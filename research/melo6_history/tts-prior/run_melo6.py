"""Disabled six-call Melo pilot. Importing this file imports only stdlib.

Approval is external to this program. The explicit CLI flag and approved plan
hash are execution interlocks, not a claim that approval has been received.
Only the supervised, armed child imports ONNX Runtime and creates one session.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'melo-native-six-phrase-pilot-v1'
MODEL_SHA = 'bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d'
TEXTS = ['你好小窝', '小窝小窝', '你好小屋', '小屋小屋', '你好你好', '今天天气很好']
MiB = 1024 ** 2
OUTPUT_CAP = 20 * MiB
LOG_CAP = 4 * MiB
PRIVATE_CAUSE_CAP = 64 * 1024
RECEIPT_CAP = 2 * MiB
PRIVATE_CAUSE_FILENAME = 'private-raw-cause.json'
PAYLOAD_CAP = OUTPUT_CAP - LOG_CAP - 2 * MiB - MiB  # blind pack + outer receipts
SAFE_ERRORS = {'ValueError', 'TypeError', 'RuntimeError', 'TimeoutError',
               'OSError', 'FileExistsError', 'FileNotFoundError', 'ImportError',
               'ModuleNotFoundError', 'MemoryError', 'AssertionError'}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(MiB), b''):
            h.update(block)
    return h.hexdigest()


def regular(path):
    return path.is_file() and not path.is_symlink()


def safe_error(error):
    name = type(error).__name__
    return name if name in SAFE_ERRORS else 'OtherException'


def safe_diagnostic(error):
    """Own-source locations and allowlisted codes, never raw exception text."""
    result = {'exception_type': safe_error(error), 'source_frames': []}
    allowed = {'run_melo6.py', 'pack_blind.py', 'conversion_recipe.py', 'supervision.py'}
    known_codes = set()
    known_functions = {}
    for name in allowed:
        tree = ast.parse((ROOT / name).read_bytes())
        known_functions[name] = {n.name for n in ast.walk(tree)
                                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'require':
                if len(node.args) == 2 and isinstance(node.args[1], ast.Constant):
                    known_codes.add(node.args[1].value)
    if type(error) is ValueError and error.args and isinstance(error.args[0], str) and error.args[0] in known_codes:
        result['validation_code'] = error.args[0]
    if isinstance(error, OSError) and isinstance(error.errno, int) and 0 < error.errno < 4096:
        result['errno'] = error.errno
    tb = error.__traceback__
    while tb is not None and len(result['source_frames']) < 16:
        path = Path(tb.tb_frame.f_code.co_filename).resolve()
        if path.name in allowed and path == ROOT / path.name:
            function = tb.tb_frame.f_code.co_name
            result['source_frames'].append({'module': path.name,
                'function': function if function in known_functions[path.name] else '<other>',
                'line': tb.tb_lineno, 'sha256': sha(path)})
        tb = tb.tb_next
    return result


def bounded_write(path, data, *, replace=False, budget_root=None):
    path = Path(path)
    require(len(data) <= 2 * MiB, 'INDIVIDUAL_FILE_CAP')
    if budget_root is not None:
        from supervision import tree_bytes
        old = path.stat().st_size if path.exists() else 0
        require(tree_bytes(budget_root) - old + len(data) <= PAYLOAD_CAP,
                'PAYLOAD_CAP')
    if not replace:
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path, value, **kwargs):
    bounded_write(path, canonical(value), **kwargs)


def retain_private_cause(runtime, error):
    """Retain raw exception chain privately, capped; never collect frame locals.

    This file is explicitly outside every future public artifact allowlist.
    Only this function's safe status/size/hash projection may be published.
    """
    trace = traceback.TracebackException.from_exception(error, capture_locals=False)
    raw = ''.join(trace.format(chain=True)).encode('utf-8', errors='backslashreplace')
    def encode(prefix_bytes):
        text = raw[:prefix_bytes].decode('utf-8', errors='ignore')
        retained = len(text.encode('utf-8'))
        value = {'schema': 'melo-private-raw-cause-v1', 'private': True,
                 'public_artifact_allowed': False, 'capture_locals': False,
                 'traceback_chain': text, 'traceback_total_bytes': len(raw),
                 'traceback_retained_bytes': retained, 'truncated': retained < len(raw),
                 'receipt_cap_bytes': PRIVATE_CAUSE_CAP}
        return value, canonical(value)
    # JSON escaping can expand even an ASCII prefix, so bound encoded bytes.
    low, high = 0, min(len(raw), PRIVATE_CAUSE_CAP)
    while low < high:
        middle = (low + high + 1) // 2
        if len(encode(middle)[1]) <= PRIVATE_CAUSE_CAP:
            low = middle
        else:
            high = middle - 1
    value, data = encode(low)
    from supervision import tree_bytes
    receipt_bytes = sum(p.stat().st_size for p in runtime.rglob('*.json') if regular(p))
    # Reserve the subsequent sanitized stage update inside existing budgets.
    require(receipt_bytes + len(data) + 4096 <= RECEIPT_CAP, 'PRIVATE_CAUSE_RECEIPT_BUDGET')
    require(tree_bytes(runtime) + len(data) + 4096 <= OUTPUT_CAP, 'PRIVATE_CAUSE_OUTPUT_BUDGET')
    private_path = runtime / PRIVATE_CAUSE_FILENAME
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {'status': 'retained_private', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'truncated': value['truncated'],
            'capture_locals': False, 'public_artifact_allowed': False}


def validate_plan(plan_path, approved_sha):
    require(regular(plan_path) and sha(plan_path) == approved_sha,
            'APPROVED_PLAN_HASH_MISMATCH')
    plan = json.loads(plan_path.read_bytes())
    require(plan['schema'] == 'melo-six-execution-plan-v1' and
            plan['experiment_id'] == EXPERIMENT, 'EXPERIMENT_IDENTITY')
    expected = {'calls_max': 6, 'attempts_per_phrase': 1, 'batch_size': 1,
                'sid': [1], 'seed': 0, 'sample_rate': 44100,
                'max_output_frames': 441000, 'noise_scale': 0.6,
                'length_scale': 1.0, 'noise_scale_w': 0.8,
                'providers': ['CPUExecutionProvider'], 'intra_op_threads': 2,
                'inter_op_threads': 1, 'call_wall_seconds': 60,
                'whole_wall_seconds': 600, 'rss_threshold_bytes': 2 * 1024**3,
                'saved_output_cap_bytes': OUTPUT_CAP, 'log_cap_bytes': LOG_CAP,
                'retries': 0}
    require(plan['execution'] == expected, 'FROZEN_EXECUTION_CHANGED')
    require(plan['model'] == {'bytes': 170429550, 'sha256': MODEL_SHA},
            'MODEL_IDENTITY')
    require(plan['phrase_order'] == TEXTS and
            plan['tone_policy'] == 'dictionary-tone3+3', 'PHRASES_OR_TONES_CHANGED')
    for item in plan['bindings']:
        name = item['path']
        require(Path(name).name == name, 'BINDING_PATH')
        path = ROOT / name
        require(regular(path) and path.stat().st_size == item['bytes'] and
                sha(path) == item['sha256'], 'BINDING_MISMATCH')
    require({i['path'] for i in plan['bindings']} ==
            {'adapter-freeze.json', 'frozen-six-inputs.json', 'onnx-static-metadata.json'},
            'BINDING_SET')
    freeze = json.loads((ROOT / 'adapter-freeze.json').read_bytes())
    for name, digest in freeze['source_sha256'].items():
        require(Path(name).name == name and regular(ROOT / name) and
                sha(ROOT / name) == digest, 'SOURCE_FREEZE_MISMATCH')
    fixtures = json.loads((ROOT / 'frozen-six-inputs.json').read_bytes())
    require(fixtures['phrase_order_frozen'] == TEXTS and
            [r['text'] for r in fixtures['fixtures']] == TEXTS, 'FIXTURE_ORDER')
    metadata = json.loads((ROOT / 'onnx-static-metadata.json').read_bytes())
    require(metadata['sha256'] == MODEL_SHA and metadata['bytes'] == 170429550,
            'STATIC_MODEL_IDENTITY')
    wanted = {'version': '2', 'model_type': 'melo-vits', 'add_blank': '1',
              'n_speakers': '1', 'speaker_id': '1', 'lang_id': '3',
              'tone_start': '0', 'sample_rate': '44100'}
    require(all(metadata['metadata'].get(k) == v for k, v in wanted.items()),
            'STATIC_METADATA_MISMATCH')
    return plan, fixtures['fixtures'], metadata


def input_arrays(fixture, np):
    inputs = fixture['inputs']
    require(set(inputs) == {'x', 'x_lengths', 'tones', 'sid', 'noise_scale',
                            'length_scale', 'noise_scale_w'}, 'INPUT_SET')
    arrays = {}
    for name, spec in inputs.items():
        dtype = 'int64' if name in {'x', 'x_lengths', 'tones', 'sid'} else 'float32'
        require(spec['dtype'] == dtype, 'INPUT_DTYPE')
        arr = np.array(spec['values'], dtype=dtype, order='C')
        require(list(arr.shape) == spec['shape'], 'INPUT_SHAPE')
        arrays[name] = arr
    length = arrays['x'].shape[1]
    require(arrays['x'].shape == (1, length) and arrays['tones'].shape == (1, length)
            and arrays['x_lengths'].tolist() == [length]
            and arrays['sid'].tolist() == [1], 'BATCH_OR_SPEAKER')
    for name, value in [('noise_scale', .6), ('length_scale', 1.), ('noise_scale_w', .8)]:
        require(np.array_equal(arrays[name], np.array([value], dtype='float32')),
                'SCALAR_VALUE')
    return arrays


def input_boundary(arrays):
    return {name: {'dtype': str(a.dtype), 'shape': list(a.shape),
                   'values': a.tolist(), 'little_endian_bytes_sha256':
                   hashlib.sha256(a.astype(a.dtype.newbyteorder('<'), copy=False)
                                  .tobytes(order='C')).hexdigest()}
            for name, a in sorted(arrays.items())}


def run_six(session, fixtures, out, plan_sha, np, stage=lambda value: None):
    """Single fakeable loop. Caller supplies the sole session; no retry path."""
    from conversion_recipe import native_wav, derivative_wav
    require(not out.exists(), 'NO_RESUME')
    out.mkdir()
    require(len(fixtures) == 6 and [x['text'] for x in fixtures] == TEXTS,
            'EXACT_SIX_FIXTURES')
    receipt = {'schema': 'melo-six-generation-receipt-v1', 'experiment_id': EXPERIMENT,
               'plan_sha256': plan_sha, 'status': 'started', 'attempts_consumed': 0,
               'deterministic_waveform_claim': False,
               'generation_rows': [{'source_id': f'melo6-{i:03d}', 'status': 'not_run',
                                    'attempt_consumed': False} for i in range(1, 7)]}
    def save():
        write_json(out / 'generation-receipt.json', receipt, replace=True, budget_root=out)
    save()
    try:
        for fixture, record in zip(fixtures, receipt['generation_rows']):
            arrays = input_arrays(fixture, np)
            claim = {'source_id': record['source_id'], 'attempt': 1,
                     'attempt_consumed': True, 'inputs': input_boundary(arrays),
                     'requested_outputs': ['y'], 'plan_sha256': plan_sha}
            # Durable write-ahead evidence exists before the only native run call.
            write_json(out / (record['source_id'] + '.started.json'), claim, budget_root=out)
            record.update(status='attempt_started', attempt_consumed=True)
            receipt['attempts_consumed'] += 1
            save()
            started = time.monotonic()
            write_json(out.parent / 'active-call.json',
                       {'pid': os.getpid(), 'source_id': record['source_id'],
                        'deadline_monotonic': started + 60}, replace=True)
            stage('session_run')
            results = session.run(['y'], arrays)
            write_json(out.parent / 'active-call.json', {'inactive': True}, replace=True)
            require(time.monotonic() - started <= 60, 'CALL_DEADLINE_EXCEEDED')
            require(isinstance(results, list) and len(results) == 1, 'OUTPUT_COUNT')
            y = results[0]
            require(isinstance(y, np.ndarray) and y.dtype == np.dtype('float32')
                    and y.ndim == 3 and y.shape[:2] == (1, 1)
                    and 0 < y.shape[2] <= 441000 and np.isfinite(y).all(),
                    'OUTPUT_GEOMETRY_OR_FINITE')
            x = y[0, 0]
            native = native_wav(x)
            pcm, conversion = derivative_wav(x)
            require(len(native) <= 1768096 and len(pcm) <= 320044, 'AUDIO_FILE_CAP')
            name = record['source_id']
            bounded_write(out / (name + '.native-f32.wav'), native, budget_root=out)
            bounded_write(out / (name + '.pcm16.wav'), pcm, budget_root=out)
            flags = []
            if not np.any(x):
                flags.append('silent')
            if np.max(np.abs(x)) >= 1:
                flags.append('source_peak_ge_one')
            if conversion['resampled_peak'] >= 1:
                flags.append('derivative_peak_ge_one')
            record.update(status='generated_candidate', source_frames=int(x.size),
                          native_wav_sha256=hashlib.sha256(native).hexdigest(),
                          pcm_wav_sha256=hashlib.sha256(pcm).hexdigest(),
                          pcm_sha256=hashlib.sha256(pcm[44:]).hexdigest(),
                          native_float_sha256=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
                          signal_flags=flags, conversion=conversion,
                          peak=float(np.max(np.abs(x))),
                          rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),
                          call_and_conversion_seconds=time.monotonic() - started)
            save()
        receipt['status'] = 'six_candidates_generated'
        save()
        return receipt
    except BaseException as error:
        for row in receipt['generation_rows']:
            if row['status'] == 'attempt_started':
                row.update(status='failed_no_retry', error_class=safe_error(error))
        receipt.update(status='failed_no_retry', error_class=safe_error(error),
                       error_code='MELO_SIX_STAGE_FAILED')
        save()
        raise
    finally:
        write_json(out.parent / 'active-call.json', {'inactive': True}, replace=True)


def validate_loaded_session(session, metadata):
    require(session.get_providers() == ['CPUExecutionProvider'], 'CPU_PROVIDER_ONLY')
    def expected(rows):
        return [(r['name'], 'tensor(int64)' if r['element_type'] == 'INT64' else 'tensor(float)',
                 [d['symbol'] if 'symbol' in d else d['value'] for d in r['shape']]) for r in rows]
    require([(r.name, r.type, r.shape) for r in session.get_inputs()] ==
            expected(metadata['graph']['inputs']), 'LOADED_INPUT_SIGNATURE')
    require([(r.name, r.type, r.shape) for r in session.get_outputs()] ==
            expected(metadata['graph']['outputs']), 'LOADED_OUTPUT_SIGNATURE')
    actual_metadata = session.get_modelmeta().custom_metadata_map
    require(all(actual_metadata.get(k) == v for k, v in metadata['metadata'].items()),
            'LOADED_METADATA')


def armed_child(model, runtime, plan_path, approved_sha):
    """Only this entry imports ORT. Called by a fresh process-group supervisor."""
    require(os.environ.get('MELO6_SUPERVISED_PLAN_SHA256') == approved_sha and
            os.getpgrp() == os.getpid(), 'SUPERVISED_CHILD_REQUIRED')
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))
    write_json(runtime / 'child-identity.json', {'pid': os.getpid(), 'pgid': os.getpgrp()}, replace=True)
    diagnostic = {'schema': 'melo-six-stage-v1', 'experiment_id': EXPERIMENT,
                  'stage': 'verify_plan', 'status': 'started', 'raw_text_disclosed': False}
    def stage(value):
        diagnostic['stage'] = value
        write_json(runtime / 'stage.json', diagnostic, replace=True)
    stage('verify_plan')
    try:
        plan, fixtures, metadata = validate_plan(plan_path, approved_sha)
        stage('verify_model')
        require(regular(model) and model.stat().st_size == 170429550 and sha(model) == MODEL_SHA,
                'MODEL_BYTES_MISMATCH')
        stage('verify_runtime_versions')
        versions = {'onnxruntime': '1.30.0', 'flatbuffers': '25.12.19',
                    'protobuf': '6.33.5', 'numpy': '2.3.5', 'scipy': '1.17.0'}
        require(all(importlib.metadata.version(k) == v for k, v in versions.items()),
                'RUNTIME_VERSION_MISMATCH')
        stage('import_runtime')
        import numpy as np
        import onnxruntime as ort
        ort.disable_telemetry_events()
        ort.set_seed(0)  # one seed before the one session; no determinism claim
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        stage('create_one_session')
        session = ort.InferenceSession(str(model), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        validate_loaded_session(session, metadata)
        write_json(runtime / 'runtime-receipt.json',
                   {'versions': versions, 'session_count': 1, 'set_seed_count': 1,
                    'seed': 0, 'waveform_determinism_claim': False,
                    'providers': session.get_providers(), 'plan_sha256': approved_sha})
        run_six(session, fixtures, runtime / 'generation', approved_sha, np, stage)
        diagnostic['status'] = 'complete'
    except BaseException as error:
        # Capture the external message/chain before producing the public-safe
        # stage record; redirected native stdout need not contain this exception.
        try:
            private_cause = retain_private_cause(runtime, error)
        except BaseException as retention_error:
            private_cause = {'status': 'retention_failed',
                             **safe_diagnostic(retention_error),
                             'public_artifact_allowed': False}
        diagnostic.update(status='failed_no_retry', **safe_diagnostic(error),
                          code='MELO_SIX_STAGE_FAILED', private_cause=private_cause)
        write_json(runtime / 'stage.json', diagnostic, replace=True)
        return 1
    write_json(runtime / 'stage.json', diagnostic, replace=True)
    return 0


def observed_group(pid, proc=Path('/proc')):
    """Do not turn inaccessible /proc observations into a measured zero RSS."""
    members = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            group = int(fields[2])
            int(fields[21])
        except FileNotFoundError:
            continue  # process disappeared during the observation
        except (OSError, ValueError, IndexError):
            raise RuntimeError('PROC_OBSERVATION_UNKNOWN') from None
        if group == pid:
            members.append(int(directory.name))
    return members


def watchdog(runtime, stop, report, *, poll_seconds=.025):
    """Separate watchdog can SIGKILL a child blocked in a native run call.

    RSS belongs to unchanged supervision.py. Disk/log checks here are sampled;
    artifact writers and child RLIMIT_FSIZE also enforce smaller saved caps.
    """
    from supervision import tree_bytes
    while not stop.wait(poll_seconds):
        identity_path = runtime / 'child-identity.json'
        if not identity_path.exists():
            continue
        identity = json.loads(identity_path.read_bytes())
        pid = identity['pid']
        require(identity['pgid'] == pid, 'CHILD_GROUP_IDENTITY')
        try:
            members = observed_group(pid)
            if not members:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError('PROC_OBSERVATION_UNKNOWN')
        except (OSError, RuntimeError):
            report.update(reason='proc_observation_unknown', rss_observation='unknown_not_zero')
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        active_path = runtime / 'active-call.json'
        reason = None
        if active_path.exists():
            active = json.loads(active_path.read_bytes())
            if not active.get('inactive'):
                require(active['pid'] == pid, 'CALL_GROUP_IDENTITY')
                if time.monotonic() >= active['deadline_monotonic']:
                    reason = 'native_call_wall_deadline'
        log = runtime / 'child.log'
        if log.exists() and log.stat().st_size > LOG_CAP:
            reason = 'sampled_log_cap'
        if tree_bytes(runtime) > OUTPUT_CAP:
            reason = 'sampled_output_cap'
        if reason:
            report.update(reason=reason, source_id=active.get('source_id') if active_path.exists() else None)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return


def watchdog_process(runtime):
    class Stop:
        def wait(self, seconds):
            time.sleep(seconds)
            return (runtime / 'watchdog.stop').exists()
    report = {}
    try:
        watchdog(runtime, Stop(), report)
    except BaseException as error:
        report.update(reason='watchdog_exception', **safe_diagnostic(error))
        identity = runtime / 'child-identity.json'
        if identity.exists():
            pid = json.loads(identity.read_bytes())['pid']
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    write_json(runtime / 'watchdog-result.json', report)
    return 0


def observe_supervision(command, observation, **kwargs):
    """Wrap the unchanged supervisor's sampling route with observation evidence."""
    import supervision
    original = supervision.group_sample
    def checked_sample(group, proc=Path('/proc')):
        try:
            observed_group(group, proc)
            sample = original(group, proc)
        except BaseException:
            observation['unknown_samples'] += 1
            raise
        if sample['members']:
            observation['valid_member_samples'] += 1
        else:
            observation['empty_member_samples'] += 1
        return sample
    supervision.group_sample = checked_sample
    try:
        return supervision.supervise(command, **kwargs)
    finally:
        supervision.group_sample = original


def annotate_rss(receipt, observation, watchdog_receipt):
    receipt['rss_observation_counts'] = observation
    unknown = (observation['valid_member_samples'] == 0 or observation['unknown_samples'] > 0
               or watchdog_receipt.get('reason') == 'proc_observation_unknown')
    receipt['rss_observation'] = 'unknown_not_zero' if unknown else 'observed_process_group_samples'
    if unknown:
        receipt.setdefault('max_sampled', {})['rss_bytes'] = None
    return receipt


def finalize(runtime, approved_sha, supervision_receipt, watchdog_receipt):
    generation = runtime / 'generation'
    generation.mkdir(exist_ok=True)
    target = generation / 'generation-receipt.json'
    if target.exists():
        receipt = json.loads(target.read_bytes())
    else:
        receipt = {'schema': 'melo-six-generation-receipt-v1', 'experiment_id': EXPERIMENT,
                   'plan_sha256': approved_sha, 'status': 'failed_no_retry',
                   'generation_rows': [{'source_id': f'melo6-{i:03d}', 'status': 'not_run',
                                        'attempt_consumed': False} for i in range(1, 7)]}
    failed = supervision_receipt.get('returncode') != 0 or bool(watchdog_receipt)
    for row in receipt['generation_rows']:
        claim = generation / (row['source_id'] + '.started.json')
        if claim.exists() and row['status'] != 'generated_candidate':
            row.update(status='failed_no_retry', attempt_consumed=True)
            failed = True
    receipt['attempts_consumed'] = sum(
        (generation / (row['source_id'] + '.started.json')).exists()
        for row in receipt['generation_rows'])
    if failed:
        receipt.update(status='failed_no_retry', error_code='SUPERVISED_STAGE_FAILED')
    write_json(target, receipt, replace=True, budget_root=generation)
    write_json(runtime / 'supervision-receipt.json', supervision_receipt)
    write_json(runtime / 'watchdog-receipt.json', watchdog_receipt)
    # Freeze only regular files; partial files remain evidence, never ASR input.
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(generation.iterdir()) if regular(p)]
    write_json(generation / 'generation-freeze.json',
               {'schema': 'melo-six-generation-freeze-v1', 'experiment_id': EXPERIMENT,
                'plan_sha256': approved_sha, 'files': files}, budget_root=generation)
    from supervision import tree_bytes
    require(tree_bytes(runtime) <= OUTPUT_CAP, 'SAVED_OUTPUT_CAP')
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-approved-melo6', action='store_true')
    p.add_argument('--plan', type=Path, default=ROOT / 'execution-plan.json')
    p.add_argument('--plan-sha256')
    p.add_argument('--model', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--_supervised-child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--_watchdog', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args._watchdog:
        require(args.out and args.out.is_dir(), 'WATCHDOG_DIRECTORY_REQUIRED')
        return watchdog_process(args.out)
    if not args.execute_approved_melo6:
        print(json.dumps({'status': 'disabled', 'experiment_id': EXPERIMENT,
                          'inference_performed': False, 'runtime_imported': False}))
        return 0
    require(args.plan_sha256 and args.model and args.out, 'PLAN_HASH_MODEL_OUT_REQUIRED')
    if args._supervised_child:
        return armed_child(args.model, args.out, args.plan, args.plan_sha256)
    started = time.monotonic()
    validate_plan(args.plan, args.plan_sha256)
    require(not args.out.exists(), 'OUTPUT_EXISTS_NO_RESUME')
    args.out.mkdir(parents=True)
    args.out = args.out.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'ORT_', 'OMP_', 'MKL_', 'OPENBLAS_'))}
    env.update(MELO6_SUPERVISED_PLAN_SHA256=args.plan_sha256,
               ORT_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONHASHSEED='0')
    # Separate process avoids preexec_fn in a multithreaded parent. The original
    # supervisor is copied unchanged and still owns child process-group cleanup.
    watcher = subprocess.Popen([sys.executable, '-B', str(ROOT / 'run_melo6.py'),
                                '--_watchdog', '--out', str(args.out)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    command = [sys.executable, '-B', str(ROOT / 'run_melo6.py'),
               '--execute-approved-melo6', '--_supervised-child',
               '--plan', str(args.plan.resolve()), '--plan-sha256', args.plan_sha256,
               '--model', str(args.model.resolve()), '--out', str(args.out)]
    observation = {'valid_member_samples': 0, 'empty_member_samples': 0, 'unknown_samples': 0}
    try:
        supervised = observe_supervision(command, observation, cwd=ROOT, env=env,
                               wall_seconds=600 - (time.monotonic() - started),
                               cpu_seconds=1200, rss_bytes=2 * 1024**3, cores=2,
                               workspace=args.out, installed=args.out / 'not-installed-here',
                               buildtmp=args.out / 'no-build', log_path=args.out / 'child.log',
                               job_deadline=started + 600)
    except BaseException as error:
        supervised = {'returncode': 1, 'termination_reason': 'supervisor_exception',
                      **safe_diagnostic(error)}
    finally:
        write_json(args.out / 'watchdog.stop', {'stop': True})
        try:
            watcher.wait(timeout=2)
        except subprocess.TimeoutExpired:
            watcher.kill()
            watcher.wait()
    result_path = args.out / 'watchdog-result.json'
    report = json.loads(result_path.read_bytes()) if result_path.exists() else {'reason': 'watchdog_result_missing'}
    supervised = annotate_rss(supervised, observation, report)
    receipt = finalize(args.out, args.plan_sha256, supervised, report)
    print(json.dumps({'status': receipt['status'], 'experiment_id': EXPERIMENT}))
    return 0 if receipt['status'] == 'six_candidates_generated' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'refused_or_failed', **safe_diagnostic(error),
                          'raw_error_text_disclosed': False}))
        raise SystemExit(1)
