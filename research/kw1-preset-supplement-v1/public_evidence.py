"""Reconstructed, model-free receipt validation and public evidence projection.

Durable authority is a strict parent chain, never progress/summary JSON. A broken
chain has unknown attempts and reserves its entire budget. Its body is not public.
This module neither imports model packages nor creates execution claims.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path, PurePosixPath
import re

from execution_common import canonical, exact_equal, require, sha
from blind_pcm import (bind_input, canonical as blind_canonical, decode, identity_map,
                       read_regular, unaliased, MAX_INPUT_BYTES)

MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
ENGINES = ('qwen06', 'qwen17')
RECOGNIZERS = ('qwen_asr', 'sensevoice', 'whisper_small')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')
REASONS = ('exception', 'timeout', 'decoder_quality', 'interrupted')
STAGE_FILES = ('stage-claim.json', 'identity-map.json', 'model-load-started.json',
               'model-load-completed.json', 'model-load-failed.json', 'stage-failed.json')
CELL_FILES = ('started.json', 'decoder-evidence.json', 'completed.json', 'failed.json')


def digest(value):
    require(type(value) is str and DIGEST.fullmatch(value) is not None, 'SHA256 required')
    return value


def same(actual, expected, message):
    require(exact_equal(actual, expected), message)


def fields(value, names, message):
    require(type(value) is dict and set(value) == set(names), message)


def json_bytes(raw):
    return decode(raw, MAX_FILE_BYTES)


def safe_name(name):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name
            and PurePosixPath(name).as_posix() == name
            and all(p not in ('', '.', '..') for p in name.split('/')), 'safe relative name')
    return name


def active(root):
    """Only an actual empty, unaliased directory takes the empty-stage path."""
    try:
        root = unaliased(root)
        return not root.is_dir() or any(root.iterdir())
    except (OSError, ValueError):
        return True


class Archive:
    """Record exactly the approved bytes read, without rereading at export time."""
    def __init__(self, root):
        self.root = unaliased(root)
        require(self.root.is_dir(), 'archive directory missing')
        self.files = {}

    def exists(self, name):
        safe_name(name)
        return os.path.lexists(self.root / name)

    def raw(self, name):
        safe_name(name)
        raw = read_regular(self.root / name, MAX_FILE_BYTES)
        self.files[name] = raw
        return raw

    def doc(self, name):
        return json_bytes(self.raw(name))

    def cells(self, allowed):
        root = self.root / 'cells'
        if not os.path.lexists(root):
            return set()
        require(unaliased(root).is_dir(), 'cell directory required')
        actual = {p.name for p in root.iterdir()}
        require(actual <= set(allowed), 'unknown cell identity')
        for name in actual:
            require(unaliased(root / name).is_dir(), 'cell directory required')
        return actual


def _metadata(root, names):
    """No body or attacker-selected filename crosses a failed boundary."""
    rows = []
    for name in sorted(set(names)):
        try:
            raw = read_regular(Path(root) / name, MAX_FILE_BYTES)
        except (OSError, ValueError):
            # A missing/unreadable object is still explicitly recorded.
            rows.append(dict(file=name, sha256=None, bytes=None))
        else:
            rows.append(dict(file=name, sha256=sha(raw), bytes=len(raw)))
    return rows


def _unverified(root, names):
    return dict(verified=False, status='UNVERIFIED', attempts=None,
                reserved_no_retry=True, outcomes=[], files={}, receipts=_metadata(root, names))


def _verified(outcomes, archive, status='VERIFIED'):
    return dict(verified=True, status=status, attempts=sum(r['attempts'] for r in outcomes),
                reserved_no_retry=True, outcomes=outcomes, files=archive.files, receipts=[])


def _load_identity(value, schema, key, identity):
    fields(value, ('schema', key, 'scientific_load_sha256', 'runtime_identity_sha256', 'source_pins_sha256'),
           'public model load identity fields')
    require(value['schema'] == schema and value[key] == identity, 'public model load identity')
    for name in ('scientific_load_sha256', 'runtime_identity_sha256', 'source_pins_sha256'):
        digest(value[name])


def _stage_failure(archive):
    if archive.exists('stage-failed.json'):
        row = archive.doc('stage-failed.json')
        fields(row, ('reason', 'retries'), 'stage failure fields')
        require(type(row['reason']) is str and row['reason'] in REASONS + ('technical_quality', 'duplicate_native'),
                'stage failure reason')
        same(row['retries'], 0, 'stage failure retry count')


def validate_tts_archive(root, engine, plan):
    from worker import validate_outcome, verify_audio_outcome
    require(engine in ENGINES, 'TTS engine')
    archive = Archive(root)
    cells = [c for c in plan['slots'] if c['engine'] == engine]
    outcomes = [dict(c, status='NOT_RUN', attempts=0, error_type=None, file=None,
                     raw_wav_sha256=None, native=None, metrics=None, termination=None) for c in cells]
    if not any(archive.root.iterdir()):
        return _verified(outcomes, archive, 'NOT_RUN')
    claim = dict(schema='kw1-tts-stage-claim-v1', engine=engine,
                 plan_sha256=sha(canonical(plan)), attempt=1)
    same(archive.doc('stage-claim.json'), claim, 'TTS stage identity')
    present = archive.cells(c['slot_id'] for c in cells)
    started = archive.exists('model-load-started.json')
    completed = archive.exists('model-load-completed.json')
    failed = archive.exists('model-load-failed.json')
    require(not (completed and failed), 'conflicting TTS load terminal')
    require(started or not (completed or failed or present), 'TTS missing load parent')
    load_hash = None
    if started:
        same(archive.doc('model-load-started.json'), dict(claim, load_seconds=1200), 'TTS load start identity')
        start_sha = sha(archive.files['model-load-started.json'])
        if completed:
            load = archive.doc('model-load-completed.json')
            fields(load, set(claim) | {'started_sha256', 'receipt'}, 'TTS load complete fields')
            same({k: load[k] for k in claim}, claim, 'TTS load complete identity')
            same(load['started_sha256'], start_sha, 'TTS load start hash')
            _load_identity(load['receipt'], 'kw1-public-tts-load-identity-v1', 'engine', engine)
            load_hash = sha(archive.files['model-load-completed.json'])
        elif failed:
            load = archive.doc('model-load-failed.json')
            fields(load, set(claim) | {'started_sha256', 'reason'}, 'TTS load failure fields')
            same({k: load[k] for k in claim}, claim, 'TTS load failure identity')
            same(load['started_sha256'], start_sha, 'TTS failed load start hash')
            require(load['reason'] in ('exception', 'timeout'), 'TTS load failure reason')
    terminal_stop = False
    seen_values, seen_wavs = set(), set()
    for i, cell in enumerate(cells):
        sid = cell['slot_id']; prefix = 'cells/' + sid + '/'
        if sid not in present:
            require(not archive.exists(sid + '.raw-float.wav'), 'orphan native WAV without call parent')
            terminal_stop = True
            continue
        require({p.name for p in (archive.root / 'cells' / sid).iterdir()} <= {'started.json', 'completed.json', 'failed.json'},
                'unknown TTS cell file')
        has_start = archive.exists(prefix + 'started.json')
        has_complete = archive.exists(prefix + 'completed.json')
        has_fail = archive.exists(prefix + 'failed.json')
        require(not (has_complete and has_fail), 'conflicting TTS cell terminal')
        require(has_start or not (has_complete or has_fail), 'TTS missing call parent')
        if not has_start:
            require(not archive.exists(sid + '.raw-float.wav'), 'orphan native WAV without call start')
            terminal_stop = True
            continue
        require(not terminal_stop and load_hash is not None, 'TTS call order/load parent')
        expected = dict(schema='kw1-tts-cell-started-v1', engine=engine, slot_id=sid,
                        plan_sha256=claim['plan_sha256'], load_completed_sha256=load_hash,
                        cell_sha256=sha(canonical(cell)), attempt=1)
        same(archive.doc(prefix + 'started.json'), expected, 'TTS call start identity')
        outcome = dict(outcomes[i], status='FAILED', attempts=1, error_type='interrupted')
        if has_complete or has_fail:
            name = prefix + ('completed.json' if has_complete else 'failed.json')
            terminal = archive.doc(name)
            wanted = {'schema', 'started_sha256', 'outcome'} | (set() if has_complete else {'reason'})
            fields(terminal, wanted, 'TTS cell terminal fields')
            require(terminal['schema'] == ('kw1-tts-cell-completed-v1' if has_complete else 'kw1-tts-cell-failed-v1'),
                    'TTS terminal schema')
            same(terminal['started_sha256'], sha(archive.files[prefix + 'started.json']), 'TTS start hash')
            outcome = terminal['outcome']
            validate_outcome(outcome, cell, 'completed' if has_complete else 'FAILED')
            if has_fail:
                same(terminal['reason'], outcome['error_type'], 'TTS failure reason binding')
            if outcome['file'] is not None:
                same(outcome['file'], sid + '.raw-float.wav', 'TTS fixed native filename')
                raw = archive.raw(outcome['file'])
                same(sha(raw), outcome['raw_wav_sha256'], 'native WAV hash')
                verify_audio_outcome(raw, outcome)
                if has_complete:
                    require(outcome['raw_wav_sha256'] not in seen_wavs and
                            outcome['native']['source_values_sha256'] not in seen_values, 'duplicate native stage output')
                    seen_wavs.add(outcome['raw_wav_sha256']); seen_values.add(outcome['native']['source_values_sha256'])
        validate_outcome(outcome, cell, outcome['status'])
        outcomes[i] = outcome
        terminal_stop = outcome['status'] != 'completed'
    _stage_failure(archive)
    return _verified(outcomes, archive)


def assess_tts_archive(root, engine, plan):
    ids = [c['slot_id'] for c in plan['slots'] if c['engine'] == engine]
    names = list(STAGE_FILES) + [f'cells/{sid}/{name}' for sid in ids for name in CELL_FILES]
    names += [sid + '.raw-float.wav' for sid in ids]
    try:
        return validate_tts_archive(root, engine, plan)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return _unverified(root, names)


def _asr_outcome(outcome, initial, successful):
    from blind_asr_worker import OUTCOME_FIELDS, validate_result
    fields(outcome, OUTCOME_FIELDS, 'ASR outcome fields')
    for key in ('opaque_id', 'recognizer', 'pcm_sha256', 'wav_sha256'):
        same(outcome[key], initial[key], 'ASR identity')
    same(outcome['attempts'], 1, 'ASR exact attempt count')
    require(outcome['status'] == ('SUCCESS' if successful else 'FAILED'), 'ASR status')
    if successful:
        require(outcome['error_type'] is None, 'ASR success error')
    else:
        require(type(outcome['error_type']) is str and outcome['error_type'] in REASONS, 'ASR error enum')
    if outcome['raw_text'] is None:
        require(not successful and outcome['completeness'] == 'unknown', 'ASR missing result')
        same(outcome['quality_flags'], [], 'ASR absent result flags')
    else:
        accepted = validate_result({k: outcome[k] for k in ('raw_text', 'completeness', 'quality_flags')})
        require(not successful or accepted, 'ASR incomplete success')


def validate_asr_archive(input_root, root, recognizer):
    from blind_asr_worker import initial_outcomes
    require(recognizer in RECOGNIZERS, 'ASR recognizer')
    job, job_raw, pcm = bind_input(input_root)
    mapping = identity_map(job, job_raw, pcm)
    outcomes = initial_outcomes(mapping, recognizer)
    archive = Archive(root)
    if not any(archive.root.iterdir()):
        return mapping, _verified(outcomes, archive, 'NOT_RUN')
    claim = dict(schema='kw1-blind-asr-stage-claim-v1', recognizer=recognizer,
                 job_sha256=sha(job_raw), identity_map_sha256=sha(blind_canonical(mapping)), attempt=1)
    same(archive.doc('stage-claim.json'), claim, 'ASR stage identity')
    present = archive.cells(r['opaque_id'] for r in outcomes)
    has_map = archive.exists('identity-map.json')
    has_start = archive.exists('model-load-started.json')
    complete = archive.exists('model-load-completed.json')
    failed = archive.exists('model-load-failed.json')
    require(has_map or not (has_start or complete or failed or present), 'ASR missing map parent')
    if has_map:
        same(archive.doc('identity-map.json'), mapping, 'ASR map identity')
    require(not (complete and failed), 'conflicting ASR load terminal')
    require(has_start or not (complete or failed), 'ASR missing load start parent')
    load_sha = None
    if has_start:
        same(archive.doc('model-load-started.json'), dict(claim, load_seconds=1200), 'ASR load start identity')
        if complete:
            load = archive.doc('model-load-completed.json')
            fields(load, set(claim) | {'receipt'}, 'ASR load complete fields')
            same({k: load[k] for k in claim}, claim, 'ASR load complete identity')
            _load_identity(load['receipt'], 'kw1-public-model-load-identity-v1', 'recognizer', recognizer)
            load_sha = sha(archive.files['model-load-completed.json'])
        elif failed:
            load = archive.doc('model-load-failed.json')
            fields(load, set(claim) | {'reason'}, 'ASR load failure fields')
            same({k: load[k] for k in claim}, claim, 'ASR load failure identity')
            require(type(load['reason']) is str and load['reason'] in ('exception', 'timeout'), 'ASR load failure reason')
    stopped = False
    for i, initial in enumerate(list(outcomes)):
        oid = initial['opaque_id']; prefix = 'cells/' + oid + '/'
        if oid not in present:
            stopped = True
            continue
        started = archive.exists(prefix + 'started.json')
        success = archive.exists(prefix + 'completed.json')
        failure = archive.exists(prefix + 'failed.json')
        evidence = archive.exists(prefix + 'decoder-evidence.json')
        require(not (success and failure), 'conflicting ASR cell terminal')
        require(started or not (success or failure or evidence), 'ASR missing call parent')
        if not started:
            stopped = True
            continue
        require(not stopped and load_sha is not None, 'ASR serial/load parent')
        expected = dict(schema='kw1-blind-asr-cell-started-v1', recognizer=recognizer,
                        opaque_id=oid, pcm_sha256=initial['pcm_sha256'], wav_sha256=initial['wav_sha256'],
                        job_sha256=sha(job_raw), identity_map_sha256=sha(blind_canonical(mapping)),
                        load_completed_sha256=load_sha, attempt=1)
        start_raw = archive.raw(prefix + 'started.json')
        outcome = dict(initial, status='FAILED', attempts=1, error_type='interrupted')
        try:
            start_doc = json_bytes(start_raw)
        except (ValueError, UnicodeError):
            # The sole admitted raw-prefix exception. Not arbitrary corrupt JSON.
            expected_raw = blind_canonical(expected)
            require(len(start_raw) < len(expected_raw) and expected_raw.startswith(start_raw), 'invalid ASR start prefix')
            require(not (success or failure or evidence), 'interrupted start has descendants')
            outcomes[i] = outcome
            stopped = True
            continue
        same(start_doc, expected, 'ASR call start identity')
        if success or failure:
            terminal = archive.doc(prefix + ('completed.json' if success else 'failed.json'))
            fields(terminal, {'schema', 'started_sha256', 'decoder_evidence_sha256', 'outcome'} |
                   (set() if success else {'reason'}), 'ASR terminal fields')
            require(terminal['schema'] == ('kw1-blind-asr-cell-completed-v1' if success else 'kw1-blind-asr-cell-failed-v1'),
                    'ASR terminal schema')
            same(terminal['started_sha256'], sha(start_raw), 'ASR terminal start binding')
            outcome = terminal['outcome']
            _asr_outcome(outcome, initial, success)
            if failure:
                same(terminal['reason'], outcome['error_type'], 'ASR failure reason binding')
            evidence_sha = terminal['decoder_evidence_sha256']
            if evidence_sha is not None:
                digest(evidence_sha)
                item = archive.doc(prefix + 'decoder-evidence.json')
                same(sha(archive.files[prefix + 'decoder-evidence.json']), evidence_sha, 'ASR decoder evidence hash')
                fields(item, ('result', 'evidence'), 'ASR decoder evidence fields')
                same(item['result'], {k: outcome[k] for k in ('raw_text', 'completeness', 'quality_flags')},
                     'ASR decoder result binding')
                fields(item['evidence'], ('schema', 'recognizer', 'scientific_result_sha256'), 'public decoder fields')
                require(item['evidence']['schema'] == 'kw1-public-decoder-evidence-v1'
                        and item['evidence']['recognizer'] == recognizer, 'public decoder schema')
                digest(item['evidence']['scientific_result_sha256'])
            else:
                require(not success and not evidence and outcome['raw_text'] is None, 'missing decoder evidence')
        else:
            # A completed evidence write without a terminal cannot qualify. It is
            # deliberately not copied, and cannot substitute for completion.
            require(not evidence, 'decoder evidence missing terminal')
        outcomes[i] = outcome
        stopped = not success
    _stage_failure(archive)
    return mapping, _verified(outcomes, archive)


def assess_asr_archive(input_root, root, recognizer, host=None):
    names = list(STAGE_FILES)
    try:
        job, _, _ = bind_input(input_root)
        names += [f"cells/{r['opaque_id']}/{name}" for r in job['clips'] for name in CELL_FILES]
        if host is not None:
            require(host.get('verified') is True and len(host.get('mapping', [])) == 16, 'unverified ASR host parent')
        return validate_asr_archive(input_root, root, recognizer)[1]
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return _unverified(root, names)


def assess_host_inputs(work, plan):
    work = Path(work)
    names = ['identity-map.json', 'blind-input/job.json']
    try:
        archive = Archive(work)
        if not archive.exists('identity-map.json') and not archive.exists('blind-input'):
            return dict(verified=True, mapping=[], files={}, receipts=[])
        mapping = archive.doc('identity-map.json')
        require(type(mapping) is list and len(mapping) == 16, 'host map fixed denominator')
        job, raw, pcm = bind_input(work / 'blind-input')
        require(len(job['clips']) == 16, 'host PCM fixed denominator')
        archive.files['blind-input/job.json'] = raw
        by_pcm = {r['opaque_id']: r for r in job['clips']}
        native = {}
        for engine in ENGINES:
            stage = validate_tts_archive(work / 'tts' / engine / 'output', engine, plan)
            native.update({r['slot_id']: r for r in stage['outcomes']})
        seen_native, seen_values, seen_pcm = set(), set(), set()
        for cell, row in zip(plan['slots'], mapping):
            fields(row, ('slot_id', 'native_sha256', 'native_values_sha256', 'derived', 'derivation'), 'host map row fields')
            same(row['slot_id'], cell['slot_id'], 'host slot order')
            source = native[cell['slot_id']]
            require(source['status'] == 'completed' and source['metrics']['technically_valid'] is True,
                    'host native completion parent')
            same(row['native_sha256'], source['raw_wav_sha256'], 'host native hash')
            same(row['native_values_sha256'], source['native']['source_values_sha256'], 'host value hash')
            derived = row['derived']; require(type(derived) is dict, 'derived map record')
            digest(derived.get('opaque_id'))
            same(derived, by_pcm[derived['opaque_id']], 'host PCM identity')
            same(derived['frames'], (2 * source['native']['frames'] + 2) // 3, 'derived frame formula')
            recipe = row['derivation']
            fields(recipe, ('method', 'up', 'down', 'window', 'padtype', 'quantization', 'numpy_version',
                            'scipy_version', 'gain_normalization', 'trimming'), 'derivation fields')
            expected = dict(method='scipy.signal.resample_poly', up=2, down=3, window=['kaiser', 5.0],
                            padtype='constant', quantization='floor_float64_times_32768_to_int16',
                            gain_normalization=False, trimming=False)
            same({k: recipe[k] for k in expected}, expected, 'fixed numerical policy')
            same(recipe['numpy_version'], '2.3.2', 'pinned NumPy runtime')
            same(recipe['scipy_version'], '1.17.1', 'pinned SciPy runtime')
            require(row['native_sha256'] not in seen_native and row['native_values_sha256'] not in seen_values
                    and derived['opaque_id'] not in seen_pcm, 'duplicate campaign audio identity')
            seen_native.add(row['native_sha256']); seen_values.add(row['native_values_sha256']); seen_pcm.add(derived['opaque_id'])
            archive.files['blind-input/' + derived['file']] = pcm[derived['opaque_id']]
            names.append('blind-input/' + derived['file'])
        return dict(verified=True, mapping=mapping, files=archive.files, receipts=[])
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return dict(verified=False, mapping=[], files={}, receipts=_metadata(work, names))


def validate_admission(value):
    names = ('schema', 'admitted', 'run_id', 'source_sha', 'workflow_id', 'config_sha256',
             'source_files', 'max_tts_calls', 'max_asr_calls', 'retry')
    fields(value, names, 'admission projection fields')
    same({k: value[k] for k in ('schema', 'admitted', 'max_tts_calls', 'max_asr_calls', 'retry')},
         dict(schema='kw1-admission-receipt-v1', admitted=True, max_tts_calls=16, max_asr_calls=48, retry=False),
         'admission policy')
    for key in ('run_id', 'workflow_id'):
        require(type(value[key]) is int and value[key] > 0, 'admission numeric identity')
    require(type(value['source_sha']) is str and re.fullmatch(r'[0-9a-f]{40}', value['source_sha']), 'admission git identity')
    digest(value['config_sha256'])
    require(type(value['source_files']) is dict and 1 <= len(value['source_files']) <= 1024, 'source metadata map')
    for path, value_sha in value['source_files'].items():
        safe_name(path); digest(value_sha)
    return copy.deepcopy(value)


def ledger(report):
    rows = []
    for slot in report['slots']:
        rows.append(dict(kind='tts', slot_id=slot['slot_id'], stage=slot['engine'],
                         status=slot['native_status'], attempts=slot['native_attempts'], reserved_no_retry=True))
    for recognizer in RECOGNIZERS:
        for slot in report['slots']:
            item = next(r for r in slot['recognitions'] if r['recognizer'] == recognizer)
            rows.append(dict(kind='asr', slot_id=slot['slot_id'], stage=recognizer,
                             status=item['status'], attempts=item['attempts'], reserved_no_retry=True))
    require(len(rows) == 64, 'fixed 64-slot ledger')
    for item in rows:
        require(item['attempts'] is None if item['status'] == 'UNVERIFIED'
                else type(item['attempts']) is int and item['attempts'] in (0, 1), 'strict ledger attempt count')
    return dict(schema='kw1-fixed-attempt-ledger-v1', denominator=64, retry=False, slots=rows)


def validate_resources(value):
    """Positive schema for supervisor measurements; limits are not measurements."""
    order = ('image', 'setup_qwen06', 'tts_qwen06', 'setup_qwen17', 'tts_qwen17',
             'setup_primary', 'qwen_asr', 'sensevoice', 'setup_whisper', 'whisper_small')
    fields(value, ('schema', 'status', 'stages', 'runtime_cleanups', 'max_wall_seconds',
                   'max_attempts_per_stage', 'raw_setup_exported', 'api_tokens_passed'), 'resource summary fields')
    same({k: value[k] for k in ('schema', 'max_wall_seconds', 'max_attempts_per_stage',
                                'raw_setup_exported', 'api_tokens_passed')},
         dict(schema='kw1-container-summary-v1', max_wall_seconds=18000, max_attempts_per_stage=1,
              raw_setup_exported=False, api_tokens_passed=False), 'resource policy')
    require(type(value['status']) is str and value['status'] in ('started', 'complete', 'stopped_no_retry'),
            'resource summary status')
    require(type(value['stages']) is list and len(value['stages']) <= len(order), 'resource stage count')
    seen = {}
    for i, row in enumerate(value['stages']):
        required = {'stage', 'status', 'attempt', 'limits', 'container_configuration_verified', 'memory_observation'}
        require(type(row) is dict and required <= row.keys() <= required | {'wall_seconds', 'exit_code', 'oom_killed', 'host'},
                'resource stage fields')
        same(row['stage'], order[i], 'resource fixed serial order')
        require(type(row['status']) is str and row['status'] in ('started', 'complete', 'failed_no_retry'), 'resource stage status')
        require(i == len(value['stages']) - 1 or row['status'] == 'complete', 'resource serial terminal order')
        same(row['attempt'], 1, 'resource attempt')
        require(type(row['container_configuration_verified']) is bool, 'resource configuration flag')
        label = row['stage']; setup = label.startswith('setup_')
        limits = dict(global_wall_seconds=18000, stage_wall_seconds=180 if label == 'image' else (1200 if setup else 4800),
                      host_available_memory_min_bytes=14 * 1024**3, host_available_disk_min_bytes=25 * 1024**3)
        if label != 'image':
            limits.update(memory_bytes=12 * 1024**3, memory_swap_bytes=12 * 1024**3, cpus=4, pids=256,
                          readonly_root=True, readonly_code=True, readonly_runtime=not setup,
                          nonroot=True, network='bridge' if setup else 'none')
            if not setup:
                limits.update(load_wall_seconds=1200, call_wall_seconds=300)
        same(row['limits'], limits, 'resource enforced limits')
        same(row['memory_observation'], dict(process_peak_rss_kib=None, cgroup_memory_peak_bytes=None,
             scope='host supervisor does not sample worker RSS or cgroup peak; configured limits are not measurements', status='unknown'),
             'resource observation scope')
        if 'wall_seconds' in row:
            require(type(row['wall_seconds']) in (int, float) and math.isfinite(row['wall_seconds'])
                    and row['wall_seconds'] >= 0, 'resource finite wall time')
        if 'exit_code' in row:
            require(type(row['exit_code']) is int and -255 <= row['exit_code'] <= 255, 'resource exit status')
        if 'oom_killed' in row:
            require(type(row['oom_killed']) is bool, 'resource OOM flag')
        if 'host' in row:
            fields(row['host'], ('available_memory_bytes', 'available_disk_bytes'), 'host resource fields')
            require(all(type(v) is int and 0 <= v < 2**63 for v in row['host'].values()), 'host resource measurements')
        seen[label] = row['status']
    if value['status'] == 'complete':
        require(len(seen) == len(order) and all(v == 'complete' for v in seen.values()), 'resource completion count')
    require(type(value['runtime_cleanups']) is list and len(value['runtime_cleanups']) <= 4, 'resource cleanup count')
    terminal = {'runtime-qwen06': 'tts_qwen06', 'runtime-qwen17': 'tts_qwen17',
                'runtime-primary': 'sensevoice', 'runtime-whisper': 'whisper_small'}
    cleanups = set()
    for row in value['runtime_cleanups']:
        fields(row, ('runtime', 'status', 'logical_regular_bytes', 'scope'), 'runtime cleanup fields')
        require(type(row['runtime']) is str and row['runtime'] in terminal and row['runtime'] not in cleanups,
                'runtime cleanup identity')
        require(seen.get(terminal[row['runtime']]) == 'complete', 'runtime cleanup after terminal container')
        require(type(row['status']) is str and row['status'] in ('started', 'removed_after_container_cleanup', 'failed_no_retry'),
                'runtime cleanup status')
        require(type(row['logical_regular_bytes']) is int and 0 <= row['logical_regular_bytes'] < 2**63,
                'runtime logical bytes not disk peak')
        same(row['scope'], 'only this campaign owned runtime; claims/results/source code retained', 'runtime cleanup scope')
        cleanups.add(row['runtime'])
    if value['status'] == 'complete':
        require(cleanups == set(terminal) and all(r['status'] == 'removed_after_container_cleanup' for r in value['runtime_cleanups']),
                'resource completion requires all runtime cleanup receipts')
    return value


def assess_resources(work):
    try:
        raw = read_regular(Path(work) / 'backend/summary.json', MAX_FILE_BYTES)
        body = validate_resources(json_bytes(raw))
        return dict(verified=True, body=body, receipts=[], files={'backend/summary.json': raw})
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return dict(verified=False, body=None, receipts=_metadata(work, ['backend/summary.json']), files={})


def resources_complete(resources):
    if resources is None or not resources['verified'] or resources['body']['status'] != 'complete':
        return False
    body = resources['body']
    # Missing observations remain unknown. Declared limits do not prove that
    # inspection, teardown, host reserves, or successful container exit occurred.
    elapsed = 0.0
    for row in body['stages']:
        wall = row.get('wall_seconds')
        if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= row['limits']['stage_wall_seconds']:
            return False
        elapsed += wall
        host = row.get('host', {})
        if host.get('available_memory_bytes', -1) < 14 * 1024**3 or host.get('available_disk_bytes', -1) < 25 * 1024**3:
            return False
        if row['stage'] != 'image' and (row['container_configuration_verified'] is not True or
                type(row.get('exit_code')) is not int or row['exit_code'] != 0 or row.get('oom_killed') is not False):
            return False
    return elapsed <= 18000


def enforce_stage_order(stages, work=None):
    """A valid isolated receipt cannot authorize an impossible later stage."""
    permitted = True
    for key, count, success in (('tts/qwen06', 4, 'completed'), ('tts/qwen17', 12, 'completed'),
                               *((('asr/' + r), 16, 'SUCCESS') for r in RECOGNIZERS)):
        stage = stages[key]
        if not permitted and stage['verified'] and stage['files']:
            require(work is not None, 'later stage ran before predecessor completed')
            stages[key] = stage = _unverified(Path(work) / key / 'output', stage['files'].keys())
        complete = stage['verified'] and len(stage['outcomes']) == count and all(
            r['status'] == success and type(r['attempts']) is int and r['attempts'] == 1 and
            (success == 'completed' or (r['completeness'] == 'complete' and r['quality_flags'] == []))
            for r in stage['outcomes'])
        permitted = permitted and complete
    return stages


def validate_host_stage_binding(host, stages):
    if not host['verified'] or not host['mapping']:
        return
    native = {}
    for engine in ENGINES:
        stage = stages['tts/' + engine]
        require(stage['verified'], 'host map stage verification changed')
        native.update({row['slot_id']: row for row in stage['outcomes']})
    require(len(native) == 16 and all(row['status'] == 'completed' for row in native.values()),
            'host map complete TTS parent')
    for item in host['mapping']:
        row = native[item['slot_id']]
        same(item['native_sha256'], row['raw_wav_sha256'], 'host map native snapshot binding')
        same(item['native_values_sha256'], row['native']['source_values_sha256'], 'host map sample snapshot binding')


def assemble_report(plan, host, stages, resources=None):
    from campaign import report, mark_unknown
    tts, asr, errors = [], {}, []
    for engine in ENGINES:
        stage = stages['tts/' + engine]
        if stage['verified']:
            tts.extend(stage['outcomes'])
        else:
            errors.append('unverified_tts_' + engine)
    for recognizer in RECOGNIZERS:
        stage = stages['asr/' + recognizer]
        if stage['verified'] and host['verified'] and len(host['mapping']) == 16:
            asr[recognizer] = [dict(r, status='success' if r['status'] == 'SUCCESS' else r['status']) for r in stage['outcomes']]
        else:
            errors.append('unverified_asr_' + recognizer)
    mapping = host['mapping'] if host['verified'] else []
    complete_calls = (not errors and len(tts) == 16 and all(r['status'] == 'completed' for r in tts)
                and len(mapping) == 16 and all(len(asr.get(name, [])) == 16 and
                all(r['status'] == 'success' for r in asr[name]) for name in RECOGNIZERS))
    resource_complete = resources_complete(resources)
    complete = complete_calls and resource_complete
    reason = ('container_resource_unverified' if resources is None or not resources['verified'] else
              'container_stage_or_cleanup_incomplete') if complete_calls and not resource_complete else 'interrupted_or_preexecution_failure'
    result = report(plan, tts, mapping, asr, None if complete else reason)
    mark_unknown(result, errors)
    result['resources'] = dict(schema='kw1-recovery-only-v1', model_calls_made=0,
                              unverified_stages=errors, counts_are_lower_bounds=bool(errors),
                              all_unverified_budget_slots_reserved=True)
    if resources is not None:
        result['resources']['container_evidence'] = dict(verified=resources['verified'],
            body=resources['body'], receipts=resources['receipts'])
    return result


def collect(plan, work):
    from plan_contract import validate_plan
    validate_plan(plan)
    host = assess_host_inputs(work, plan)
    stages = {'tts/' + engine: assess_tts_archive(Path(work) / 'tts' / engine / 'output', engine, plan)
              for engine in ENGINES}
    stages.update({'asr/' + name: assess_asr_archive(Path(work) / 'blind-input', Path(work) / 'asr' / name / 'output', name, host)
                   for name in RECOGNIZERS})
    enforce_stage_order(stages, work)
    try:
        validate_host_stage_binding(host, stages)
    except (ValueError, KeyError, TypeError):
        host = dict(verified=False, mapping=[], files={}, receipts=_metadata(work, ['identity-map.json', 'blind-input/job.json']))
        for name in RECOGNIZERS:
            key = 'asr/' + name
            stages[key] = _unverified(Path(work) / key / 'output', stages[key]['files'].keys() or STAGE_FILES)
    resources = assess_resources(work)
    result = assemble_report(plan, host, stages, resources)
    return host, stages, resources, result, ledger(result)
