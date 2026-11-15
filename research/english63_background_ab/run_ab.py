#!/usr/bin/env python3
"""One public CPU-only background A/B; --verify-result-dir is stdlib and zero-forward."""
from __future__ import annotations

import argparse
import ctypes
import io
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import wave
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ab_contract as c


def atomic_json(path, value):
    """Progress replacement leaves no extra result filename, even on failure."""
    descriptor, temporary = tempfile.mkstemp(prefix='en63-ab-progress-', dir=path.parent.parent.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write((c.p.canonical(value) + '\n').encode())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def preserve_weights(output, temporary, completed):
    """Keep each completed checkpoint even if a later arm consumes its budget and fails."""
    descriptor, path = tempfile.mkstemp(prefix='en63-ab-weights-', dir=output.parent.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for kind, arm in completed:
                name = arm + '-' + kind + '.npz'
                raw = (temporary / name).read_bytes()
                c.npz_model(raw, kind)
                archive.writestr(name, raw)
        os.replace(path, output / 'WEIGHTS.zip')
    finally:
        Path(path).unlink(missing_ok=True)


def trained_inputs(root):
    pin = c.FIXED['source_artifacts']['trained']
    c.need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'AB_TRAINED_ROOT')
    terminal = (root / 'TERMINAL.json').read_bytes()
    summary = (root / 'model/SUMMARY.json').read_bytes()
    c.need(c.digest(terminal) == pin['terminal_sha256'] and c.digest(summary) == pin['summary_sha256'],
           'AB_TRAINED_FIXED_EVIDENCE')
    all_files = {x.relative_to(root).as_posix(): c.digest(x.read_bytes()) for x in root.rglob('*') if x.is_file()}
    c.need(not any(x.is_symlink() for x in root.rglob('*')) and
           c.exact({k: v for k, v in all_files.items() if k != 'TERMINAL.json'},
                   c.parse_json(terminal)['files_sha256']), 'AB_ALL_TRAINED_MEMBERS')
    for kind in c.KINDS:
        c.need(all_files['model/' + kind + '-weights.npz'] == pin['weights_sha256'][kind], 'AB_INITIAL_WEIGHT_BYTES')
    return terminal, summary, all_files


def bounded_child(target, seconds, cpu):
    """Hard wall bound for a single isolated training worker, including Torch setup."""
    c.need(seconds > 0, 'AB_TRAINING_GROUP_DEADLINE')
    sys.stdout.flush()
    sys.stderr.flush()
    start = time.monotonic()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        try:
            # Keep the outer entrypoint's process group and also die if its
            # supervising Python process vanishes without a group kill.
            c.need(ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == parent,
                   'AB_TRAINING_PARENT_DEATH_GUARD')
            os.sched_setaffinity(0, {cpu})
            target()
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        os._exit(0)
    timed_out = False
    while True:
        child, status, usage = os.wait4(pid, os.WNOHANG)
        if child:
            break
        if time.monotonic() - start >= seconds:
            timed_out = True
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(pid, 0)
            break
        time.sleep(.02)
    return {'child_exit_code': os.waitstatus_to_exitcode(status), 'child_wall_seconds': time.monotonic() - start,
            'child_peak_rss_kib': usage.ru_maxrss, 'timed_out': timed_out}


def train_worker(kind, arm, trained, data, natural, destination, progress_path):
    # Imported only after public Actions/claim validation, in a hard-bounded child.
    import numpy as np
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(c.ROOT / 'research/english63_online_pilot'))
    import online_pilot as online
    runtime = online.configure_runtime()
    torch.manual_seed(c.PROFILE['seeds'][kind])
    initial_path = trained / 'model' / (kind + '-weights.npz')
    initial_meta, initial_payload = c.npz_model(initial_path.read_bytes(), kind)
    model = online.load_new_weights(initial_path, kind, c.p.c.SOURCE_WEIGHTS[kind])
    x, lengths = online.pilot.pad_tensor(data['train'], data['features'])
    labels = torch.tensor([row['weak_wholeclip_class'] for row in data['train']], dtype=torch.long)
    natural_x = torch.from_numpy(np.stack(natural).astype(np.float32)) if arm == 'treatment' else None
    natural_lengths = torch.full((32,), 298, dtype=torch.long)
    natural_labels = torch.zeros(32, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    model.train()
    progress = c.p.read_json(progress_path)
    record = progress['arms'][-1]
    record['runtime'] = runtime
    record['warm_start_parameter_payload_sha256'] = initial_meta['parameter_payload_sha256']
    for step in range(1, 49):
        progress['optimizer_updates_started'] += 1
        progress['active_training'] = {'kind': kind, 'arm': arm, 'step_started': step}
        atomic_json(progress_path, progress)
        optimizer.zero_grad(set_to_none=True)
        def old_loss():
            return F.cross_entropy(online.bag_logits(online.frame_logits(model, kind, x), lengths), labels)
        def natural_loss(index):
            batch = natural_x[index * 32:(index + 1) * 32]
            return F.cross_entropy(online.bag_logits(online.frame_logits(model, kind, batch), natural_lengths), natural_labels)
        values = c.accumulate_loss(old_loss, natural_loss, arm == 'treatment')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        progress['optimizer_updates_completed'] += 1
        record['updates'].append({'step': step, **values, 'gradient_norm': float(norm)})
        record['optimizer_updates'] = step
        atomic_json(progress_path, progress)
        if step == 1 or step % 12 == 0:
            print(c.p.canonical({'kind': kind, 'arm': arm, 'optimizer_updates': step, **values}), flush=True)
    model.eval()
    online.pilot.save_safe_npz(destination, model)
    final_meta, final_payload = c.npz_model(destination.read_bytes(), kind)
    c.need(final_payload != initial_payload, 'AB_NO_WEIGHT_PAYLOAD_CHANGE')
    record['weights'] = final_meta
    record['changed_state_tensors'] = c.changed_tensors(initial_payload, final_payload, kind)
    preserve_weights(progress_path.parent, destination.parent, c.ARMS[:c.ARMS.index((kind, arm)) + 1])
    record['status'] = 'WEIGHTS_SAVED'
    progress['active_training'] = None
    atomic_json(progress_path, progress)


def prepare_data(args, temporary, request_raw, claim_raw, identity, head):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    sys.path.insert(0, str(c.ROOT / 'research/english63_online_pilot'))
    import online_pilot as online
    terminal_raw, summary_raw, trained_files = trained_inputs(args.trained_result_dir)
    data = online.load_data(args.tts_zip, args.review_json)
    reference_raw = c.s.reference_input(args.negative_evidence_dir)
    reference = c.s.reference_rows(reference_raw)
    original_pcm, natural_rows, decoder = c.s.materialize_pcm(args.source_tar, reference, temporary)
    plan = c.natural_plan(natural_rows)
    windows, natural_features = [], []
    with original_pcm.open('rb') as source:
        for row in plan['train_windows']:
            source.seek(2 * (row['concatenation_start_sample'] + row['crop_start_sample']))
            raw = source.read(96000)
            c.need(len(raw) == 96000, 'AB_WINDOW_PCM_LENGTH')
            floats = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
            features = ((online.pilot.features(floats, data['bank']) - data['mean']) / data['std']).astype(np.float32)
            c.need(features.shape == (298, 32) and bool(np.isfinite(features).all()), 'AB_WINDOW_FEATURES')
            natural_features.append(features)
            windows.append({**row, 'window_pcm_sha256': c.digest(raw),
                            'feature_payload_sha256': c.digest(features.astype('<f4').tobytes())})
        offsets = {r['member']: r['concatenation_start_sample'] for r in natural_rows}
        natural_dev = temporary / 'natural-study-development.pcm'
        with natural_dev.open('xb') as out:
            for row in plan['study_dev_rows']:
                source.seek(offsets[row['member']] * 2)
                raw = source.read(row['sample_count'] * 2)
                c.need(len(raw) == row['sample_count'] * 2, 'AB_DEV_PCM_LENGTH')
                out.write(raw)
    tts_rows, tts_paths = [], []
    with zipfile.ZipFile(args.tts_zip) as archive:
        c.need(len(archive.namelist()) == len(set(archive.namelist())), 'AB_TTS_ZIP_DUPLICATES')
        for row in data['rows']:
            raw = archive.read('english-seven-voice-actual/capture/cells/' + row['cell_id'] + '/audio-16k.wav')
            c.need(c.digest(raw) == row['wav_sha256'], 'AB_TTS_WAV_HASH')
            with wave.open(io.BytesIO(raw), 'rb') as wav:
                count = wav.getnframes()
                c.need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 16000 and
                       wav.getcomptype() == 'NONE', 'AB_TTS_PCM_FORMAT')
                pcm = wav.readframes(count)
                c.need(len(pcm) == count * 2 and not wav.readframes(1), 'AB_TTS_PCM_LENGTH')
            c.need(c.digest(pcm) == row['pcm_sha256'], 'AB_TTS_PCM_HASH')
            tts_rows.append({**row, 'source_samples': count})
            if row['action'] == 'EXPOSED_DEVELOPMENT_NO_TUNE':
                path = temporary / (row['cell_id'] + '.pcm')
                path.write_bytes(pcm)
                tts_paths.append(path)
    original_train = [r for r in tts_rows if r['action'] == 'TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY']
    train_features = [{'cell_id': r['cell_id'], 'frames': len(data['features'][r['cell_id']]),
                       'feature_payload_sha256': c.digest(data['features'][r['cell_id']].astype('<f4').tobytes())}
                      for r in original_train]
    segments = [{'source_id': 'natural-study-development', 'samples': 11124244, 'weak_wholeclip_class': 0,
                 'rows': plan['study_dev_rows']}]
    for row in tts_rows:
        if row['action'] == 'EXPOSED_DEVELOPMENT_NO_TUNE':
            segments.append({'source_id': row['cell_id'], 'samples': row['source_samples'],
                             'weak_wholeclip_class': row['weak_wholeclip_class'],
                             'rows': [{'member': row['cell_id'], 'sample_count': row['source_samples'],
                                       'concatenation_start_sample': 0}]})
    pcm_paths = [natural_dev, *tts_paths]
    (temporary / 'evaluation.txt').write_text(''.join(str(x) + '\n' for x in pcm_paths))
    inputs = {'schema': 'english63-background-ab-inputs-v1', 'source_artifacts': c.FIXED['source_artifacts'],
              'request_raw_utf8': request_raw.decode(), 'request_sha256': c.digest(request_raw),
              'claim_raw_utf8': claim_raw.decode(), 'claim_sha256': c.digest(claim_raw),
              'trained_terminal_raw_utf8': terminal_raw.decode(), 'trained_summary_raw_utf8': summary_raw.decode(),
              'trained_files_sha256': trained_files, 'review_raw_utf8': args.review_json.read_text(),
              'reference_raw_utf8': reference_raw.decode(), 'natural_rows': natural_rows, 'natural_plan': plan,
              'natural_windows': windows, 'pcm_decoder': decoder, 'original_train_rows': original_train,
              'original_train_features': train_features, 'tts_rows': tts_rows, 'evaluation_segments': segments,
              'natural_dev_pcm_sha256': c.s.sha_file(natural_dev),
              'evaluation_pcm_sha256': {x.name: c.s.sha_file(x) for x in pcm_paths}, 'dataset_audit': data['audit']}
    c.verify_inputs(inputs, {'public_run': identity, 'pipeline_head': head})
    # Only these thirty old clips and the 128 natural crops reach any optimizer.
    train_data = {'train': data['train'], 'features': {r['cell_id']: data['features'][r['cell_id']] for r in data['train']}}
    return inputs, train_data, natural_features, pcm_paths


def train_all(args, temporary, data, natural, cpu):
    path = args.output_dir / 'TRAINING.json'
    progress = {'schema': 'english63-background-ab-training-v1', 'status': 'STARTED_NO_RETRY',
                'optimizer_updates_started': 0, 'optimizer_updates_completed': 0, 'active_training': None, 'arms': []}
    atomic_json(path, progress)
    start = time.monotonic()
    for kind, arm in c.ARMS:
        record = {'kind': kind, 'arm': arm, 'status': 'STARTED_NO_RETRY', 'optimizer_updates': 0,
                  'seed': c.PROFILE['seeds'][kind], 'warm_start_sha256': c.p.c.SOURCE_WEIGHTS[kind],
                  'optimizer': 'fresh-AdamW', 'learning_rate': .001, 'weight_decay': .01, 'gradient_clip': 1,
                  'original_train_rows': 30, 'natural_train_rows': 128 if arm == 'treatment' else 0,
                  'natural_coefficient': .5 if arm == 'treatment' else 0, 'train_development_rows': 0,
                  'train_nearphone_rows': 0, 'model_promotion_allowed': False, 'updates': []}
        progress['arms'].append(record)
        progress['active_training'] = {'kind': kind, 'arm': arm, 'step_started': 0}
        atomic_json(path, progress)
        destination = temporary / (arm + '-' + kind + '.npz')
        remaining = start + 720 - time.monotonic()
        result = bounded_child(lambda: train_worker(kind, arm, args.trained_result_dir, data, natural,
                                                    destination, path), min(180, remaining), cpu)
        progress = c.p.read_json(path)
        progress['arms'][-1].update(result)
        if result['child_exit_code'] != 0 or result['timed_out']:
            progress['status'] = 'FAILED_NO_RETRY'
            progress['arms'][-1]['status'] = 'FAILED_NO_RETRY'
            atomic_json(path, progress)
            raise ValueError('AB_TRAINING_CHILD_FAILED_NO_RETRY')
        c.need(progress['arms'][-1]['status'] == 'WEIGHTS_SAVED' and
               progress['arms'][-1]['optimizer_updates'] == 48, 'AB_WORKER_CLOSED')
        progress['arms'][-1]['status'] = 'COMPLETED'
        atomic_json(path, progress)
    progress['group_wall_seconds'] = time.monotonic() - start
    progress['status'] = 'COMPLETED'
    progress['active_training'] = None
    c.need(progress['group_wall_seconds'] <= 720, 'AB_TRAINING_GROUP_LIMIT')
    atomic_json(path, progress)
    return progress


def build_all(args, temporary, source, identity, head):
    c.need(platform.system() == 'Linux' and not (c.s.HERE / 'en63_generated.h').exists() and
           not (HERE / 'en63_generated.h').exists(), 'AB_HEADER_INCLUDE_SCOPE')
    compiler = shutil.which('gcc')
    c.need(compiler is not None, 'AB_GCC_REQUIRED')
    compiler = str(Path(compiler).absolute())
    real = Path(compiler).resolve()
    compiler_sha = c.s.sha_file(real)
    # Preserve the old exporter's full source-summary and normalizer checks.
    exported = c.exporter.export(args.trained_result_dir, temporary / 'baseline-export')
    baseline = (temporary / 'baseline-export/en63_generated.h').read_bytes()
    c.need(exported['header_sha256'] == c.PROFILE['baseline_header_sha256'], 'AB_BASELINE_HEADER_PIN')
    weight_raw = {name: (temporary / name).read_bytes() for name in c.WEIGHT_FILES}
    weights = {name: c.npz_model(raw, 'dsc_tcn' if 'dsc_tcn' in name else 'gru') for name, raw in weight_raw.items()}
    variants, runtime = [], {}
    for arm in c.VARIANTS:
        destination = temporary / arm
        destination.mkdir()
        header = baseline if arm == 'baseline' else c.replace_models(baseline, {k: weights[arm + '-' + k + '.npz'] for k in c.KINDS})
        (destination / 'en63_generated.h').write_bytes(header)
        command = c.command_for(compiler, c.ROOT, destination)
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=c.p.native_environment())
        c.need(result.returncode == 0, 'AB_COMPILE_FAILED:' + result.stderr)
        binary = (destination / 'ab_replay').read_bytes()
        runtime[arm + '.h'], runtime[arm + '.bin'] = header, binary
        variants.append({'arm': arm, 'command': command, 'exit_code': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr,
                         'header_sha256': c.digest(header), 'binary_sha256': c.digest(binary)})
    c.need(c.s.sha_file(real) == compiler_sha, 'AB_COMPILER_CHANGED')
    build = {'schema': 'english63-background-ab-build-v1', 'compiler': compiler, 'compiler_realpath': str(real),
             'compiler_sha256': compiler_sha, 'compiler_version': subprocess.check_output(
                 [compiler, '--version'], text=True, timeout=15, env=c.p.native_environment()),
             'source_root': str(c.ROOT), 'build_root': str(temporary), 'output_root': str(args.output_dir),
             'entrypoint_pid': os.getpid(),
             'source_sha256': source, 'public_run': identity, 'pipeline_head': head, 'variants': variants,
             'physical_ssc305': False, 'machine': platform.machine(), 'libc': list(platform.libc_ver())}
    c.need(c.zip_members((args.output_dir / 'WEIGHTS.zip').read_bytes(), c.WEIGHT_FILES) == weight_raw,
           'AB_ALL_COMPLETED_WEIGHTS_PRESERVED')
    for name, members in (('RUNTIME.zip', runtime),):
        with zipfile.ZipFile(args.output_dir / name, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for member, raw in sorted(members.items()):
                archive.writestr(member, raw)
    c.p.c.write_json(args.output_dir / 'BUILD.json', build)
    return build, runtime


def native_pass(output, temporary, arm, kind, cpu, deadline):
    remaining = deadline - time.monotonic()
    c.need(remaining > 0, 'AB_NATIVE_GROUP_BUDGET')
    command = [str(temporary / arm / 'ab_replay'), '--supervise', kind, str(temporary / 'evaluation.txt'),
               str(output / (arm + '-' + kind + '.trace')), str(cpu), str(os.getpid())]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=c.p.native_environment())
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=min(120, remaining))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=10)
    return {'arm': arm, 'kind': kind, 'command': command, 'exit_code': process.returncode,
            'status': 'COMPLETED' if process.returncode == 0 and not timed_out else 'FAILED_NO_RETRY',
            'timed_out': timed_out, 'raw_stdout': stdout, 'raw_stderr': stderr}


def run(args):
    identity = c.p.require_public_actions()  # Gate before reading audio, importing Torch, or any model operation.
    output = args.output_dir
    c.need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
           not any(x.is_symlink() for x in (output, *output.parents)), 'AB_CREATE_ONLY_OUTPUT')
    inputs_paths = (args.trained_result_dir, args.tts_zip, args.review_json, args.negative_evidence_dir, args.source_tar)
    c.need(all(x is not None and x.is_absolute() and not x.is_symlink() for x in inputs_paths), 'AB_ABSOLUTE_INPUTS')
    head = subprocess.check_output(['git', '-C', str(c.ROOT), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    request_path, claim_path = output.parent / 'REQUEST.json', output.parent.parent / 'claim/CLAIM.json'
    request_raw, claim_raw = request_path.read_bytes(), claim_path.read_bytes()
    c.validate_request(request_raw, head)
    c.need(c.exact(c.parse_json(claim_raw), c.claim_expected({'request_sha256': c.digest(request_raw)},
                                                          {'public_run': identity, 'pipeline_head': head})), 'AB_PRIOR_CLAIM')
    cpus = sorted(os.sched_getaffinity(0))
    c.need(cpus and all(c.p.is_int(x, 0, 1023) for x in cpus), 'AB_ALLOWED_CPUS')
    before = c.source_hashes()
    output.mkdir()
    c.p.c.write_json(output / 'PROTOCOL.json', c.protocol())
    with zipfile.ZipFile(output / 'SOURCE.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in c.SOURCES:
            archive.writestr(name, (c.ROOT / name).read_bytes())
    with tempfile.TemporaryDirectory(prefix='en63-background48-') as temp:
        temporary = Path(temp)
        inputs, data, natural, pcm_paths = prepare_data(args, temporary, request_raw, claim_raw, identity, head)
        c.p.c.write_json(output / 'INPUTS.json', inputs)
        training = train_all(args, temporary, data, natural, cpus[0])
        build, runtime = build_all(args, temporary, before, identity, head)
        measurements = {'schema': 'english63-background-ab-measurements-v1', 'public_run': identity,
                        'pipeline_head': head, 'initial_allowed_cpus': cpus, 'selected_cpu': cpus[0], 'passes': []}
        (output / 'native.stderr').touch(exist_ok=False)
        models = []
        start = time.monotonic()
        for arm, kind in c.PASSES:
            measurements['passes'].append({'arm': arm, 'kind': kind, 'status': 'STARTED_NO_RETRY'})
            atomic_json(output / 'MEASUREMENTS.json', measurements)
            record = native_pass(output, temporary, arm, kind, cpus[0], start + 180)
            measurements['passes'][-1] = record
            atomic_json(output / 'MEASUREMENTS.json', measurements)
            with (output / 'native.stderr').open('a') as stderr:
                stderr.write(record['raw_stderr'])
            observation = c.trace_observation((output / (arm + '-' + kind + '.trace')).read_bytes(), inputs['evaluation_segments'])
            models.append(c.pass_observation(record, arm, kind, build, cpus[0], observation))
        measurements['native_group_wall_seconds'] = time.monotonic() - start
        c.need(measurements['native_group_wall_seconds'] <= 180, 'AB_NATIVE_GROUP_LIMIT')
        measurements['final_allowed_cpus'] = sorted(os.sched_getaffinity(0))
        _, _, parent_after = trained_inputs(args.trained_result_dir)
        runtime_after = {a + suffix: c.s.sha_file(temporary / a / ('en63_generated.h' if suffix == '.h' else 'ab_replay'))
                         for a in c.VARIANTS for suffix in ('.h', '.bin')}
        measurements['source_integrity'] = {
            'source_before': before, 'source_after': c.source_hashes(),
            'corpus_before': c.s.CORPUS['sha256'], 'corpus_after': c.s.sha_file(args.source_tar),
            'request_before': c.digest(request_raw), 'request_after': c.s.sha_file(request_path),
            'claim_before': c.digest(claim_raw), 'claim_after': c.s.sha_file(claim_path),
            'trained_files_before': inputs['trained_files_sha256'], 'trained_files_after': parent_after,
            'evaluation_pcm_before': inputs['evaluation_pcm_sha256'],
            'evaluation_pcm_after': {x.name: c.s.sha_file(x) for x in pcm_paths},
            'runtime_before': {name: c.digest(raw) for name, raw in runtime.items()}, 'runtime_after': runtime_after}
        atomic_json(output / 'MEASUREMENTS.json', measurements)
        summary = c.summary_from({**inputs, 'file_sha256': c.s.sha_file(output / 'INPUTS.json')}, build,
                                 {**training, 'file_sha256': c.s.sha_file(output / 'TRAINING.json')},
                                 {**measurements, 'file_sha256': c.s.sha_file(output / 'MEASUREMENTS.json')}, models)
        c.p.c.write_json(output / 'SUMMARY.json', summary)
        receipt = c.saved_receipt(output)
        c.p.c.write_json(output / 'REPLAY.json', receipt)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trained-result-dir', 'tts-zip', 'review-json', 'negative-evidence-dir', 'source-tar',
                 'output-dir', 'verify-result-dir'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    runtime = (args.trained_result_dir, args.tts_zip, args.review_json, args.negative_evidence_dir, args.source_tar, args.output_dir)
    if args.verify_result_dir is not None:
        c.need(not any(x is not None for x in runtime), 'AB_SAVED_ONLY_ARGUMENTS')
        result = c.verify_saved(args.verify_result_dir)
    else:
        c.need(all(x is not None for x in runtime), 'AB_RUNTIME_ARGUMENTS')
        result = run(args)
    print(c.p.canonical(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
