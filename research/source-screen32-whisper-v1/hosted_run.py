"""One explicitly activated free-hosted run. No local inference entry point.

Uses the unchanged source-screen32-v1 container constructor, runtime proof and
supervisor; only the fixed Python command tail is replaced. Old evidence remains
byte-for-byte historical. New third-route raw outputs are not training evidence.
"""
from pathlib import Path
import json
import os
import re
import shutil
import signal
import struct
import sys
import time
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
from common import (EXPERIMENT, BRANCH, LIMITS, BOUNDARY, require, read_json, write_once,
                    atomic, file_hash, canonical, bind_input, identity, regular, wav_bytes, sha)
from asr_native_contracts import validate_native_capture


def historical_host():
    old = ROOT / 'research/source-screen32-v1'
    sys.path.insert(0, str(old))
    spec = importlib.util.spec_from_file_location('retained_screen32_host', old / 'hosted_run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def admission(env):
    release = read_json(HERE / 'execution-release.json')
    require(set(release) == {'schema', 'experiment', 'approved', 'source_freeze_sha256'} and
            release['schema'] == 'whisper-screen-execution-release-v1' and
            release['experiment'] == EXPERIMENT and release['approved'] is True, 'reviewed activation required')
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + BRANCH,
        'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux'}
    require(all(env.get(k) == v for k, v in expected.items()), 'one-shot free public hosted identity')
    require(re.fullmatch('[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch('[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')), 'immutable run/source identity')
    event = read_json(env['GITHUB_EVENT_PATH'])
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('private') is False and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'],
            'first public branch creation required')
    freeze_path = HERE / 'execution-freeze.json'
    require(file_hash(freeze_path) == release['source_freeze_sha256'], 'reviewed source freeze identity')
    freeze = read_json(freeze_path)
    require(set(freeze) == {'schema', 'files', 'blind_code'} and
            freeze['schema'] == 'whisper-screen-source-freeze-v1', 'source freeze schema')
    for name, digest in freeze['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                name.startswith(('research/', 'tools/', 'tests/')), 'source freeze path')
        regular(ROOT / name)
        require(file_hash(ROOT / name) == digest, 'reviewed source drift')
    require((ROOT / '.github/workflows/source-screen32-whisper-third-v1.yml').read_bytes() ==
            (HERE / 'workflow.yml').read_bytes(), 'activated workflow drift')
    require(set(freeze['blind_code']) <= set(freeze['files']) and len(freeze['blind_code']) == 8,
            'fixed blind code membership')
    allowed = {'research/source-screen32-whisper-v1/' + name for name in
               ('common.py', 'adapter.py', 'worker.py', 'setup_runtime.py', 'runtime-lock.json', 'source-lock.json')}
    allowed |= {'tools/asr_native_contracts.py', 'research/source-screen32-v1/runtime_scope.py'}
    require(set(freeze['blind_code']) == allowed, 'no producer or peer context in ASR mount')
    return freeze


def container_command(host, name, image, code, runtime, output, stage, inputs):
    require(stage in ('setup', 'sensevoice'), 'fixed supervisor stage')
    args = host.container_args(name, image, code, runtime, output, 'asr', stage, inputs)
    old_base = '/code/research/source-screen32-v1/'
    if stage == 'setup':
        old = ['python', '-I', '-B', old_base + 'setup_adapter.py', '--profile', 'asr',
               '--stage', 'setup', '--runtime', '/runtime']
        new = ['python', '-I', '-B', '/code/research/source-screen32-whisper-v1/setup_runtime.py']
    else:
        old = ['/runtime/venv/bin/python', '-I', '-B', old_base + 'asr_worker.py', '--model', 'sensevoice',
               '--runtime', '/runtime', '--input', '/input', '--output', '/output']
        new = ['/runtime/venv/bin/python', '-I', '-B', '/code/research/source-screen32-whisper-v1/worker.py']
    require(args[-len(old):] == old, 'unchanged fixed constructor tail required')
    return args[:-len(old)] + new


def terminal_rows(inputs, output):
    """Full fixed denominator; an interrupted started claim always consumes its attempt."""
    job = bind_input(inputs); rows = []
    config_path = output / 'native-config.json'
    config = None; config_error = None
    if config_path.exists():
        try: config = read_json(config_path)
        except Exception as error: config_error = type(error).__name__
    for clip in job['clips']:
        directory = output / clip['audio_id']; started = directory / 'started.json'
        completed = directory / 'completed.json'; raw_path = directory / 'raw-output.json'
        row = {'audio_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'],
               'attempts': int(started.exists()), 'status': 'not_run', 'raw_output_sha256': None,
               'raw_text': None, 'stop_reason': None, 'validation_errors': [], **BOUNDARY}
        if started.exists():
            row['status'] = 'failed_no_retry'
            try:
                if raw_path.exists():
                    raw = read_json(raw_path)
                    row.update(raw_output_sha256=file_hash(raw_path), raw_text=raw['raw_text'],
                               stop_reason=raw['native_capture']['termination']['stop_reason'])
                if completed.exists():
                    done = read_json(completed)
                    require(done['audio_id'] == clip['audio_id'] and done['wav_sha256'] == clip['wav_sha256'] and
                            done['started_sha256'] == file_hash(started) and
                            done['raw_output_sha256'] == file_hash(raw_path) and
                            done['input_capture_sha256'] == file_hash(directory / 'input-capture.json') and
                            done['pre_finalize_sha256'] == file_hash(directory / 'pre-finalize.json'), 'completion byte bindings')
                    raw = read_json(raw_path)
                    require(raw['decoder_completion_validated'] is True and config is not None, 'no fabricated completion')
                    validate_native_capture(config, raw['native_capture'])
                    start = read_json(started)
                    require(start['audio_id'] == clip['audio_id'] and start['wav_sha256'] == clip['wav_sha256'] and
                            start['attempt'] == 1 and start['input_job_sha256'] == file_hash(inputs / 'job.json') and
                            start['native_config_sha256'] == file_hash(config_path), 'started/config/input binding')
                    from adapter import termination
                    count = raw['native_capture']['termination']['step_count']
                    expected_steps = {f'update-{i:03d}.json' for i in range(1, count + 1)}
                    require({p.name for p in directory.glob('update-*.json')} == expected_steps, 'exact durable update set')
                    steps = [read_json(directory / f'update-{i:03d}.json') for i in range(1, count + 1)]
                    require(termination(steps, config['effective_config']) == raw['native_capture']['termination'],
                            'raw termination derived from actual durable steps')
                    pre = read_json(directory / 'pre-finalize.json')
                    require(pre == {'token_ids': steps[-1]['token_ids'], 'last_update_completed': True, 'trace_count': count},
                            'pre-finalize actual step sequence binding')
                    captured_input = read_json(directory / 'input-capture.json')
                    pcm, frames = wav_bytes(inputs / clip['audio_path'], clip['wav_sha256'])
                    floats = b''.join(struct.pack('<f', value[0] / 32768.0) for value in struct.iter_unpack('<h', pcm))
                    require(captured_input['native_input'] == raw['native_capture']['input'] and
                            raw['native_capture']['input']['audio_frames'] == frames and
                            captured_input['pcm16_sha256'] == sha(pcm) and
                            captured_input['float32_source_sha256'] == sha(floats) and
                            captured_input['float32_padded_sha256'] == sha(floats + bytes((480000 - frames) * 4)),
                            'independently reconstructed before/after PCM binding')
                    require(raw['returned_token_ids'] == raw['native_capture']['termination']['generated_token_ids'][:-1],
                            'actual returned token suffix excludes only natural EOT')
                    row['status'] = 'completed_native_eot'
            except Exception as error:
                row['status'] = 'failed_no_retry'
                row['validation_errors'].append(type(error).__name__)
            if config_error is not None:
                row['validation_errors'].append('native_config_' + config_error)
        elif completed.exists() or raw_path.exists():
            row['validation_errors'].append('output_without_started_marker')
        rows.append(row)
    stopped = False
    for row in rows:
        if stopped and row['attempts']:
            row['validation_errors'].append('attempt_after_failed_or_untouched_clip')
        stopped = stopped or row['status'] != 'completed_native_eot'
    return rows


def collect_public(root):
    """Explicit evidence directories only; never runtime, checkpoint, input, or log mounts."""
    artifact = root / 'artifact'; artifact.mkdir()
    files = {}
    for part in ('host-receipt.json', 'image-transfer.json', 'container-setup.json',
                 'container-sensevoice.json', 'terminal-outcomes.json', 'setup', 'whisper'):
        source = root / part
        if not source.exists(): continue
        paths = list(source.rglob('*')) if source.is_dir() else [source]
        for path in paths:
            require(not path.is_symlink(), 'no evidence aliases')
            if not path.is_file(): continue
            regular(path, 8 * 1024**2)
            if path.name.endswith('.tmp'): continue
            name = path.relative_to(root).as_posix()
            require('small.pt' not in name and all(re.fullmatch('[A-Za-z0-9_.-]+', p) for p in Path(name).parts),
                    'public evidence filename boundary')
            destination = artifact / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination); files[name] = identity(destination)
    require(sum(v['bytes'] for v in files.values()) <= LIMITS['output_bytes'], 'artifact bound')
    write_once(artifact / 'artifact-freeze.json', {'schema': 'whisper-screen-artifact-freeze-v1',
               'files': files, 'scope': 'new third-route source-screen raw observations only', **BOUNDARY})


def main(env=None):
    started = time.monotonic()
    env = os.environ if env is None else env
    freeze = admission(env)  # No acquisition/runtime calls before this gate.
    host = historical_host()
    sys.path.insert(0, str(ROOT / 'research/source-screen32-v1'))
    import sense_continuation
    lock = sense_continuation.load_lock(host)
    sense_continuation.verify_retained(lock, host)
    root = Path(env['RUNNER_TEMP']) / (EXPERIMENT + '-' + env['GITHUB_RUN_ID']); root.mkdir()
    code, runtime, inputs = root / 'code', root / 'runtime', root / 'input'
    code.mkdir(); runtime.mkdir()
    sense_continuation.prepare_input(inputs, lock, host)
    for name in freeze['blind_code']:
        target = code / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    deadline = started + LIMITS['job_seconds']
    receipt = {'schema': 'whisper-screen-host-receipt-v1', 'status': 'started',
        'experiment': EXPERIMENT, 'head_sha': env['GITHUB_SHA'], 'run_id': env['GITHUB_RUN_ID'],
        'run_attempt': 1, 'run_number': 1, 'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json'),
        'fixed_model': 'openai/whisper-small-native', 'new_model_selection_after_results': False,
        'prospective_three_model_roster': False,
        'interpretation': 'third model selected after historical two-route screen; exploratory source-screen comparison',
        'prior_runs_unchanged': [38018029787, 38021467257], 'limits': LIMITS,
        'stages': {'setup': 'not_run', 'whisper': 'not_run'}, **BOUNDARY}
    atomic(root / 'host-receipt.json', receipt)
    # Expose destination early, so completed partial evidence can upload after failure.
    if env.get('GITHUB_OUTPUT'):
        with open(env['GITHUB_OUTPUT'], 'a') as stream: stream.write('evidence_root=' + str(root) + '\n')
    failed = False
    def interrupted(signum, frame):
        raise RuntimeError('host termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        require(shutil.disk_usage(root).free >= 16 * 1024**3, 'host disk reserve')
        receipt['capacity'] = host.host_capacity()
        image_lock = read_json(ROOT / 'research/source-screen32-v1/container-lock.json')
        host.pull_image(image_lock, root / 'image-transfer.json')
        for stage, label in (('setup', 'setup'), ('sensevoice', 'whisper')):
            host.host_capacity()
            output = root / label; output.mkdir()
            name = EXPERIMENT + '-' + env['GITHUB_RUN_ID'] + '-' + label
            args = container_command(host, name, image_lock['image'], code, runtime, output, stage, inputs)
            receipt['stages'][label] = 'started'; atomic(root / 'host-receipt.json', receipt)
            receipt.setdefault('container_argv_sha256', {})[label] = __import__('hashlib').sha256(canonical(args)).hexdigest()
            host.supervise_container(args, name, output, stage, deadline)
            receipt['stages'][label] = 'complete'; atomic(root / 'host-receipt.json', receipt)
        receipt['status'] = 'all_sixteen_completed'
    except BaseException as error:
        failed = True
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        for stage, status in receipt['stages'].items():
            if status == 'started': receipt['stages'][stage] = 'failed_no_retry'
    finally:
        output = root / 'whisper'; output.mkdir(exist_ok=True)
        try:
            rows = terminal_rows(inputs, output)
            write_once(root / 'terminal-outcomes.json', rows)
            receipt['completed'] = sum(r['status'] == 'completed_native_eot' for r in rows)
            receipt['attempts'] = sum(r['attempts'] for r in rows)
            receipt['terminal_validation_errors'] = sum(bool(r['validation_errors']) for r in rows)
            if receipt['terminal_validation_errors']:
                failed = True; receipt['status'] = 'terminal_validation_failed'
            require(receipt['status'] != 'all_sixteen_completed' or receipt['completed'] == 16, 'sixteen complete required')
        except Exception as error:
            failed = True; receipt.update(status='terminal_validation_failed', terminal_error_type=type(error).__name__)
        receipt['wall_seconds'] = time.monotonic() - started
        atomic(root / 'host-receipt.json', receipt)
        collect_public(root)
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'stopped_no_retry', 'error_type': type(error).__name__}))
        raise SystemExit(1)
