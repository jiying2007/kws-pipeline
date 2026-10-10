"""Fixed free-hosted Docker orchestration; never accepts an arbitrary command.

Not auto-dispatched. A reviewed source freeze, explicit admission, first branch
creation and workflow run/attempt one are all required before any image pull.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from contract import expected_plan, validate_generation_ledger, blind_job, require, canonical
from tts_worker import atomic, file_hash, initial_ledger
from runtime_scope import MEMORY, PIDS

GIB = 1024 ** 3
IMAGE_TRANSFER_RESERVE = 64 * 1024 ** 2
BRANCH = 'research/qwen16-voicedesign-once-v1'
EXPERIMENT = 'qwen16-voicedesign-once-v1'


def bounded_json(path, maximum=1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum, 'bounded regular JSON')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON field'); result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def verify_admission(env):
    release = bounded_json(HERE / 'execution-release.json')
    require(release == {'schema': 'screen32-execution-release-v1', 'experiment': EXPERIMENT,
             'approved': True, 'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json')},
            'exact reviewed source admission required')
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + BRANCH,
                'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1'}
    require(all(env.get(key) == value for key, value in expected.items()), 'one-shot workflow identity')
    require(re.fullmatch(r'[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch(r'[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')), 'immutable head/run identity')
    event = bounded_json(env['GITHUB_EVENT_PATH'])
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('private') is False and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'],
            'first public branch creation required')
    freeze = bounded_json(HERE / 'execution-freeze.json')
    require(freeze.get('schema') == 'screen32-execution-source-freeze-v1', 'source freeze schema')
    require(set(freeze) == {'schema', 'files', 'profiles'}, 'source freeze fields')
    for name, digest in freeze['files'].items():
        require(name.startswith('research/') and '..' not in Path(name).parts and
                re.fullmatch(r'[0-9a-f]{64}', digest), 'source path/hash')
        path = ROOT / name
        require(path.is_file() and not path.is_symlink() and file_hash(path) == digest, 'reviewed source drift')
    workflow = ROOT / '.github/workflows/source-screen32-run.yml'
    require(workflow.is_file() and workflow.read_bytes() == (HERE / 'workflow.yml').read_bytes(), 'activated workflow drift')
    return freeze


def docker(*args, timeout=30):
    result = subprocess.run(['docker', *args], check=True, timeout=timeout, capture_output=True, text=True)
    require(len(result.stdout) <= 2 * 1024 ** 2, 'oversized Docker response')
    return result.stdout


def host_capacity():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':'); memory[key] = value.split()
    available = int(memory['MemAvailable'][0]) * 1024
    require(available >= MEMORY + 2 * GIB, 'current host memory reserve unavailable')
    info = json.loads(docker('info', '--format', '{{json .}}'))
    require(info.get('CgroupVersion') == '2' and info.get('OSType') == 'linux' and
            info.get('Architecture') in ('x86_64', 'amd64') and info.get('NCPU', 0) >= 4,
            'existing supported Linux cgroup-v2 Docker required')
    require(any('seccomp' in item for item in info.get('SecurityOptions', [])), 'Docker seccomp unavailable')
    return {'available_memory_bytes': available, 'cpu_count': info['NCPU'], 'cgroup_version': '2'}


def network_received():
    return {path.parent.parent.name: int(path.read_text())
            for path in Path('/sys/class/net').glob('*/statistics/rx_bytes') if path.parent.parent.name != 'lo'}


def pull_image(lock, receipt_path):
    """Monitored pull reservation, NOT a hard Docker-daemon transfer cutoff.

    A client cancellation does not prove daemon downloads stopped. On any failure,
    no setup is started and the retained receipt leaves cumulative bytes unknown.
    The daemon and host security settings are never changed or shut down here.
    """
    observation = {'schema': 'screen32-image-transfer-observation-v1', 'status': 'started',
        'image': lock['image'], 'reservation_bytes': IMAGE_TRANSFER_RESERVE,
        'reservation_kind': 'monitored_host_receive_observation_not_hard_transfer_limit',
        'host_received_delta_bytes': None, 'daemon_completion_verified': False,
        'cumulative_image_transfer_bytes_verified': False, 'pull_client_attempts': 1}
    atomic(receipt_path, observation)
    process = None
    try:
        before = network_received(); require(before, 'host transfer counters unavailable')
        started = time.monotonic()
        process = subprocess.Popen(['docker', 'pull', '--platform', 'linux/amd64', lock['image']],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while process.poll() is None:
            after = network_received()
            require(after.keys() == before.keys() and all(after[key] >= before[key] for key in before),
                    'host transfer counter identity changed')
            observation['host_received_delta_bytes'] = sum(after[key] - before[key] for key in before)
            require(observation['host_received_delta_bytes'] <= IMAGE_TRANSFER_RESERVE,
                    'observed image reservation exceeded; no setup or generation')
            require(time.monotonic() - started <= 180, 'image pull client wall limit')
            time.sleep(0.1)
        require(process.returncode == 0, 'single image pull failed; no retry')
        after = network_received()
        require(after.keys() == before.keys() and all(after[key] >= before[key] for key in before), 'transfer counters changed')
        observation['host_received_delta_bytes'] = sum(after[key] - before[key] for key in before)
        require(observation['host_received_delta_bytes'] <= IMAGE_TRANSFER_RESERVE, 'observed image reservation exceeded')
        image = json.loads(docker('image', 'inspect', lock['image']))[0]
        require(image['Id'] == lock['manifest']['config']['digest'] and image['Architecture'] == 'amd64'
                and image['Os'] == 'linux', 'exact official image config/platform mismatch')
        observation.update(status='completed_observed_within_reservation', daemon_completion_verified=True,
                           image_config_digest=image['Id'],
                           byte_scope='all_host_receive_traffic_during_completed_pull_conservative_not_exact_daemon_accounting')
        return dict(observation)
    except BaseException as error:
        observation.update(status='stopped_before_setup_daemon_completion_unverified',
                           error_type=type(error).__name__, daemon_fetch_may_continue=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
        atomic(receipt_path, observation)


def stage_code(freeze, profile, destination):
    names = freeze['profiles'][profile]
    require(type(names) is list and len(names) == len(set(names)) and names, 'source membership')
    for name in names:
        require(name in freeze['files'], 'unreviewed staged source')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        require(file_hash(target) == freeze['files'][name], 'staged source identity')


def container_args(name, image, code, runtime, output, profile, stage, input_dir=None):
    require(profile in ('tts', 'asr') and stage in ('setup', 'tts', 'qwen06', 'sensevoice'), 'fixed container stage')
    require((profile == 'tts') == (stage in ('setup', 'tts') and input_dir is None), 'profile/input separation')
    uid, gid = os.getuid(), os.getgid(); require(uid != 0, 'hosted unprivileged runner account required')
    args = ['create', '--name', name, '--platform', 'linux/amd64', '--pull', 'never',
            '--memory', str(MEMORY), '--memory-swap', str(MEMORY), '--cpus', '4',
            '--pids-limit', str(PIDS), '--cgroupns', 'private', '--ipc', 'private',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', f'{uid}:{gid}', '--network', 'bridge' if stage == 'setup' else 'none',
            '--ulimit', 'core=0', '--ulimit', 'nofile=1024:1024',
            '--ulimit', 'fsize=' + str(4 * GIB if stage == 'setup' else 8 * 1024 ** 2),
            '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1',
            '--tmpfs', f'/scratch:rw,noexec,nosuid,nodev,size=268435456,uid={uid},gid={gid},mode=0700',
            '--tmpfs', f'/tmp:rw,noexec,nosuid,nodev,size=134217728,uid={uid},gid={gid},mode=0700',
            '--mount', f'type=bind,src={code},dst=/code,readonly',
            '--mount', f'type=bind,src={runtime},dst=/runtime' + ('' if stage == 'setup' else ',readonly'),
            '--mount', f'type=bind,src={output},dst=/output',
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONNOUSERSITE=1',
            '--env', 'ORT_DISABLE_TELEMETRY=1', '--env', 'CUDA_VISIBLE_DEVICES=',
            '--env', 'OMP_NUM_THREADS=4', '--env', 'MKL_NUM_THREADS=4',
            '--env', 'TOKENIZERS_PARALLELISM=false']
    base = '/code/research/source-screen32-v1/'
    if stage == 'setup':
        command = ['python', '-I', '-B', base + 'setup_adapter.py', '--profile', profile,
                   '--stage', 'setup', '--runtime', '/runtime']
    elif stage == 'tts':
        command = ['/runtime/venv/bin/python', '-I', '-B', base + 'tts_worker.py',
                   '--runtime', '/runtime', '--output', '/output']
    else:
        require(input_dir is not None, 'audio-only input required')
        args += ['--mount', f'type=bind,src={input_dir},dst=/input,readonly']
        command = ['/runtime/venv/bin/python', '-I', '-B', base + 'asr_worker.py', '--model', stage,
                   '--runtime', '/runtime', '--input', '/input', '--output', '/output']
    return args + [image, *command]


def validate_container_inspect(record, stage):
    host = record['HostConfig']
    require(host.get('Memory') == MEMORY and host.get('MemorySwap') == MEMORY and
            host.get('NanoCpus') == 4_000_000_000 and host.get('PidsLimit') == PIDS,
            'Docker effective host configuration differs')
    require(host.get('ReadonlyRootfs') is True and host.get('Privileged') is False and
            host.get('CgroupnsMode') == 'private' and host.get('PidMode') == '' and
            host.get('IpcMode') == 'private' and host.get('CapAdd') in (None, []) and
            set(host.get('CapDrop') or []) == {'ALL'} and
            any(item in ('no-new-privileges', 'no-new-privileges:true') for item in host.get('SecurityOpt', [])),
            'container privilege/isolation mismatch')
    require(host.get('NetworkMode') == ('bridge' if stage == 'setup' else 'none'), 'container network mode')
    mounts = {row['Destination']: row for row in record['Mounts'] if row['Type'] == 'bind'}
    expected = {'/code', '/runtime', '/output'} | ({'/input'} if stage in ('qwen06', 'sensevoice') else set())
    require(set(mounts) == expected and mounts['/code']['RW'] is False and
            mounts['/runtime']['RW'] is (stage == 'setup') and mounts['/output']['RW'] is True and
            ('/input' not in mounts or mounts['/input']['RW'] is False), 'read-only input mount boundary')


def supervise_container(args, name, output, stage, deadline):
    started = time.monotonic(); phase_start = started; previous = None
    maximum = 1200 if stage == 'setup' else 4800
    created = False
    try:
        docker(*args); created = True
        record = json.loads(docker('inspect', name))[0]
        validate_container_inspect(record, stage)
        docker('start', name)
        while True:
            now = time.monotonic()
            require(now < deadline and now - started <= maximum, 'stage/job hard wall limit')
            progress_path = output / 'progress.json'
            if stage != 'setup' and progress_path.exists():
                progress = bounded_json(progress_path, 4096)
                require(set(progress) == {'stage', 'index', 'started_monotonic'} and
                        progress['stage'] in ('load', 'cell') and type(progress['index']) is int and
                        0 <= progress['index'] <= 16, 'worker progress shape')
                current = (progress['stage'], progress['index'])
                if current != previous:
                    require(previous is None or current[1] > previous[1], 'worker progress cannot restart')
                    previous = current; phase_start = now
                require(now - phase_start <= (300 if current[0] == 'cell' else 1200), 'cell/load wall limit')
            elif stage != 'setup':
                require(now - started <= 1200, 'worker startup wall limit')
            require(sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) <= 64 * 1024 ** 2,
                    'output disk bound exceeded')
            state = json.loads(docker('inspect', '--format', '{{json .State}}', name))
            if not state['Running']:
                require(state['ExitCode'] == 0 and not state.get('OOMKilled'), 'container failed; no restart')
                return {'status': 'complete', 'wall_seconds': now - started, 'exit_code': state['ExitCode']}
            time.sleep(0.5)
    finally:
        if created:
            # Killing/removing the one dedicated container kills all its children.
            docker('rm', '--force', name)


def terminal_tts(output):
    target = output / 'tts-receipt.json'
    if target.exists():
        receipt = bounded_json(target)
        for row in receipt['outcomes']:
            if row['status'] == 'attempt_started': row['status'] = 'failed_no_retry'
        if receipt['status'] == 'started': receipt['status'] = 'stopped_no_retry'
    else:
        receipt = {'schema': 'screen32-tts-receipt-v1', 'status': 'stopped_before_first_cell',
                   'ledger': initial_ledger(), 'outcomes': [], 'human_review': 'PENDING',
                   'human_gold': False, 'training_admitted': False}
    # A kill can land after the exclusive claim and before the next receipt replace.
    # Claim presence consumes the cell even if its tiny body was interrupted.
    claims = {path.name: path for path in output.glob('*.claim')}
    allowed = {row['cell_id'] + '.claim' for row in receipt['ledger'][:16]}
    require(set(claims) <= allowed, 'unexpected TTS durable claim')
    claimed = [index for index, row in enumerate(receipt['ledger'][:16]) if row['cell_id'] + '.claim' in claims]
    require(claimed == list(range(len(claimed))), 'non-prefix TTS claims')
    outcomes = {row['cell_id']: row for row in receipt['outcomes']}
    for index, row in enumerate(receipt['ledger'][:16]):
        claim = claims.get(row['cell_id'] + '.claim')
        if claim is None:
            require(row['attempts'] == 0, 'attempt receipt without durable TTS claim')
            continue
        require(not claim.is_symlink() and claim.stat().st_nlink == 1 and claim.stat().st_size <= 64,
                'invalid TTS claim type/size')
        valid_body = claim.read_bytes() == (row['cell_id'] + '\n').encode()
        if row['status'] != 'GENERATED' or not valid_body:
            row.update(status='FAILED_NO_RETRY', attempts=1, audio=None)
            outcome = outcomes.get(row['cell_id'])
            if outcome is None:
                outcome = {'cell_id': row['cell_id'], 'attempts': 1}
                receipt['outcomes'].append(outcome)
            outcome.update(status='failed_no_retry', terminal_reason='durable_claim_without_completed_candidate')
            receipt['status'] = 'stopped_no_retry'
    validate_generation_ledger(expected_plan(), receipt['ledger'])
    atomic(target, receipt)
    return receipt


def pack_blind(output, destination, ledger):
    """Copy only byte-revalidated audio. No archive, prompts or producer receipts."""
    job = blind_job(expected_plan(), ledger)
    destination.mkdir(); (destination / 'audio').mkdir()
    by_hash = {row['audio']['wav_sha256']: row for row in ledger if row['status'] == 'GENERATED'}
    for clip in job['clips']:
        row = by_hash[clip['wav_sha256']]; source = output / (row['cell_id'] + '.wav')
        require(not source.is_symlink() and file_hash(source) == clip['wav_sha256'], 'derived WAV drift')
        with wave.open(str(source), 'rb') as handle:
            require((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes()) ==
                    (16000, 1, 2, row['audio']['frames']), 'blind PCM geometry')
            pcm = handle.readframes(handle.getnframes())
        require(hashlib.sha256(pcm).hexdigest() == row['audio']['pcm_sha256'], 'blind decoded PCM drift')
        shutil.copyfile(source, destination / clip['audio_path'])
    atomic(destination / 'job.json', job)
    require(sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()) <= 8 * 1024 ** 2,
            'blind handoff bound')
    return {'job_sha256': file_hash(destination / 'job.json'), 'clips': len(job['clips']),
            'files': {str(path.relative_to(destination)): file_hash(path)
                      for path in sorted(destination.rglob('*')) if path.is_file()}}


def validate_blind_input(directory):
    from contract import validate_blind_job
    job = bounded_json(directory / 'job.json', 65536)
    validate_blind_job(job)
    require(len(job['clips']) <= 16 and [row['wav_sha256'] for row in job['clips']] ==
            sorted(row['wav_sha256'] for row in job['clips']), 'fixed Qwen blind subset/order')
    expected = {'job.json'} | {row['audio_path'] for row in job['clips']}
    actual = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'blind aliases/special files')
        if path.is_file(): actual.add(path.relative_to(directory).as_posix())
    require(actual == expected, 'only audio and blind job may enter ASR')
    require(sum((directory / name).stat().st_size for name in actual) <= 8 * 1024 ** 2, 'blind byte cap')
    for row in job['clips']:
        require(file_hash(directory / row['audio_path']) == row['wav_sha256'], 'blind file hash')
    return job


def freeze_primary(root, job):
    """Audio-only post-decode projection; raw receipts remain the authority."""
    from contract import freeze_disputes
    primary, inputs = {}, {}
    ids = [clip['audio_id'] for clip in job['clips']]
    for model in ('qwen06', 'sensevoice'):
        path = root / model / 'terminal-outcomes.json'
        rows = bounded_json(path)
        require([row['opaque_id'] for row in rows] == ids, 'terminal ASR denominator')
        inputs[model] = file_hash(path)
        primary[model] = []
        for row in rows:
            flags = list(row['quality_flags'])
            if row['status'] == 'success' and row['completeness'] != 'complete':
                flags.append('decoder_completeness_' + row['completeness'])
            primary[model].append({'audio_id': row['opaque_id'],
                'status': 'complete' if row['status'] == 'success' else 'not_run' if row['status'] == 'not_run' else 'failed',
                'raw_text': row['raw_text'] if row['status'] == 'success' else None, 'quality_flags': flags})
    atomic(root / 'primary-raw.json', primary)
    raw = (root / 'primary-raw.json').read_bytes()
    atomic(root / 'primary-raw-freeze.json', {'schema': 'screen32-primary-raw-freeze-v1',
          'primary_raw_sha256': file_hash(root / 'primary-raw.json'), 'terminal_outcomes_sha256': inputs,
          'intended_text_joined': False, 'human_gold': False, 'training_admitted': False})
    atomic(root / 'frozen-disputes.json', freeze_disputes(job, raw, file_hash(root / 'primary-raw.json')))


def public_evidence(root, phase):
    """Allowlist bounded outputs only; never copy setup/runtime/model/download trees."""
    destination = root / 'artifact'; destination.mkdir()
    sources = [root / 'host-receipt.json']; incomplete = []
    for name in ('primary-raw.json', 'primary-raw-freeze.json', 'frozen-disputes.json', 'image-transfer.json'):
        if (root / name).exists(): sources.append(root / name)
    setup_receipt = root / 'runtime/setup-receipt.json'
    if setup_receipt.exists():
        summary = {'schema': 'screen32-setup-public-v1', 'completed_verification': False,
                   'verification_evidence': 'UNAVAILABLE_OR_TRUNCATED', 'model_calls': 0}
        try:
            value = bounded_json(setup_receipt, 8 * 1024 ** 2)
            require(type(value) is dict, 'setup receipt shape')
            summary.update(completed_verification='verification' in value,
                           verification_evidence='BOUNDED_RECEIPT_PARSED')
            for field in ('actual_download_bytes', 'wall_seconds'):
                item = value.get(field)
                if type(item) in (int, float) and 0 <= item <= 64 * GIB:
                    summary[field] = item
            exception = value.get('exception')
            if type(exception) is dict and re.fullmatch(r'[A-Za-z]{1,64}', str(exception.get('type', ''))):
                summary['exception_type'] = exception['type']
        except (ValueError, TypeError, KeyError, OSError):
            pass
        atomic(root / 'setup-public.json', summary); sources.append(root / 'setup-public.json')
    names = ('tts',) if phase == 'tts' else ('qwen06', 'sensevoice')
    for name in names:
        directory = root / name
        for path in directory.rglob('*'):
            require(not path.is_symlink() and (path.is_dir() or path.is_file()), 'unsafe output file type')
            if path.is_file():
                known_temporary = path.name in {'tts-receipt.json.tmp', 'progress.json.tmp', 'runtime-observations.json.tmp', 'kernel-scope.json.tmp', 'worker-failure.json.tmp'} or re.fullmatch(r'\.progress-(?:[0-9]|1[0-6])\.tmp', path.name)
                if known_temporary:
                    incomplete.append(str(path.relative_to(root))); continue
                require(path.suffix in ('.json', '.wav', '.claim') and path.stat().st_nlink == 1 and
                        path.stat().st_size <= 8 * 1024 ** 2, 'output file bound/type')
                require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', part) for part in path.relative_to(directory).parts),
                        'output filename boundary')
                sources.append(path)
    require(sum(path.stat().st_size for path in sources) <= 64 * 1024 ** 2, 'whole evidence artifact bound')
    files = {}
    for source in sources:
        relative = source.relative_to(root)
        target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); files[str(relative)] = file_hash(target)
    atomic(destination / 'artifact-freeze.json', {'schema': 'screen32-evidence-freeze-v1',
          'files': files, 'incomplete_temporary_files_excluded': incomplete,
          'human_gold': False, 'training_admitted': False})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tts', 'asr'), required=True)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    freeze = verify_admission(os.environ)
    require((args.phase == 'asr') == (args.input is not None), 'exact input/phase boundary')
    root = Path(os.environ['RUNNER_TEMP']) / (EXPERIMENT + '-' + os.environ['GITHUB_RUN_ID'] + '-' + args.phase)
    root.mkdir(exist_ok=False)  # Durable local stage claim before any acquisition.
    require(shutil.disk_usage(root).free >= 16 * GIB, 'current 16GiB disk headroom required')
    started = time.monotonic(); deadline = started + 7200
    input_job = None
    if args.phase == 'asr':
        input_job = validate_blind_input(args.input)
        require(file_hash(args.input / 'job.json') == os.environ.get('SCREEN32_BLIND_JOB_SHA256'), 'producer-bound blind input')
    code, runtime = root / 'code', root / 'runtime'; code.mkdir(); runtime.mkdir()
    stage_code(freeze, args.phase, code)
    receipt = {'schema': 'screen32-host-run-v1', 'status': 'started', 'phase': args.phase,
               'head_sha': os.environ['GITHUB_SHA'], 'run_id': os.environ['GITHUB_RUN_ID'],
               'stages': {stage: {'status': 'not_run'} for stage in (['setup', 'tts'] if args.phase == 'tts' else ['setup', 'qwen06', 'sensevoice'])}, 'human_review': 'PENDING', 'human_gold': False, 'training_admitted': False}
    target = root / 'host-receipt.json'
    atomic(target, receipt)
    try:
        if input_job is not None and not input_job['clips']:
            receipt['status'] = 'empty_generated_subset'
            return
        receipt['capacity'] = host_capacity()
        lock = bounded_json(HERE / 'container-lock.json')
        receipt['image'] = pull_image(lock, root / 'image-transfer.json')
        stages = ['setup', 'tts'] if args.phase == 'tts' else ['setup', 'qwen06', 'sensevoice']
        for stage in stages:
            # Recheck current host reserve immediately before creating each scope.
            host_capacity()
            output = root / stage; output.mkdir()
            name = EXPERIMENT + '-' + os.environ['GITHUB_RUN_ID'] + '-' + args.phase + '-' + stage
            command = container_args(name, lock['image'], code, runtime, output, args.phase, stage, args.input)
            receipt['stages'][stage] = {'status': 'started'}; atomic(target, receipt)
            receipt['stages'][stage] = supervise_container(command, name, output, stage, deadline)
            atomic(target, receipt)
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        for record in receipt['stages'].values():
            if record['status'] == 'started': record['status'] = 'failed_no_retry'
        raise
    finally:
        if args.phase == 'tts':
            output = root / 'tts'; output.mkdir(exist_ok=True)
            terminal = terminal_tts(output)
            receipt['blind'] = pack_blind(output, root / 'blind', terminal['ledger'])
        else:
            from asr_worker import terminal_outcomes
            for model in ('qwen06', 'sensevoice'):
                output = root / model; output.mkdir(exist_ok=True)
                terminal_outcomes(args.input, output, model)
            freeze_primary(root, input_job)
        receipt['wall_seconds'] = time.monotonic() - started
        atomic(target, receipt)
        public_evidence(root, args.phase)
        # Only locations of bounded evidence are exposed; no raw runtime logs.
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as handle:
                handle.write('evidence_root=' + str(root) + '\n')
                if args.phase == 'tts':
                    handle.write('blind_job_sha256=' + receipt['blind']['job_sha256'] + '\n')
                    handle.write('blind_clips=' + str(receipt['blind']['clips']) + '\n')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status': 'stopped_no_retry', 'error_type': type(error).__name__}))
        raise SystemExit(1)
