"""Admitted bounded acquisition, then the unchanged blind inference constructor."""
from pathlib import Path
import re
import shutil
import signal
import subprocess
import math
import sys
import time
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
from common import LIMITS, BOUNDARY, require, read_json, atomic, write_once, identity, regular, canonical
import hosted_run as original
import bounded_host


class CampaignDocker:
    """Historical operations plus one USR1 seal of the current admitted broker.

    Record uncertain create before invoking Docker, so cleanup remains possible.
    A signal requires a confirmed create; no unknown/stale name or retry is valid.
    """
    def __init__(self, host, run_id):
        require(re.fullmatch('[1-9][0-9]{0,19}', run_id), 'fixed campaign run identity')
        self.host = host
        assets = ['wheel-' + str(i).zfill(2) for i in range(1, 23)] + ['source', 'model']
        self.allowed = {'kws-bound-' + run_id + '-' + asset + '-broker' for asset in assets}
        self.current = None
        self.confirmed = False
        self.signalled = False

    def __call__(self, *args, timeout=30):
        require(bool(args), 'fixed Docker operation')
        if args[0] == 'kill':
            require(len(args) == 3 and args[1] == '--signal=USR1' and
                    args[2] in self.allowed and args[2] == self.current and
                    self.confirmed and not self.signalled, 'only current broker one-shot seal')
            require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 30,
                    'bounded broker signal timeout')
            self.signalled = True
            try:
                result = subprocess.run(['docker', *args], check=True, timeout=timeout, capture_output=True)
                require(len(result.stdout) + len(result.stderr) <= 2 * 1024**2, 'bounded broker signal response')
                return result.stdout.decode('utf-8')
            except BaseException as error:
                error.screen32_diagnostic = self.host.public_failure(error, 'broker_seal')
                raise
        broker = None
        if args[0] == 'create':
            require(args.count('--name') == 1, 'one fixed container name')
            index = args.index('--name')
            require(index + 1 < len(args), 'container name required')
            name = args[index + 1]
            if name.endswith('-broker'):
                require(name in self.allowed and self.current is None, 'single current campaign broker')
                broker = name
                self.current = name; self.confirmed = False; self.signalled = False
        result = self.host.docker(*args, timeout=timeout)
        if broker is not None:
            self.confirmed = True
        if args[0] == 'rm' and args == ('rm', '--force', self.current):
            self.current = None; self.confirmed = False; self.signalled = False
        return result


def collect_public(root):
    """Only fixed JSON evidence and hash-locked official source may be published."""
    artifact = root / 'artifact'; artifact.mkdir()
    allowed = {'host-receipt.json', 'image-transfer.json', 'container-sensevoice.json', 'terminal-outcomes.json'}
    allowed |= {'setup/' + n for n in ('runtime-lock.json', 'source-lock.json', 'asset-lock.json',
        'installed-inventory.json', 'download-receipt.json', 'setup-receipt.json', 'phased-setup-failure.json')}
    allowed |= {'setup/phase-' + phase + '-' + suffix + '.json'
                for phase in ('install', 'source', 'model', 'finalize') for suffix in ('progress', 'complete')}
    allowed |= {'whisper/' + n for n in ('blind-job.json', 'kernel-scope.json', 'progress.json',
        'native-config.json', 'load-receipt.json', 'worker-complete.json', 'worker-failure.json')}
    job = read_json(root / 'input/job.json')
    for clip in job['clips']:
        prefix = 'whisper/' + clip['audio_id'] + '/'
        allowed |= {prefix + n for n in ('started.json', 'completed.json', 'input-capture.json',
                                        'pre-finalize.json', 'raw-output.json')}
        allowed |= {prefix + 'update-' + str(i).zfill(3) + '.json' for i in range(1, 449)}
    allowed |= {'setup/bounded-acquisition/' + n for n in bounded_host.public_metadata_names()}
    official = { 'setup/official-source/' + r['path']: r for r in read_json(BASE / 'source-lock.json')['files'] }
    files = {}
    for name in sorted(allowed | set(official)):
        path = root / name
        if not path.exists(): continue
        regular(path, 8 * 1024**2)
        if name in official:
            row = official[name]
            require(identity(path) == {'bytes': row['size_bytes'], 'sha256': row['sha256']}, 'official source byte identity')
        else:
            read_json(path)
        destination = artifact / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination); files[name] = identity(destination)
    require(sum(v['bytes'] for v in files.values()) <= LIMITS['output_bytes'], 'artifact bound')
    write_once(artifact / 'artifact-freeze.json', {'schema': 'whisper-screen-artifact-freeze-v1',
        'files': files, 'scope': 'new third-route source-screen raw observations only', **BOUNDARY})


def execute(env, freeze, started, experiment, source_freeze_sha256, campaign=None):
    """Shared fixed runtime, called only after an entry point's reviewed admission."""
    host = original.historical_host()
    sys.path.insert(0, str(ROOT / 'research/source-screen32-v1'))
    import sense_continuation
    lock = sense_continuation.load_lock(host)
    sense_continuation.verify_retained(lock, host)
    root = Path(env['RUNNER_TEMP']) / (experiment + '-' + env['GITHUB_RUN_ID']); root.mkdir()
    code, runtime, inputs = root / 'code', root / 'runtime', root / 'input'
    setup_code = root / 'setup-code'
    code.mkdir(); setup_code.mkdir(); runtime.mkdir()
    sense_continuation.prepare_input(inputs, lock, host)
    for kind, destination in (('inference_code', code), ('setup_code', setup_code)):
        for name in freeze[kind]:
            target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    deadline = started + LIMITS['job_seconds']
    receipt = {'schema': 'whisper-screen-host-receipt-v1', 'status': 'started',
        'experiment': experiment, 'head_sha': env['GITHUB_SHA'], 'run_id': env['GITHUB_RUN_ID'],
        'run_attempt': 1, 'run_number': 1, 'source_freeze_sha256': source_freeze_sha256,
        'fixed_model': 'openai/whisper-small-native', 'new_model_selection_after_results': False,
        'prospective_three_model_roster': False,
        'interpretation': 'third model selected after historical two-route screen; exploratory source-screen comparison',
        'prior_runs_unchanged': [38018029787, 38021467257], 'limits': LIMITS,
        'stages': {'setup': 'not_run', 'whisper': 'not_run'}, **BOUNDARY}
    if campaign is not None:
        receipt['campaign'] = campaign
        receipt['prior_runs_unchanged'].extend(r['run_id'] for r in campaign['prior_runs'])
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
        host.host_capacity()
        output = root / 'setup'; output.mkdir(mode=0o700)
        receipt['stages']['setup'] = 'started'; atomic(root / 'host-receipt.json', receipt)
        bounded_host.acquire_runtime(docker=CampaignDocker(host, env['GITHUB_RUN_ID']), setup_code=setup_code, runtime=runtime,
            output=output, root=root, job_deadline=deadline, run_id=env['GITHUB_RUN_ID'],
            image=image_lock['image'])
        require(time.monotonic() < deadline, 'host deadline before inference')
        receipt['stages']['setup'] = 'complete'; atomic(root / 'host-receipt.json', receipt)
        host.host_capacity()
        output = root / 'whisper'; output.mkdir()
        name = experiment + '-' + env['GITHUB_RUN_ID'] + '-whisper'
        args = original.container_command(host, name, image_lock['image'], code, runtime,
                                          output, 'sensevoice', inputs)
        receipt['stages']['whisper'] = 'started'; atomic(root / 'host-receipt.json', receipt)
        receipt.setdefault('container_argv_sha256', {})['whisper'] = __import__('hashlib').sha256(canonical(args)).hexdigest()
        host.supervise_container(args, name, output, 'sensevoice', deadline)
        receipt['stages']['whisper'] = 'complete'; atomic(root / 'host-receipt.json', receipt)
        receipt['status'] = 'all_sixteen_completed'
    except BaseException as error:
        failed = True
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        for stage, status in receipt['stages'].items():
            if status == 'started': receipt['stages'][stage] = 'failed_no_retry'
    finally:
        output = root / 'whisper'; output.mkdir(exist_ok=True)
        try:
            rows = original.terminal_rows(inputs, output)
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

