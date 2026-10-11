"""One source-PR, zero-model Docker lifecycle smoke; no native admission.

Only cpu_gate.py is staged. The child is the fixed stdlib ``python -I -B -c pass``.
The actual MeteredDocker owns cleanup and final CPU accounting; this module does
not introduce another removal state machine. Failures stop without a second run.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import hosted_run as host
from campaign_contract import canonical, digest, identity, read_json, require
from cpu_budget import CpuLedger

REPOSITORY = 'jiying2007/kws-pipeline'
JOB = 'cleanup-lifecycle-smoke'
BRANCH = 'fix/qwen12-native-cleanup-source-20261011'
IMAGE = 'docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a'
IMAGE_CONFIG = 'sha256:febd0be41adb897a0ab8f1f1c693d8912669ea60c4940e076e9946b60e210ef0'
GATE = 'research/qwen12-native-admission-v1/cpu_gate.py'
COMMAND = ['python', '-I', '-B', '-c', 'pass']
HOST_SECONDS = 180
IMAGE_SECONDS = 120
LIFECYCLE_SECONDS = 30
CPU_SECONDS = 60
RECEIPT_BYTES = 32768
SOURCE_FILES = (
    '.github/workflows/qwen12-native-admission-v1.yml',
    'research/qwen12-native-admission-v1/cleanup_lifecycle_smoke.py',
    'research/qwen12-native-admission-v1/hosted_run.py',
    'research/qwen12-native-admission-v1/campaign_contract.py',
    'research/qwen12-native-admission-v1/execution-release.json',
    'research/qwen12-native-admission-v1/cpu_budget.py', GATE,
    'research/source-screen32-v1/hosted_run.py',
    'research/source-screen32-v1/container-lock.json',
    'research/source-screen32-v1/contract.py',
    'research/source-screen32-v1/tts_worker.py',
    'research/source-screen32-v1/qwen_adapter.py',
    'research/source-screen32-v1/runtime_scope.py',
)


def admission(env, event):
    """No Docker, filesystem mutation, network, or release-gate dependency."""
    expected = {'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_EVENT_NAME': 'pull_request',
                'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_JOB': JOB, 'GITHUB_ACTIONS': 'true',
                'GITHUB_HEAD_REF': BRANCH,
                'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64'}
    require(all(env.get(k) == v for k, v in expected.items()), 'fixed first-attempt hosted source PR')
    require(re.fullmatch('[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch('[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')), 'actual immutable run identity')
    require(event.get('action') == 'opened', 'only the original source PR opening')
    number = event.get('number')
    require(type(number) is int and number > 0 and env.get('GITHUB_REF') == f'refs/pull/{number}/merge',
            'exact PR merge ref')
    pr = event.get('pull_request', {})
    for repo in (event.get('repository', {}), pr.get('head', {}).get('repo', {}),
                 pr.get('base', {}).get('repo', {})):
        require(repo.get('full_name') == REPOSITORY and repo.get('private') is False,
                'canonical public same-repository PR only')
    require(pr.get('head', {}).get('ref') == BRANCH and
            re.fullmatch('[0-9a-f]{40}', pr.get('head', {}).get('sha', '')), 'actual PR source head SHA')
    # REST/event mergeability is computed asynchronously. This observation is
    # optional metadata, not proof of the Actions-selected merge commit.
    merge_observed = pr.get('merge_commit_sha')
    require(merge_observed is None or (type(merge_observed) is str and
            re.fullmatch('[0-9a-f]{40}', merge_observed)), 'optional observed PR merge SHA')
    return {'repository': REPOSITORY, 'run_id': env['GITHUB_RUN_ID'], 'run_attempt': 1,
            'job_key': JOB, 'executed_head_sha': pr['head']['sha'], 'github_sha': env['GITHUB_SHA'],
            'pull_request_merge_commit_sha_observed': merge_observed,
            'pull_request_head_sha': pr['head']['sha'], 'pull_request_number': number}


def source_binding(run, root=ROOT):
    """Compare each actual input byte sequence with this run's Git object."""
    root = Path(root)
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], check=True,
                                capture_output=True, timeout=5)
        require(len(result.stdout) <= 256 * 1024 and not result.stderr, 'bounded source Git object')
        return result.stdout
    require(git('rev-parse', 'HEAD').decode().strip() == run['executed_head_sha'], 'checkout equals actual PR head SHA')
    sources = {}
    for name in SOURCE_FILES:
        actual = identity(root / name)
        body = git('show', run['executed_head_sha'] + ':' + name)
        require(actual == {'bytes': len(body), 'sha256': digest(body)}, 'source input differs from executed Git object')
        sources[name] = actual
    binding = {'run': run, 'source_files': sources, 'image': IMAGE,
               'image_config': IMAGE_CONFIG, 'command': COMMAND,
               'staged_code': {GATE: sources[GATE]}, 'runtime_files': [], 'input_files': []}
    return {**binding, 'binding_sha256': digest(canonical(binding))}


@contextmanager
def phase_alarm(deadline):
    """An earlier phase alarm cannot reset the absolute outer host allowance."""
    previous = signal.getsignal(signal.SIGALRM)
    old_remaining, old_interval = signal.getitimer(signal.ITIMER_REAL)
    old_deadline = time.monotonic() + old_remaining if old_remaining else None
    selected = min(deadline, old_deadline) if old_deadline is not None else deadline
    def expired(signum, frame):
        # Re-arm the host timer before legacy pull_image's bounded finally runs.
        if old_deadline is not None and old_deadline > time.monotonic():
            signal.signal(signal.SIGALRM, previous)
            signal.setitimer(signal.ITIMER_REAL, old_deadline - time.monotonic())
        raise TimeoutError('fixed smoke phase elapsed')
    require(selected > time.monotonic(), 'smoke phase remaining allowance')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.000001, selected - time.monotonic()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if old_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, old_deadline - time.monotonic()), old_interval)


class SmokeCpuLedger(CpuLedger):
    """Reuse cumulative registration/sample/final-close, with a smaller threshold."""
    def check(self):
        exceeded = super().check()
        if sum(row['usage_usec'] for row in self.entries.values()) >= CPU_SECONDS * 1000000:
            self.failed = True
            exceeded.append('smoke_total')
        return exceeded

    def receipt(self):
        value = super().receipt()
        value.update(thresholds_seconds={'setup': CPU_SECONDS}, total_threshold_seconds=CPU_SECONDS)
        return value


def bounded_text(path):
    with Path(path).open() as source:
        text = source.read(8193)
    require(len(text) <= 8192, 'bounded kernel scope text')
    return text


def observed_kernel_scope(row):
    """Read real cgroup limits while MeteredDocker still holds the gate closed."""
    cgroup = row['cpu_path'].parent
    require(cgroup.stat().st_ino == row['cgroup_inode'], 'same live cgroup')
    expected = {'memory.max': str(12 * 1024 ** 3), 'memory.swap.max': '0',
                'cpu.max': '400000 100000', 'pids.max': '256'}
    actual = {name: ' '.join(bounded_text(cgroup / name).split()) for name in expected}
    require(actual == expected, 'observed smoke kernel limits')
    pids = bounded_text(cgroup / 'cgroup.procs').split()
    require(len(pids) == 1 and pids[0].isdigit() and int(pids[0]) > 0, 'one pre-release gate process')
    status = dict(line.split(':', 1) for line in bounded_text(Path('/proc') / pids[0] / 'status').splitlines() if ':' in line)
    require(status.get('NoNewPrivs', '').strip() == '1' and status.get('Seccomp', '').strip() == '2',
            'observed no-new-privileges and seccomp')
    require(all(re.fullmatch('0+', status.get(key, '').strip()) for key in
                ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')), 'observed zero capabilities')
    require(status.get('Uid', '').split() == [str(os.getuid())] * 4 and os.getuid() != 0,
            'observed non-root gate identity')
    return {'limits': actual, 'non_root': True, 'capabilities': 'none',
            'no_new_privileges': True, 'seccomp_filter': True, 'observed_before_gate_release': True}


class SmokeDocker(host.MeteredDocker):
    """Keep the production lifecycle; add this smoke's smaller gate and scope."""
    def __init__(self, root, deadline):
        super().__init__(root, deadline)
        self.ledger = SmokeCpuLedger()
        self.kernel_scope = {}

    def check(self):
        super().check()
        for name, row in self.names.items():
            if row['registered'] and name not in self.kernel_scope:
                self.kernel_scope[name] = observed_kernel_scope(row)


def stage_gate(root, binding):
    code = root / 'code'
    target = code / GATE
    target.parent.mkdir(parents=True)
    require(identity(ROOT / GATE) == binding['staged_code'][GATE], 'current gate matches source binding')
    shutil.copyfile(ROOT / GATE, target)
    require(identity(target) == binding['staged_code'][GATE], 'staged gate bytes')
    for name in ('runtime', 'input', 'output'):
        (root / name).mkdir(mode=0o700)
    return code


def smoke_arguments(original, docker, name, code):
    args = original.container_args(name, IMAGE, code, docker.root / 'runtime',
        docker.root / 'output', 'asr', 'qwen06', docker.root / 'input')
    index = args.index(IMAGE)
    args = args[:index + 1] + COMMAND
    return host.gated_arguments(args, name, docker, '/code/' + GATE)


def validate_created(original, record, args, docker, name, image_env):
    expected = {'/code': (docker.root / 'code', False), '/runtime': (docker.root / 'runtime', False),
                '/input': (docker.root / 'input', False), '/output': (docker.root / 'output', True),
                '/gate': (docker.names[name]['gate'], True)}
    mounts = record['Mounts']
    require(len(mounts) == len(expected), 'only five exact smoke mounts')
    for target, (source, writable) in expected.items():
        rows = [r for r in mounts if r.get('Destination') == target]
        require(len(rows) == 1 and rows[0].get('Type') == 'bind' and
                rows[0].get('Source') == str(source) and rows[0].get('RW') is writable,
                'actual exact smoke mount source and mode')
    original.validate_container_inspect({**record, 'Mounts': [m for m in mounts if m['Destination'] != '/gate']}, 'qwen06')
    config = record['Config']
    requested_env = [args[i + 1] for i, value in enumerate(args[:-1]) if value == '--env']
    effective_env = dict(item.split('=', 1) for item in image_env)
    effective_env.update(dict(item.split('=', 1) for item in requested_env))
    observed_env = config.get('Env') or []
    require(len(observed_env) == len(effective_env) and
            dict(item.split('=', 1) for item in observed_env) == effective_env, 'no extra inherited environment')
    require(config.get('Image') == IMAGE and record.get('Image') == IMAGE_CONFIG and
            config.get('Cmd') == args[args.index(IMAGE) + 1:] and config.get('Entrypoint') in (None, []) and
            config.get('User') == f'{os.getuid()}:{os.getgid()}', 'actual fixed image, static command and non-root identity')
    hc = record['HostConfig']
    require(hc.get('RestartPolicy', {}).get('Name') in ('', 'no') and
            not hc.get('Devices') and not hc.get('DeviceRequests') and not hc.get('Binds'),
            'no restart, devices, or hidden binds')
    tmpfs = dict(item.split(':', 1) for i, item in enumerate(args) if i and args[i - 1] == '--tmpfs')
    require(hc.get('Tmpfs') == tmpfs, 'exact bounded tmpfs')
    require(sorted(hc.get('Ulimits') or [], key=lambda r: r['Name']) == sorted([
        {'Name': 'core', 'Soft': 0, 'Hard': 0}, {'Name': 'nofile', 'Soft': 1024, 'Hard': 1024},
        {'Name': 'fsize', 'Soft': 8388608, 'Hard': 8388608}], key=lambda r: r['Name']), 'exact process resource limits')


def lifecycle(original, docker, code, run, image_env, *, result=None):
    result = {} if result is None else result
    result.update(status='started', primary_error=None, cleanup_error=None)
    name = 'kws-cleanup-smoke-' + run['run_id'] + '-1'
    args = smoke_arguments(original, docker, name, code)
    phase = 'create'
    started = time.monotonic()
    outer_deadline = docker.deadline
    docker.deadline = min(outer_deadline, started + LIFECYCLE_SECONDS)
    try:
        with phase_alarm(docker.deadline):
            docker(*args)
            phase = 'inspect'
            record = json.loads(docker('inspect', name))[0]
            validate_created(original, record, args, docker, name, image_env)
            result['actual_contract_verified_before_start'] = True
            phase = 'start'
            docker('start', name)
            phase = 'wait'
            while not docker.completed(name):
                time.sleep(.2)
            result['gate_result'] = docker.gate_result(name)
            require(time.monotonic() < docker.deadline, 'lifecycle wall ceiling')
    except BaseException as error:
        result['primary_error'] = {'phase': phase, **host.safe_error(error)}
    finally:
        result['lifecycle_seconds'] = time.monotonic() - started
        # Production cleanup has its own single 15-second window even if the
        # lifecycle/host allowance is gone. Its overrun remains a failed receipt.
        signal.setitimer(signal.ITIMER_REAL, 0)
        docker.deadline = outer_deadline
        cleanup_started = time.monotonic()
        try:
            docker.finish()
        except BaseException as error:
            result['cleanup_error'] = host.safe_error(error)
        finally:
            result['cleanup_wall_seconds'] = time.monotonic() - cleanup_started
        result['kernel_scope'] = docker.kernel_scope
        result['status'] = 'failed_no_retry' if result['primary_error'] or result['cleanup_error'] else 'complete'
        docker.deadline = outer_deadline
    return result


def acquire_image(original, lock, root, deadline):
    """Reuse exactly one historical pull, tightening inspect and outer phase time."""
    require(lock.get('image') == IMAGE and lock.get('platform') == 'linux/amd64' and
            lock.get('manifest', {}).get('config', {}).get('digest') == IMAGE_CONFIG, 'fixed reviewed official image')
    saved = original.docker
    observed = {}
    def bounded_docker(*args, timeout=5):
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'image inspect host allowance')
        raw = saved(*args, timeout=min(timeout, 5, remaining))
        if args[:2] == ('image', 'inspect'):
            records = json.loads(raw)
            require(len(records) == 1, 'one image inspect')
            observed['environment'] = records[0]['Config']['Env']
        return raw
    original.docker = bounded_docker
    try:
        with phase_alarm(min(deadline, time.monotonic() + IMAGE_SECONDS)):
            receipt = original.pull_image(lock, root / 'image-transfer.json')
    finally:
        original.docker = saved
    require(receipt['daemon_completion_verified'] is True and 'environment' in observed, 'observed image completion')
    return receipt, observed['environment']


def emit_receipt(receipt):
    raw = canonical(receipt)
    require(len(raw) + 1 <= RECEIPT_BYTES, 'smoke receipt at most 32 KiB')
    print(raw.decode(), flush=True)


def execute(env, event):
    run = admission(env, event)
    started = time.monotonic()
    deadline = started + HOST_SECONDS
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise TimeoutError('smoke host interrupted or deadline exhausted')
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, HOST_SECONDS)
    receipt = {'schema': 'qwen12-cleanup-lifecycle-smoke-v1', **run, 'status': 'failed_no_retry',
               'model_calls': 0, 'audio_files': 0, 'package_installs': 0, 'training_admitted': False,
               'limits': {'host_seconds': HOST_SECONDS, 'image_client_phase_seconds': IMAGE_SECONDS,
                          'image_failed_client_cleanup_seconds': 20, 'lifecycle_seconds': LIFECYCLE_SECONDS,
                          'cleanup_seconds': 15, 'docker_lifecycle_call_seconds': 5,
                          'sampled_container_cpu_seconds': CPU_SECONDS, 'receipt_bytes': RECEIPT_BYTES},
               'cpu_scope': 'controlled container only; excludes host, image pull, and Docker daemon',
               'storage': 'bounded_stdout_receipt_no_artifact_no_cache', 'primary_error': None}
    root = None
    docker = None
    try:
        receipt['inputs'] = binding = source_binding(run)
        release = read_json(HERE / 'execution-release.json')
        require(release.get('enabled') is False, 'native release must remain disabled')
        root = Path(env['RUNNER_TEMP']) / ('cleanup-lifecycle-smoke-' + run['run_id'] + '-1')
        root.mkdir(mode=0o700)  # Existing directory means a second execution; never reuse.
        code = stage_gate(root, binding)
        old = ROOT / 'research/source-screen32-v1'
        sys.path.insert(0, str(old))
        original = host.load('qwen12_cleanup_smoke_original_host', old / 'hosted_run.py')
        image, image_env = acquire_image(original, read_json(old / 'container-lock.json'), root, deadline)
        receipt['image_acquisition'] = image
        docker = SmokeDocker(root, deadline)
        receipt['lifecycle'] = {}
        lifecycle(original, docker, code, run, image_env, result=receipt['lifecycle'])
        require(receipt['lifecycle']['status'] == 'complete' and time.monotonic() < deadline, 'complete bounded lifecycle')
        require(all(not list((root / name).iterdir()) for name in ('runtime', 'input', 'output')), 'no runtime, audio or output bodies')
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt['primary_error'] = host.safe_error(error)
        if root is not None and (root / 'image-transfer.json').is_file() and 'image_acquisition' not in receipt:
            try:
                receipt['image_acquisition'] = read_json(root / 'image-transfer.json', RECEIPT_BYTES)
            except BaseException as read_error:
                receipt['image_receipt_error'] = host.safe_error(read_error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Retain failures independently: returning from lifecycle is not a
        # prerequisite for keeping final CPU, owned-container and gate evidence.
        if docker is not None:
            for key, value in (('cpu', lambda: docker.ledger.receipt()),
                               ('cleanup', lambda: read_json(root / 'cleanup-receipt.json', RECEIPT_BYTES))):
                try:
                    receipt[key] = value()
                except BaseException as error:
                    receipt[key + '_receipt_error'] = host.safe_error(error)
                    receipt['status'] = 'failed_no_retry'
            if 'cleanup' not in receipt:
                receipt['cleanup_in_memory'] = docker.cleanup
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        receipt['wall_seconds'] = time.monotonic() - started
        emit_receipt(receipt)
    return 0 if receipt['status'] == 'complete' else 1


def main():
    require(len(sys.argv) == 1, 'no caller-selected command or options')
    env = dict(os.environ)
    return execute(env, read_json(env['GITHUB_EVENT_PATH'], 1024 * 1024))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        emit_receipt({'schema': 'qwen12-cleanup-lifecycle-smoke-v1', 'status': 'failed_no_retry',
                      'error': host.safe_error(error)})
        raise SystemExit(2)
