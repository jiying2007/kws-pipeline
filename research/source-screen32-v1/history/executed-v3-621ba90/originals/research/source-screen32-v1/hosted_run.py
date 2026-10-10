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
IMAGE_TRANSFER_RESERVE = 128 * 1024 ** 2
BRANCH = 'research/qwen16-voicedesign-once-v3'
EXPERIMENT = 'qwen16-voicedesign-once-v3'


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


def verify_first_created_source(env, branch, number, freeze_sha, workflow_name, template_name):
    """Shared fixed first-push checks; callers choose only screen or probe identity."""
    expected = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + branch,
                'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': str(number)}
    require(all(env.get(key) == value for key, value in expected.items()), 'one-shot workflow identity')
    require(re.fullmatch(r'[0-9a-f]{40}', env.get('GITHUB_SHA', '')) and
            re.fullmatch(r'[1-9][0-9]{0,19}', env.get('GITHUB_RUN_ID', '')), 'immutable head/run identity')
    event = bounded_json(env['GITHUB_EVENT_PATH'])
    require(event.get('created') is True and event.get('deleted') is False and
            event.get('after') == env['GITHUB_SHA'] and
            event.get('repository', {}).get('private') is False and
            event.get('repository', {}).get('full_name') == expected['GITHUB_REPOSITORY'],
            'first public branch creation required')
    require(file_hash(HERE / 'execution-freeze.json') == freeze_sha, 'reviewed freeze identity')
    freeze = bounded_json(HERE / 'execution-freeze.json')
    require(freeze.get('schema') == 'screen32-execution-source-freeze-v1', 'source freeze schema')
    require(set(freeze) == {'schema', 'files', 'profiles'}, 'source freeze fields')
    for name, digest in freeze['files'].items():
        require(name.startswith('research/') and '..' not in Path(name).parts and
                re.fullmatch(r'[0-9a-f]{64}', digest), 'source path/hash')
        path = ROOT / name
        require(path.is_file() and not path.is_symlink() and file_hash(path) == digest, 'reviewed source drift')
    workflow = ROOT / '.github/workflows' / workflow_name
    require(workflow.is_file() and workflow.read_bytes() == (HERE / template_name).read_bytes(), 'activated workflow drift')
    return freeze


def verify_admission(env):
    release = bounded_json(HERE / 'execution-release.json')
    require(set(release) == {'schema', 'experiment', 'approved', 'source_freeze_sha256', 'successful_probe'} and
            release['schema'] == 'screen32-execution-release-v1' and release['experiment'] == EXPERIMENT and
            release['approved'] is True, 'exact reviewed source admission required')
    freeze = verify_first_created_source(env, BRANCH, 1, release['source_freeze_sha256'],
                                        'source-screen32-run-v3.yml', 'workflow.yml')
    verify_successful_probe(release['successful_probe'], freeze, release['source_freeze_sha256'])
    return freeze


def verify_probe_admission(env):
    release = bounded_json(HERE / 'probe-release.json')
    require(set(release) == {'schema', 'approved', 'attempt', 'source_freeze_sha256'} and
            release['schema'] == 'screen32-probe-release-v1' and release['approved'] is True and
            type(release['attempt']) is int and release['attempt'] in (1, 2), 'exact bounded probe admission required')
    attempt = release['attempt']
    freeze = verify_first_created_source(env, 'research/qwen16-container-probe-' + str(attempt), attempt,
              release['source_freeze_sha256'], 'source-screen32-probe.yml', 'probe-workflow.yml')
    return freeze, attempt


# Only fixed public phrases/options can leave Docker stderr. No raw text, paths,
# URLs, environment values or unknown tokens are included in public evidence.
DOCKER_ERROR_PHRASES = (
    'unknown flag', 'unknown shorthand flag', 'invalid argument', 'invalid option',
    'invalid ulimit', 'invalid ulimit type', 'invalid mount config', 'invalid mount path', 'invalid bind mount',
    'invalid mode', 'unknown log opt', 'invalid log opt', 'invalid tmpfs option', 'invalid container name', 'invalid reference format', 'invalid value',
    'unsupported', 'not supported', 'permission denied', 'operation not permitted',
    'read-only file system', 'no such file or directory', 'no such container',
    'no space left on device', 'cannot allocate memory', 'out of memory',
    'OCI runtime create failed', 'OCI runtime start failed', 'failed to create task',
    'failed to mount', 'error mounting', 'failed to set rlimit', 'failed to write',
    'unable to apply cgroup configuration', 'connection refused', 'cannot connect to the Docker daemon',
    'context deadline exceeded', 'conflict', 'is already in use', 'not found',
    'failed to initialize logging driver', 'compression cannot be enabled when max file count is 1',
)
DOCKER_ERROR_OPTIONS = ('--memory', '--memory-swap', '--cpus', '--pids-limit', '--cgroupns', '--ipc',
    '--read-only', '--cap-drop', '--security-opt', '--user', '--network', '--ulimit', '--log-driver',
    '--log-opt', '--tmpfs', '--mount', '--platform', '--pull', 'core', 'nofile', 'fsize', 'max-file', 'max-size', 'compress', 'uid', 'gid', 'mode', 'size', 'no-new-privileges')


def public_failure(error, operation):
    """A small positive-allowlisted summary; the original exception stays intact."""
    original_stderr = getattr(error, 'stderr', None)
    representation = 'captured_bytes' if isinstance(original_stderr, bytes) else 'utf8_reencoded_text' if isinstance(original_stderr, str) else 'not_provided'
    raw = original_stderr or b''
    if isinstance(raw, str): raw = raw.encode('utf-8', errors='replace')
    if not isinstance(raw, bytes): raw = b''
    sample = raw[:16384].decode('utf-8', errors='replace').lower()
    result = {'operation': operation, 'error_type': type(error).__name__,
              'returncode': getattr(error, 'returncode', None),
              'stderr': {'representation': representation, 'observed_bytes': len(raw), 'examined_prefix_bytes': min(len(raw), 16384),
                  'prefix_truncated': len(raw) > 16384,
                  'safe_fragments': [phrase for phrase in DOCKER_ERROR_PHRASES if phrase.lower() in sample],
                  'fixed_option_mentions': [option for option in DOCKER_ERROR_OPTIONS if re.search(r'(?<![A-Za-z0-9_-])' + re.escape(option.lower()) + r'(?![A-Za-z0-9_-])', sample)],
                  'fixed_argument_values': [value for value in (
                      'core=0', 'nofile=1024:1024', 'fsize=4294967296', 'fsize=8388608',
                      'max-file=1', 'max-size=1m', 'compress=false', 'size=268435456', 'size=134217728',
                      'uid=' + str(os.getuid()), 'gid=' + str(os.getgid()), 'mode=0700',
                      'noexec', 'nosuid', 'nodev')
                      if re.search(r'(?<![A-Za-z0-9_=:-])' + re.escape(value) + r'(?![A-Za-z0-9_:-])', sample)],
                  'sha256': hashlib.sha256(raw).hexdigest(),
                  'unclassified': not any(phrase.lower() in sample for phrase in DOCKER_ERROR_PHRASES),
                  'arbitrary_content_omitted': bool(raw),
                  'policy': 'positive_allowlist_no_raw_text'}}
    if type(result['returncode']) is not int: result['returncode'] = None
    if isinstance(error, subprocess.TimeoutExpired): result['timed_out'] = True
    return result


def docker(*args, timeout=30):
    # No command line, raw stdout or stderr is copied into the public receipt.
    operation = 'image_inspect' if args[:2] == ('image', 'inspect') else args[0]
    require(operation in ('info', 'image_inspect', 'create', 'inspect', 'start', 'rm'), 'fixed Docker operation')
    try:
        result = subprocess.run(['docker', *args], check=True, timeout=timeout, capture_output=True)
        require(len(result.stdout) <= 2 * 1024 ** 2, 'oversized Docker response')
        return result.stdout.decode('utf-8')
    except BaseException as error:
        error.screen32_diagnostic = public_failure(error, operation)
        raise


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
    """Observe one stable default-route interface, never sum stacked links.

    This is host-interface traffic, not per-process/daemon accounting. Reading
    routes sends no packets and changes no host settings. Ambiguous defaults fail.
    """
    defaults = set()
    for line in Path('/proc/net/route').read_text().splitlines()[1:]:
        fields = line.split()
        require(len(fields) == 11, 'malformed IPv4 route observation')
        if fields[1] == fields[7] == '00000000' and int(fields[3], 16) & 1 and not int(fields[3], 16) & 0x200:
            defaults.add(fields[0])
    for line in Path('/proc/net/ipv6_route').read_text().splitlines():
        fields = line.split()
        require(len(fields) == 10, 'malformed IPv6 route observation')
        if fields[0] == '0' * 32 and fields[1] == '00' and int(fields[8], 16) & 1 and not int(fields[8], 16) & 0x200:
            defaults.add(fields[9])
    require(len(defaults) == 1 and 'lo' not in defaults, 'single non-loopback default interface required')
    counters = {}
    for path in Path('/sys/class/net').glob('*/statistics/rx_bytes'):
        device = path.parent.parent
        if device.name == 'lo': continue
        require(re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', device.name), 'interface name boundary')
        counters[device.name] = {'rx_bytes': int(path.read_text()), 'ifindex': int((device / 'ifindex').read_text())}
    require(0 < len(counters) <= 64 and defaults <= counters.keys() and
            all(row['rx_bytes'] >= 0 and row['ifindex'] > 0 for row in counters.values()), 'bounded interface counters')
    return {'default_interface': next(iter(defaults)), 'interfaces': counters}


def record_network_delta(before, after, observation):
    observation['network_after'] = after
    device = before['default_interface']
    require(after['default_interface'] == device, 'default interface changed')
    first, last = before['interfaces'][device], after['interfaces'][device]
    require(first['ifindex'] == last['ifindex'] and last['rx_bytes'] >= first['rx_bytes'],
            'default interface identity/counter changed')
    delta = last['rx_bytes'] - first['rx_bytes']
    observation['default_interface_received_delta_bytes'] = delta
    # Retain the old aggregate only as diagnostic evidence, never an image-byte
    # claim or gate. Virtual devices may observe the same traffic again.
    comparable = before['interfaces'].keys() == after['interfaces'].keys() and all(
        row['ifindex'] == after['interfaces'][name]['ifindex'] and
        row['rx_bytes'] <= after['interfaces'][name]['rx_bytes'] for name, row in before['interfaces'].items())
    observation['all_interface_delta_diagnostic_only_bytes'] = sum(
        after['interfaces'][name]['rx_bytes'] - row['rx_bytes'] for name, row in before['interfaces'].items()) if comparable else None
    require(delta <= IMAGE_TRANSFER_RESERVE, 'observed default-interface reservation exceeded; no setup or generation')


def pull_image(lock, receipt_path):
    """Monitored pull reservation, NOT a hard Docker-daemon transfer cutoff.

    A client cancellation does not prove daemon downloads stopped. On any failure,
    no setup is started and the retained receipt leaves cumulative bytes unknown.
    The daemon and host security settings are never changed or shut down here.
    """
    observation = {'schema': 'screen32-image-transfer-observation-v2', 'status': 'started',
        'image': lock['image'], 'reservation_bytes': IMAGE_TRANSFER_RESERVE,
        'reservation_kind': 'monitored_default_interface_receive_observation_not_hard_transfer_limit',
        'declared_config_and_compressed_layer_bytes': lock['config_and_compressed_layer_bytes'],
        'declared_manifest_bytes': lock['manifest_bytes'],
        'default_interface_received_delta_bytes': None, 'daemon_completion_verified': False,
        'cumulative_image_transfer_bytes_verified': False, 'pull_client_attempts': 0,
        'byte_scope': 'selected_default_interface_includes_protocol_and_unrelated_host_traffic_not_exact_daemon_accounting'}
    atomic(receipt_path, observation)
    process = None
    try:
        before = network_received(); observation['network_before'] = before
        atomic(receipt_path, observation)
        started = time.monotonic()
        process = subprocess.Popen(['docker', 'pull', '--platform', 'linux/amd64', lock['image']],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        observation['pull_client_attempts'] = 1
        while process.poll() is None:
            record_network_delta(before, network_received(), observation)
            require(time.monotonic() - started <= 180, 'image pull client wall limit')
            time.sleep(0.1)
        observation['pull_client_exit_code'] = process.returncode
        record_network_delta(before, network_received(), observation)
        require(process.returncode == 0, 'single image pull failed; no retry')
        image = json.loads(docker('image', 'inspect', lock['image']))[0]
        require(image['Id'] == lock['manifest']['config']['digest'] and image['Architecture'] == 'amd64'
                and image['Os'] == 'linux', 'exact official image config/platform mismatch')
        observation.update(status='completed_observed_within_reservation', daemon_completion_verified=True,
                           image_config_digest=image['Id'])
        return dict(observation)
    except BaseException as error:
        observation.update(status='stopped_before_setup_daemon_completion_unverified',
                           error_type=type(error).__name__, daemon_fetch_may_continue=process is not None,
                           after_snapshot_available='network_after' in observation)
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
            '--log-opt', 'compress=false',
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



def setup_probe_args(name, image, code, runtime, output):
    """Replace only the exact setup worker tail; no caller-selected command."""
    args = container_args(name, image, code, runtime, output, 'tts', 'setup')
    tail = ['python', '-I', '-B', '/code/research/source-screen32-v1/setup_adapter.py',
            '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
    require(args[-len(tail):] == tail, 'exact setup command tail changed')
    prefix = args[:-len(tail)]
    return prefix + ['python', '-I', '-B', '/code/research/source-screen32-v1/probe_worker.py']


def setup_contract_sha256(image):
    """Fixed create prefix; only per-run name/paths and non-root UID/GID vary."""
    args = container_args('<container>', image, Path('/<code>'), Path('/<runtime>'), Path('/<output>'), 'tts', 'setup')
    tail = ['python', '-I', '-B', '/code/research/source-screen32-v1/setup_adapter.py',
            '--profile', 'tts', '--stage', 'setup', '--runtime', '/runtime']
    require(args[-len(tail):] == tail, 'setup contract tail changed')
    prefix = args[:-len(tail)]
    prefix[prefix.index('--user') + 1] = '<uid>:<gid>'
    for index, value in enumerate(prefix):
        if index and prefix[index - 1] == '--tmpfs':
            prefix[index] = value.replace('uid=' + str(os.getuid()) + ',', 'uid=<uid>,').replace('gid=' + str(os.getgid()) + ',', 'gid=<gid>,')
    return hashlib.sha256(canonical({'create_prefix': prefix, 'setup_tail': tail})).hexdigest()


def verify_probe_kernel(value, freeze):
    require(value.get('schema') == 'screen32-probe-kernel-v1' and value.get('status') == 'complete' and
            value.get('python_version') == '3.12.14' and value.get('model_calls') == 0 and
            value.get('runtime_installs') == 0 and value.get('human_gold') is False and
            value.get('training_admitted') is False, 'complete no-model kernel receipt required')
    require(value.get('worker_sha256') == freeze['files']['research/source-screen32-v1/probe_worker.py'] and
            value.get('runtime_scope_sha256') == freeze['files']['research/source-screen32-v1/runtime_scope.py'], 'probe worker source identity')
    kernel = value['kernel']
    require(kernel.get('schema') == 'screen32-kernel-scope-v1' and kernel.get('limits') == {
        'memory.max': str(MEMORY), 'memory.swap.max': '0', 'cpu.max': '400000 100000', 'pids.max': str(PIDS)} and
        type(kernel.get('uid')) is int and kernel['uid'] > 0 and kernel.get('private_cgroup_root') is True and
        kernel.get('read_only_root') is True and kernel.get('capabilities') == 'none' and
        kernel.get('no_new_privileges') is True and kernel.get('seccomp_filter') is True and
        kernel.get('network_disabled') is False, 'actual setup kernel scope proof required')
    interfaces = kernel.get('network_interfaces')
    require(type(interfaces) is list and 2 <= len(interfaces) <= 16 and interfaces == sorted(set(interfaces)) and
            'lo' in interfaces and all(re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', item) for item in interfaces), 'setup bridge interface proof')


def verify_successful_probe(reference, freeze, freeze_sha):
    """Verify retained raw evidence; a success boolean alone can never admit TTS."""
    require(type(reference) is dict and set(reference) == {'directory', 'artifact_freeze_sha256', 'source_head_sha'} and
            re.fullmatch(r'research/source-screen32-v1/evidence/probe-[12]-[1-9][0-9]{0,19}', reference['directory']) and
            re.fullmatch(r'[0-9a-f]{64}', reference['artifact_freeze_sha256']) and
            re.fullmatch(r'[0-9a-f]{40}', reference['source_head_sha']), 'reviewed successful probe reference required')
    directory = ROOT / reference['directory']
    require(directory.is_dir() and not directory.is_symlink(), 'probe evidence directory')
    manifest = bounded_json(directory / 'artifact-freeze.json')
    require(file_hash(directory / 'artifact-freeze.json') == reference['artifact_freeze_sha256'] and
            manifest.get('schema') == 'screen32-evidence-freeze-v1' and manifest.get('human_gold') is False and
            manifest.get('training_admitted') is False, 'probe artifact freeze identity')
    names = {'host-receipt.json', 'image-transfer.json', 'container-setup.json', 'setup/probe-kernel.json'}
    require(set(manifest['files']) == names and not manifest.get('incomplete_temporary_files_excluded'), 'complete fixed probe evidence membership')
    actual = set()
    for path in directory.rglob('*'):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), 'probe evidence file type')
        if path.is_file(): actual.add(str(path.relative_to(directory)))
    require(actual == names | {'artifact-freeze.json'}, 'no extra probe evidence files')
    values = {}
    for name in names:
        values[name] = bounded_json(directory / name, 32768)
        require(file_hash(directory / name) == manifest['files'][name], 'probe raw evidence hash')
    receipt = values['host-receipt.json']; attempt = receipt.get('probe_attempt')
    require(receipt.get('schema') == 'screen32-probe-host-v1' and receipt.get('status') == 'complete' and
            type(attempt) is int and attempt in (1, 2) and receipt.get('run_number') == attempt and
            receipt.get('run_attempt') == 1 and receipt.get('head_sha') == reference['source_head_sha'] and
            receipt.get('source_freeze_sha256') == freeze_sha and receipt.get('container_start_attempts') == 1 and
            receipt.get('model_calls') == 0 and receipt.get('runtime_installs') == 0 and
            receipt.get('stages', {}).get('setup', {}).get('status') == 'complete', 'successful exact-source probe required')
    require(reference['directory'].endswith('/probe-' + str(attempt) + '-' + str(receipt.get('run_id'))), 'probe run/path identity')
    lock = bounded_json(HERE / 'container-lock.json')
    require(receipt.get('container_contract_sha256') == setup_contract_sha256(lock['image']), 'same setup container contract required')
    image = values['image-transfer.json']
    require(image.get('status') == 'completed_observed_within_reservation' and image.get('daemon_completion_verified') is True and
            image.get('image') == lock['image'] and image.get('image_config_digest') == lock['manifest']['config']['digest'], 'exact image proof')
    diagnostic = values['container-setup.json']
    require(diagnostic.get('status') == 'complete' and diagnostic.get('created_confirmed') is True and
            diagnostic.get('primary_failure') is None and diagnostic.get('cleanup_failure') is None and
            diagnostic.get('container_state') == {'Running': False, 'ExitCode': 0, 'OOMKilled': False} and
            diagnostic.get('operations') == {name: {'status': 'complete'} for name in
                ('create', 'inspect_configuration', 'start', 'watch', 'cleanup')}, 'completed create/start/kernel/cleanup proof')
    verify_probe_kernel(values['setup/probe-kernel.json'], freeze)
    return receipt


def validate_container_inspect(record, stage):
    host = record['HostConfig']
    require(host.get('LogConfig') == {'Type': 'local', 'Config': {
        'max-size': '1m', 'max-file': '1', 'compress': 'false'}}, 'exact bounded local log configuration required')
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
    created = False; primary = None; primary_traceback = None; cleanup_error = None; result = None
    path = output.parent / ('container-' + stage + '.json')
    diagnostic = {'schema': 'screen32-container-diagnostic-v1', 'stage': stage, 'status': 'started',
        'created_confirmed': False, 'active_operation': None, 'primary_failure': None, 'cleanup_failure': None,
        'operations': {name: {'status': 'not_run'} for name in ('create', 'inspect_configuration', 'start', 'watch', 'cleanup')},
        'human_gold': False, 'training_admitted': False}

    def mark(operation, status):
        diagnostic['active_operation'] = operation
        diagnostic['operations'][operation]['status'] = status
        atomic(path, diagnostic)

    def save_without_masking():
        try: atomic(path, diagnostic)
        except BaseException as error:
            diagnostic['diagnostic_write_failure_type'] = type(error).__name__
            return error
        return None

    operation = 'create'
    try:
        mark(operation, 'started')
        docker(*args); created = True; diagnostic['created_confirmed'] = True
        mark(operation, 'complete')
        operation = 'inspect_configuration'; mark(operation, 'started')
        record = json.loads(docker('inspect', name))[0]
        validate_container_inspect(record, stage); mark(operation, 'complete')
        operation = 'start'; mark(operation, 'started')
        docker('start', name); mark(operation, 'complete')
        operation = 'watch'; mark(operation, 'started')
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
            require(type(state.get('Running')) is bool and type(state.get('ExitCode')) is int and
                    -255 <= state['ExitCode'] <= 255 and type(state.get('OOMKilled')) is bool, 'bounded container state')
            if not state['Running']:
                diagnostic['container_state'] = {key: state.get(key) for key in ('Running', 'ExitCode', 'OOMKilled')}
                require(state['ExitCode'] == 0 and not state.get('OOMKilled'), 'container failed; no restart')
                mark(operation, 'complete')
                result = {'status': 'complete', 'wall_seconds': now - started, 'exit_code': state['ExitCode']}
                break
            time.sleep(0.5)
    except BaseException as error:
        primary, primary_traceback = error, error.__traceback__
        diagnostic['status'] = 'failed_no_retry'
        diagnostic['operations'][operation]['status'] = 'failed_no_retry'
        diagnostic['failure_operation'] = operation
        diagnostic['primary_failure'] = getattr(error, 'screen32_diagnostic', public_failure(error, operation))
        save_without_masking()  # Preserve the pre-cleanup failure even if removal also fails.
    finally:
        if created:
            diagnostic['active_operation'] = 'cleanup'
            diagnostic['operations']['cleanup']['status'] = 'started'
            save_without_masking()
            try:
                # Only the one confirmed-created dedicated container is removed.
                docker('rm', '--force', name)
                diagnostic['operations']['cleanup']['status'] = 'complete'
            except BaseException as error:
                cleanup_error = error
                diagnostic['operations']['cleanup']['status'] = 'failed_no_retry'
                diagnostic['cleanup_failure'] = getattr(error, 'screen32_diagnostic', public_failure(error, 'cleanup'))
        diagnostic['status'] = 'failed_no_retry' if primary is not None or cleanup_error is not None else 'complete'
        diagnostic['wall_seconds'] = time.monotonic() - started
        write_error = save_without_masking()
    if primary is not None: raise primary.with_traceback(primary_traceback)
    if cleanup_error is not None: raise cleanup_error
    if write_error is not None: raise write_error
    return result


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
    for stage in ('setup', 'tts', 'qwen06', 'sensevoice'):
        path = root / ('container-' + stage + '.json')
        if path.exists():
            bounded_json(path, 32768)
            sources.append(path)
    if phase == 'probe' and (root / 'setup/probe-kernel.json').exists():
        bounded_json(root / 'setup/probe-kernel.json', 32768)
        sources.append(root / 'setup/probe-kernel.json')
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
    names = () if phase == 'probe' else ('tts',) if phase == 'tts' else ('qwen06', 'sensevoice')
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



def run_probe(env):
    freeze, attempt = verify_probe_admission(env)  # Before capacity checks or image acquisition.
    root = Path(env['RUNNER_TEMP']) / ('qwen16-container-probe-' + str(attempt) + '-' + env['GITHUB_RUN_ID'])
    root.mkdir(exist_ok=False)
    require(shutil.disk_usage(root).free >= 16 * GIB, 'current 16GiB disk headroom required')
    started = time.monotonic()
    code, runtime, output = root / 'code', root / 'runtime', root / 'setup'
    for path in (code, runtime, output): path.mkdir()
    stage_code(freeze, 'probe', code)
    receipt = {'schema': 'screen32-probe-host-v1', 'status': 'started', 'probe_attempt': attempt,
        'head_sha': env['GITHUB_SHA'], 'run_id': env['GITHUB_RUN_ID'], 'run_number': int(env['GITHUB_RUN_NUMBER']),
        'run_attempt': 1, 'source_freeze_sha256': file_hash(HERE / 'execution-freeze.json'),
        'stages': {'setup': {'status': 'not_run'}}, 'container_start_attempts': 0,
        'model_calls': 0, 'runtime_installs': 0, 'human_gold': False, 'training_admitted': False}
    target = root / 'host-receipt.json'; atomic(target, receipt)
    try:
        receipt['capacity'] = host_capacity()
        lock = bounded_json(HERE / 'container-lock.json')
        receipt['container_contract_sha256'] = setup_contract_sha256(lock['image'])
        receipt['image'] = pull_image(lock, root / 'image-transfer.json')
        host_capacity()
        name = 'qwen16-container-probe-' + str(attempt) + '-' + env['GITHUB_RUN_ID']
        args = setup_probe_args(name, lock['image'], code, runtime, output)
        receipt['actual_create_argv_sha256'] = hashlib.sha256(canonical(args)).hexdigest()
        receipt['stages']['setup'] = {'status': 'started'}; atomic(target, receipt)
        receipt['stages']['setup'] = supervise_container(args, name, output, 'setup', started + 300)
        require(not list(runtime.iterdir()), 'probe runtime must remain empty')
        verify_probe_kernel(bounded_json(output / 'probe-kernel.json', 32768), freeze)
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='stopped_no_retry', error_type=type(error).__name__)
        receipt['failure'] = getattr(error, 'screen32_diagnostic', public_failure(error, 'probe_orchestration'))
        if receipt['stages']['setup']['status'] == 'started': receipt['stages']['setup']['status'] = 'failed_no_retry'
        raise
    finally:
        diagnostic = root / 'container-setup.json'
        if diagnostic.exists():
            try:
                status = bounded_json(diagnostic, 32768)['operations']['start']['status']
                require(status in ('not_run', 'started', 'complete', 'failed_no_retry'), 'probe start state')
                receipt['container_start_attempts'] = int(status != 'not_run')
            except (ValueError, KeyError, TypeError, OSError): receipt['container_start_attempts'] = None
        receipt['wall_seconds'] = time.monotonic() - started
        atomic(target, receipt); public_evidence(root, 'probe')
        if env.get('GITHUB_OUTPUT'):
            with open(env['GITHUB_OUTPUT'], 'a') as handle: handle.write('evidence_root=' + str(root) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tts', 'asr', 'probe'), required=True)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.phase == 'probe':
        require(args.input is None, 'probe accepts no input')
        return run_probe(os.environ)
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
        receipt['failure'] = getattr(error, 'screen32_diagnostic', public_failure(error, 'host_orchestration'))
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
