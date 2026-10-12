"""Fixed ASR18 runtime boundary; imported only after actual host admission.

No historical host module is imported. RETAINED_NODE_SHA256 measures exact
AST source segments statically copied from source-screen32-v1/hosted_run.py.
Those pure kernel/image helpers keep their original public format/provenance.
No model, container or network operation happens at import time.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time

from campaign_contract import (EXPERIMENT, FAMILIES, LIMITS, MIB, GIB, require,
    canonical, digest, identity, read_json, read_bytes, write_once, durable_mkdir)
from resource_host import atomic, safe_error, gated_arguments
from runtime_scope import MEMORY, PIDS
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMAGE_TRANSFER_RESERVE = 128 * MIB
RETAINED_NODE_SHA256 = {'DOCKER_ERROR_PHRASES': '1ae05f0fb3387531d34974ddd2dfb7014eba77e6244c7b870230019e4016ffbd', 'DOCKER_ERROR_OPTIONS': '17d7fe5e5b2f6511ff2588ad00128a463811fadead073ffba5f7149c1dff2530', 'public_failure': '8456f5ef0c229f65b978dfe6a319db32b62b9e4180a57c79946d10bed347ded2', 'docker': '6052e8bd5feeac32481a43be46548fded9fe1306143d4dd81a0d0cb01fca727a', 'host_capacity': 'd1a9fec5d656302b6a05a453d0a92819a9de889169cd3bb1b97d450cdc4322cb', 'network_received': '38272469729e24f116cd25009ba038bc8c4adf8d589450fb04189341272c10e9', 'record_network_delta': 'd175320b37097864dabe839ce48bd82da4276960cde9b94350299913eb7231e5', 'pull_image': 'ef9f4def02a91054367fb5d300c81b89fd6e56ddd270551211e48f01ad4b5c08', 'container_args': '3eb05b58fe62db3f454b4c23bedd099c764ac06d964f98b1663449f09c040e57', 'validate_container_inspect': '717dbeb84509dd646db3d0f9230d6d5d087ffffbb2e36ab6dcf9e3badb2ab126'}

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


def stage_source(files, destination):
    durable_mkdir(destination)
    for name, expected in files.items():
        require(type(name) is str and not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')), 'closed source path')
        body = read_bytes(ROOT / name, cap=8 * MIB)
        require(expected == {'bytes': len(body), 'sha256': digest(body)}, 'closed source identity')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            require(stream.write(body) == len(body), 'complete source copy')
        target.chmod(0o444)


def worker_arguments(*, name, image, code, runtime, output, inputs, family, mode, order, authority,
                     run_identity, freeze_sha, claim_sha, frozen=None, policy=None):
    require(family in FAMILIES and mode in ('prepare', 'decode'), 'fixed native family/mode')
    stage = 'sensevoice' if family == 'whisper' else family
    args = container_args(name, image, code, runtime, output, 'asr', stage, inputs)
    index = args.index(image)
    extras = ['--mount', f'type=bind,src={order},dst=/order,readonly',
        '--mount', f'type=bind,src={authority},dst=/authority,readonly']
    env = {'KWS_EXECUTED_HEAD_SHA': run_identity['executed_head_sha'],
           'KWS_RUN_ID': str(run_identity['run_id']), 'KWS_SOURCE_FREEZE_SHA256': freeze_sha,
           'KWS_SOURCE_CLAIM_SHA256': claim_sha}
    if mode == 'decode':
        require(frozen is not None and policy is not None, 'all-three policy/own frozen config')
        extras += ['--mount', f'type=bind,src={frozen},dst=/frozen,readonly',
            '--mount', f'type=bind,src={policy},dst=/policy,readonly']
        env['KWS_POLICY_SHA256'] = identity(policy / 'policy.json')['sha256']
    for key, value in env.items():
        require(type(value) is str and re.fullmatch('[a-zA-Z0-9_-]{1,128}', value), 'fixed scalar worker environment')
        extras += ['--env', key + '=' + value]
    command = ['/runtime/venv/bin/python', '-I', '-B',
        '/code/research/k1-punctuation6-asr18-v1/native_worker.py', family, mode]
    return args[:index] + extras + [image] + command


def validate_launch(record, args, stage):
    """Exact allowlisted environment and all mount sources, including added gates."""
    require(record['Config'].get('Entrypoint') in (None, []), 'no inherited entrypoint')
    image = record['Config']['Image']; index = args.index(image)
    require(record['Config']['Cmd'] == args[index + 1:], 'fixed isolated native command')
    expected = {}
    for i, arg in enumerate(args[:index]):
        if arg == '--mount':
            fields = args[i + 1].split(',')
            spec = dict(part.split('=', 1) for part in fields if '=' in part)
            expected[spec['dst']] = (spec['src'], 'readonly' not in fields)
    binds = [row for row in record['Mounts'] if row['Type'] == 'bind']
    require(len(binds) == len(expected) and len({m['Destination'] for m in binds}) == len(binds), 'exact unique worker mounts')
    for row in binds:
        require(expected.get(row['Destination']) == (row['Source'], row['RW']) and
                row.get('Propagation') == 'rprivate', 'no substituted worker mount')
    standard = {'/code', '/runtime', '/output'} | ({'/input'} if stage != 'setup' else set())
    validate_container_inspect({**record, 'Mounts': [m for m in record['Mounts'] if m['Destination'] in standard]}, stage)
    supplied = {args[i + 1].split('=', 1)[0]: args[i + 1].split('=', 1)[1]
                for i, arg in enumerate(args[:index]) if arg == '--env'}
    observed = {}
    for entry in record['Config']['Env']:
        key, value = entry.split('=', 1)
        require(key not in observed, 'unique container environment')
        observed[key] = value
    # Official locked Python image defaults only; never host environment copy.
    image_defaults = {'PATH', 'LANG', 'GPG_KEY', 'PYTHON_VERSION', 'PYTHON_SHA256'}
    require(set(observed) <= set(supplied) | image_defaults and all(observed.get(k) == v for k, v in supplied.items()),
            'no credential/history/label environment')
    host = record['HostConfig']
    require(not any(host.get(key) for key in ('Binds', 'VolumesFrom', 'Devices', 'DeviceRequests', 'ExtraHosts', 'Links', 'GroupAdd', 'PortBindings', 'PublishAllPorts')),
            'no additional inherited host resource')
    require(all(m['Type'] in ('bind', 'tmpfs') for m in record['Mounts']), 'no additional volume')


def supervise(docker, args, output, *, family, mode, deadline):
    name = args[args.index('--name') + 1]
    args = gated_arguments(args, name, docker, '/code/research/k1-punctuation6-asr18-v1/cpu_gate.py')
    began = time.monotonic()
    row = {'name': name, 'family': family, 'mode': mode, 'status': 'started',
           'worker_exit_success': False, 'primary_error': None, 'cleanup_error': None,
           'wall_seconds': 0.0, 'gate_result': None}
    docker.supervisions.append(row); docker.supervision_failure = None
    primary = cleanup = retention = None
    try:
        docker(*args)
        validate_launch(json.loads(docker('inspect', name))[0], args,
                        'sensevoice' if family == 'whisper' else family)
        docker('start', name)
        first_seen = {}
        while not docker.completed(name):
            now = time.monotonic()
            require(now < deadline, 'fixed phase wall ceiling')
            docker.sample_bytes(output, 'phase_output', LIMITS['output_bytes'])
            if mode == 'decode':
                for path in output.glob('clip-*'):
                    if path.is_dir() and not (path / 'completed.json').exists():
                        first_seen.setdefault(path.name, now)
                        require(now - first_seen[path.name] < 120, 'host observed per-clip wall ceiling')
            time.sleep(.2)
        row['worker_exit_success'] = True
    except BaseException as error:
        primary = error; row['primary_error'] = safe_error(error)
    finally:
        try:
            if name in docker.names and docker.names[name]['create_attempted']:
                docker.remove(name)
        except BaseException as error:
            cleanup = error; row['cleanup_error'] = safe_error(error)
        row.update(wall_seconds=time.monotonic() - began, gate_result=docker.gate_result(name),
                   status='complete' if primary is None and cleanup is None else 'failed')
        try:
            atomic(docker.root / 'supervision-receipt.json', {'schema': 'k1-asr18-supervision-v1', 'stages': docker.supervisions})
        except BaseException as error:
            retention = error; row['status'] = 'failed'
    error = primary or cleanup or retention
    if error is not None:
        docker.supervision_failure = (error, row)
        raise error
    return row['wall_seconds']


def whisper_setup(docker, setup_code, runtime, output, root, run_id, deadline):
    """Retained exact bounded setup, with only its fixed supervisor gate adapter."""
    path = ROOT / 'research/source-screen32-whisper-v1/attempt-3/bounded_host.py'
    spec = importlib.util.spec_from_file_location('k1_asr18_bounded_setup', path)
    bounded = importlib.util.module_from_spec(spec); spec.loader.exec_module(bounded)
    original = bounded.Supervisor
    class Gated(original):
        def create(self, role, stage, mounts, command, file_limit):
            name = f'kws-bound-{run_id}-{stage}-{role}'
            gate = docker.gate(name)
            return super().create(role, stage, [*mounts, bounded.mount(gate, '/gate', True)],
                ['python', '-I', '-B', '/opt/cpu_gate.py', *command], file_limit)
        def state(self, name):
            return not docker.completed(name)
    bounded.Supervisor = Gated
    return bounded.acquire_runtime(docker=docker, setup_code=setup_code, runtime=runtime,
        output=output, root=root, run_id=str(run_id), job_deadline=deadline)


def setup_runtimes(root, freeze, docker, inputs, run_identity):
    """One setup path; never retries, probes audio or loads recognition models."""
    require(shutil.disk_usage(root).free >= LIMITS['initial_free_bytes'], '12GiB initial free reserve')
    host_capacity()
    image = read_json(ROOT / 'research/source-screen32-v1/container-lock.json')
    pull_image(image, root / 'image-transfer.json'); docker.check()
    code, setup_code = root / 'code', root / 'setup-code'
    stage_source(freeze['inference_code'], code); stage_source(freeze['setup_code'], setup_code)
    # Required only by the retained gated Whisper broker entrypoint. This
    # additional alias is exact new cpu_gate bytes, never old worker code.
    gate = setup_code / 'research/source-screen32-whisper-v1/attempt-3/cpu_gate.py'
    with gate.open('xb') as stream: stream.write(read_bytes(HERE / 'cpu_gate.py'))
    gate.chmod(0o444)
    runtime = durable_mkdir(root / 'asr-runtime'); out = durable_mkdir(root / 'asr-setup')
    name = EXPERIMENT + '-' + str(run_identity['run_id']) + '-setup'
    args = container_args(name, image['image'], setup_code, runtime, out, 'asr', 'setup', inputs)
    index = args.index(image['image'])
    args = args[:index + 1] + ['python', '-I', '-B', '/code/research/k1-punctuation6-asr18-v1/setup_adapter.py',
        '--profile', 'asr', '--stage', 'setup', '--runtime', '/runtime']
    began = time.monotonic(); deadline = min(docker.deadline, began + LIMITS['setup_seconds'])
    supervise(docker, args, out, family='setup', mode=None, deadline=deadline)
    setup = read_json(runtime / 'setup-receipt.json', 32 * MIB)
    require(type(setup['actual_download_bytes']) is int and 0 <= setup['actual_download_bytes'] <= 4 * GIB,
            'ASR complete response body cap')
    # No command strings or exception text are projected into public evidence.
    write_once(root / 'asr-setup-public.json', {'schema': 'k1-asr18-setup-summary-v1',
        'status': 'complete', 'model_calls': 0, 'actual_download_bytes': setup['actual_download_bytes'],
        'setup_receipt_sha256': identity(runtime / 'setup-receipt.json')['sha256']})
    whisper = durable_mkdir(root / 'whisper-runtime'); whisper_out = durable_mkdir(root / 'whisper-setup')
    require(time.monotonic() < deadline, 'combined setup deadline')
    whisper_receipt = whisper_setup(docker, setup_code, whisper, whisper_out, root, run_identity['run_id'], deadline)
    elapsed = time.monotonic() - began
    require(elapsed <= LIMITS['setup_seconds'], 'combined setup ceiling')
    return {'runtimes': {'qwen06': runtime, 'sensevoice': runtime, 'whisper': whisper},
            'code': code, 'image': image['image'], 'seconds': elapsed,
            'asr_body_bytes': setup['actual_download_bytes'], 'whisper': whisper_receipt}


def whisper_public_names():
    """Finite retained setup evidence roster, never arbitrary .json matches."""
    names={'runtime-lock.json','source-lock.json','asset-lock.json','installed-inventory.json',
        'download-receipt.json','setup-receipt.json','phased-setup-failure.json'}
    names|={'phase-'+phase+'-'+suffix+'.json' for phase in ('install','source','model','finalize') for suffix in ('progress','complete')}
    bounded={'budget-initial.json','receipt.json','install-offline-scope.json','finalize-offline-scope.json'}
    bounded|={f'budget-{i:03d}-{suffix}.json' for i in range(24) for suffix in ('started','final')}
    bounded|={asset+suffix for asset in [f'wheel-{i:02d}' for i in range(1,23)]+['source','model'] for suffix in
        ('-broker.json','-broker-scope.json','-worker-scope.json','-broker-kernel.json','-worker-result.json','-worker-kernel.json','-failure.json')}
    bounded|={f'bounded-host-cleanup-{i:03d}.json' for i in range(26)}
    return names|{'bounded-acquisition/'+name for name in bounded}
