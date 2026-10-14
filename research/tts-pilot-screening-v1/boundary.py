"""Retained pure Docker boundary helpers, copied without old campaign admission."""
from pathlib import Path
import hashlib,json,os,re,subprocess,time
from screen import require, canonical
MEMORY=12*1024**3
PIDS=256
GIB=1024**3
IMAGE_TRANSFER_RESERVE=128*1024**2
def atomic(path,value):
    path=Path(path); tmp=path.with_suffix('.tmp'); tmp.write_bytes(canonical(value)); os.replace(tmp,path)
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
