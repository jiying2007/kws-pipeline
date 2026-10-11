"""Dormant, source-only genuine pip worker. No acquisition on import.

The supervisor supplies a fresh <=512 MiB tmpfs and a fixed Unix CONNECT
broker inside a network-none container. Filesystem limits are NOT network byte
limits. Broker policy and activation authorization are independent prerequisites.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import selectors
import signal
import socket
import stat
import subprocess
import sys
import time

LOOPBACK = ('127.0.0.1', 18080)
PROXY = 'http://127.0.0.1:18080'
QUOTA = 512 * 1024 ** 2
CAPTURE_LIMIT = 128 * 1024
REQUIRED_OPTIONS = ('--resume-retries', '--retries', '--require-hashes',
                    '--no-deps', '--no-index', '--only-binary', '--keyring-provider')
LOCK_SHA256 = '881443c6e66acb99223137bd8342788468f814d0a1c3e73ba33a27da3ad1fe6d'


class WorkerFailure(Exception):
    """Only fixed, non-sensitive reason codes cross the supervisor boundary."""


def require(value, code):
    if not value:
        raise WorkerFailure(code)


RESULT_SCHEMA = 'bounded-pip-worker-result-v2'
SCOPE_SCHEMA = 'bounded-kernel-scope-v1'
STAGES = frozenset(('scope', 'deadline', 'manifest', 'storage', 'config',
                    'pip_version_probe', 'pip_help_probe', 'pip_config_probe',
                    'acquisition', 'verification', 'export', 'complete'))
REASONS = frozenset(('worker_internal_failure', 'worker_scope', 'deadline_bound',
    'deadline_record', 'host_deadline', 'deadline', 'invalid_manifest',
    'lock_identity', 'asset_index', 'filename', 'size', 'tmpfs_path',
    'tmpfs_not_fresh', 'tmpfs_required', 'tmpfs_quota', 'pip_config_present',
    'pip_config_guard', 'pip_version_probe', 'pip_capability_probe',
    'pip_effective_config', 'pip_output_limit', 'broker_socket', 'shim_failed',
    'shim_broker', 'shim_connections', 'pip_acquisition_failed', 'output_filename',
    'output_size', 'output_hash', 'candidate_path', 'candidate_size',
    'evidence_path', 'evidence_write_failed'))
RESULT_FIELDS = frozenset(('schema', 'status', 'stage', 'reason', 'asset_index',
                           'pip_version', 'filename', 'size_bytes', 'sha256',
                           'help_exit_code', 'missing_options', 'resume_policy',
                           'reviewed_source_matches'))


def validate_result(value):
    require(type(value) is dict and set(value) == RESULT_FIELDS and
            value['schema'] == RESULT_SCHEMA, 'evidence_write_failed')
    require(value['status'] in ('ok', 'failed') and value['stage'] in STAGES,
            'evidence_write_failed')
    require(value['asset_index'] is None or (type(value['asset_index']) is int and
            0 <= value['asset_index'] < 22), 'evidence_write_failed')
    require(value['pip_version'] is None or (type(value['pip_version']) is str and
            re.fullmatch(r'[0-9]{1,6}\.[0-9]{1,6}(?:\.[0-9]{1,6})?', value['pip_version'])),
            'evidence_write_failed')
    require(value['help_exit_code'] is None or (type(value['help_exit_code']) is int and
            -255 <= value['help_exit_code'] <= 255), 'evidence_write_failed')
    require(type(value['missing_options']) is list and
            all(type(flag) is str and flag in REQUIRED_OPTIONS for flag in value['missing_options']) and
            len(set(value['missing_options'])) == len(value['missing_options']), 'evidence_write_failed')
    require(value['resume_policy'] in (None, 'explicit-zero-resume', 'audited-25.0.1-no-resume') and
            (value['reviewed_source_matches'] is None or type(value['reviewed_source_matches']) is bool),
            'evidence_write_failed')
    if value['resume_policy'] == 'audited-25.0.1-no-resume':
        require(value['pip_version'] == '25.0.1' and type(value['reviewed_source_matches']) is bool,
                'evidence_write_failed')
    else:
        require(value['reviewed_source_matches'] is None and
                (value['resume_policy'] is None or value['pip_version'] != '25.0.1'),
                'evidence_write_failed')
    if value['status'] == 'failed':
        require(value['reason'] in REASONS and all(value[x] is None for x in
                ('filename', 'size_bytes', 'sha256')), 'evidence_write_failed')
    else:
        require(value['reason'] is None and value['stage'] == 'complete' and
                value['asset_index'] is not None and value['pip_version'] is not None and
                value['help_exit_code'] == 0 and
                (not value['missing_options'] or (value['resume_policy'] == 'audited-25.0.1-no-resume' and
                 value['missing_options'] == ['--resume-retries'])) and
                value['resume_policy'] is not None and
                (value['resume_policy'] != 'audited-25.0.1-no-resume' or value['reviewed_source_matches'] is True) and
                type(value['filename']) is str and len(value['filename']) <= 255 and
                re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', value['filename']) and
                type(value['size_bytes']) is int and 0 < value['size_bytes'] < QUOTA and
                type(value['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', value['sha256']),
                'evidence_write_failed')
    return value


def safe_result(index, stage, version, outcome=None, reason=None, capability=None):
    value = {'schema': RESULT_SCHEMA, 'status': 'ok' if outcome is not None else 'failed',
             'stage': stage if stage in STAGES else 'scope',
             'reason': None if outcome is not None else
                 (reason if reason in REASONS else 'worker_internal_failure'),
             'asset_index': index if type(index) is int and 0 <= index < 22 else None,
             'pip_version': version if type(version) is str and
                 re.fullmatch(r'[0-9]{1,6}\.[0-9]{1,6}(?:\.[0-9]{1,6})?', version) else None,
             'filename': None, 'size_bytes': None, 'sha256': None,
             'help_exit_code': None, 'missing_options': [], 'resume_policy': None,
             'reviewed_source_matches': None}
    if capability is not None:
        require(set(capability) == {'schema', 'pip_version', 'help_exit_code', 'missing_options',
                                    'policy', 'reviewed_source_matches'} and
                capability['schema'] == 'bounded-pip-capability-v1' and
                capability['pip_version'] == value['pip_version'], 'evidence_write_failed')
        value.update(help_exit_code=capability['help_exit_code'],
                     missing_options=capability['missing_options'],
                     resume_policy=capability['policy'],
                     reviewed_source_matches=capability['reviewed_source_matches'])
    if outcome is not None:
        require(set(outcome) == {'status', 'asset_index', 'pip_version', 'filename',
                                'size_bytes', 'sha256'}, 'evidence_write_failed')
        require(outcome['status'] == 'ok' and outcome['asset_index'] == value['asset_index'] and
                outcome['pip_version'] == value['pip_version'], 'evidence_write_failed')
        value.update({k: outcome[k] for k in ('filename', 'size_bytes', 'sha256')})
        value['stage'] = 'complete'
    return validate_result(value)


def sanitized_scope(snapshot):
    # Only bounded typed kernel facts, never raw /proc status or exception text.
    limits = {'memory.max': '12616466432', 'memory.swap.max': '0',
              'cpu.max': '300000 100000', 'pids.max': '240'}
    require(snapshot['role'] == 'worker' and type(snapshot['uid']) is int and
            0 < snapshot['uid'] < 2**32 and snapshot['limits'] == limits and
            snapshot['interfaces'] == ['lo'] and snapshot['readonly_root'] is True and
            snapshot['private_readonly_cgroup'] is True, 'worker_scope')
    require(snapshot['cgroup'] == '0::/', 'worker_scope')
    keys = ('NoNewPrivs', 'Seccomp', 'NSpid', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')
    status = {key: snapshot['status'][key] for key in keys}
    require(status['NoNewPrivs'] == '1' and status['Seccomp'] == '2' and
            type(status['NSpid']) is str and re.fullmatch(r'[0-9]{1,10}', status['NSpid']) and
            all(type(status[k]) is str and re.fullmatch(r'0{1,16}', status[k])
                for k in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')), 'worker_scope')
    return {'schema': SCOPE_SCHEMA, 'role': 'worker', 'uid': snapshot['uid'],
            'cgroup': '0::/', 'limits': limits, 'interfaces': ['lo'],
            'readonly_root': True, 'private_readonly_cgroup': True, 'status': status}



class Evidence:
    def __init__(self, root='/worker-evidence'):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve() == self.root and
                self.root.is_dir() and not any(self.root.iterdir()), 'evidence_path')
        self.stage = 'scope'
        self.pip_version = None
        self.capability = None

    def mark(self, stage):
        require(stage in STAGES, 'evidence_write_failed')
        self.stage = stage

    def write(self, name, value):
        require(name in ('worker-result.json', 'worker-scope.json'), 'evidence_write_failed')
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode('ascii')
        require(len(raw) <= 4096, 'evidence_write_failed')
        with (self.root / name).open('xb') as target:
            target.write(raw); target.flush(); os.fsync(target.fileno())


def verify_worker_scope():
    # -I deliberately removes the script directory from sys.path. Load only the
    # sibling trusted, read-only supervisor source mounted by the fixed launcher.
    path = Path(__file__).resolve().with_name('bounded_host.py')
    spec = importlib.util.spec_from_file_location('_trusted_bounded_host', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.verify_effective_scope('worker')
    except (ValueError, OSError, KeyError):
        raise WorkerFailure('worker_scope') from None


def worker_deadline(seconds, deadline_path='/run/kws/deadline.json'):
    require(type(seconds) is int and 0 < seconds <= 300, 'deadline_bound')
    path = Path(deadline_path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            0 < info.st_size <= 256, 'deadline_record')
    record = json.loads(path.read_bytes())
    require(type(record) is dict and set(record) == {'deadline_monotonic'},
            'deadline_record')
    limit = record['deadline_monotonic']
    now = time.monotonic()
    require(type(limit) in (int, float) and math.isfinite(limit) and
            now < limit <= now + 1200, 'host_deadline')
    return min(now + seconds, limit)


def read_asset(manifest, index):
    path = Path(manifest)
    require(not path.is_symlink() and path.is_file(), 'invalid_manifest')
    require(path.stat().st_size <= 1024 * 1024, 'invalid_manifest')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == LOCK_SHA256, 'lock_identity')
    rows = json.loads(raw)['files']
    require(len(rows) == 22 and type(index) is int and 0 <= index < 22,
            'asset_index')
    row = rows[index]
    require(re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', row['filename']), 'filename')
    require(type(row['size_bytes']) is int and 0 < row['size_bytes'] < QUOTA,
            'size')
    return row


def clean_env(root):
    root = str(Path(root))
    return {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': root + '/home',
            'TMPDIR': root + '/tmp', 'XDG_CONFIG_HOME': root + '/home',
            'XDG_CONFIG_DIRS': root + '/home', 'LANG': 'C.UTF-8',
            'LC_ALL': 'C.UTF-8', 'PIP_CONFIG_FILE': '/dev/null',
            'PYTHONNOUSERSITE': '1'}


def ensure_no_configs(env, prefix=None):
    # --isolated alone does not suppress global or site pip configuration.
    # PIP_CONFIG_FILE=/dev/null suppresses file loading in supported pip. Also
    # fail closed if any expected global/site/user file exists, and verify the
    # effective isolated config is empty with the real runtime CLI below.
    prefix = sys.prefix if prefix is None else prefix
    candidates = ['/etc/pip.conf', '/etc/xdg/pip/pip.conf',
                  str(Path(prefix) / 'pip.conf'),
                  str(Path(env['HOME']) / '.pip/pip.conf'),
                  str(Path(env['HOME']) / 'pip/pip.conf')]
    require(not any(os.path.lexists(p) for p in candidates), 'pip_config_present')
    require(env.get('PIP_CONFIG_FILE') == '/dev/null', 'pip_config_guard')


def base_argv(python=sys.executable):
    return [python, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
            '--no-input', '--no-cache-dir']


def pip_argv(root, python=sys.executable, contract=None):
    root = Path(root)
    retries = ['--retries', '0', '--resume-retries', '0']
    if contract is not None:
        require(set(contract) == {'retry_argv', 'evidence'}, 'pip_capability_probe')
        retries = contract['retry_argv']
        evidence = contract['evidence']
        require(retries in (['--retries', '0'], ['--retries', '0', '--resume-retries', '0']),
                'pip_capability_probe')
        require(evidence['help_exit_code'] == 0, 'pip_capability_probe')
        if retries != ['--retries', '0']:
            require(not evidence['missing_options'] and evidence['policy'] == 'explicit-zero-resume' and
                    evidence['reviewed_source_matches'] is None and evidence['pip_version'] != '25.0.1',
                    'pip_capability_probe')
        if retries == ['--retries', '0']:
            require(evidence['pip_version'] == '25.0.1' and
                    evidence['policy'] == 'audited-25.0.1-no-resume' and
                    evidence['reviewed_source_matches'] is True and
                    evidence['missing_options'] in ([], ['--resume-retries']), 'pip_capability_probe')
    return base_argv(python) + [
        'download', '--no-index', '--no-deps', '--require-hashes',
        '--only-binary=:all:'] + retries + [
        '--keyring-provider', 'disabled', '--proxy', PROXY,
        '--timeout', '30', '--progress-bar', 'off',
        '--dest', str(root / 'output'), '--requirement', str(root / 'one.txt')]


def load_compatibility():
    path = Path(__file__).resolve().with_name('pip_compatibility.py')
    spec = importlib.util.spec_from_file_location('_trusted_pip_compatibility', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed_pip_root():
    """Locate without importing/executing pip, in this exact isolated interpreter."""
    spec = importlib.util.find_spec('pip')
    require(spec is not None and spec.submodule_search_locations is not None and
            len(spec.submodule_search_locations) == 1 and spec.origin is not None,
            'pip_capability_probe')
    root = Path(spec.submodule_search_locations[0])
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and
            Path(spec.origin) == root / '__init__.py', 'pip_capability_probe')
    return root


def requirement(row):
    return row['name'] + ' @ ' + row['url'] + ' --hash=sha256:' + row['sha256'] + '\n'


def stop_process(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_capped(argv, env, deadline, file_limit=None, popen=subprocess.Popen):
    """Bound both pipes together. Never return raw output to host diagnostics."""
    require(time.monotonic() < deadline, 'deadline')
    def limits():
        if file_limit is not None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    proc = popen(argv, env=env, stdin=subprocess.DEVNULL,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 start_new_session=True, preexec_fn=limits)
    chunks = [bytearray(), bytearray()]
    selector = selectors.DefaultSelector()
    try:
        for i, pipe in enumerate((proc.stdout, proc.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, i)
        while selector.get_map():
            left = deadline - time.monotonic()
            require(left > 0, 'deadline')
            for key, _ in selector.select(min(left, 0.1)):
                part = os.read(key.fileobj.fileno(), 8192)
                if not part:
                    selector.unregister(key.fileobj)
                else:
                    require(sum(map(len, chunks)) + len(part) <= CAPTURE_LIMIT,
                            'pip_output_limit')
                    chunks[key.data].extend(part)
        require(time.monotonic() < deadline, 'deadline')
        try:
            code = proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise WorkerFailure('deadline') from None
        return code, bytes(chunks[0]), bytes(chunks[1])
    finally:
        stop_process(proc)
        selector.close()
        proc.stdout.close()
        proc.stderr.close()


def probe_capabilities(env, deadline, runner=run_capped, reporter=None, pip_root=None):
    if reporter: reporter.mark('pip_version_probe')
    code, version, error = runner(base_argv() + ['--version'], env, deadline)
    match = re.match(rb'^pip ([0-9]+\.[0-9]+(?:\.[0-9]+)?) ', version)
    require(code == 0 and not error and match, 'pip_version_probe')
    version_text = match.group(1).decode('ascii')
    if reporter:
        reporter.pip_version = version_text
        reporter.mark('pip_help_probe')
    code, help_text, _ = runner(base_argv() + ['download', '--help'], env, deadline)
    compatibility = load_compatibility()
    if version_text == compatibility.AUDITED_VERSION and pip_root is None:
        try:
            pip_root = installed_pip_root()
        except (WorkerFailure, OSError):
            pip_root = None  # The audited contract fails closed and retains help diagnostics.
    try:
        contract = compatibility.select_contract(version_text, code, help_text, pip_root)
    except compatibility.CompatibilityFailure as error:
        if reporter: reporter.capability = error.evidence
        raise WorkerFailure('pip_capability_probe') from None
    if reporter: reporter.capability = contract['evidence']
    if reporter: reporter.mark('pip_config_probe')
    code, config, error = runner(base_argv() + ['config', 'list'], env, deadline)
    require(code == 0 and not config.strip() and not error.strip(),
            'pip_effective_config')
    return contract


def ensure_tmpfs(root, mountinfo='/proc/self/mountinfo'):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(),
            'tmpfs_path')
    require(not any(root.iterdir()), 'tmpfs_not_fresh')
    entries = Path(mountinfo).read_text().splitlines()
    matched = []
    for line in entries:
        fields = line.split()
        split = fields.index('-')
        if fields[4] == str(root):
            matched.append((fields[split + 1], fields[split + 3]))
    require(len(matched) == 1 and matched[0][0] == 'tmpfs', 'tmpfs_required')
    # statvfs total blocks is the kernel-enforced tmpfs maximum, not free space.
    fs = os.statvfs(root)
    require(0 < fs.f_blocks * fs.f_frsize <= QUOTA, 'tmpfs_quota')


def verify_output(root, row):
    output = Path(root) / 'output'
    items = list(output.iterdir())
    require(len(items) == 1 and items[0].name == row['filename'], 'output_filename')
    path = items[0]
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            info.st_size == row['size_bytes'], 'output_size')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    require(digest.hexdigest() == row['sha256'], 'output_hash')


def serve_shim(listener, broker_socket, deadline):
    """Opaque loopback-to-Unix relay; no TLS parsing, credentials or logs.

    Executed only in a forked shim child before the parent starts pip. A small
    process-local selector bounds userspace buffering and concurrent sockets.
    Broker independently owns egress policy and encrypted-stream counters.
    """
    listener.setblocking(False)
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ)
    peers, pending = {}, {}
    eof, write_closed = set(), set()
    def drop(sock):
        other = peers.pop(sock, None)
        for item in (sock, other):
            if item is not None:
                peers.pop(item, None)
                pending.pop(item, None)
                eof.discard(item); write_closed.discard(item)
                try:
                    selector.unregister(item)
                except KeyError:
                    pass
                item.close()
    try:
        while time.monotonic() < deadline:
            for key, mask in selector.select(min(0.2, max(0, deadline-time.monotonic()))):
                sock = key.fileobj
                if sock is listener:
                    client, _ = listener.accept()
                    if len(peers) >= 8:
                        client.close()
                        raise WorkerFailure('shim_connections')
                    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    upstream.settimeout(min(5, max(.001, deadline-time.monotonic())))
                    try:
                        upstream.connect(broker_socket)
                    except OSError:
                        client.close(); upstream.close()
                        raise WorkerFailure('shim_broker') from None
                    for a, b in ((client, upstream), (upstream, client)):
                        a.setblocking(False); peers[a] = b; pending[a] = bytearray()
                        selector.register(a, selectors.EVENT_READ)
                    continue
                if sock not in peers:
                    continue
                if mask & selectors.EVENT_WRITE:
                    try:
                        written = sock.send(pending[sock])
                        del pending[sock][:written]
                    except BlockingIOError:
                        pass
                    except OSError:
                        drop(sock); continue
                if mask & selectors.EVENT_READ:
                    other = peers[sock]
                    try:
                        data = sock.recv(min(16384, 65536-len(pending[other])))
                    except BlockingIOError:
                        data = None
                    except OSError:
                        drop(sock); continue
                    if data == b'':
                        eof.add(sock)
                    if data:
                        pending[other].extend(data)
                for item in list(peers):
                    if peers[item] in eof and not pending[item] and item not in write_closed:
                        try:
                            item.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        write_closed.add(item)
                    events = (selectors.EVENT_READ if item not in eof and
                              len(pending[peers[item]]) < 65536 else 0)
                    if pending[item]:
                        events |= selectors.EVENT_WRITE
                    try:
                        selector.get_key(item)
                        if events: selector.modify(item, events)
                        else: selector.unregister(item)
                    except KeyError:
                        if events: selector.register(item, events)
                for item in list(peers):
                    if item in peers and item in eof and peers[item] in eof and not pending[item] and not pending[peers[item]]:
                        drop(item)
    finally:
        for sock in list(peers):
            if sock in peers: drop(sock)
        listener.close(); selector.close()


@contextmanager
def shim(broker_socket, deadline):
    """Fixed loopback proxy for a trusted, single-threaded container worker.

    The deadline is the supervisor-derived absolute monotonic deadline, never
    extended here. The sole outbound connection is the mounted Unix socket.
    """
    require(time.monotonic() < deadline, 'deadline')
    require(stat.S_ISSOCK(os.stat(broker_socket).st_mode), 'broker_socket')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(LOOPBACK); listener.listen(4)
        pid = os.fork()
    except BaseException:
        listener.close()
        raise
    if pid == 0:
        try:
            serve_shim(listener, broker_socket, deadline)
        except BaseException:
            os._exit(1)
        os._exit(0)
    listener.close()
    try:
        yield PROXY
        require(os.waitpid(pid, os.WNOHANG) == (0, 0), 'shim_failed')
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def export_candidate(root, candidate_root, row):
    candidate = Path(candidate_root)
    require(candidate.is_absolute() and candidate.resolve() == candidate and
            candidate.is_dir() and not any(candidate.iterdir()), 'candidate_path')
    source = Path(root) / 'output' / row['filename']
    target = candidate / row['filename']
    copied = 0
    try:
        with source.open('rb') as src, target.open('xb') as dst:
            while True:
                block = src.read(min(65536, row['size_bytes'] - copied + 1))
                if not block:
                    break
                copied += len(block)
                require(copied <= row['size_bytes'], 'candidate_size')
                dst.write(block)
            require(copied == row['size_bytes'], 'candidate_size')
            dst.flush(); os.fsync(dst.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def acquire(index, manifest, root, broker_socket, candidate_root, seconds=300, reporter=None):
    require(type(seconds) is int and 0 < seconds <= 300, 'deadline_bound')
    snapshot = verify_worker_scope()
    if reporter:
        reporter.write('worker-scope.json', sanitized_scope(snapshot))
        reporter.mark('deadline')
    deadline = worker_deadline(seconds)
    if reporter: reporter.mark('manifest')
    row = read_asset(manifest, index)
    if reporter: reporter.mark('storage')
    ensure_tmpfs(root)
    root = Path(root)
    for directory in ('home', 'tmp', 'output'):
        (root / directory).mkdir(mode=0o700)
    env = clean_env(root)
    if reporter: reporter.mark('config')
    ensure_no_configs(env)
    contract = probe_capabilities(env, min(deadline, time.monotonic() + 30), reporter=reporter)
    version = contract['evidence']['pip_version']
    (root / 'one.txt').write_text(requirement(row), encoding='ascii')
    if reporter: reporter.mark('acquisition')
    with shim(broker_socket, deadline):
        code, _, _ = run_capped(pip_argv(root, contract=contract), env, deadline, row['size_bytes'])
        # TLS is opaque to the broker: do not infer status 403 from pip logs.
        require(code == 0, 'pip_acquisition_failed')
        if reporter: reporter.mark('verification')
        verify_output(root, row)
        require(time.monotonic() < deadline, 'deadline')
        if reporter: reporter.mark('export')
        export_candidate(root, candidate_root, row)
        require(time.monotonic() < deadline, 'deadline')
        return {'status': 'ok', 'asset_index': index, 'pip_version': version,
                'filename': row['filename'], 'size_bytes': row['size_bytes'],
                'sha256': row['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-index', type=int, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--broker-socket', required=True)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args()
    reporter = None
    outcome = None
    reason = None
    try:
        reporter = Evidence()
        outcome = acquire(args.asset_index, args.manifest, args.output_root,
                          args.broker_socket, args.candidate_root, args.seconds, reporter)
    except WorkerFailure as error:
        reason = str(error)
    except BaseException:
        reason = 'worker_internal_failure'
    result = safe_result(args.asset_index, reporter.stage if reporter else 'scope',
                         reporter.pip_version if reporter else None, outcome, reason,
                         reporter.capability if reporter else None)
    if reporter:
        try:
            reporter.write('worker-result.json', result)
        except BaseException:
            result = safe_result(args.asset_index, reporter.stage, reporter.pip_version,
                                 reason='evidence_write_failed', capability=reporter.capability)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
