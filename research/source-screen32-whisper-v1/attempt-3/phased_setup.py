"""Fixed, source-only phased setup. No torch import or native inference.

The host alone owns the aggregate TLS ledger. These phases cannot reset it.
Only source/model use the old bounded stdlib body downloader; pip's transport
accounting is intentionally never represented as response-body accounting.
"""
from pathlib import Path
from contextlib import contextmanager
import os
import math
import shutil
import signal
import subprocess
import sys
import time
import tempfile
import urllib.request
import venv

HERE = Path(__file__).resolve().parent
LOCKS = HERE.parent
sys.path.insert(0, str(LOCKS))
sys.path.insert(0, str(HERE))
from common import (BOUNDARY, MODEL_SHA, MODEL_URL, SOURCE_REVISION, LIMITS,
                    atomic, file_hash, identity, read_json, verify_inventory, write_once)
from setup_runtime import (Downloader, FixedRedirect, SetupDiagnostics, SetupFault,
                           check, public_reason)

PHASES = ('install', 'source', 'model', 'finalize')
MODEL_BYTES = 483617219
BOOTSTRAP_TMP = Path('/wheel-tmpfs')
LOCK_HASHES = {
    'runtime-lock.json': '881443c6e66acb99223137bd8342788468f814d0a1c3e73ba33a27da3ad1fe6d',
    'source-lock.json': '0635fb8759fb56439f59db6956612f78cc8903c239f81f3eb7f9b0c47eb99649'}


class Diagnostics(SetupDiagnostics):
    def __init__(self, output, phase):
        check(phase in PHASES, 'entry_contract')
        self.phase = phase
        super().__init__(output)
        self.state['schema'] = 'whisper-screen-phased-progress-v1'
        self.state['phase'] = phase
        self.state['byte_scope'] = 'Source/model selected response bodies only; pip bytes excluded; aggregate TLS ledger belongs to host'

    def save(self):
        path = self.output / ('phase-' + self.phase + '-progress.json')
        if not self.claimed:
            write_once(path, self.state)
            self.claimed = True
        else:
            atomic(path, self.state)

    def fail(self, error):
        self.state['status'] = 'failed_no_retry'
        self.state['failure'] = public_reason(error)
        self.save()
        atomic(self.output / 'phased-setup-failure.json', {
            'schema': 'whisper-screen-phased-failure-v1', 'phase': self.phase,
            'status': 'failed_no_retry', 'model_calls': 0, 'retry_count': 0,
            **public_reason(error), **BOUNDARY})


def load_locks(directory=LOCKS):
    check(all(file_hash(directory / name) == digest for name, digest in LOCK_HASHES.items()),
          'sha256_mismatch')
    runtime, source = (read_json(directory / name) for name in LOCK_HASHES)
    wheels, sources = runtime['files'], source['files']
    check(len(wheels) == 22 and len({r['name'] for r in wheels}) == 22 and
          len({r['filename'] for r in wheels}) == 22, 'wheel_set')
    check(source['source_revision'] == SOURCE_REVISION and len(sources) == 22,
          'source_revision')
    for row in wheels:
        name = row['filename']
        check(Path(name).name == name and name.endswith('.whl'), 'wheel_filename')
    for row in sources:
        name = row['path']
        check(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source_path')
    return wheels, sources


def wheel_inventory(wheels):
    return {r['filename']: {'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in wheels}


def installed_inventory(root):
    inventory = read_json(root / 'installed-inventory.json')
    check(inventory.get('schema') == 'whisper-screen-installed-runtime-v1', 'validation_rejected')
    members = inventory['files']
    check(sum(r['bytes'] for r in members.values()) <= 3 * 1024**3, 'installed_size')
    verify_inventory(root / 'venv/lib/python3.12/site-packages', members)
    return inventory


@contextmanager
def bootstrap_environment(env):
    """Bound ensurepip temporary files and remove inherited secrets/config.

    EnvBuilder forwards the process environment to its bootstrap subprocess.
    Reset tempfile's cached choice so it cannot silently use a previous path.
    Restore both process state and cache, including when bootstrap fails.
    """
    previous, cached = dict(os.environ), tempfile.tempdir
    try:
        os.environ.clear()
        os.environ.update(env)
        tempfile.tempdir = None
        check(BOOTSTRAP_TMP.is_dir() and not BOOTSTRAP_TMP.is_symlink() and
              tempfile.gettempdir() == str(BOOTSTRAP_TMP), 'entry_contract')
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)
        tempfile.tempdir = cached


class PhasedSetup:
    """Injectable paths/dependencies for offline fixture tests; CLI paths are fixed."""
    def __init__(self, root, output, locks=LOCKS):
        self.root, self.output, self.locks = Path(root), Path(output), Path(locks)

    def prior(self, phase):
        value = read_json(self.output / ('phase-' + phase + '-complete.json'))
        check(value.get('status') == 'complete' and value.get('phase') == phase and
              value.get('lock_sha256') == LOCK_HASHES, 'validation_rejected')
        return value

    def run(self, phase, diagnostic, scope, opener=None):
        check(phase in PHASES and not (self.output / 'phased-setup-failure.json').exists(), 'entry_contract')
        check(not (self.output / ('phase-' + phase + '-complete.json')).exists(), 'file_exists')
        wheels, sources = load_locks(self.locks)
        diagnostic.bind_locks(LOCK_HASHES)
        if phase == 'install':
            self.install(wheels, diagnostic)
        else:
            self.prior('install')
            installed_inventory(self.root)
            if phase == 'source':
                self.materialize(sources, 'source', diagnostic, opener)
            elif phase == 'model':
                self.prior('source')
                verify_inventory(self.root / 'source', {r['path']: {'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in sources})
                self.materialize([{'path': 'small.pt', 'url': MODEL_URL,
                                   'sha256': MODEL_SHA, 'size_bytes': MODEL_BYTES}], 'model', diagnostic, opener)
            else:
                self.finalize(wheels, sources, scope)
        write_once(self.output / ('phase-' + phase + '-complete.json'), {
            'schema': 'whisper-screen-phase-complete-v1', 'phase': phase, 'status': 'complete',
            'lock_sha256': LOCK_HASHES, 'model_calls': 0, 'source_builds': 0,
            'selected_source_model_body_bytes': diagnostic.state['response_body_bytes_observed'], **BOUNDARY})
        diagnostic.state['status'] = 'complete'
        diagnostic.save()

    def install(self, wheels, diagnostic):
        root = self.root
        check({p.name for p in root.iterdir()} == {'wheels'}, 'fresh_directories')
        check(shutil.disk_usage(root).free >= 8 * 1024**3, 'disk_reserve')
        verify_inventory(root / 'wheels', wheel_inventory(wheels))
        for name in LOCK_HASHES:
            shutil.copyfile(self.locks / name, self.output / name)
        for name in ('home', 'tmp'):
            (root / name).mkdir()
        # An allowlisted environment cannot inherit proxy/index/user-site settings.
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(root / 'home'),
               'TMPDIR': str(BOOTSTRAP_TMP), 'PIP_CONFIG_FILE': '/dev/null',
               'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
               'PYTHONDONTWRITEBYTECODE': '1'}
        diagnostic.enter('create_venv')
        with bootstrap_environment(env):
            venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(root / 'venv')
        python = str(root / 'venv/bin/python')
        diagnostic.enter('install_wheels')
        subprocess.run([python, '-I', '-B', '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--no-compile', '--disable-pip-version-check',
                        *[str(root / 'wheels' / r['filename']) for r in wheels]],
                       env=env, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diagnostic.enter('check_dependencies')
        subprocess.run([python, '-I', '-B', '-m', 'pip', 'check'], env=env, check=True,
                       timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        diagnostic.enter('installed_inventory')
        installed = root / 'venv/lib/python3.12/site-packages'
        members = {p.relative_to(installed).as_posix(): identity(p) for p in installed.rglob('*') if p.is_file()}
        check(bool(members) and sum(r['bytes'] for r in members.values()) <= 3 * 1024**3, 'installed_size')
        verify_inventory(installed, members)
        write_once(root / 'installed-inventory.json', {'schema': 'whisper-screen-installed-runtime-v1', 'files': members})
        shutil.copyfile(root / 'installed-inventory.json', self.output / 'installed-inventory.json')

    def materialize(self, rows, group, diagnostic, opener):
        check(opener is not None, 'entry_contract')
        diagnostic.enter(group + '_download')
        download = Downloader(diagnostic)
        download.opener = opener
        for index, row in enumerate(rows, 1):
            diagnostic.begin_asset(group, index, row)
            download.get(row, self.root / group / row['path'])

    def finalize(self, wheels, sources, scope):
        self.prior('source'); self.prior('model')
        root, output = self.root, self.output
        verify_inventory(root / 'wheels', wheel_inventory(wheels))
        source_members = {r['path']: {'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in sources}
        verify_inventory(root / 'source', source_members)
        model = {'bytes': MODEL_BYTES, 'sha256': MODEL_SHA}
        verify_inventory(root / 'model', {'small.pt': model})
        acquisition = validated_acquisition(Path('/acquisition/receipt.json'))
        shutil.copytree(root / 'source', output / 'official-source')
        asset = {'schema': 'whisper-screen-loaded-assets-lock-v1', 'model_id': 'openai/whisper-small-native',
                 'files': {'small.pt': model, **{k: v for k, v in source_members.items() if k.startswith('whisper/assets/')}}}
        write_once(root / 'asset-lock.json', asset)
        shutil.copyfile(root / 'asset-lock.json', output / 'asset-lock.json')
        write_once(output / 'download-receipt.json', {
            'schema': 'whisper-screen-download-bounded-tls-v1',
            'files': {**{'wheels/' + k: v for k, v in wheel_inventory(wheels).items()},
                      **{'source/' + k: v for k, v in source_members.items()}, 'model/small.pt': model},
            'acquisition': acquisition, 'retry_count': 0, 'model_loaded': False, **BOUNDARY})
        write_once(output / 'setup-receipt.json', {'schema': 'whisper-screen-setup-v1', 'status': 'complete',
            'kernel_scope': scope, 'runtime_lock_sha256': LOCK_HASHES['runtime-lock.json'],
            'source_lock_sha256': LOCK_HASHES['source-lock.json'],
            'asset_lock_sha256': file_hash(root / 'asset-lock.json'),
            'installed_inventory_sha256': file_hash(root / 'installed-inventory.json'),
            'model_calls': 0, 'source_builds': 0, 'acquisition': acquisition, **BOUNDARY})


def validated_acquisition(path):
    """Host-owned, mounted read-only; fixed numeric metadata only."""
    data = read_json(path)
    check(data.get('schema') == 'bounded-host-acquisition-v1' and
          data.get('status') == 'complete', 'validation_rejected')
    count = data.get('total_read_bytes')
    assets = data.get('assets')
    check(type(count) is int and 0 < count <= LIMITS['download_bytes'] and
          data.get('aggregate_cap') == LIMITS['download_bytes'] and
          data.get('byte_scope') == 'bidirectional_tunnel_socket_recv_before_forwarding', 'byte_cap')
    check(type(assets) is list and len(assets) == 24 and
          [row.get('kind') for row in assets] == ['wheel'] * 22 + ['source', 'model'], 'validation_rejected')
    observed, charged = 0, 0
    clean = []
    for index, row in enumerate(assets, 1):
        read, debit = row.get('observed_ciphertext_bytes'), row.get('budget_charge_bytes')
        check(row.get('status') == 'verified' and type(read) is int and
              type(debit) is int and 0 < read <= debit <= LIMITS['download_bytes'], 'byte_cap')
        observed += read; charged += debit
        clean.append({'index': index, 'kind': row['kind'], 'observed_ciphertext_bytes': read,
                      'budget_charge_bytes': debit, 'status': 'verified'})
    check(observed == count and charged <= LIMITS['download_bytes'], 'byte_cap')
    return {'schema': data['schema'], 'status': 'complete', 'total_read_bytes': count,
            'aggregate_cap': LIMITS['download_bytes'], 'assets': clean, 'retry_count': 0,
            'byte_scope': data['byte_scope'],
            'excludes': 'CONNECT headers, kernel buffering and network overhead'}


def main():
    diagnostic = None
    try:
        check(HERE == Path('/code/research/source-screen32-whisper-v1/attempt-3') and
              len(sys.argv) == 2 and sys.argv[1] in PHASES, 'entry_contract')
        check(sys.version_info[:2] == (3, 12), 'python_version')
        phase = sys.argv[1]
        diagnostic = Diagnostics(Path('/output'), phase)
        diagnostic.enter('entry')
        def interrupted(signum, frame):
            raise SetupFault('interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        sys.path.insert(0, '/opt')
        if phase in ('source', 'model'):
            from bounded_host import verify_effective_scope
            scope = verify_effective_scope('worker')
        else:
            from setup_runtime import verify_runtime_scope
            scope = verify_runtime_scope(network_required=True)
        setup = PhasedSetup('/runtime', '/output')
        if phase in ('source', 'model'):
            from pip_acquire import shim
            deadline = read_json(Path('/run/kws/deadline.json'))['deadline_monotonic']
            check(type(deadline) in (int, float) and math.isfinite(deadline) and
                  time.monotonic() < deadline <= time.monotonic() + LIMITS['setup_seconds'], 'entry_contract')
            for key in list(os.environ):
                if key.lower().endswith('_proxy'):
                    del os.environ[key]
            with shim('/run/kws/broker.sock', deadline) as proxy_url:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': proxy_url}), FixedRedirect())
                setup.run(phase, diagnostic, scope, opener)
        else:
            check(not Path('/run/kws').exists(), 'entry_contract')
            setup.run(phase, diagnostic, scope)
        return 0
    except BaseException as error:
        try:
            if diagnostic is not None:
                diagnostic.fail(error)
        except BaseException:
            pass
        print({'status': 'failed_no_retry', **public_reason(error)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
