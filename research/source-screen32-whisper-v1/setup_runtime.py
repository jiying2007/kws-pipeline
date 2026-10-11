"""Acquire the fixed official native distribution inside the admitted hosted scope.

No model import/inference, source build, editable install, floating resolution,
cache reuse, alternate model, retry, credential, or caller-selected destination.
"""
from pathlib import Path
import os
import copy
import http.client
import re
import signal
import socket
import ssl
import time
import urllib.error
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import venv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'source-screen32-v1'))
from common import (MODEL_SHA, MODEL_URL, LIMITS, BOUNDARY, require, read_json,
                    write_once, identity, file_hash, verify_inventory, atomic)
from runtime_scope import verify_runtime_scope

HOSTS = {'raw.githubusercontent.com', 'files.pythonhosted.org',
         'download.pytorch.org', 'download-r2.pytorch.org', 'openaipublic.azureedge.net'}


STAGES = frozenset(('entry', 'kernel_scope', 'python_version', 'fresh_directories',
    'disk_reserve', 'read_locks', 'copy_locks', 'source_lock_validation', 'source_download',
    'model_download', 'wheel_lock_validation', 'wheel_download', 'download_receipt',
    'runtime_environment', 'create_venv', 'install_wheels', 'check_dependencies',
    'installed_inventory', 'retain_public_source', 'asset_lock', 'setup_receipt', 'complete'))
REASONS = frozenset(('entry_contract', 'python_version', 'fresh_directories', 'disk_reserve',
    'source_revision', 'source_path', 'wheel_set', 'wheel_filename', 'unapproved_url',
    'unapproved_redirect', 'byte_cap', 'declared_length', 'sha256_mismatch', 'installed_size',
    'diagnostic_identity', 'duplicate_asset', 'validation_rejected', 'http_error', 'url_error',
    'tls_error', 'timeout', 'file_exists', 'permission_denied', 'connection_error', 'incomplete_read', 'os_error',
    'subprocess_exit', 'interrupted', 'runtime_error', 'unexpected_exception'))


class SetupFault(ValueError):
    def __init__(self, code, http_status=None):
        self.code = code if type(code) is str and code in REASONS else 'unexpected_exception'
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        super().__init__(self.code)


def check(ok, code):
    if not ok:
        raise SetupFault(code)


def public_reason(error):
    """Only fixed categories and bounded numeric codes, never str/repr/log/URL."""
    result = {'reason_code': 'unexpected_exception'}
    if isinstance(error, SetupFault):
        result['reason_code'] = error.code if type(error.code) is str and error.code in REASONS else 'unexpected_exception'
        if type(error.http_status) is int and 100 <= error.http_status <= 599:
            result['http_status'] = error.http_status
    elif isinstance(error, urllib.error.HTTPError):
        result['reason_code'] = 'http_error'
        if type(error.code) is int and 100 <= error.code <= 599:
            result['http_status'] = error.code
    elif isinstance(error, urllib.error.URLError):
        cause = error.reason
        result['reason_code'] = ('tls_error' if isinstance(cause, ssl.SSLError) else
                                 'timeout' if isinstance(cause, (TimeoutError, socket.timeout)) else 'url_error')
    elif isinstance(error, ssl.SSLError): result['reason_code'] = 'tls_error'
    elif isinstance(error, (TimeoutError, socket.timeout, subprocess.TimeoutExpired)):
        result['reason_code'] = 'timeout'
    elif isinstance(error, subprocess.CalledProcessError):
        result['reason_code'] = 'subprocess_exit'
        if type(error.returncode) is int and -255 <= error.returncode <= 255:
            result['returncode'] = error.returncode
    elif isinstance(error, FileExistsError): result['reason_code'] = 'file_exists'
    elif isinstance(error, PermissionError): result['reason_code'] = 'permission_denied'
    elif isinstance(error, (KeyboardInterrupt, SystemExit)): result['reason_code'] = 'interrupted'
    elif isinstance(error, ConnectionError): result['reason_code'] = 'connection_error'
    elif isinstance(error, http.client.IncompleteRead): result['reason_code'] = 'incomplete_read'
    elif isinstance(error, OSError): result['reason_code'] = 'os_error'
    elif isinstance(error, ValueError): result['reason_code'] = 'validation_rejected'
    elif isinstance(error, RuntimeError): result['reason_code'] = 'runtime_error'
    if isinstance(error, OSError) and type(error.errno) is int and 0 <= error.errno <= 4096:
        result['errno'] = error.errno
    return result


class SetupDiagnostics:
    """Durable allowlisted metadata, including partial acquisition, without raw logs."""
    def __init__(self, output):
        self.output = Path(output)
        self.started = time.monotonic()
        self.claimed = False
        self.state = {'schema': 'whisper-screen-setup-progress-v1', 'status': 'started',
            'stage': 'entry', 'completed_stages': [], 'lock_sha256': {}, 'asset': None, 'assets': [], 'response_body_bytes_observed': 0,
            'verified_assets': 0, 'model_calls': 0, 'retry_count': 0,
            'network_transfer_total_known': False,
            'byte_scope': 'Selected response body bytes returned to downloader; excludes protocol, unreturned redirect/error bodies and internal buffering',
            **BOUNDARY}

    def save(self):
        self.state['elapsed_seconds'] = time.monotonic() - self.started
        if not self.claimed:
            write_once(self.output / 'setup-progress.json', self.state)
            self.claimed = True
        else:
            atomic(self.output / 'setup-progress.json', self.state)

    def enter(self, stage):
        check(stage in STAGES, 'diagnostic_identity')
        previous = self.state['stage']
        if self.claimed and previous != stage and previous not in self.state['completed_stages']:
            self.state['completed_stages'].append(previous)
        self.state['stage'] = stage
        self.state['asset'] = None
        self.save()

    def bind_locks(self, identities):
        check(set(identities) == {'runtime-lock.json', 'source-lock.json'} and
              all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) is not None
                  for v in identities.values()), 'diagnostic_identity')
        self.state['lock_sha256'] = dict(identities)
        self.save()

    def begin_asset(self, group, index, row):
        check(group in ('source', 'model', 'wheel') and type(index) is int and 1 <= index <= 128,
              'diagnostic_identity')
        digest, size = row.get('sha256'), row.get('size_bytes')
        check(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None and
              (size is None or type(size) is int and 0 <= size <= 1024**3), 'diagnostic_identity')
        check(not any(a['group'] == group and a['index'] == index for a in self.state['assets']),
              'duplicate_asset')
        # Ordinal + digest maps only to the separately retained frozen lock.
        # No external filename, URL, headers, exception message or body is copied.
        asset = {'group': group, 'index': index, 'expected_sha256': digest, 'expected_bytes': size,
                 'status': 'started', 'response_body_bytes_observed': 0, 'observed_sha256': None, 'http_status': None}
        self.state['assets'].append(asset); self.state['asset'] = asset
        self.save()

    def http_response(self, status):
        if type(status) is int and 100 <= status <= 599:
            self.state['asset']['http_status'] = status
            self.save()

    def body_read(self, count):
        check(type(count) is int and 0 <= count <= 1024**2 and self.state['asset'] is not None,
              'diagnostic_identity')
        self.state['response_body_bytes_observed'] += count
        self.state['asset']['response_body_bytes_observed'] += count
        self.save()

    def hashed(self, digest):
        check(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None,
              'diagnostic_identity')
        self.state['asset']['observed_sha256'] = digest
        self.save()

    def verified(self):
        self.state['asset']['status'] = 'verified'
        self.state['verified_assets'] += 1
        self.save()

    def fail(self, error):
        self.state['status'] = 'failed_no_retry'; self.state['failure'] = public_reason(error)
        if self.state['asset'] is not None and self.state['asset']['status'] == 'started':
            self.state['asset']['status'] = 'failed_no_retry'
            if 'http_status' in self.state['failure']:
                self.state['asset']['http_status'] = self.state['failure']['http_status']
        self.save()
        value = copy.deepcopy(self.state); value['schema'] = 'whisper-screen-setup-failure-v1'
        atomic(self.output / 'setup-failure.json', value)


class FixedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlsplit(newurl)
        if not (url.scheme == 'https' and url.hostname in HOSTS and not url.username and not url.password):
            raise SetupFault('unapproved_redirect', http_status=code)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Downloader:
    def __init__(self, diagnostic):
        self.total = 0
        self.diagnostic = diagnostic
        self.opener = urllib.request.build_opener(FixedRedirect())

    def get(self, row, target):
        check(self.diagnostic.state['asset'] is not None and
              self.diagnostic.state['asset']['expected_sha256'] == row.get('sha256') and
              self.diagnostic.state['asset']['expected_bytes'] == row.get('size_bytes'), 'diagnostic_identity')
        url = urllib.parse.urlsplit(row['url'])
        check(url.scheme == 'https' and url.hostname in HOSTS and not url.username
              and not url.password, 'unapproved_url')
        limit = row.get('size_bytes', 1024**3)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.opener.open(row['url'], timeout=45) as source, target.open('xb') as destination:
            self.diagnostic.http_response(getattr(source, 'status', None))
            size = 0
            while True:
                block = source.read(min(1024**2, limit - size + 1))
                if not block: break
                size += len(block); self.total += len(block)
                self.diagnostic.body_read(len(block))
                check(size <= limit and self.total <= LIMITS['download_bytes'], 'byte_cap')
                destination.write(block)
        check('size_bytes' not in row or size == row['size_bytes'], 'declared_length')
        digest = file_hash(target); self.diagnostic.hashed(digest)
        check(digest == row['sha256'], 'sha256_mismatch')
        result = identity(target)
        self.diagnostic.verified()
        return result


def execute_setup(diagnostic):
    diagnostic.enter('entry')
    check(HERE == Path('/code/research/source-screen32-whisper-v1') and len(sys.argv) == 1,
            'entry_contract')
    diagnostic.enter('kernel_scope')
    scope = verify_runtime_scope(network_required=False)
    diagnostic.enter('python_version')
    check(sys.version_info[:2] == (3, 12), 'python_version')
    root, output = Path('/runtime'), Path('/output')
    diagnostic.enter('fresh_directories')
    check(not list(root.iterdir()) and {p.name for p in output.iterdir()} == {'setup-progress.json'},
          'fresh_directories')
    diagnostic.enter('disk_reserve')
    check(shutil.disk_usage(root).free >= 8 * 1024**3, 'disk_reserve')
    diagnostic.enter('read_locks')
    locks = {name: read_json(HERE / name) for name in ('runtime-lock.json', 'source-lock.json')}
    diagnostic.bind_locks({name: file_hash(HERE / name) for name in locks})
    diagnostic.enter('copy_locks')
    for name in locks:
        shutil.copyfile(HERE / name, output / name)
    download = Downloader(diagnostic); acquired = {}
    diagnostic.enter('source_lock_validation')
    source_rows = locks['source-lock.json']['files']
    check(locks['source-lock.json']['source_revision'] ==
            '86098128c0b4f24f0e2aa2994de830614b474227', 'source_revision')
    diagnostic.enter('source_download')
    for index, row in enumerate(source_rows, 1):
        diagnostic.begin_asset('source', index, row)
        name = row['path']
        check(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source_path')
        acquired['source/' + name] = download.get(row, root / 'source' / name)
    diagnostic.enter('wheel_lock_validation')
    wheels = locks['runtime-lock.json']['files']
    check(len(wheels) == len({row['name'] for row in wheels}), 'wheel_set')
    diagnostic.enter('wheel_download')
    for index, row in enumerate(wheels, 1):
        diagnostic.begin_asset('wheel', index, row)
        check(Path(row['filename']).name == row['filename'] and row['filename'].endswith('.whl'), 'wheel_filename')
        acquired['wheels/' + row['filename']] = download.get(row, root / 'wheels' / row['filename'])
    diagnostic.enter('runtime_environment')
    os.environ.update(HOME='/runtime/home', TMPDIR='/runtime/tmp', PIP_NO_INDEX='1',
                      PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ('home', 'tmp'): (root / name).mkdir()
    diagnostic.enter('create_venv')
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(root / 'venv')
    args = [str(root / 'venv/bin/python'), '-I', '-B', '-m', 'pip', 'install',
            '--no-index', '--no-deps', '--no-compile', '--disable-pip-version-check',
            *[str(root / 'wheels' / r['filename']) for r in wheels]]
    diagnostic.enter('install_wheels')
    subprocess.run(args, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diagnostic.enter('check_dependencies')
    subprocess.run([str(root / 'venv/bin/python'), '-I', '-B', '-m', 'pip', 'check'],
                   check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diagnostic.enter('installed_inventory')
    installed = root / 'venv/lib/python3.12/site-packages'
    members = {p.relative_to(installed).as_posix(): identity(p)
               for p in installed.rglob('*') if p.is_file()}
    check(sum(r['bytes'] for r in members.values()) <= 3 * 1024**3, 'installed_size')
    verify_inventory(installed, members)
    inventory = {'schema': 'whisper-screen-installed-runtime-v1', 'files': members}
    write_once(root / 'installed-inventory.json', inventory)
    shutil.copyfile(root / 'installed-inventory.json', output / 'installed-inventory.json')
    # Fail before acquiring checkpoint bytes if any locked runtime dependency fails.
    diagnostic.enter('model_download')
    model = {'url': MODEL_URL, 'sha256': MODEL_SHA}
    diagnostic.begin_asset('model', 1, model)
    acquired['model/small.pt'] = download.get(model, root / 'model/small.pt')
    diagnostic.enter('download_receipt')
    write_once(output / 'download-receipt.json', {'schema': 'whisper-screen-download-v1',
        'files': acquired, 'total_download_bytes': download.total, 'retry_count': 0,
        'model_loaded': False, **BOUNDARY})
    # Public source/tokenizer/frontend bytes only. The checkpoint is never an artifact.
    diagnostic.enter('retain_public_source')
    shutil.copytree(root / 'source', output / 'official-source')
    diagnostic.enter('asset_lock')
    asset = {'schema': 'whisper-screen-loaded-assets-lock-v1', 'model_id': 'openai/whisper-small-native',
             'files': {'small.pt': acquired['model/small.pt'],
                       **{r['path']: acquired['source/' + r['path']] for r in source_rows
                          if r['path'].startswith('whisper/assets/')}}}
    write_once(root / 'asset-lock.json', asset)
    shutil.copyfile(root / 'asset-lock.json', output / 'asset-lock.json')
    diagnostic.enter('setup_receipt')
    write_once(output / 'setup-receipt.json', {'schema': 'whisper-screen-setup-v1', 'status': 'complete',
        'kernel_scope': scope, 'runtime_lock_sha256': file_hash(HERE / 'runtime-lock.json'),
        'source_lock_sha256': file_hash(HERE / 'source-lock.json'),
        'asset_lock_sha256': file_hash(root / 'asset-lock.json'),
        'installed_inventory_sha256': file_hash(root / 'installed-inventory.json'),
        'model_calls': 0, 'source_builds': 0, 'total_download_bytes': download.total, **BOUNDARY})

    diagnostic.enter('complete')
    diagnostic.state['status'] = 'complete'
    diagnostic.save()


def main():
    diagnostic = None
    def interrupted(signum, frame):
        raise SetupFault('interrupted')
    try:
        diagnostic = SetupDiagnostics(Path('/output'))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        execute_setup(diagnostic)
        return 0
    except BaseException as error:
        reason = public_reason(error)
        try:
            if diagnostic is not None:
                diagnostic.fail(error)
                saved = True
            else:
                saved = False
        except BaseException:
            saved = False
        # The outer launcher need not expose container stdout or arbitrary logs.
        print({'status': 'failed_no_retry', 'diagnostic_saved': saved, **reason})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
