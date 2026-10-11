"""Acquire the fixed official native distribution inside the admitted hosted scope.

No model import/inference, source build, editable install, floating resolution,
cache reuse, alternate model, retry, credential, or caller-selected destination.
"""
from pathlib import Path
import os
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
                    write_once, identity, file_hash, verify_inventory)
from runtime_scope import verify_runtime_scope

HOSTS = {'raw.githubusercontent.com', 'files.pythonhosted.org',
         'download.pytorch.org', 'download-r2.pytorch.org', 'openaipublic.azureedge.net'}


class FixedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlsplit(newurl)
        require(url.scheme == 'https' and url.hostname in HOSTS and not url.username
                and not url.password, 'unreviewed download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Downloader:
    def __init__(self):
        self.total = 0
        self.opener = urllib.request.build_opener(FixedRedirect())

    def get(self, row, target):
        url = urllib.parse.urlsplit(row['url'])
        require(url.scheme == 'https' and url.hostname in HOSTS and not url.username
                and not url.password, 'official download URL required')
        limit = row.get('size_bytes', 1024**3)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.opener.open(row['url'], timeout=45) as source, target.open('xb') as destination:
            size = 0
            while True:
                block = source.read(min(1024**2, limit - size + 1))
                if not block: break
                size += len(block); self.total += len(block)
                require(size <= limit and self.total <= LIMITS['download_bytes'], 'download hard byte cap')
                destination.write(block)
        require('size_bytes' not in row or size == row['size_bytes'], 'declared download length')
        require(file_hash(target) == row['sha256'], 'download SHA256 mismatch')
        return identity(target)


def main():
    require(HERE == Path('/code/research/source-screen32-whisper-v1') and len(sys.argv) == 1,
            'fixed hosted setup entry')
    scope = verify_runtime_scope(network_required=False)
    require(sys.version_info[:2] == (3, 12), 'frozen CPython minor')
    root, output = Path('/runtime'), Path('/output')
    require(not list(root.iterdir()) and not list(output.iterdir()), 'fresh runtime and setup output required')
    require(shutil.disk_usage(root).free >= 8 * 1024**3, 'setup disk reserve')
    locks = {name: read_json(HERE / name) for name in ('runtime-lock.json', 'source-lock.json')}
    for name in locks:
        shutil.copyfile(HERE / name, output / name)
    download = Downloader(); acquired = {}
    source_rows = locks['source-lock.json']['files']
    require(locks['source-lock.json']['source_revision'] ==
            '86098128c0b4f24f0e2aa2994de830614b474227', 'fixed source revision')
    for row in source_rows:
        name = row['path']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source member path')
        acquired['source/' + name] = download.get(row, root / 'source' / name)
    model = {'url': MODEL_URL, 'sha256': MODEL_SHA}
    acquired['model/small.pt'] = download.get(model, root / 'model/small.pt')
    wheels = locks['runtime-lock.json']['files']
    require(len(wheels) == len({row['name'] for row in wheels}), 'unique wheel set')
    for row in wheels:
        require(Path(row['filename']).name == row['filename'] and row['filename'].endswith('.whl'), 'wheel-only setup')
        acquired['wheels/' + row['filename']] = download.get(row, root / 'wheels' / row['filename'])
    write_once(output / 'download-receipt.json', {'schema': 'whisper-screen-download-v1',
        'files': acquired, 'total_download_bytes': download.total, 'retry_count': 0,
        'model_loaded': False, **BOUNDARY})
    os.environ.update(HOME='/runtime/home', TMPDIR='/runtime/tmp', PIP_NO_INDEX='1',
                      PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ('home', 'tmp'): (root / name).mkdir()
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(root / 'venv')
    args = [str(root / 'venv/bin/python'), '-I', '-B', '-m', 'pip', 'install',
            '--no-index', '--no-deps', '--no-compile', '--disable-pip-version-check',
            *[str(root / 'wheels' / r['filename']) for r in wheels]]
    subprocess.run(args, check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([str(root / 'venv/bin/python'), '-I', '-B', '-m', 'pip', 'check'],
                   check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    installed = root / 'venv/lib/python3.12/site-packages'
    members = {p.relative_to(installed).as_posix(): identity(p)
               for p in installed.rglob('*') if p.is_file()}
    require(sum(r['bytes'] for r in members.values()) <= 3 * 1024**3, 'installed byte cap')
    verify_inventory(installed, members)
    inventory = {'schema': 'whisper-screen-installed-runtime-v1', 'files': members}
    write_once(root / 'installed-inventory.json', inventory)
    shutil.copyfile(root / 'installed-inventory.json', output / 'installed-inventory.json')
    # Public source/tokenizer/frontend bytes only. The checkpoint is never an artifact.
    shutil.copytree(root / 'source', output / 'official-source')
    asset = {'schema': 'whisper-screen-loaded-assets-lock-v1', 'model_id': 'openai/whisper-small-native',
             'files': {'small.pt': acquired['model/small.pt'],
                       **{r['path']: acquired['source/' + r['path']] for r in source_rows
                          if r['path'].startswith('whisper/assets/')}}}
    write_once(root / 'asset-lock.json', asset)
    shutil.copyfile(root / 'asset-lock.json', output / 'asset-lock.json')
    write_once(output / 'setup-receipt.json', {'schema': 'whisper-screen-setup-v1', 'status': 'complete',
        'kernel_scope': scope, 'runtime_lock_sha256': file_hash(HERE / 'runtime-lock.json'),
        'source_lock_sha256': file_hash(HERE / 'source-lock.json'),
        'asset_lock_sha256': file_hash(root / 'asset-lock.json'),
        'installed_inventory_sha256': file_hash(root / 'installed-inventory.json'),
        'model_calls': 0, 'source_builds': 0, 'total_download_bytes': download.total, **BOUNDARY})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(type(error).__name__)
        raise SystemExit(1)
