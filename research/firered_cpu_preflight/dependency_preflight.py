#!/usr/bin/env python3
"""UNEXECUTED candidate: isolated dependency-only CPython 3.11 Linux preflight.
Fetches official package files, verifies index/PyPI hashes, installs into a NEW
venv, and runs tiny import/tensor/fbank checks. Never fetches model/audio assets.
This captures CPU wheel hashes for review; it does not qualify the ASR model.
"""
import argparse
import email.parser
import hashlib
import html.parser
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import venv
import zipfile

HERE = Path(__file__).resolve().parent
OFFICIAL_CPU_HOSTS = {'download.pytorch.org', 'download-r2.pytorch.org'}

class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for key, value in attrs:
                if key == 'href' and value:
                    self.hrefs.append(value)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def get_cpu_index_entry(package):
    with urllib.request.urlopen(package['index_url'], timeout=60) as response:
        parser = Links()
        parser.feed(response.read(10_000_000).decode('utf-8'))
    matches = []
    for href in parser.hrefs:
        url = urllib.parse.urljoin(package['index_url'], href)
        parsed = urllib.parse.urlsplit(url)
        filename = urllib.parse.unquote(parsed.path.rsplit('/', 1)[-1])
        if filename != package['filename']:
            continue
        if parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_CPU_HOSTS:
            raise RuntimeError('Unexpected CPU wheel host')
        expected = urllib.parse.parse_qs(parsed.fragment).get('sha256', [])
        if len(expected) != 1 or len(expected[0]) != 64 or any(c not in '0123456789abcdef' for c in expected[0]):
            raise RuntimeError('Official CPU index did not provide one valid SHA256')
        matches.append((urllib.parse.urlunsplit(parsed._replace(fragment='')), expected[0]))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one exact CPU wheel index entry: {package["filename"]}')
    return matches[0]

def download_checked(url, expected, target):
    with urllib.request.urlopen(url, timeout=120) as response:
        final = urllib.parse.urlsplit(response.url)
        if final.scheme != 'https' or final.hostname not in OFFICIAL_CPU_HOSTS:
            raise RuntimeError('Unexpected CPU download redirect')
        count = 0
        with target.open('xb') as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                count += len(chunk)
                if count > 2_000_000_000:
                    raise RuntimeError('CPU wheel exceeds bounded download limit')
                out.write(chunk)
    actual = sha256(target)
    if actual != expected:
        raise RuntimeError(f'CPU wheel hash mismatch for {target.name}')
    return count

def read_metadata(path):
    with zipfile.ZipFile(path) as wheel:
        names = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise RuntimeError('Wheel must contain exactly one METADATA')
        raw = wheel.read(names[0]).decode('utf-8')
    parsed = email.parser.Parser().parsestr(raw)
    return raw, parsed

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True, help='NEW directory for this isolated job')
    ap.add_argument('--allow-network-and-install', action='store_true', help='Required explicit execution gate')
    args = ap.parse_args()
    if not args.allow_network_and_install:
        ap.error('Not executed: --allow-network-and-install is required')
    if sys.version_info[:2] != (3, 11) or sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This candidate is only for CPython 3.11 Linux x86_64')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wheelhouse = output / 'wheelhouse'
    wheelhouse.mkdir()
    if shutil.disk_usage(output).free < 5 * 1024**3:
        raise RuntimeError('Require at least 5 GiB free for this dependency-only stage')
    proposal = json.loads((HERE / 'proposal.freeze-unready.json').read_text())
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    # The invoking runner interpreter is only used to create a new isolated environment.
    # Never install into its environment or the user's existing environment.
    venv_path = output / 'venv'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(venv_path)
    python = venv_path / 'bin' / 'python'
    receipt = {'asr_qualified': False, 'model_or_audio_downloaded': False,
               'python': sys.version, 'platform': platform.platform(),
               'proposal': 'proposal.freeze-unready.json', 'cpu_index_receipts': [], 'wheel_metadata': []}
    for package in proposal['packages']:
        if package['name'] not in ('torch', 'torchaudio'):
            continue
        url, expected = get_cpu_index_entry(package)
        path = wheelhouse / package['filename']
        size = download_checked(url, expected, path)
        receipt['cpu_index_receipts'].append({'name': package['name'], 'version': package['version'],
            'index_url': package['index_url'], 'url': url, 'filename': path.name,
            'sha256_from_index': expected, 'sha256_downloaded': sha256(path), 'size_bytes': size})
    subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'download', '--no-deps',
        '--no-cache-dir', '--dest', str(wheelhouse), '-r',
        str(HERE / 'pypi-subset.cp311-linux-x86_64.requirements.txt')], check=True, env=env)
    # Verify every file, including the metadata-only PyPI proposal, before installing.
    for package in proposal['packages']:
        path = wheelhouse / package['filename']
        if package['sha256'] is not None and sha256(path) != package['sha256']:
            raise RuntimeError(f'PyPI wheel hash mismatch: {path.name}')
        raw, metadata = read_metadata(path)
        canonical = lambda value: re.sub(r'[-_.]+', '-', value).lower()
        if canonical(metadata['Name']) != canonical(package['name']) or metadata['Version'] != package['version']:
            raise RuntimeError('Wheel METADATA identity differs from exact proposal')
        (output / (path.name + '.METADATA')).write_text(raw)
        receipt['wheel_metadata'].append({'name': metadata['Name'], 'version': metadata['Version'],
            'filename': path.name, 'requires_python': metadata['Requires-Python'],
            'requires_dist': metadata.get_all('Requires-Dist', [])})
    # Use the packaging implementation bundled in new venv's pip, without adding a package.
    closure_check = r'''
import json, sys
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
r=json.load(open(sys.argv[1]))
installed={canonicalize_name(p['name']):Version(p['version']) for p in r['wheel_metadata']}
assert len(installed)==14, installed
for name in installed:
    assert not name.startswith(('nvidia-', 'cuda-')) and name != 'triton', name
for package in r['wheel_metadata']:
    for raw in package['requires_dist']:
        req=Requirement(raw)
        if req.marker is not None and not req.marker.evaluate({'extra':''}):
            continue
        name=canonicalize_name(req.name)
        assert name in installed, (package['name'],raw,'not in proposed closure')
        assert installed[name] in req.specifier, (package['name'],raw,installed[name])
print('All 14 exact wheel metadata files satisfy the active runtime dependency closure')
'''
    receipt_file = output / 'dependency-receipt.json'
    receipt_file.write_text(json.dumps(receipt, indent=2) + '\n')
    subprocess.run([str(python), '-I', '-c', closure_check, str(receipt_file)], check=True, env=env)
    wheels = [str(wheelhouse / p['filename']) for p in proposal['packages']]
    subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--no-index',
                    '--no-deps', '--no-cache-dir', *wheels], check=True, env=env)
    subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'check'], check=True, env=env)
    tiny_checks = r'''
import importlib.metadata, json
import torch, torchaudio, numpy as np, kaldiio, kaldi_native_fbank as knf, sentencepiece
assert torch.__version__ == '2.10.0+cpu', torch.__version__
assert torchaudio.__version__ == '2.10.0+cpu', torchaudio.__version__
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(4)
a=np.arange(6,dtype=np.float32).reshape(2,3)
assert torch.equal(torch.from_numpy(a) @ torch.ones(3,1), torch.tensor([[3.],[12.]]))
o=knf.FbankOptions(); o.mel_opts.num_bins=80; o.frame_opts.dither=0
f=knf.OnlineFbank(o); f.accept_waveform(16000,[0.0]*4000)
assert f.num_frames_ready > 0 and len(f.get_frame(0))==80
versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
assert not any(n.lower().startswith(('nvidia-','cuda-')) or n.lower()=='triton' for n in versions)
print(json.dumps({'imports_ok':True,'tiny_torch_numpy_check':True,'tiny_fbank_check':True,
 'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
 'installed_versions':versions,'model_or_audio_loaded':False},sort_keys=True))
'''
    result = subprocess.run([str(python), '-I', '-c', tiny_checks], check=True, env=env,
                            capture_output=True, text=True)
    receipt['tiny_checks'] = json.loads(result.stdout.strip())
    receipt['dependency_preflight_passed'] = True
    receipt['freeze_requires_receipt_review'] = True
    receipt_file.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Dependency-only preflight passed. Review {receipt_file}; ASR remains unqualified.')

if __name__ == '__main__':
    main()
