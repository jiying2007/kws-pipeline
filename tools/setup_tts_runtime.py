#!/usr/bin/env python3
"""Install the active Python 3.12 TTS role and fixed public Base assets only.

Reuses the retained TTS artifact lock and the active hash-checked installer
primitives. No model construction, generation, old campaign or API call occurs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import venv

import research_runtime
from setup_asr_runtime import acquire, safe_relative, save, sha_file, wheel_identity

ROOT = Path(__file__).resolve().parents[1]
ASSET_LOCK = ROOT / 'configs/runtime/qwen-base-tts-model-lock.json'
SOURCE_LOCK = ROOT / 'research/qwen6_tts/model-locks.json'


def locked_packages(manifest):
    research_runtime.verify_reference_locks(manifest, 'tts')
    rows = {}
    for reference in manifest['roles']['tts']['reference_locks']:
        for row in json.loads((ROOT / reference['path']).read_text())['files']:
            name = row['name'].lower().replace('_', '-')
            if name in rows and any(rows[name][key] != row[key] for key in ('version', 'filename', 'sha256')):
                raise ValueError('TTS runtime artifact conflict: ' + name)
            rows[name] = row
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reuse-downloads', type=Path)
    parser.add_argument('--setup-minutes', type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.setup_minutes <= 90:
        parser.error('setup budget must be 1..90 minutes')
    manifest = research_runtime.load_manifest()
    observed = research_runtime.check_interpreter(manifest)
    if (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
        raise ValueError('retained wheels require Linux x86_64')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt = root / 'setup-attempts' / (attempt + '.json')
    record = {'schema': 'active-tts-setup-v1', 'attempt': attempt, 'status': 'running', 'stage': 'download',
              'python': observed, 'active_python': manifest['python']['version'], 'models_executed': 0,
              'commands': [], 'installer_sha256': sha_file(__file__),
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'lock_sha256': {str(path.relative_to(ROOT)): sha_file(path) for path in
                              (research_runtime.MANIFEST, ASSET_LOCK, SOURCE_LOCK)}}
    save(receipt, record)
    deadline = time.monotonic() + args.setup_minutes * 60
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_NO_CACHE_DIR='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', MAX_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4',
               ORT_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')

    def command(argv, stage):
        record['stage'] = stage
        record['commands'].append({'argv': [str(value) for value in argv], 'stage': stage})
        save(receipt, record)
        subprocess.run([str(value) for value in argv], check=True, env=env, timeout=max(1, deadline - time.monotonic()))

    try:
        packages = locked_packages(manifest)
        for reference in manifest['roles']['tts']['reference_locks']:
            record['lock_sha256'][reference['path']] = reference['sha256']
        reused = []
        for row in packages.values():
            target = root / 'downloads' / safe_relative(row['filename'])
            old = args.reuse_downloads / row['filename'] if args.reuse_downloads is not None else None
            if not target.exists() and old is not None and old.is_file() and old.stat().st_size == row['size_bytes'] and sha_file(old) == row['sha256']:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, target)
                reused.append(row['filename'])
        jobs = [(row, root / 'downloads' / safe_relative(row['filename'])) for row in packages.values()]
        assets = json.loads(ASSET_LOCK.read_text())
        jobs.extend((row, root / 'models/base' / safe_relative(row['filename'])) for row in assets['files'])
        record['reused_download_files'] = reused
        with ThreadPoolExecutor(max_workers=4) as pool:
            record['acquired'] = list(pool.map(lambda pair: acquire(*pair, deadline), jobs))
        save(receipt, record)
        python = root / 'venv/bin/python'
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(root / 'venv')
        command([python, '-B', ROOT / 'tools/research_runtime.py', 'check'], 'check_python')
        prebuilt, sources, build_tools = [], [], []
        for row in packages.values():
            path = root / 'downloads' / row['filename']
            if path.suffix == '.whl':
                wheel_identity(path, row)
                (build_tools if row.get('build_tool') else prebuilt).append(path)
            else:
                sources.append((row, path))
        pip = [python, '-I', '-m', 'pip', 'install', '--no-index', '--no-deps', '--disable-pip-version-check']
        command([*pip, *build_tools], 'install_build_tools')
        command([*pip, *prebuilt], 'install_runtime_wheels')
        built = []
        for row, path in sources:
            destination = root / 'built-wheels' / row['name']
            destination.mkdir(parents=True, exist_ok=True)
            matches = list(destination.glob('*.whl'))
            if not matches:
                command([python, '-I', '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
                         '--wheel-dir', destination, path], 'build_' + row['name'])
                matches = list(destination.glob('*.whl'))
            if len(matches) != 1:
                raise ValueError('expected one built TTS runtime wheel')
            built.append(wheel_identity(matches[0], row))
        record['built_wheels'] = built
        command([*pip, *(row['path'] for row in built)], 'install_built_wheels')
        command([python, '-I', '-m', 'pip', 'check'], 'pip_check')
        command([python, '-B', ROOT / 'tools/research_runtime.py', 'check', '--role', 'tts', '--packages'], 'check_tts_role')
        source_check = """
import hashlib,importlib.metadata,json,sys
from pathlib import Path
lock=json.loads(Path(sys.argv[1]).read_text())['qwen_tts']['source_lock']
for row in lock:
 p=importlib.metadata.distribution(row['distribution']).locate_file(row['relative_path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('TTS package source drift: '+row['relative_path'])
print(json.dumps({'verified_tts_source_files':len(lock)}))
"""
        command([python, '-I', '-c', source_check, SOURCE_LOCK], 'check_tts_package_sources')
        inventory_code = "import importlib.metadata,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()},sort_keys=True))"
        inventory = json.loads(subprocess.check_output([str(python), '-I', '-c', inventory_code], env=env, text=True))
        for name, row in packages.items():
            if inventory.get(name) != row['version']:
                raise ValueError('installed TTS version mismatch: ' + name)
        record.update(status='complete', stage='complete', installed_packages=inventory,
                      completed_at=datetime.now(timezone.utc).isoformat())
        save(receipt, record)
        save(root / 'setup.json', record)
        print(json.dumps({'status': 'complete', 'packages': len(packages), 'models_executed': 0,
                          'python': str(python), 'receipt': str(receipt)}), flush=True)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        save(receipt, record)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
