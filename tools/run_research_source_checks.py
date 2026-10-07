#!/usr/bin/env python3
"""Run only explicit offline checks in an ephemeral historical source projection.

This command never downloads fixtures and never installs a dependency. Archived
workflows are copied as data into a temporary directory, never into the checkout.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from verify_research_sources import MANIFEST, relative, verify

ALLOWLIST = 'research/consolidation/source-offline-allowlist.json'
SUPPORT = ['research/consolidation/source_tests/test_n1_companion_offline.py',
           'research/consolidation/prepare_n1_saved_fixtures.py',
           'research/consolidation/source_fixtures/n1-saved-inputs.json']

def selected_checks(entries, suite, only=()):
    seen = set()
    selected = []
    for row in entries:
        if (not isinstance(row.get('id'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,180}', row['id'])
                or row['id'] in seen):
            raise ValueError('invalid or duplicate offline check id')
        seen.add(row['id'])
        cwd = row.get('cwd')
        if cwd != '.':
            relative(cwd)
        argv = row.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(s, str) for s in argv):
            raise ValueError('invalid check arguments')
        if argv[0] not in ('python3', 'env'):
            raise ValueError('only reviewed Python checks are allowed')
        if argv[0] == 'env':
            allowed_env = {'A20_QWEN20_DATA_ROOT', 'A20_N0_DATA_ROOT', 'MELO5_LEADING_FIXTURES'}
            position = 1
            assigned = set()
            while position < len(argv) and '=' in argv[position]:
                name, value = argv[position].split('=', 1)
                if name not in allowed_env or name in assigned or not value.startswith('../offline-fixtures/'):
                    raise ValueError('unsupported fixture environment assignment')
                relative(value[len('../offline-fixtures/'):])
                assigned.add(name)
                position += 1
            if not assigned or position >= len(argv) or argv[position] != 'python3':
                raise ValueError('env must contain fixture assignments followed by python3')
        if any(s.startswith('--execute') or s == 'workflow_dispatch' for s in argv):
            raise ValueError('execution flag prohibited in offline allowlist')
        # A single, exact test file is required for each unittest discovery.
        if 'discover' in argv and ('-p' not in argv or any('*' in s or '?' in s for s in argv)):
            raise ValueError('unbounded test discovery prohibited')
        saved = any('PREPARED_FIXTURE:' in s for s in row.get('requirements', []))
        if (suite == 'synthetic' and saved) or (suite == 'saved' and not saved):
            continue
        if only and row['id'] not in only:
            continue
        selected.append(row)
    if only and set(only) - seen:
        raise ValueError('unknown check id')
    if not selected:
        raise ValueError('no checks selected')
    return selected

def make_projection(root, projected, manifest):
    # Closed manifests, rather than test/source globbing, determine copied files.
    core = json.loads((root / 'research/consolidation/core-2026-10-07.json').read_text())
    paths = {r['path'] for r in manifest['files']} | {r['path'] for r in core['files']} | set(SUPPORT)
    for name in sorted(paths):
        relative(name)
        source = root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError('missing/nonregular projection source: ' + name)
        target = projected / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for row in manifest['source_projections']:
        target = projected / row['original_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row['archive_path'], target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--suite', choices=('synthetic', 'saved'), default='synthetic')
    p.add_argument('--only', action='append', default=[])
    p.add_argument('--fixtures-root', type=Path)
    p.add_argument('--report-dir', type=Path)
    p.add_argument('--list', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    verify(root, manifest)
    entries = json.loads((root / ALLOWLIST).read_text())
    selected = selected_checks(entries, args.suite, args.only)
    if args.list:
        print(json.dumps(selected, indent=2))
        return
    if args.suite == 'saved' and (args.fixtures_root is None or not args.fixtures_root.is_dir()):
        p.error('saved suite requires an explicitly prepared --fixtures-root; no implicit downloads')
    reports = args.report_dir.resolve() if args.report_dir else None
    if reports:
        reports.mkdir(parents=True, exist_ok=True)
    results = []
    clean_env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    clean_env['PYTHONDONTWRITEBYTECODE'] = '1'
    with tempfile.TemporaryDirectory(prefix='kws-offline-source-') as temp:
        projected = Path(temp) / 'repo'
        projected.mkdir()
        make_projection(root, projected, manifest)
        if args.fixtures_root:
            shutil.copytree(args.fixtures_root, Path(temp) / 'offline-fixtures')
        for row in selected:
            start = time.monotonic()
            try:
                run = subprocess.run(row['argv'], cwd=projected / row['cwd'], env=clean_env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                output = run.stdout
                code = run.returncode
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b''
                code = 124
            result = {'id': row['id'], 'status': 'PASS' if code == 0 else 'FAIL',
                      'returncode': code, 'seconds': round(time.monotonic() - start, 3),
                      'scope': row['scope']}
            results.append(result)
            if reports:
                (reports / (row['id'] + '.log')).write_bytes(output)
            print(json.dumps(result), flush=True)
            if code:
                print(output.decode('utf-8', errors='replace'), flush=True)
    verify(root, manifest)
    if reports:
        (reports / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    failed = sum(r['status'] != 'PASS' for r in results)
    print(f'{len(results) - failed}/{len(results)} explicit {args.suite} checks passed; no live acquisition')
    raise SystemExit(bool(failed))

if __name__ == '__main__':
    main()
