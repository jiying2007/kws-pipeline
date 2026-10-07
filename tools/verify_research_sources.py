#!/usr/bin/env python3
"""Verify public-source retention and keep historical workflows inactive."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

MANIFEST = 'research/consolidation/source-retention-2026-10-07.json'
ARM = 'research/cosyvoice3_cross_voice/arm.json'
DISABLED_ARM = {'mode': 'disabled', 'reviewed_parent_sha': None,
                'payload_sha256': None, 'batch_id': 'cosy30-cross-voice-20261004-v1'}

def relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('noncanonical relative path')
    return name

def check_file(root, row):
    name = relative(row['path'])
    path = root / name
    if any((root / Path(*Path(name).parts[:n])).is_symlink()
           for n in range(1, len(Path(name).parts) + 1)):
        raise ValueError('symlink in retained source path: ' + name)
    if not path.is_file():
        raise ValueError('missing retained file: ' + name)
    if row['mode'] not in ('100644', '100755'):
        raise ValueError('unsupported mode: ' + name)
    if os.name == 'posix':
        actual = '100755' if path.stat().st_mode & 0o111 else '100644'
        if actual != row['mode']:
            raise ValueError('retained file mode changed: ' + name)
    b = path.read_bytes()
    if type(row['bytes']) is not int or len(b) != row['bytes']:
        raise ValueError('retained byte count changed: ' + name)
    blob = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
    if blob != row['git_blob_sha1'] or hashlib.sha256(b).hexdigest() != row['sha256']:
        raise ValueError('retained source identity changed: ' + name)

def verify(root, manifest):
    if manifest.get('schema') != 'research-source-retention-v1':
        raise ValueError('unsupported source retention schema')
    if manifest.get('repository') != 'jiying2007/kws-pipeline':
        raise ValueError('unexpected source repository')
    if not re.fullmatch('[0-9a-f]{40}', manifest.get('base_commit', '')):
        raise ValueError('invalid base commit')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('retained files must be nonempty')
    by_path = {}
    for row in files:
        if row['path'] in by_path:
            raise ValueError('duplicate retained file')
        if not row['path'].startswith('research/'):
            raise ValueError('source import outside isolated research tree')
        check_file(root, row)
        by_path[row['path']] = row
    mapped = set()
    source_ids = set()
    for row in manifest.get('provenance', []):
        if not re.fullmatch('[0-9a-f]{40}', row.get('source_commit', '')):
            raise ValueError('invalid source commit')
        key = (row['source_commit'], relative(row['original_path']), row['path'])
        if key in source_ids:
            raise ValueError('duplicate source provenance')
        source_ids.add(key)
        expected = by_path.get(row['path'])
        if expected is None or any(row[k] != expected[k]
                for k in ('mode', 'bytes', 'sha256', 'git_blob_sha1')):
            raise ValueError('source provenance does not bind retained bytes')
        wanted = 'https://github.com/jiying2007/kws-pipeline/blob/' + row['source_commit'] + '/' + row['original_path']
        if row.get('source_url') != wanted:
            raise ValueError('source URL does not bind declared identity')
        mapped.add(row['path'])
    if mapped != set(by_path):
        raise ValueError('retained file lacks provenance')
    projection_paths = set()
    for row in manifest.get('source_projections', []):
        original = relative(row['original_path'])
        if original in projection_paths:
            raise ValueError('duplicate projection path')
        projection_paths.add(original)
        if not (original.startswith('.github/workflows/') or original == ARM):
            raise ValueError('projection exceeds archived workflow/arm scope')
        archive = relative(row['archive_path'])
        retained = by_path.get(archive)
        if not archive.startswith('research/consolidation/archive/') or retained is None:
            raise ValueError('projection is not backed by retained archive')
        if any(row[k] != retained[k] for k in ('bytes', 'sha256', 'git_blob_sha1')):
            raise ValueError('projection identity changed')
        if not any(p['source_commit'] == row['source_commit'] and p['original_path'] == original
                   and p['path'] == archive for p in manifest['provenance']):
            raise ValueError('projection lacks original-path provenance')
        if original != ARM and (root / original).exists():
            raise ValueError('historical workflow is active: ' + original)
    baseline = manifest.get('baseline_active_workflows', [])
    paths = set()
    for row in baseline:
        if not row['path'].startswith('.github/workflows/') or row['path'] in paths:
            raise ValueError('invalid baseline workflow inventory')
        check_file(root, row)
        paths.add(row['path'])
    additions = manifest.get('allowed_new_active_workflows')
    if additions != ['.github/workflows/research-source-consolidation.yml']:
        raise ValueError('unexpected active workflow addition')
    actual = {str(p.relative_to(root)) for p in (root / '.github/workflows').glob('*') if p.is_file()}
    if actual != paths | set(additions):
        raise ValueError('active workflow inventory changed')
    for row in manifest.get('pointer_only', []):
        if row.get('path') is not None or row.get('kind') != 'source-pointer-only':
            raise ValueError('invalid pointer-only disposition')
        original = relative(row['original_path'])
        if (root / original).exists():
            raise ValueError('pointer-only historical audio unexpectedly copied')
    if json.loads((root / ARM).read_text()) != DISABLED_ARM:
        raise ValueError('Cosy30 source projection is not disarmed')
    # Original freeze identities are checked through the recorded relocation map.
    relocated = {p['original_path']: p['archive_path'] for p in manifest['source_projections']}
    for name in ('research/token_preparation/SOURCE-FREEZE.json',
                 'research/fixed300/SOURCE-FREEZE.json', 'research/cosy49/SOURCE-FREEZE.json'):
        if name not in by_path:
            continue
        frozen = json.loads((root / name).read_text())
        seen_frozen = set()
        for row in frozen['files']:
            original = relative(row['path'])
            if original in seen_frozen:
                raise ValueError('duplicate frozen source path')
            seen_frozen.add(original)
            retained = by_path.get(relocated.get(original, original))
            if retained is None or any(row[k] != retained[k] for k in ('bytes', 'sha256')):
                raise ValueError('original source-freeze identity changed: ' + original)
    cosy = [p for p in manifest['provenance']
            if p.get('source_branch') == 'research/a20-cosy30-cross-voice-v1']
    if cosy:
        values = {p['original_path']: {'sha256': p['sha256'], 'bytes': p['bytes']}
                  for p in cosy if p['original_path'] != ARM}
        identity = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=False).encode()).hexdigest()
        original_arm = json.loads((root / relocated[ARM]).read_text())
        if identity != original_arm['payload_sha256']:
            raise ValueError('historical Cosy30 reviewed payload identity changed')
    return len(by_path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root.resolve()
    n = verify(root, json.loads((root / MANIFEST).read_text()))
    print(f'PASS: {n} exact retained files; historical workflows inactive; Cosy30 disarmed')

if __name__ == '__main__':
    main()
