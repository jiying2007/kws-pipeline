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

# A single reviewed source maintenance exception. Historical inventory/provenance
# stay byte-for-byte semantically fixed; current code and the original archive
# are checked separately. This is not a general path override facility.
CURRENT_REVISION_PATH = 'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py'
CURRENT_REVISION_ARCHIVE = 'research/consolidation/archive/source-revisions/prepare-r1-hotpath-20261012/qwen_loading_gate_v2.py'
CURRENT_REVISION_ID = 'prepare-r1-hotpath-20261012'
CURRENT_REVISION_SUPERSEDES = 'd6ca13151f98a8fc80c2fbb1f21832a9524191db'
ORIGINAL_SOURCE = {'path': 'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py', 'mode': '100644', 'bytes': 20790, 'git_blob_sha1': '73557f9608f338b406c4135b38ccb3c81dfb05d3', 'sha256': '1b0cbd7b96e24925bfcf72438edea9dfb286860d2a990cdbad48104ae595d3fb'}
ORIGINAL_FILES_SHA256 = '4aaccd87a86900a6cbc7301b078841713621544ed483e16213411574b986cd7a'
ORIGINAL_PROVENANCE_SHA256 = 'e494ec723ff1570b36698eccb424ffb767e533d6c4acd7e4ba6078c2f9da3e7a'
IDENTITY_FIELDS = frozenset(('path', 'mode', 'bytes', 'sha256', 'git_blob_sha1'))


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def validate_current_source_revision(root, manifest):
    revision = manifest.get('current_source_revision')
    if revision is None:
        if 'current_source_revision' in manifest:
            raise ValueError('current source revision must be a fixed object')
        return None
    fields = {'revision_id', 'path', 'archive_path', 'supersedes_commit',
              'original', 'original_provenance', 'current'}
    if not isinstance(revision, dict) or set(revision) != fields:
        raise ValueError('current source revision fields')
    for key, wanted in (('revision_id', CURRENT_REVISION_ID),
                        ('path', CURRENT_REVISION_PATH),
                        ('archive_path', CURRENT_REVISION_ARCHIVE),
                        ('supersedes_commit', CURRENT_REVISION_SUPERSEDES)):
        if revision[key] != wanted:
            raise ValueError('current source revision exceeds fixed scope: ' + key)
    if (canonical_sha256(manifest.get('files')) != ORIGINAL_FILES_SHA256 or
            canonical_sha256(manifest.get('provenance')) != ORIGINAL_PROVENANCE_SHA256):
        raise ValueError('original 718 inventory/provenance changed')
    original = revision['original']
    if original != ORIGINAL_SOURCE or type(original.get('bytes')) is not int:
        raise ValueError('current source revision original identity changed')
    retained = [row for row in manifest['files'] if row['path'] == CURRENT_REVISION_PATH]
    if retained != [original]:
        raise ValueError('current source revision does not bind complete original row')
    provenance = [row for row in manifest['provenance'] if row['path'] == CURRENT_REVISION_PATH]
    if revision['original_provenance'] != provenance or len(provenance) != 3:
        raise ValueError('current source revision original provenance changed')
    archive = dict(original, path=CURRENT_REVISION_ARCHIVE)
    check_file(root, archive)
    current = revision['current']
    if (not isinstance(current, dict) or set(current) != IDENTITY_FIELDS or
            current.get('path') != CURRENT_REVISION_PATH or
            current.get('mode') != original['mode'] or
            type(current.get('bytes')) is not int or current['bytes'] <= 0 or
            not isinstance(current.get('sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', current['sha256']) or
            not isinstance(current.get('git_blob_sha1'), str) or
            not re.fullmatch('[0-9a-f]{40}', current['git_blob_sha1'])):
        raise ValueError('current source revision exact identity fields')
    if current['sha256'] == original['sha256'] or current['git_blob_sha1'] == original['git_blob_sha1']:
        raise ValueError('current source revision must identify distinct derived bytes')
    check_file(root, current)
    return revision


def retained_source_path(manifest, name):
    """Fixed original-byte path for verification and historical test projection.

    Callers must first validate the complete manifest with verify().
    """
    if manifest.get('current_source_revision') is not None and name == CURRENT_REVISION_PATH:
        return CURRENT_REVISION_ARCHIVE
    return name

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
    validate_current_source_revision(root, manifest)
    by_path = {}
    for row in files:
        if row['path'] in by_path:
            raise ValueError('duplicate retained file')
        if not row['path'].startswith('research/'):
            raise ValueError('source import outside isolated research tree')
        check_file(root, dict(row, path=retained_source_path(manifest, row['path'])))
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
    approved = ['.github/workflows/research-source-consolidation.yml',
                '.github/workflows/qwen12-native-admission-v1.yml',
                '.github/workflows/k1-punctuation6-asr18-v1.yml',
                '.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml',
                '.github/workflows/k1-evidence-transfer-qualification-v1.yml',
                '.github/workflows/k1-atomic-claim-qualification-v1.yml',
                '.github/workflows/k1-punctuation6-asr18-r4.yml',
                '.github/workflows/k1-punctuation6-asr18-r5.yml']
    if additions != approved:
        raise ValueError('unexpected active workflow addition')
    # The original 54-workflow baseline and the separately allowed additions
    # both bind CURRENT bytes. Historical maintenance receipts are evidence,
    # not the source of the current identity and not an existence-only escape.
    current_additions = manifest.get('current_added_active_workflows')
    if not isinstance(current_additions, list) or len(current_additions) != len(approved):
        raise ValueError('missing or duplicate current added workflow identity')
    current_paths = []
    for row in current_additions:
        if not isinstance(row, dict) or row.get('path') not in approved:
            raise ValueError('unexpected current added workflow identity')
        check_file(root, row)
        current_paths.append(row['path'])
    if sorted(current_paths) != sorted(approved) or paths.intersection(current_paths):
        raise ValueError('current added workflow identities do not match allowance')
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
