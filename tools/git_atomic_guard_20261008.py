#!/usr/bin/python3
"""Pinned private pre-push hook: every intended old/new ref must be outgoing."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ZERO = '0' * 40
ANCHOR = 'refs/tags/archive/branches-2026-10-07-prune-anchor'
LOCAL = 'refs/transfer/archive-target'

def validate_rows(raw, binding, remote_name, remote_url):
    if remote_name != binding['remote'] or remote_url != binding['remote']:
        raise ValueError('remote mismatch')
    if binding['anchor'] != ANCHOR or binding['local_ref'] != LOCAL:
        raise ValueError('anchor binding')
    if not re.fullmatch('[0-9a-f]{40}', binding['archive_oid']) or binding['archive_oid'] == ZERO:
        raise ValueError('archive OID')
    expected = {(LOCAL, binding['archive_oid'], ANCHOR, ZERO)}
    names = set()
    for row in binding['delete_refs']:
        name, oid = row['name'], row['before_oid']
        if not name.startswith('refs/heads/') or name == 'refs/heads/main' or name in names:
            raise ValueError('invalid deletion ref')
        if not re.fullmatch('[0-9a-f]{40}', oid) or oid == ZERO:
            raise ValueError('invalid deletion OID')
        names.add(name)
        expected.add(('(delete)', ZERO, name, oid))
    if len(raw) > 32768 or not raw.endswith(b'\n'):
        raise ValueError('input framing or size')
    rows = []
    for line in raw.decode('ascii').splitlines(keepends=True):
        if not line.endswith('\n') or any(ord(x) < 32 or ord(x) == 127 for x in line[:-1]):
            raise ValueError('control character')
        row = tuple(line[:-1].split(' '))
        if len(row) != 4 or any(not field for field in row):
            raise ValueError('row framing')
        rows.append(row)
    if len(rows) != len(expected) or len(set(rows)) != len(rows) or set(rows) != expected:
        raise ValueError('outgoing set mismatch')
    return {'outgoing_rows': len(rows), 'anchor_old_oid': ZERO}

def main():
    raw = Path(os.environ['KWS_GIT_BINDING']).read_bytes()
    if len(raw) > 32768 or hashlib.sha256(raw).hexdigest() != os.environ['KWS_GIT_BINDING_SHA256']:
        raise ValueError('binding mismatch')
    binding = json.loads(raw)
    if len(sys.argv) != 3:
        raise ValueError('hook arguments')
    result = validate_rows(sys.stdin.buffer.read(32769), binding, sys.argv[1], sys.argv[2])
    # This receipt supplements installation validation, never replaces it.
    Path(os.environ['KWS_GIT_HOOK_RECEIPT']).write_text(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Pinned pre-push guard rejected outgoing ref set', file=sys.stderr)
        sys.exit(1)
