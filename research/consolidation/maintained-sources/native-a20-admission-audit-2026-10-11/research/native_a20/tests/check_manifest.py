"""Offline exact source inventory, identity and transport/privacy checks."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
expected = manifest['files']
actual = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*')
          if p.is_file() and '__pycache__' not in p.parts and p.name != 'SOURCE_MANIFEST.json'}
assert set(actual) == set(expected), (set(actual)-set(expected), set(expected)-set(actual))
for name, path in actual.items():
    raw = path.read_bytes()
    assert len(raw) == expected[name]['bytes'], name
    assert hashlib.sha256(raw).hexdigest() == expected[name]['sha256'], name
    raw.decode('utf-8')
    # Public source must never accidentally contain executor/user delivery records.
    forbidden = [r'/work'+'space/', r'lib'+'rary_file_id', r'sedim'+'ent://',
                 r'Sent'+'inel_[a-f0-9]', r'cloud_browser_'+'handoff',
                 r'chatgpt\.com/backend-api/', r'BEGIN '+'PRIVATE KEY']
    for pattern in forbidden:
        assert not re.search(pattern, raw.decode('utf-8')), (name, pattern)
provenance = json.loads((ROOT/'PROVENANCE.json').read_text())
for row in provenance['preserved_native_sources']:
    assert expected[row['path']]['sha256'] == row['source_sha256'], row['path']
assert provenance['shipping_files_changed'] is False
assert provenance['native_math_or_decoder_changed'] is False
identity = (ROOT/'baseline/export/a20_identity.h').read_text()
assert 'a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d' in identity
print('PASS exact public source inventory, native identity and transport/privacy checks')
