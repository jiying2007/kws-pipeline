"""Fixed, source-local zero-model authority; no network or inferred approval."""
from pathlib import Path
import hashlib
import json
import stat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = 'research/k1-evidence-transfer-qualification-v1/'
ASR = 'research/k1-punctuation6-asr18-v1/'
WORKFLOW = '.github/workflows/k1-evidence-transfer-qualification-v1.yml'
ROSTER = tuple(sorted((REL + 'qualification.py', REL + 'source_authority.py',
    REL + 'fixture-lock.json', REL + 'qualification-control.json', WORKFLOW,
    ASR + 'evidence_publisher.py', ASR + 'evidence_transport.py',
    ASR + 'campaign_contract.py', ASR + 'asr18_metadata.py', ASR + 'failure_observer.py',
    ASR + 'execution-release.json', ASR + 'publication-release.json')))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def build(root=ROOT):
    root = Path(root)
    require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
            'fixed unaliased source root')
    rows = {}
    for name in ROSTER:
        path = root / name
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'no source symlink')
        mode = path.stat().st_mode
        require(stat.S_ISREG(mode) and not mode & 0o111, 'ordinary fixed source file')
        raw = path.read_bytes()
        require(len(raw) <= 1024 ** 2, 'bounded source file')
        rows[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                     'mode': '100644'}
    return {'schema': 'k1-hosted-transfer-source-freeze-v1', 'files': rows,
            'execution_authority': 'fixed qualification control only; no model authority'}


if __name__ == '__main__':
    import sys
    require(sys.argv[1:] == [], 'fixed source freeze takes no arguments')
    sys.stdout.buffer.write(canonical(build()))
