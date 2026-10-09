"""Complete real-file audit fixture. No mocking, model, audio or historical data."""
import hashlib
import json
from pathlib import Path
import numpy as np


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def trace(helper, counts):
    rows = sum(counts)
    data = {'input': np.zeros((rows, 400), np.float32), 'counts': np.array(counts, np.int64)}
    for i, n in enumerate(counts):
        data.update({f'cmvn{i}': np.zeros((n, 400), np.float32),
                     f'logits{i}': np.zeros((n, 6), np.float32),
                     f'cache{i}': np.zeros((1, 128, 11, 4), np.float32)})
        data.update({f'stage{i}_{s}': np.zeros((n, d), np.float32) for s, d in enumerate(helper.DIMS)})
    return data


def build(root, helper):
    root = Path(root)
    base = root / 'review'
    attempt = base / 'attempt'
    arm = attempt / 'D20'
    arm.mkdir(parents=True)
    (root / 'synthetic-source.txt').write_text('public synthetic source\n')
    deps = root / 'reference-dependency-recovery-20261008/INSTALLED-BYTE-AUDIT.json'
    write(deps, {'synthetic': True})
    files = []
    for path in (root / 'synthetic-source.txt', deps):
        files.append({'path': str(path.relative_to(root)), 'sha256': helper.sha(path), 'bytes': path.stat().st_size})
    write(base / 'SOURCE-FREEZE.json', {'files': files})
    write(base / 'PARENT-RELEASE.json', {'source_freeze_sha256': helper.sha(base / 'SOURCE-FREEZE.json')})
    write(attempt / 'STARTED.json', {'release_sha256': helper.sha(base / 'PARENT-RELEASE.json')})
    records = []
    features = []
    for n in helper.LENGTHS:
        features.append({'rows': n, 'sha256': hashlib.sha256(np.zeros((n, 400), np.float32).tobytes()).hexdigest()})
        ragged = []
        remaining = n
        while remaining:
            count = min(remaining, (1, 3, 7, 11)[len(ragged) % 4])
            ragged.append(count)
            remaining -= count
        for part, counts in (('whole', [n]), ('ragged', ragged)):
            name = f'features{n}-{part}'
            path = arm / (name + '.npz')
            np.savez(path, **trace(helper, counts))
            records.append({'name': name, 'path': path.name, 'rows': n, 'sha256': helper.sha(path)})
    write(base / 'FIXTURES.json', {'features': features})
    pcm = {}
    geometry = {}
    for n, rows in helper.PCM.items():
        name = f'pcm{n}'
        path = arm / (name + '.npz')
        np.savez(path, **trace(helper, [rows] if rows else []))
        records.append({'name': name, 'path': path.name, 'rows': rows, 'sha256': helper.sha(path)})
        geometry[str(n)] = [rows] if rows else []
        if rows:
            pcm[f'rows_{n}_0'] = np.zeros((rows, 400), np.float32)
    np.savez(attempt / 'official-pcm.npz', **pcm)
    pcm_hash = helper.sha(attempt / 'official-pcm.npz')
    write(attempt / 'official-pcm.json', {'sha256': pcm_hash, 'geometry': geometry})
    write(arm / 'REFERENCE-FROZEN.json', {'records': records, 'torch_rows': 479, 'official_pcm_sha256': pcm_hash})
    report = helper.audit(root, attempt)
    ticket = helper.ticket(report)
    reference = trace(helper, [1])
    identity = {
        'arm': 'D20', 'fixture': 'features1-whole',
        'input_sha256': hashlib.sha256(reference['input'].tobytes()).hexdigest(),
        'source_sha256': helper.sha(base / 'SOURCE-FREEZE.json'),
        'dependency_sha256': helper.sha(deps),
        'reference_sha256': helper.trace_digest(reference),
        'admission_ticket_sha256': helper.digest(ticket), 'gate_profile': 'same-input',
    }
    return attempt, reference, identity, ticket
