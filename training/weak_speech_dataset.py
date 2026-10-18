#!/usr/bin/env python3
"""Research-only weak dataset using the current C-compatible training frontend.

No optimizer, product launcher, qualification admission, or model export. Returns
(feature matrix, numeric CTC target IDs), the same pair used by train_ctc.Manifest.
Callers may tensorize these pairs in an explicitly research-only experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from corpus_identity import inspect_pcm16_wav
from frontend_spec import features_pcm16


def load_examples(package: Path, *, feature_dim=32, frontend='logmel'):
    package = package.resolve(strict=True)
    manifest = json.loads((package / 'MANIFEST.json').read_bytes())
    for name, digest in manifest.items():
        path = (package / name).resolve(strict=True)
        if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('result package checksum mismatch')
    if 'weak-train.jsonl' not in manifest or 'report.json' not in manifest:
        raise ValueError('missing weak manifest identity')
    report = json.loads((package / 'report.json').read_bytes())
    admitted = {r['item_id']: r for r in report['items'] if r['status'] == 'machine_candidate'}
    rows = [json.loads(line) for line in (package / 'weak-train.jsonl').read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if (row['label_source'] != 'weak-machine-agreement' or row['purpose'] != 'noncommercial-research-only'
                or row['human_gold'] is not False or row['product_qualified'] is not False or row['split'] != 'train'):
            raise ValueError('research weak scope required')
        expected = admitted[row['source_id']]
        if row['source_id'] in seen:
            raise ValueError('duplicate weak example')
        seen.add(row['source_id'])
        for field, original in [('actual_text', 'machine_text'), ('target_ids', 'target_ids'),
                                ('file_sha256', 'file_sha256'), ('pcm_sha256', 'pcm_sha256'),
                                ('speaker_id', 'lineage_group'), ('split', 'split'), ('kind', 'actual_kind'), ('tokens', 'tokens')]:
            if row[field] != expected[original]:
                raise ValueError('weak example differs from screened result')
        # Conditioned metadata is label provenance, not optional decoration.
        protocol = 'source-language-conditioned-asr-v1'
        if protocol in expected.get('asr_protocols', []) or protocol in row.get('asr_protocols', []):
            for field in ('asr_protocols', 'source_language_conditions', 'detected_language',
                          'language_is_model_independent_observation', 'intended_text'):
                if field not in row or field not in expected or row[field] != expected[field]:
                    raise ValueError('conditioned weak provenance differs from screened result: ' + field)
            if (row['detected_language'] is not None
                    or row['language_is_model_independent_observation'] is not False):
                raise ValueError('source-language conditioning is not independent language detection')
        path = (package / row['audio']).resolve(strict=True)
        if row['audio'] not in manifest or not path.is_relative_to(package):
            raise ValueError('audio not retained in package')
        measured = inspect_pcm16_wav(path)
        if any(row[k] != measured[k] for k in ('file_sha256', 'pcm_sha256')):
            raise ValueError('weak audio identity mismatch')
        targets = row['target_ids']
        if not targets or any(type(t) is not int or t <= 0 for t in targets):
            raise ValueError('nonblank CTC targets required')
        with wave.open(str(path), 'rb') as reader:
            raw = reader.readframes(reader.getnframes())
        samples = list(struct.unpack('<' + 'h' * (len(raw) // 2), raw))
        acoustic = features_pcm16(samples, feature_dim=feature_dim, frontend=frontend)
        minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
        if len(acoustic) < minimum:
            raise ValueError('insufficient CTC frames')
        yield acoustic, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    examples = list(load_examples(args.package))
    print(json.dumps({'examples': len(examples), 'feature_frames': sum(len(x) for x, _ in examples),
                      'model_calls': 0, 'training_runs': 0, 'scope': 'research-weak-loader-only'}))


if __name__ == '__main__':
    main()
