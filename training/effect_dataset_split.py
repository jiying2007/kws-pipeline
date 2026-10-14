#!/usr/bin/env python3
"""Offline group split planning. Outputs are NOT training-admission receipts.

Use audited JSONL rows with explicit label evidence. Never launches ASR/TTS or
training; existing training_admission remains the authority for consumption.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'eval')]
from audit_dataset import inspect_wav, sha256_file
from effect_evidence import normalize, separation

SOURCE_ID_FIELDS = ('source_id', 'source_family', 'source_family_id', 'family_id',
                    'derivation_family', 'derivation_family_id')
PCM_FIELDS = ('pcm_sha256', 'source_pcm', 'source_pcm_sha256', 'original_pcm_sha256', 'reference_pcm_sha256')
AUDIO_ID_FIELDS = ('file_sha256', 'source_wav_sha256', 'source_audio_sha256',
                   'original_audio_sha256', 'reference_audio_sha256')
GROUP_FIELDS = ('speaker_id', 'session_id', *SOURCE_ID_FIELDS, *AUDIO_ID_FIELDS)
REQUIRED = ('speaker_id', 'session_id', 'source_family_id')
SPLITS = ('train', 'development', 'sealed_acceptance', 'quarantine')
UNKNOWN = frozenset(('', 'unknown', 'none', 'null', 'pending', 'n/a', 'na',
                     'unset', 'unspecified', 'missing', 'tbd', '-', '?'))


def known(value):
    """Identity-only predicate, aligned with model_comparison.compare.identity."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return normalized not in UNKNOWN and not any(
        normalized.startswith(word + separator) for word in UNKNOWN if len(word) > 1
        for separator in ('_', '-', ':', ' '))


def nonempty_text(value):
    # Natural-language observations must not be interpreted as identity tokens.
    return isinstance(value, str) and bool(value.strip())


SPLIT_ALIASES = {
    'train': 'train', 'training': 'train',
    'development': 'development', 'dev': 'development',
    'selection': 'development', 'calibration': 'development',
    'validation': 'development', 'val': 'development',
    'test': 'sealed_acceptance', 'qualification': 'sealed_acceptance',
    'sealed_acceptance': 'sealed_acceptance', 'quarantine': 'quarantine',
}


def historical_partitions(row):
    """All retained split history is binding; unknown roles fail closed."""
    roles = set()
    for field in ('split', 'planned_split', 'previous_split'):
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or value.strip().lower() not in SPLIT_ALIASES:
            raise ValueError(f'unsupported historical {field}: {value!r}')
        roles.add(SPLIT_ALIASES[value.strip().lower()])
    if row.get('development_only') is True or row.get('previously_used_for_tuning') is True:
        roles.add('development')
    if row.get('previously_used_for_training') is True:
        roles.add('train')
    return roles


def quality_tier(row):
    """Evidence categories, not inferred correctness or admission authority."""
    review = row.get('human_review')
    if isinstance(review, dict) and review.get('status') == 'complete' and all(
            known(review.get(k)) for k in ('reviewer_id', 'listened_at_utc')) and nonempty_text(review.get('heard_text')) and (
            review.get('asr_or_target_seen_before_listening') is False):
        return 'human_review_recorded'
    evidence = row.get('asr_evidence', [])
    if not isinstance(evidence, list):
        raise ValueError('asr_evidence must be a list')
    complete = [x for x in evidence if x.get('status') == 'complete' and isinstance(x.get('raw_text'), str)]
    families = [x.get('family') for x in complete]
    if len(families) != len(set(families)) or any(not known(x) for x in families):
        raise ValueError('ASR evidence requires distinct named families')
    texts = [normalize(x['raw_text']) for x in complete]
    if len(set(texts)) > 1:
        return 'machine_disagreement'
    if len(complete) >= 3 and len(complete) == len(evidence) and texts[0]:
        return 'machine_consensus'
    return 'unreviewed_or_incomplete'


def plan(rows, seed='effect-split-v1'):
    rows = [dict(row) for row in rows]
    ids = [row.get('record_id') for row in rows]
    if any(not known(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique non-empty record_id required')
    parent = list(range(len(rows)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        row['quality_tier'] = quality_tier(row)
        identities = []
        for field in (*GROUP_FIELDS, *PCM_FIELDS):
            value = row.get(field)
            if not known(value):
                continue
            value = value.strip()
            if field in (*PCM_FIELDS, *AUDIO_ID_FIELDS) and (
                    len(value) != 64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError(f'invalid audio identity digest: {field}')
            namespace = ('pcm' if field in PCM_FIELDS else 'audio-file' if field in AUDIO_ID_FIELDS
                         else 'source' if field in SOURCE_ID_FIELDS else field)
            identities.append((namespace, value))
        # Path linkage also protects derivatives when source PCM is unavailable.
        identities += [('path:' + row.get('audio_root_id', ''), row[field])
                       for field in ('path', 'source_path') if known(row.get(field))]
        for identity in identities:
            if identity in owners:
                parent[root(i)] = root(owners[identity])
            else:
                owners[identity] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    results = []
    for members in groups.values():
        group_id = hashlib.sha256('\n'.join(sorted(r['record_id'] for r in members)).encode()).hexdigest()
        missing = any(not all(known(r.get(f)) for f in REQUIRED) for r in members)
        unusable = any(r.get('audio_status') != 'verified_pcm16' for r in members)
        tiers = {r['quality_tier'] for r in members}
        historical = set().union(*(historical_partitions(row) for row in members))
        if len(historical) > 1:
            raise ValueError('conflicting historical partitions share a source component: ' +
                             ', '.join(sorted(historical)))
        if historical:
            split = next(iter(historical))
            reason = 'preserved_historical_partition_never_randomized'
        elif unusable or missing or tiers & {'machine_disagreement', 'unreviewed_or_incomplete'}:
            split, reason = 'quarantine', 'missing_audio_identity_or_label_evidence'
        elif 'machine_consensus' in tiers:
            split, reason = 'train', 'machine_consensus_training_candidate_only'
        else:
            bucket = int(hashlib.sha256((seed + group_id).encode()).hexdigest()[:8], 16) % 100
            split = 'train' if bucket < 70 else 'development' if bucket < 85 else 'sealed_acceptance'
            reason = 'group_hash_partition_reviewed_candidate'
        for row in members:
            if 'split' in row and row['split'] != split:
                row['previous_split'] = row['split']
            if row.get('human_gold') is True:
                row['input_human_gold_claim'] = True
            row.update(group_id=group_id, planned_split=split, split=split, split_reason=reason,
                       human_gold=False, training_admitted=False, product_qualification_allowed=False)
            results.append(row)
    results.sort(key=lambda r: r['record_id'])
    return {'schema': 'effect-dataset-split-plan-v1', 'seed': seed, 'rows': results,
            'groups': len(groups), 'counts': dict(Counter(r['planned_split'] for r in results)),
            'quality_counts': dict(Counter(r['quality_tier'] for r in results)),
            'missing_metadata': {f: sum(not known(r.get(f)) for r in rows) for f in REQUIRED},
            'separation': {a + ':' + b: separation([r for r in results if r['planned_split'] == a],
                                                      [r for r in results if r['planned_split'] == b])
                           for a, b in (('train', 'development'), ('train', 'sealed_acceptance'),
                                        ('development', 'sealed_acceptance'))},
            'training_admitted': False, 'product_qualification_allowed': False,
            'note': 'A planned sealed partition is not qualification evidence. Existing admission, live-byte audit and independent evaluation are still required.'}


def write_outputs(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    for split in SPLITS:
        with (output / (split + '.jsonl')).open('w') as stream:
            for row in report['rows']:
                if row['planned_split'] == split:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audio-root', type=Path)
    p.add_argument('--root', action='append', default=[], help='logical audio root NAME=PATH')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', default='effect-split-v1')
    args = p.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    roots = {}
    for value in args.root:
        name, raw = value.split('=', 1)
        if name in roots or not name or not raw:
            raise ValueError('unique NAME=PATH roots required')
        roots[name] = Path(raw)
    for row in rows:
        from corpus_identity import resolve_audio_path
        root_id = row.get('audio_root_id')
        if root_id and root_id not in roots:
            raise ValueError(f'missing --root mapping for {root_id}')
        audio_root = roots[root_id] if root_id else args.audio_root or args.manifest.parent
        path = resolve_audio_path(row, audio_root)
        frames, duration, pcm = inspect_wav(path)
        wav = sha256_file(path)
        for key, actual in [('pcm_sha256', pcm), ('file_sha256', wav), ('wav_sha256', wav)]:
            if key in row and row[key] != actual:
                raise ValueError(f'{row.get("record_id")}: {key} mismatch')
        row.update(pcm_sha256=pcm, file_sha256=wav, frames=frames, duration_s=duration,
                   audio_status='verified_pcm16')
    write_outputs(plan(rows, args.seed), args.output)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
