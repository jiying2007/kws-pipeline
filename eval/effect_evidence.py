#!/usr/bin/env python3
"""Fixed-denominator, non-gold diagnostic summaries. No recognizer invocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

PUNCTUATION = frozenset('。，！？、；：…,.!?;: \t\n\r')
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
CELLS = tuple(f'k1p6-{number:03d}' for number in range(1, 7))
DEFAULT_POLICY = {'policy_id': 'xiaowo-exact-diagnostic-v1', 'target': '你好小窝', 'accepted_alternatives': []}
LINEAGE = ('file_sha256', 'pcm_sha256', 'source_family_id', 'speaker_id', 'session_id', 'derivation_family_id', 'reference_audio_sha256')


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('transcript must be text')
    return ''.join(char for char in text if char not in PUNCTUATION)


def summarize(result, policy=None):
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    if (not isinstance(policy.get('policy_id'), str) or not policy['policy_id'] or
            not isinstance(policy.get('target'), str) or not policy['target'] or
            not isinstance(policy.get('accepted_alternatives'), list) or
            any(not isinstance(s, str) or not s or normalize(s) != s for s in policy['accepted_alternatives'])):
        raise ValueError('explicit canonical near-sound policy required')
    accepted = {policy['target'], *policy['accepted_alternatives']}
    slots = {}; identities = {}; sources = {}
    for row in result['rows']:
        key = (row['cell_id'], row['family'])
        if key[0] not in CELLS or key[1] not in FAMILIES or key in slots:
            raise ValueError('unexpected or duplicate fixed-campaign slot')
        identity = {field: row[field] for field in ('audio_id', 'wav_sha256', 'pcm_sha256', 'audio_frames')}
        if any(re.fullmatch('[0-9a-f]{64}', identity[field]) is None for field in ('wav_sha256', 'pcm_sha256')):
            raise ValueError('missing audio byte identity')
        if key[0] in identities and identities[key[0]] != identity:
            raise ValueError('ASR families disagree on input audio identity')
        identities[key[0]] = identity
        if row['status'] == 'complete':
            transcript = normalize(row['raw_transcript'])
            if transcript != row['normalized_transcript']:
                raise ValueError('normalization changed beyond punctuation')
            slots[key] = transcript
        else:
            slots[key] = None
        sources[key] = row['status']
    per_cell = []
    for cell in CELLS:
        texts = [slots.get((cell, family)) for family in FAMILIES]
        complete = all(text is not None for text in texts)
        consensus = texts[0] if complete and len(set(texts)) == 1 else None
        per_cell.append({'cell_id': cell, 'audio_identity': identities.get(cell), 'three_complete': complete,
                         'machine_consensus': consensus, 'target_consensus': consensus == policy['target'],
                         'policy_accepted_consensus': consensus in accepted, 'human_gold': False,
                         'training_admitted': False})
    return {'schema': 'fixed-six-asr-diagnostic-v1', 'policy': policy, 'denominator': 6,
            'expected_slots': 18, 'complete_slots': sum(value is not None for value in slots.values()),
            'target_consensus_count': sum(row['target_consensus'] for row in per_cell),
            'policy_accepted_consensus_count': sum(row['policy_accepted_consensus'] for row in per_cell),
            'other_consensus_count': sum(row['machine_consensus'] is not None and not row['target_consensus'] for row in per_cell),
            'disagreement_or_incomplete_count': sum(row['machine_consensus'] is None for row in per_cell),
            'cells': per_cell, 'human_gold': False, 'training_admitted': False,
            'product_qualification_allowed': False}


def listening_template(summary):
    # Deliberately exclude intended text, machine outputs and source-family names.
    return {'schema': 'independent-listening-template-v1', 'denominator': 6,
            'instructions': 'Listen independently before viewing ASR or target text. Record what is heard; uncertain is valid. No blank or target is inferred.',
            'records': [{'blind_id': f'listen-{i:03d}', 'audio_identity': row['audio_identity'],
                         'reviewer_id': None, 'listened_at_utc': None, 'asr_or_target_seen_before_listening': None,
                         'status': 'pending', 'heard_text': None, 'phonetic_notes': None,
                         'acoustic_complete': None, 'uncertainty_reason': None,
                         'human_gold': False, 'training_admitted': False}
                        for i, row in enumerate(summary['cells'], 1)]}


def separation(left, right):
    """Missing group metadata stays unknown even when every byte hash differs."""
    output = {}
    for field in LINEAGE:
        def known(row):
            value = row.get(field)
            return isinstance(value, str) and bool(value.strip()) and value.lower() not in {'unknown', 'none', 'null', 'n/a'}
        a, b = {r[field] for r in left if known(r)}, {r[field] for r in right if known(r)}
        overlap = a & b
        complete = bool(left and right) and all(known(r) for r in left + right)
        output[field] = {'status': 'overlap' if overlap else ('verified_disjoint' if complete else 'unknown'),
                         'overlap_count': len(overlap), 'left_known': sum(known(r) for r in left),
                         'right_known': sum(known(r) for r in right)}
    output['fully_disjoint'] = all(row['status'] == 'verified_disjoint' for row in output.values())
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--posterior', type=Path, required=True)
    p.add_argument('--policy', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--listening-template', type=Path, required=True)
    args = p.parse_args()
    data = args.posterior.read_bytes()
    summary = summarize(json.loads(data), json.loads(args.policy.read_text()) if args.policy else None)
    summary['posterior_sha256'] = hashlib.sha256(data).hexdigest()
    for path, value in ((args.output, summary), (args.listening_template, listening_template(summary))):
        if path.exists():
            raise ValueError('fresh output required; never overwrite completed listening records')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
