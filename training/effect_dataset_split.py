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
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'training'), str(ROOT / 'eval')]
from audit_dataset import inspect_wav, sha256_file
from effect_evidence import normalize, separation
from source_identity import SOURCE_ID_FIELDS, source_identity_namespace

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
                         else source_identity_namespace(field))
            identities.append((namespace, value))
        # Preset voices are conservative leakage links, never real speaker IDs.
        # Do not namespace by engine: two engines may share the same preset.
        if known(row.get('preset_voice_id')):
            identities.append(('preset-voice', row['preset_voice_id'].strip()))
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



# This is an explicit, offline eligibility audit, not a new admission authority.
EVALUATION_ROLES = ('calibration', 'selection', 'sealed_test')
ROLE_ALIASES = {**SPLIT_ALIASES, 'calibration': 'calibration',
                'selection': 'selection', 'test': 'sealed_test',
                'qualification': 'sealed_test', 'sealed_acceptance': 'sealed_test',
                'sealed_test': 'sealed_test', 'screening': 'screening',
                'target_informed_selection': 'target_informed_selection'}


def check_role_eligibility(rows, required_roles=EVALUATION_ROLES):
    """Audit explicit roles without allocating or rewriting any input partition.

    New evaluation candidates require evaluation_role, history_complete=True,
    exposure_history=[] (or recorded prior uses), history_evidence_id, and
    source_kind='real'. Synthetic preset voices cannot establish real speakers.
    Exposure events are objects with a known ``role``; extra evidence is retained.
    Existing split aliases remain binding. A broad development history never
    proves independent calibration/selection. Supply the complete support set,
    including training/exposed ancestors, together: absent records cannot be
    audited. No boolean here certifies live bytes, gold labels or admission.
    """
    import copy
    sys.path.insert(0, str(ROOT / 'tools'))
    from speech_asr_consensus import has_machine_consensus

    def machine_candidate(value):
        if isinstance(value, list):
            return any(machine_candidate(item) for item in value)
        if not isinstance(value, dict):
            return False
        return (has_machine_consensus(value) or
                any(value.get(key) is True for key in
                    ('machine_train_only_candidate', 'three_model_training_candidate')) or
                value.get('label_origin') in ('machine-consensus', 'machine_consensus') or
                value.get('label_source') == 'three_asr_machine_consensus' or
                value.get('tier') in ('machine_three_asr_training_candidate', 'machine_high_confidence_candidate') or
                any(machine_candidate(item) for item in value.values()))

    def metadata_blockers(value):
        # JSON booleans must not be silently confused with integers or strings.
        # Validate nested provenance too; an outer tag cannot clear old evidence.
        boolean_fields = ('development_only', 'previously_used_for_tuning',
                          'previously_used_for_training', 'previously_exposed',
                          'test_results_seen', 'history_complete',
                          'machine_train_only_candidate', 'three_model_training_candidate',
                          'asr_or_target_seen_before_listening', 'human_gold')
        label_values = {
            'label_origin': ('human-review', 'human_review', 'human',
                             'machine-consensus', 'machine_consensus'),
            'label_source': ('human-review', 'human_review', 'independent-human-review',
                             'three_asr_machine_consensus', 'machine-three-asr-consensus-v1'),
            'tier': ('human_review_recorded',),
            'quality_tier': ('human_review_recorded',),
        }
        reasons = set()
        if isinstance(value, list):
            for item in value:
                reasons.update(metadata_blockers(item))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in boolean_fields and type(item) is not bool:
                    reasons.add('invalid_boolean_metadata:' + key)
                if key in ('previously_used_for_training', 'previously_used_for_tuning',
                           'development_only', 'previously_exposed', 'test_results_seen') and item is True:
                    reasons.add('retained_exposure_flag:' + key)
                if key == 'history_complete' and item is False:
                    reasons.add('retained_incomplete_history')
                if key in label_values and item not in label_values[key]:
                    reasons.add('unknown_or_unsupported_label_metadata:' + key)
                reasons.update(metadata_blockers(item))
        return reasons

    rows = copy.deepcopy(list(rows))
    required_roles = tuple(required_roles)
    if not required_roles or len(set(required_roles)) != len(required_roles) or any(
            role not in EVALUATION_ROLES for role in required_roles):
        raise ValueError('required_roles must be distinct explicit evaluation roles')
    # Reuse all existing identity validation and transitive lineage grouping.
    # Historical conflicts continue to raise rather than being silently repaired.
    grouping = plan(rows)
    by_id = {row['record_id']: row for row in rows}
    groups = defaultdict(list)
    for planned in grouping['rows']:
        groups[planned['group_id']].append(by_id[planned['record_id']])
    results, support = [], []
    for group_id, members in sorted(groups.items()):
        history, assigned, blockers = set(), set(), set()
        for row in members:
            blockers.update(metadata_blockers(row))
            row_history = set()
            for field in ('split', 'planned_split', 'previous_split'):
                if row.get(field) is not None:
                    value = row[field].strip().lower()
                    row_history.add(ROLE_ALIASES[value])
            if row.get('development_only') is True or row.get('previously_used_for_tuning') is True:
                row_history.add('development')
            if row.get('previously_used_for_training') is True:
                row_history.add('train')
            events = row.get('exposure_history')
            if row.get('history_complete') is not True or not isinstance(events, list):
                blockers.add('missing_complete_exposure_history')
                events = []
            for event in events:
                role = event.get('role') if isinstance(event, dict) else None
                if not isinstance(role, str) or role.strip().lower() not in ROLE_ALIASES:
                    blockers.add('unknown_exposure_role')
                else:
                    row_history.add(ROLE_ALIASES[role.strip().lower()])
            role = row.get('evaluation_role')
            if role is not None and role not in EVALUATION_ROLES:
                blockers.add('unknown_evaluation_role')
            elif role is not None:
                assigned.add(role)
            else:
                # Only exact retained roles can supply an assignment; development
                # cannot be guessed into calibration or selection.
                exact_roles = row_history & set(EVALUATION_ROLES)
                assigned.update(exact_roles)
                if not exact_roles:
                    blockers.add('missing_explicit_role')
            if row.get('source_kind') not in ('real', 'synthetic'):
                blockers.add('unknown_source_kind')
            elif row.get('source_kind') == 'synthetic':
                blockers.add('synthetic_voice_not_independent_real_speaker')
            if not known(row.get('history_evidence_id')):
                blockers.add('missing_exposure_history_evidence')
            if not all(known(row.get(field)) for field in REQUIRED):
                blockers.add('missing_source_speaker_or_session_lineage')
            if row.get('audio_status') != 'verified_pcm16' or not known(row.get('pcm_sha256')):
                blockers.add('missing_verified_pcm_identity')
            if quality_tier(row) != 'human_review_recorded' or machine_candidate(row) or row.get('label_origin') == 'machine_consensus':
                blockers.add('not_independently_human_reviewed')
            if row.get('previously_exposed') is True or row.get('test_results_seen') is True:
                blockers.add('previously_exposed')
            history.update(row_history)
            support.append({'record_id': row['record_id'], 'group_id': group_id,
                            'historical_roles': sorted(row_history),
                            'exposure_history': copy.deepcopy(row.get('exposure_history')),
                            'history_complete': row.get('history_complete') is True})
        for row in members:
            role_blockers = {}
            for role in required_roles:
                reasons = set(blockers)
                if assigned != {role}:
                    reasons.add('missing_or_conflicting_explicit_role')
                if history - {role}:
                    reasons.add('incompatible_historical_support')
                if role == 'sealed_test' and history:
                    reasons.add('sealed_test_already_exposed')
                role_blockers[role] = sorted(reasons)
            results.append({'record_id': row['record_id'], 'group_id': group_id,
                            'role_eligibility': {role: not reasons for role, reasons in role_blockers.items()},
                            'role_blockers': role_blockers,
                            'fresh_heldout': not history and any(not r for r in role_blockers.values()),
                            'input': row,
                            'human_gold': False, 'training_admitted': False,
                            'product_qualification_allowed': False})
    return {'schema': 'effect-role-eligibility-v1', 'required_roles': list(required_roles),
            'rows': sorted(results, key=lambda row: row['record_id']),
            'historical_support': sorted(support, key=lambda row: row['record_id']),
            'training_admitted': False, 'product_qualification_allowed': False,
            'note': 'Eligibility is conditional on supplied support metadata; not live-byte verification, gold certification, or admission.'}


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
