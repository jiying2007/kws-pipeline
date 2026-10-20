#!/usr/bin/env python3
"""Saved-event offline comparison, not inference, threshold selection or promotion."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'tools')]
from score_events import score, validate_recordings, validate_detections
from statistical_bounds import wilson_upper, poisson_rate_upper
from context_policy import valid_sha, policy_unknowns, unique_object

TIERS = ('synthetic', 'public', 'real_device')
IDENTITY = ('model_sha256', 'frontend_sha256', 'decoder_sha256', 'thresholds_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def read_rows(path):
    return [json.loads(line, object_pairs_hook=unique_object)
            for line in Path(path).read_text().splitlines() if line.strip()]


def validate_plan(plan):
    require(plan['schema'] == 'offline-effect-plan-v1', 'unsupported plan')
    require(type(plan.get('family_held_out', False)) is bool, 'family_held_out must be boolean')
    for field in ('references_sha256', 'source_split_sha256', 'frozen_plan_receipt_sha256'):
        require(valid_sha(plan[field]), 'missing frozen identity: ' + field)
    require(isinstance(plan['keyword_ids'], list) and bool(plan['keyword_ids']) and
            all(type(k) is int and 0 <= k <= 0xFFFFFFFF for k in plan['keyword_ids']) and
            len(set(plan['keyword_ids'])) == len(plan['keyword_ids']), 'explicit unique keyword IDs required')
    require(plan['test_used_for_selection'] is False, 'test selection forbidden')
    require(plan['comparison'] in ('fixed_operating_point', 'calibration_matched_far'), 'unknown comparison')
    require(plan['split_role'] in ('development', 'held_out_test'), 'unknown split role')
    require(not policy_unknowns(plan['context_policy']), 'complete context policy required')
    matching = plan['event_matching']
    require(type(matching['pre_tolerance_ms']) in (int, float) and matching['pre_tolerance_ms'] == 0, 'pre-onset credit forbidden')
    post = matching['post_tolerance_ms']
    require(type(post) in (int, float) and math.isfinite(post) and post >= 0, 'invalid post tolerance')
    require(set(plan['arms']) == {'control', 'candidate'}, 'exactly two predeclared arms required')
    for arm in plan['arms'].values():
        require(all(valid_sha(arm.get(field)) for field in IDENTITY), 'incomplete arm identity')
        require(arm['context_policy'] == plan['context_policy'], 'arm context differs')
    if plan['comparison'] == 'fixed_operating_point':
        require(plan['arms']['control']['thresholds_sha256'] == plan['arms']['candidate']['thresholds_sha256'],
                'fixed operating point requires identical thresholds')
    else:
        cal = plan['calibration']
        require(valid_sha(cal['references_sha256']) and valid_sha(cal['selection_receipt_sha256']), 'calibration identity required')
        require(cal['references_sha256'] != plan['references_sha256'], 'calibration must differ from evaluated references')
        require(cal['selection_split'] == 'calibration' and cal['test_used'] is False, 'calibration-only selection required')
        require(cal['source_groups_disjoint'] is True, 'calibration/evaluation source separation required')
        require(type(cal['target_far_per_hour']) in (int, float) and math.isfinite(cal['target_far_per_hour'])
                and cal['target_far_per_hour'] >= 0, 'invalid declared calibration FAR target')


def summarize(names, references, detections, metadata, post, keywords):
    refs = {n: references[n] for n in names}
    if not refs:
        return None
    result, fas, frs = score(refs, {n: detections.get(n, []) for n in names}, 0, post)
    continuous = {n for n in names if not refs[n]['expected'] and metadata[n]['continuous_negative']}
    hours = sum(refs[n]['duration_s'] for n in continuous) / 3600
    continuous_fas = [r for r in fas if r['recording'] in continuous]
    result['frr'] = result['frr'] if result['expected'] else None
    result['frr_upper_95'] = wilson_upper(len(frs), result['expected'], .95) if result['expected'] else None
    for keyword in keywords:
        result['per_keyword'].setdefault(str(keyword), dict(expected=0, matched=0, false_rejects=0, false_accepts=0, frr=None))
    for keyword, item in result['per_keyword'].items():
        item['frr'] = item['frr'] if item['expected'] else None
        item['frr_upper_95'] = wilson_upper(item['false_rejects'], item['expected'], .95) if item['expected'] else None
        fa = sum(r['keyword_id'] == int(keyword) for r in continuous_fas)
        item['continuous_negative_false_accepts'] = fa
        item['continuous_negative_far_per_hour'] = fa / hours if hours else None
        item['continuous_negative_far_upper_95_per_hour'] = poisson_rate_upper(fa, hours, .95) if hours else None
    # The canonical scorer's all-audio FAR is not continuous-background evidence.
    result['clip_negative_hours'] = result.pop('negative_recording_audio_hours')
    result['clip_negative_false_accepts'] = result.pop('negative_recording_false_accepts')
    for key in list(result):
        if key.startswith('negative_recording_far') or key == 'far_per_hour':
            del result[key]
    result['continuous_negative'] = {
        'hours': hours, 'recordings': len(continuous),
        'longest_seconds': max((refs[n]['duration_s'] for n in continuous), default=0),
        'false_accepts': len(continuous_fas),
        'far_per_hour': len(continuous_fas) / hours if hours else None,
        'far_upper_95_per_hour': poisson_rate_upper(len(continuous_fas), hours, .95) if hours else None,
    }
    result['false_accept_rows'] = fas
    result['false_reject_rows'] = frs
    return result


def known_group(value):
    return (isinstance(value, str) and bool(value.strip()) and value == value.strip()
            and value.lower() not in {'unknown', 'none', 'null', 'n/a'})


def lineage_value(item, field):
    aliases = {'room_id': ('room_id', 'rir_room_id', 'RIRroom_id'),
               'room_config_id': ('room_config_id', 'rir_config_id', 'RIRconfig_id')}
    values = [item[key] for key in aliases.get(field, (field,)) if key in item]
    require(not values or all(value == values[0] for value in values), 'conflicting lineage aliases: ' + field)
    return values[0] if values else None


def validate_source_split(plan, rows, split):
    """Fail closed on exact-byte and declared lineage leakage, not a rights audit."""
    records = split['records']
    require(isinstance(records, list) and bool(records), 'source split records required')
    fields = ['root_recording_id', 'session_id', 'speaker_id', 'preset_identity', 'room_id', 'room_config_id']
    seen = set()
    groups = {field: {} for field in fields}
    family_roles = {}
    evaluation = {}
    for item in records:
        pcm, role, family = item['pcm_sha256'], item['split_role'], item.get('source_family')
        require(valid_sha(pcm) and pcm not in seen, 'duplicate or missing PCM across source splits')
        require(known_group(family), 'source_family required')
        require(role in ('training', 'calibration', 'development', 'held_out_test'), 'unknown source split role')
        seen.add(pcm)
        if plan.get('family_held_out', False):
            require(family not in family_roles or family_roles[family] == role, 'source family crosses held-out splits')
            family_roles[family] = role
        for field in fields:
            group = lineage_value(item, field)
            if field != 'root_recording_id' and group is None:
                continue
            require(known_group(group), 'missing source lineage: ' + field)
            key = (family, group)
            require(key not in groups[field] or groups[field][key] == role,
                    'source lineage crosses splits: ' + field)
            groups[field][key] = role
        if role == plan['split_role']:
            evaluation[pcm] = item
    require(set(evaluation) == {row['pcm_sha256'] for row in rows}, 'evaluated PCM does not equal frozen split')
    for row in rows:
        frozen = evaluation[row['pcm_sha256']]
        for field in ['source_family'] + fields:
            require(lineage_value(row, field) == lineage_value(frozen, field),
                    'reference lineage differs from frozen split')


def build_report(plan, rows, control_rows, candidate_rows):
    validate_plan(plan)
    refs = validate_recordings(rows)
    metadata = {r['recording']: r for r in rows}
    seen_pcm = set()
    continuous_groups = {field: set() for field in ('root_recording_id', 'session_id')}
    for row in rows:
        require(row['evidence_tier'] in TIERS, 'unknown evidence tier')
        require(isinstance(row['scenario'], str) and row['scenario'].strip(), 'scenario required')
        require(valid_sha(row['pcm_sha256']) and row['pcm_sha256'] not in seen_pcm, 'missing or duplicate PCM identity')
        seen_pcm.add(row['pcm_sha256'])
        require(type(row['continuous_negative']) is bool, 'explicit continuous declaration required')
        if row['continuous_negative']:
            require(not row['expected'], 'positive is not continuous negative')
            for field, seen_groups in continuous_groups.items():
                group = row.get(field)
                require(known_group(row.get('source_family')), 'continuous source_family required')
                require(known_group(group), 'continuous source lineage required: ' + field)
                key = (row['source_family'], group)
                require(key not in seen_groups, 'duplicate continuous source exposure: ' + field)
                seen_groups.add(key)
            context = plan['context_policy']
            require(context['appended_context_samples'] == 0 and context['appended_context_kind'] == 'none'
                    and context['eof_flush'] is False and context['eof_padding'] == 'none',
                    'continuous FAR excludes appended, flushed or padded context')
            require(row['continuity']['kind'] == 'original_uninterrupted_recording', 'concatenated/repeated clips are not continuous exposure')
            require(row['continuity']['reset_count'] == 0 and type(row['continuity']['reset_count']) is int,
                    'continuous exposure contains resets')
            require(valid_sha(row['continuity']['execution_receipt_sha256']), 'continuous execution receipt required')
    observed_keywords = {event['keyword_id'] for row in rows for event in row['expected']}
    observed_keywords.update(event['keyword_id'] for event in control_rows + candidate_rows)
    require(observed_keywords <= set(plan['keyword_ids']), 'undeclared keyword')
    arms = {name: validate_detections(data, refs) for name, data in
            (('control', control_rows), ('candidate', candidate_rows))}
    tiers = {}
    for tier in TIERS:
        names = [n for n in refs if metadata[n]['evidence_tier'] == tier]
        tiers[tier] = {'present': bool(names), 'arms': {}, 'scenarios': {}}
        for arm, detections in arms.items():
            tiers[tier]['arms'][arm] = summarize(names, refs, detections, metadata, plan['event_matching']['post_tolerance_ms'] / 1000, plan['keyword_ids'])
        for scenario in sorted({metadata[n]['scenario'] for n in names}):
            subset = [n for n in names if metadata[n]['scenario'] == scenario]
            tiers[tier]['scenarios'][scenario] = {arm: summarize(subset, refs, dets, metadata,
                plan['event_matching']['post_tolerance_ms'] / 1000, plan['keyword_ids']) for arm, dets in arms.items()}
        if names:
            a, b = (tiers[tier]['arms'][arm] for arm in ('control', 'candidate'))
            tiers[tier]['candidate_minus_control'] = {key: b[key] - a[key]
                for key in ('matched', 'false_rejects', 'false_accepts')}
    return {'schema': 'offline-effect-report-v1', 'plan': plan, 'tiers': tiers,
            'production_qualified': False, 'product_qualification_allowed': False, 'shipping_approved': False,
            'numeric_streaming_gate': 'unchanged; required independently before candidate execution',
            'limitations': ['Input declarations and receipts are bound, not independently attested.',
                'Evidence tier is declared provenance, not proof of license rights, gold labels, speaker independence or device qualification.',
                'Null/missing optional session, speaker or room provenance remains unknown; continuous FAR requires known session and root.',
                'No pooling across evidence tiers; no threshold search or statistical superiority claim.',
                'Wilson/Poisson bounds assume binomial/Poisson sampling; correlated/reused sources need grouped analysis.',
                'Calibration matched-FAR is a predeclared calibration target, not a claim of equal test FAR.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'references', 'source-split', 'control-detections', 'candidate-detections', 'execution-receipt', 'output'):
        p.add_argument('--' + flag, required=True, type=Path)
    args = p.parse_args()
    plan = read_json(args.plan)
    require(digest(args.references) == plan['references_sha256'], 'reference hash mismatch')
    require(digest(args.source_split) == plan['source_split_sha256'], 'source split hash mismatch')
    rows = read_rows(args.references)
    split = read_json(args.source_split)
    validate_source_split(plan, rows, split)
    receipt = read_json(args.execution_receipt)
    require(receipt['plan_sha256'] == digest(args.plan), 'execution plan mismatch')
    require(receipt['references_sha256'] == digest(args.references), 'execution references mismatch')
    for arm in ('control', 'candidate'):
        require(receipt['arms'][arm]['identity'] == plan['arms'][arm], 'execution identity mismatch')
        require(receipt['arms'][arm]['detections_sha256'] == digest(getattr(args, arm + '_detections')), 'execution detections mismatch')
    result = build_report(plan, rows, read_rows(args.control_detections), read_rows(args.candidate_detections))
    result['input_sha256'] = {key: digest(getattr(args, key)) for key in
        ('plan', 'references', 'source_split', 'control_detections', 'candidate_detections', 'execution_receipt')}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
