#!/usr/bin/env python3
"""Offline research admission and saved-prediction metrics. Never runs a model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parent
ROLES = ('inventory', 'train', 'dev', 'heldout')
STRATA = ('dataset', 'tier', 'speaker', 'generator_voice', 'noise', 'far_field', 'hard_negative')


class Invalid(ValueError):
    """Malformed, inconsistent, or ineligible evidence."""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise Invalid('non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=bad)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_check(value, schema, path='$'):
    """Validate the deliberately small JSON Schema subset used by our schemas."""
    if 'anyOf' in schema:
        for choice in schema['anyOf']:
            try:
                schema_check(value, choice, path)
                return
            except Invalid:
                pass
        raise Invalid(path + ': does not match any allowed schema')
    kinds = {'object': dict, 'array': list, 'string': str, 'boolean': bool,
             'null': type(None), 'integer': int, 'number': (int, float)}
    kind = schema.get('type')
    if kind:
        require(isinstance(value, kinds[kind]) and
                not (kind in ('integer', 'number') and isinstance(value, bool)),
                path + ': wrong type; expected ' + kind)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        require(math.isfinite(value), path + ': non-finite number')
    if 'enum' in schema:
        require(value in schema['enum'], path + ': not an allowed value')
    if 'const' in schema:
        require(type(value) is type(schema['const']) and value == schema['const'],
                path + ': wrong constant')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        require(all(k in value for k in schema.get('required', [])), path + ': missing required field')
        if schema.get('additionalProperties') is False:
            require(set(value) <= set(props), path + ': unknown field(s)')
        for key, item in value.items():
            if key in props:
                schema_check(item, props[key], path + '.' + key)
    if isinstance(value, list):
        require(len(value) >= schema.get('minItems', 0), path + ': too few items')
        if schema.get('uniqueItems'):
            require(len({json.dumps(x, sort_keys=True) for x in value}) == len(value),
                    path + ': duplicate array values')
        for i, item in enumerate(value):
            schema_check(item, schema.get('items', {}), path + '[' + str(i) + ']')
    if isinstance(value, str):
        require(len(value) >= schema.get('minLength', 0), path + ': empty string')
        if 'pattern' in schema:
            require(re.search(schema['pattern'], value) is not None, path + ': invalid format')
    for key, predicate in [('minimum', lambda x, y: x >= y),
                           ('maximum', lambda x, y: x <= y),
                           ('exclusiveMinimum', lambda x, y: x > y),
                           ('exclusiveMaximum', lambda x, y: x < y)]:
        if key in schema:
            require(predicate(value, schema[key]), path + ': out of range')


def overlaps(a, b):
    return max(a['start_s'], b['start_s']) < min(a['end_s'], b['end_s'])


def group_keys(asset):
    s = asset['source']
    keys = [('source_family', s['family']), ('leakage_group', asset['leakage_group']),
            ('derivation_family', asset['derivation_family']),
            ('pcm', asset['pcm_sha256']), ('wav', asset['wav_sha256'])]
    if s['session'] is not None:
        keys.append(('recording_session', (s['dataset'], s['session'])))
    if s['speaker'] is not None:
        keys.append(('speaker', s['speaker']))
    if s['generator'] is not None:
        g = s['generator']
        keys += [('generator_voice', (g['model'], g['voice'])),
                 ('generator_prompt', (g['model'], g['prompt_family']))]
    return [(kind, value) for kind, value in keys if value is not None]


def validate_manifest(manifest):
    schema_check(manifest, load(ROOT / 'schemas/manifest.schema.json'))
    require(manifest['protocol']['input_order'] == [a['id'] for a in manifest['assets']],
            'manifest assets must match frozen input order')
    keys, ids = defaultdict(list), set()
    keywords = set(manifest['keywords'])
    for a in manifest['assets']:
        ident = a['id']
        require(ident not in ids, 'duplicate asset id: ' + ident)
        ids.add(ident)
        require(math.isclose(a['duration_s'], a['frames'] / a['sample_rate_hz'],
                             rel_tol=0, abs_tol=1e-9), ident + ': frame/duration mismatch')
        require(set(a['declared_keywords']) <= keywords, ident + ': unknown declared keyword')
        admitted = a['role'] != 'inventory'
        if admitted:
            require(a['role'] in a['license']['approved_roles'] and
                    a['license']['review_status'] == 'reviewed_research_scope',
                    ident + ': role lacks a reviewed data-use permission')
            require(a['exposure_review_complete'], ident + ': exposure review incomplete')
            require(a['wav_sha256'] is not None and a['pcm_sha256'] is not None,
                    ident + ': admitted audio needs WAV and PCM hashes')
            require(a['source']['tier'] not in ('unknown', 'numeric_fixture'),
                    ident + ': unqualified source tier')
            require(a['source']['session'] is not None, ident + ': missing session/group identity')
            if a['source']['tier'] == 'synthetic':
                require(a['source']['generator'] is not None, ident + ': missing generator identity')
            if a['source']['tier'] == 'open_human':
                require(a['source']['speaker'] is not None, ident + ': missing speaker identity')
        if a['role'] in ('dev', 'heldout') and not a['exposures']:
            require(manifest['protocol']['split_frozen_before_predictions'],
                    ident + ': independent scoring requires pre-result split freeze')
        if a['role'] == 'heldout':
            require(not a['exposures'], ident + ': exposed data cannot be heldout')
            require(manifest['protocol']['split_frozen_before_predictions'],
                    'heldout requires a pre-result split freeze')
            require(manifest['protocol']['heldout_open_authorized'],
                    'heldout opening not authorized in protocol')
        events = a['positive_events']
        negatives = a['negative_intervals']
        if a['label_strength'] not in ('verified', 'human_confirmed'):
            require(not events and not negatives, ident + ': weak/unknown labels cannot assert metric truth')
        if events or negatives or eligible(a):
            require(a['label_binding'] == 'source_bound', ident + ': asserted truth lacks source-bound labels')
        event_ids = set()
        for event in events:
            require(event['id'] not in event_ids, ident + ': duplicate positive event id')
            event_ids.add(event['id'])
            require(event['keyword'] in keywords, ident + ': unregistered keyword')
            require(0 <= event['start_s'] < event['end_s'] <= a['duration_s'],
                    ident + ': invalid event window')
            tail = event['word_tail_s']
            require((tail is None) == (event['word_tail_evidence_ref'] is None),
                    ident + ': word-tail truth must have an evidence reference')
            if tail is not None:
                require(in_window(tail, event, a['duration_s']), ident + ': tail outside event window')
        # Windows must be disjoint, including across keywords, so matching cannot be ambiguous.
        intervals = sorted(events + negatives, key=lambda x: x['start_s'])
        for interval in negatives:
            require(a['continuous'] and not a['replayed_or_looped'],
                    ident + ': negative hours require original continuous, non-replayed audio')
            require(interval['verified_absent_keywords'] == manifest['keywords'],
                    ident + ': negative verification must cover the complete keyword set in order')
            require(0 <= interval['start_s'] < interval['end_s'] <= a['duration_s'],
                    ident + ': invalid continuous negative interval')
        for before, after in zip(intervals, intervals[1:]):
            require(not overlaps(before, after), ident + ': overlapping truth windows/intervals')
        for key in group_keys(a):
            keys[key].append(a)
    conflicts = []
    for key, group in keys.items():
        active = {a['role'] for a in group if a['role'] != 'inventory'}
        if len(active) > 1:
            conflicts.append({'kind': key[0], 'assets': sorted({a['id'] for a in group})})
        if 'heldout' in active and any(a['exposures'] for a in group):
            conflicts.append({'kind': key[0] + '_prior_exposure', 'assets': sorted({a['id'] for a in group})})
    require(not conflicts, 'cross-split or heldout exposure leakage: ' + json.dumps(conflicts, ensure_ascii=False))
    # Byte-identical audio cannot increase eligible event counts or continuous hours, even in one split.
    for kind in ('wav', 'pcm'):
        for key, group in keys.items():
            if key[0] == kind:
                scored = [a for a in group if a['role'] in ('dev', 'heldout')]
                require(len(scored) < 2, 'duplicate scored audio identity: ' + ', '.join(a['id'] for a in scored))
    for key, group in keys.items():
        if key[0] in ('derivation_family', 'recording_session'):
            negatives = [a for a in group if a['role'] in ('dev', 'heldout') and a['negative_intervals']]
            require(len(negatives) < 2, 'repeated negative recording/derivation family: ' + ', '.join(a['id'] for a in negatives))
    return {a['id']: a for a in manifest['assets']}


def validate_predictions(predictions, manifest, manifest_sha256):
    schema_check(predictions, load(ROOT / 'schemas/predictions.schema.json'))
    require(predictions['manifest_sha256'] == manifest_sha256, 'predictions bind a different manifest')
    require(predictions['protocol_sha256'] == manifest['protocol']['sha256'], 'protocol identity mismatch')
    for key in ('model_sha256', 'source_sha256', 'decoder_config_sha256'):
        require(manifest['protocol'][key] is not None and predictions[key] == manifest['protocol'][key],
                'frozen baseline identity mismatch: ' + key)
    assets = {a['id']: a for a in manifest['assets']}
    require(predictions['state_policy'] == manifest['protocol']['state_policy'], 'state policy mismatch')
    require([a['asset_id'] for a in predictions['assets']] == manifest['protocol']['input_order'],
            'prediction assets must match frozen input order')
    runs = {}
    for run in predictions['assets']:
        ident = run['asset_id']
        require(ident in assets and ident not in runs, 'unknown or duplicate prediction asset: ' + ident)
        a = assets[ident]
        require(run['wav_sha256'] == a['wav_sha256'] and run['pcm_sha256'] == a['pcm_sha256'],
                ident + ': prediction input identity mismatch')
        require(run['complete'] and run['processed_duration_s'] == a['duration_s'],
                ident + ': incomplete processing cannot imply misses or zero false alarms')
        decision_ids = set()
        for d in run['decisions']:
            require(d['id'] not in decision_ids, ident + ': duplicate decision id')
            decision_ids.add(d['id'])
            require(d['keyword'] in manifest['keywords'], ident + ': unregistered decision keyword')
            require(0 <= d['decision_time_s'] <= a['duration_s'], ident + ': decision time out of audio range')
        runs[ident] = run
    # Require all assets, including empty decision lists; omitted negatives cannot manufacture 0 FA/h.
    require(set(runs) == set(assets), 'prediction coverage must equal the complete manifest asset set')
    for resource in predictions['recorded_resources']:
        name, value = resource['name'], resource['value']
        unit = 'bytes' if name.endswith('_bytes') else ('count' if name in ('io_syscalls', 'minor_faults', 'major_faults') else 'seconds')
        require(resource['unit'] == unit, 'recorded resource unit mismatch: ' + name)
        require(name == 'rss_drift_bytes' or value >= 0, 'negative recorded resource: ' + name)
        if unit in ('bytes', 'count'):
            require(value == int(value), 'fractional recorded byte/count field: ' + name)
    return runs


def wilson(misses, total, confidence):
    if not total:
        return None
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = misses / total
    denom = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [0.0 if misses == 0 else max(0.0, mid - radius),
            1.0 if misses == total else min(1.0, mid + radius)]


def poisson_upper_count(count, confidence):
    """Exact one-sided Poisson mean upper bound, inverted CDF, in log space."""
    log_alpha = math.log1p(-confidence)
    if count == 0:
        return -log_alpha
    def log_cdf(mu):
        # At this upper-bound search mu >= count, the last summand is maximal.
        term = total = 1.0
        for i in range(count, 0, -1):
            term *= i / mu
            total += term
            if term < total * 1e-16:
                break
        return -mu + count * math.log(mu) - math.lgamma(count + 1) + math.log(total)
    low, high = float(count), count + 10 * math.sqrt(count + 1) + 10
    while log_cdf(high) > log_alpha:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if log_cdf(mid) > log_alpha:
            low = mid
        else:
            high = mid
    return high


def summary(values):
    if not values:
        return {'count': 0, 'p50_s': None, 'p95_s': None, 'p99_s': None, 'max_s': None}
    ordered = sorted(values)
    def quantile(q):
        return ordered[max(0, math.ceil(len(ordered) * q) - 1)]
    return {'count': len(values), 'p50_s': quantile(.5), 'p95_s': quantile(.95),
            'p99_s': quantile(.99), 'max_s': ordered[-1]}


def eligible(a):
    return (a['role'] in ('dev', 'heldout') and not a['exposures']
            and a['label_strength'] in ('verified', 'human_confirmed'))


def strata(a):
    s, g = a['source'], a['source']['generator']
    return {'dataset': s['dataset'], 'tier': s['tier'], 'speaker': s['speaker'] or 'unknown',
            'generator_voice': (g['model'] + ':' + g['voice']) if g else 'not_applicable_or_unknown',
            **a['strata']}


def in_window(time, window, duration):
    # EOF finish callbacks belong to the final interval only; all interior boundaries are half-open.
    return window['start_s'] <= time < window['end_s'] or (time == duration == window['end_s'])


def metrics(assets, runs, keywords, protocol):
    counts = {k: {'eligible_positive_events': 0, 'hits': 0, 'misses': 0,
                  'duplicate_decisions': 0, 'wrong_keyword_decisions_in_positive_windows': 0,
                  'false_alarm_events': 0, 'latencies': [], 'missing_word_tail_hits': 0} for k in keywords}
    negative_s, unscored = 0.0, 0
    for a in assets:
        decisions = sorted(runs[a['id']]['decisions'], key=lambda d: (d['decision_time_s'], d['id']))
        used = set()
        for event in a['positive_events']:
            c = counts[event['keyword']]
            c['eligible_positive_events'] += 1
            window = [d for d in decisions if in_window(d['decision_time_s'], event, a['duration_s'])]
            matching = [d for d in window if d['keyword'] == event['keyword']]
            for d in window:
                used.add(d['id'])
                if d['keyword'] != event['keyword']:
                    c['wrong_keyword_decisions_in_positive_windows'] += 1
            if matching:
                c['hits'] += 1
                c['duplicate_decisions'] += len(matching) - 1
                if event['word_tail_s'] is None:
                    c['missing_word_tail_hits'] += 1
                else:
                    c['latencies'].append(matching[0]['decision_time_s'] - event['word_tail_s'])
            else:
                c['misses'] += 1
        for interval in a['negative_intervals']:
            negative_s += interval['end_s'] - interval['start_s']
            for d in decisions:
                if in_window(d['decision_time_s'], interval, a['duration_s']):
                    counts[d['keyword']]['false_alarm_events'] += 1
                    used.add(d['id'])
        unscored += len(decisions) - len(used)
    hours = negative_s / 3600
    output = {}
    confidence = protocol['confidence']
    for keyword, c in counts.items():
        n, misses, alarms = c['eligible_positive_events'], c['misses'], c['false_alarm_events']
        latency = summary(c.pop('latencies'))
        latency['status'] = 'DESCRIPTIVE_TRUE_WORD_TAIL' if latency['count'] else 'NOT_QUALIFIED'
        output[keyword] = {**c, 'recall': (n - misses) / n if n else None,
            'frr': misses / n if n else None,
            'frr_wilson_interval': wilson(misses, n, confidence) if protocol['iid_binomial_assumption'] else None,
            'negative_hours': hours, 'observed_fa_per_hour': alarms / hours if hours else None,
            'fa_per_hour_poisson_upper': poisson_upper_count(alarms, confidence) / hours
                if hours and protocol['stationary_poisson_assumption'] else None,
            'word_tail_latency': latency}
    total_alarms = sum(c['false_alarm_events'] for c in counts.values())
    return {'asset_count': len(assets), 'per_keyword': output, 'all_keyword_false_alarm_events': total_alarms,
        'negative_hours': hours, 'all_keyword_observed_fa_per_hour': total_alarms / hours if hours else None,
        'all_keyword_fa_per_hour_poisson_upper': poisson_upper_count(total_alarms, confidence) / hours
            if hours and protocol['stationary_poisson_assumption'] else None,
        'decisions_outside_verified_truth': unscored}


def admission(manifest):
    validate_manifest(manifest)
    excluded = []
    for a in manifest['assets']:
        if not eligible(a):
            excluded.append({'asset_id': a['id'], 'role': a['role'], 'label_strength': a['label_strength'],
                             'reasons': ['inventory/train role, exposed data, or weak/unknown truth: no independent scoring']})
    return {'schema_version': 1, 'research_only': True, 'shipping_approved': False,
            'assets': len(manifest['assets']), 'scoring_eligible_assets': sum(eligible(a) for a in manifest['assets']),
            'qualified_heldout_assets': sum(eligible(a) and a['role'] == 'heldout' for a in manifest['assets']),
            'excluded': excluded, 'thresholds': None, 'status': 'ADMISSION_CHECKED_NOT_PRODUCT_QUALIFIED'}


def score(manifest, predictions, manifest_sha256):
    report = admission(manifest)
    runs = validate_predictions(predictions, manifest, manifest_sha256)
    report['identity'] = {k: predictions[k] for k in ('manifest_sha256', 'protocol_sha256',
                         'model_sha256', 'source_sha256', 'decoder_config_sha256', 'run_id')}
    report['assumptions'] = {
        'confidence': manifest['protocol']['confidence'], 'frr_interval': 'two-sided Wilson; pointwise, no multiplicity correction',
        'iid_binomial_assumption': manifest['protocol']['iid_binomial_assumption'],
        'fa_bound': 'one-sided exact Poisson; homogeneous stationary rate and independent event counts/exposure',
        'stationary_poisson_assumption': manifest['protocol']['stationary_poisson_assumption'],
        'zero_events': 'A zero observed count does not establish a zero population FAR',
        'domain': 'Only the admitted source domain; synthetic/open-source is not product-board evidence',
        'latency': 'First correct callback-available audio timestamp minus annotated true word tail, same audio timeline; negative values retained; nearest-rank descriptive quantiles; no capture-to-wake claim',
        'annotation_and_exposure': 'Schema validates supplied claims; independent annotation, permissions, split freeze and one-time heldout access require external review'}
    report['partitions'] = {}
    for role in ('dev', 'heldout'):
        selected = [a for a in manifest['assets'] if a['role'] == role and eligible(a)]
        part = metrics(selected, runs, manifest['keywords'], manifest['protocol'])
        part['strata'] = {}
        for dimension in STRATA:
            groups = defaultdict(list)
            for a in selected:
                groups[strata(a)[dimension]].append(a)
            part['strata'][dimension] = {value: metrics(group, runs, manifest['keywords'], manifest['protocol'])
                                         for value, group in sorted(groups.items())}
        report['partitions'][role] = part
    report['exploratory'] = [{'asset_id': a['id'], 'label_strength': a['label_strength'],
        'role': a['role'], 'exposures': a['exposures'], 'duration_s': a['duration_s'],
        'decision_count': len(runs[a['id']]['decisions']), 'declared_keywords': a['declared_keywords'],
        'metrics_qualified': False} for a in manifest['assets'] if not eligible(a)]
    # No inferred resource baselines or percentage targets. Supplied recorded fields are only echoed.
    report['recorded_resources'] = predictions['recorded_resources']
    report['resource_status'] = 'RECORDED_FIELDS_ONLY_NOT_VERIFIED_OR_GATED'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('admit', 'score'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        manifest = load(args.manifest)
        if args.command == 'score':
            require(args.predictions is not None, 'score requires --predictions')
            result = score(manifest, load(args.predictions), sha(args.manifest))
        else:
            require(args.predictions is None, 'admit does not consume predictions')
            result = admission(manifest)
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            require(args.output.resolve() not in {args.manifest.resolve(),
                    args.predictions.resolve() if args.predictions else None}, 'output must not overwrite inputs')
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    except (Invalid, ValueError, OSError, OverflowError) as exc:
        print('INVALID: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
