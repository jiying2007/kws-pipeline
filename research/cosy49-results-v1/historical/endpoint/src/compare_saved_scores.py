"""Pure comparison of already validated complete Qwen20 score records.

No audio, frontend, model, probability recomputation or decoder execution.
Upstream scoring must verify trace completeness and frozen raw/source identities.
Greedy transcript accuracy deliberately has no influence on the wake gate.
"""
from collections import Counter
from contracts import require


def event_counts(row):
    require(row.get('complete') is True and row.get('feed_finish_records_validated') is True,
            'incomplete/invalid trace is not a negative')
    require(all(type(e['keyword']) is int and e['keyword'] in (1, 2) for e in row['events']), 'keyword')
    return Counter(e['keyword'] for e in row['events'])


def compare(original, candidate):
    require(len(original) == len(candidate) == 20, 'all twenty reviewed sources required')
    old = {r['recording']: r for r in original}
    new = {r['recording']: r for r in candidate}
    require(len(old) == len(new) == 20 and set(old) == set(new), 'exact source set')
    gains, regressions, details = [], [], []
    for name, a in old.items():
        b = new[name]
        for key in ('pcm_sha256', 'wav_sha256', 'kind', 'keyword_id', 'historical_split', 'voice', 'declared_text'):
            require(a[key] == b[key], 'label/input drift: ' + name + '/' + key)
        ca, cb = event_counts(a), event_counts(b)
        positive = a['kind'] == 'positive'
        if positive:
            k = a['keyword_id']
            if ca[k] and not cb[k]: regressions.append([name, 'lost_original_positive'])
            if not ca[k] and cb[k]: gains.append([name, 'recovered_original_miss'])
            if cb[3-k] > ca[3-k]: regressions.append([name, 'new_wrong_keyword_event'])
            if max(cb[k]-1, 0) > max(ca[k]-1, 0): regressions.append([name, 'new_duplicate_matching_event'])
        else:
            for k in (1, 2):
                if cb[k] > ca[k]: regressions.append([name, 'new_negative_event_K'+str(k)])
            if name == 'qwen3-repeat-nihao-eric' and sum(ca.values()) > 0 and not cb:
                gains.append([name, 'removed_original_repeat_nihao_false_event'])
        if max(sum(cb.values())-1, 0) > max(sum(ca.values())-1, 0):
            regressions.append([name, 'additional_repeat_event'])
        details.append(dict(recording=name, old_events=a['events'], candidate_events=b['events'],
                            old_counts=dict(ca), candidate_counts=dict(cb)))
    return dict(schema='a20-exposed-development-candidate-screen-v1', all20=details,
                gains=gains, regressions=regressions,
                merits_next_research_stage=bool(gains) and not regressions,
                no_improvement_ends_candidate=not gains, qualification=False,
                greedy_exact_used_for_wake_gate=False)
