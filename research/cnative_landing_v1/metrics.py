"""Keyword metrics for already-collapsed CTC ids; no model or transcript access.

Whole-sequence equality, complete-keyword containment, and nonempty output are
separate measurements. None is an event-level continuous-stream FAR.
"""
import argparse
import hashlib
import json
from pathlib import Path

TARGETS = {'K1': (1, 2, 3, 4), 'K2': (3, 4, 3, 4)}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_ids(ids, *, collapsed=True):
    require(isinstance(ids, (list, tuple)), 'ids must be a sequence')
    require(len(ids) <= 100000, 'sequence limit')
    low = 1 if collapsed else 0
    require(all(type(x) is int and low <= x <= 4 for x in ids), 'invalid local token id')
    return tuple(ids)


def collapse_ctc(frame_ids):
    ids = validate_ids(frame_ids, collapsed=False)
    out, previous = [], None
    for value in ids:
        if value != previous and value != 0:
            out.append(value)
        previous = value
    return tuple(out)


def contains_keyword(ids, keyword):
    require(keyword in TARGETS, 'unknown keyword')
    ids = validate_ids(ids)
    target = TARGETS[keyword]
    return any(ids[i:i+len(target)] == target for i in range(len(ids)-len(target)+1))


def confusion(rows, decisions):
    require(len(rows)==len(decisions), 'decision/row length mismatch')
    require(all(type(d) is bool for d in decisions), 'decisions must be bool')
    known = [(r, d) for r, d in zip(rows, decisions) if r.get('known', True)]
    tp = sum(r['gold_present'] and d for r, d in known)
    fn = sum(r['gold_present'] and not d for r, d in known)
    fp = sum(not r['gold_present'] and d for r, d in known)
    tn = sum(not r['gold_present'] and not d for r, d in known)
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'positive_rows': tp+fn, 'negative_rows': fp+tn,
            'recall': tp/(tp+fn) if tp+fn else None,
            'row_fpr': fp/(fp+tn) if fp+tn else None,
            'far_per_hour': None}


def score_rows(rows, keyword):
    require(keyword in TARGETS, 'unknown keyword')
    require(type(rows) is list and len(rows) <= 100000, 'bounded row list required')
    seen, exact, complete, nonempty = set(), [], [], []
    for r in rows:
        require(type(r) is dict, 'row object required')
        sid = r.get('source_id')
        require(type(sid) is str and bool(sid) and sid not in seen, 'missing/duplicate source id')
        seen.add(sid)
        known = r.get('known', True)
        require(type(known) is bool, 'known must be boolean')
        require((type(r.get('gold_present')) is bool) if known else r.get('gold_present') is None,
                'known label must be bool; unknown label must be null')
        ids = validate_ids(r.get('greedy_local_ids'))
        exact.append(ids == TARGETS[keyword])
        complete.append(contains_keyword(ids, keyword))
        nonempty.append(bool(ids))
    return {'keyword': keyword, 'rows': len(rows),
            'unknown_rows': sum(not r.get('known', True) for r in rows),
            'exact_sequence': confusion(rows, exact),
            'complete_keyword_contains': confusion(rows, complete),
            'nonempty_diagnostic_only': confusion(rows, nonempty)}


def audit_archive(model_dir):
    """Append-only reinterpretation: retain the input result bytes unchanged."""
    reports = []
    for p in sorted(Path(model_dir).glob('result-causal-ds-tcn-layernorm-ctc-seed*.json')):
        raw = p.read_bytes()
        require(len(raw) < 4*1024**2, 'result size limit')
        old = json.loads(raw)
        require(type(old.get('seed')) is int, 'integer seed required')
        require([x['keyword'] for x in old['dev']] == ['K1', 'K2'], 'closed head order')
        heads = []
        for head in old['dev']:
            new = score_rows(head['rows'], head['keyword'])
            old_values = {k: head[k] for k in ('exact_recall', 'false_positive_rate', 'tp','fp','fn','tn')}
            nonempty = new['nonempty_diagnostic_only']
            new['legacy_values'] = old_values
            new['legacy_counts_match_nonempty'] = all(old_values[k] == nonempty[k] for k in ('tp','fp','fn','tn'))
            new['legacy_named_exact_recall_matches_nonempty'] = old_values['exact_recall'] == nonempty['recall']
            new['legacy_named_exact_recall_matches_exact_sequence'] = old_values['exact_recall'] == new['exact_sequence']['recall']
            heads.append(new)
        reports.append({'seed': old['seed'], 'input_name': p.name,
                        'input_sha256': hashlib.sha256(raw).hexdigest(), 'heads': heads})
    require(len(reports) == 3 and len({r['seed'] for r in reports}) == 3, 'three distinct seeds required')
    return {'schema': 'kws-complete-keyword-metric-audit-v1',
            'input_scope': '34 previously exposed weak-label development rows per seed',
            'original_files_modified': False, 'new_model_forwards': 0,
            'new_training_updates': 0, 'new_asr_calls': 0, 'new_tts_calls': 0,
            'continuous_far_measured': False, 'model_promoted': False,
            'results': reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit_archive(a.model_dir)
    # Exclusive creation: an earlier audit receipt may never be overwritten.
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
