"""Cosy49 admission/objective/gates. Import is stdlib-only; no launcher or I/O.

The optional tensor adapter imports Torch only when explicitly called by a later,
separately reviewed runtime. These helpers do not confer execution authority.
"""
from collections import Counter
from fractions import Fraction
import math

TOKENS = ('<blank>', '你', '好', '小', '窝', '屋')
D20 = 'D20-weak-rehearsal'
Q12 = 'Qwen20-reviewed'
C17 = 'Cosy30-human-actual'
COUNTS = {D20: 20, Q12: 12, C17: 17}
EXPECTED_ALIASES = tuple('ABCDEFGHIJKLMNOPR')
STATE_SHA = 'c05623683b4616badda5535eefb0bfaecde63efbf3bfbceec6e7440a5fc19eb2'
STEPS, SEED = 300, 610104


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(text):
    require(type(text) is str and bool(text), 'nonempty actual transcript required')
    require(all(c in TOKENS[1:] for c in text), 'unknown/blank token prohibited')
    return [TOKENS.index(c) for c in text]


def minimum_frames(target):
    require(type(target) is list and bool(target), 'nonempty target list')
    require(all(type(t) is int and 1 <= t <= 5 for t in target), 'nonblank six-class vocabulary')
    return len(target) + sum(x == y for x, y in zip(target, target[1:]))


def cohort_weights(cohorts):
    require(Counter(cohorts) == Counter(COUNTS), 'exact train49 source cohorts')
    return [Fraction(1, 3 * COUNTS[c]) for c in cohorts]


def validate_rows(rows):
    require(type(rows) is list and len(rows) == 49, 'exact train49')
    require(len({r['recording'] for r in rows}) == 49, 'unique recordings')
    require(len({r['wav_sha256'] for r in rows}) == 49, 'unique WAV identities')
    require(len({r['pcm_sha256'] for r in rows}) == 49, 'unique PCM identities')
    require(all(r['role'] == 'train' for r in rows), 'development/sealed prohibited')
    cohort_weights([r['cohort'] for r in rows])
    cosy = [r for r in rows if r['cohort'] == C17]
    require(tuple(r['alias'] for r in cosy) == EXPECTED_ALIASES, 'exact Cosy17 order/exclusions')
    for r in rows:
        require(encode(r['text']) == r['target_ids'], 'actual full target binding')
        require(len(r['target_ids']) in (2, 3, 4), 'two/three/four-token targets')
        require(type(r['model_rows']) is int and minimum_frames(r['target_ids']) <= r['model_rows'] <= 95, 'feasible native lengths')
        if r['cohort'] == C17:
            require(r['text'] == r['human_actual_transcript'], 'Cosy observed target required')
            if r['alias'] in ('D', 'I'):
                require(r['text'] == {'D': '小窝', 'I': '小屋'}[r['alias']], 'D/I actual two-token target')
                require(r['complete_wake_target'] is False, 'D/I are not complete wake positives')
    require(sum(r['model_rows'] for r in rows) == 2505, 'fixed valid feature geometry')
    return rows


def objective(nll, rows):
    """Mathematical scalar test oracle; nonnegativity is NOT a production FP32 gate."""
    validate_rows(rows)
    require(len(nll) == 49, '49 source losses')
    require(all(math.isfinite(x) and x >= 0 for x in nll), 'finite nonnegative CTC')
    return math.fsum(float(w) * x / len(r['target_ids']) for w, x, r in zip(cohort_weights([r['cohort'] for r in rows]), nll, rows))


def tensor_ctc_loss(logits, lengths, targets, cohorts):
    """Future drop-in replacement for old control.tensor_ctc_loss; NOT run in prep.

    Exact old float32 CPU/reduction/zero_infinity=False interface, changed only
    to train49, source-cohort weights, and legitimate two-token actual words.
    """
    weights = cohort_weights(cohorts)
    require(len(lengths) == len(targets) == 49, '49 source lengths/targets')
    require(all(type(n) is int and minimum_frames(y) <= n <= 95 and len(y) in (2, 3, 4) for n, y in zip(lengths, targets)), 'feasible target lengths')
    import torch
    require(tuple(logits.shape) == (49, 95, 6) and logits.dtype == torch.float32 and logits.device.type == 'cpu', 'CPU float32 train49 logits')
    require(torch.isfinite(logits).all().item(), 'finite logits')
    target_lengths = torch.tensor([len(y) for y in targets], dtype=torch.long)
    losses = torch.nn.functional.ctc_loss(
        logits.log_softmax(-1).transpose(0, 1),
        torch.tensor([t for y in targets for t in y], dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long), target_lengths,
        blank=0, reduction='none', zero_infinity=False)
    require(torch.isfinite(losses).all().item(), 'nonfinite CTC stops attempt')
    normalized = losses / target_lengths
    return (normalized * torch.tensor([float(w) for w in weights], dtype=normalized.dtype)).sum(), normalized


def terminal_fit(initial_normalized, terminal_normalized, cohorts):
    """Train-fit diagnostic only. No step selection or acoustic quality decision."""
    weights = cohort_weights(cohorts)
    require(len(initial_normalized) == len(terminal_normalized) == 49, 'complete initial/terminal train49 diagnostics')
    require(all(math.isfinite(x) for x in initial_normalized + terminal_normalized), 'finite fit values')
    first = math.fsum(float(w) * x for w, x in zip(weights, initial_normalized))
    last = math.fsum(float(w) * x for w, x in zip(weights, terminal_normalized))
    group = {c: {'initial': math.fsum(x for x, k in zip(initial_normalized, cohorts) if k == c) / n,
                 'terminal': math.fsum(x for x, k in zip(terminal_normalized, cohorts) if k == c) / n} for c, n in COUNTS.items()}
    improved = last < first and group[C17]['terminal'] < group[C17]['initial']
    return dict(status='TRAIN_FIT_IMPROVED' if improved else 'TRAIN_FIT_NOT_IMPROVED', initial=first, terminal=last,
                cohorts=group, quality_pass=False, diagnostic_only=True, stop_training=False, retrain_allowed=False, intermediate_selection=False)


def _events(rows):
    require(set(rows) == set(EXPECTED_ALIASES) | set('SUVWX'), 'all22 aliases required')
    for alias, row in rows.items():
        require(row.get('complete') is True and row.get('error') is None, 'partial/error trace is never a negative')
        require(type(row.get('events')) is list, 'complete event list')
        require(all(type(k) is int and k in (1, 2) for k in row['events']), 'keyword IDs only; all repeated/wrong events retained')
    return rows


def cosy_endpoint_gate(original, candidate):
    """Pure proposed gate on complete already-audited events, not a trace auditor.

    Original exact A20 behavioral signature must agree with frozen Cosy22.
    W is descriptive only. Every other source protects hits and all event types.
    """
    original, candidate = _events(original), _events(candidate)
    expected = {a: [] for a in original}
    expected.update(A=[1], B=[2], H=[1], N=[1], O=[2])
    if any(original[a]['events'] != expected[a] for a in expected):
        return dict(status='STOP_BASELINE_DRIFT', quality_pass=False)
    regression, changes = [], []
    positives = {'A': 1, 'B': 2, 'C': 1, 'O': 2}
    for a in sorted(original):
        before, after = original[a]['events'], candidate[a]['events']
        if before != after:
            changes.append(dict(alias=a, original=before, candidate=after))
        if a == 'W':
            continue
        if a in positives:
            k = positives[a]
            if k in before and k not in after:
                regression.append((a, 'lost_correct_keyword'))
            if any(Counter(after)[j] > Counter(before)[j] for j in (1, 2) if j != k):
                regression.append((a, 'added_wrong_keyword'))
            if Counter(after)[k] > max(1, Counter(before)[k]):
                regression.append((a, 'added_repeat'))
        elif any(Counter(after)[j] > Counter(before)[j] for j in (1, 2)):
            regression.append((a, 'added_nonwake_event'))
    gains = (["C_miss_recovered"] if 1 in candidate['C']['events'] else [])
    gains += [a + '_false_events_removed' for a in ('H', 'N') if not candidate[a]['events']]
    status = 'STOP_COSY_REGRESSION' if regression else ('FIT_ENDPOINT_GAIN' if gains else 'STOP_NO_TARGETED_ENDPOINT_GAIN')
    return dict(status=status, targeted_gains=gains, regressions=regression, all_event_changes=changes,
                W_interpretation='words confirmed; abrupt ending; final-syllable cutoff uncertain; no recall verdict',
                heldout_or_generalization_pass=False, quality_pass=False)
