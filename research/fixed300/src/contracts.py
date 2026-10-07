"""Standard-library contracts and small mathematical CTC oracle. No Torch/DSP."""
import hashlib
import math
from collections import Counter
from pathlib import Path

TOKENS = ('<blank>', '你', '好', '小', '窝', '屋')
HEAD = ('backbone.out_linear2.linear.weight', 'backbone.out_linear2.linear.bias')
CMVN = ('global_cmvn.mean', 'global_cmvn.istd')
D20 = 'D20-weak-rehearsal'
Q12 = 'Qwen20-reviewed'
INVENTORY_SHA = '4d8c7dcc8864d25bd13dbf00eec6df781ca1ef8625fb5e4e2c0349105c929aee'
PROPOSAL_SHA = 'efcaf56e9b0c1583e7d0adee54242afd2c777b645302c439c6fb1f53534f5900'
CHECKPOINT_SHA = '5b347c0ce7df6e8ad3649c9186a11acab4b3e35971950934490c511d2b04bc39'
STATE_SHA = 'c05623683b4616badda5535eefb0bfaecde63efbf3bfbceec6e7440a5fc19eb2'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(text):
    require(type(text) is str and bool(text), 'nonempty full transcript required')
    require(all(c in TOKENS[1:] for c in text), 'OOV must be excluded, never blanked')
    return [TOKENS.index(c) for c in text]


def validate_target(target, allow_empty=False):
    require(type(target) in (list, tuple), 'target sequence')
    require(allow_empty or bool(target), 'empty training target excluded')
    require(all(type(t) is int and 1 <= t < len(TOKENS) for t in target),
            'target has blank, OOV or noninteger token')


def minimum_frames(target):
    validate_target(target, allow_empty=True)
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def collapse(path):
    require(all(type(t) is int and 0 <= t < len(TOKENS) for t in path), 'path token')
    return [t for i, t in enumerate(path) if t and (i == 0 or t != path[i - 1])]


def logadd(values):
    m = max(values, default=-math.inf)
    return m if m == -math.inf else m + math.log(math.fsum(math.exp(v - m) for v in values))


def ctc_nll(log_probs, target):
    """Tiny float64 forward recursion, independent of Torch's CTC implementation.

    Empty targets are permitted for mathematical fixtures only, not this corpus.
    Input rows are six-class finite/-inf log probabilities with row sum one.
    An impossible alignment returns +inf; nothing is zeroed or truncated.
    """
    validate_target(target, allow_empty=True)
    for row in log_probs:
        require(len(row) == 6 and all(v <= 0 and not math.isnan(v) for v in row),
                'six nonpositive log probabilities')
        require(abs(math.fsum(math.exp(v) for v in row) - 1) <= 1e-12, 'probability row sum')
    if not log_probs:
        return 0.0 if not target else math.inf
    ext = [0]
    for t in target:
        ext.extend((t, 0))
    alpha = [-math.inf] * len(ext)
    alpha[0] = log_probs[0][0]
    if target:
        alpha[1] = log_probs[0][target[0]]
    for row in log_probs[1:]:
        new = []
        for s, t in enumerate(ext):
            prior = [alpha[s]]
            if s:
                prior.append(alpha[s - 1])
            if s >= 2 and t != 0 and t != ext[s - 2]:
                prior.append(alpha[s - 2])
            new.append(logadd(prior) + row[t])
        alpha = new
    return -logadd(alpha[-2:] if target else alpha[-1:])


def cohort_weights(cohorts):
    require(Counter(cohorts) == Counter({D20: 20, Q12: 12}), 'exact 20+12 cohorts')
    return [1 / 40 if c == D20 else 1 / 24 for c in cohorts]


def cohort_loss(nll, targets, cohorts):
    require(len(nll) == len(targets) == len(cohorts), 'source count')
    weights = cohort_weights(cohorts)
    for loss, target in zip(nll, targets):
        validate_target(target)
        require(len(target) in (3, 4), 'three/four token full transcript')
        require(math.isfinite(loss) and loss >= 0, 'finite source CTC NLL')
    return math.fsum(w * loss / len(y) for w, loss, y in zip(weights, nll, targets))


def train_rows(inventory):
    rows = inventory['rows']
    require(len(rows) == 40 and len({r['recording'] for r in rows}) == 40, '40 unique sources')
    require(Counter(r['role'] for r in rows) == Counter(train=32, development_a=4, development_b=4), 'split')
    selected = [r for r in rows if r['role'] == 'train']
    cohort_weights([r['cohort'] for r in selected])
    require(len({r['pcm_sha256'] for r in rows}) == 40, 'exact PCM duplicates')
    require(set(r['voice'] for r in selected).isdisjoint(r['voice'] for r in rows if r['role'] != 'train'), 'voice overlap')
    require(sum(r['model_rows'] for r in selected) == 1812 and max(r['model_rows'] for r in selected) == 95, 'shape')
    for row in rows:
        require(row['tokens'] == list(TOKENS) and encode(row['text']) == row['target_ids'], 'target mapping')
        require(len(row['target_ids']) in (3, 4), 'target length')
        require(type(row['model_rows']) is int and row['model_rows'] >= minimum_frames(row['target_ids']), 'CTC infeasible')
        p = Path(row['path'])
        require(not p.is_absolute() and '..' not in p.parts, 'relative source path')
    return selected


def validate_feature_manifest(manifest, rows, producer_sha):
    require(manifest['schema'] == 'a20-precmvn-features-v1', 'feature schema')
    require(manifest['input_inventory_sha256'] == INVENTORY_SHA, 'feature inventory binding')
    require(manifest['producer_build_sha256'] == producer_sha, 'feature producer binding')
    require(manifest['normalization'] == 'PRE_CMVN' and manifest['dtype'] == '<f4', 'no second CMVN or dtype conversion')
    require(len(manifest['rows']) == 32, 'feature count')
    for src, feat in zip(rows, manifest['rows']):
        require(feat['recording'] == src['recording'], 'fixed feature source order')
        require(feat['pcm_sha256'] == src['pcm_sha256'] and feat['wav_sha256'] == src['wav_sha256'], 'audio binding')
        require(feat['shape'] == [src['model_rows'], 400], 'feature row shape')
        require(feat['bytes'] == src['model_rows'] * 1600, 'feature bytes')
        require(feat['file'] == src['recording'] + '.f32le', 'feature filename')
        require(type(feat['sha256']) is str and len(feat['sha256']) == 64 and
                all(c in '0123456789abcdef' for c in feat['sha256']), 'feature hash')
    return manifest['rows']
