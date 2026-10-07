"""Source-retained original decoder on canonical probabilities, without Torch.

The fixed six-class selector schedule is reconstructed. Historical Torch selector
checks survive only where restored evidence explicitly supports them; this new
source/runtime requires independent recovery review.
"""
import ast
import collections
import copy
from fractions import Fraction
import logging
import math
import sys
import numpy as np
from assets import ROOT, PINS, sha

ORIGINAL = '../kws-alternate-fsmn-precheck-v1/reference/upstream/wekws/bin/stream_kws_ctc.py'
SUFFIX_FIX = '../kws-alternate-fsmn-precheck-v1/decoder_tail_fix.py'


def top3(values):
    """Fixed k3/n6 nth_element at2 followed by strict-value sort[0,2)."""
    assert len(values) == 6 and all(math.isfinite(v) for v in values)
    order, low, high, depth = list(range(6)), 0, 6, 4
    def greater(a, b):
        return values[a] > values[b]
    while high - low > 3:
        assert depth > 0, 'unreviewed heap fallback'
        depth -= 1
        a, b, c = low + 1, low + (high - low) // 2, high - 1
        if greater(order[a], order[b]):
            median = b if greater(order[b], order[c]) else (c if greater(order[a], order[c]) else a)
        else:
            median = a if greater(order[a], order[c]) else (c if greater(order[b], order[c]) else b)
        order[low], order[median] = order[median], order[low]
        first, last, pivot = low + 1, high, order[low]
        while True:
            while greater(order[first], pivot):
                first += 1
            last -= 1
            while greater(pivot, order[last]):
                last -= 1
            if first >= last:
                break
            order[first], order[last] = order[last], order[first]
            first += 1
        if first <= 2:
            low = first
        else:
            high = first
    for i in range(low + 1, high):
        value, j = order[i], i
        while j > low and greater(value, order[j - 1]):
            order[j] = order[j - 1]
            j -= 1
        order[j] = value
    if greater(order[1], order[0]):
        order[0], order[1] = order[1], order[0]
    return order[:3]


class _List(list):
    def tolist(self):
        return list(self)


class _Scalar(float):
    def item(self):
        return float(self)


class _Row(list):
    def topk(self, count):
        assert count == 3
        indices = top3(self)
        return _List(self[i] for i in indices), _List(indices)

    def __getitem__(self, index):
        return _Scalar(super().__getitem__(index))


def retained_methods():
    path = ROOT / ORIGINAL
    assert sha(path) == PINS[ORIGINAL]
    source = ast.parse(path.read_text())
    beam = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'ctc_prefix_beam_search')
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'KeyWordSpotter')
    names = {'decode_keywords', 'execute_detection', 'reset'}
    methods = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    forward = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
    # Remove only original frontend/model/softmax preparation; retain the entire
    # original post-softmax decode/detect/reset/clock/expiry/return AST unchanged.
    assert len(forward.body) == 10 and isinstance(forward.body[6], ast.For)
    assert isinstance(forward.body[7], ast.AugAssign) and isinstance(forward.body[8], ast.If)
    forward.name = 'process_probs'
    forward.args.args[1].arg = 'probs'
    forward.body = forward.body[6:]
    methods.append(forward)
    fix_path = ROOT / SUFFIX_FIX
    assert sha(fix_path) == PINS[SUFFIX_FIX]
    fix = next(n for n in ast.parse(fix_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'is_sublist')
    namespace = dict(math=math, logging=logging, defaultdict=collections.defaultdict)
    tree = ast.Module(body=[beam, fix, *methods], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path) + '[retained-decoder-only]', 'exec'), namespace)
    return namespace


class CanonicalDecoder:
    def __init__(self):
        assert sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
        methods = retained_methods()
        self._process = methods['process_probs']
        for name in ['execute_detection', 'reset']:
            setattr(self, name, methods[name].__get__(self))
        self._decode = methods['decode_keywords'].__get__(self)
        self.downsampling, self.resolution = 3, .01
        self.score_beam, self.path_beam = 3, 20
        self.threshold, self.min_frames, self.max_frames, self.interval_frames = 0.0, 5, 250, 50
        self.keywords_idxset = set(range(6))
        self.keywords_token = {'你好小窝': {'token_id': (1, 2, 3, 4)},
                               '小窝小窝': {'token_id': (3, 4, 3, 4)}}
        self.reset_all()

    def reset_all(self):
        self.reset()
        self.total_frames, self.last_active_pos, self.result = 0, -1, {}
        self.rows_decoded = 0

    def decode_keywords(self, time, probabilities):
        self.rows_decoded += 1
        self._decode(time, probabilities)

    def process(self, probabilities):
        assert isinstance(probabilities, np.ndarray) and probabilities.dtype == np.float32
        assert probabilities.ndim == 2 and probabilities.shape[1] == 6
        assert np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
        for row in probabilities:
            total = sum((Fraction.from_float(float(v)) for v in row), Fraction(0))
            assert abs(total - 1) <= Fraction(1, 100000)
        self.rows_decoded = 0
        returned = ({} if len(probabilities) == 0 else
                    dict(self._process(self, [_Row(map(float, row)) for row in probabilities])))
        return dict(return_value=returned, rows_decoded=self.rows_decoded,
                    decoder_total_frames=self.total_frames, last_active_pos=self.last_active_pos,
                    hit_score=self.hit_score, eof_flush=False, decoder_state=copy.deepcopy(self.cur_hyps))
