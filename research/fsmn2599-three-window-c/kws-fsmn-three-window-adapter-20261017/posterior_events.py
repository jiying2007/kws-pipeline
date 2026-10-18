"""Candidate-event adapter; stdlib only, no acoustic model or file decoder.

Full 2599-class normalized posteriors are required. theta=0 is diagnostic only.
The retained beam is source-pinned; documented event-loop fixes are intentional.
"""
import ast
import hashlib
import heapq
import logging
import math
from collections import defaultdict
from pathlib import Path

CONFIG = {
    'classes': 2599, 'blank': 0, 'score_beam': 3, 'path_beam': 20,
    'token_probability_floor_exclusive': 0.05, 'threshold': 0.0,
    'min_frames': 5, 'max_frames': 250, 'interval_frames': 50,
    'downsampling': 3, 'resolution': 0.01, 'prefix_cap': 128,
    'topk_ties': 'ascending_token_id', 'mode': 'candidate_events_only',
    'expiry': 'after_each_row_best_first_node_age_gt_250_clear_all_hypotheses',
}
PINS = {
    'stream_kws_ctc.py': '2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53',
    'decoder_tail_fix.py': '5fc55921a555c78cb08c12018850652d98d4283594f875366284dc1df0e71f4d',
}
KEYWORDS = (('你好小窝', (1027, 1401, 1462, 2093)),
            ('小窝小窝', (1462, 2093, 1462, 2093)))


class DecoderError(RuntimeError):
    """The recording is invalid; earlier emitted candidate events are not final."""


class _List(list):
    def tolist(self):
        return list(self)


class _Scalar(float):
    def item(self):
        return float(self)


class _Row(tuple):
    def __getitem__(self, index):
        return _Scalar(super().__getitem__(index))

    def topk(self, count):
        if count != 3:
            raise DecoderError('fixed score beam must equal three')
        indices = heapq.nsmallest(count, range(len(self)),
                                  key=lambda i: (-self[i], i))
        return _List(self[i] for i in indices), _List(indices)


def _retained():
    trees = {}
    for name, expected in PINS.items():
        data = (Path(__file__).parent / 'vendor' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise DecoderError('source identity mismatch: ' + name)
        trees[name] = ast.parse(data)
    source = trees['stream_kws_ctc.py']
    beam = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                and n.name == 'ctc_prefix_beam_search')
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef)
               and n.name == 'KeyWordSpotter')
    detect = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'execute_detection')
    # Only this retained method is patched: each attempted detection starts at 1.
    detect.body.insert(0, ast.parse('self.hit_score = 1.0').body[0])
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'reset')
    suffix = next(n for n in trees['decoder_tail_fix.py'].body
                  if isinstance(n, ast.FunctionDef) and n.name == 'is_sublist')
    tree = ast.fix_missing_locations(ast.Module(body=[beam, suffix, detect, reset],
                                               type_ignores=[]))
    namespace = dict(math=math, logging=logging, defaultdict=defaultdict)
    exec(compile(tree, 'pinned-wekws[posterior-only]', 'exec'), namespace)
    return namespace


class PosteriorEvents:
    """One instance per head; same fixed rule. No stored posterior/event history.

    reset_all is a real-recording boundary only. step consumes exactly one row;
    process is a lazy iterator over candidate events, not an all-stream list.
    Any DecoderError invalidates the entire recording until explicit reset_all.
    """
    def __init__(self, recording_id):
        methods = _retained()
        self._beam = methods['ctc_prefix_beam_search']
        self._detect = methods['execute_detection'].__get__(self)
        self._reset = methods['reset'].__get__(self)
        for name in ('threshold', 'min_frames', 'max_frames', 'interval_frames',
                     'downsampling', 'resolution', 'score_beam', 'path_beam'):
            setattr(self, name, CONFIG[name])
        self.keywords_token = {word: {'token_id': ids} for word, ids in KEYWORDS}
        self.keywords_idxset = {0, 1027, 1401, 1462, 2093}
        self.reset_all(recording_id)

    def reset_all(self, recording_id):
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError('a nonempty recording identity is required')
        self.recording_id = recording_id
        self._reset()
        self.total_frames, self.last_active_pos, self.result = 0, -1, {}
        self.rows_decoded = self.events_emitted = self.expiry_resets = 0
        self.max_prefix_seen = self.max_next_hypotheses = 0
        self.zero_mass_rows = 0
        self.invalid_reason = None

    def _check_hypotheses(self, hypotheses, max_length, max_count):
        if len(hypotheses) > max_count:
            raise DecoderError('hypothesis count exceeded static bound')
        for prefix, (pb, pnb, nodes) in hypotheses:
            if len(prefix) != len(nodes) or len(prefix) > max_length:
                raise DecoderError('prefix/node capacity or alignment failure')
            if not (math.isfinite(pb) and math.isfinite(pnb) and pb >= 0 and pnb >= 0):
                raise DecoderError('invalid beam probability')

    def step(self, probabilities):
        if self.invalid_reason is not None:
            raise DecoderError('recording already invalid: ' + self.invalid_reason)
        try:
            if len(probabilities) != CONFIG['classes']:
                raise DecoderError('expected the full 2599-class posterior')
            row = _Row(float(p) for p in probabilities)
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in row):
                raise DecoderError('nonfinite or out-of-range posterior')
            if abs(math.fsum(row) - 1.0) > 1e-5:
                raise DecoderError('posterior must already sum to one; no renormalization')
            self._check_hypotheses(self.cur_hyps, CONFIG['prefix_cap'], self.path_beam)
            next_hyps = self._beam(self.total_frames, row, self.cur_hyps,
                                   self.keywords_idxset, self.score_beam)
            self._check_hypotheses(next_hyps, CONFIG['prefix_cap'], 80)
            self.max_next_hypotheses = max(self.max_next_hypotheses, len(next_hyps))
            self.cur_hyps = next_hyps[:self.path_beam]
            if self.cur_hyps and not any(pb + pnb > 0 for _, (pb, pnb, _) in self.cur_hyps):
                self.zero_mass_rows += 1  # Observability only; no new normalization.
            self.max_prefix_seen = max(self.max_prefix_seen,
                                       max((len(x[0]) for x in self.cur_hyps), default=0))
            self._detect(0)
            event = None
            if self.activated:
                start = round(self.result['start'] / self.resolution)
                end = round(self.result['end'] / self.resolution)
                event = dict(recording_id=self.recording_id,
                             event_index=self.events_emitted,
                             keyword_id=1 if self.result['keyword'] == KEYWORDS[0][0] else 2,
                             keyword=self.result['keyword'], score=self.result['score'],
                             start_frame=start, end_frame=end,
                             decision_frame=self.total_frames,
                             output_row=self.rows_decoded, grid_seconds=self.resolution,
                             diagnostic_candidate=True)
                self.events_emitted += 1
                self._reset()  # Preserve global clock and last_active_pos.
            self.rows_decoded += 1
            self.total_frames += self.downsampling
            # Retained official best-path expiry, now on a fixed one-row tick.
            if self.cur_hyps and self.cur_hyps[0][0]:
                first = int(self.cur_hyps[0][1][2][0]['frame'])
                if self.total_frames - first > self.max_frames:
                    self._reset()
                    self.expiry_resets += 1
            return event
        except (DecoderError, TypeError, ValueError, OverflowError) as exc:
            self.invalid_reason = str(exc)
            raise DecoderError(self.invalid_reason) from exc

    def process(self, rows):
        for row in rows:
            event = self.step(row)
            if event is not None:
                yield event

    def summary(self):
        return dict(recording_id=self.recording_id, rows_decoded=self.rows_decoded,
                    total_frames=self.total_frames, events_emitted=self.events_emitted,
                    expiry_resets=self.expiry_resets, max_prefix_seen=self.max_prefix_seen,
                    max_next_hypotheses=self.max_next_hypotheses,
                    zero_mass_rows=self.zero_mass_rows,
                    numerical_support_ok=self.zero_mass_rows == 0,
                    valid=self.invalid_reason is None, invalid_reason=self.invalid_reason)
