"""Fixed 126/42-row wrapper with three explicit source-pinned beam corrections.

Synthetic-posterior preparation only. No acoustic model, audio I/O or runner.
V1 remains available as posterior_events.py, byte-identical to its delivered copy.
"""
import heapq
import math
from posterior_events import CONFIG as V1_CONFIG, DecoderError, KEYWORDS, _Row, _List, _retained
from beam_minimal_patch_candidate import patched_methods

CONFIG = dict(V1_CONFIG, window_rows=126, hop_rows=42, live_windows=3,
              expiry='fixed_window_lifetime_only',
              merge_order='score_desc,end_asc,keyword_id_asc,window_start_asc,node_frames_asc',
              duplicate_key='keyword_id_and_four_absolute_node_frames',
              local_reset='every_duration_valid_candidate_before_global_merge',
              refractory='shared_end_frame_difference_at_least_50',
              beam_patch='two_strict_positive_mass_branches_and_one_last_node_copy_on_write',
              mode='candidate_events_only_v2_not_admitted_for_real_data')


class _SharedRow(_Row):
    """Compute the full-alphabet top3 once; expose fresh list views to callers."""
    def __new__(cls, probabilities):
        obj = super().__new__(cls, probabilities)
        obj._cache = None
        obj.topk_computations = 0
        return obj

    def topk(self, count):
        if count != 3:
            raise DecoderError('fixed score beam must equal three')
        if self._cache is None:
            probs, ids = super().topk(count)
            self._cache = (tuple(probs), tuple(ids))
            self.topk_computations += 1
        return _List(self._cache[0]), _List(self._cache[1])


class _Window:
    def __init__(self, start, methods):
        self.start = start
        self.stop = start + CONFIG['window_rows']
        self._beam = methods['ctc_prefix_beam_search']
        self._detect = methods['execute_detection'].__get__(self)
        self._reset = methods['reset'].__get__(self)
        self._suffix = methods['is_sublist']
        self.threshold, self.min_frames, self.max_frames = 0.0, 5, 250
        self.interval_frames, self.resolution = 0, .01
        self.keywords_token = {word: {'token_id': ids} for word, ids in KEYWORDS}
        self.keywords_idxset = {0, 1027, 1401, 1462, 2093}
        self.last_active_pos, self.total_frames = -1, 3 * start
        self.result = {}
        self._reset()

    def step(self, row_index, row):
        if not self.start <= row_index < self.stop:
            raise DecoderError('window used outside its fixed lifetime')
        self.total_frames = 3 * row_index
        next_hyps = self._beam(self.total_frames, row, self.cur_hyps,
                               self.keywords_idxset, CONFIG['score_beam'])
        if len(next_hyps) > 80:
            raise DecoderError('temporary hypothesis bound exceeded')
        for prefix, (pb, pnb, nodes) in next_hyps:
            if len(prefix) != len(nodes) or len(prefix) > row_index - self.start + 1:
                raise DecoderError('window prefix length/alignment invariant failed')
            if not (math.isfinite(pb) and math.isfinite(pnb) and pb >= 0 and pnb >= 0):
                raise DecoderError('invalid beam probability')
        self.cur_hyps = next_hyps[:CONFIG['path_beam']]
        metrics = dict(next_hypotheses=len(next_hyps),
                       prefix_length=max((len(x[0]) for x in self.cur_hyps), default=0),
                       empty_beam=not self.cur_hyps,
                       zero_mass=bool(self.cur_hyps) and
                       not any(pb + pnb > 0 for _, (pb, pnb, _) in self.cur_hyps))
        self._detect(0)
        if not self.activated:
            return None, metrics
        candidate = None
        # Recover the same first path/keyword selected by retained detection.
        for prefix, (_, _, nodes) in self.cur_hyps:
            for keyword_id, (word, labels) in enumerate(KEYWORDS, 1):
                offset = self._suffix(prefix, labels)
                if offset != -1:
                    selected = nodes[offset:offset + len(labels)]
                    frames = tuple(int(node['frame']) for node in selected)
                    candidate = dict(keyword_id=keyword_id, keyword=word,
                                     node_frames=frames, start_frame=frames[0],
                                     end_frame=frames[-1], score=float(self.hit_score),
                                     window_start=self.start, decision_row=row_index,
                                     decision_frame=3 * row_index)
                    break
            if candidate is not None:
                break
        if candidate is None:
            raise DecoderError('retained detector activated without candidate nodes')
        self._reset()  # Consume local candidate even if global refractory rejects it.
        return candidate, metrics


class ThreeWindowEvents:
    def __init__(self, recording_id):
        self._methods = patched_methods()
        self.reset_all(recording_id)

    def reset_all(self, recording_id):
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError('a nonempty real-recording identity is required')
        self.recording_id = recording_id
        self.windows = []
        self._seen, self._seen_heap = set(), []
        self.last_accepted_end = -1
        self.rows_decoded = self.beam_updates = self.topk_computations = 0
        self.windows_started = self.windows_retired = self.eof_windows_closed = 0
        self.candidates_seen = self.duplicate_candidates = self.refractory_rejections = 0
        self.events_emitted = self.zero_mass_window_rows = self.empty_beam_window_rows = 0
        self.max_live_windows = self.max_prefix_seen = self.max_next_hypotheses = 0
        self.max_dedup_keys = self.max_live_node_references = 0
        self.finished = False
        self.invalid_reason = None

    @staticmethod
    def _order(candidate):
        return (-candidate['score'], candidate['end_frame'], candidate['keyword_id'],
                candidate['window_start'], candidate['node_frames'])

    def _merge(self, candidates, row_index):
        current_frame = 3 * row_index
        while self._seen_heap and self._seen_heap[0][0] <= current_frame - 378:
            _, key = heapq.heappop(self._seen_heap)
            self._seen.remove(key)
        # Stable winner: highest score, earliest window start for an identical key.
        unique = {}
        for c in sorted(candidates, key=self._order):
            key = (c['keyword_id'], c['node_frames'])
            if key in unique or key in self._seen:
                self.duplicate_candidates += 1
                continue
            unique[key] = c
        events = []
        for key, c in sorted(unique.items(), key=lambda item: self._order(item[1])):
            self._seen.add(key)
            heapq.heappush(self._seen_heap, (c['end_frame'], key))
            if self.last_accepted_end != -1 and c['end_frame'] - self.last_accepted_end < 50:
                self.refractory_rejections += 1
                continue
            self.last_accepted_end = c['end_frame']
            event = dict(c, recording_id=self.recording_id, event_index=self.events_emitted,
                         diagnostic_candidate=True, grid_seconds=.01)
            self.events_emitted += 1
            events.append(event)
        if len(self._seen) > 378:
            raise DecoderError('deduplication bound exceeded')
        self.max_dedup_keys = max(self.max_dedup_keys, len(self._seen))
        return events

    def step(self, probabilities):
        if self.invalid_reason is not None:
            raise DecoderError('recording already invalid: ' + self.invalid_reason)
        try:
            if self.finished:
                raise DecoderError('posterior supplied after real-recording EOF')
            if len(probabilities) != 2599:
                raise DecoderError('expected full 2599-class posterior')
            row = _SharedRow(float(p) for p in probabilities)
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in row):
                raise DecoderError('nonfinite or out-of-range posterior')
            if abs(math.fsum(row) - 1.) > 1e-5:
                raise DecoderError('posterior must sum to one; no renormalization')
            n = self.rows_decoded
            if n % 42 == 0:
                self.windows.append(_Window(n, self._methods))
                self.windows_started += 1
            if len(self.windows) > 3:
                raise DecoderError('live window bound exceeded')
            self.max_live_windows = max(self.max_live_windows, len(self.windows))
            candidates = []
            for window in self.windows:
                candidate, metrics = window.step(n, row)
                self.beam_updates += 1
                self.zero_mass_window_rows += int(metrics['zero_mass'])
                self.empty_beam_window_rows += int(metrics['empty_beam'])
                self.max_prefix_seen = max(self.max_prefix_seen, metrics['prefix_length'])
                self.max_next_hypotheses = max(self.max_next_hypotheses, metrics['next_hypotheses'])
                if candidate is not None:
                    candidates.append(candidate)
            self.topk_computations += row.topk_computations
            if row.topk_computations != 1:
                raise DecoderError('global top3 must be computed exactly once per row')
            self.max_live_node_references = max(self.max_live_node_references,
                sum(len(hyp[0]) for window in self.windows for hyp in window.cur_hyps))
            self.candidates_seen += len(candidates)
            events = self._merge(candidates, n)
            self.rows_decoded += 1
            active = [w for w in self.windows if w.stop > self.rows_decoded]
            self.windows_retired += len(self.windows) - len(active)
            self.windows = active
            return events
        except (DecoderError, TypeError, ValueError, OverflowError) as exc:
            self.invalid_reason = str(exc)
            raise DecoderError(self.invalid_reason) from exc

    def process(self, rows):
        for row in rows:
            yield from self.step(row)

    def finish(self):
        """Close partial windows only. No posterior padding or new events."""
        if self.invalid_reason is not None:
            raise DecoderError('cannot finish an invalid recording')
        if not self.finished:
            self.eof_windows_closed += len(self.windows)
            self.windows.clear()
            self._seen.clear()
            self._seen_heap.clear()
            self.finished = True
        return self.summary()

    def summary(self):
        names = ('rows_decoded', 'beam_updates', 'topk_computations', 'windows_started',
                 'windows_retired', 'eof_windows_closed', 'candidates_seen',
                 'duplicate_candidates', 'refractory_rejections', 'events_emitted',
                 'zero_mass_window_rows', 'empty_beam_window_rows', 'max_live_windows',
                 'max_prefix_seen', 'max_next_hypotheses', 'max_dedup_keys',
                 'max_live_node_references', 'finished', 'invalid_reason')
        result = {name: getattr(self, name) for name in names}
        result.update(recording_id=self.recording_id, live_windows=len(self.windows),
                      dedup_keys=len(self._seen), valid=self.invalid_reason is None,
                      numerical_support_ok=self.zero_mass_window_rows == 0
                      and self.empty_beam_window_rows == 0)
        return result
