import itertools
import hashlib
import math
import sys
import unittest
from pathlib import Path
from posterior_events import DecoderError, KEYWORDS, PosteriorEvents
from test_posterior_events import row
from three_window_events import CONFIG, ThreeWindowEvents, _SharedRow


def candidate(keyword, frames, score, start=0, decision=100):
    return dict(keyword_id=keyword, keyword=KEYWORDS[keyword - 1][0],
                node_frames=tuple(frames), start_frame=frames[0], end_frame=frames[-1],
                score=score, window_start=start, decision_row=decision,
                decision_frame=3 * decision)


class ThreeWindowFixtures(unittest.TestCase):
    def test_01_canonical_score_and_absolute_time(self):
        d = ThreeWindowEvents('fixture')
        events = list(d.process(row(t) for t in [0] * 90 + list(KEYWORDS[0][1])))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['node_frames'], (270, 273, 276, 279))
        self.assertAlmostEqual(events[0]['score'], .81)
        self.assertEqual(d.duplicate_candidates, 2)

    def test_02_old_expiry_counterexample_is_covered(self):
        tokens = [1401] + [0] * 80 + list(KEYWORDS[0][1])
        old = PosteriorEvents('fixture')
        self.assertEqual(list(old.process(row(t) for t in tokens)), [])
        d = ThreeWindowEvents('fixture')
        events = list(d.process(row(t) for t in tokens))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['node_frames'], (243, 246, 249, 252))
        self.assertEqual(events[0]['decision_row'], 84)

    def test_03_long_blank_does_not_zero_beam_mass(self):
        d = ThreeWindowEvents('fixture')
        self.assertEqual(list(d.process(row(0, .5) for _ in range(1100))), [])
        self.assertEqual(d.zero_mass_window_rows, 0)
        self.assertEqual(d.empty_beam_window_rows, 0)
        self.assertTrue(all(w.cur_hyps[0][1][0] > 0 for w in d.windows))
        events = list(d.process(row(t) for t in KEYWORDS[0][1]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['node_frames'], (3300, 3303, 3306, 3309))

    def test_04_geometry_all_42_phases(self):
        for phase in range(42):
            for span in range(2, 84):
                start = 84 + phase
                window = start // 42 * 42
                self.assertLessEqual(window, start)
                self.assertLessEqual(start + span, window + 125)

    def test_05_actual_maximum_span_at_boundary_phases(self):
        for phase in (0, 1, 40, 41):
            d = ThreeWindowEvents('fixture')
            start = 84 + phase
            positions = (start, start + 27, start + 55, start + 83)
            by_row = dict(zip(positions, KEYWORDS[0][1]))
            events = list(d.process(row(by_row.get(i, 0))
                                    for i in range(positions[-1] + 1)))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['node_frames'], tuple(3 * x for x in positions))
            self.assertEqual(events[0]['end_frame'] - events[0]['start_frame'], 249)

    def test_06_chunk_partition_invariance(self):
        tokens = [0] * 40 + list(KEYWORDS[0][1]) + [0] * 38 + list(KEYWORDS[1][1])
        rows = [row(t) for t in tokens]
        ref = ThreeWindowEvents('fixture')
        expected = list(ref.process(rows))
        expected_summary = ref.finish()
        for size in (1, 7, 42, 50, 126):
            d = ThreeWindowEvents('fixture')
            events = []
            for start in range(0, len(rows), size):
                events.extend(d.process(rows[start:start + size]))
            self.assertEqual(events, expected)
            self.assertEqual(d.finish(), expected_summary)

    def test_07_start_and_partial_eof(self):
        d = ThreeWindowEvents('fixture')
        self.assertEqual(d.windows_started, 0)
        self.assertEqual(list(d.process(row(t) for t in KEYWORDS[0][1][:3])), [])
        before = d.rows_decoded
        summary = d.finish()
        self.assertEqual(summary['rows_decoded'], before)
        self.assertEqual(summary['eof_windows_closed'], 1)
        self.assertEqual(summary['live_windows'], 0)
        self.assertEqual(summary['events_emitted'], 0)
        self.assertEqual(d.finish(), summary)
        with self.assertRaisesRegex(DecoderError, 'EOF'):
            d.step(row(2093))

    def test_08_window_retirement_and_beam_update_budget(self):
        d = ThreeWindowEvents('fixture')
        list(d.process(row(0) for _ in range(125)))
        self.assertEqual([w.start for w in d.windows], [0, 42, 84])
        d.step(row(0))
        self.assertEqual([w.start for w in d.windows], [42, 84])
        self.assertEqual(d.windows_retired, 1)
        d.step(row(0))
        self.assertEqual([w.start for w in d.windows], [42, 84, 126])
        self.assertEqual(d.beam_updates, 3 * 127 - 126)
        self.assertEqual(d.topk_computations, 127)

    def test_09_exact_duplicate_prefers_score_then_window(self):
        a = candidate(1, (270, 273, 276, 279), .7, 0)
        b = candidate(1, (270, 273, 276, 279), .8, 42)
        c = candidate(1, (270, 273, 276, 279), .8, 0)
        outputs = []
        for items in itertools.permutations((a, b, c)):
            d = ThreeWindowEvents('fixture')
            outputs.append(d._merge(items, 100))
            self.assertEqual(d.duplicate_candidates, 2)
        self.assertTrue(all(x == outputs[0] for x in outputs))
        self.assertEqual(outputs[0][0]['score'], .8)
        self.assertEqual(outputs[0][0]['window_start'], 0)

    def test_10_same_frame_multiple_distinct_events(self):
        a = candidate(1, (0, 3, 6, 9), .9)
        b = candidate(2, (60, 63, 66, 69), .8)
        for items in ((a, b), (b, a)):
            d = ThreeWindowEvents('fixture')
            events = d._merge(items, 100)
            self.assertEqual([e['end_frame'] for e in events], [9, 69])
            self.assertEqual([e['decision_frame'] for e in events], [300, 300])

    def test_11_same_frame_sorting_precedes_refractory(self):
        a = candidate(1, (0, 3, 6, 9), .8)
        b = candidate(2, (60, 63, 66, 69), .9)
        d = ThreeWindowEvents('fixture')
        self.assertEqual([e['end_frame'] for e in d._merge((a, b), 100)], [69])
        self.assertEqual(d.refractory_rejections, 1)

    def test_12_shared_refractory_and_local_candidate_consumption(self):
        d = ThreeWindowEvents('fixture')
        tokens = [0] * 90 + list(KEYWORDS[0][1]) + list(KEYWORDS[1][1])
        tokens += [0] * 13 + list(KEYWORDS[1][1])
        events = list(d.process(row(t) for t in tokens))
        self.assertEqual([e['keyword_id'] for e in events], [1, 2])
        self.assertEqual([e['end_frame'] for e in events], [279, 342])
        self.assertGreater(d.refractory_rejections, 0)

    def test_13_cache_is_computed_once_and_not_mutable_by_consumers(self):
        shared = _SharedRow(row(1027, .8))
        p, ids = shared.topk(3)
        original = (tuple(p), tuple(ids))
        p[0] = 0.; ids[0] = 2598; p.append(9.)
        for _ in range(3):
            again = shared.topk(3)
            self.assertEqual((tuple(again[0]), tuple(again[1])), original)
        self.assertEqual(shared.topk_computations, 1)

    def test_14_full_class_top3_and_no_renormalization(self):
        values = [0.] * 2599
        values[1], values[2], values[1027] = .5, .44, .06
        d = ThreeWindowEvents('fixture')
        d.step(values)
        self.assertEqual(d.windows[0].cur_hyps[0][1][2][0]['prob'], .06)
        values[1], values[2], values[3], values[1027] = .34, .33, .32, .01
        other = ThreeWindowEvents('fixture')
        other.step(values)
        self.assertEqual(other.windows[0].cur_hyps[0][0], ())

    def test_15_state_is_bounded_on_long_nonkeyword_prefix(self):
        d = ThreeWindowEvents('fixture')
        self.assertEqual(list(d.process(row(1027 if i % 2 == 0 else 1401)
                                        for i in range(1000))), [])
        self.assertEqual(d.max_prefix_seen, 126)
        self.assertEqual(d.max_live_windows, 3)
        self.assertLessEqual(d.max_live_node_references, 3 * 20 * 126)
        self.assertLessEqual(d.max_next_hypotheses, 80)
        self.assertEqual(d.beam_updates, 2874)
        self.assertEqual(d.topk_computations, 1000)
        self.assertEqual(d.windows_started, d.windows_retired + len(d.windows))

    def test_16_dedup_history_expires_and_has_fixed_bound(self):
        d = ThreeWindowEvents('fixture')
        for n in range(4, 504):
            end = 3 * n
            candidates = [candidate(1, (end-9, end-6, end-3, end), .8, decision=n),
                          candidate(2, (end-9, end-6, end-3, end), .8, decision=n),
                          candidate(1, (end-12, end-6, end-3, end), .8, decision=n)]
            d._merge(candidates, n)
        self.assertEqual(len(d._seen), 378)
        self.assertEqual(len(d._seen_heap), 378)
        self.assertEqual(d.max_dedup_keys, 378)

    def test_17_invalid_input_and_real_meeting_reset(self):
        d = ThreeWindowEvents('fixture')
        with self.assertRaises(DecoderError):
            d.step([math.nan] * 2599)
        self.assertFalse(d.summary()['valid'])
        d.reset_all('new-meeting')
        events = list(d.process(row(t) for t in KEYWORDS[0][1]))
        self.assertEqual(events[0]['recording_id'], 'new-meeting')
        self.assertEqual(events[0]['event_index'], 0)
        self.assertEqual(events[0]['node_frames'], (0, 3, 6, 9))

    def test_18_empty_meeting_and_no_torch(self):
        d = ThreeWindowEvents('fixture')
        self.assertEqual(list(d.process([])), [])
        self.assertEqual(d.finish()['windows_started'], 0)
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(CONFIG['threshold'], 0)

    def test_19_full_beam_staging_is_bounded(self):
        values = [0.] * 2599
        values[0], values[1027], values[1401] = .4, .3, .3
        d = ThreeWindowEvents('fixture-full-beam')
        self.assertEqual(list(d.process(values for _ in range(260))), [])
        self.assertTrue(any(len(w.cur_hyps) == 20 for w in d.windows))
        self.assertLessEqual(d.max_next_hypotheses, 80)
        self.assertLessEqual(d.max_live_node_references, 7560)
        self.assertLessEqual(d.max_prefix_seen, 126)

    def test_20_v1_source_remains_byte_identical(self):
        source = Path(__file__).with_name('posterior_events.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '56d6ada2c531de25f72a713db77b07ef9d8326b8b1e238bd776d40155d8584ec')

    def test_21_later_higher_duplicate_does_not_rewrite_event(self):
        d = ThreeWindowEvents('fixture')
        first = candidate(1, (270, 273, 276, 279), .7)
        later = candidate(1, (270, 273, 276, 279), .9, 42, 101)
        self.assertEqual(d._merge([first], 100)[0]['score'], .7)
        self.assertEqual(d._merge([later], 101), [])
        self.assertEqual(d.duplicate_candidates, 1)

    def test_22_cross_keyword_score_tie_is_fixed(self):
        first = candidate(1, (270, 273, 276, 279), .8)
        second = candidate(2, (270, 273, 276, 279), .8)
        for order in ((first, second), (second, first)):
            d = ThreeWindowEvents('fixture')
            self.assertEqual([e['keyword_id'] for e in d._merge(order, 100)], [1])
            self.assertEqual(d.refractory_rejections, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
