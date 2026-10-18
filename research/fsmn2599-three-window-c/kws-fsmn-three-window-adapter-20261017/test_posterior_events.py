import math
import sys
import unittest
from posterior_events import CONFIG, DecoderError, KEYWORDS, PosteriorEvents, _Row


def row(token, probability=.9):
    values = [0.0] * 2599
    values[token] = probability
    values[2] = 1.0 - probability
    return values


def seeded(decoder, tokens, frames, probability=.9):
    decoder.cur_hyps = [(tuple(tokens), (1.0, 0.0, [
        dict(token=t, frame=f, prob=probability) for t, f in zip(tokens, frames)]))]


class PosteriorFixtures(unittest.TestCase):
    def test_01_canonical_score_and_ids(self):
        d = PosteriorEvents('synthetic')
        events = list(d.process(row(x) for x in KEYWORDS[0][1]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['keyword_id'], 1)
        self.assertEqual((events[0]['start_frame'], events[0]['end_frame']), (0, 9))
        self.assertAlmostEqual(events[0]['score'], .81)

    def test_02_second_keyword(self):
        d = PosteriorEvents('synthetic')
        self.assertEqual(list(d.process(row(x) for x in KEYWORDS[1][1]))[0]['keyword_id'], 2)

    def test_03_chunk_partition_invariance(self):
        rows = [row(x) for x in KEYWORDS[0][1]] + [row(0)] * 17 + [
            row(x) for x in KEYWORDS[1][1]]
        reference = PosteriorEvents('synthetic')
        expected = list(reference.process(rows))
        for size in (1, 2, 3, 5, 8, 25):
            d = PosteriorEvents('synthetic')
            events = []
            for start in range(0, len(rows), size):
                events.extend(d.process(rows[start:start + size]))
            self.assertEqual(events, expected)
            self.assertEqual(d.summary(), reference.summary())

    def test_04_two_events_one_chunk_no_dropped_rows(self):
        d = PosteriorEvents('synthetic')
        tokens = list(KEYWORDS[0][1]) + [0] * 17 + list(KEYWORDS[1][1])
        events = list(d.process(row(t) for t in tokens))
        self.assertEqual([e['keyword_id'] for e in events], [1, 2])
        self.assertEqual(d.rows_decoded, len(tokens))
        self.assertEqual(d.total_frames, 3 * len(tokens))

    def test_05_final_suffix_is_found(self):
        d = PosteriorEvents('synthetic')
        seeded(d, (1401,) + KEYWORDS[0][1], (0, 3, 6, 9, 12))
        d.total_frames = 15
        self.assertEqual(d.step(row(0))['start_frame'], 3)

    def test_06_rejected_score_does_not_accumulate(self):
        d = PosteriorEvents('synthetic')
        seeded(d, KEYWORDS[0][1], (0, 1, 2, 3), .8)
        d.total_frames = 6
        self.assertIsNone(d.step(row(0)))  # Too short, independent of theta.
        first = d.hit_score
        self.assertIsNone(d.step(row(0)))
        self.assertAlmostEqual(first, .64)
        self.assertEqual(d.hit_score, first)

    def test_07_refractory_global_and_inclusive(self):
        for end, allowed in ((49, False), (50, True)):
            d = PosteriorEvents('synthetic')
            d.last_active_pos = 0
            seeded(d, KEYWORDS[1][1], (end - 9, end - 6, end - 3, end))
            d.total_frames = end + 1
            self.assertEqual(d.step(row(0)) is not None, allowed)

    def test_08_duration_inclusive(self):
        for duration, allowed in ((4, False), (5, True), (250, True), (251, False)):
            d = PosteriorEvents('synthetic')
            seeded(d, KEYWORDS[0][1], (0, 1, 2, duration))
            d.total_frames = duration + 1
            self.assertEqual(d.step(row(0)) is not None, allowed)

    def test_09_official_expiry_after_row(self):
        d = PosteriorEvents('synthetic')
        d.step(row(1401))
        list(d.process(row(0) for _ in range(82)))
        self.assertEqual(d.total_frames, 249)
        self.assertEqual(d.expiry_resets, 0)
        d.step(row(0))
        self.assertEqual(d.total_frames, 252)
        self.assertEqual(d.expiry_resets, 1)
        self.assertEqual(d.cur_hyps, [((), (1.0, 0.0, []))])

    def test_10_expiry_boundary_can_miss_valid_word(self):
        d = PosteriorEvents('synthetic')
        tokens = [1401] + [0] * 80 + list(KEYWORDS[0][1])
        self.assertEqual(list(d.process(row(x) for x in tokens)), [])
        self.assertEqual(d.expiry_resets, 1)
        fresh = PosteriorEvents('synthetic-control')
        self.assertEqual(len(list(fresh.process(row(x) for x in KEYWORDS[0][1]))), 1)

    def test_11_full_alphabet_top3_blocks_target(self):
        values = [0.] * 2599
        values[1], values[2], values[3], values[1027] = .34, .33, .32, .01
        d = PosteriorEvents('synthetic')
        d.step(values)
        self.assertEqual(d.cur_hyps[0][0], ())

    def test_12_probability_mass_not_renormalized(self):
        values = [0.] * 2599
        values[1], values[2], values[1027] = .50, .44, .06
        d = PosteriorEvents('synthetic')
        d.step(values)
        self.assertEqual(d.cur_hyps[0][1][2][0]['prob'], .06)

    def test_13_fixed_ties_and_probability_floor(self):
        values = [0.] * 2599
        for token in (0, 1027, 1401, 1462):
            values[token] = .25
        self.assertEqual(_Row(values).topk(3)[1], [0, 1027, 1401])
        d = PosteriorEvents('synthetic')
        d.step(row(1027, .05))
        self.assertEqual(d.cur_hyps[0][0], ())

    def test_14_invalid_rows_latch_failure(self):
        invalid = ([0.] * 6, [0.] * 2599, [math.nan] * 2599,
                   [math.inf] + [0.] * 2598, [-.1, 1.1] + [0.] * 2597)
        for values in invalid:
            d = PosteriorEvents('synthetic')
            with self.assertRaises(DecoderError):
                d.step(values)
            self.assertFalse(d.summary()['valid'])
            with self.assertRaises(DecoderError):
                d.step(row(0))

    def test_15_capacity_failure_not_silent_reset(self):
        d = PosteriorEvents('synthetic')
        seeded(d, (1027, 1401) * 64, [0] * 128)
        with self.assertRaisesRegex(DecoderError, 'capacity'):
            d.step(row(1462))
        self.assertFalse(d.summary()['valid'])
        self.assertEqual(d.expiry_resets, 0)
        d.reset_all('new-recording')
        self.assertTrue(d.summary()['valid'])

    def test_16_streaming_iteration_and_no_torch(self):
        d = PosteriorEvents('synthetic')
        events = d.process(row(t) for t in KEYWORDS[0][1])
        self.assertEqual(d.rows_decoded, 0)
        self.assertEqual(next(events)['output_row'], 3)
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(CONFIG['threshold'], 0)

    def test_17_known_empty_prefix_underflow_is_visible(self):
        d = PosteriorEvents('synthetic-known-limitation')
        self.assertEqual(list(d.process(row(0, .5) for _ in range(1100))), [])
        self.assertEqual(d.cur_hyps, [((), (0.0, 0.0, []))])
        self.assertGreater(d.zero_mass_rows, 0)
        self.assertFalse(d.summary()['numerical_support_ok'])
        # Preserved upstream weakness: a node score can fire from zero path mass.
        event = list(d.process(row(t) for t in KEYWORDS[0][1]))[0]
        self.assertAlmostEqual(event['score'], .81)

    def test_18_nonempty_prefix_is_routinely_expired(self):
        d = PosteriorEvents('synthetic')
        self.assertEqual(list(d.process(row(1027 if i % 2 == 0 else 1401)
                                        for i in range(336))), [])
        self.assertEqual(d.expiry_resets, 4)
        self.assertEqual(d.max_prefix_seen, 84)
        self.assertTrue(d.summary()['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
