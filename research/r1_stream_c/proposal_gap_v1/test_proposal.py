#!/usr/bin/env python3
"""Fictional tokens only. Running this file does not execute a model or C binary."""
import copy
import json
from pathlib import Path
import unittest

from proposal import Proposal, WORDS, UINT64_MAX, replay


class ProposalTests(unittest.TestCase):
    def test_frozen_single_candidate(self):
        plan = json.loads(Path(__file__).with_name("frozen_candidate.json").read_text())
        self.assertEqual(plan["max_consecutive_blank_frames"], 20)
        self.assertEqual(plan["max_keyword_span_frames"], 150)
        self.assertEqual(plan["same_keyword_cooldown_frames"], 50)
        self.assertTrue(plan["single_candidate_no_gap_sweep"])

    def test_expiry_boundary_both_words(self):
        for word in WORDS:
            for blanks, accepted in ((0, True), (19, True), (20, False), (21, False)):
                tokens = [*word[:2], *([0] * blanks), *word[2:]]
                self.assertEqual(len(replay(tokens, 0)["events"]), 1)
                self.assertEqual(bool(replay(tokens, 20)["events"]), accepted)

    def test_sustained_nonblank_is_not_silence(self):
        tokens = [1027] * 101 + [1401, 1462, 2093]
        self.assertEqual(replay(tokens, 20)["events"], replay(tokens, 0)["events"])
        self.assertEqual(len(replay(tokens, 20)["events"]), 1)

    def test_nonconsecutive_blank_total_does_not_expire(self):
        tokens = [1027] + [0] * 10 + [1401] + [0] * 10 + [1462] + [0] * 10 + [2093]
        self.assertEqual(len(replay(tokens, 20)["events"]), 1)

    def test_span_boundary_unchanged(self):
        for repeats, expected in ((148, 1), (149, 0)):
            tokens = [1027] * repeats + [1401, 1462, 2093]
            self.assertEqual(len(replay(tokens, 20)["events"]), expected)
            self.assertEqual(replay(tokens, 20)["events"], replay(tokens, 0)["events"])

    def test_expiry_preserves_same_word_cooldown(self):
        tokens = list(WORDS[1]) + [0] * 20 + list(WORDS[1])
        result = replay(tokens, 20)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["prefix_expirations"], 1)
        tokens += [0] * 22 + list(WORDS[1])
        self.assertEqual([e["model_frame"] for e in replay(tokens, 20)["events"]], [3, 53])

    def test_keyword_cooldowns_remain_separate(self):
        tokens = list(WORDS[0]) + [0] * 20 + list(WORDS[1])
        self.assertEqual([e["keyword_id"] for e in replay(tokens, 20)["events"]], [1, 2])

    def test_long_blank_run_saturates_and_expires_once(self):
        decoder = Proposal(20)
        for token in [1027] + [0] * 1000:
            decoder.step(token)
        self.assertEqual((decoder.blanks, decoder.expirations), (20, 1))
        self.assertEqual(decoder.rows, 1001)
        self.assertEqual(decoder.ids, [])

    def test_invalid_input_is_atomic(self):
        decoder = Proposal(20)
        decoder.step(1027)
        for token, frame in ((True, 1), (-1, 1), (2601, 1), (0, 0), (0, 2)):
            before = copy.deepcopy(decoder.__dict__)
            with self.assertRaises(ValueError):
                decoder.step(token, frame)
            self.assertEqual(decoder.__dict__, before)
        decoder.rows = UINT64_MAX - 50
        with self.assertRaises(ValueError):
            decoder.step(0)

    def test_invalid_limit_rejected(self):
        for value in (-1, 151, True, 20.0, None):
            with self.assertRaises(ValueError):
                Proposal(value)

    def test_repeated_token_ctc_and_unknown_token_behavior(self):
        self.assertEqual(len(replay([1027, 0, 1027, 1401, 1462, 2093], 20)["events"]), 1)
        self.assertEqual(replay([1027, 1401, 12, 1462, 2093], 20)["events"], [])


if __name__ == "__main__":
    unittest.main()
