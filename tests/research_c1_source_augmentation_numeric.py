#!/usr/bin/env python3
"""Tiny invented-logit checks only; no model construction/forward/optimizer."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_c1_source_augmentation as stage
from test_research_c1_source_augmentation import fixture


class TinyNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

    def test_complete_observed_score_keeps_weight_and_full_target(self):
        torch = self.torch
        vocabulary, original = fixture()
        row = stage.project_row(original, vocabulary)
        feature = torch.zeros((row["expected_c1_frames"], 32), dtype=torch.float32)
        logits = torch.zeros((len(feature), 1, 2601), dtype=torch.float32)
        logits[:, 0, 0] = 4
        for i, token in enumerate(row["target_ids"]):
            logits[5 + 3 * i, 0, token] = 9
        model_pin = {"model_id": "invented", "checkpoint": {"sha256": "0" * 64}}
        with patch.object(stage.training, "build_fresh", side_effect=AssertionError("NO_MODEL")):
            item = stage.observed_score(row, model_pin, b"\0\0" * row["samples"], 0, feature, logits,
                logits, feature, 14, {}, vocabulary)
        self.assertEqual(item["evaluation_status"], "evaluated")
        self.assertEqual(item["target_text"], original["full_text"])
        self.assertEqual(item["target_ids"], row["target_ids"])
        self.assertAlmostEqual(item["ctc_loss_confidence_weighted"], item["ctc_loss_target_length_normalized"] * .5)
        self.assertTrue(item["keywords"][0]["complete_sequence_present"])

    def test_oov_numeric_result_keeps_known_keyword_but_no_fulltext_loss(self):
        torch = self.torch
        vocabulary, original = fixture(text="你好小窝龘")
        row = stage.project_row(original, vocabulary)
        feature = torch.zeros((row["expected_c1_frames"], 32), dtype=torch.float32)
        logits = torch.zeros((len(feature), 1, 2601), dtype=torch.float32)
        logits[:, 0, 0] = 4
        model_pin = {"model_id": "invented", "checkpoint": {"sha256": "0" * 64}}
        item = stage.observed_score(row, model_pin, b"\0\0" * row["samples"], 0, feature, logits,
            logits, feature, 14, {}, vocabulary)
        self.assertEqual(item["evaluation_status"], "not-evaluable")
        self.assertEqual(item["full_text"], original["full_text"])
        self.assertEqual(item["observed_label"]["ctc_weight"], .5)
        self.assertNotIn("ctc_loss_target_length_normalized", item)
        self.assertTrue(item["keywords"][0]["known"])
        self.assertEqual(item["keywords"][0]["reference"], 1)


if __name__ == "__main__":
    unittest.main()
