#!/usr/bin/env python3
"""Metadata/PCM fixtures and optional small-logit math; never create a model."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_c1_known_silence as silence


class MetadataTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_every_row_has_the_same_predeclared_cyclic_exposure(self):
        rows = [{"source_id": "fixture-%d" % i, "ctc_weight": .5} for i in range(8)]
        draws = silence.exposure_plan(rows)
        self.assertEqual(len(draws), 8000)
        self.assertEqual(Counter(d["offset_ms"] for d in draws), {0: 2672, 200: 2664, 500: 2664})
        for i in range(8):
            row = draws[i::8]
            self.assertEqual([d["row_visit_index"] for d in row], list(range(1000)))
            self.assertEqual(Counter(d["offset_ms"] for d in row), {0: 334, 200: 333, 500: 333})
            self.assertTrue(all(d["row_index"] == i and d["source_id"] == rows[i]["source_id"] and d["ctc_weight"] == .5 for d in row))
            self.assertTrue(all(d["known_zero_prefix_frames"] == {0: 0, 200: 9, 500: 24}[d["offset_ms"]] for d in row))

    def test_known_zero_geometry_excludes_the_window_crossing_original_audio(self):
        for offset, frames in ((0, 0), (200, 9), (500, 24)):
            shape = silence.onset.geometry(16000, offset)
            self.assertEqual(shape["fully_digital_silence_frames"], frames)
            if frames:
                self.assertLessEqual((frames - 1) * 320 + 400, shape["added_samples"])
                self.assertGreater(frames * 320 + 400, shape["added_samples"])

    def test_sweeps_weighting_and_budget_expansions_are_rejected(self):
        original = {"schema": silence.SCHEMA, **silence.fixed_contract()}
        silence.validate_contract(original)
        for key, value in (("optimizer_steps", 8000), ("auxiliary_scale", .5), ("auxiliary_scale", 1),
                ("auxiliary_uses_speech_weak_weight", True), ("before_forward_calls", 24),
                ("training_forward_limit", 8001), ("final_forward_limit", 48), ("automatic_retry", True),
                ("offsets_ms", [False, 200, 500]), ("single_variable_ablation", True)):
            changed = copy.deepcopy(original); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                silence.validate_contract(changed)

    def test_mechanism_gate_uses_every_silent_frame_and_keeps_tiny_fit_separate(self):
        rows = [{"source_id": "fixture-%d" % i, "offset_ms": offset,
            "known_zero_prefix_frames": {0: 0, 200: 9, 500: 24}[offset], "known_zero_nonblank_frames": 0,
            "exact_token_match": i < 5} for i in range(8) for offset in (0, 200, 500)]
        conditions = {str(offset): {"exact_token_matches": 5, "edit_numerator": 7} for offset in (0, 200, 500)}
        with patch.object(silence.onset, "summarize", side_effect=lambda _: copy.deepcopy(conditions)):
            result = silence.decisions(rows)
            self.assertTrue(result["minimum_mechanism_gate_passed"])
            self.assertFalse(result["strict_tiny_fit_passed"])
            rows[2]["known_zero_nonblank_frames"] = 1
            self.assertFalse(silence.decisions(rows)["minimum_mechanism_gate_passed"])
            rows[2]["known_zero_nonblank_frames"] = 0
            conditions["500"]["edit_numerator"] = 8
            self.assertFalse(silence.decisions(rows)["minimum_mechanism_gate_passed"])

    def test_direct_unsupervised_run_fails_before_output_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); recipe = root / "recipe.json"; lock = root / "fixture.lock"
            silence.io.write_json(recipe, {"schema": silence.SCHEMA, **silence.fixed_contract(), "heavy_lock": str(lock)})
            with self.assertRaisesRegex(ValueError, "D1_HASH_BOUND_DURABLE_QUEUE_REQUIRED"):
                silence.run(recipe, root / "output", lock)
            self.assertFalse((root / "output").exists())
            self.assertNotIn("torch", sys.modules)


class TensorLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The original deterministic one-thread CPU setup defines classes only;
        # these fixtures construct no encoder, checkpoint, optimizer or model.
        cls.torch, _ = silence.training.torch_runtime()

    def logits(self):
        return self.torch.linspace(-.8, .9, 30 * 2601).reshape(30, 1, 2601).requires_grad_()

    def test_known_prefix_equals_empty_target_ctc_in_value_and_gradient(self):
        t = self.torch
        for count in (9, 24):
            with self.subTest(count=count):
                x = self.logits(); y = x.detach().clone().requires_grad_()
                actual = silence.prefix_blank_ctc(x, count)
                expected = t.nn.functional.ctc_loss(y[:count].log_softmax(2), t.empty(0, dtype=t.long),
                    t.tensor([count]), t.tensor([0]), blank=0, reduction="none", zero_infinity=False)[0] / count
                self.assertTrue(t.allclose(actual, expected, rtol=1e-6, atol=1e-6))
                gx, = t.autograd.grad(actual, x); gy, = t.autograd.grad(expected, y)
                self.assertTrue(t.allclose(gx, gy, rtol=3e-4, atol=1e-6))
                self.assertEqual(int(t.count_nonzero(gx[count:])), 0)
                self.assertGreater(int(t.count_nonzero(gx[:count])), 0)

    def test_zero_offset_has_zero_auxiliary_value_and_gradient(self):
        x = self.logits(); loss = silence.prefix_blank_ctc(x, 0)
        gradient, = self.torch.autograd.grad(loss, x)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(int(self.torch.count_nonzero(gradient)), 0)

    def test_mean_normalization_does_not_amplify_longer_silence(self):
        x = self.torch.zeros(30, 1, 2601); x[:, 0, 0] = 1.0
        self.assertTrue(self.torch.allclose(silence.prefix_blank_ctc(x, 9), silence.prefix_blank_ctc(x, 24), rtol=1e-6, atol=1e-6))
        observed = silence.prefix_observation(x, 24)
        self.assertEqual(observed["known_zero_nonblank_frames"], 0)
        self.assertGreater(observed["prefix_blank_probability_minimum"], 0)

    def test_invalid_masks_cannot_touch_other_original_audio_frames(self):
        x = self.logits()
        for count in (-1, 1, 10, 25, True, 9.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                silence.prefix_blank_ctc(x, count)


if __name__ == "__main__":
    numeric = "--tensor-fixtures" in sys.argv
    if numeric:
        sys.argv.remove("--tensor-fixtures")
    unittest.main(defaultTest="TensorLossTests" if numeric else "MetadataTests")
