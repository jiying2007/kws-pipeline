#!/usr/bin/env python3
"""No Torch/model imports: PCM, window, fixed-budget and saved-input boundaries."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_c1_onset_diagnostic as onset


class OnsetContractTests(unittest.TestCase):
    def test_import_has_no_torch_or_models(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_pcm_is_prepended_without_trimming_or_gain(self):
        original = b"\x00\x80\xff\x7f\x00\x00\xff\xff" * 200
        for offset, samples, frames in ((0, 0, 0), (200, 3200, 10), (500, 8000, 25)):
            padded, shape = onset.prepend_pcm(original, offset)
            self.assertEqual(padded[:samples * 2], b"\x00\x00" * samples)
            self.assertEqual(padded[samples * 2:], original)
            self.assertEqual(shape["shift_frames"], frames)
            self.assertEqual(shape["frames"], (len(original) // 2 - 400) // 320 + 1 + frames)
        self.assertEqual(onset.prepend_pcm(original, 0)[0], original)

    def test_partial_frontend_window_is_not_counted_as_pure_silence(self):
        self.assertEqual(onset.geometry(800, 200)["fully_digital_silence_frames"], 9)
        self.assertEqual(onset.geometry(800, 500)["fully_digital_silence_frames"], 24)
        # Frame 9 starts at 180 ms and ends at 205 ms, crossing original PCM onset.
        events = onset.emission_locations([0] * 8 + [1, 2, 3], 200)
        self.assertEqual([e["window_entirely_in_added_silence"] for e in events], [True, False, False])
        self.assertAlmostEqual(events[1]["original_window_start_seconds"], -0.020)
        self.assertAlmostEqual(events[1]["original_window_end_seconds"], 0.005)
        self.assertEqual(events[2]["original_window_start_seconds"], 0.0)

    def test_ctc_blank_allows_repeated_token_emission(self):
        events = onset.emission_locations([0, 4, 4, 0, 4, 5, 5], 0)
        self.assertEqual([(e["token_id"], e["frame_index"]) for e in events], [(4, 1), (4, 4), (5, 5)])

    def test_unsupported_offsets_and_malformed_pcm_fail(self):
        for offset in (True, 200.0, -20, 20, 201, 1000):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                onset.prepend_pcm(b"\0\0" * 800, offset)
        for pcm in (b"", b"\0" * 801, bytearray(b"\0\0" * 800), b"\0\0" * 399):
            with self.assertRaises(ValueError):
                onset.prepend_pcm(pcm, 200)

    def test_budgets_and_baseline_reuse_cannot_silently_expand(self):
        original = {"schema": onset.SCHEMA, **onset.fixed_contract()}
        onset.validate_contract(original)
        for key, value in (("new_forward_limit", 24), ("optimizer_steps", 1), ("worker_seconds", 301),
                ("queue_minutes", 6), ("automatic_retry", True), ("reset_each_condition", False),
                ("baseline", "new-forward"), ("offsets_ms", [False, 200, 500])):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                onset.validate_contract(changed)

    def test_input_byte_change_and_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "final.pt"
            payload.write_bytes(b"invented fixture, not a checkpoint")
            onset.io.write_json(root / "MANIFEST.json", {"final.pt": onset.io.file_record(payload)})
            digest = hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()
            onset.verify_manifest(root, digest)
            payload.write_bytes(b"changed fixture")
            with self.assertRaisesRegex(ValueError, "ONSET_FILE_CHANGED"):
                onset.verify_manifest(root, digest)
            (root / "unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ONSET_FILE_ROSTER_CHANGED"):
                onset.verify_manifest(root, digest)

    def test_incomplete_observation_set_fails_before_summary(self):
        with self.assertRaisesRegex(ValueError, "ONSET_OBSERVATION_ORDER_OR_COUNT"):
            onset.summarize([])

    def test_direct_unqueued_run_fails_before_output_or_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, recipe, output = root / "fixture.lock", root / "recipe.json", root / "output"
            onset.io.write_json(recipe, {"schema": onset.SCHEMA, **onset.fixed_contract(), "heavy_lock": str(lock)})
            with self.assertRaisesRegex(ValueError, "D1_HASH_BOUND_DURABLE_QUEUE_REQUIRED"):
                onset.run(recipe, output, lock)
            self.assertFalse(output.exists())
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
