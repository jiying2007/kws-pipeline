"""Source-only contracts for the R1 x86/ARM numerical replay tools."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class CrossArchSourceContracts(unittest.TestCase):
    def setUp(self):
        path = ROOT / "research/r1_stream_c/cross_arch_validate.py"
        spec = importlib.util.spec_from_file_location("r1_cross_arch_validate", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.dtype = self.module.FRAME_DTYPE

    def capture(self, frames=2):
        rows = np.zeros(frames, dtype=self.dtype)
        rows["frame"] = np.arange(frames, dtype=np.uint64)
        rows["source_end"] = 720 + 480 * np.arange(frames, dtype=np.uint64)
        rows["delivery_end"] = np.array([960, 1280][:frames], dtype=np.uint64)
        rows["logits"][:, 0] = 1.0
        rows["logits"][:, 1] = 0.5
        return rows

    def test_fixed_r1_payload_identities(self):
        self.assertEqual(self.module.MODEL_BYTES, 3028860)
        self.assertEqual(self.module.TABLE_BYTES, 83840)
        self.assertEqual(self.module.CHECKPOINT_SHA256,
                         "99ac2239807ce5a93c8723a01df2e53c446f868245caccb2e7809deb6e943571")

    def test_binary64_numerical_contract_is_fixed(self):
        self.assertEqual(self.module.FEATURE_ATOL, 1e-8)
        self.assertEqual(self.module.FEATURE_RTOL, 1e-8)
        self.assertEqual(self.module.FEATURE_MAX_ABS, 1e-8)
        self.assertEqual(self.module.LOGIT_ATOL, 1e-8)
        self.assertEqual(self.module.LOGIT_RTOL, 1e-8)
        self.assertEqual(self.module.LOGIT_MAX_ABS, 1e-6)
        self.assertEqual(self.module.POSTERIOR_MAX_ABS, 1e-8)

    def test_structured_capture_geometry(self):
        self.assertEqual(self.module.FRAME_BYTES, 24032)
        self.assertEqual(self.dtype.itemsize, 24032)
        self.assertEqual(self.dtype.fields["features"][0].shape, (400,))
        self.assertEqual(self.dtype.fields["logits"][0].shape, (2601,))

    def test_capture_magic_and_exact_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.bin"
            rows = self.capture()
            path.write_bytes(self.module.MAGIC + rows.tobytes())
            actual = self.module.read_capture(path, 2)
            self.assertEqual(actual.shape, (2,))
            self.assertTrue(np.array_equal(actual, rows))

    def test_capture_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.bin"
            path.write_bytes(b"BADMAGIC" + self.capture().tobytes())
            with self.assertRaises(ValueError):
                self.module.read_capture(path, 2)

    def test_capture_rejects_truncated_or_extra_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.bin"
            path.write_bytes(self.module.MAGIC + self.capture().tobytes()[:-1])
            with self.assertRaises(ValueError):
                self.module.read_capture(path, 2)
            path.write_bytes(self.module.MAGIC + self.capture().tobytes() + b"x")
            with self.assertRaises(ValueError):
                self.module.read_capture(path, 2)

    def test_valid_cross_arch_record_passes(self):
        x86 = self.capture()
        arm = self.capture()
        arm["features"][0, 0] = 1e-15
        arm["logits"][1, 1] += 1e-13
        result = self.module.compare_record(x86, arm, 1600)
        self.assertTrue(result["source_clock_equal"])
        self.assertTrue(result["argmax_equal"])
        self.assertLessEqual(result["max_feature_abs_difference"], 1e-8)
        self.assertLessEqual(result["max_logit_abs_difference"], 1e-6)
        self.assertLessEqual(result["max_posterior_abs_difference"], 1e-8)

    def test_feature_gate_rejects_excessive_difference(self):
        x86 = self.capture()
        arm = self.capture()
        arm["features"][0, 0] = 1e-5
        with self.assertRaisesRegex(ValueError, "feature numerical contract"):
            self.module.compare_record(x86, arm, 1600)

    def test_logit_gate_rejects_excessive_difference(self):
        x86 = self.capture()
        arm = self.capture()
        arm["logits"][0, 1] += 1e-4
        with self.assertRaisesRegex(ValueError, "logit numerical contract"):
            self.module.compare_record(x86, arm, 1600)

    def test_source_clock_mismatch_is_rejected(self):
        x86 = self.capture()
        arm = self.capture()
        arm["delivery_end"][1] = 1600
        with self.assertRaisesRegex(ValueError, "clock mismatch"):
            self.module.compare_record(x86, arm, 1600)

    def test_invalid_source_dependency_clock_is_rejected(self):
        x86 = self.capture()
        arm = self.capture()
        x86["source_end"][0] = 721
        arm["source_end"][0] = 721
        with self.assertRaisesRegex(ValueError, "dependency clock"):
            self.module.compare_record(x86, arm, 1600)

    def test_invalid_delivery_clock_is_rejected(self):
        x86 = self.capture()
        arm = self.capture()
        x86["delivery_end"][1] = 1599
        arm["delivery_end"][1] = 1599
        with self.assertRaisesRegex(ValueError, "delivery clock"):
            self.module.compare_record(x86, arm, 1600)

    def test_unknown_eof_padding_is_rejected(self):
        x86 = self.capture()
        arm = self.capture()
        with self.assertRaisesRegex(ValueError, "EOF residual"):
            self.module.compare_record(x86, arm, 1760)

    def test_greedy_sequence_collapses_repeats_and_blank(self):
        logits = np.zeros((7, 2601), dtype=np.float64)
        for row, token in enumerate((1, 1, 0, 2, 2, 0, 1)):
            logits[row, token] = 1.0
        self.assertEqual(self.module.greedy_sequence(logits), [1, 2, 1])

    def test_softmax_is_finite_and_normalized(self):
        logits = np.array([[1000.0, 999.0, -1000.0],
                           [-2.0, 0.0, 2.0]], dtype=np.float64)
        posterior = self.module.stable_softmax(logits)
        self.assertTrue(np.isfinite(posterior).all())
        self.assertTrue(np.allclose(posterior.sum(axis=1), 1.0, atol=1e-15))

    def test_input_manifest_rejects_duplicate_source(self):
        valid = {
            "schema": "kws-r1-arm-input-recovery-v1",
            "records": [
                {"index": 0, "source_id": "same", "split": "train",
                 "samples": 800, "pcm_bytes": 1600, "pcm_sha256": "a" * 64,
                 "feature_shape": [2, 400]},
                {"index": 1, "source_id": "same", "split": "development",
                 "samples": 800, "pcm_bytes": 1600, "pcm_sha256": "b" * 64,
                 "feature_shape": [2, 400]}
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                self.module.load_input_manifest(path)

    def test_input_manifest_rejects_role_leakage(self):
        valid = {
            "schema": "kws-r1-arm-input-recovery-v1",
            "records": [
                {"index": 0, "source_id": "one", "split": "test",
                 "samples": 800, "pcm_bytes": 1600, "pcm_sha256": "a" * 64,
                 "feature_shape": [2, 400]}
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError, "source role"):
                self.module.load_input_manifest(path)

    def test_capture_driver_has_no_audio_padding_or_label_logic(self):
        source = (ROOT / "research/r1_stream_c/cross_arch_capture.c").read_text()
        self.assertIn("r1d_create", source)
        self.assertIn("r1d_feed", source)
        self.assertIn("r1d_finish", source)
        self.assertIn('R1DREF1\\0', source)
        self.assertIn("synthetic_eof", source)
        self.assertNotIn("threshold", source.lower())
        self.assertNotIn("wake event", source.lower())

    def test_recovery_tool_is_offline_and_hash_bound(self):
        source = (ROOT / "research/r1_stream_c/cross_arch_recover_inputs.py").read_text()
        self.assertIn("CAPSULE_SHA256", source)
        self.assertIn("MODEL_CARD_SHA256", source)
        self.assertIn("CHECKPOINT_SHA256", source)
        self.assertIn("binding[\"pcm_sha256\"]", source)
        self.assertNotIn("urllib", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("subprocess", source)
        self.assertIn("raw_audio_published", source)

    def test_recovery_tool_enforces_original_pcm_format(self):
        source = (ROOT / "research/r1_stream_c/cross_arch_recover_inputs.py").read_text()
        self.assertIn("(1, 2, 16000)", source)
        self.assertIn("capsule WAV member hash mismatch", source)
        self.assertIn("output must be a fresh absolute directory", source)

    def test_workflow_is_manual_and_source_only(self):
        workflow = (ROOT / ".github/workflows/r1-reference64-cross-arch.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("push:", workflow)
        self.assertIn("numerics.py --mode reference64", workflow)
        self.assertIn("private_audio_loaded':False", workflow)
        self.assertIn("model_weights_loaded':False", workflow)
        self.assertIn("qemu-arm-static", workflow)


if __name__ == "__main__":
    unittest.main()
