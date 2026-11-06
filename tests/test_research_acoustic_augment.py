import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import research_acoustic_augment as a


class PolicyTest(unittest.TestCase):
    def parent(self):
        return {"ctc_mask": True, "ctc_weight": 0.5, "target_ids": [1, 2],
                "full_text": "你好", "keyword_mask": [True, True],
                "keyword_targets": [1, 0], "keyword_weight": [0.5, 0.5]}

    def test_noise_never_creates_blank_ctc_or_absent_keyword_truth(self):
        parent = self.parent()
        original = copy.deepcopy(parent)
        observed = a.inherited_labels(parent, True)
        self.assertIs(observed["ctc_mask"], False)
        self.assertEqual(observed["ctc_weight"], 0.0)
        self.assertIsNone(observed["target_ids"])
        self.assertIsNone(observed["full_text"])
        self.assertEqual(observed["keyword_mask"], [True, False])
        self.assertEqual(observed["keyword_targets"], [1, None])
        self.assertEqual(parent, original)

    def test_pure_acoustic_variant_retains_complete_target(self):
        row = a.inherited_labels(self.parent(), False)
        self.assertEqual(row["target_ids"], [1, 2])
        self.assertEqual(row["full_text"], "你好")
        self.assertIs(row["human_gold"], False)

    def test_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNSAFE"):
            a.safe_path(ROOT, "../elsewhere")


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
                     "numerical runtime unavailable")
class NumericalTest(unittest.TestCase):
    def test_snr_uses_active_speech_and_common_gain_preserves_ratio(self):
        import numpy as np
        clean = np.r_[np.zeros(320), np.full(640, 0.9), np.zeros(320)]
        noise = np.full(192, 0.8)
        mixed, metrics = a.noise_mix(clean, noise, 0, 0)
        self.assertEqual(metrics["active_samples"], 640)
        self.assertAlmostEqual(metrics["measured_component_snr_db"], 0.0, places=10)
        self.assertLess(metrics["common_peak_gain"], 1)
        self.assertLessEqual(float(np.max(np.abs(mixed))), 0.98)
        self.assertTrue(metrics["noise_tiled"])

    def test_all_requested_snr_values(self):
        import numpy as np
        rng = np.random.default_rng(5)
        for snr in (0, 5, 10, 20):
            _, metrics = a.noise_mix(rng.normal(0, 0.15, 2000),
                                    rng.normal(0, 0.2, 1000), snr, 5)
            self.assertAlmostEqual(metrics["measured_component_snr_db"], snr, places=8)

    def test_impulse_delay_and_full_tail(self):
        import numpy as np
        clean = np.array([0.1, 0.2, 0.05])
        rir = np.array([0., 0., 1., 0.5])
        rendered, metrics = a.convolve(clean, rir)
        np.testing.assert_allclose(rendered, np.convolve(clean, rir / np.linalg.norm(rir)), atol=1e-12)
        self.assertEqual(len(rendered), len(clean) + len(rir) - 1)
        self.assertEqual(metrics["rir_onset_samples"], 2)
        self.assertIsNone(metrics["physical_distance_m"])

    def test_low_level_is_preserved_and_not_silence_filtered(self):
        import numpy as np
        tiny = np.array([0.0001, -0.0001] * 320)
        body, _ = a.wav_bytes(tiny)
        decoded = a.audio(body)
        self.assertEqual(decoded.shape, (640, 1))
        self.assertGreater(float(np.max(np.abs(decoded))), 0)

    def test_nonfinite_and_zero_noise_fail(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError, "ZERO_ENERGY_NOISE"):
            a.noise_mix(np.ones(640), np.zeros(640), 10, 0)
        with self.assertRaisesRegex(ValueError, "INVALID_SAMPLES"):
            a.wav_bytes(np.array([float("nan")]))


if __name__ == "__main__":
    unittest.main()
