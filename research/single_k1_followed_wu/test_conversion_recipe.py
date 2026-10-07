import io
import unittest
import wave
import numpy as np
from scipy.io.wavfile import read
from conversion_recipe import native_wav, derivative_wav, quantize_pcm16


class FixedRecipeProof(unittest.TestCase):
    def test_prior_twelve_sample_quantization_fixture(self):
        x = (-1.5, -1., -.5, -3/65536, -1/65536, 0., 1/65536, 3/65536, .5, 1-1/32768, 1., 1.5)
        expected = [-32768, -32768, -16384, -2, -1, 0, 0, 1, 16384, 32767, 32767, 32767]
        self.assertEqual(quantize_pcm16(x).tolist(), expected)

    def test_float_wav_preserves_native_values_in_memory(self):
        x = np.asarray([-1.5, -.25, 0, .25, 1.5], dtype=np.float32)
        rate, got = read(io.BytesIO(native_wav(x)))
        self.assertEqual(rate, 44100)
        self.assertEqual(got.dtype, x.dtype)
        self.assertEqual(got.tobytes(), x.tobytes())

    def test_fixed_ratio_and_canonical_pcm_geometry_in_memory(self):
        x = np.zeros(441, dtype=np.float32)
        x[220] = .5
        raw, receipt = derivative_wav(x)
        with wave.open(io.BytesIO(raw)) as f:
            self.assertEqual((f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()), (16000, 1, 2, 160))
        self.assertEqual(len(raw), 44 + 2 * 160)
        self.assertEqual(receipt['derivative_frames'], 160)
        self.assertFalse(receipt['gain_normalization'])
        self.assertFalse(receipt['trimmed'])

    def test_invalid_outputs_are_rejected_without_crop(self):
        for x in (np.array([], dtype=np.float32), np.array([np.nan], dtype=np.float32),
                  np.zeros(441001, dtype=np.float32), np.zeros((2, 10), dtype=np.float32)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                derivative_wav(x)


if __name__ == '__main__':
    unittest.main()
