"""No-numpy, no-Torch, no-audio source-only checks for measured PCM parity."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import pcm_frontend_parity as pcm

HERE = Path(__file__).resolve().parent


class EnglishPCMFrontendSourceCases(unittest.TestCase):
    def test_retained_real_audio_receipt_is_source_bound_and_nonpromotable(self):
        measured = json.loads((HERE/'PCM_FRONTEND_PARITY.json').read_text())
        self.assertTrue(pcm.verify_receipt(measured))
        self.assertEqual([r['chunk_samples'] for r in measured['plans']],
                         list(pcm.CHUNKS))
        self.assertEqual(sum(r['clips'] for r in measured['plans']), 315)
        self.assertFalse(measured['model_inference_executed'])

    def test_forged_result_fields_fail_closed_without_audio(self):
        measured = json.loads((HERE/'PCM_FRONTEND_PARITY.json').read_text())
        changes = (
            lambda r: r.__setitem__('model_promotion_allowed', True),
            lambda r: r.__setitem__('continuous_wake_detector_tested', True),
            lambda r: r.__setitem__('formal_training_admitted', True),
            lambda r: r.__setitem__('model_inference_executed', True),
            lambda r: r.__setitem__('runner_sha256', '0'*64),
            lambda r: r.__setitem__('source_review_json_sha256', '0'*64),
            lambda r: r['plans'][0].__setitem__('max_abs_logmel_error', 0.02),
            lambda r: r['plans'][1].__setitem__('frames', 13011),
            lambda r: r['plans'][2].__setitem__('max_buffer_samples', 9999),
            lambda r: r['plans'][3].__setitem__('clips', 62),
        )
        for change in changes:
            tampered = copy.deepcopy(measured)
            change(tampered)
            with self.assertRaises(ValueError):
                pcm.verify_receipt(tampered)

    def test_streaming_pcm_frame_geometry_with_hostile_boundaries(self):
        wave = list(range(9000))
        expected = [wave[n:n+pcm.FRAME]
                    for n in range(0, len(wave)-pcm.FRAME+1, pcm.HOP)]
        for chunk in (1, 159, 160, 161, 3199):
            framer = pcm.FrameBuffer()
            actual = []
            for start in range(0, len(wave), chunk):
                actual.extend(framer.push(wave[start:start+chunk]))
            self.assertEqual(actual, expected)
            self.assertEqual(framer.finish(), len(expected))
            self.assertLess(framer.maximum_buffer_samples,
                            pcm.FRAME + chunk)
        for invalid in ([], [0]*3200):
            with self.assertRaises(ValueError):
                pcm.FrameBuffer().push(invalid)
        partial = pcm.FrameBuffer()
        partial.push([0]*399)
        with self.assertRaises(ValueError):
            partial.finish()


if __name__ == '__main__':
    unittest.main()
