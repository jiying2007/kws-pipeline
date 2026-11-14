"""No-audio, no-model negative tests for original PCM versus artificial 660ms tail."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pretrained_live_timing as timing


def invented():
    rows=[]
    for i in range(63):
        rows.append({
            "cell_id":"en-"+format(i,"024x"),
            "source_family_id":"qwen3-preset-dylan",
            "requested_text":"Hey Mero","requested_intent":"wake",
            "source_pcm_samples":33361+(17 if i==0 else 0),
            "pcm_sha256":"a"*64,
            "original_with_tail_event_count":int(i<9),
            "original_after_audio_event_count":int(i<4),
            "events":[],"decode_steps":1,
            "human_gold":False,"training_admitted":False,
        })
    return {
        "schema":timing.POLICY,
        "status":"MEASURED_UNPADDED_SYNTHETIC_SOURCE_EVENT_DIAGNOSTIC",
        "runner_sha256":timing.sha(Path(timing.__file__).read_bytes()),
        "baseline_artifact_sha256":timing.BASELINE_ARTIFACT_SHA,
        "source_tts_sha256":timing.original.source.TTS_SHA256,
        "source_review_sha256":timing.original.source.REVIEW_SHA256,
        "model_sha256":timing.original.MODEL_SHA,
        "source_clips":63,"input_pcm_samples":timing.original.EXPECTED_PCM_SAMPLES,
        "modes":list(timing.MODES),
        "by_mode":{mode:{
            "saved_baseline_json_sha256":timing.BASELINE_SHA[mode],
            "original_with_artificial_tail_events":9,
            "original_after_real_pcm_events":4,
            "raw_pcm_events":0,
            "eof_only_events":0,
            "records":copy.deepcopy(rows),
            "model_promotion_allowed":False,
        } for mode in timing.MODES},
        "new_model_training_or_audio_generation":False,
        "human_gold":False,
        "continuous_household_far_per_hour":"NOT_RUN",
        "independent_human_frr":"NOT_RUN",
        "end_of_word_latency_p99":"NOT_RUN",
        "ssc305_qualified":False,
        "model_promotion_allowed":False,
    }


class EnglishUnpaddedTimingSourceCases(unittest.TestCase):
    def test_model_free_safe_receipt_and_inference_scope(self):
        self.assertTrue(timing.verify_retained_receipt(invented()))
        self.assertEqual(timing.MODES,("fp32_c16","int8_c16"))
        self.assertEqual(timing.INPUT_HOP_SAMPLES,1600)
        self.assertEqual(timing.EXPECTED_PREVIOUS_EVENTS,9)

    def test_forged_qualification_source_or_timing_never_pass(self):
        base=invented()
        changes=(
            lambda r:r.__setitem__("runner_sha256","0"*64),
            lambda r:r.__setitem__("source_tts_sha256","0"*64),
            lambda r:r.__setitem__("source_review_sha256","0"*64),
            lambda r:r.__setitem__("model_promotion_allowed",True),
            lambda r:r.__setitem__("human_gold",True),
            lambda r:r.__setitem__("continuous_household_far_per_hour",0.0),
            lambda r:r.__setitem__("end_of_word_latency_p99",0.0),
            lambda r:r.__setitem__("ssc305_qualified",True),
            lambda r:r.__setitem__("new_model_training_or_audio_generation",True),
            lambda r:r["by_mode"]["fp32_c16"].__setitem__("raw_pcm_events",1),
            lambda r:r["by_mode"]["fp32_c16"].__setitem__("original_after_real_pcm_events",3),
            lambda r:r["by_mode"]["fp32_c16"]["records"][0].__setitem__("training_admitted",True),
            lambda r:r["by_mode"]["fp32_c16"]["records"][0].__setitem__("source_pcm_samples",999),
            lambda r:r["by_mode"]["fp32_c16"]["records"][0].__setitem__("original_with_tail_event_count",0),
            lambda r:r["by_mode"]["int8_c16"]["records"][0].__setitem__("events",
                [{"phase":"artificial_tail","keyword":"HEY_MERO","input_samples_available":33361}]),
            lambda r:r["by_mode"]["int8_c16"]["records"][0].__setitem__("cell_id",
                r["by_mode"]["int8_c16"]["records"][1]["cell_id"]),
        )
        for mutation in changes:
            bad=copy.deepcopy(base)
            mutation(bad)
            with self.assertRaises(ValueError):
                timing.verify_retained_receipt(bad)

    def test_unpinned_reference_refuses_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"invented-old-artifact.zip"
            path.write_bytes(b"fake-zip-not-source")
            with self.assertRaisesRegex(ValueError,"EN_LIVE_BASELINE_ARTIFACT_DRIFT"):
                timing.source_locked_baselines(path)


    def test_actual_original_event_receipt_is_hash_bound_and_nonpromotable(self):
        report=json.loads((HERE/"UNPADDED_LIVE_RESULT.json").read_text())
        self.assertEqual(report["schema"],
                         "english63-unpadded-event-measured-summary-v1")
        self.assertEqual(report["status"],
                         "ACTUAL_MODEL_REPLAY_VERIFIED_NOT_QUALIFIED")
        self.assertEqual(report["execution"]["runner_sha256"],
                         timing.sha(Path(timing.__file__).read_bytes()))
        self.assertEqual(report["source"]["original_audio_zip_sha256"],
                         timing.original.source.TTS_SHA256)
        self.assertEqual(report["source"]["review_json_sha256"],
                         timing.original.source.REVIEW_SHA256)
        self.assertEqual(report["source"]["official_pretrained_model_archive_sha256"],
                         timing.original.MODEL_SHA)
        self.assertEqual(report["source"]["prior_event_artifact_sha256"],
                         timing.BASELINE_ARTIFACT_SHA)
        self.assertEqual(report["source"]["clip_count"],63)
        self.assertEqual(report["source"]["source_pcm_samples"],2101760)
        self.assertEqual(len(report["original_pcm_event_cell_ids"]),5)
        self.assertEqual(len(report["original_extra_tail_dependent_event_cell_ids"]),4)
        self.assertFalse(set(report["original_pcm_event_cell_ids"]) &
                         set(report["original_extra_tail_dependent_event_cell_ids"]))
        for mode in timing.MODES:
            numbers=report["results"][mode]
            self.assertEqual(numbers["previous_with_660ms_tail"],9)
            self.assertEqual(numbers["prior_only_after_real_audio"],4)
            self.assertEqual(numbers["original_pcm_events_without_tail"],5)
            self.assertEqual(numbers["eof_only_events_without_padding"],0)
        self.assertFalse(report["interpretation"]["model_promotion_allowed"])
        self.assertFalse(report["interpretation"]["natural_acoustic_truth"])
        self.assertEqual(report["interpretation"]["continuous_environment_far_per_hour"],
                         "NOT_RUN")
        self.assertFalse(report["interpretation"]["true_end_of_word_latency_measured"])


if __name__=="__main__":
    unittest.main()
