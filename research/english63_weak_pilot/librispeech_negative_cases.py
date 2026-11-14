"""Invented no-model/no-audio hostile tests for independently sourced speech."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import librispeech_negative_events as neg


def invented_receipt():
    names=[]
    for i in range(10000):
        path="LibriSpeech/dev-clean/1993/147149/1993-147149-"+str(i)+".flac"
        if neg.selected(path):
            names.append(path)
        if len(names)==400:
            break
    assert len(names)==400
    return {
        "schema":neg.POLICY,
        "status":"MEASURED_REAL_READ_SPEECH_SEGMENT_EVENTS_NOT_FIELD_FAR",
        "archive_sha256":neg.ARCHIVE_SHA,
        "model_sha256":neg.quant.MODEL_SHA,
        "runner_sha256":neg.quant.sha256_bytes(Path(neg.__file__).read_bytes()),
        "subset_rule":"SHA256(member_name) first8 big-endian mod5 == 0",
        "license":"CC-BY-4.0",
        "sampled_utterances":len(names),
        "speaker_count":40,
        "source_audio_seconds":4000,
        "models_tested":list(neg.MODES),
        "events_by_mode":{x:0 for x in neg.MODES},
        "records":[{
            "member":name,
            "flac_sha256":"a"*64,
            "transcript_sha256":"b"*64,
            "sample_count":160000,
            "speaker_source_id":"1993",
            "events":{x:0 for x in neg.MODES},
            "human_gold":False,
            "trusted_negative":False,
            "training_admitted":False,
        } for name in names],
        "new_asr_tts_training_calls":0,
        "human_acoustic_gold":False,
        "exact_source_transcripts_exclude_requested_words":True,
        "continuous_environment_far_per_hour":"NOT_RUN",
        "human_frr":"NOT_RUN",
        "target_ssc305":"NOT_RUN",
        "model_promotion_allowed":False,
    }


class EnglishNaturalSpeechSourceCases(unittest.TestCase):
    def test_fixed_hash_sampling_and_transcript_exclusion(self):
        a="LibriSpeech/dev-clean/1993/147149/1993-147149-2.flac"
        self.assertEqual(neg.selected(a),neg.selected(a))
        self.assertNotEqual(neg.selected(a),None)
        for path in ("../data.flac","LibriSpeech/test-clean/1/2/3.flac","/root/a.flac"):
            with self.assertRaises(ValueError):
                neg.selected(path)
        self.assertTrue(neg.no_requested_wake("THE CHILD WAS LOOKING OUTSIDE"))
        self.assertFalse(neg.no_requested_wake("  hey, MERO!  "))
        self.assertFalse(neg.no_requested_wake("oh, Hey Vosen, stop"))
        self.assertTrue(neg.no_requested_wake("HERO MERIT"))
        self.assertEqual(neg.MODEL_SHA if hasattr(neg,"MODEL_SHA")
                         else neg.quant.MODEL_SHA,
                         neg.quant.MODEL_SHA)

    def test_synthetic_report_remains_nonpromotable(self):
        self.assertTrue(neg.verify_receipt(invented_receipt()))

    def test_unsafe_transcript_or_nongold_source_fails_closed(self):
        r=invented_receipt()
        changes=(
            lambda v:v.__setitem__("archive_sha256","0"*64),
            lambda v:v.__setitem__("model_sha256","0"*64),
            lambda v:v.__setitem__("runner_sha256","0"*64),
            lambda v:v.__setitem__("model_promotion_allowed",True),
            lambda v:v.__setitem__("human_acoustic_gold",True),
            lambda v:v.__setitem__("exact_source_transcripts_exclude_requested_words",False),
            lambda v:v.__setitem__("continuous_environment_far_per_hour",0.0),
            lambda v:v.__setitem__("target_ssc305","PASS"),
            lambda v:v.__setitem__("new_asr_tts_training_calls",1),
            lambda v:v.__setitem__("sampled_utterances",399),
            lambda v:v.__setitem__("speaker_count",0),
            lambda v:v.__setitem__("source_audio_seconds",0),
            lambda v:v.__setitem__("models_tested",["int8_c16"]),
            lambda v:v["records"][0].__setitem__("trusted_negative",True),
            lambda v:v["records"][0].__setitem__("training_admitted",True),
            lambda v:v["records"][0].__setitem__("human_gold",True),
            lambda v:v["records"][0].__setitem__("events",{"fp32_c16":1,"int8_c16":0}),
            lambda v:v["records"][0].__setitem__("member","LibriSpeech/test-clean/a.flac"),
        )
        for change in changes:
            broken=copy.deepcopy(r)
            change(broken)
            with self.assertRaises(ValueError):
                neg.verify_receipt(broken)


if __name__=="__main__":
    unittest.main()
