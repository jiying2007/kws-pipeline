"""No-model, no-audio hostile native continuous audiobook source tests."""
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
import english_librispeech_contiguous_c as study


def synthetic_receipt():
    names=[]
    for i in range(5000):
        person=str(1000+i%40)
        name="LibriSpeech/dev-clean/"+person+"/1/"+person+"-1-"+str(i)+".flac"
        if study.selected(name):
            names.append(name)
        if len(names)==study.SOURCE_CLIPS:
            break
    assert len(names)==study.SOURCE_CLIPS
    count,remaining=divmod(study.SOURCE_PCM_SAMPLES,study.SOURCE_CLIPS)
    roster=[]
    for i,name in enumerate(names):
        person=study.NAME.fullmatch(name).group(1)
        roster.append({
            "member":name,
            "speaker_source_id":person,
            "flac_sha256":"a"*64,
            "transcript_sha256":"b"*64,
            "sample_count":count+int(i<remaining),
            "human_gold":False,
            "trusted_negative":False,
            "training_admitted":False,
        })
    assert len({r["speaker_source_id"] for r in roster})==40
    sessions={}
    for order in study.ORDERS:
        order_list=study.ordered_members(roster,order)
        for partition in study.CHUNK_PLANS:
            sessions[partition+"__"+order]={
                "ordered_source_sha256":study.order_digest(order_list),
                "processed_samples":study.SOURCE_PCM_SAMPLES,
                "recordings":study.SOURCE_CLIPS,
                "event_count":0,"eof_only_events":0,
                "near_splice_event_count_200ms":0,
                "events":[],
                "no_interclip_padding":True,
                "no_interclip_reset":True,
            }
    return {
        "schema":study.POLICY,
        "status":"MEASURED_CONTIGUOUS_REAL_READ_SPEECH_EVENTS_NOT_HOUSEHOLD_FAR",
        "runner_sha256":study.digest(Path(study.__file__).read_bytes()),
        "dataset_archive_sha256":study.ARCHIVE_SHA,
        "segmented_reference_artifact_sha256":study.BASELINE_ZIP_SHA,
        "segmented_reference_json_sha256":study.BASELINE_JSON_SHA,
        "original_model_archive_sha256":study.native.quant.MODEL_SHA,
        "original_host_runtime_archive_sha256":study.native.host.ARCHIVE_SHA,
        "data_license":study.LICENSE,
        "selected_utterances":study.SOURCE_CLIPS,
        "natural_speaker_count":study.SOURCE_SPEAKERS,
        "source_samples_per_session":study.SOURCE_PCM_SAMPLES,
        "source_audio_sec_per_session":study.SOURCE_PCM_SAMPLES/16000,
        "model_mode":"int8_c16","keyword":study.KEYWORD,"keyword_id":3,
        "orders":list(study.ORDERS),"pcm_partitions":list(study.CHUNK_PLANS),
        "no_pcm_inserted_between_recordings":True,
        "stream_reset_between_recordings":False,
        "source_roster":roster,"actual_sessions":sessions,
        "new_training_tts_asr_calls":0,
        "human_acoustic_gold":False,
        "continuous_household_far_per_hour":"NOT_RUN",
        "independent_human_frr":"NOT_RUN",
        "ssc305_target_qualification":"NOT_RUN",
        "model_promotion_allowed":False
    }


class EnglishNaturalContiguousHostSourceCases(unittest.TestCase):
    def test_fixed_one_hour_original_source_and_quarantined_scope(self):
        good=synthetic_receipt()
        self.assertTrue(study.verify_result(good))
        self.assertEqual(study.SOURCE_PCM_SAMPLES,58646164)
        self.assertEqual(study.SOURCE_CLIPS,517)
        self.assertEqual(study.SOURCE_SPEAKERS,40)
        self.assertEqual(len(good["actual_sessions"]),4)
        self.assertTrue(good["no_pcm_inserted_between_recordings"])
        self.assertFalse(good["stream_reset_between_recordings"])
        self.assertEqual(good["continuous_household_far_per_hour"],"NOT_RUN")

    def test_source_or_scientific_promotion_tampering_fails_closed(self):
        base=synthetic_receipt()
        corruptions=(
            lambda r:r.__setitem__("runner_sha256","0"*64),
            lambda r:r.__setitem__("dataset_archive_sha256","0"*64),
            lambda r:r.__setitem__("segmented_reference_artifact_sha256","0"*64),
            lambda r:r.__setitem__("segmented_reference_json_sha256","0"*64),
            lambda r:r.__setitem__("original_model_archive_sha256","0"*64),
            lambda r:r.__setitem__("original_host_runtime_archive_sha256","0"*64),
            lambda r:r.__setitem__("data_license","unknown"),
            lambda r:r.__setitem__("source_samples_per_session",58646163),
            lambda r:r.__setitem__("natural_speaker_count",39),
            lambda r:r.__setitem__("model_mode","int8_c8"),
            lambda r:r.__setitem__("no_pcm_inserted_between_recordings",False),
            lambda r:r.__setitem__("stream_reset_between_recordings",True),
            lambda r:r.__setitem__("continuous_household_far_per_hour",0.0),
            lambda r:r.__setitem__("human_acoustic_gold",True),
            lambda r:r.__setitem__("independent_human_frr",0.0),
            lambda r:r.__setitem__("ssc305_target_qualification","PASS"),
            lambda r:r.__setitem__("model_promotion_allowed",True),
            lambda r:r.__setitem__("new_training_tts_asr_calls",1),
            lambda r:r["source_roster"][0].__setitem__("trusted_negative",True),
            lambda r:r["source_roster"][0].__setitem__("human_gold",True),
            lambda r:r["source_roster"][0].__setitem__("member","../../audio.flac"),
            lambda r:r["source_roster"][0].__setitem__("sample_count",3),
            lambda r:r["source_roster"][0].__setitem__("flac_sha256","0"),
            lambda r:r["actual_sessions"]["fixed320__source_member"].__setitem__(
                "event_count",1),
            lambda r:r["actual_sessions"]["network1600__speaker_round_robin"].__setitem__(
                "no_interclip_reset",False),
            lambda r:r["actual_sessions"]["fixed320__speaker_round_robin"].__setitem__(
                "ordered_source_sha256","0"*64),
        )
        for edit in corruptions:
            bad=copy.deepcopy(base)
            edit(bad)
            with self.assertRaises(ValueError):
                study.verify_result(bad)

    def test_any_event_requires_exact_C_abi_and_original_source_position(self):
        ok=synthetic_receipt()
        for k,v in ok["actual_sessions"].items():
            first=study.ordered_members(ok["source_roster"],k.split("__")[1])[0]
            v["events"]=[{
                "keyword":"HEY_MERO","keyword_id":3,
                "available_samples":320,"eof_flush":False,
                "source_interval_member":first,
                "near_splice_boundary_200ms":True,
                "exact_splice_boundary":False,
            }]
            v["event_count"]=1
        self.assertTrue(study.verify_result(ok))
        for key,bad in (("keyword_id",2),
                        ("available_samples",study.SOURCE_PCM_SAMPLES+1),
                        ("source_interval_member","WRONG")):
            forged=copy.deepcopy(ok)
            forged["actual_sessions"]["fixed320__source_member"]["events"][0][key]=bad
            with self.assertRaises(ValueError):
                study.verify_result(forged)

    def test_fake_source_artifact_rejected_before_any_native_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            f=Path(root)/"saved-reference.zip"
            f.write_bytes(b"untrusted-invented-evidence")
            with self.assertRaisesRegex(ValueError,"EN_NAT_CONT_BASELINE_ARTIFACT_IDENTITY"):
                study.reference_readback(f)


if __name__=="__main__":
    unittest.main()
