"""Model-free source/receipt regression tests for contiguous C KWS research."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_contiguous_pcm_c as cont


def invented_report():
    cells=["en-"+format(n,"024x") for n in range(63)]
    recordings=[]
    sample=0
    for i,cell in enumerate(cells):
        length=33361+(17 if i==0 else 0)
        recordings.append({
            "cell_id":cell,
            "start_sample":sample,
            "end_sample":sample+length,
            "source_family_id":"qwen3-preset-dylan",
            "requested_text":"Hey Mero",
            "human_gold":False,
        })
        sample+=length
    assert sample==cont.TOTAL_SOURCE_SAMPLES
    measures={}
    for order in cont.ORDERS:
        for pattern in cont.CHUNK_PLANS:
            measures[pattern+"__"+order]={
                "ordered_cell_ids":cells,
                "source_interval_boundaries":recordings,
                "events":[],
                "event_count":0,
                "eof_event_count":0,
                "no_reset_between_source_clips":True,
                "no_interclip_silence_inserted":True,
            }
    return {
        "schema":cont.POLICY,
        "status":"EXECUTED_CONTIGUOUS_SYNTHETIC_SOURCE_EVENTS_NOT_FIELD_FAR",
        "runner_sha256":cont.hashlib.sha256(Path(cont.__file__).read_bytes()).hexdigest(),
        "source_audio_sha256":cont.native.quant.source.TTS_SHA256,
        "source_review_sha256":cont.native.quant.source.REVIEW_SHA256,
        "reference_unpadded_zip_sha256":cont.native.REFERENCE_ARTIFACT_SHA,
        "official_model_sha256":cont.native.quant.MODEL_SHA,
        "model_mode":"int8_c16",
        "source_clips":63,
        "source_samples_per_replay":cont.TOTAL_SOURCE_SAMPLES,
        "order_variants":list(cont.ORDERS),
        "caller_partitions":list(cont.CHUNK_PLANS),
        "measurements":measures,
        "human_gold":False,
        "independent_human_frr":"NOT_RUN",
        "continuous_household_far_per_hour":"NOT_RUN",
        "ssc305_target_qualification":"NOT_RUN",
        "model_promotion_allowed":False,
        "new_training_or_generated_audio":False,
    }


class EnglishContiguousSourceCases(unittest.TestCase):
    def test_fixed_source_and_contiguous_research_boundaries(self):
        self.assertTrue(cont.verify_receipt(invented_report()))
        self.assertEqual(cont.CLIPS,63)
        self.assertEqual(cont.TOTAL_SOURCE_SAMPLES,2101760)
        self.assertEqual(set(cont.CHUNK_PLANS),
                         {"fixed320","network1600","uneven"})
        self.assertEqual(len(cont.ORDERS),2)
        self.assertEqual(cont.KEYWORD,"HEY_MERO")

    def test_invented_field_success_qualification_source_or_sample_is_rejected(self):
        base=invented_report()
        changes=(
            lambda r:r.__setitem__("runner_sha256","0"*64),
            lambda r:r.__setitem__("source_audio_sha256","0"*64),
            lambda r:r.__setitem__("source_review_sha256","0"*64),
            lambda r:r.__setitem__("reference_unpadded_zip_sha256","0"*64),
            lambda r:r.__setitem__("official_model_sha256","0"*64),
            lambda r:r.__setitem__("model_promotion_allowed",True),
            lambda r:r.__setitem__("human_gold",True),
            lambda r:r.__setitem__("continuous_household_far_per_hour",0.0),
            lambda r:r.__setitem__("independent_human_frr",0.0),
            lambda r:r.__setitem__("ssc305_target_qualification","PASS"),
            lambda r:r.__setitem__("new_training_or_generated_audio",True),
            lambda r:r.__setitem__("source_samples_per_replay",2101761),
            lambda r:r.__setitem__("source_clips",62),
            lambda r:r["measurements"]["fixed320__cell_id"].__setitem__(
                "no_reset_between_source_clips",False),
            lambda r:r["measurements"]["network1600__cell_id"].__setitem__(
                "no_interclip_silence_inserted",False),
            lambda r:r["measurements"]["uneven__cell_id"][
                "source_interval_boundaries"][0].__setitem__("end_sample",1234),
            lambda r:r["measurements"]["fixed320__cell_id"][
                "ordered_cell_ids"].__setitem__(0,"WRONG"),
            lambda r:r["measurements"]["fixed320__cell_id"].__setitem__(
                "event_count",1),
        )
        for edit in changes:
            bad=copy.deepcopy(base)
            edit(bad)
            with self.assertRaises(ValueError):
                cont.verify_receipt(bad)

    def test_event_payload_can_be_emitted_only_with_valid_c_id_and_source_position(self):
        report=invented_report()
        for order in cont.ORDERS:
            for pattern in cont.CHUNK_PLANS:
                r=report["measurements"][pattern+"__"+order]
                r["events"]=[{"keyword":"HEY_MERO","id":3,
                              "available_samples":320,
                              "eof_flush":False,
                              "source_interval_cell_id":r["ordered_cell_ids"][0],
                              "source_boundary_ambiguous":False}]
                r["event_count"]=1
        self.assertTrue(cont.verify_receipt(report))
        forged=copy.deepcopy(report)
        forged["measurements"]["network1600__cell_id"]["events"][0]["id"]=2
        with self.assertRaises(ValueError):
            cont.verify_receipt(forged)


    def test_actual_source_locked_contiguous_result_is_nonpromotable(self):
        report=json.loads((HERE.parent/"ENGLISH_CONTIGUOUS_RESULT.json").read_text())
        self.assertEqual(report["schema"],"english63-contiguous-native-c-actual-readback-summary-v1")
        self.assertEqual(report["status"],
                         "MEASURED_SYNTHETIC_CONTIGUOUS_HOST_KWS_EVENTS_NOT_FIELD_QUALIFICATION")
        original=report["source"]
        self.assertEqual(original["runner_sha256"],
                         cont.hashlib.sha256(Path(cont.__file__).read_bytes()).hexdigest())
        self.assertEqual(original["native_c_library_sha256"],
                         "b43d258c127ce5e8563b81273f4f36ad7f728a9968ca46caa970c57d8f954d3e")
        self.assertEqual(original["official_model_sha256"],cont.native.quant.MODEL_SHA)
        self.assertEqual(original["original_audio_zip_sha256"],
                         cont.native.quant.source.TTS_SHA256)
        self.assertEqual(original["original_review_sha256"],
                         cont.native.quant.source.REVIEW_SHA256)
        self.assertEqual(original["unpadded_per_clip_baseline_zip_sha256"],
                         cont.native.REFERENCE_ARTIFACT_SHA)
        experiment=report["experiment"]
        self.assertEqual(experiment["continuous_sessions"],6)
        self.assertEqual(experiment["source_clips_per_session"],63)
        self.assertEqual(experiment["source_audio_samples_per_session"],cont.TOTAL_SOURCE_SAMPLES)
        self.assertEqual(experiment["ordered_variants"],list(cont.ORDERS))
        self.assertEqual(experiment["caller_pcm_partitions"],list(cont.CHUNK_PLANS))
        self.assertTrue(experiment["same_order_same_events_and_sample_positions_all_partitions"])
        self.assertFalse(experiment["inserted_inter_utterance_silence"])
        self.assertFalse(experiment["reset_between_source_clips"])
        self.assertEqual(experiment["eof_only_event_count_each_session"],0)
        self.assertEqual(set(report["results"]),set(cont.ORDERS))
        for order,expected,other in (("cell_id",11,4),
                                     ("source_family_then_requested_text",7,2)):
            evidence=report["results"][order]
            self.assertEqual(evidence["actual_events_per_caller_partition"],expected)
            self.assertEqual(evidence["events_in_other_requested_source_intervals"],other)
            self.assertEqual(evidence["events_at_exact_source_splice_boundary"],1)
            self.assertEqual(len(evidence["event_available_sample_positions"]),expected)
            self.assertEqual(sorted(evidence["event_available_sample_positions"]),
                             evidence["event_available_sample_positions"])
        lim=report["scientific_limits"]
        self.assertTrue(lim["contiguous_source_is_synthetically_spliced_recordings"])
        self.assertFalse(lim["requested_text_is_human_acoustic_gold"])
        self.assertFalse(lim["other_prompt_intervals_are_confirmed_false_alarms"])
        self.assertEqual(lim["continuous_household_FAR_per_hour"],"NOT_RUN")
        self.assertEqual(lim["SSC305_qualified"],"NOT_RUN")
        self.assertFalse(lim["model_promotion_allowed"])


if __name__=="__main__":
    unittest.main()
