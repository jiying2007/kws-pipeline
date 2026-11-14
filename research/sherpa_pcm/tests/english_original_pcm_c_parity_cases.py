"""No model, audio or external binary required: C research source identity tests."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as research
from build_official_x64 import ARCHIVE_SHA, ARCHIVE_BYTES, LIBRARIES, verify_original_archive


class EnglishNativeMeroSourceCases(unittest.TestCase):
    def test_native_event_abi_is_opt_in_mero_only(self):
        src=(HERE.parent/"pcm_kws.cc").read_text()
        head=(HERE.parent/"pcm_kws.h").read_text()
        self.assertIn('!std::strcmp(r->keyword,"HEY_MERO")?3:0',src)
        self.assertIn('!std::strcmp(r->keyword,"你好小窝")?1',src)
        self.assertIn('!std::strcmp(r->keyword,"小窝小窝")?2',src)
        self.assertIn("HEY_MERO",head)
        self.assertEqual(research.POLICY,
                         "english63-native-c-mero-unpadded-event-parity-v1")
        self.assertEqual(research.KEYWORD,"HEY_MERO")
        self.assertEqual(set(research.PATS),{"fixed320","network1600","uneven"})
        self.assertEqual(research.REFERENCE_JSON_SHA,
                         "d5cfbd295299ff8c0ee7524538e44c9755ad25a943f5178455816bc691dbe521")

    def test_unverified_original_results_never_reach_native_model(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"source.zip"
            path.write_bytes(b"invented-not-Actions-source")
            with self.assertRaisesRegex(ValueError,"EN_C_BASELINE_ZIP_DRIFT"):
                research.verify_saved_baseline(path)

    def test_unsafe_runtime_or_missing_build_receipt_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/"libsherpa-onnx-c-api.so"
            fake.write_bytes(b"invented-not-actual")
            missing=Path(d)/"build.json"
            with self.assertRaisesRegex(ValueError,"EN_C_NATIVE_PATH"):
                research.verify_build_receipt(fake,missing)


    def test_original_official_x64_runtime_has_distinct_immutable_sherpa_hash(self):
        self.assertEqual(ARCHIVE_SHA,
                         "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4")
        self.assertEqual(ARCHIVE_BYTES,28156791)
        self.assertEqual(LIBRARIES["libsherpa-onnx-c-api.so"],(
            5124192,"b8351ca1632571ac108adbb317bcc4bf7cfe84b72690e3017316b0da3e1e344f"))
        self.assertEqual(LIBRARIES["libonnxruntime.so"],(
            27026609,"4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e"))
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/"official.tar.bz2"
            fake.write_bytes(b"not-original-runtime")
            with self.assertRaisesRegex(ValueError,"EN_C_OFFICIAL_HOST_ARCHIVE_CHANGED"):
                verify_original_archive(fake)


    def test_retained_actual_native_model_receipt_is_hash_bound_and_nonpromotable(self):
        import json
        manifest=json.loads((HERE.parent/"ENGLISH_REAL_PARITY_RESULT.json").read_text())
        self.assertEqual(manifest["schema"],
                         "english63-actual-native-c-int8-parity-readback-v1")
        self.assertEqual(manifest["status"],
                         "SOURCE_VERIFIED_HOST_NATIVE_C_RESEARCH_EXECUTION_NOT_QUALIFIED")
        observed=manifest["evidence"]
        self.assertEqual(observed["c_adapter_source_sha256"],
                         research.sha((HERE.parent/"pcm_kws.cc").read_bytes()))
        self.assertEqual(observed["real_c_parity_runner_sha256"],
                         research.sha(Path(research.__file__).read_bytes()))
        self.assertEqual(observed["official_model_archive_sha256"],research.quant.MODEL_SHA)
        self.assertEqual(observed["official_host_release_sha256"],ARCHIVE_SHA)
        self.assertEqual(observed["official_host_c_api_sha256"],
                         LIBRARIES["libsherpa-onnx-c-api.so"][1])
        self.assertEqual(observed["official_host_ort_sha256"],
                         LIBRARIES["libonnxruntime.so"][1])
        self.assertEqual(observed["original_qwen_audio_zip_sha256"],research.quant.source.TTS_SHA256)
        self.assertEqual(observed["unpadded_python_baseline_zip_sha256"],
                         research.REFERENCE_ARTIFACT_SHA)
        d=manifest["experiment"]
        self.assertEqual(d["source_original_synthetic_pcm16_clips"],63)
        self.assertEqual(len(d["input_partition_patterns"]),3)
        self.assertEqual(len(d["source_wake_clip_ids"]),5)
        self.assertEqual(len(set(d["source_wake_clip_ids"])),5)
        self.assertTrue(d["all_patterns_same_source_cell_event_counts"])
        self.assertEqual(d["per_pattern_actual_events"],5)
        self.assertEqual(d["per_pattern_baseline_python_events"],5)
        self.assertEqual(d["per_pattern_changed_event_count_cells"],0)
        self.assertEqual(d["per_pattern_eof_flush_events"],0)
        self.assertEqual(d["event_id"],3)
        self.assertTrue(d["no_artificial_tail"])
        self.assertFalse(manifest["limits"]["human_gold"])
        self.assertFalse(manifest["limits"]["final_model_promotion_allowed"])
        self.assertEqual(manifest["limits"]["continuous_household_far_per_hour"],"NOT_RUN")
        self.assertEqual(manifest["limits"]["true_ssc305_arm_board"],"NOT_RUN")


    def test_actual_research_candidate_cannot_be_mistaken_for_deployed_model(self):
        import json
        m=json.loads((HERE.parent/"ENGLISH_INT8_C16_RESEARCH_CANDIDATE.json").read_text())
        self.assertEqual(m["schema"],"english-research-model-candidate-v1")
        self.assertEqual(m["status"],
                         "HOST_NATIVE_C_REPRODUCED_RESEARCH_ONLY_NO_PROMOTION")
        self.assertEqual(m["keyword_alias"],research.KEYWORD)
        self.assertEqual(m["keyword_c_abi_id"],3)
        self.assertEqual(m["second_keyword_status"],
                         "BLOCKED_UNVERIFIED_OFFICIAL_LEXICON_OOV")
        self.assertEqual(m["model"]["original_release_sha256"],
                         research.quant.MODEL_SHA)
        self.assertEqual(m["runtime"]["official_host_runtime_archive_sha256"],
                         ARCHIVE_SHA)
        self.assertEqual(m["runtime"]["official_host_C_api_sha256"],
                         LIBRARIES["libsherpa-onnx-c-api.so"][1])
        self.assertEqual(m["runtime"]["official_host_ONNX_Runtime_sha256"],
                         LIBRARIES["libonnxruntime.so"][1])
        self.assertEqual(m["evidence"]["actual_C_abi_run_id"],43547980797)
        self.assertEqual(m["evidence"]["real_native_C_tests"]["total_replays"],189)
        self.assertEqual(m["evidence"]["real_native_C_tests"]["per_pattern_deviation_vs_python"],0)
        self.assertEqual(m["evidence"]["synthetic_source"]["after_artificial_tail_only"],4)
        self.assertEqual(m["evidence"]["natural_source"]["sampled_utterances"],517)
        self.assertFalse(m["runtime"]["SSC305_SDK_ABI_qualified"])
        self.assertFalse(m["runtime"]["SSC305_CPU_RSS_p99_qualified"])
        self.assertFalse(m["model_selected_for_shipping"])
        self.assertFalse(m["model_promotion_allowed"])
        self.assertEqual(m["true_independent_FAR_per_hour"],"NOT_RUN")
        self.assertEqual(m["true_independent_FRR"],"NOT_RUN")
        self.assertGreaterEqual(len(m["pending_gates"]),5)


if __name__=="__main__":
    unittest.main()
