"""Source-only, no Torch, no NumPy, no Sherpa, no-audio hostile quant tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretrained_quant_events as q


def synthetic_result(mode):
    return {
        "schema": q.SCHEMA,
        "status": "REAL_PRETRAINED_SYNTHETIC_EVENTS_NOT_QUALIFICATION",
        "model_archive_sha256": q.MODEL_SHA,
        "source_tts_zip_sha256": q.source.TTS_SHA256,
        "source_review_json_sha256": q.source.REVIEW_SHA256,
        "runner_sha256": q.sha256_bytes(Path(q.__file__).read_bytes()),
        "mode": mode, "source_clips": 63,
        "source_pcm_samples": q.EXPECTED_PCM_SAMPLES,
        "keyword_line": q.KEYWORD_TEXT.strip(),
        "total_events": 0,
        "host_inference_wall_sec_excludes_load": 1.2,
        "host_process_peak_rss_kib_includes_setup": 1000,
        "model_file_bytes": {"encoder":1,"decoder":1,"joiner":1},
        "new_tts_asr_training_calls": 0,
        "actual_human_speakers": 0, "human_gold": False,
        "training_admitted": False, "independent_frr": "NOT_RUN",
        "continuous_far_per_hour": "NOT_RUN", "ssc305_qualified": False,
        "model_promotion_allowed": False,
        "clips": [{
            "cell_id": "en-" + format(i,"024x"),
            "source_family_id": "qwen3-preset-dylan",
            "role": "train-candidate",
            "requested_text": "Hey Mero",
            "requested_intent": "wake",
            "wav_sha256": "a"*64, "pcm_sha256": "b"*64,
            "events": [], "human_gold": False,
            "training_admitted": False, "trusted_negative": False,
        } for i in range(63)]
    }


class EnglishPretrainedQuantSourceCases(unittest.TestCase):
    def test_fixed_quant_variants_and_official_lexicon_contract(self):
        self.assertEqual(list(q.MODES), ["fp32_c16","int8_c16","int8_c8"])
        names = {m:q.model_component_names(m) for m in q.MODES}
        self.assertTrue(names["fp32_c16"]["encoder"].endswith("64.onnx"))
        self.assertTrue(names["int8_c16"]["encoder"].endswith("64.int8.onnx"))
        self.assertTrue(names["int8_c8"]["joiner"].endswith("64.int8.onnx"))
        self.assertTrue(all(v["decoder"].endswith("64.onnx")
                            for v in names.values()))
        self.assertEqual(q.KEYWORD_TEXT,
                         "HH EY1 M EH1 R OW0 @HEY_MERO\n")
        self.assertNotIn("VOSEN", q.KEYWORD_TEXT)
        self.assertEqual(q.EXPECTED_PCM_SAMPLES, 2101760)

    def test_invented_safe_receipt_and_untrusted_status(self):
        for mode in q.MODES:
            report=synthetic_result(mode)
            self.assertTrue(q.verify_receipt(report))
            self.assertFalse(report["model_promotion_allowed"])
            self.assertEqual(report["continuous_far_per_hour"], "NOT_RUN")

    def test_no_false_promotion_or_changed_source_digests(self):
        value=synthetic_result("fp32_c16")
        changes=(
            lambda r:r.__setitem__("model_promotion_allowed", True),
            lambda r:r.__setitem__("human_gold", True),
            lambda r:r.__setitem__("training_admitted", True),
            lambda r:r.__setitem__("independent_frr", 0.0),
            lambda r:r.__setitem__("continuous_far_per_hour", 0.0),
            lambda r:r.__setitem__("ssc305_qualified", True),
            lambda r:r.__setitem__("model_archive_sha256", "0"*64),
            lambda r:r.__setitem__("source_tts_zip_sha256", "0"*64),
            lambda r:r.__setitem__("source_review_json_sha256", "0"*64),
            lambda r:r.__setitem__("runner_sha256", "0"*64),
            lambda r:r.__setitem__("source_clips", 62),
            lambda r:r.__setitem__("mode", "int8_bogus"),
            lambda r:r["clips"][0].__setitem__("trusted_negative", True),
            lambda r:r["clips"][0].__setitem__("requested_intent", "ordinary_negative"),
            lambda r:r["clips"][0].__setitem__("events", [{"keyword":"HEY_VOSEN","is_streaming_event":True}]),
        )
        for f in changes:
            forged=copy.deepcopy(value)
            f(forged)
            with self.assertRaises(ValueError):
                q.verify_receipt(forged)

    def test_original_source_roles_and_prompts_never_become_gold(self):
        row={
            "cell_id": "en-" + "a"*24,
            "voice_id": "dylan", "source_family_id": "qwen3-preset-dylan",
            "role": "train-candidate", "requested_text": "Hey Mero",
            "requested_intent": "wake", "human_gold":False,
            "training_admitted":False,"trusted_negative":False,
            "phonetic_label_authenticated":False,
            "independent_evaluation":False
        }
        self.assertTrue(q.correct_spelling_and_role(row))
        for k,v in (("human_gold",True),("phonetic_label_authenticated",True),
                    ("source_family_id","qwen3-preset-eric"),
                    ("requested_intent","ordinary_negative"),
                    ("role","development-only")):
            forged={**row,k:v}
            with self.assertRaises(ValueError):
                q.correct_spelling_and_role(forged)

    def test_comparison_of_invented_receipts_only_never_selects_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            paths=[]
            for mode in q.MODES:
                f=root/(mode+".json")
                f.write_text(json.dumps(synthetic_result(mode)))
                paths.append(f)
            out=root/"comparison.json"
            value=q.compare_receipts(paths,out)
            self.assertEqual(set(value["modes"]),set(q.MODES))
            self.assertTrue(out.is_file())
            self.assertFalse(value["model_promotion_allowed"])
            self.assertFalse(value["model_selection_made"])
            with self.assertRaises(ValueError):
                q.compare_receipts(paths,out)


if __name__ == "__main__":
    unittest.main()
