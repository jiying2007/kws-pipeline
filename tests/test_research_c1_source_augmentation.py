#!/usr/bin/env python3
"""Small invented metadata tests; no network, queue, tensor runtime or model."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_c1_source_augmentation as stage


def fixture(text="你好小窝", category="full-positive", intended=1):
    vocabulary = {"schema": "kws-d1-fulltext-vocabulary-v2", "output_count": 2601,
        "token_to_id": {char: i + 1 for i, char in enumerate("你好小窝今天请把灯打开吴")}}
    row = {"consumer": "d1-verified-full-transcript-v2", "purpose": "technical-research-only",
        "human_gold": False, "independent_test": False, "split": "train", "source_role": "train-candidate",
        "source_id": "invented-cell", "source_family_id": "invented-family", "speaker_id": "invented-speaker",
        "source_plan_sha256": stage.NEW_PLAN_SHA, "global_source_key": [stage.NEW_PLAN_SHA, "invented-cell"],
        "label_status": "machine_candidate", "full_text": text, "ctc_mask": True, "ctc_weight": .5,
        "keyword_targets": [int(word in text) for word in stage.fit.KEYWORDS], "keyword_mask": [True, True],
        "keyword_weight": [.5, .5], "samples": 16000, "generation_category": category,
        "intended_keyword_id": intended, "pcm_sha256": "1" * 64, "file_sha256": "2" * 64,
        "audio": "invented.wav", "audio_path": "invented.wav"}
    return vocabulary, row


class MetadataTests(unittest.TestCase):
    def test_import_has_no_numerical_runtime(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_observed_positive_is_kept_under_negative_generation_intent(self):
        vocabulary, row = fixture(category="ordinary-negative", intended=None)
        actual = stage.project_row(row, vocabulary)
        self.assertTrue(actual["ctc_mask"])
        self.assertTrue(actual["intent_label_conflict_diagnostic"])
        self.assertEqual(actual["keyword_targets"], [1, 0])
        self.assertEqual(actual["full_text"], row["full_text"])
        self.assertEqual(actual["ctc_weight"], .5)

    def test_observed_negative_is_kept_under_positive_generation_intent(self):
        vocabulary, row = fixture(text="今天")
        actual = stage.project_row(row, vocabulary)
        self.assertTrue(actual["ctc_mask"])
        self.assertTrue(actual["intent_label_conflict_diagnostic"])
        self.assertEqual(actual["keyword_targets"], [0, 0])
        self.assertEqual(actual["target_ids"], stage.fit.fulltext_targets("今天", vocabulary)["target_ids"])

    def test_uncertain_is_not_made_negative_from_absent_keyword(self):
        vocabulary, row = fixture()
        row.update(label_status="uncertain", full_text=None, ctc_mask=False, ctc_weight=0.0,
            keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0.0, 0.0])
        actual = stage.project_row(row, vocabulary)
        self.assertFalse(actual["ctc_mask"])
        self.assertIsNone(actual["target_ids"])
        self.assertEqual(actual["keyword_targets"], [None, None])
        self.assertEqual(actual["keyword_mask"], [False, False])

    def test_target_free_geometry_is_a_temporary_nonlabel_copy(self):
        vocabulary, row = fixture()
        row.update(label_status="uncertain", full_text=None, ctc_mask=False, ctc_weight=0.0,
            keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0.0, 0.0])
        actual = stage.project_row(row, vocabulary)
        original = copy.deepcopy(actual)
        geometry = stage.feature_geometry_row(actual)
        self.assertEqual(geometry, {**original, "ctc_minimum_frames": 0})
        self.assertEqual(actual, original)
        self.assertIsNone(actual["ctc_minimum_frames"])
        self.assertNotEqual(stage.fit.row_identity(geometry), stage.fit.row_identity(actual))
        for changed in ({"target_ids": [1]}, {"ctc_weight": .5}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "MISSING_TRAIN_CTC_GEOMETRY"):
                stage.feature_geometry_row({**actual, **changed})

    def test_target_free_geometry_never_relaxes_known_target_bound(self):
        vocabulary, row = fixture()
        known = stage.project_row(row, vocabulary)
        self.assertIs(stage.feature_geometry_row(known), known)
        with self.assertRaisesRegex(ValueError, "MISSING_TRAIN_CTC_GEOMETRY"):
            stage.feature_geometry_row({**known, "ctc_minimum_frames": None})
        infeasible = {**known, "ctc_mask": False, "ctc_minimum_frames": 10000}
        self.assertIs(stage.feature_geometry_row(infeasible), infeasible)

    def test_wrong_keyword_label_is_rejected(self):
        vocabulary, row = fixture()
        row["keyword_targets"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "UNKNOWN_NOT_NEGATIVE"):
            stage.project_row(row, vocabulary)

    def test_c1_geometry_does_not_inherit_shorter_fsmn_gate(self):
        vocabulary, row = fixture(text="你好小窝" * 3)
        row["samples"] = 5680
        legacy = stage.d1_recipe.project_row(row, vocabulary)
        actual = stage.project_row(row, vocabulary)
        self.assertFalse(legacy["ctc_mask"])
        self.assertEqual(legacy["ctc_reason"], "common_unpadded_ctc_geometry_infeasible")
        self.assertTrue(actual["ctc_mask"])
        self.assertEqual(actual["expected_c1_frames"], 17)
        self.assertEqual(actual["ctc_minimum_frames"], 12)

    def test_oov_keeps_complete_text_and_original_weak_label(self):
        vocabulary, row = fixture(text="你好小窝龘")
        actual = stage.project_row(row, vocabulary)
        self.assertFalse(actual["ctc_mask"])
        self.assertEqual(actual["ctc_weight"], 0)
        self.assertEqual(actual["observed_label"]["ctc_weight"], .5)
        self.assertEqual(actual["full_text"], "你好小窝龘")
        self.assertEqual(actual["oov_characters"], ["龘"])
        self.assertEqual(actual["keyword_targets"], [1, 0])

    def test_repeated_target_needs_explicit_blank_frame(self):
        vocabulary, row = fixture(text="好好", category="ordinary-negative", intended=None)
        row["samples"] = 720  # two C1 frames, whereas repeated target needs three
        actual = stage.project_row(row, vocabulary)
        self.assertFalse(actual["ctc_mask"])
        self.assertEqual(actual["ctc_minimum_frames"], 3)
        self.assertEqual(actual["ctc_reason"], "unpadded_c1_ctc_geometry_infeasible")

    def test_failed_native_cannot_acquire_nonzero_weak_label(self):
        vocabulary, row = fixture()
        row["generation_outcome"] = "failed-native"
        with self.assertRaisesRegex(ValueError, "FAILED_NATIVE_LABEL"):
            stage.project_row(row, vocabulary)

    def test_100_row_cycle_is_exact_8000_with_no_dev_draw(self):
        _, row = fixture()
        rows = [{**copy.deepcopy(row), "source_id": "row-%03d" % i} for i in range(100)]
        draws = stage.exposure_plan(rows)
        self.assertEqual(len(draws), 8000)
        self.assertEqual(set(Counter(d["source_id"] for d in draws).values()), {80})
        self.assertEqual(Counter(d["offset_ms"] for d in draws), {0: 2700, 200: 2700, 500: 2600})
        rows[-1]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "TRAIN_ONLY"):
            stage.exposure_plan(rows)

    def test_role_leak_and_old_source_reuse_rejected(self):
        _, row = fixture()
        other = {**row, "source_id": "new", "source_family_id": "new-family", "speaker_id": "new-speaker",
            "pcm_sha256": "3" * 64, "file_sha256": "4" * 64}
        stage.role_check([row], [other])
        for key in ("source_family_id", "speaker_id", "pcm_sha256", "file_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "OVERLAPS_OLD"):
                stage.role_check([row], [{**other, key: row[key]}])

    def test_contract_rejects_budget_or_training_change(self):
        base = {**stage.contract("augmentation2000"), "job_id": "c1-newsource64-augmentation2000", "final_forward_limit": 402}
        stage.validate_contract(base)
        for key, value in (("optimizer_steps", 2001), ("worker_seconds", 541), ("threads", 8),
            ("final_forward_limit", 403), ("automatic_retry", True), ("auxiliary_scale", .5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.validate_contract({**base, key: value})

    def test_terminal_counts_require_baseline_zero_optimizer(self):
        job = {**stage.contract("baseline-evaluation"), "final_forward_limit": 192}
        status = {"model_constructions": 1, "actual_optimizer_steps": 0, "training_forwards_started": 0,
            "training_forwards_completed": 0, "final_forwards_started": 192, "final_forwards_completed": 192}
        stage.validate_terminal_counts(status, job)
        with self.assertRaises(ValueError):
            stage.validate_terminal_counts({**status, "actual_optimizer_steps": 1}, job)

    def test_frozen_queue_budgets_are_300_and_900_seconds(self):
        for mode, minutes, seconds in (("baseline-evaluation", 5, 270), ("augmentation2000", 15, 840)):
            job = {**stage.contract(mode), "job_id": "c1-newsource64-" + mode,
                "final_forward_limit": 24, "heavy_lock": "/invented.lock"}
            with patch.object(stage.training, "read_json", return_value=job), \
                patch.object(stage.training, "sha_file", return_value="0" * 64):
                queue = stage.queue_definition("/invented/recipe.json", "/invented/python")
            self.assertEqual(queue["global_budget_minutes"], minutes)
            self.assertEqual(queue["jobs"][0]["timeout_minutes"], minutes)
            self.assertEqual(job["worker_seconds"], seconds)

    def test_absent_weak_rows_can_remain_failed_unknown_or_unstarted(self):
        cells = [{"cell_id": "cell-%02d" % i} for i in range(64)]
        value = {"schema": stage.ACCOUNTING, "terminal": True, "plan_sha256": stage.NEW_PLAN_SHA,
            "selected_cell_ids": [c["cell_id"] for c in cells], "source_pins": [{"invented": True}],
            "entries": [{**c, "tts_status": ("failed", "unknown", "not_started")[i % 3],
                "asr_status": "not_run", "weak_row_present": False, "evidence": [{"invented": True}]}
                for i, c in enumerate(cells)]}
        with patch.object(stage, "checked"):
            self.assertEqual(stage.validate_accounting(value, cells, []), value)
            missing = copy.deepcopy(value)
            missing["entries"].pop()
            with self.assertRaisesRegex(ValueError, "TERMINAL_ACCOUNTING"):
                stage.validate_accounting(missing, cells, [])

    def test_declare_retains_unsupported_train_in_final_numeric_roster(self):
        vocabulary, row = fixture()
        unknown = {**row, "source_id": "unsupported", "ctc_mask": False, "ctc_weight": 0,
            "ctc_reason": "full_transcript_out_of_vocabulary"}
        data = {"base": {k: {} for k in ("initial_checkpoint", "initial_binding", "initial_source_hashes",
            "model_description", "canonical_parameter_names", "vocabulary_sha256", "vocabulary_file", "tokens_file", "heavy_lock")},
            "new": [row, unknown], "train": [row], "development": [], "audio": {}, "aggregate": {},
            "loader_sources": {}, "missing_weak_rows": [], "base_summary": {"final_float_state_sha256": "state"}}
        data["base"]["excluded_original_rows"] = []
        with patch.object(stage, "assemble", return_value=data), patch.object(stage, "pin", return_value={}), \
            patch.object(stage.training, "sha_file", return_value="invented"):
            job = stage.declare({"baseline_output": "/invented"}, "augmentation2000")
        self.assertEqual([r["source_id"] for r in job["evaluation_rows"]], ["invented-cell", "unsupported"])
        self.assertEqual(job["final_forward_limit"], 6)


if __name__ == "__main__":
    unittest.main()
