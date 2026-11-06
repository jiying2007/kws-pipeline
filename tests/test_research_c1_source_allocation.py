#!/usr/bin/env python3
"""Invented metadata only: no model, optimizer, tensor runtime or queue launch."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training")]
import research_c1_source_allocation as stage


def row(name, number, split="train", known=True):
    return {"source_id": name, "source_family_id": "family-" + name, "speaker_id": "speaker-" + name,
        "split": split, "source_role": "train-candidate" if split == "train" else "development-only",
        "ctc_mask": known, "ctc_weight": (.5 if number % 2 else 1.0) if known else 0.0,
        "target_ids": [1] if known else None, "full_text": "甲" if known else None,
        "pcm_sha256": "%064x" % number, "file_sha256": "%064x" % (1000 + number)}


def fixture():
    old = [row("old-train-%02d" % i, i + 1) for i in range(52)]
    new = [row("new-train-%02d" % i, i + 101) for i in range(44)]
    unknown = [row("unknown-%d" % i, i + 201, known=False) for i in range(4)]
    old_dev = [row("old-dev-%02d" % i, i + 301, split="development") for i in range(18)]
    new_dev = [row("new-dev-%02d" % i, i + 401, split="development") for i in range(16)]
    train, dev = old + new, old_dev + new_dev
    return {**stage.engine.contract("augmentation2000"), "job_id": "c1-newsource64-augmentation2000",
        "train": train, "development": dev, "all_new_rows": copy.deepcopy(new + unknown + new_dev),
        "evaluation_rows": copy.deepcopy(train + unknown + dev), "new_zero_loss_rows": [r["source_id"] for r in unknown],
        "excluded_original_rows": [row("excluded-%d" % i, i + 501, known=False) for i in range(2)],
        "final_forward_limit": 402, "heavy_lock": "/invented/shared.lock", "audio_files": {},
        "initial_checkpoint": {"path": "/invented/initial.pt"}, "vocabulary_file": {"path": "/invented/vocabulary.json"},
        "tokens_file": {"path": "/invented/tokens.txt"}, "inputs": {"frozen_original_input": True},
        "source_hashes": {"invented-original-source": "1" * 64}}


def budget():
    return {"schema": "kws-posterior-allocation-continuation-stage-v1",
        "global_runner_minutes_ceiling": 350, "python_version": "3.12.16",
        "heavy_model_concurrency": 1, "threads": 4, "interop_threads": 1,
        "shared_heavy_lock": "/invented/shared.lock", "model_execution_requires_frozen_followup_decision": True,
        "new_tts_calls": 0, "new_asr_calls": 0, "unknown_model_retries": 0,
        "paid_model_api_calls": 0, "external_codex_or_work_jobs": 0,
        "reservations_runner_minutes": {"optional_single_allocation_training_and_audit": 20, "other": 330},
        "phases": [{"id": "source-allocation-candidate", "new_training_forwards_max": 8000,
            "new_optimizer_steps_max": 2000, "new_final_forwards_max": 402, "model_constructions_max": 1,
            "worker_seconds_max": 840, "queue_seconds_max": 900}]}


class AllocationTests(unittest.TestCase):
    def test_fixed_batches_fair_rows_and_per_row_offset_rotation(self):
        control = fixture()
        new_ids = stage.validate_roster(control)
        draws = stage.exposure_plan(control["train"], new_ids)
        self.assertEqual(len(draws), 8000)
        self.assertEqual(Counter(d["source_id"] in new_ids for d in draws), {False: 6000, True: 2000})
        for start in range(0, 8000, 4):
            self.assertEqual([d["source_id"] in new_ids for d in draws[start:start + 4]], [False, False, False, True])
        for new_cohort, counts in ((False, {115: 32, 116: 20}), (True, {45: 24, 46: 20})):
            per_row = Counter(d["source_id"] for d in draws if (d["source_id"] in new_ids) == new_cohort)
            self.assertEqual(Counter(per_row.values()), counts)
        for index, sample in enumerate(control["train"]):
            visits = [d for d in draws if d["source_id"] == sample["source_id"]]
            self.assertEqual([d["row_visit_index"] for d in visits], list(range(len(visits))))
            self.assertEqual([d["offset_ms"] for d in visits], [stage.engine.OFFSETS[i % 3] for i in range(len(visits))])
            self.assertTrue(all(d["row_index"] == index and d["ctc_weight"] == sample["ctc_weight"] for d in visits))

    def test_deterministic_without_mutating_rows(self):
        control = fixture()
        saved = copy.deepcopy(control)
        new_ids = stage.validate_roster(control)
        self.assertEqual(stage.exposure_plan(control["train"], new_ids),
            stage.exposure_plan(copy.deepcopy(control["train"]), new_ids))
        self.assertEqual(control, saved)

    def test_unknown_and_development_have_no_draws(self):
        control = fixture()
        draws = stage.exposure_plan(control["train"], stage.validate_roster(control))
        forbidden = set(control["new_zero_loss_rows"]) | {r["source_id"] for r in control["development"]}
        self.assertFalse(forbidden & {d["source_id"] for d in draws})
        self.assertEqual(len(control["evaluation_rows"]), 134)

    def test_unknown_or_development_cannot_be_inserted_into_train(self):
        for update in ({"ctc_mask": False, "ctc_weight": 0, "target_ids": None},
            {"split": "development"}, {"source_role": "development-only"}, {"ctc_weight": float("nan")},
            {"ctc_weight": True}):
            control = fixture()
            new_ids = stage.validate_roster(control)
            control["train"][0].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "TRAIN_ONLY_KNOWN"):
                stage.exposure_plan(control["train"], new_ids)

    def test_duplicate_source_and_wrong_cohort_size_rejected(self):
        control = fixture()
        new_ids = stage.validate_roster(control)
        control["train"][1]["source_id"] = control["train"][0]["source_id"]
        with self.assertRaisesRegex(ValueError, "TRAIN_ONLY_KNOWN"):
            stage.exposure_plan(control["train"], new_ids)
        control = fixture()
        with self.assertRaisesRegex(ValueError, "SOURCE_COHORT_COUNTS"):
            stage.exposure_plan(control["train"], new_ids | {control["train"][0]["source_id"]})

    def test_original_and_new_source_identity_overlap_rejected(self):
        for key in ("speaker_id", "source_family_id", "pcm_sha256", "file_sha256"):
            control = fixture()
            control["all_new_rows"][0][key] = control["train"][0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "OVERLAPS_OLD"):
                stage.validate_roster(control)

    def test_development_identity_leak_rejected(self):
        control = fixture()
        control["all_new_rows"][-1]["source_family_id"] = control["all_new_rows"][0]["source_family_id"]
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_LEAK"):
            stage.validate_roster(control)

    def test_new_training_row_cannot_change_its_frozen_weak_label(self):
        control = fixture()
        control["train"][-1]["ctc_weight"] = .9
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_CHANGED"):
            stage.validate_roster(control)

    def test_unknown_rows_remain_in_final_roster(self):
        control = fixture()
        control["evaluation_rows"].pop(96)
        with self.assertRaisesRegex(ValueError, "UNKNOWN_OR_EVALUATION_CHANGED"):
            stage.validate_roster(control)

    def test_contract_rejects_sweep_and_budget_changes(self):
        job = {**stage.contract(), "job_id": stage.JOB_ID, "final_forward_limit": 402}
        stage.validate_contract(job)
        for key, value in (("optimizer_steps", 2001), ("training_forward_limit", 8001),
            ("worker_seconds", 841), ("queue_minutes", 16), ("batch_size", 5),
            ("final_forward_limit", 403), ("automatic_retry", True), ("auxiliary_scale", .5),
            ("execution_authorized", False), ("source_allocation", {"old_per_batch": 2, "new_per_batch": 2})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "FIXED_CONTRACT"):
                stage.validate_contract({**job, key: value})

    def test_aggregate_and_phase_budget_are_finite(self):
        original = budget()
        stage.validate_budget(original, "/invented/shared.lock")
        changes = [
            ("global_runner_minutes_ceiling", 351), ("heavy_model_concurrency", 2),
            ("new_tts_calls", 1), ("new_asr_calls", 1), ("unknown_model_retries", 1),
            ("shared_heavy_lock", "/other.lock"),
            ("reservations_runner_minutes", {"optional_single_allocation_training_and_audit": 20, "other": 331}),
        ]
        for key, value in changes:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "BUDGET"):
                stage.validate_budget({**original, key: value}, "/invented/shared.lock")
        for key, value in (("new_final_forwards_max", 403), ("new_optimizer_steps_max", 2001),
            ("model_constructions_max", 2), ("queue_seconds_max", 901)):
            changed = copy.deepcopy(original)
            changed["phases"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "PHASE_BUDGET"):
                stage.validate_budget(changed, "/invented/shared.lock")

    def test_saved_posterior_decision_pins_control_and_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = Path(temp) / "posterior.json"
            comparison = Path(temp) / "comparison.json"
            comparison.write_text('{"fixture":"invented-comparison"}\n')
            summary_data = {"schema": "kws-c1-posterior-audit-v1", "status": "completed",
                "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0,
                "inputs": {"comparison": stage.engine.pin(comparison),
                    "augmentation": {"manifest_sha256": stage.CONTROL_MANIFEST_SHA}},
                "workload": {"new_numeric_conditions": 570, "reused_conditions": 210,
                    "unscored_conditions": 24, "paired_model_conditions": 804}}
            summary.write_text(json.dumps(summary_data))
            request = {"control": {"manifest_sha256": stage.CONTROL_MANIFEST_SHA}, "budget": {"invented": "budget"}}
            value = {"schema": stage.DECISION, "decision": "run-single-fixed-allocation-3to1",
                "posterior_summary": stage.engine.pin(summary), "control_completed_evidence": request["control"],
                "budget": request["budget"], "rationale": "Invented fixture justification.",
                "one_model_invocation": True, "no_threshold_or_loss_tuning": True}
            with patch.object(stage, "CONTROL_COMPARISON_SHA", summary_data["inputs"]["comparison"]["sha256"]):
                stage.validate_decision(value, request)
                for key, replacement in (("decision", "do-not-run"), ("control_completed_evidence", {"wrong": "control"}),
                    ("budget", {"wrong": "budget"}), ("one_model_invocation", False),
                    ("no_threshold_or_loss_tuning", False), ("rationale", " ")):
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, "POSTERIOR_DECISION"):
                        stage.validate_decision({**value, key: replacement}, request)
                for key, replacement in (("status", "failed"), ("schema", "different-diagnostic"),
                    ("new_model_forwards", 1), ("new_model_constructions", 1),
                    ("inputs", {**summary_data["inputs"], "augmentation": {"manifest_sha256": "0" * 64}}),
                    ("workload", {**summary_data["workload"], "new_numeric_conditions": 569})):
                    summary.write_text(json.dumps({**summary_data, key: replacement}))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, "POSTERIOR"):
                        stage.validate_decision({**value, "posterior_summary": stage.engine.pin(summary)}, request)
                summary.write_text('{"status":"changed"}\n')
                with self.assertRaisesRegex(ValueError, "PIN_CHANGED"):
                    stage.validate_decision(value, request)

    def test_preview_does_not_grant_execution(self):
        control = fixture()
        request = {"schema": stage.REQUEST, "control": {}, "budget": {"path": "/invented/budget.json"}, "decision": None}
        with patch.object(stage, "control_evidence", return_value=(Path("/invented/output"), control, {})), \
            patch.object(stage.engine, "checked", side_effect=lambda v: Path(v["path"])), \
            patch.object(stage.training, "read_json", return_value=budget()), \
            patch.object(stage, "source_hashes", return_value={"explicit-new-source": "2" * 64}), \
            patch.object(stage.engine, "pin", return_value={"invented": "pin"}):
            preview = stage.declare(request, allow_pending=True)
            self.assertFalse(preview["execution_authorized"])
            for key in ("train", "development", "evaluation_rows", "all_new_rows", "excluded_original_rows",
                "new_zero_loss_rows", "initial_checkpoint", "optimizer"):
                self.assertEqual(preview[key], control[key])
            with self.assertRaisesRegex(ValueError, "FIXED_CONTRACT"):
                stage.validate_contract(preview)
            with self.assertRaisesRegex(ValueError, "POSTERIOR_DECISION_REQUIRED"):
                stage.declare(request)

    def test_redeclare_rejects_changed_rows_initial_or_sampling(self):
        control = fixture()
        job = {**control, **stage.contract(), "job_id": stage.JOB_ID, "inputs": {"fixture": True}}
        for field, replacement in (("train", []), ("initial_checkpoint", {"changed": True}),
            ("exposure_plan", {"draws": []})):
            with self.subTest(field=field), patch.object(stage, "declare", return_value=job), \
                self.assertRaisesRegex(ValueError, "RECIPE_CHANGED"):
                stage.validate_recipe({**job, field: replacement})

    def test_legacy_plan_and_default_engine_policy_remain_original(self):
        control = fixture()
        new_ids = stage.validate_roster(control)
        original = stage.engine.exposure_plan(control["train"])
        stage.exposure_plan(control["train"], new_ids)
        self.assertEqual(original, stage.engine.exposure_plan(control["train"]))
        self.assertEqual(Counter(d["source_id"] in new_ids for d in original), {False: 4348, True: 3652})
        policy = stage.engine.execution_policy()
        self.assertEqual(policy.schema, stage.engine.SCHEMA)
        self.assertEqual(policy.code, stage.engine.CODE)
        self.assertIs(policy.validate_recipe, stage.engine.validate_recipe)
        with patch.object(stage.training, "read_json", return_value=control), \
            patch.object(stage.training, "sha_file", return_value="0" * 64):
            queue = stage.engine.queue_definition("/invented/recipe.json", "/invented/python")
        self.assertEqual(queue["jobs"][0]["argv"][3], str(ROOT / stage.engine.SOURCE))
        self.assertEqual(queue["global_budget_minutes"], 15)

    def test_allocation_queue_uses_explicit_entry_point_with_same_limits(self):
        job = {**fixture(), **stage.contract(), "job_id": stage.JOB_ID}
        with patch.object(stage.training, "read_json", return_value=job), \
            patch.object(stage.training, "sha_file", return_value="0" * 64):
            queue = stage.engine.queue_definition("/invented/recipe.json", "/invented/python",
                policy=stage.execution_policy())
        self.assertEqual(queue["jobs"][0]["argv"][3], str(ROOT / stage.SOURCE))
        self.assertEqual(queue["jobs"][0]["timeout_minutes"], 15)
        self.assertEqual(queue["jobs"][0]["job_id"], stage.JOB_ID)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
