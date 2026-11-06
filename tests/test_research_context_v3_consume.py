#!/usr/bin/env python3
"""Frozen cohort/exclusion and real queue/Gate tests; no model constructions."""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools"), str(ROOT / "tests")]
import research_context_v3_consume as probe
import research_context_v3_inputs as inputs
import research_d1_vocab as vocab
from test_research_context_v3_inputs import observed_row, ordinary_row, TOKENS
from test_research_d1_fit_diagnostic import base_recipe

q, cp = probe.training.io_contract, probe.checkpoints
REAL_STAGE_SOURCES = probe.stage_sources
SAVED = "--saved-tensors" in sys.argv
if SAVED:
    sys.argv.remove("--saved-tensors")


def fixture_plan():
    references, cells = [], []
    for index in range(24):
        family, speaker = f"fam-{index:02d}", f"spk-{index:02d}"
        role = "train-candidate" if index < 18 else "development-only"
        references.append({"source_family_id": family, "speaker_group_id": speaker, "role": role})
        for kind in range(3):
            cells.append({"cell_id": f"bulk-fixture-{index:02d}-{kind}", "source_family_id": family,
                "speaker_group_id": speaker, "role": role,
                "generation_category": "ordinary-negative" if kind == 2 else "full-positive"})
    # Cell order need not equal family lexical order, and ASR never chooses it.
    cells = cells[9:] + cells[:9]
    return {"schema": "kws-context-supplement-plan-v3", "references": references, "cells": cells}


def fixture_rows(plan):
    vocabulary = vocab.make_vocabulary(TOKENS.read_bytes())
    rows = []
    for cell in plan["cells"]:
        kind = int(cell["cell_id"][-1])
        row = ordinary_row() if kind == 2 else observed_row("你好小窝请把灯打开" if kind == 0 else "小窝小窝请把门关上")
        row.update(source_id=cell["cell_id"], source_family_id=cell["source_family_id"],
            speaker_id=cell["speaker_group_id"], source_role=cell["role"],
            split="train" if cell["role"] == "train-candidate" else "development", label_reason="fixture-machine-agreement")
        if kind == 1:
            row["intended_keyword_id"] = 2
        row["global_source_key"] = [row["source_plan_sha256"], row["source_id"]]
        rows.append(inputs.project_row(row, vocabulary))
    return rows


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.plan = fixture_plan()
        self.selection = probe.selection_layout(self.plan)
        self.rows = fixture_rows(self.plan)

    def test_pre_asr_family_order_controls_exact_18_plus_18_and_100_train_draws(self):
        train, dev, ledger = probe.enrollment(list(reversed(self.rows)), self.selection, self.plan)
        self.assertEqual([r["source_id"] for r in train], self.selection["train_declared_source_ids"])
        self.assertEqual([r["source_id"] for r in dev], self.selection["development_declared_source_ids"])
        self.assertEqual((len(train), len(dev), len(ledger)), (18, 18, 72))
        recipe = probe.probe_recipe(train, base_recipe())
        self.assertEqual((recipe["optimizer_steps"], len(recipe["exposure_plan"]["draws"])), (25, 100))
        draws = {d["source_id"] for d in recipe["exposure_plan"]["draws"]}
        self.assertEqual(draws, set(self.selection["train_declared_source_ids"]))
        self.assertFalse(draws & set(self.selection["development_declared_source_ids"]))
        ordinary = [r for r in train if r["generation_category"] == "ordinary-negative"]
        self.assertEqual(len(ordinary), 6)
        self.assertTrue(all(r["target_ids"] and len(r["target_ids"]) == len(r["full_text"]) > 4 and
                            r["keyword_targets"] == [0, 0] and 0 not in r["target_ids"] for r in ordinary))
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_OPTIMIZER_FORBIDDEN"):
            probe.probe_recipe(train + dev[:1], base_recipe())

    def test_unknown_failed_oov_and_missing_stay_excluded_without_replacement(self):
        selected = self.selection["train_declared_source_ids"]
        changed = {selected[0]: "failed-native", selected[1]: "uncertain", selected[2]: "oov"}
        rows = []
        for original in self.rows:
            row = copy.deepcopy(original)
            if row["source_id"] == selected[3]:
                continue
            if row["source_id"] in changed:
                kind = changed[row["source_id"]]
                row.update(ctc_mask=False, ctc_weight=0, target_ids=None, quarantined=True,
                    keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0],
                    ctc_reason="full_transcript_out_of_vocabulary" if kind == "oov" else "no_complete_observed_transcript")
                if kind != "oov": row["label_status"] = "uncertain"
                if kind == "failed-native": row.update(generation_outcome="failed-native", full_text=None)
            rows.append(row)
        train, dev, ledger = probe.enrollment(rows, self.selection, self.plan)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(dev), 18)
        self.assertEqual(len(ledger), 72)
        self.assertTrue(set(r["source_id"] for r in train) <= set(selected))
        audit = {r["source_id"]: r for r in ledger}
        self.assertFalse(audit[selected[3]]["observed"])
        self.assertEqual(audit[selected[3]]["reason"], "unobserved_no_audio_or_label")
        self.assertTrue(all(not audit[i]["optimizer_allowed"] and audit[i]["ctc_weight"] == 0 for i in selected[:4]))
        self.assertEqual(audit[selected[0]]["generation_outcome"], "failed-native")

    def test_all_18_dev_remain_evaluation_only_when_zero_are_eligible(self):
        for row in self.rows:
            if row["split"] == "development":
                row.update(ctc_mask=False, ctc_weight=0, target_ids=None, quarantined=True,
                    label_status="uncertain", ctc_reason="no_complete_observed_transcript")
        train, dev, ledger = probe.enrollment(self.rows, self.selection, self.plan)
        self.assertEqual((len(train), len(dev)), (18, 0))
        planned_dev = [r for r in ledger if r["planned_use"] == "evaluation"]
        self.assertEqual(len(planned_dev), 18)
        self.assertTrue(all(not r["optimizer_allowed"] and not r["evaluation_allowed"] for r in planned_dev))
        self.assertEqual(probe.observed_summary([]), {"eligible_rows": 0, "status": "no_eligible_rows", "metrics": None})

    def test_missing_selected_ordinary_cannot_use_other_families_to_fill_coverage(self):
        rows = [r for r in self.rows if not (r["source_id"] in self.selection["train_declared_source_ids"] and
                                            r["generation_category"] == "ordinary-negative")]
        self.assertTrue(any(r["generation_category"] == "ordinary-negative" and r["split"] == "train" for r in rows))
        with self.assertRaisesRegex(ValueError, "PREDECLARED_COVERAGE_UNAVAILABLE"):
            probe.enrollment(rows, self.selection, self.plan)

    def test_split_speaker_family_and_duplicate_aliases_are_rejected(self):
        for key, value in (("speaker_id", "dev-speaker"), ("source_role", "development-only"), ("split", "development")):
            rows = copy.deepcopy(self.rows); rows[0][key] = value
            with self.assertRaisesRegex(ValueError, "OBSERVED_ROLE_CHANGED"):
                probe.enrollment(rows, self.selection, self.plan)
        with self.assertRaisesRegex(ValueError, "OBSERVED_ROSTER"):
            probe.enrollment(self.rows + self.rows[:1], self.selection, self.plan)
        plan = copy.deepcopy(self.plan)
        plan["references"][-1]["speaker_group_id"] = plan["references"][0]["speaker_group_id"]
        with self.assertRaisesRegex(ValueError, "FAMILY_SPLIT_CHANGED"):
            probe.selection_layout(plan)


class OriginalIdentityTests(unittest.TestCase):
    def test_predeclared_selection_rejects_family_budget_optimizer_and_dev_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vocabulary.json").write_text('{"explicit_fixture":true}\n')
            plan = fixture_plan()
            plan["vocabulary"] = {"file_sha256": q.file_record(root / "vocabulary.json")["sha256"]}
            plan_path = root / "plan.json"; q.write_json(plan_path, plan)
            original = {"manifest_sha256": "a" * 64, "initial": {"file_sha256": "b" * 64},
                        "base_recipe": {**base_recipe(), "seed": probe.SEED}}
            request = {"anchor": str(root), "anchor_manifest_sha256": "a" * 64, "plan": probe.pin(plan_path)}
            selection = {"schema": probe.SELECTION, **probe.selection_layout(plan), "candidate": probe.CANDIDATE,
                "optimizer_steps": 25, "sample_exposures": 100, "batch_size": 4, "seed": probe.SEED,
                "worker_timeout_seconds": 120, "queue_timeout_minutes": 10, "automatic_retries": 0,
                "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01},
                "family_rule": "first-six-lexicographically-sorted-declared-train-families; all-six-declared-development-families",
                "required_train_coverage": ["observed-K1", "observed-K2", "ordinary-negative-full-text"],
                "human_gold": False, "quality_acceptance_claim": False, "new_model_calls": 0,
                "selection_used_asr_results": False, "selection_used_model_predictions": False,
                "plan_canonical_sha256": probe.digest(plan), "plan_file_sha256": request["plan"]["sha256"],
                "anchor_manifest_sha256": "a" * 64,
                "full2601_vocabulary_file_sha256": plan["vocabulary"]["file_sha256"],
                "initial_checkpoint": {"path": str(root / "initial/c1-ctc2601.pt"), "sha256": "b" * 64},
                "created_at_utc": "2026-11-06T00:00:00+00:00"}
            probe.validate_selection(selection, plan, original, request)
            mutations = [lambda v: v.update(optimizer_steps=26), lambda v: v.update(selection_used_asr_results=True),
                lambda v: v["optimizer"].update(lr=.0001), lambda v: v["train_source_family_ids"].reverse(),
                lambda v: v["development_declared_source_ids"].pop(),
                lambda v: v.update(train_declared_source_ids=v["development_declared_source_ids"])]
            for mutate in mutations:
                value = copy.deepcopy(selection); mutate(value)
                with self.assertRaisesRegex(ValueError, "PREDECLARED_SELECTION_CHANGED"):
                    probe.validate_selection(value, plan, original, request)

    def test_original_algorithm_and_effective_runtime_may_not_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); anchor = root / "anchor"
            name = "configs/runtime/research-runtime.json"
            old = {"python": {"version": "3.12.16"}, "roles": {"core": {"packages": {"torch": "2.13.0+cpu"}}},
                   "validation": {"old": True}}
            for base in (root, anchor / "execution-source"):
                (base / name).parent.mkdir(parents=True); (base / name).write_text(json.dumps(old))
            current = {**old, "validation": {"new-evidence-metadata": True}}
            (root / name).write_text(json.dumps(current))
            records = {"training/frontend.py": "a" * 64, name: "new-runtime-hash"}
            original = {"source_hashes": {**records, name: "old-runtime-hash"}}
            with mock.patch.object(probe, "ROOT", root), mock.patch.object(probe.training, "source_records", return_value=records):
                probe.verify_original_implementation(original, anchor)
                original["source_hashes"]["training/frontend.py"] = "b" * 64
                with self.assertRaisesRegex(ValueError, "ORIGINAL_ALGORITHM_SOURCE_CHANGED"):
                    probe.verify_original_implementation(original, anchor)
                original["source_hashes"]["training/frontend.py"] = "a" * 64
                current["python"] = {"version": "3.12.17"}; (root / name).write_text(json.dumps(current))
                with self.assertRaisesRegex(ValueError, "ORIGINAL_EFFECTIVE_RUNTIME_CHANGED"):
                    probe.verify_original_implementation(original, anchor)

    def test_source_contract_includes_every_executed_consumer_dependency(self):
        actual = probe.source_records()
        self.assertEqual(len(actual), len(probe.CODE))
        self.assertTrue(set(probe.historical.CODE) <= set(actual))
        self.assertIn("training/research_context_v3_consume.py", actual)
        self.assertIn("training/research_context_v3_inputs.py", actual)
        self.assertIn("tools/research_d2_checkpoints.py", actual)

    @unittest.skipUnless(SAVED, "explicit tensor-only readback check; no model execution")
    def test_saved_adamw_steps_defaults_parameter_map_and_moments(self):
        import torch
        parameters = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}
        names = list(parameters)
        spec = {"name": "AdamW", "lr": .001, "weight_decay": .01}
        group = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01,
            "amsgrad": False, "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None, "decoupled_weight_decay": True, "params": [0, 1]}
        saved = {"param_groups": [group], "state": {i: {"step": torch.tensor(25.), "exp_avg": torch.zeros_like(p),
                  "exp_avg_sq": torch.ones_like(p)} for i, p in enumerate(parameters.values())}}
        result = probe.optimizer_readback(saved, parameters, names, spec)
        self.assertEqual((result["steps_per_parameter"], result["parameter_names"]), (25, names))
        for mutate, reason in ((lambda x: x["param_groups"][0].update(lr=.002), "SETTINGS_CHANGED"),
            (lambda x: x["param_groups"][0].update(params=[1, 0]), "SETTINGS_CHANGED"),
            (lambda x: x["state"][0].update(step=torch.tensor(24.)), "STEP_CHANGED"),
            (lambda x: x["state"][1].update(exp_avg=torch.zeros(3)), "MOMENTS_CHANGED")):
            wrong = copy.deepcopy(saved); mutate(wrong)
            with self.assertRaisesRegex(ValueError, reason): probe.optimizer_readback(wrong, parameters, names, spec)


class QueueOwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        recipe = self.root / "recipe.json"; q.write_json(recipe, {"explicit_no_model_fixture": True})
        self.marker = self.root / "called-once.json"
        code = """import json,os,sys
from pathlib import Path
output=Path(sys.argv[1]);marker=Path(sys.argv[2])
value={'owner':os.environ['KWS_CONTEXT_V3_INGESTION_OWNER'],'intent':os.environ['KWS_CONTEXT_V3_INGESTION_INTENT'],'output':str(output)}
with marker.open('x') as stream:json.dump(value,stream)
(output/'proof.json').write_text(json.dumps(value));(output/'MANIFEST.json').write_text('{}')
"""
        request = {"schema": q.SCHEMA, "queue_id": "explicit-ingestion-owner-fixture", "global_budget_minutes": 1,
            "jobs": [{"job_id": probe.JOB_ID, "recipe": str(recipe), "recipe_sha256": q.file_record(recipe)["sha256"],
                "argv": [sys.executable, "-I", "-S", "-B", "-c", code, "{output_dir}", str(self.marker)],
                "cwd": str(self.root), "depends_on": [], "timeout_minutes": .1, "outputs": ["proof.json", "MANIFEST.json"]}]}
        q.write_json(self.root / "queue-request.json", request)
        self.lock = self.root / "heavy-model.lock"
        q.create(self.root / "queue", self.root / "queue-request.json", ROOT / "configs/research/active-plan-20261105.json", self.lock)
        self.value = {"schema": probe.STAGE, "root": str(self.root), "queue": probe.pin(self.root / "queue/queue.json"),
            "recipe": probe.pin(self.root / "queue/recipes" / (probe.JOB_ID + ".json")),
            "heavy_lock": str(self.lock), "runtime": {"executable": sys.executable}, "slot_wait_seconds": 60}
        self.path = self.root / "stage.json"; q.write_json(self.path, self.value)
        p = mock.patch.object(probe, "load_stage", return_value=self.value); p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(probe, "stage_sources", side_effect=lambda _: [self.path, *cp.regular_tree(self.root / "queue")])
        p.start(); self.addCleanup(p.stop)
        def audit_fixture(argv, **_kwargs):
            self.assertEqual(argv[3], "audit")
            self.assertTrue((self.root / "checkpoints/training-terminal/ack.json").is_file())
            q.write_json(Path(argv[argv.index("--receipt") + 1]), {"explicit_fixture": True, "new_model_forwards": 0})
            return SimpleNamespace(returncode=0, stdout=b'fixture audit\n', stderr=b'')
        self.audit = mock.patch.object(probe.subprocess, "run", side_effect=audit_fixture)
        self.audit.start(); self.addCleanup(self.audit.stop)
        self.gate = cp.Gate(max_wait_seconds=0, on_wait=self.save_fixture)

    @staticmethod
    def save_fixture(folder, descriptor):
        receipt = folder / "explicit-fixture-provider.json"
        q.write_json(receipt, {"schema": cp.SAVE, "status": "saved", "provider": "explicit-test-fixture",
            "object_id": "fixture-only", "provider_response": {"saved_id": "fixture-only"},
            "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
        cp.acknowledge(folder, receipt)

    def test_production_queue_expands_exact_output_dir_and_recipe(self):
        recipe = self.root / "recipe.json"
        definition = probe.queue_definition(recipe, sys.executable, self.lock)
        command = q.expanded_argv(self.root / "queue", definition["jobs"][0])
        self.assertEqual(command[command.index("--output") + 1], str(self.root / "queue/jobs" / probe.JOB_ID / "outputs"))
        self.assertEqual(command[command.index("--recipe") + 1], str(self.root / "queue/recipes" / (probe.JOB_ID + ".json")))
        self.assertFalse(any("{" in part for part in command))

    def test_capsule_keeps_both_required_original_and_queued_recipe_paths(self):
        prepared, anchor = self.root / "prepared", self.root / "anchor"
        prepared.mkdir(); anchor.mkdir()
        for name in ("MANIFEST.json", "preparation.json"):
            q.write_json(anchor / name, {"explicit_source_list_fixture": True})
        selection_checkpoint = self.root / "selection-checkpoint"; selection_checkpoint.mkdir()
        for name in ("capsule.json", "phase-marker.json", "provider-save.json", "ack.json"):
            q.write_json(selection_checkpoint / name, {"explicit_source_list_fixture": True})
        q.write_json(prepared / "request.json", {"schema": probe.PREP, **{key: probe.pin(self.path) for key in
            ("plan", "selection", "campaign", "campaign_validator")},
            "selection_checkpoint": str(selection_checkpoint), "anchor": str(anchor)})
        q.write_json(prepared / "campaign-closure.json", {"closures": []})
        value = {**self.value, "original_recipe": probe.pin(self.root / "recipe.json")}
        # This tests the concrete source inventory only; the real admission
        # and original-manifest validators have their own separate contracts.
        with mock.patch.object(probe, "validate_job", return_value={"prepared": str(prepared),
                 "prepared_manifest_sha256": "a" * 64}), mock.patch.object(probe.historical, "verify_files"):
            sources = REAL_STAGE_SOURCES(value)
        original, queued = self.root / "recipe.json", Path(value["recipe"]["path"])
        self.assertNotEqual(original, queued)
        self.assertEqual(probe.pin(original)["sha256"], probe.pin(queued)["sha256"])
        self.assertIn(original, sources); self.assertIn(queued, sources)
        folder = self.root / "source-list-capsule"
        descriptor = cp.create_capsule(folder, {"explicit_source_list_fixture": True}, sources)
        cp.verify_capsule(folder, current_sources=True)
        import zipfile
        with zipfile.ZipFile(descriptor["capsule"]["path"]) as archive:
            members = q.loads(archive.read("MANIFEST.json"))["files"].values()
        retained = {item["path"] for item in members}
        self.assertTrue({str(original), str(queued)} <= retained)

    def test_real_queue_runs_once_after_live_ack_and_terminal_is_saved_before_readback(self):
        result, code = probe.execute(self.path, checkpoint_gate=self.gate)
        self.assertEqual((result["status"], code), ("completed", 0))
        proof = json.loads(self.marker.read_text())
        self.assertEqual(proof["owner"], self.gate.owner)
        self.assertEqual(proof["intent"], str(self.root / "checkpoints/before-training"))
        for name in ("before-training", "training-terminal", "readback-complete"):
            cp.verify_ack(self.root / "checkpoints" / name)
        with mock.patch.object(q, "run", wraps=q.run) as launch:
            probe.execute(self.path, checkpoint_gate=cp.Gate(max_wait_seconds=0, on_wait=self.save_fixture))
            launch.assert_not_called()
        self.assertNotIn(probe.OWNER_ENV, os.environ)

    def test_old_pending_intent_cannot_be_reused_even_with_a_saved_ack(self):
        folder = self.root / "checkpoints/before-training"
        self.gate.checkpoint(folder, {**probe.stage_scope(self.path, self.value), "phase": "before-training"},
                             probe.stage_sources(self.value), intent=True)
        with mock.patch.object(q, "run", wraps=q.run) as launch:
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
                probe.execute(self.path, checkpoint_gate=cp.Gate(max_wait_seconds=0, on_wait=self.save_fixture))
            launch.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_missing_ack_blocks_all_queue_launches(self):
        with mock.patch.object(q, "run", wraps=q.run) as launch:
            with self.assertRaisesRegex(ValueError, "SAVE_NOT_ACKNOWLEDGED"):
                probe.execute(self.path, checkpoint_gate=cp.Gate(max_wait_seconds=0))
            launch.assert_not_called()
        self.assertFalse((self.root / "queue/jobs" / probe.JOB_ID / "claim.json").exists())

    def test_busy_slot_without_claim_keeps_same_live_gate_until_single_launch(self):
        real_run, calls = q.run, []
        def temporarily_busy(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                report = q.inspect(args[0]); report["status"] = "waiting_for_heavy_model_slot"
                return report, 75
            return real_run(*args, **kwargs)
        with mock.patch.object(q, "run", side_effect=temporarily_busy):
            result, code = probe.execute(self.path, checkpoint_gate=self.gate)
        self.assertEqual((code, len(calls), result["status"]), (0, 2, "completed"))
        self.assertEqual(json.loads(self.marker.read_text())["owner"], self.gate.owner)
        self.assertEqual(q.loads((self.root / "queue/jobs" / probe.JOB_ID / "claim.json").read_bytes())["attempt"], 1)

    def test_orphan_is_authenticated_and_durably_closed_without_rerun(self):
        root, _, request, queue_sha = q.load_queue(self.root / "queue")
        q.write_json(root / "jobs" / probe.JOB_ID / "claim.json", {"schema": "kws-local-research-job-claim-v1",
            **q.job_binding(request["jobs"][0], queue_sha), "attempt": 1, "automatic_retry": False, "explicit_orphan_fixture": True})
        with mock.patch.object(q, "run", wraps=q.run) as launch:
            result, code = probe.execute(self.path, checkpoint_gate=self.gate)
            launch.assert_not_called()
        self.assertEqual((result["status"], code), ("unknown", 1))
        context = cp.verify_capsule(self.root / "checkpoints/training-terminal")["context"]
        self.assertEqual(context["terminal_state"], "unknown")
        self.assertEqual(context["queue_result"], probe.pin(root / "jobs" / probe.JOB_ID / "result.json"))
        self.assertFalse(self.marker.exists())

    def test_worker_requires_exact_stage_queue_scope_and_inherited_job_paths(self):
        folder = self.root / "checkpoints/before-training"
        scope = {**probe.stage_scope(self.path, self.value), "phase": "before-training"}
        self.gate.checkpoint(folder, scope, probe.stage_sources(self.value), intent=True)
        recipe = Path(self.value["recipe"]["path"]); output = self.root / "queue/jobs" / probe.JOB_ID / "outputs"
        environment = {probe.OWNER_ENV: self.gate.owner, probe.INTENT_ENV: str(folder),
            "KWS_RESEARCH_JOB_ID": probe.JOB_ID, "KWS_RESEARCH_RECIPE": str(recipe), "KWS_RESEARCH_OUTPUT_DIR": str(output)}
        with mock.patch.dict(os.environ, environment):
            probe.verify_worker_intent(recipe, self.lock, output)
            with self.assertRaisesRegex(ValueError, "WORKER_INTENT_SCOPE_CHANGED"):
                probe.verify_worker_intent(recipe, self.lock, self.root / "wrong-output")
            self.value["queue"] = {**self.value["queue"], "sha256": "a" * 64}
            with self.assertRaisesRegex(ValueError, "WORKER_INTENT_SCOPE_CHANGED"):
                probe.verify_worker_intent(recipe, self.lock, output)

    def test_readback_timeout_is_saved_after_training_and_is_not_retried(self):
        with mock.patch.object(probe.subprocess, "run", side_effect=subprocess.TimeoutExpired(["fixture-readback"], 180)):
            result, code = probe.execute(self.path, checkpoint_gate=self.gate)
        self.assertEqual((result["status"], code), ("readback_failed", 1))
        cp.verify_ack(self.root / "checkpoints/training-terminal")
        cp.verify_ack(self.root / "checkpoints/readback-complete")
        self.assertEqual(q.loads((self.root / "readback-failure.json").read_bytes())["exit_code"], "timed_out")
        with mock.patch.object(q, "run", wraps=q.run) as launch, mock.patch.object(probe.subprocess, "run") as replay:
            result, code = probe.execute(self.path, checkpoint_gate=cp.Gate(max_wait_seconds=0, on_wait=self.save_fixture))
            launch.assert_not_called(); replay.assert_not_called()
        self.assertEqual((result["status"], code), ("readback_failed", 1))


if __name__ == "__main__":
    unittest.main()
