#!/usr/bin/env python3
"""Pure metadata and invented call fixtures; no model or tensor runtime."""
from __future__ import annotations

import copy
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_context52_known_silence as stage
from test_research_c1_context_development import fixture as dev_fixture, invented_observations


def fixture():
    vocabulary, dev = dev_fixture()
    rows = []
    for i in range(52):
        kind = i % 2 if i < 34 else 2
        row = copy.deepcopy(dev[kind])
        weight = 1.0 if i < 6 else .5
        row.update(source_id="train-%02d" % i, source_family_id="train-family-%d" % (i // 3),
            speaker_id="train-speaker-%d" % (i // 3), split="train", source_role="train-candidate",
            intended_keyword_id=kind + 1 if kind < 2 else None, ctc_weight=weight,
            keyword_weight=[weight, weight], pcm_sha256=stage.fit.digest(["invented-train", i]))
        rows.append(row)
    for i, row in enumerate(dev):
        row["intended_keyword_id"] = i % 3 + 1 if i % 3 < 2 else None
    excluded = []
    for i in range(2):
        row = copy.deepcopy(rows[i])
        row.update(source_id="failed-%d" % i, ctc_mask=False, ctc_weight=0.0, full_text=None, target_ids=None,
            generation_outcome="failed-native", keyword_mask=[False, False], keyword_targets=[None, None], keyword_weight=[0.0, 0.0])
        excluded.append(row)
    return vocabulary, rows, dev, rows + excluded + dev


def recipe(rows, dev):
    return {"schema": stage.SCHEMA, **stage.fixed_contract(), "inputs": {}, "train": rows,
        "development": dev, "exposure_plan": {"draws": stage.exposure_plan(rows)}}


def ledger(job):
    events = []
    for step in range(stage.STEPS):
        recorded = []
        for draw in job["exposure_plan"]["draws"][step * 4:(step + 1) * 4]:
            row = job["train"][draw["row_index"]]
            ctc, aux = row["ctc_weight"] / 4, .25 if draw["offset_ms"] else 0.0
            recorded.append({**draw, "label_binding_sha256": stage.fit.row_identity(row),
                **{k: row[k] for k in ("full_text", "target_ids", "pcm_sha256", "source_family_id", "ctc_mask")},
                "keyword_loss_used": False, "auxiliary_scale": 1.0, "raw_length_normalized_ctc": 1.0,
                "weighted_batch_contribution": ctc, "aux_blank_nll_mean": aux * 4,
                "aux_batch_contribution": aux, "total_batch_contribution": ctc + aux})
        primary = sum(r["weighted_batch_contribution"] for r in recorded)
        auxiliary = sum(r["aux_batch_contribution"] for r in recorded)
        events.append({"optimizer_step": step + 1, "draw_offset": step * 4, "draws": recorded,
            "gradient_norm_before_clip": 1.0, "loss": primary + auxiliary,
            "fulltext_ctc_batch_loss": primary, "aux_batch_loss": auxiliary})
    return events


class MetadataTests(unittest.TestCase):
    def test_no_tensor_runtime_is_imported(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_fixed_contract_rejects_budget_loss_tuning_and_type_changes(self):
        original = {"schema": stage.SCHEMA, **stage.fixed_contract()}
        stage.validate_contract(original)
        for key, value in (("training_rows", 17), ("training_forward_limit", 8001), ("optimizer_steps", 1999),
                ("worker_seconds", 600), ("queue_minutes", 11), ("batch_size", 8), ("auxiliary_scale", .5),
                ("final_forward_limit", 211), ("before_forward_calls", 1), ("development_optimizer_draws", 1),
                ("automatic_retry", True), ("threshold_selection", True), ("independent_test", True),
                ("single_variable_data_size_ablation", True), ("training_ledger_replays_complete_optimization", True),
                ("offsets_ms", [False, 200, 500]), ("known_zero_frames", {"0": False, "200": 9, "500": 24})):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.validate_contract(changed)

    def test_all52_original_train_and_all18_development_are_required(self):
        vocabulary, rows, dev, all_rows = fixture()
        admitted, excluded = stage.validate_roster(all_rows, dev, vocabulary, [])
        self.assertEqual(admitted, rows)
        self.assertEqual(len(excluded), 2)
        for mutate in (lambda rs: rs.pop(), lambda rs: rs[0].update(split="development"),
                lambda rs: rs[0].update(source_role="development-only"), lambda rs: rs[0].update(ctc_weight=0),
                lambda rs: rs[34].update(full_text="", target_ids=[]), lambda rs: rs[0].update(target_ids=[1]),
                lambda rs: rs[0].update(expected_c1_frames=100), lambda rs: rs[0].update(keyword_mask=[False, True]),
                lambda rs: rs[52].update(ctc_weight=.5), lambda rs: rs[0].update(ctc_minimum_frames=100)):
            changed = copy.deepcopy(all_rows)
            mutate(changed)
            with self.assertRaises(ValueError):
                stage.validate_roster(changed, dev, vocabulary, [])

    def test_source_identity_overlap_is_rejected_before_any_training(self):
        vocabulary, rows, dev, all_rows = fixture()
        for key in ("source_family_id", "speaker_id", "pcm_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ORIGINAL_TINY_SOURCE_OVERLAP"):
                stage.validate_roster(all_rows, dev, vocabulary, [{"source_family_id": "other-family",
                    "speaker_id": "other-speaker", key: rows[0][key]}])
        dev[0]["reference_pcm_sha256"] = rows[0]["pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "TRAIN_SOURCE_OVERLAP"):
            stage.validate_roster(all_rows, dev, vocabulary, [])

    def test_exposure_plan_has_fixed_row_visits_offsets_and_no_dev_draws(self):
        _, rows, dev, _ = fixture()
        draws = stage.exposure_plan(rows)
        self.assertEqual(len(draws), 8000)
        self.assertEqual(Counter(d["offset_ms"] for d in draws), {0: 2696, 200: 2652, 500: 2652})
        counts = Counter(d["source_id"] for d in draws)
        self.assertEqual([counts[r["source_id"]] for r in rows], [154] * 44 + [153] * 8)
        for i, row in enumerate(rows):
            visits = [d for d in draws if d["source_id"] == row["source_id"]]
            self.assertEqual([d["row_visit_index"] for d in visits], list(range(len(visits))))
            self.assertEqual([d["offset_ms"] for d in visits], [stage.onset.OFFSETS[j % 3] for j in range(len(visits))])
        with self.assertRaisesRegex(ValueError, "TRAIN_ONLY"):
            stage.exposure_plan(rows[:-1] + dev[:1])

    def test_prerequisite_uses_completion_and_saved_audit_not_accuracy(self):
        summary = {"schema": stage.diagnostic.SCHEMA, **stage.diagnostic.fixed_contract(), "status": "completed",
            "started_forward_calls": 108, "completed_forward_calls": 108, "model_constructions": 2, "model_states_unchanged": True}
        audit = {"schema": stage.diagnostic.SCHEMA, "status": "verified", "saved_conditions_verified": 108,
            "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0}
        stage.validate_prerequisite(summary, audit)
        stage.validate_prerequisite({**summary, "conditions": {"invented_accuracy": 0.0}}, audit)
        stage.validate_prerequisite({**summary, "conditions": {"invented_accuracy": 1.0}}, audit)
        for changed_summary, changed_audit in (({**summary, "status": "failed"}, audit),
                ({**summary, "completed_forward_calls": 107}, audit), (summary, {**audit, "status": "running"}),
                (summary, {**audit, "saved_conditions_verified": 107}), (summary, {**audit, "new_model_forwards": 1})):
            with self.assertRaises(ValueError):
                stage.validate_prerequisite(changed_summary, changed_audit)

    def test_recipe_pins_complete_rows_masks_order_checkpoint_and_types(self):
        _, rows, dev, _ = fixture()
        original = {**recipe(rows, dev), "initial_checkpoint": {"sha256": "a" * 64}}
        with patch.object(stage, "assemble", return_value=(original, {})):
            stage.validate_recipe(copy.deepcopy(original))
            for mutate in (lambda job: job["train"].reverse(), lambda job: job["development"].reverse(),
                    lambda job: job["train"][0].update(full_text="改写"), lambda job: job["train"][0].update(ctc_weight=True),
                    lambda job: job["train"][0].update(keyword_mask=[False, True]),
                    lambda job: job["initial_checkpoint"].update(sha256="b" * 64)):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "RECIPE_OR_INPUT_CHANGED"):
                    stage.validate_recipe(changed)

    def test_loss_audit_rejects_cancelling_per_draw_error_and_rewritten_source(self):
        _, rows, dev, _ = fixture()
        job = recipe(rows, dev)
        events = ledger(job)
        self.assertEqual(stage.ledger_readback(events, job, rows)["draw_contributions_checked"], 8000)
        changed = copy.deepcopy(events)
        changed[0]["draws"][0]["total_batch_contribution"] += .1
        changed[0]["draws"][1]["total_batch_contribution"] -= .1
        with self.assertRaisesRegex(ValueError, "EACH_DRAW_TOTAL_CHANGED"):
            stage.ledger_readback(changed, job, rows)
        changed = copy.deepcopy(events)
        changed[0]["draws"][0]["full_text"] = "not the declared full transcript"
        with self.assertRaisesRegex(ValueError, "DRAW_SOURCE_CHANGED"):
            stage.ledger_readback(changed, job, rows)

    def test_identical_shape_parameter_name_swap_cannot_self_certify(self):
        names = ["block0.bias", "block1.bias"]
        swapped = list(reversed(names))
        groups = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01,
            "amsgrad": False, "maximize": False, "foreach": None, "capturable": False, "differentiable": False,
            "fused": None, "decoupled_weight_decay": True, "params": [0, 1]}
        saved = {"param_groups": [groups], "state": {0: {}, 1: {}}}
        with self.assertRaisesRegex(ValueError, "ADAMW_PARAMETER_MAP"):
            stage.optimizer_readback(saved, {name: "invented same-shape bias" for name in swapped}, swapped, names, None)

    def test_final_roster_is210_split_conditions_without_tiny8_gates(self):
        vocabulary, rows, dev, _ = fixture()
        observed = []
        for row in rows + dev:
            for item in invented_observations([row], vocabulary)[:3]:
                observed.append({**item, "split": row["split"]})
        result = stage.conditions(observed, {"train": rows, "development": dev})
        self.assertEqual(set(result), {"train", "development"})
        self.assertEqual([result["train"][str(ms)]["known_zero_frames"] for ms in (0, 200, 500)], [0, 468, 1248])
        self.assertEqual([result["development"][str(ms)]["known_zero_frames"] for ms in (0, 200, 500)], [0, 162, 432])
        self.assertNotIn("strict_tiny_fit_passed", result)
        for changed in (observed[:-1], observed + observed[:1], list(reversed(observed))):
            with self.assertRaisesRegex(ValueError, "ALL_FINAL_CONDITIONS"):
                stage.conditions(changed, {"train": rows, "development": dev})

    def test_queue_request_is_one_bounded_job_and_does_not_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recipe.json"
            stage.io.write_json(path, {"schema": stage.SCHEMA, **stage.fixed_contract(), "heavy_lock": "/fixture.lock"})
            request = stage.queue_definition(path, "/fixed-python")
            self.assertEqual(request["global_budget_minutes"], 10)
            self.assertEqual(len(request["jobs"]), 1)
            self.assertEqual(request["jobs"][0]["timeout_minutes"], 10)
            self.assertEqual(request["jobs"][0]["depends_on"], [])
            self.assertIn(stage.SOURCE, request["jobs"][0]["argv"][3])
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_unqueued_call_fails_before_output_or_model_runtime(self):
        @contextmanager
        def invented_lease(_lock):
            yield SimpleNamespace(supervised=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, lock, output = root / "recipe.json", root / "fixture.lock", root / "outputs"
            stage.io.write_json(path, {"schema": stage.SCHEMA, **stage.fixed_contract(), "heavy_lock": str(lock)})
            with patch.object(stage.training, "heavy_model_slot", invented_lease), self.assertRaisesRegex(ValueError, "DURABLE_QUEUE_REQUIRED"):
                stage.run(path, output, lock)
            self.assertFalse(output.exists())
            self.assertNotIn("torch", sys.modules)

    def test_failed_terminal_status_or_queue_cannot_pass_audit_with_stale_completed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "jobs" / stage.JOB_ID / "outputs"
            output.mkdir(parents=True)
            summary = {"status": "completed", "recipe_sha256": "a" * 64}
            stage.io.write_json(output / "status.json", {**summary, "status": "failed"})
            with self.assertRaisesRegex(ValueError, "DIVERGENT_TERMINAL_STATUS"):
                stage.completed_queue(output, summary)
            (output / "status.json").unlink()
            stage.io.write_json(output / "status.json", summary)
            stage.io.write_json(output.parent / "result.json", {"schema": stage.io.RECEIPT_SCHEMA, "status": "failed"})
            with self.assertRaisesRegex(ValueError, "COMPLETED_QUEUE_RECEIPT_REQUIRED"):
                stage.completed_queue(output, summary)
            self.assertNotIn("torch", sys.modules)


class CallAccountingTests(unittest.TestCase):
    class Feature:
        def unsqueeze(self, _dimension):
            return self
        def equal(self, value):
            return value is self

    def test_forward_proxy_calls_once_and_failure_retains_started_claim(self):
        _, rows, dev, _ = fixture()
        feature = self.Feature()
        calls = []
        def underlying(value):
            calls.append(value)
            return "invented logits object; no tensor runtime"
        evidence = [{"source_id": rows[0]["source_id"], "offset_ms": 0, "feature_sha256": "a" * 64}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage.diagnostic, "check_logits"):
            output = Path(temporary)
            proxy = stage.TrainingCalls(underlying, {(0, 0): feature}, evidence, rows, recipe(rows, dev), output, None)
            result = proxy(feature)
            self.assertEqual(result, "invented logits object; no tensor runtime")
            self.assertEqual(calls, [feature])
            self.assertEqual(proxy.count, 1)
            self.assertEqual([r["phase"] for r in stage.context.lines(output / "forward-calls.jsonl")], ["started", "completed"])
        def fails_after_entering(_value):
            raise RuntimeError("invented interrupted forward")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            proxy = stage.TrainingCalls(fails_after_entering, {(0, 0): feature}, evidence, rows, recipe(rows, dev), output, None)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                proxy(feature)
            self.assertEqual(proxy.count, 0)
            self.assertEqual([r["phase"] for r in stage.context.lines(output / "forward-calls.jsonl")], ["started"])
            self.assertNotIn("torch", sys.modules)

    def test_optimizer_proxy_requires_four_completed_calls_and_preserves_failed_intent(self):
        class Optimizer:
            def step(self):
                raise RuntimeError("invented optimizer interruption")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            forward = SimpleNamespace(count=3)
            proxy = stage.OptimizerCalls(Optimizer(), forward, output)
            with self.assertRaisesRegex(ValueError, "OPTIMIZER_CALL_ORDER"):
                proxy.step()
            self.assertFalse((output / "optimizer-calls.jsonl").exists())
            forward.count = 4
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                proxy.step()
            self.assertEqual(proxy.count, 0)
            self.assertEqual(stage.context.lines(output / "optimizer-calls.jsonl"),
                [{"optimizer_step": 1, "completed_training_forwards": 4, "phase": "started"}])


if __name__ == "__main__":
    unittest.main()
