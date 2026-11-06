#!/usr/bin/env python3
"""No-model queue/Gate fixtures for the explicit metadata continuation.

The provider response below is deliberately a fixture in a temporary directory.
No fixture is written into an actual experiment/checkpoint root.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_context_v3_metadata_handoff as owner
q, cp = owner.q, owner.cp
RETAINED_CAMPAIGN = None
RETAINED_VALIDATOR = None


def fixture_save(folder, descriptor):
    receipt = folder.parent / (folder.name + "-explicit-fixture-save.json")
    q.write_json(receipt, {"schema": cp.SAVE, "status": "saved", "provider": "temporary-test-fixture",
        "object_id": "not-a-real-provider-object", "provider_response": {"fixture_object": "not-a-real-provider-object"},
        "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
    cp.acknowledge(folder, receipt)


class HandoffBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "continuation"; self.root.mkdir()
        self.job = "generate-005"
        self.stage_path = self.base / "stage.json"; q.write_json(self.stage_path, {"fixture": True})
        self.registration = self.base / "registration.json"; q.write_json(self.registration, {"fixture": True})
        self.parent_path = self.base / "campaign.json"
        self.original_budget = {"actual_known_job_nanoseconds": 0, "tts_claims_consumed": 40,
            "native_return_receipts": 40, "asr_claims_consumed": 71, "asr_results_saved": 71,
            "unresolved_jobs": [], "failed_jobs": []}
        initial = self.base / "initial.json"; q.write_json(initial, {"budget": self.original_budget})
        self.value = {"schema": owner.SCHEMA, "root": str(self.root), "registration": str(self.registration),
            "original_validator": owner.pin(self.stage_path),
            "initial_status": owner.pin(initial), "driver_wall_seconds": 5400,
            "heavy_lock": str(self.base / "heavy.lock"), "pipeline_files": {},
            "original_reader": {"root": str(self.base), "source_files_sha256": {}},
            "metadata_reader": {"revision": "context-v3-metadata-r2", "root": str(self.base), "source_files_sha256": {}}}
        self.parent = {"stage": owner.pin(self.stage_path), "asr_python": str(Path(sys.executable).absolute()),
            "pipeline_files": {}, "generation_queue": {"path": str(self.base / "unused-generation")}}
        q.write_json(self.parent_path, self.parent)
        self.value["parent_campaign"] = owner.pin(self.parent_path)
        self.handoff = self.root / "handoff.json"; q.write_json(self.handoff, self.value)
        self.old = SimpleNamespace(ROOT=self.base, budget=lambda _: copy.deepcopy(self.original_budget),
            validate_static=mock.Mock(), selective=SimpleNamespace(run=mock.Mock(return_value={"state": {"status": "complete"}})))
        self.stage = {"prepared": str(self.base), "queue_root": str(self.base / "unused-generation")}
        self.path = owner.where(self.value, self.job); self.path["chunk"].mkdir(parents=True)
        q.write_json(self.path["request"], {"fixture": "no-model-request"})
        declaration = self.path["chunk"] / "asr-queue-request.json"
        q.write_json(declaration, owner.asr_definition(self.value, self.parent, self.job))
        q.create(self.path["queue"], declaration, ROOT / "configs/research/active-plan-20261105.json", self.value["heavy_lock"])
        self.gate = cp.Gate(max_wait_seconds=0, on_wait=fixture_save)
        self.source_patch = mock.patch.object(owner, "sources", side_effect=lambda *_:
            [self.handoff, self.registration, *cp.regular_tree(self.path["chunk"])])
        self.source_patch.start(); self.addCleanup(self.source_patch.stop)

    def before_asr(self):
        owner.phase(self.gate, self.value, self.old, self.parent, self.stage, self.job, "before-asr")
        return self.root / "checkpoints" / self.job / "before-asr"

    def orphan(self):
        root, _, request, queue_sha = q.load_queue(self.path["queue"])
        job = request["jobs"][0]
        q.write_json(root / "jobs" / owner.ASR_JOB / "claim.json", {
            "schema": "kws-local-research-job-claim-v1", **q.job_binding(job, queue_sha),
            "attempt": 1, "automatic_retry": False, "explicit_orphan_fixture": True})

    def test_saved_ack_cannot_authorize_different_chunk_or_reader(self):
        folder = self.before_asr()
        scope = owner.phase_scope(self.value, self.parent, self.job, "before-asr")
        owner.verify_scope(folder, scope, intent=True, owner=self.gate.owner)
        for key, changed in (("job_id", "generate-006"), ("reader", {"revision": "context-v3-r1"})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ACK_SCOPE_CHANGED"):
                owner.verify_scope(folder, {**scope, key: changed}, intent=True, owner=self.gate.owner)

    def test_another_invocation_cannot_reuse_saved_pending_intent(self):
        self.before_asr()
        other = cp.Gate(max_wait_seconds=0, on_wait=fixture_save)
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
            owner.phase(other, self.value, self.old, self.parent, self.stage, self.job, "before-asr")
        self.assertFalse((self.path["queue"] / "jobs/asr-selective/claim.json").exists())

    def test_queue_argument_is_bound_to_new_guarded_owner_and_copied_recipe(self):
        root, _, request, _ = q.load_queue(self.path["queue"])
        argv = q.expanded_argv(root, request["jobs"][0])
        self.assertIn(str(ROOT / "tools/research_context_v3_metadata_handoff.py"), argv)
        self.assertIn("asr-worker", argv)
        self.assertEqual(argv[-1], str(root / "recipes/asr-selective.json"))
        self.assertEqual(request["jobs"][0]["timeout_minutes"], 10)

    def test_asr_worker_requires_live_owner_and_delegates_path_not_namespace(self):
        folder = self.before_asr()
        request = self.path["queue"] / "recipes/asr-selective.json"
        with mock.patch.object(owner, "static", return_value=self.value), \
             mock.patch.object(owner, "original_module", return_value=self.old), \
             mock.patch.object(owner, "dependency_provenance"), \
             mock.patch.dict(os.environ, {owner.ASR_OWNER_ENV: self.gate.owner, owner.ASR_INTENT_ENV: str(folder),
                "KWS_RESEARCH_RECIPE_SHA256": q.file_record(request)["sha256"], "KWS_RESEARCH_JOB_ID": owner.ASR_JOB}):
            result = owner.asr_worker(self.handoff, self.job, request)
            self.assertEqual(result["state"]["status"], "complete")
            self.old.selective.run.assert_called_once_with(request)
            with mock.patch.dict(os.environ, {owner.ASR_OWNER_ENV: "different-owner"}), \
                 self.assertRaisesRegex(ValueError, "ACK_SCOPE_CHANGED"):
                owner.asr_worker(self.handoff, self.job, request)
        self.old.selective.run.assert_called_once()

    def test_missing_own_source_member_is_rejected_even_with_genuine_ack(self):
        folder = self.before_asr()
        changed = copy.deepcopy(self.value)
        changed["pipeline_files"] = {"tools/research_context_v3_metadata_handoff.py": q.file_record(Path(owner.__file__))}
        with self.assertRaisesRegex(ValueError, "INTENT_SOURCE_MEMBERSHIP"):
            owner.refresh_intent(folder, changed, self.old, self.parent, self.stage, self.job, "before-asr", self.gate.owner)

    def test_busy_slot_retains_same_live_intent_without_new_claim(self):
        folder = self.before_asr(); report = q.inspect(self.path["queue"])
        waiting = {**report, "status": "waiting_for_heavy_model_slot"}
        returned = {**report, "status": "fixture_no_model_completed"}
        owners = []
        def refresh():
            owner.refresh_intent(folder, self.value, self.old, self.parent, self.stage, self.job, "before-asr", self.gate.owner)
            owners.append(self.gate.owner)
        with mock.patch.object(q, "run", side_effect=[(waiting, 75), (returned, 0)]) as execute, \
             mock.patch.object(owner.time, "sleep"):
            result, code = owner.queued_attempt(self.path["queue"], self.value, self.old, self.parent,
                time.monotonic() + 1200, owner.ASR_JOB, refresh)
        self.assertEqual((result["status"], code), ("fixture_no_model_completed", 0))
        self.assertEqual(owners, [self.gate.owner, self.gate.owner]); self.assertEqual(execute.call_count, 2)
        self.assertFalse((self.path["queue"] / "jobs/asr-selective/claim.json").exists())

    def test_deadline_after_saved_intent_never_calls_queue(self):
        self.before_asr()
        with mock.patch.object(q, "run") as execute, self.assertRaisesRegex(ValueError, "LAUNCH_BUDGET"):
            owner.queued_attempt(self.path["queue"], self.value, self.old, self.parent,
                time.monotonic() + 300, owner.ASR_JOB, lambda: None)
        execute.assert_not_called()

    def test_interrupted_claim_authenticates_unknown_before_owner_terminal(self):
        folder = self.before_asr()
        def interrupted(*_args, **_kwargs):
            self.orphan()
            raise KeyboardInterrupt("explicit interrupted claimed-queue fixture")
        with mock.patch.object(q, "run", side_effect=interrupted):
            try:
                owner.queued_attempt(self.path["queue"], self.value, self.old, self.parent,
                    time.monotonic() + 1200, owner.ASR_JOB, lambda: cp.verify_capsule(folder, current_sources=True))
            except KeyboardInterrupt as error:
                owner.persist_exception(self.gate, self.value, self.old, self.parent, self.stage, self.job, "asr", error)
            else:
                self.fail("fixture must interrupt after its claim")
        result = owner.read(self.path["queue"] / "jobs/asr-selective/result.json")
        self.assertEqual(result["status"], "unknown"); self.assertIsNone(result["actual_elapsed_ns"])
        terminal = self.root / "checkpoints" / self.job / "asr-terminal"
        cp.verify_ack(terminal)
        self.assertEqual(owner.read(terminal / "capsule.json")["context"]["result"],
                         owner.pin(self.path["queue"] / "jobs/asr-selective/result.json"))
        failure = owner.read(self.root / (self.job + "-failure.json"))
        self.assertEqual(failure["classification"], "model_phase_interrupted")
        self.assertIsNotNone(failure["model_phase"]["claim"])
        self.assertFalse(failure["budget"]["actual_total_is_complete"])
        cp.verify_ack(self.root / "checkpoints" / self.job / "owner-terminal")
        self.assertFalse((self.root / "checkpoints" / self.job / "nonmodel-terminal").exists())

    def test_zero_native_export_has_no_fabricated_view_or_asr_receipt(self):
        # A pure completion can retain a zero-observed chunk without claiming ASR.
        root = self.base / "zero"; root.mkdir()
        value = {**self.value, "root": str(root)}; q.write_json(root / "handoff.json", value)
        path = owner.where(value, "generate-006"); path["projection"].mkdir(parents=True)
        q.write_json(path["projection"] / "report.json", {"rows": 32, "observed_only": True})
        result_path = self.base / "generation-result.json"; q.write_json(result_path, {"status": "completed", "fixture": True})
        receipt = owner.seal_chunk(value, "generate-006", {"included": [], "excluded": ["fixture-cell"],
                                                        "job_result": owner.pin(result_path)})
        self.assertEqual(receipt["status"], "completed_no_asr_eligible_return")
        self.assertIsNone(receipt["asr_result"]); self.assertIsNone(receipt["statistics"])
        self.assertFalse(path["view"].exists()); owner.check_chunk(value, "generate-006")

    def test_unrelated_completion_ack_cannot_close_chunk_or_final_handoff(self):
        complete = self.path["complete"]
        q.write_json(complete, {"fixture": "completion"})
        folder = self.root / "checkpoints" / self.job / "chunk-complete"
        expected = {**owner.phase_scope(self.value, self.parent, self.job, "chunk-complete"),
                    "completion": owner.pin(complete)}
        self.gate.checkpoint(folder, {**expected, "job_id": "generate-006"}, [complete], intent=False)
        with mock.patch.object(owner, "check_chunk", return_value={}), self.assertRaisesRegex(ValueError, "ACK_SCOPE_CHANGED"):
            owner.check_chunk_closure(self.value, self.parent, self.job)
        q.write_json(self.root / "complete.json", {"fixture": "final"})
        final = self.root / "checkpoints/handoff-complete"
        self.gate.checkpoint(final, {**owner.final_scope(self.value), "phase": "unrelated"},
                             [self.root / "complete.json"], intent=False)
        with self.assertRaisesRegex(ValueError, "ACK_SCOPE_CHANGED"):
            owner.verify_scope(final, owner.final_scope(self.value))

    def test_cached_data_module_from_other_root_is_rejected(self):
        value = {"metadata_reader": {"root": str(self.base), "revision": "context-v3-metadata-r2", "source_files_sha256": {}}}
        with mock.patch.object(owner, "reader", return_value=value["metadata_reader"]), \
             mock.patch.dict(sys.modules, {"context_supplement_v3": SimpleNamespace(__file__="/different-root/context_supplement_v3.py")}), \
             self.assertRaisesRegex(ValueError, "MIXED_METADATA_MODULES"):
            owner.metadata_loader(value)

    def test_cli_propagates_incomplete_asr_exit_status(self):
        with mock.patch.object(owner, "asr_worker", return_value={"state": {"status": "incomplete"}}):
            self.assertEqual(owner.main(["asr-worker", "--handoff", str(self.handoff), "--job-id", self.job,
                                        "--request", str(self.path["request"])]), 1)


class HandoffLayoutTests(unittest.TestCase):
    def test_create_only_layout_supports_the_real_original_exporter(self):
        import research_context_v3_campaign as original
        generation = original.generation
        common = generation.common
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            prepared, stash, native = base / "prepared", base / "fixture-output", base / "generation"
            prepared.mkdir(); stash.mkdir(); native.mkdir()
            job, ident = "generate-005", "bulk-000000000000000000000005"
            stage_path, queue_root = prepared / "stage.json", prepared / "queue"
            plan = {"fixture": "no-model export parent regression", "cells": [{"cell_id": ident}]}
            q.write_json(prepared / "plan.json", plan)
            stage = {"prepared": str(prepared), "queue_root": str(queue_root), "generation_root": str(native),
                "chunks": [[], [], [], [], [ident]], "generation": common.GENERATION,
                "worker_timeout_seconds": generation.WORKER_SECONDS}
            q.write_json(stage_path, stage)
            recipe = {"schema": generation.JOB_SCHEMA, "stage": owner.pin(stage_path), "job_id": job,
                "chunk_index": 4, "selected_cells": [ident], "max_calls": 1,
                "generation": common.GENERATION, "worker_timeout_seconds": generation.WORKER_SECONDS}
            recipe_path = prepared / "recipe.json"; q.write_json(recipe_path, recipe)
            q.write_json(stash / "inventory.json", {"files": {}, "generation_files": {}, "generation_root": str(native)})
            q.write_json(stash / "summary.json", {"schema": "kws-context-v3-generation-result-v1", "status": "completed",
                "job_id": job, "recipe_sha256": q.file_record(recipe_path)["sha256"], "claims": 1,
                "generation_root": str(native), "rows": [{"cell_id": ident, "native_returned": False}],
                "explicit_no_model_fixture": True})
            command = "import os,shutil,sys;shutil.copytree(sys.argv[1],os.environ['KWS_RESEARCH_OUTPUT_DIR'],dirs_exist_ok=True)"
            declaration = prepared / "queue-request.json"
            q.write_json(declaration, {"schema": q.SCHEMA, "queue_id": "handoff-layout-no-model-fixture", "global_budget_minutes": 1,
                "jobs": [{"job_id": job, "recipe": str(recipe_path), "recipe_sha256": q.file_record(recipe_path)["sha256"],
                    "argv": [sys.executable, "-I", "-S", "-B", "-c", command, str(stash)], "cwd": str(prepared),
                    "depends_on": [], "timeout_minutes": .1, "outputs": ["summary.json", "inventory.json"]}]})
            q.create(queue_root, declaration, ROOT / "configs/research/active-plan-20261105.json", base / "heavy.lock")
            self.assertEqual(q.run(queue_root)[1], 0)
            before = q.inspect(queue_root)
            root = base / "handoff"; root.mkdir()
            context = {"root": str(root), "stage": owner.pin(stage_path)}
            # Only the upstream 72-cell/model source admission is replaced by
            # this explicit fixture. Queue receipts, ensure_export, export_job,
            # and the failing/successful filesystem operations are real.
            with mock.patch.object(generation, "load_stage", return_value=(stage, prepared, plan,
                    SimpleNamespace(p=SimpleNamespace(canonical=q.canonical)))):
                with self.assertRaises(FileNotFoundError):
                    original.ensure_export(context, job)
                owner.create_chunk_layout(root)
                self.assertEqual({p.name for p in (root / "chunks").iterdir()}, set(owner.NEW))
                self.assertTrue(all(not list(p.iterdir()) for p in (root / "chunks").iterdir()))
                # File-only durable capsules do not retain empty directories.
                # Reproduce that restore condition after fresh declaration.
                (root / "chunks" / job).rmdir()
                with self.assertRaises(FileNotFoundError):
                    original.ensure_export(context, job)
                self.assertEqual(owner.ensure_chunk_parent(root, job), root / "chunks" / job)
                exported = original.ensure_export(context, job)
                self.assertEqual(exported["included"], [])
                self.assertEqual(exported["excluded"], [{"cell_id": ident, "reason": "no_native_return"}])
                self.assertEqual(exported["new_asr_calls"], 0)
                self.assertEqual(original.ensure_export(context, job), exported)
            result = root / "chunks" / job / "export/export.json"
            saved = result.read_bytes()
            with self.assertRaisesRegex(ValueError, "CREATE_ONLY_CHUNK_LAYOUT"):
                owner.create_chunk_layout(root)
            owner.ensure_chunk_parent(root, job)
            self.assertEqual(result.read_bytes(), saved)
            self.assertEqual(q.inspect(queue_root), before)
            for blocked in ("generate-006", "generate-007"):
                parent = root / "chunks" / blocked
                parent.rmdir()
                if blocked == "generate-006": parent.write_bytes(b"existing-file")
                else: parent.symlink_to(root / "chunks" / job, target_is_directory=True)
                with self.assertRaises((ValueError, FileExistsError)):
                    owner.ensure_chunk_parent(root, blocked)
                self.assertTrue(parent.is_symlink() if blocked == "generate-007" else parent.read_bytes() == b"existing-file")
            self.assertEqual(result.read_bytes(), saved)


class RetainedOriginalIntentTests(unittest.TestCase):
    def test_actual_original_gen6_accepts_only_one_live_source_bound_fixture(self):
        if RETAINED_CAMPAIGN is None:
            self.skipTest("requires explicit --retained-campaign/--retained-validator; zero models")
        old, parent, stage, _ = owner.load_original(owner.pin(RETAINED_VALIDATOR), RETAINED_CAMPAIGN)
        provenance = owner.dependency_provenance(old, parent)
        self.assertEqual(len(provenance), 15)
        job = "generate-006"
        recipe_path = Path(stage["queue_root"]) / "recipes" / (job + ".json")
        recipe = owner.read(recipe_path)
        before = owner.q.inspect(stage["queue_root"], reconcile=False)
        self.assertEqual(before["jobs"][job]["status"], "pending")
        self.assertFalse((Path(stage["prepared"]) / "checkpoints" / job / "before-generation").exists())
        with tempfile.TemporaryDirectory() as temporary:
            # Only the fixture checkpoint destination differs. The authentic
            # original stage/recipe pins and original verifier are unchanged.
            fixture_stage = {**stage, "prepared": temporary}
            folder = Path(temporary) / "checkpoints" / job / "before-generation"
            scope = old.generation.checkpoint_scope(parent["stage"]["path"], fixture_stage, recipe_path, recipe, "before-generation")
            gate = cp.Gate(max_wait_seconds=0, on_wait=fixture_save)
            gate.checkpoint(folder, scope, [Path(owner.__file__), Path(RETAINED_VALIDATOR),
                Path(parent["stage"]["path"]), recipe_path], intent=True)
            env = {old.generation.INTENT_FOLDER_ENV: str(folder), old.generation.INTENT_OWNER_ENV: gate.owner}
            with mock.patch.dict(os.environ, env):
                old.generation.verify_generation_intent(parent["stage"]["path"], fixture_stage, recipe_path, recipe)
                with mock.patch.dict(os.environ, {old.generation.INTENT_OWNER_ENV: "different-live-owner"}), \
                     self.assertRaisesRegex(ValueError, "INTENT_SCOPE_CHANGED"):
                    old.generation.verify_generation_intent(parent["stage"]["path"], fixture_stage, recipe_path, recipe)
            other = cp.Gate(max_wait_seconds=0, on_wait=fixture_save)
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
                other.checkpoint(folder, scope, [Path(owner.__file__)], intent=True)
        self.assertEqual(before, owner.q.inspect(stage["queue_root"], reconcile=False))
        self.assertFalse((Path(stage["queue_root"]) / "jobs" / job / "claim.json").exists())
        self.assertFalse((Path(stage["prepared"]) / "checkpoints" / job / "before-generation").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--retained-campaign", type=Path)
    parser.add_argument("--retained-validator", type=Path)
    args, remaining = parser.parse_known_args()
    RETAINED_CAMPAIGN, RETAINED_VALIDATOR = args.retained_campaign, args.retained_validator
    if (RETAINED_CAMPAIGN is None) != (RETAINED_VALIDATOR is None):
        raise SystemExit("both retained paths are required")
    unittest.main(argv=[sys.argv[0], *remaining])
