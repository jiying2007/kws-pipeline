#!/usr/bin/env python3
"""Real queue/Gate receipts and explicit no-model phase fixtures for v3."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import importlib
import os
from pathlib import Path
import shutil
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import research_context_v3_campaign as campaign
import test_research_context_v3_generate as owner_fixture

q, cp = campaign.queue, campaign.checkpoints


class ContextV3CampaignTests(unittest.TestCase):
    def setUp(self):
        self.fixture = owner_fixture.ContextV3GenerationTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.stage_path, self.stage, self.prepared, self.marker = self.fixture.guarded_fixture(job_count=2)
        self.root = self.base / "campaign"; self.root.mkdir()
        self.qroot, _, self.request, queue_sha = q.load_queue(self.prepared / "queue")
        self.value = {"schema": campaign.SCHEMA, "root": str(self.root),
            "stage": campaign.pin(self.stage_path),
            "generation_queue": {"path": str(self.qroot), "queue_sha256": queue_sha},
            "job_ids": ["generate-001", "generate-002"], "total_budget_minutes": 180,
            "asr_job_minutes": 10, "asr_runtime": str(self.base / "asr"), "asr_python": sys.executable}
        (self.base / "asr").mkdir()
        self.path = self.root / "campaign.json"; q.write_json(self.path, self.value)
        # These are deliberately narrow synthetic phase inputs. Production
        # checkpoint_sources has separate current-chunk/chain tests below.
        patch = mock.patch.object(campaign, "checkpoint_sources", side_effect=lambda *_:
            [self.path, self.stage_path, *cp.regular_tree(self.qroot)])
        patch.start(); self.addCleanup(patch.stop)
        self.gate = cp.Gate(max_wait_seconds=0, on_wait=self.fixture.fixture_save)

    def load_fixture(self):
        return mock.patch.object(campaign, "load", return_value=(self.value, self.stage, self.request))

    def orphan(self, root, identifier):
        root, _, request, queue_sha = q.load_queue(root)
        job = next(item for item in request["jobs"] if item["job_id"] == identifier)
        q.write_json(root / "jobs" / identifier / "claim.json", {
            "schema": "kws-local-research-job-claim-v1", **q.job_binding(job, queue_sha),
            "attempt": 1, "automatic_retry": False, "explicit_orphan_fixture": True})

    def complete_generation(self):
        status, code = campaign.run_generation(self.value, self.stage, "generate-001", self.gate, None)
        self.assertEqual(code, 0)
        self.assertEqual(status["jobs"]["generate-001"]["status"], "completed")

    def test_real_generation_queue_runs_only_after_saved_intent_with_exact_live_owner(self):
        self.complete_generation()
        proof = json.loads(self.marker.read_bytes())
        folder = self.prepared / "checkpoints/generate-001/before-generation"
        self.assertEqual(proof["owner"], self.gate.owner)
        self.assertEqual(proof["intent"], str(folder))
        cp.verify_ack(folder)
        self.assertNotIn(campaign.generation.INTENT_OWNER_ENV, os.environ)
        self.assertFalse((self.qroot / "jobs/generate-002/claim.json").exists())

    def test_unacknowledged_intent_and_new_invocation_cannot_launch_a_pending_job(self):
        first = cp.Gate(max_wait_seconds=0)
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            with self.assertRaisesRegex(ValueError, "SAVE_NOT_ACKNOWLEDGED"):
                campaign.run_generation(self.value, self.stage, "generate-001", first, None)
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
                campaign.run_generation(self.value, self.stage, "generate-001", self.gate, None)
            execute.assert_not_called()
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.qroot / "jobs/generate-001/claim.json").exists())

    def test_saved_but_unlaunched_intent_is_not_reusable_by_another_invocation(self):
        campaign.phase_checkpoint(self.gate, self.value, self.stage, "generate-001", "before-generation")
        other = cp.Gate(max_wait_seconds=0, on_wait=self.fixture.fixture_save)
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
                campaign.run_generation(self.value, self.stage, "generate-001", other, None)
            execute.assert_not_called()
        self.assertFalse(self.marker.exists())

    def test_insufficient_wall_budget_after_intent_ack_does_not_launch(self):
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            with self.assertRaisesRegex(ValueError, "WALL_BUDGET_EXHAUSTED"):
                campaign.run_generation(self.value, self.stage, "generate-001", self.gate, None,
                                        deadline=time.monotonic() + 300)
            execute.assert_not_called()
        cp.verify_ack(self.prepared / "checkpoints/generate-001/before-generation")
        self.assertFalse((self.qroot / "jobs/generate-001/claim.json").exists())

    def test_dead_generation_claim_gets_authenticated_terminal_capsule_before_halt(self):
        self.orphan(self.qroot, "generate-001")
        with self.load_fixture(), mock.patch.object(q, "run", wraps=q.run) as execute:
            result, code = campaign.advance(self.path, checkpoint_gate=self.gate)
            execute.assert_not_called()
        self.assertEqual((result["status"], code), ("failed_or_unknown", 1))
        folder = self.prepared / "checkpoints/generate-001/generation-terminal"
        descriptor = cp.verify_capsule(folder); cp.verify_ack(folder)
        self.assertEqual(descriptor["context"]["queue_result_status"], "unknown")
        self.assertEqual(descriptor["context"]["queue_result"], campaign.pin(self.qroot / "jobs/generate-001/result.json"))
        receipt = campaign.read(self.qroot / "jobs/generate-001/result.json")
        self.assertIsNone(receipt["actual_elapsed_ns"])
        self.assertFalse(receipt["actual_duration_known"])
        self.assertFalse((self.qroot / "jobs/generate-002/claim.json").exists())

    def test_dead_asr_claim_is_closed_before_unresolved_wait_without_relaunch(self):
        self.complete_generation()
        where = campaign.paths(self.root, "generate-001"); where["chunk"].mkdir(parents=True)
        q.write_json(where["asr_request"], {"explicit_fixture": True})
        definition = campaign.asr_queue_definition(self.value, "generate-001")
        q.write_json(where["asr_queue_request"], definition)
        q.create(where["asr_queue"], where["asr_queue_request"], owner_fixture.ACTIVE_PLAN,
                 Path(self.value["asr_runtime"]) / "heavy-model.lock")
        self.orphan(where["asr_queue"], campaign.ASR_JOB)
        with self.load_fixture(), mock.patch.object(q, "run", wraps=q.run) as execute:
            result, code = campaign.advance(self.path, checkpoint_gate=self.gate)
            execute.assert_not_called()
        self.assertEqual((result["status"], code), ("failed_or_unknown", 1))
        folder = self.root / "checkpoints/generate-001/asr-terminal"
        descriptor = cp.verify_capsule(folder); cp.verify_ack(folder)
        self.assertEqual(descriptor["context"]["terminal_state"], "unknown")
        self.assertEqual(descriptor["context"]["asr_result"],
                         campaign.pin(where["asr_queue"] / "jobs/asr-selective/result.json"))
        self.assertFalse((self.qroot / "jobs/generate-002/claim.json").exists())

    def phase_fixture(self, *, included=True, weak_failure=False, deadline=None):
        self.complete_generation()
        events = []
        exported = {"included": [{"cell_id": "fixture"}] if included else [], "excluded": []}
        def phase(_gate, _value, _stage, _job, name, **_):
            events.append(name)
        def asr_run(*_args, **_kwargs):
            events.append("asr-model-fixture")
            return {"status": "completed", "jobs": {campaign.ASR_JOB: {"status": "completed"}}}, 0
        def weak(*_):
            events.append("weak-view")
            if weak_failure: raise ValueError("explicit-weak-replay-failure")
            return {"statistics": {"explicit_fixture": True}}
        def project(*_): events.append("projection")
        def seal(*_):
            events.append("seal")
            return {"native_included": int(included), "statistics": {},
                    "cumulative_full2601": {"observed_rows": 1, "eligible_rows": 1}}
        patches = [self.load_fixture(), mock.patch.object(campaign, "phase_checkpoint", side_effect=phase),
            mock.patch.object(campaign, "ensure_export", return_value=exported),
            mock.patch.object(campaign, "ensure_inputs", return_value=self.base / "fixture-inputs.json"),
            mock.patch.object(campaign, "ensure_source_map", return_value=self.base / "fixture-map.json"),
            mock.patch.object(campaign, "ensure_asr_request", return_value={}),
            mock.patch.object(campaign, "ensure_asr_queue", return_value={"jobs": {campaign.ASR_JOB: {"status": "pending"}}}),
            mock.patch.object(q, "run", side_effect=asr_run),
            mock.patch.object(campaign, "ensure_view", side_effect=weak),
            mock.patch.object(campaign, "nonmodel_worker", side_effect=project),
            mock.patch.object(campaign, "seal_chunk", side_effect=seal)]
        for patch in patches: patch.start(); self.addCleanup(patch.stop)
        return events, lambda: campaign.advance(self.path, max_chunks_this_call=1,
                                                checkpoint_gate=self.gate, deadline=deadline)

    def test_chunk_sequence_persists_asr_before_weak_and_fulltext_before_next_generation(self):
        events, execute = self.phase_fixture()
        with redirect_stdout(io.StringIO()): result, code = execute()
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(events, ["generation-complete", "before-asr", "asr-model-fixture", "asr-complete",
                                  "weak-view", "projection", "seal", "chunk-complete"])
        self.assertFalse((self.qroot / "jobs/generate-002/claim.json").exists())

    def test_pure_weak_failure_occurs_after_durable_asr_and_before_any_next_chunk(self):
        events, execute = self.phase_fixture(weak_failure=True)
        with self.assertRaisesRegex(ValueError, "explicit-weak-replay-failure"): execute()
        self.assertEqual(events[-2:], ["asr-complete", "weak-view"])
        self.assertNotIn("projection", events)
        self.assertNotIn("chunk-complete", events)
        self.assertFalse((self.qroot / "jobs/generate-002/claim.json").exists())

    def test_zero_native_export_closes_without_asr_or_fabricated_weak_rows(self):
        events, execute = self.phase_fixture(included=False)
        with redirect_stdout(io.StringIO()): result, code = execute()
        self.assertEqual(code, 0)
        self.assertEqual(events, ["generation-complete", "seal", "chunk-complete"])

    def test_asr_wall_reservation_is_checked_after_ack_before_model_queue(self):
        events, execute = self.phase_fixture(deadline=time.monotonic() + 300)
        with self.assertRaisesRegex(ValueError, "WALL_BUDGET_EXHAUSTED"): execute()
        self.assertEqual(events, ["generation-complete", "before-asr"])
        self.assertNotIn("asr-model-fixture", events)

    def test_terminal_pin_authenticates_queue_claim_identity(self):
        self.orphan(self.qroot, "generate-001")
        status, pinned = campaign.terminal_receipt(self.qroot, "generate-001")
        self.assertEqual(status, "unknown")
        value = campaign.read(pinned["path"]); value["claim_sha256"] = "0" * 64
        Path(pinned["path"]).write_bytes(q.canonical(value))
        with self.assertRaisesRegex(ValueError, "RESULT_IDENTITY_CHANGED"):
            campaign.terminal_receipt(self.qroot, "generate-001")


@unittest.skipUnless(os.environ.get("KWS_CONTEXT_DATA_ROOT"), "set KWS_CONTEXT_DATA_ROOT for real-format child replay")
class ContextV3WorkerIntegrationTests(unittest.TestCase):
    def test_fresh_workers_replay_native_asr_and_full2601_with_ordinary_text_targets(self):
        data_root = Path(os.environ["KWS_CONTEXT_DATA_ROOT"]).absolute()
        sys.path.insert(0, str(data_root / "automation"))
        fixtures = importlib.import_module("test_active_context_v3_weak_view")
        fixture = fixtures.ContextV3WeakViewTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root = fixture.root / "campaign-worker"; root.mkdir()
        where = campaign.paths(root, "generate-001"); where["chunk"].mkdir(parents=True)
        where["export"].mkdir()
        shutil.copyfile(fixture.archive, where["export"] / "source-original.zip")
        shutil.copytree(fixture.inputs, where["inputs"])
        shutil.copytree(fixture.asr, where["asr_output"])
        tokens = ROOT / "tests/fixtures/d1-fulltext/fsmn-official-tokens.txt"
        vocabulary = fixture.root / "worker-vocabulary.json"
        vocabulary.write_bytes(fixture.vocabulary_raw)
        setup = fixture.root / "fixture-asr-setup.json"
        q.write_json(setup, {"status": "complete", "explicit_fixture": True})
        value = {"schema": campaign.SCHEMA, "root": str(root), "stage": campaign.pin(fixture.stage_path),
            "max_chunks": 9, "total_budget_minutes": 180, "asr_job_minutes": 10, "asr_budget_seconds": 570,
            "threads": 4, "model_concurrency": 1, "automatic_retries": 0,
            "max_tts_calls": 72, "max_asr_calls": 216, "driver_wall_seconds": 10800, "human_gold": False,
            "job_ids": [f"generate-{index:03d}" for index in range(1, 10)],
            "durable_checkpoints": {"schema": cp.SCHEMA, "required": True,
                "intent_owner": "one-live-driver-invocation", "before_every_model_phase": True,
                "closure_before_next_chunk": True, "pending_intent_restart_allowed": False},
            "data_root": str(data_root), "data_files": campaign.data_records(data_root),
            "pipeline_files": campaign.pipeline_records(), "policy": campaign.pin(fixture.policy_path),
            "fsmn_tokens": campaign.pin(tokens), "vocabulary": campaign.pin(vocabulary),
            "asr_setup": campaign.pin(setup), "plan_sha256": fixture.plan_sha}
        q.write_json(root / "campaign.json", value)
        exported = {"plan": str(fixture.plan_path), "plan_sha256": fixture.plan_sha,
            "generation_root": str(fixture.generation), "native_root": str(fixture.native),
            "job_recipe": campaign.pin(fixture.recipe_path),
            "archive": q.file_record(where["export"] / "source-original.zip")}
        q.write_json(where["export"] / "export.json", exported)
        for task in ("source-map", "weak-view", "projection"):
            result = campaign.nonmodel_worker(value, "generate-001", task)
            self.assertEqual(result["new_model_calls"], 0)
        report = campaign.read(where["projection"] / "report.json")
        self.assertEqual(report["observed_rows"], 8)
        self.assertEqual(report["planned_rows"], 72)
        self.assertEqual(report["eligible_rows_by_category"]["ordinary-negative"], 2)
        rows = [q.loads(line) for line in (where["projection"] / "rows.jsonl").read_bytes().splitlines()]
        ordinary = [row for row in rows if row["generation_category"] == "ordinary-negative"]
        for row in ordinary:
            self.assertEqual(row["keyword_targets"], [0, 0])
            self.assertEqual(len(row["target_ids"]), len(row["full_text"]))
            self.assertTrue(row["target_ids"])
        failed = next(row for row in rows if row["generation_outcome"] == "failed-native")
        self.assertIsNone(failed["target_ids"])
        self.assertEqual(failed["keyword_weight"], [0, 0])
        before = q.file_record(where["projection"] / "MANIFEST.json")
        campaign.nonmodel_worker(value, "generate-001", "projection")
        self.assertEqual(before, q.file_record(where["projection"] / "MANIFEST.json"))


if __name__ == "__main__":
    unittest.main()
