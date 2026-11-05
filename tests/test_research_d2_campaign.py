#!/usr/bin/env python3
"""No-model campaign regressions with real subprocess queues and file receipts.

Only audio/model/weak-view adapters are substituted by explicit synthetic
fixtures. The campaign declaration, registration, queue jobs, process exits,
claims, result hashes, budgets and state transitions use production code.
"""
from __future__ import annotations

from argparse import Namespace
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_d2_campaign as campaign

queue = campaign.queue
ACTIVE_PLAN = ROOT / "configs/research/active-plan-20261105.json"
LOAD_DATA = campaign.load_data


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source, self.data_root, self.runtime = [self.base / n for n in ("source", "data", "asr-runtime")]
        self.prepared, self.output = self.base / "stage", self.base / "campaign"
        for path in (self.source, self.data_root, self.runtime, self.prepared):
            path.mkdir()
        self.events = self.base / "events.txt"
        self.job_ids = ["generate-001", "generate-002"]
        self.stage_path, self.generation_queue = self.prepared / "stage.json", self.prepared / "queue"
        self.stage = {"schema": campaign.generation.SCHEMA, "prepared": str(self.prepared),
                      "queue_root": str(self.generation_queue), "heavy_lock": str(self.runtime / "heavy-model.lock"),
                      "chunks": [["cell-001"], ["cell-002"]]}
        queue.write_json(self.stage_path, self.stage)
        self.plan = {"fixture": True, "cells": [{"cell_id": x[0]} for x in self.stage["chunks"]]}
        queue.write_json(self.prepared / "plan.json", self.plan)
        (self.runtime / "venv/bin").mkdir(parents=True)
        (self.runtime / "venv/bin/python").symlink_to(sys.executable)
        source_names = set(campaign.generation.OWN_FILES) | {"tools/research_d2_campaign.py",
                       "tools/research_queue_supervisor.py", "tools/active_asr_selective.py"}
        for name in source_names:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture source " + name)
        (self.data_root / "automation").mkdir()
        (self.data_root / "automation/active_d2_weak_view.py").write_text("fixture adapter")
        self.policy = self.base / "policy.json"
        queue.write_json(self.policy, campaign.selective.POLICY)
        self.asr_script = self.source / "tools/active_asr_selective.py"
        self.asr_script.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "r=json.loads(Path(sys.argv[-1]).read_text())\n"
            "p=Path(r['output']);(p/'audit').mkdir(parents=True)\n"
            "with Path(r['fixture_events']).open('a') as f:f.write('asr:'+r['fixture_job']+'\\n')\n"
            "(p/'audit/final-summary.json').write_text(json.dumps({'state':{'status':'complete'},'fixture':True}))\n"
            "(p/'audit/file-hashes.json').write_text(json.dumps({'fixture':True}))\n")
        self.data = SimpleNamespace(PRODUCER_FILES=("automation/active_d2_weak_view.py",),
                                    build_source_map=self.source_map, create_view=self.create_view,
                                    verify_view=self.verify_view)
        patches = (
            mock.patch.object(campaign, "ROOT", self.source),
            mock.patch.object(campaign, "load_data", return_value=self.data),
            mock.patch.object(campaign.generation, "load_stage", side_effect=self.load_stage),
            mock.patch.object(campaign.generation, "export_job", side_effect=self.export_job),
            mock.patch.object(campaign.selective, "source_identity", return_value={"tools/active_asr_selective.py": "fixture"}),
            mock.patch.object(campaign.batch, "prepare", side_effect=self.prepare_inputs),
            mock.patch.object(campaign.batch, "verify_inputs", side_effect=campaign.read),
            mock.patch.object(campaign.selective, "declare", side_effect=self.declare_asr),
            mock.patch.object(campaign.selective, "verify_request", side_effect=self.verify_asr_request),
            mock.patch.object(campaign.selective, "seal_or_verify", side_effect=self.audit_asr),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.args = Namespace(stage=self.stage_path, output=self.output, data_root=self.data_root,
                              asr_runtime=self.runtime, policy=self.policy, fsmn_tokens=None,
                              max_chunks=2, total_minutes=1, asr_job_minutes=.05,
                              asr_budget_seconds=1, threads=4)

    def load_stage(self, path):
        self.assertEqual(Path(path), self.stage_path)
        return campaign.read(path), self.prepared, self.plan, None

    def make_generation_queue(self, *, fail_first=False):
        jobs = []
        for index, job_id in enumerate(self.job_ids):
            recipe = self.prepared / (job_id + ".json")
            queue.write_json(recipe, {"fixture": True, "job_id": job_id})
            command = (
                "import json,os,sys;from pathlib import Path;"
                "p=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR']);"
                "p.joinpath('summary.json').write_text(json.dumps({'fixture':True}));"
                "p.joinpath('inventory.json').write_text(json.dumps({'fixture':True}));"
                "f=Path(sys.argv[1]).open('a');f.write('gen:'+sys.argv[2]+'\\n');f.close();"
                "sys.exit(int(sys.argv[3]))")
            jobs.append({"job_id": job_id, "recipe": str(recipe), "recipe_sha256": queue.file_record(recipe)["sha256"],
                         "argv": [sys.executable, "-I", "-S", "-B", "-c", command, str(self.events), job_id,
                                  "7" if fail_first and index == 0 else "0"],
                         "cwd": str(self.prepared), "depends_on": [] if index == 0 else [self.job_ids[index - 1]],
                         "timeout_minutes": .05, "outputs": ["summary.json", "inventory.json"]})
        request = {"schema": queue.SCHEMA, "queue_id": "fixture-generation", "global_budget_minutes": 1, "jobs": jobs}
        path = self.prepared / "request.json"
        queue.write_json(path, request)
        queue.create(self.generation_queue, path, ACTIVE_PLAN, self.runtime / "heavy-model.lock")

    def declare(self, **changes):
        for key, value in changes.items():
            setattr(self.args, key, value)
        self.value = campaign.declare(self.args)
        self.path = self.output / "campaign.json"
        return self.value

    def event_rows(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def export_job(self, stage, job_id, output):
        qroot, _qmeta, request, qsha = queue.load_queue(self.generation_queue)
        definition = next(j for j in request["jobs"] if j["job_id"] == job_id)
        result = queue.result_for(qroot, definition, qsha)
        queue.require(result["status"] == "completed", "fixture export needs a real completed receipt")
        output = Path(output)
        output.mkdir()
        archive = output / "source-original.zip"
        archive.write_bytes(("explicit synthetic archive fixture " + job_id).encode())
        value = {"schema": "kws-d2-asr-export-v1", "stage": campaign.pin(stage), "job_id": job_id,
                 "job_result": campaign.pin(qroot / "jobs" / job_id / "result.json"),
                 "job_recipe": campaign.pin(qroot / "recipes" / (job_id + ".json")),
                 "plan": str(self.prepared / "plan.json"), "plan_sha256": "1" * 64,
                 "archive": queue.file_record(archive), "generation_root": str(self.prepared / "generation"),
                 "native_root": str(qroot / "jobs" / job_id / "outputs/native"),
                 "included": [{"cell_id": job_id, "observed_text": None, "training_admitted": False}], "excluded": []}
        queue.write_json(output / "export.json", value)
        return value

    def prepare_inputs(self, args):
        args.output.mkdir()
        queue.write_json(args.output / "inputs.json", {"fixture": True, "source_archive_sha256": args.source_sha256,
                                                      "clips": [{"id": "clip-000"}]})

    def source_map(self, inputs, sha, archive, **options):
        return {"fixture": True, "inputs_sha256": sha, "archive": queue.file_record(archive),
                "job_id": Path(options["job_recipe_path"]).stem}

    def declare_asr(self, args):
        value = {"fixture": True, "runtime": str(args.runtime), "output": str(args.output),
                 "inputs": {"path": str(args.inputs)}, "source_map": {"path": str(args.source_map)},
                 "policy": {"path": str(args.policy)}, "threads": args.threads, "budget_seconds": args.budget_seconds,
                 "fixture_events": str(self.events), "fixture_job": args.request.parent.name}
        queue.write_json(args.request, value)
        return value

    def verify_asr_request(self, path):
        return campaign.read(path), None, None, None

    def audit_asr(self, path):
        request = campaign.read(path)
        return campaign.read(Path(request["output"]) / "audit/final-summary.json")

    def create_view(self, inputs, expected_inputs_sha, archive, asr_root, output, **options):
        job_id = Path(options["job_recipe_path"]).stem
        report = {"fixture": True, "inputs_sha256": expected_inputs_sha,
                  "source_archive_sha256": queue.file_record(archive)["sha256"],
                  "asr_adapter": {"final_audit_sha256": options["selective_audit_sha256"]},
                  "source_adapter": {"plan_sha256": options["plan_sha256"], "job_owner": {
                      "job_recipe_sha256": options["job_recipe_sha256"],
                      "stage_sha256": queue.file_record(options["stage_path"])["sha256"], "job_id": job_id}},
                  "statistics": {"source_items": 1, "machine_candidates": 0, "uncertain": 1}}
        output.mkdir()
        queue.write_json(output / "report.json", report)
        queue.write_json(output / "MANIFEST.json", {"report.json": queue.file_record(output / "report.json")})
        return report

    def verify_view(self, output):
        expected = campaign.read(output / "MANIFEST.json")
        queue.require(expected == {"report.json": queue.file_record(output / "report.json")}, "fixture weak bytes changed")
        return campaign.read(output / "report.json")

    def test_completion_order_and_combined_actual_budget_use_real_queue_receipts(self):
        self.make_generation_queue()
        value = self.declare()
        result, code = campaign.run(self.path)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.event_rows(), ["gen:generate-001", "asr:generate-001", "gen:generate-002", "asr:generate-002"])
        report = campaign.budget(value)
        actual = sum(s["actual_elapsed_ns"] for s in report["generation"]["jobs"].values())
        actual += sum(s["actual_elapsed_ns"] for q in report["asr"].values() for s in q["jobs"].values())
        self.assertEqual(report["actual_known_job_nanoseconds"], actual)
        self.assertGreater(actual, 0)
        self.assertEqual(campaign.read(self.output / "heartbeat.json")["state"], "completed")
        previous = self.event_rows()
        again, code = campaign.run(self.path)
        self.assertEqual(code, 0)
        self.assertEqual(again["new_completed_chunks"], 0)
        self.assertEqual(self.event_rows(), previous)

    def test_one_chunk_cap_resumes_same_declaration_without_new_claims_for_first(self):
        self.make_generation_queue()
        self.declare()
        first, code = campaign.run(self.path, max_chunks_this_call=1)
        self.assertEqual((first["status"], code), ("paused", 0))
        self.assertEqual(self.event_rows(), ["gen:generate-001", "asr:generate-001"])
        second, code = campaign.run(self.path, max_chunks_this_call=1)
        self.assertEqual((second["status"], code), ("completed", 0))
        self.assertEqual(second["new_completed_chunks"], 1)
        self.assertEqual(len(self.event_rows()), 4)

    def test_adopts_authentic_completed_generation_asr_and_view_without_model_repeats(self):
        self.make_generation_queue()
        queue.run(self.generation_queue, max_jobs=1)
        value = self.declare(max_chunks=1)
        where = campaign.paths(self.output, self.job_ids[0])
        where["chunk"].mkdir(parents=True)
        exported = campaign.ensure_export(value, self.job_ids[0])
        inputs = campaign.ensure_inputs(value, self.job_ids[0], exported)
        mapping = campaign.ensure_source_map(value, self.job_ids[0], exported, inputs, self.data)
        campaign.ensure_asr_request(value, self.job_ids[0], inputs, mapping)
        campaign.ensure_asr_queue(value, self.job_ids[0])
        queue.run(where["asr_queue"], max_jobs=1)
        campaign.ensure_view(value, self.job_ids[0], exported, inputs, self.data)
        previous = self.event_rows()
        result, code = campaign.run(self.path)
        self.assertEqual((result["status"], code), ("completed", 0))
        self.assertEqual(self.event_rows(), previous)

    def test_same_stage_second_driver_and_changed_declaration_are_rejected(self):
        self.make_generation_queue()
        self.declare()
        other = copy.copy(self.args)
        other.output = self.base / "other-campaign"
        with self.assertRaisesRegex(ValueError, "ALREADY_REGISTERED"):
            campaign.declare(other)
        value = campaign.read(self.path)
        value["total_budget_minutes"] = 500
        self.path.write_bytes(queue.canonical(value))
        with self.assertRaisesRegex(ValueError, "REGISTRATION_CHANGED"):
            campaign.run(self.path)

    def test_frozen_source_change_blocks_before_any_claim(self):
        self.make_generation_queue()
        self.declare()
        self.asr_script.write_text(self.asr_script.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "SOURCE_CHANGED"):
            campaign.run(self.path)
        self.assertEqual(self.event_rows(), [])

    def test_cached_shared_data_module_must_match_the_declared_producer_bytes(self):
        relatives = {
            "active_d2_weak_view": "automation/active_d2_weak_view.py",
            "active_asr_weak_view": "automation/active_asr_weak_view.py",
            "d2_pool": "research/auto-tts-voice-clone-20261102/d2_pool.py",
            "bulk_pool": "research/auto-tts-voice-clone-20261102/bulk_pool.py",
            "kws_data_contract": "kws_data_contract/__init__.py",
            "kws_data_contract.identity": "kws_data_contract/identity.py",
            "kws_data_contract.source_registry": "kws_data_contract/source_registry.py",
            "kws_data_contract.reference_text": "kws_data_contract/reference_text.py",
        }
        modules = {}
        for name, relative in relatives.items():
            path = self.data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("fixture module " + name)
            modules[name] = SimpleNamespace(__file__=str(path))
        with mock.patch.dict(sys.modules, modules), mock.patch.object(campaign.importlib, "import_module",
                return_value=modules["active_d2_weak_view"]):
            self.assertIs(LOAD_DATA(self.data_root), modules["active_d2_weak_view"])
            stale = self.base / "stale-pool.py"
            stale.write_text("different frozen producer bytes")
            modules["d2_pool"].__file__ = str(stale)
            with self.assertRaisesRegex(ValueError, "CACHED_DATA_SOURCE_CHANGED"):
                LOAD_DATA(self.data_root)

    def test_insufficient_combined_budget_reserves_downstream_asr_before_tts(self):
        self.make_generation_queue()
        self.declare(total_minutes=.07)
        result, code = campaign.run(self.path)
        self.assertEqual((result["status"], code), ("budget_exhausted", 3))
        self.assertEqual(self.event_rows(), [])
        self.assertTrue(all(s["status"] == "pending" for s in queue.inspect(self.generation_queue)["jobs"].values()))

    def test_real_failed_generation_is_terminal_and_never_retried(self):
        self.make_generation_queue(fail_first=True)
        self.declare()
        first, code = campaign.run(self.path)
        self.assertEqual(code, 1)
        again, code = campaign.run(self.path)
        self.assertEqual((again["status"], code), ("failed_or_unknown", 1))
        self.assertEqual(self.event_rows(), ["gen:generate-001"])

    def test_unknown_generation_receipt_is_not_treated_as_unclaimed(self):
        self.make_generation_queue()
        self.declare()
        queue.run(self.generation_queue, max_jobs=1)
        result_file = self.generation_queue / "jobs/generate-001/result.json"
        result_file.unlink()
        result, code = campaign.run(self.path)
        self.assertEqual((result["status"], code), ("failed_or_unknown", 1))
        self.assertEqual(self.event_rows(), ["gen:generate-001"])
        self.assertFalse(result_file.exists())

    def test_unknown_asr_receipt_blocks_next_chunk_without_repeating_either_model_phase(self):
        self.make_generation_queue()
        self.declare()
        campaign.run(self.path, max_chunks_this_call=1)
        where = campaign.paths(self.output, self.job_ids[0])
        result_file = where["asr_queue"] / "jobs/asr-selective/result.json"
        result_file.unlink()
        result, code = campaign.run(self.path)
        self.assertEqual((result["status"], code), ("failed_or_unknown", 1))
        self.assertEqual(self.event_rows(), ["gen:generate-001", "asr:generate-001"])
        self.assertFalse(result_file.exists())

    def test_no_eligible_return_finishes_without_asr_or_synthetic_training_rows(self):
        self.make_generation_queue()
        self.declare(max_chunks=1)
        original = self.export_job
        def empty_export(*args):
            value = original(*args)
            value["included"] = []
            value["excluded"] = [{"cell_id": self.job_ids[0], "reason": "no_native_return"}]
            (Path(args[2]) / "export.json").write_bytes(queue.canonical(value))
            return value
        with mock.patch.object(campaign.generation, "export_job", side_effect=empty_export):
            result, code = campaign.run(self.path)
        self.assertEqual((result["status"], code), ("completed", 0))
        where = campaign.paths(self.output, self.job_ids[0])
        receipt = campaign.read(where["complete"])
        self.assertEqual(receipt["status"], "completed_no_asr_eligible_return")
        self.assertEqual(receipt["native_included"], 0)
        self.assertIsNone(receipt["statistics"])
        self.assertFalse(where["view"].exists())
        self.assertFalse(where["asr_queue"].exists())
        self.assertEqual(self.event_rows(), ["gen:generate-001"])

    def test_partial_non_model_output_is_not_overwritten_or_regenerated(self):
        self.make_generation_queue()
        self.declare()
        queue.run(self.generation_queue, max_jobs=1)
        target = campaign.paths(self.output, self.job_ids[0])["export"]
        target.mkdir(parents=True)
        (target / "partial.bin").write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "PARTIAL_EXPORT_NOT_RETRIED"):
            campaign.run(self.path)
        self.assertEqual((target / "partial.bin").read_bytes(), b"original")
        self.assertEqual(self.event_rows(), ["gen:generate-001"])

    def test_reuse_rejects_changed_export_bytes_before_next_chunk(self):
        self.make_generation_queue()
        self.declare()
        campaign.run(self.path, max_chunks_this_call=1)
        path = campaign.paths(self.output, self.job_ids[0])["export"] / "source-original.zip"
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "EXISTING_EXPORT_CHANGED"):
            campaign.run(self.path)
        self.assertEqual(self.event_rows(), ["gen:generate-001", "asr:generate-001"])

    def test_status_rechecks_sealed_artifact_pins_without_running_a_model(self):
        self.make_generation_queue()
        self.declare()
        campaign.run(self.path, max_chunks_this_call=1)
        self.assertEqual(campaign.status(self.path)["completed_chunks"], self.job_ids[:1])
        target = campaign.paths(self.output, self.job_ids[0])["view"] / "MANIFEST.json"
        target.write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "RECEIPT_FILE_CHANGED"):
            campaign.status(self.path)
        self.assertEqual(len(self.event_rows()), 2)

    def test_wait_resume_preserves_invocation_chunk_cap(self):
        self.make_generation_queue()
        self.declare()
        observed = []
        def synthetic_advance(path, *, max_chunks_this_call, progress):
            observed.append(max_chunks_this_call)
            if len(observed) == 1:
                target = campaign.paths(self.output, self.job_ids[0])["complete"]
                target.parent.mkdir(parents=True)
                queue.write_json(target, {"fixture": True})
                return {"status": "waiting_for_worker"}, 75
            return {"status": "completed"}, 0
        with mock.patch.object(campaign, "advance", side_effect=synthetic_advance):
            campaign.run(self.path, max_chunks_this_call=2, wait_seconds=.02)
        self.assertEqual(observed, [2, 1])

    def test_driver_lock_rejects_a_concurrent_host(self):
        self.make_generation_queue()
        self.declare()
        with queue.exclusive(campaign.driver_lock(self.stage_path)):
            with self.assertRaises(BlockingIOError):
                campaign.run(self.path)
        self.assertEqual(self.event_rows(), [])


if __name__ == "__main__":
    unittest.main()
