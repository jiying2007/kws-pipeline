#!/usr/bin/env python3
"""Context v3 orchestration boundaries using real files and stdlib queue fixtures.

No model is imported or run. Planner/model-asset validation is stubbed only in
the preparation fixture; kws-data owns that separate contract. Export fixtures
use real queue claims, terminal receipts and output hashes from dummy commands.
For cross-worktree development set KWS_CONTEXT_V3_GENERATE_SOURCE_ROOT; CI defaults to
this repository's ordinary tools directory.
"""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("KWS_CONTEXT_V3_GENERATE_SOURCE_ROOT", str(ROOT))).resolve()
spec = importlib.util.spec_from_file_location("tested_research_context_v3_generate", SOURCE_ROOT / "tools/research_context_v3_generate.py")
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)
queue = generation.queue
common = generation.common
ACTIVE_PLAN = SOURCE_ROOT / "configs/research/active-plan-20261105.json"


class PlannerFixture:
    """Isolate orchestration tests from the other repository's model assets."""
    LOCK_SCHEMA = "fixture-model-lock-v1"
    p = SimpleNamespace(canonical=common.canonical)

    @staticmethod
    def verify_plan(plan):
        return plan

    @staticmethod
    def rebuild_plan(plan, manifest, audio_root):
        return copy.deepcopy(plan)

    @staticmethod
    def asset_lock(root, value, revision):
        return None


class ContextV3GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.plan = {
            "schema": "kws-context-supplement-plan-v3", "stage": "context-supplement-v3", "batch_id": "fixture-d2",
            "model_revision": "0" * 40, "references": [], "generation": copy.deepcopy(common.GENERATION),
            "d2_parent": {"plan_sha256": "1" * 64, "plan": {"fixture": True}},
            "context_parent": {"plan_sha256": "2" * 64, "plan": {"fixture": "v1"}},
            "cells": [{"cell_id": f"bulk-{index:024d}"} for index in range(72)],
            "budgets": {"max_total_calls": 72, "max_calls_per_invocation": 8, "max_retries": 0},
        }
        self.ids = [row["cell_id"] for row in self.plan["cells"]]
        self.generation_root = self.base / "generation"

    def saved_claim(self, ident, terminal="completed"):
        folder = self.generation_root / "cells" / ident
        folder.mkdir(parents=True, exist_ok=True)
        common.once(folder / "started.json", {"fixture": True, "cell_id": ident})
        if terminal is not None:
            common.once(folder / (terminal + ".json"), {"fixture": True, "cell_id": ident})
        return folder

    def test_partition_first_eight_and_next_chunk_are_exact_and_unique(self):
        before = copy.deepcopy(self.plan)
        groups = generation.partition(self.plan, 8)
        self.assertEqual(groups, [self.ids[i:i+8] for i in range(0,72,8)])
        self.assertEqual(self.plan, before)
        for invalid in (0, 9, 17, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generation.partition(self.plan, invalid)
        duplicate = copy.deepcopy(self.plan)
        duplicate["cells"][1] = copy.deepcopy(duplicate["cells"][0])
        with self.assertRaisesRegex(ValueError, "CELL_ROSTER"):
            generation.partition(duplicate, 8)

    def test_prefix_accepts_only_unstarted_first_eight_or_exact_finished_prefix(self):
        generation.check_prefix(self.plan, self.generation_root, self.ids[:8])
        self.assertFalse(self.generation_root.exists())
        for index, ident in enumerate(self.ids[:8]):
            self.saved_claim(ident, "failed" if index == 2 else "completed")
        before = {p: p.read_bytes() for p in self.generation_root.rglob("*") if p.is_file()}
        generation.check_prefix(self.plan, self.generation_root, self.ids[8:16])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_replaying_claimed_first_chunk_is_rejected_even_with_success_receipts(self):
        for ident in self.ids[:8]:
            self.saved_claim(ident)
        with self.assertRaisesRegex(ValueError, "ALREADY_CLAIMED"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[:8])

    def test_partial_and_out_of_order_claims_cannot_be_silently_skipped(self):
        self.saved_claim(self.ids[0])
        with self.assertRaisesRegex(ValueError, "ALREADY_CLAIMED"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[:8])
        with self.assertRaisesRegex(ValueError, "ALREADY_CLAIMED"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[8:16])
        for ident in self.ids[1:8]:
            self.saved_claim(ident)
        self.saved_claim(self.ids[12])
        with self.assertRaisesRegex(ValueError, "ALREADY_CLAIMED"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[8:16])

    def test_unknown_or_double_terminal_prefix_blocks_next_generation(self):
        for ident in self.ids[:8]:
            self.saved_claim(ident)
        target = self.generation_root / "cells" / self.ids[3]
        (target / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "UNIQUE_TERMINAL"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[8:16])
        common.once(target / "completed.json", {"fixture": True})
        common.once(target / "failed.json", {"fixture": True})
        with self.assertRaisesRegex(ValueError, "UNIQUE_TERMINAL"):
            generation.check_prefix(self.plan, self.generation_root, self.ids[8:16])

    def test_changed_selection_and_changed_immutable_bytes_are_rejected(self):
        for selected in ([], self.ids[:2][::-1], [self.ids[0], self.ids[2]], [self.ids[0]] * 2, ["foreign"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                generation.check_prefix(self.plan, self.generation_root, selected)
        source = self.base / "source.json"
        source.write_bytes(b'{"source":"original"}\n')
        pins = {source.name: common.record(source)}
        generation.check_records(self.base, pins)
        source.write_bytes(b'{"source":"changed"}\n')
        with self.assertRaisesRegex(ValueError, "PINNED_FILE_CHANGED"):
            generation.check_records(self.base, pins)
        with self.assertRaises(ValueError):
            generation.check_records(self.base, {"../elsewhere": pins[source.name]})

    def preparation_fixture(self):
        source, producer, runtime, audio = [self.base / name for name in ("source", "producer", "runtime", "audio")]
        for path in (source, producer, runtime, audio):
            path.mkdir()
        for name in generation.OWN_FILES:
            common.once_bytes(source / name, ("fixture source: " + name).encode())
        assets = source / "configs/runtime/qwen-base-tts-model-lock.json"
        assets.write_bytes(common.canonical({"model_id": "fixture-only", "revision": "0" * 40, "files": []}))
        for name in generation.DATA_FILES:
            common.once_bytes(producer / name, ("fixture producer: " + name).encode())
        common.once(runtime / "setup.json", {"status": "complete", "installed_packages": {}})
        (runtime / "venv/bin").mkdir(parents=True)
        (runtime / "venv/bin/python").symlink_to(sys.executable)
        plan_path, manifest = self.base / "plan.json", self.base / "sources.json"
        common.once(plan_path, self.plan)
        common.once(manifest, {"references": [], "fixture": True})
        args = Namespace(output=self.base / "prepared-a", plan=plan_path, sources=manifest,
                         data_root=producer, audio_root=audio, runtime=runtime,
                         heavy_lock=self.base / "heavy.lock", chunk_size=8,
                         global_minutes=90, job_minutes=10, active_plan=ACTIVE_PLAN)
        return source, args

    def test_prepare_pins_all_inputs_and_keeps_first_eight_in_one_job(self):
        source, args = self.preparation_fixture()
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            generation.prepare(args)
            stage_path = args.output / "stage.json"
            stage, prepared, _, _ = generation.load_stage(stage_path)
            self.assertEqual(stage["chunks"], [self.ids[i:i+8] for i in range(0,72,8)])
            self.assertEqual(stage["context_v1_cells_generated_again"], 0)
            request = common.read(args.output / "queue-request.json")
            self.assertEqual(request["global_budget_minutes"], 90)
            self.assertEqual([job["timeout_minutes"] for job in request["jobs"]], [10] * 9)
            self.assertEqual(request["jobs"][1]["depends_on"], [request["jobs"][0]["job_id"]])
            for relative in ("plan.json", "source-manifest.json", "model-assets.json", "runtime-setup.json",
                             "producer-source/" + generation.DATA_FILES[0]):
                path = prepared / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    generation.load_stage(stage_path)
                path.write_bytes(original)
            own = source / generation.OWN_FILES[0]
            own.write_bytes(own.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                generation.load_stage(stage_path)

    def test_v1_plan_is_rejected_even_when_the_planner_fixture_accepts_it(self):
        source, args = self.preparation_fixture()
        value = common.read(args.plan)
        value["schema"] = "kws-context-supplement-plan-v1"
        args.plan.write_bytes(common.canonical(value))
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            with self.assertRaisesRegex(ValueError, "INCREMENT_REQUIRED"):
                generation.prepare(args)

    def test_checkpoint_contains_current_chunk_and_intent_receipts_without_prior_audio(self):
        source, args = self.preparation_fixture()
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            generation.prepare(args)
        prepared = args.output
        stage = common.read(prepared / "stage.json")
        current = prepared / "queue/jobs/generate-002/outputs/native/current.wav"
        prior = prepared / "queue/jobs/generate-001/outputs/native/prior.wav"
        current_cell = prepared / "generation/cells" / self.ids[8] / "completed.json"
        prior_cell = prepared / "generation/cells" / self.ids[0] / "completed.json"
        for path in (current, prior, current_cell, prior_cell):
            common.once_bytes(path, b"explicit-fixture-only")
        intent = prepared / "checkpoints/generate-002/before-generation"
        metadata = [intent / name for name in ("capsule.json", "ack.json", "provider-save.json", "phase-marker.json")]
        old_zip = intent / "checkpoint-fixture.zip"
        for path in [*metadata, old_zip]:
            common.once(path, {"explicit_fixture_only": True})
        files = set(generation.checkpoint_sources(stage, prepared, "generate-002"))
        self.assertTrue({current, current_cell, *metadata} <= files)
        self.assertFalse({prior, prior_cell, old_zip} & files)
        self.assertIn(prepared / "execution-source/tools/research_d2_checkpoints.py", files)
        self.assertIn(prepared / "execution-source/tools/research_queue.py", files)
        self.assertIn(prepared / "runtime-setup.json", files)
        self.assertIn(prepared / "model-assets.json", files)

    def test_changed_context_parent_has_a_distinct_registration_identity(self):
        before = generation.campaign_identity(self.plan)
        changed = copy.deepcopy(self.plan)
        changed["context_parent"]["plan_sha256"] = "3" * 64
        self.assertNotEqual(generation.campaign_identity(changed), before)
        self.assertEqual(generation.campaign_identity(copy.deepcopy(self.plan)), before)

    def test_semantically_identical_plan_cannot_register_again_with_different_json_format(self):
        source, args = self.preparation_fixture()
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            generation.prepare(args)
            args.plan.write_text(json.dumps(self.plan, indent=2, ensure_ascii=False) + "\n")
            args.output = self.base / "prepared-b"
            with self.assertRaises((ValueError, FileExistsError)):
                generation.prepare(args)

    def test_same_campaign_scope_cannot_be_registered_with_a_different_plan(self):
        source, args = self.preparation_fixture()
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            generation.prepare(args)
            changed = copy.deepcopy(self.plan)
            changed["model_revision"] = "2" * 40
            args.plan.write_bytes(common.canonical(changed))
            args.output = self.base / "prepared-b"
            with self.assertRaises((ValueError, FileExistsError)):
                generation.prepare(args)

    def test_prepare_rejects_call_budget_smaller_than_full_declared_roster(self):
        source, args = self.preparation_fixture()
        changed = copy.deepcopy(self.plan)
        changed["budgets"]["max_total_calls"] = 8
        args.plan.write_bytes(common.canonical(changed))
        with mock.patch.object(generation, "ROOT", source), mock.patch.object(generation, "load_context_v3_producer", return_value=PlannerFixture()):
            with self.assertRaises(ValueError):
                generation.prepare(args)

    def test_job_recipe_rejects_chunk_type_selection_and_generation_drift(self):
        path = self.base / "stage.json"
        stage = {"chunks": [self.ids[i:i+8] for i in range(0,72,8)], "generation": common.GENERATION, "worker_timeout_seconds": generation.WORKER_SECONDS}
        common.once(path, stage)
        recipe = {"schema": generation.JOB_SCHEMA, "stage": {"path": str(path), **common.record(path)},
                  "job_id": "generate-001", "chunk_index": 0, "selected_cells": self.ids[:8],
                  "max_calls": 8, "generation": common.GENERATION, "worker_timeout_seconds": generation.WORKER_SECONDS}
        generation.validate_job_recipe(recipe, stage, path)
        changed_values = (("chunk_index", False), ("chunk_index", -1), ("chunk_index", 0.0),
                          ("job_id", "generate-002"), ("selected_cells", self.ids[8:16]),
                          ("max_calls", 9), ("generation", {**common.GENERATION, "temperature": .5}))
        for field, value in changed_values:
            changed = {**recipe, field: value}
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "JOB_SCOPE"):
                generation.validate_job_recipe(changed, stage, path)
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "JOB_STAGE_CHANGED"):
            generation.validate_job_recipe(recipe, stage, path)

    def guarded_fixture(self, job_count=1):
        """One real toy process; the separate72-cell planner is not a model."""
        prepared=self.base/"guarded-prepared";prepared.mkdir()
        qroot=prepared/"queue";stage_path=prepared/"stage.json"
        stage={"prepared":str(prepared),"queue_root":str(qroot),
               "generation_root":str(prepared/"generation"),"chunks":[self.ids[i:i+1] for i in range(job_count)],
               "generation":common.GENERATION,"worker_timeout_seconds":generation.WORKER_SECONDS}
        common.once(stage_path,stage);common.once(prepared/"plan.json",self.plan)
        recipe={"schema":generation.JOB_SCHEMA,"stage":{"path":str(stage_path),**common.record(stage_path)},
                "job_id":"generate-001","chunk_index":0,"selected_cells":self.ids[:1],
                "max_calls":1,"generation":common.GENERATION,"worker_timeout_seconds":generation.WORKER_SECONDS}
        recipe_path=prepared/"recipe.json";common.once(recipe_path,recipe)
        marker=prepared/"toy-called-once.json"
        command="""import os,json,sys
from pathlib import Path
value={'fixture':True,'owner':os.environ['KWS_CONTEXT_V3_INTENT_OWNER'],'intent':os.environ['KWS_CONTEXT_V3_INTENT_FOLDER']}
with Path(sys.argv[1]).open('x') as f:json.dump(value,f)
(Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])/'proof.json').write_text(json.dumps(value))
"""
        request={"schema":queue.SCHEMA,"queue_id":"context-guard-toy","global_budget_minutes":1,
                 "jobs":[{"job_id":"generate-001","recipe":str(recipe_path),
                    "recipe_sha256":common.sha_file(recipe_path),
                    "argv":[sys.executable,"-I","-S","-B","-c",command,str(marker)],
                    "cwd":str(prepared),"depends_on":[],"timeout_minutes":.1,"outputs":["proof.json"]}]}
        if job_count == 2:
            second_path = prepared / "recipe-second.json"
            second = {**recipe, "job_id": "generate-002", "chunk_index": 1, "selected_cells": self.ids[1:2]}
            common.once(second_path, second)
            request["jobs"].append({**request["jobs"][0], "job_id": "generate-002", "recipe": str(second_path),
                "recipe_sha256": common.sha_file(second_path), "depends_on": ["generate-001"],
                "argv": [sys.executable,"-I","-S","-B","-c",command,str(prepared/"second-toy-called.json")]})
        common.once(prepared/"request.json",request)
        queue.create(qroot,prepared/"request.json",ACTIVE_PLAN,self.base/"guarded-heavy.lock")
        return stage_path,stage,prepared,marker

    def fixture_save(self,folder,descriptor):
        import research_d2_checkpoints as checkpoints
        receipt=folder/"explicit-fixture-provider.json"
        common.once(receipt,{"schema":checkpoints.SAVE,"status":"saved","provider":"explicit-fixture",
            "object_id":"fixture-only","provider_response":{"saved_id":"fixture-only"},
            "artifact":descriptor["capsule"],"member_manifest_sha256":descriptor["member_manifest_sha256"]})
        checkpoints.acknowledge(folder,receipt)

    def test_guarded_queue_calls_once_only_after_real_capsule_ack_then_seals_terminal(self):
        import research_d2_checkpoints as checkpoints
        path,stage,prepared,marker=self.guarded_fixture()
        phases=[];real_gate=checkpoints.Gate
        def save(folder,descriptor):
            phases.append(descriptor["context"]["phase"])
            self.assertEqual(marker.exists(),not descriptor["context"]["launch_intent"])
            self.fixture_save(folder,descriptor)
        def gate(**kwargs):return real_gate(wait_seconds=.01,max_wait_seconds=0,on_wait=save)
        files=lambda *_:[path,prepared/"plan.json",*checkpoints.regular_tree(prepared/"queue")]
        args=Namespace(stage=path,checkpoint_wait_seconds=0)
        with mock.patch.object(generation,"load_stage",return_value=(stage,prepared,self.plan,PlannerFixture())), \
             mock.patch.object(generation,"checkpoint_sources",side_effect=files), \
             mock.patch.object(checkpoints,"Gate",side_effect=gate):
            result=generation.guarded_next(args)
            self.assertEqual(result["status"],"completed")
            self.assertEqual(phases,["before-generation","generation-complete"])
            original=marker.read_bytes()
            self.assertEqual(generation.guarded_next(args)["new_jobs"],0)
            self.assertEqual(marker.read_bytes(),original)
        before=checkpoints.read(prepared/"checkpoints/generate-001/before-generation/capsule.json")
        after=checkpoints.read(prepared/"checkpoints/generate-001/generation-complete/capsule.json")
        self.assertEqual(before["context"]["owner_invocation"],after["context"]["owner_invocation"])
        self.assertEqual(common.read(marker)["owner"],before["context"]["owner_invocation"])

    def test_unacknowledged_old_intent_cannot_launch_even_when_queue_is_pending(self):
        import research_d2_checkpoints as checkpoints
        path,stage,prepared,marker=self.guarded_fixture()
        files=lambda *_:[path,prepared/"plan.json",*checkpoints.regular_tree(prepared/"queue")]
        with mock.patch.object(generation,"load_stage",return_value=(stage,prepared,self.plan,PlannerFixture())), \
             mock.patch.object(generation,"checkpoint_sources",side_effect=files), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"SAVE_NOT_ACKNOWLEDGED"):
                generation.guarded_next(Namespace(stage=path,checkpoint_wait_seconds=0))
            self.assertFalse(marker.exists())
            with self.assertRaisesRegex(ValueError,"AMBIGUOUS_INTENT_AFTER_RESTART"):
                generation.guarded_next(Namespace(stage=path,checkpoint_wait_seconds=0))
        self.assertEqual(queue.inspect(prepared/"queue")["jobs"]["generate-001"]["status"],"pending")
        self.assertFalse(marker.exists())

    def test_lost_claim_is_saved_as_unknown_before_halt_without_later_launch(self):
        import research_d2_checkpoints as checkpoints
        path, stage, prepared, marker = self.guarded_fixture(job_count=2)
        qroot, _, request, queue_sha = queue.load_queue(prepared / "queue")
        claim = {"schema": "kws-local-research-job-claim-v1", **queue.job_binding(request["jobs"][0], queue_sha),
                 "attempt": 1, "automatic_retry": False, "explicit_orphan_fixture": True}
        common.once(qroot / "jobs/generate-001/claim.json", claim)
        phases = []; real_gate = checkpoints.Gate
        def save(folder, descriptor):
            phases.append(descriptor["context"]["phase"])
            self.assertEqual(descriptor["context"]["queue_result_status"], "unknown")
            self.fixture_save(folder, descriptor)
        files = lambda *_: [path, prepared / "plan.json", *checkpoints.regular_tree(qroot)]
        with mock.patch.object(generation, "load_stage", return_value=(stage,prepared,self.plan,PlannerFixture())), \
             mock.patch.object(generation, "checkpoint_sources", side_effect=files), \
             mock.patch.object(checkpoints, "Gate", side_effect=lambda **_: real_gate(max_wait_seconds=0,on_wait=save)), \
             mock.patch.object(queue, "run", wraps=queue.run) as run:
            with self.assertRaisesRegex(ValueError, "UNRESOLVED_QUEUE"):
                generation.guarded_next(Namespace(stage=path,checkpoint_wait_seconds=0))
            run.assert_not_called()
        self.assertEqual(phases, ["generation-terminal"])
        checkpoints.verify_ack(prepared / "checkpoints/generate-001/generation-terminal")
        result = common.read(qroot / "jobs/generate-001/result.json")
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["actual_elapsed_ns"])
        self.assertFalse((qroot / "jobs/generate-002/claim.json").exists())
        self.assertFalse(marker.exists())
        self.assertFalse((prepared / "second-toy-called.json").exists())

    def test_worker_requires_persisted_exact_live_intent_before_model_construction(self):
        import research_d2_checkpoints as checkpoints
        path,stage,prepared,_marker=self.guarded_fixture()
        recipe_path=prepared/"queue/recipes/generate-001.json";recipe=common.read(recipe_path)
        with self.assertRaisesRegex(ValueError,"PERSISTED_LAUNCH_INTENT_REQUIRED"):
            generation.verify_generation_intent(path,stage,recipe_path,recipe)
        folder=prepared/"checkpoints/generate-001/before-generation"
        gate=checkpoints.Gate(max_wait_seconds=0,on_wait=self.fixture_save)
        gate.checkpoint(folder,generation.checkpoint_scope(path,stage,recipe_path,recipe,"before-generation"),
                        [path,recipe_path],intent=True)
        env={generation.INTENT_FOLDER_ENV:str(folder),generation.INTENT_OWNER_ENV:gate.owner}
        with mock.patch.dict(os.environ,env):
            generation.verify_generation_intent(path,stage,recipe_path,recipe)
            with mock.patch.dict(os.environ,{generation.INTENT_OWNER_ENV:"different-live-driver"}):
                with self.assertRaisesRegex(ValueError,"INTENT_SCOPE_CHANGED"):
                    generation.verify_generation_intent(path,stage,recipe_path,recipe)
            changed={**recipe,"selected_cells":self.ids[1:2]}
            with self.assertRaisesRegex(ValueError,"INTENT_SCOPE_CHANGED"):
                generation.verify_generation_intent(path,stage,recipe_path,changed)

    def test_declared_campaign_prevents_standalone_generation_from_skipping_asr_closure(self):
        path, stage, prepared, marker = self.guarded_fixture()
        common.once(prepared / "campaign-driver.json", {"explicit_campaign_fixture": True})
        with mock.patch.object(generation, "load_stage", return_value=(stage,prepared,self.plan,PlannerFixture())), \
             mock.patch.object(queue, "run", wraps=queue.run) as run:
            with self.assertRaisesRegex(ValueError, "CAMPAIGN_OWNS_STAGE"):
                generation.guarded_next(Namespace(stage=path,checkpoint_wait_seconds=0))
            run.assert_not_called()
        self.assertFalse(marker.exists())
        self.assertFalse((prepared / "checkpoints").exists())

    def test_worker_deadline_cannot_be_swallowed_as_a_cell_quality_failure(self):
        with mock.patch.object(generation,"_run_model",side_effect=generation.WorkerDeadline("fixture")):
            with self.assertRaises(generation.WorkerDeadline):generation.run(Namespace())
        self.assertEqual(generation.signal.getitimer(generation.signal.ITIMER_REAL),(0.0,0.0))

    def test_synchronized_stage_generation_mutation_still_fails_independent_plan_binding(self):
        source,args=self.preparation_fixture()
        with mock.patch.object(generation,"ROOT",source), \
             mock.patch.object(generation,"load_context_v3_producer",return_value=PlannerFixture()):
            generation.prepare(args)
            path=args.output/"stage.json";stage=common.read(path)
            stage["generation"]={**common.GENERATION,"temperature":.5}
            path.write_bytes(common.canonical(stage))
            with mock.patch.object(common,"GENERATION",stage["generation"]):
                with self.assertRaisesRegex(ValueError,"PLAN_GENERATION_CHANGED"):
                    generation.load_stage(path)

    def export_fixture(self, *, failed=False, duplicate=False):
        """Create real queue receipts around a plainly synthetic output fixture."""
        prepared = self.base / "export-prepared"
        prepared.mkdir()
        stage_path, qroot = prepared / "stage.json", prepared / "queue"
        generation_root, stash = prepared / "generation", prepared / "fixture-output"
        stash.mkdir()
        ident, job_id = self.ids[0], "generate-001"
        plan = {**self.plan, "cells": self.plan["cells"][:1]}
        common.once(prepared / "plan.json", plan)
        stage = {"schema": generation.SCHEMA, "prepared": str(prepared), "queue_root": str(qroot),
                 "generation_root": str(generation_root), "chunks": [[ident]], "max_new_calls": 1,
                 "generation": common.GENERATION, "worker_timeout_seconds": generation.WORKER_SECONDS}
        common.once(stage_path, stage)
        recipe = {"schema": generation.JOB_SCHEMA, "stage": {"path": str(stage_path), **common.record(stage_path)},
                  "job_id": job_id, "chunk_index": 0, "selected_cells": [ident],
                  "max_calls": 1, "generation": common.GENERATION, "worker_timeout_seconds": generation.WORKER_SECONDS}
        recipe_path = prepared / "recipe.json"
        common.once(recipe_path, recipe)
        native = stash / "native" / ident
        native.mkdir(parents=True)
        pcm = struct.pack("<" + "f" * 48, *([.1, -.1] * 24))
        fmt = struct.pack("<HHIIHH", 3, 1, 24000, 24000 * 4, 4, 32)
        wav = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
        common.once_bytes(native / "returned-native.wav", wav)
        common.once(native / "native.json", {"schema": "active-tts-native-return-v1", "shape": [48],
                    "sample_rate_hz": 24000, "dtype": "<f4", "all_finite": True,
                    "wav": common.record(native / "returned-native.wav"), "technical_gate_applied": False})
        common.once(native / "returned.json", {"fixture": True, "generation_seconds": .01})
        common.once(native / "call.json", {"fixture": True, "cell_id": ident})
        cell_root = generation_root / "cells" / ident
        common.once(cell_root / "started.json", {"fixture": True, "cell_id": ident})
        common.once(cell_root / ("failed.json" if failed else "completed.json"), {"fixture": True, "cell_id": ident})
        row = {"cell_id": ident, "generation_outcome": "failed-native" if failed else "completed",
               "native_returned": True, "observed_text": None, "training_admitted": False}
        rows = [row, copy.deepcopy(row)] if duplicate else [row]
        common.once(stash / "inventory.json", {
            "files": {p.relative_to(stash).as_posix(): common.record(p) for p in stash.rglob("*") if p.is_file()},
            "generation_root": str(generation_root),
            "generation_files": {p.relative_to(generation_root).as_posix(): common.record(p)
                                 for p in generation_root.rglob("*") if p.is_file()}})
        summary = {"schema": "kws-context-v3-generation-result-v1", "status": "completed",
                   "recipe_sha256": common.sha_file(recipe_path), "job_id": job_id,
                   "claims": len(rows), "native_generation_calls": len(rows), "native_returns": len(rows),
                   "rows": rows, "outcomes": dict(Counter(row["generation_outcome"] for row in rows)),
                   "generation_root": str(generation_root), "asr_calls": 0, "human_gold": False}
        common.once(stash / "summary.json", summary)
        command = "from pathlib import Path;import os,shutil,sys;shutil.copytree(sys.argv[1],os.environ['KWS_RESEARCH_OUTPUT_DIR'],dirs_exist_ok=True)"
        request = {"schema": queue.SCHEMA, "queue_id": "d2-export-fixture", "global_budget_minutes": 1,
                   "jobs": [{"job_id": job_id, "recipe": str(recipe_path),
                              "recipe_sha256": common.sha_file(recipe_path),
                              "argv": [sys.executable, "-I", "-S", "-B", "-c", command, str(stash)],
                              "cwd": str(prepared), "depends_on": [], "timeout_minutes": .1,
                              "outputs": ["summary.json", "inventory.json"]}]}
        request_path = prepared / "request.json"
        common.once(request_path, request)
        queue.create(qroot, request_path, ACTIVE_PLAN, self.base / "export-heavy.lock")
        self.assertEqual(queue.run(qroot)[1], 0)
        return stage_path, stage, prepared, plan, qroot / "jobs" / job_id, wav

    def export(self, fixture):
        stage_path, stage, prepared, plan, job, _wav = fixture
        with mock.patch.object(generation, "load_stage", return_value=(stage, prepared, plan, PlannerFixture())):
            return generation.export_job(stage_path, job.name, self.base / "asr-export")

    def test_failed_native_exports_original_audio_without_a_text_or_training_label(self):
        fixture = self.export_fixture(failed=True)
        result = self.export(fixture)
        self.assertEqual(result["new_asr_calls"], 0)
        self.assertIsNone(result["recognizer_context_text"])
        self.assertEqual(len(result["included"]), 1)
        self.assertEqual(result["included"][0]["generation_outcome"], "failed-native")
        self.assertIsNone(result["included"][0]["observed_text"])
        self.assertFalse(result["included"][0]["training_admitted"])
        with zipfile.ZipFile(self.base / "asr-export/source-original.zip") as archive:
            self.assertEqual(archive.read("cells/" + self.ids[0] + "/native-f32.wav"), fixture[-1])

    def test_export_rejects_tampered_queue_result_binding(self):
        fixture = self.export_fixture()
        result_path = fixture[4] / "result.json"
        result = common.read(result_path)
        result["claim_sha256"] = "f" * 64
        result_path.write_bytes(common.canonical(result))
        with self.assertRaises((ValueError, KeyError)):
            self.export(fixture)
        self.assertFalse((self.base / "asr-export").exists())

    def test_export_rejects_changed_summary_instead_of_trusting_its_rows(self):
        fixture = self.export_fixture()
        path = fixture[4] / "outputs/summary.json"
        summary = common.read(path)
        summary["rows"][0]["observed_text"] = "fixture invented transcript"
        path.write_bytes(common.canonical(summary))
        with self.assertRaises(ValueError):
            self.export(fixture)
        self.assertFalse((self.base / "asr-export").exists())

    def test_export_rejects_modified_inventory_and_audio_even_when_file_lists_are_cleared(self):
        fixture = self.export_fixture()
        output = fixture[4] / "outputs"
        inventory = common.read(output / "inventory.json")
        inventory["files"] = {}
        inventory["generation_files"] = {}
        (output / "inventory.json").write_bytes(common.canonical(inventory))
        (output / "native" / self.ids[0] / "returned-native.wav").write_bytes(b"not a waveform")
        with self.assertRaises(ValueError):
            self.export(fixture)
        self.assertFalse((self.base / "asr-export").exists())

    def test_export_rejects_duplicate_rows_even_if_the_queue_authenticates_their_bytes(self):
        fixture = self.export_fixture(duplicate=True)
        with self.assertRaises(ValueError):
            self.export(fixture)
        self.assertFalse((self.base / "asr-export").exists())


if __name__ == "__main__":
    unittest.main()
