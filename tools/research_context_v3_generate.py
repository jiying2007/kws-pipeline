#!/usr/bin/env python3
"""Run immutable fixed-vocabulary new24 context v3 supplements through the existing bounded research queue.

Generation claims are never retried. Every native return precedes quality
screening; failed quality and missing returns remain distinct. The frozen D1 and D2 plans are
only ancestors and are never submitted again by this entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_tts_smoke as common
import research_queue as queue
import research_runtime
import research_no_network
from research_job_lock import heavy_model_slot

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kws-context-v3-generation-stage-v1"
JOB_SCHEMA = "kws-context-v3-generation-job-v1"
WORKER_SECONDS = 570
INTENT_FOLDER_ENV = "KWS_CONTEXT_V3_INTENT_FOLDER"
INTENT_OWNER_ENV = "KWS_CONTEXT_V3_INTENT_OWNER"
DATA_FILES = common.SOURCE_FILES + (str(common.BULK / "d2_pool.py"),
                                    str(common.BULK / "context_supplement.py"),
                                    str(common.BULK / "context_supplement_v2.py"),
                                    str(common.BULK / "context_supplement_v3.py"))
OWN_FILES = ("tools/research_context_v3_generate.py", "tools/active_tts_smoke.py",
             "tools/research_queue.py", "tools/research_job_lock.py",
             "tools/research_runtime.py", "tools/research_no_network.py", "tools/research_d2_checkpoints.py",
             "configs/runtime/research-runtime.json", "research/qwen6_tts/runtime-lock.json",
             "research/qwen6_tts/model-locks.json", "configs/runtime/qwen-base-tts-model-lock.json")


def require(ok, reason):
    common.require(ok, reason)


def load_context_v3_producer(snapshot):
    """Adapt the independent context contract without changing global modules."""
    from types import SimpleNamespace
    base = common.load_bulk(snapshot)
    context = importlib.import_module("context_supplement_v3")
    require(Path(context.__file__).resolve() ==
            (Path(snapshot) / common.BULK / "context_supplement_v3.py").resolve(),
            "CONTEXT_PRODUCER_IMPORT_CHANGED")
    return SimpleNamespace(p=base.p, LOCK_SCHEMA=base.LOCK_SCHEMA,
        asset_lock=base.asset_lock, QwenBulkClone=base.QwenBulkClone,
        verify_plan=context.verify_plan, rebuild_plan=context.rebuild, run=context.run)


def source_records(root, names):
    return {name: common.record(Path(root) / name) for name in names}


def check_records(root, records):
    for name, expected in records.items():
        require(common.record(Path(root) / common.relative(name)) == expected,
                "CONTEXT_PINNED_FILE_CHANGED:" + name)


def partition(plan, chunk_size):
    require(type(chunk_size) is int and 1 <= chunk_size <= 8, "CONTEXT_CHUNK_BUDGET")
    ids = [cell["cell_id"] for cell in plan["cells"]]
    require(ids and len(ids) == len(set(ids)) and len(ids) == 72, "CONTEXT_CELL_ROSTER")
    require(plan["budgets"]["max_total_calls"] == len(ids), "CONTEXT_FULL_STAGE_CALL_BUDGET_REQUIRED")
    return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]


def campaign_identity(plan):
    """A changed serialization/reference cannot re-register the same campaign."""
    scope = {"parent_plan_sha256": plan["context_parent"]["plan_sha256"],
             "d2_parent_plan_sha256": plan["d2_parent"]["plan_sha256"],
             "stage": plan["stage"], "batch_id": plan["batch_id"]}
    return hashlib.sha256(common.canonical(scope)).hexdigest()


def validate_job_recipe(recipe, stage, stage_path):
    require(recipe.get("schema") == JOB_SCHEMA and
            recipe.get("stage") == {"path": str(stage_path), **common.record(stage_path)},
            "CONTEXT_JOB_STAGE_CHANGED")
    index = recipe.get("chunk_index")
    require(type(index) is int and 0 <= index < len(stage["chunks"]) and
            recipe.get("job_id") == f"generate-{index + 1:03d}" and
            recipe.get("selected_cells") == stage["chunks"][index] and
            recipe.get("max_calls") == len(stage["chunks"][index]) and
            recipe.get("worker_timeout_seconds") == stage["worker_timeout_seconds"] == WORKER_SECONDS and
            recipe.get("generation") == stage["generation"], "CONTEXT_JOB_SCOPE")


def prepare(args):
    output = Path(args.output).absolute()
    require(not output.exists() and output.parent.is_dir(), "CONTEXT_CREATE_ONLY_PREPARATION")
    require(args.global_minutes == 90 and args.job_minutes == 10,
            "CONTEXT_EXPLICIT_RUNNER_BUDGET")
    queue.validate_plan(common.read(args.active_plan))
    output.mkdir(mode=0o700)
    for name in DATA_FILES:
        common.once_bytes(output / "producer-source" / name, (args.data_root / name).read_bytes())
    bulk = load_context_v3_producer(output / "producer-source")
    plan = common.read(args.plan)
    require(plan.get("schema") == "kws-context-supplement-plan-v3", "CONTEXT_INCREMENT_REQUIRED")
    bulk.verify_plan(plan)
    require(plan["generation"] == common.GENERATION, "CONTEXT_PLAN_GENERATION_CHANGED")
    manifest = common.read(args.sources)
    require(bulk.p.canonical(bulk.rebuild_plan(plan, manifest, args.audio_root)) ==
            bulk.p.canonical(plan), "CONTEXT_SOURCE_PLAN_CHANGED")
    common.once_bytes(output / "plan.json", args.plan.read_bytes())
    common.once_bytes(output / "source-manifest.json", args.sources.read_bytes())
    for ref in plan["references"]:
        for name in (ref["audio_path"], ref["rights_record_path"]):
            target = output / "reference-assets" / common.relative(name)
            if not target.exists():
                common.once_bytes(target, (args.audio_root / name).read_bytes())
    assets = common.read(ROOT / "configs/runtime/qwen-base-tts-model-lock.json")
    model_lock = {"schema": bulk.LOCK_SCHEMA, "model_id": assets["model_id"],
                  "revision": assets["revision"], "files": [
        {"path": row["filename"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in assets["files"]]}
    bulk.asset_lock(args.runtime / "models/base", model_lock, plan["model_revision"])
    common.once(output / "model-assets.json", model_lock)
    common.once_bytes(output / "runtime-setup.json", (args.runtime / "setup.json").read_bytes())
    for name in OWN_FILES:
        common.once_bytes(output / "execution-source" / name, (ROOT / name).read_bytes())
    chunks = partition(plan, args.chunk_size)
    stage = {"schema": SCHEMA, "prepared": str(output), "runtime": str(args.runtime.absolute()),
             "heavy_lock": str(args.heavy_lock.absolute()), "queue_root": str(output / "queue"),
             "generation_root": str(output / "generation"), "created_at": datetime.now(timezone.utc).isoformat(),
             "plan": common.record(output / "plan.json"),
             "source_manifest": common.record(output / "source-manifest.json"),
             "model_assets": common.record(output / "model-assets.json"),
             "runtime_setup": common.record(output / "runtime-setup.json"),
             "data_source": source_records(output / "producer-source", DATA_FILES),
             "execution_source": source_records(ROOT, OWN_FILES), "chunks": chunks,
             "global_budget_minutes": args.global_minutes, "job_timeout_minutes": args.job_minutes,
             "worker_timeout_seconds": WORKER_SECONDS,
             "max_new_calls": len(plan["cells"]), "generation": plan["generation"],
             "model_concurrency": 1, "automatic_retries": 0,
             "d1_cells_generated_again": 0, "d2_cells_generated_again": 0, "context_v1_cells_generated_again": 0, "context_v2_cells_generated_again": 0,
             "historical_prompts_generated_again": 0, "human_gold": False,
             "durable_checkpoint_policy": "external-save-before-and-after-each-generation-v1",
             "allocation": {"generation_runner_minutes": args.global_minutes,
                            "selective_asr_runner_minutes_ceiling": 90,
                            "aggregate_runner_minutes_ceiling": args.global_minutes + 90,
                            "maximum_asr_calls": 216, "prior_allocations_reused": False}}
    common.once(output / "stage.json", stage)
    jobs = []
    stage_pin = {"path": str(output / "stage.json"), **common.record(output / "stage.json")}
    for number, cells in enumerate(chunks):
        ident = f"generate-{number + 1:03d}"
        recipe_path = output / "recipes" / (ident + ".json")
        recipe = {"schema": JOB_SCHEMA, "stage": stage_pin, "job_id": ident,
                  "chunk_index": number, "selected_cells": cells,
                  "max_calls": len(cells), "generation": plan["generation"],
                  "worker_timeout_seconds": WORKER_SECONDS}
        common.once(recipe_path, recipe)
        jobs.append({"job_id": ident, "recipe": str(recipe_path),
                     "recipe_sha256": common.sha_file(recipe_path),
                     "argv": [str(args.runtime / "venv/bin/python"), "-B", str(ROOT / "tools/research_context_v3_generate.py"),
                              "run", "--recipe", str(recipe_path), "--output", "{output_dir}",
                              "--heavy-lock", str(args.heavy_lock.absolute())],
                     "cwd": str(ROOT), "depends_on": [] if number == 0 else [f"generate-{number:03d}"],
                     "timeout_minutes": args.job_minutes, "outputs": ["summary.json", "inventory.json"]})
    request = {"schema": queue.SCHEMA, "queue_id": output.name,
               "global_budget_minutes": args.global_minutes, "jobs": jobs}
    queue.validate_request(request)
    common.once(output / "queue-request.json", request)
    # A process restart resumes this same queue; identical plans cannot be
    # registered in a second output directory and regenerate their cells.
    registrations = args.runtime / "context-v3-campaigns"
    registrations.mkdir(exist_ok=True, mode=0o700)
    with queue.exclusive(registrations / "registration.lock"):
        registration = registrations / (campaign_identity(plan) + ".json")
        require(not registration.exists(), "CONTEXT_CAMPAIGN_ALREADY_REGISTERED")
        state = queue.create(output / "queue", output / "queue-request.json", args.active_plan, args.heavy_lock)
        common.once(registration, {"stage": stage_pin,
            "canonical_plan_sha256": hashlib.sha256(bulk.p.canonical(plan)).hexdigest(),
            "campaign_identity": campaign_identity(plan)})
    return {"status": "prepared", "stage": str(output / "stage.json"),
            "new_context_cells": stage["max_new_calls"], "jobs": len(jobs), "queue": state}


def load_stage(path):
    stage = common.read(path)
    require(stage["schema"] == SCHEMA and stage["model_concurrency"] == 1 and
            stage["automatic_retries"] == 0 and stage["d1_cells_generated_again"] == 0 and
            stage["d2_cells_generated_again"] == 0 and
            stage["context_v1_cells_generated_again"] == 0 and
            stage["context_v2_cells_generated_again"] == 0 and
            stage["historical_prompts_generated_again"] == 0 and
            stage["durable_checkpoint_policy"] == "external-save-before-and-after-each-generation-v1" and
            stage["worker_timeout_seconds"] == WORKER_SECONDS and
            stage["global_budget_minutes"] == 90 and stage["job_timeout_minutes"] == 10 and
            stage["allocation"] == {"generation_runner_minutes": stage["global_budget_minutes"],
                "selective_asr_runner_minutes_ceiling": 90,
                "aggregate_runner_minutes_ceiling": stage["global_budget_minutes"] + 90,
                "maximum_asr_calls": 216, "prior_allocations_reused": False} and
            stage["generation"] == common.GENERATION, "CONTEXT_STAGE_POLICY")
    root = Path(stage["prepared"])
    for key, name in (("plan", "plan.json"), ("source_manifest", "source-manifest.json"),
                      ("model_assets", "model-assets.json"), ("runtime_setup", "runtime-setup.json")):
        require(common.record(root / name) == stage[key], "CONTEXT_STAGE_INPUT_CHANGED:" + name)
    check_records(root / "producer-source", stage["data_source"])
    require(set(stage["data_source"]) == set(DATA_FILES), "CONTEXT_DATA_SOURCE_ROSTER")
    check_records(ROOT, stage["execution_source"])
    require(set(stage["execution_source"]) == set(OWN_FILES), "CONTEXT_EXECUTION_SOURCE_ROSTER")
    plan = common.read(root / "plan.json")
    require([value for group in stage["chunks"] for value in group] ==
            [cell["cell_id"] for cell in plan["cells"]] and
            stage["max_new_calls"] == len(plan["cells"]) == plan["budgets"]["max_total_calls"] and
            all(1 <= len(group) <= 8 for group in stage["chunks"]), "CONTEXT_CHUNK_COVERAGE")
    bulk = load_context_v3_producer(root / "producer-source")
    bulk.verify_plan(plan)
    require(stage["generation"] == plan["generation"], "CONTEXT_PLAN_GENERATION_CHANGED")
    require(bulk.p.canonical(bulk.rebuild_plan(plan, common.read(root / "source-manifest.json"),
                root / "reference-assets")) == bulk.p.canonical(plan), "CONTEXT_SOURCE_PLAN_CHANGED")
    return stage, root, plan, bulk


def check_prefix(plan, generation_root, selected):
    ids = [cell["cell_id"] for cell in plan["cells"]]
    require(selected and len(selected) <= 8 and selected == ids[ids.index(selected[0]):ids.index(selected[0]) + len(selected)],
            "CONTEXT_JOB_SELECTION_CHANGED")
    prior = ids[:ids.index(selected[0])]
    started = [ident for ident in ids if (generation_root / "cells" / ident / "started.json").exists()]
    require(started == prior, "CONTEXT_PREFIX_NOT_EXACT_OR_ALREADY_CLAIMED")
    for ident in prior:
        folder = generation_root / "cells" / ident
        require((folder / "completed.json").exists() != (folder / "failed.json").exists(),
                "CONTEXT_PRIOR_CLAIM_WITHOUT_UNIQUE_TERMINAL")


def checkpoint_scope(stage_path, stage, recipe_path, recipe, phase):
    _root, _meta, _request, queue_sha = queue.load_queue(Path(stage["queue_root"]))
    return {"schema": "kws-context-v3-checkpoint-scope-v1", "phase": phase,
            "stage": {"path": str(stage_path), **common.record(stage_path)},
            "queue_sha256": queue_sha, "job_id": recipe["job_id"],
            "recipe_sha256": common.sha_file(recipe_path), "cell_ids": recipe["selected_cells"],
            "max_new_calls": recipe["max_calls"], "automatic_retries": 0,
            "resume_pending_intent_after_restart": False}


def verify_generation_intent(stage_path, stage, recipe_path, recipe):
    import research_d2_checkpoints as checkpoints
    folder = Path(stage["prepared"]) / "checkpoints" / recipe["job_id"] / "before-generation"
    require(os.environ.get(INTENT_FOLDER_ENV) == str(folder) and os.environ.get(INTENT_OWNER_ENV),
            "CONTEXT_PERSISTED_LAUNCH_INTENT_REQUIRED")
    descriptor = checkpoints.verify_capsule(folder, current_sources=True)
    checkpoints.verify_ack(folder)
    expected = {**checkpoint_scope(stage_path, stage, recipe_path, recipe, "before-generation"),
                "owner_invocation": os.environ[INTENT_OWNER_ENV], "launch_intent": True}
    require(descriptor["context"] == expected, "CONTEXT_LAUNCH_INTENT_SCOPE_CHANGED")


def checkpoint_sources(stage, prepared, job_id):
    """One chunk's native evidence plus static inputs and durable chain links."""
    import research_d2_checkpoints as checkpoints
    files = [prepared / name for name in ("stage.json", "plan.json", "source-manifest.json",
             "model-assets.json", "runtime-setup.json", "queue-request.json")]
    for name in ("producer-source", "execution-source", "reference-assets", "recipes"):
        path = prepared / name
        if path.exists():
            files.extend(checkpoints.regular_tree(path))
    qroot = Path(stage["queue_root"])
    files.extend(qroot / name for name in ("queue.json", "request.json", "active-plan.json"))
    files.extend(checkpoints.regular_tree(qroot / "recipes"))
    recipe = common.read(qroot / "recipes" / (job_id + ".json"))
    validate_job_recipe(recipe, stage, prepared / "stage.json")
    files.extend(checkpoints.regular_tree(qroot / "jobs" / job_id))
    generation_root = Path(stage["generation_root"])
    if (generation_root / "plan-lock.json").exists():
        files.append(generation_root / "plan-lock.json")
    for ident in recipe["selected_cells"]:
        path = generation_root / "cells" / ident
        if path.exists():
            files.extend(checkpoints.regular_tree(path))
    registration = Path(stage["runtime"]) / "context-v3-campaigns" / (campaign_identity(common.read(prepared / "plan.json")) + ".json")
    files.append(registration)
    # The completion capsule includes its preceding intent's descriptor, actual
    # acknowledgement and provider response. Earlier chunk ZIPs remain separate
    # immutable objects: restoring the whole campaign requires that chain.
    for ack in sorted((prepared / "checkpoints").glob("*/*/ack.json")):
        files.extend((ack, ack.parent / "provider-save.json", ack.parent / "capsule.json", ack.parent / "phase-marker.json"))
    return files


def persist_terminal(gate, stage_path, stage, prepared, qroot, job, queue_sha):
    result = queue.result_for(qroot, job, queue_sha)
    require(result["status"] not in ("pending", "running"), "CONTEXT_JOB_NOT_TERMINAL")
    recipe_path = qroot / "recipes" / (job["job_id"] + ".json")
    recipe = common.read(recipe_path)
    validate_job_recipe(recipe, stage, stage_path)
    phase = "generation-complete" if result["status"] == "completed" else "generation-terminal"
    scope = checkpoint_scope(stage_path, stage, recipe_path, recipe, phase)
    scope.update(queue_result_status=result["status"],
                 queue_result={"path": str(qroot / "jobs" / job["job_id"] / "result.json"),
                               **common.record(qroot / "jobs" / job["job_id"] / "result.json")})
    gate.checkpoint(prepared / "checkpoints" / job["job_id"] / phase, scope,
                    checkpoint_sources(stage, prepared, job["job_id"]), intent=False)
    return result


def guarded_next(args):
    """Persist intent, run at most one declared job, then persist its terminal."""
    import research_d2_checkpoints as checkpoints
    stage_path = args.stage.absolute()
    stage, prepared, plan, _bulk = load_stage(stage_path)
    announced = set()
    def on_wait(folder, descriptor):
        if str(folder) not in announced:
            print(json.dumps({"status": "awaiting_durable_save", "checkpoint": str(folder),
                              "capsule": descriptor["capsule"]}, sort_keys=True), flush=True)
            announced.add(str(folder))
    gate = checkpoints.Gate(wait_seconds=1, max_wait_seconds=args.checkpoint_wait_seconds, on_wait=on_wait)
    # The maintained campaign uses this same lock during declaration and run.
    # Once it owns the stage, standalone generation must not skip its required
    # ASR/weak-view/chunk-save ordering between generation jobs.
    with queue.exclusive(prepared / "campaign-driver.lock"):
        require(not (prepared / "campaign-driver.json").exists(),
                "CONTEXT_CAMPAIGN_OWNS_STAGE_USE_CAMPAIGN_DRIVER")
        qroot, _meta, request, queue_sha = queue.load_queue(Path(stage["queue_root"]))
        status = queue.inspect(qroot, reconcile=True)
        require(not any(row["status"] == "running" for row in status["jobs"].values()),
                "CONTEXT_RUNNING_QUEUE_REQUIRES_WAIT")
        # A process may have ended after an authentic terminal but before the
        # save acknowledgement. Seal that evidence before any new launch.
        for prior in request["jobs"]:
            if status["jobs"][prior["job_id"]]["status"] not in ("pending", "running"):
                persist_terminal(gate, stage_path, stage, prepared, qroot, prior, queue_sha)
        require(not status["unresolved_jobs"], "CONTEXT_UNRESOLVED_QUEUE_REQUIRES_RECONCILIATION")
        if status["status"] == "completed":
            return {"status": "completed", "new_jobs": 0, "durable_terminal_saved": True}
        ready = [job for job in request["jobs"] if status["jobs"][job["job_id"]]["status"] == "pending" and
                 all(status["jobs"][dep]["status"] == "completed" for dep in job["depends_on"])]
        require(bool(ready), "CONTEXT_NO_NEW_READY_JOB")
        job = ready[0]
        recipe_path = qroot / "recipes" / (job["job_id"] + ".json")
        recipe = common.read(recipe_path)
        validate_job_recipe(recipe, stage, stage_path)
        check_prefix(plan, Path(stage["generation_root"]), recipe["selected_cells"])
        folder = prepared / "checkpoints" / job["job_id"] / "before-generation"
        gate.checkpoint(folder, checkpoint_scope(stage_path, stage, recipe_path, recipe, "before-generation"),
                        checkpoint_sources(stage, prepared, job["job_id"]), intent=True)
        old = {key: os.environ.get(key) for key in (INTENT_FOLDER_ENV, INTENT_OWNER_ENV)}
        os.environ.update({INTENT_FOLDER_ENV: str(folder), INTENT_OWNER_ENV: gate.owner})
        try:
            while True:
                report, code = queue.run(qroot, max_jobs=1)
                if code != 75 or report["status"] != "waiting_for_heavy_model_slot":
                    break
                time.sleep(1)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        result = persist_terminal(gate, stage_path, stage, prepared, qroot, job, queue_sha)
        return {"status": result["status"], "job_id": job["job_id"], "new_jobs": 1,
                "queue_code": code, "new_asr_calls": 0, "durable_terminal_saved": True}


class WorkerDeadline(BaseException):
    """Not caught by the per-cell quality-failure handler; no subsequent calls."""


def run(args):
    previous = signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "CONTEXT_NESTED_WORKER_TIMER")
    def timeout(_signum, _frame):
        raise WorkerDeadline("CONTEXT_WORKER_570_SECOND_BUDGET_EXHAUSTED")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WORKER_SECONDS)
    try:
        return _run_model(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_model(args):
    recipe = common.read(args.recipe)
    require(recipe["schema"] == JOB_SCHEMA and
            common.record(recipe["stage"]["path"]) == {k: recipe["stage"][k] for k in ("bytes", "sha256")},
            "CONTEXT_JOB_STAGE_CHANGED")
    require(os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == common.sha_file(args.recipe),
            "CONTEXT_QUEUE_RECIPE_BINDING")
    stage, prepared, plan, bulk = load_stage(recipe["stage"]["path"])
    validate_job_recipe(recipe, stage, recipe["stage"]["path"])
    verify_generation_intent(recipe["stage"]["path"], stage, args.recipe, recipe)
    require(str(args.heavy_lock.absolute()) == stage["heavy_lock"], "CONTEXT_HEAVY_LOCK_CHANGED")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    generation_root = Path(stage["generation_root"])
    started = time.monotonic()
    with heavy_model_slot(args.heavy_lock) as lease:
        require(lease.supervised and lease.inherited, "CONTEXT_DURABLE_QUEUE_REQUIRED")
        check_prefix(plan, generation_root, recipe["selected_cells"])
        manifest = research_runtime.load_manifest()
        research_runtime.check_interpreter(manifest)
        research_runtime.check_packages(manifest, "tts")
        setup = common.read(prepared / "runtime-setup.json")
        require(setup["status"] == "complete", "CONTEXT_TTS_RUNTIME_INCOMPLETE")
        for name, version in setup["installed_packages"].items():
            require(importlib.metadata.version(name) == version, "CONTEXT_RUNTIME_PACKAGE_CHANGED:" + name)
        common.once(output / "invocation.json", {"recipe_sha256": common.sha_file(args.recipe),
            "selected_cells": recipe["selected_cells"], "supervised": True,
            "heavy_lock": stage["heavy_lock"], "pid": os.getpid(), "max_calls": recipe["max_calls"]})
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "4"
        os.environ.update(HF_HOME=str(Path(stage["runtime"]) / "cache"))
        research_no_network.set_offline_environment()
        common.once(output / "network-guard.json", research_no_network.install())
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        model_root = Path(stage["runtime"]) / "models/base"
        bulk.asset_lock(model_root, common.read(prepared / "model-assets.json"), plan["model_revision"])
        delegate = bulk.QwenBulkClone(model_root, device="cpu")
        model = delegate._model()
        require(model.model.tts_model_type == "base", "CONTEXT_BASE_MODEL_REQUIRED")
        backend = common.BoundedBackend(delegate, output, recipe)
        result = bulk.run(plan, common.read(prepared / "source-manifest.json"), prepared / "reference-assets",
                          generation_root, backend, max_new_calls=recipe["max_calls"])
        require(result["attempts_this_call"] == len(recipe["selected_cells"]), "CONTEXT_ACTUAL_CLAIM_COUNT")
        rows = []
        for ident in recipe["selected_cells"]:
            folder, native = generation_root / "cells" / ident, output / "native" / ident
            completed, failed = (folder / "completed.json").exists(), (folder / "failed.json").exists()
            require(completed != failed, "CONTEXT_MISSING_OR_AMBIGUOUS_TERMINAL")
            returned = (native / "native.json").exists() and (native / "returned.json").exists()
            rows.append({"cell_id": ident, "generation_outcome": "completed" if completed else
                         ("failed-native" if returned else "failed-without-return"),
                         "native_returned": returned, "observed_text": None,
                         "training_admitted": False})
        files = {p.relative_to(output).as_posix(): common.record(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
        generation_files = {str(p.relative_to(generation_root)): common.record(p)
                            for ident in recipe["selected_cells"]
                            for p in sorted((generation_root / "cells" / ident).rglob("*")) if p.is_file()}
        common.once(output / "inventory.json", {"files": files, "generation_root": str(generation_root),
                                                 "generation_files": generation_files})
        summary = {"schema": "kws-context-v3-generation-result-v1", "status": "completed",
                   "recipe_sha256": common.sha_file(args.recipe), "job_id": recipe["job_id"],
                   "claims": len(rows), "native_generation_calls": backend.calls,
                   "native_returns": sum(row["native_returned"] for row in rows),
                   "outcomes": dict(Counter(row["generation_outcome"] for row in rows)), "rows": rows,
                   "prompt_creations": delegate.prompt_creations, "wall_seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "generation_root": str(generation_root), "asr_calls": 0, "human_gold": False}
        common.once(output / "summary.json", summary)
    return summary


def export_job(stage_path, job_id, output):
    stage, prepared, plan, _bulk = load_stage(stage_path)
    output = Path(output).absolute()
    require(not output.exists(), "CONTEXT_ASR_EXPORT_CREATE_ONLY")
    qroot, _qmeta, request, queue_sha = queue.load_queue(Path(stage["queue_root"]))
    definitions = [item for item in request["jobs"] if item["job_id"] == job_id]
    require(len(definitions) == 1, "CONTEXT_UNKNOWN_JOB")
    definition = definitions[0]
    result = queue.result_for(qroot, definition, queue_sha)
    require(result["status"] == "completed", "CONTEXT_COMPLETED_JOB_REQUIRED")
    recipe_path = qroot / "recipes" / (job_id + ".json")
    recipe = common.read(recipe_path)
    validate_job_recipe(recipe, stage, str(Path(stage_path).absolute()))
    job = qroot / "jobs" / job_id
    native_root = job / "outputs"
    summary = common.read(native_root / "summary.json")
    inventory = common.read(native_root / "inventory.json")
    require(summary.get("schema") == "kws-context-v3-generation-result-v1" and
            summary.get("status") == "completed" and summary.get("job_id") == job_id and
            summary.get("recipe_sha256") == definition["recipe_sha256"] and
            [row["cell_id"] for row in summary["rows"]] == recipe["selected_cells"] and
            summary["claims"] == recipe["max_calls"] and
            summary["generation_root"] == inventory["generation_root"] == stage["generation_root"],
            "CONTEXT_RESULT_SCOPE_CHANGED")
    check_records(native_root, inventory["files"])
    check_records(stage["generation_root"], inventory["generation_files"])
    buffer, included, excluded = io.BytesIO(), [], []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for row in summary["rows"]:
            ident = row["cell_id"]
            if not row["native_returned"]:
                excluded.append({"cell_id": ident, "reason": "no_native_return"})
                continue
            native = common.read(native_root / "native" / ident / "native.json")
            if not (native["all_finite"] and native["sample_rate_hz"] == 24000 and
                    len(native["shape"]) == 1 and 1 <= native["shape"][0] <= 720000):
                excluded.append({"cell_id": ident, "reason": "invalid_native_asr_geometry"})
                continue
            body = (native_root / "native" / ident / "returned-native.wav").read_bytes()
            archive.writestr(zipfile.ZipInfo("cells/" + ident + "/native-f32.wav"), body)
            included.append(row)
    output.mkdir(mode=0o700)
    common.once_bytes(output / "source-original.zip", buffer.getvalue())
    index = {"schema": "kws-context-v3-asr-export-v1", "stage": {"path": str(stage_path), **common.record(stage_path)},
             "job_id": job_id, "job_result": {"path": str(job / "result.json"), **common.record(job / "result.json")},
             "job_recipe": {"path": str(recipe_path), **common.record(recipe_path)},
             "plan": str(prepared / "plan.json"),
             "plan_sha256": hashlib.sha256(_bulk.p.canonical(plan)).hexdigest(),
             "plan_file_sha256": common.sha_file(prepared / "plan.json"),
             "generation_root": stage["generation_root"], "native_root": str(native_root / "native"),
             "archive": common.record(output / "source-original.zip"), "included": included,
             "excluded": excluded, "new_asr_calls": 0, "recognizer_context_text": None}
    common.once(output / "export.json", index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for field in ("plan", "sources", "audio-root", "data-root", "runtime", "output", "heavy-lock"):
        prep.add_argument("--" + field, type=Path, required=True)
    prep.add_argument("--active-plan", type=Path, default=ROOT / "configs/research/active-plan-20261105.json")
    prep.add_argument("--chunk-size", type=int, default=8)
    prep.add_argument("--global-minutes", type=float, required=True)
    prep.add_argument("--job-minutes", type=float, default=10)
    execute = sub.add_parser("run")
    for field in ("recipe", "output", "heavy-lock"):
        execute.add_argument("--" + field, type=Path, required=True)
    export = sub.add_parser("export-asr")
    export.add_argument("--stage", type=Path, required=True)
    export.add_argument("--job-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    guarded = sub.add_parser("guarded-next", help="persist and execute at most one declared generation chunk")
    guarded.add_argument("--stage", type=Path, required=True)
    guarded.add_argument("--checkpoint-wait-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    elif args.command == "guarded-next":
        result = guarded_next(args)
    else:
        result = export_job(args.stage, args.job_id, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
