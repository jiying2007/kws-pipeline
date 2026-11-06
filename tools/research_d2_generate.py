#!/usr/bin/env python3
"""Run immutable D2 increments through the existing bounded research queue.

Generation claims are never retried. Every native return precedes quality
screening; failed quality and missing returns remain distinct. The D1 plan is
only an ancestor and is never submitted again by this entry point.
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
SCHEMA = "kws-d2-generation-stage-v1"
JOB_SCHEMA = "kws-d2-generation-job-v1"
DATA_FILES = common.SOURCE_FILES + (str(common.BULK / "d2_pool.py"),)
OWN_FILES = ("tools/research_d2_generate.py", "tools/active_tts_smoke.py",
             "tools/research_queue.py", "tools/research_job_lock.py",
             "tools/research_runtime.py", "tools/research_no_network.py",
             "configs/runtime/research-runtime.json", "research/qwen6_tts/runtime-lock.json",
             "research/qwen6_tts/model-locks.json", "configs/runtime/qwen-base-tts-model-lock.json")


def require(ok, reason):
    common.require(ok, reason)


def source_records(root, names):
    return {name: common.record(Path(root) / name) for name in names}


def check_records(root, records):
    for name, expected in records.items():
        require(common.record(Path(root) / common.relative(name)) == expected,
                "D2_PINNED_FILE_CHANGED:" + name)


def partition(plan, chunk_size):
    require(type(chunk_size) is int and 1 <= chunk_size <= 16, "D2_CHUNK_BUDGET")
    ids = [cell["cell_id"] for cell in plan["cells"]]
    require(ids and len(ids) == len(set(ids)) and len(ids) <= 1536, "D2_CELL_ROSTER")
    require(plan["budgets"]["max_total_calls"] == len(ids), "D2_FULL_STAGE_CALL_BUDGET_REQUIRED")
    return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]


def campaign_identity(plan):
    """A changed serialization/reference cannot re-register the same campaign."""
    scope = {"parent_plan_sha256": plan["d1_parent"]["plan_sha256"],
             "stage": plan["stage"], "batch_id": plan["batch_id"]}
    return hashlib.sha256(common.canonical(scope)).hexdigest()


def validate_job_recipe(recipe, stage, stage_path):
    require(recipe.get("schema") == JOB_SCHEMA and
            recipe.get("stage") == {"path": str(stage_path), **common.record(stage_path)},
            "D2_JOB_STAGE_CHANGED")
    index = recipe.get("chunk_index")
    require(type(index) is int and 0 <= index < len(stage["chunks"]) and
            recipe.get("job_id") == f"generate-{index + 1:03d}" and
            recipe.get("selected_cells") == stage["chunks"][index] and
            recipe.get("max_calls") == len(stage["chunks"][index]) and
            recipe.get("generation") == stage["generation"], "D2_JOB_SCOPE")


def prepare(args):
    output = Path(args.output).absolute()
    require(not output.exists() and output.parent.is_dir(), "D2_CREATE_ONLY_PREPARATION")
    require(1 <= args.global_minutes <= 999 and 1 <= args.job_minutes <= 999,
            "D2_EXPLICIT_RUNNER_BUDGET")
    queue.validate_plan(common.read(args.active_plan))
    output.mkdir(mode=0o700)
    for name in DATA_FILES:
        common.once_bytes(output / "producer-source" / name, (args.data_root / name).read_bytes())
    bulk = common.load_bulk(output / "producer-source")
    plan = common.read(args.plan)
    require(plan.get("schema") == "kws-d2-increment-plan-v1", "D2_INCREMENT_REQUIRED")
    bulk.verify_plan(plan)
    manifest = common.read(args.sources)
    require(bulk.p.canonical(bulk.rebuild_plan(plan, manifest, args.audio_root)) ==
            bulk.p.canonical(plan), "D2_SOURCE_PLAN_CHANGED")
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
             "max_new_calls": len(plan["cells"]), "generation": common.GENERATION,
             "model_concurrency": 1, "automatic_retries": 0,
             "d1_cells_generated_again": 0, "human_gold": False}
    common.once(output / "stage.json", stage)
    jobs = []
    stage_pin = {"path": str(output / "stage.json"), **common.record(output / "stage.json")}
    for number, cells in enumerate(chunks):
        ident = f"generate-{number + 1:03d}"
        recipe_path = output / "recipes" / (ident + ".json")
        recipe = {"schema": JOB_SCHEMA, "stage": stage_pin, "job_id": ident,
                  "chunk_index": number, "selected_cells": cells,
                  "max_calls": len(cells), "generation": common.GENERATION}
        common.once(recipe_path, recipe)
        jobs.append({"job_id": ident, "recipe": str(recipe_path),
                     "recipe_sha256": common.sha_file(recipe_path),
                     "argv": [str(args.runtime / "venv/bin/python"), str(ROOT / "tools/research_d2_generate.py"),
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
    registrations = args.runtime / "d2-campaigns"
    registrations.mkdir(exist_ok=True, mode=0o700)
    with queue.exclusive(registrations / "registration.lock"):
        registration = registrations / (campaign_identity(plan) + ".json")
        require(not registration.exists(), "D2_CAMPAIGN_ALREADY_REGISTERED")
        state = queue.create(output / "queue", output / "queue-request.json", args.active_plan, args.heavy_lock)
        common.once(registration, {"stage": stage_pin,
            "canonical_plan_sha256": hashlib.sha256(bulk.p.canonical(plan)).hexdigest(),
            "campaign_identity": campaign_identity(plan)})
    return {"status": "prepared", "stage": str(output / "stage.json"),
            "new_cells": stage["max_new_calls"], "jobs": len(jobs), "queue": state}


def load_stage(path):
    stage = common.read(path)
    require(stage["schema"] == SCHEMA and stage["model_concurrency"] == 1 and
            stage["automatic_retries"] == 0 and stage["d1_cells_generated_again"] == 0 and
            stage["generation"] == common.GENERATION, "D2_STAGE_POLICY")
    root = Path(stage["prepared"])
    for key, name in (("plan", "plan.json"), ("source_manifest", "source-manifest.json"),
                      ("model_assets", "model-assets.json"), ("runtime_setup", "runtime-setup.json")):
        require(common.record(root / name) == stage[key], "D2_STAGE_INPUT_CHANGED:" + name)
    check_records(root / "producer-source", stage["data_source"])
    require(set(stage["data_source"]) == set(DATA_FILES), "D2_DATA_SOURCE_ROSTER")
    check_records(ROOT, stage["execution_source"])
    require(set(stage["execution_source"]) == set(OWN_FILES), "D2_EXECUTION_SOURCE_ROSTER")
    plan = common.read(root / "plan.json")
    require([value for group in stage["chunks"] for value in group] ==
            [cell["cell_id"] for cell in plan["cells"]] and
            stage["max_new_calls"] == len(plan["cells"]) == plan["budgets"]["max_total_calls"] and
            all(1 <= len(group) <= 16 for group in stage["chunks"]), "D2_CHUNK_COVERAGE")
    bulk = common.load_bulk(root / "producer-source")
    bulk.verify_plan(plan)
    require(bulk.p.canonical(bulk.rebuild_plan(plan, common.read(root / "source-manifest.json"),
                root / "reference-assets")) == bulk.p.canonical(plan), "D2_SOURCE_PLAN_CHANGED")
    return stage, root, plan, bulk


def check_prefix(plan, generation_root, selected):
    ids = [cell["cell_id"] for cell in plan["cells"]]
    require(selected and selected == ids[ids.index(selected[0]):ids.index(selected[0]) + len(selected)],
            "D2_JOB_SELECTION_CHANGED")
    prior = ids[:ids.index(selected[0])]
    started = [ident for ident in ids if (generation_root / "cells" / ident / "started.json").exists()]
    require(started == prior, "D2_PREFIX_NOT_EXACT_OR_ALREADY_CLAIMED")
    for ident in prior:
        folder = generation_root / "cells" / ident
        require((folder / "completed.json").exists() != (folder / "failed.json").exists(),
                "D2_PRIOR_CLAIM_WITHOUT_UNIQUE_TERMINAL")


def run(args):
    recipe = common.read(args.recipe)
    require(recipe["schema"] == JOB_SCHEMA and
            common.record(recipe["stage"]["path"]) == {k: recipe["stage"][k] for k in ("bytes", "sha256")},
            "D2_JOB_STAGE_CHANGED")
    require(os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == common.sha_file(args.recipe),
            "D2_QUEUE_RECIPE_BINDING")
    stage, prepared, plan, bulk = load_stage(recipe["stage"]["path"])
    validate_job_recipe(recipe, stage, recipe["stage"]["path"])
    require(str(args.heavy_lock.absolute()) == stage["heavy_lock"], "D2_HEAVY_LOCK_CHANGED")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    generation_root = Path(stage["generation_root"])
    started = time.monotonic()
    with heavy_model_slot(args.heavy_lock) as lease:
        require(lease.supervised and lease.inherited, "D2_DURABLE_QUEUE_REQUIRED")
        check_prefix(plan, generation_root, recipe["selected_cells"])
        manifest = research_runtime.load_manifest()
        research_runtime.check_interpreter(manifest)
        research_runtime.check_packages(manifest, "tts")
        setup = common.read(prepared / "runtime-setup.json")
        require(setup["status"] == "complete", "D2_TTS_RUNTIME_INCOMPLETE")
        for name, version in setup["installed_packages"].items():
            require(importlib.metadata.version(name) == version, "D2_RUNTIME_PACKAGE_CHANGED:" + name)
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
        require(model.model.tts_model_type == "base", "D2_BASE_MODEL_REQUIRED")
        backend = common.BoundedBackend(delegate, output, recipe)
        result = bulk.run(plan, common.read(prepared / "source-manifest.json"), prepared / "reference-assets",
                          generation_root, backend, max_new_calls=recipe["max_calls"])
        require(result["attempts_this_call"] == len(recipe["selected_cells"]), "D2_ACTUAL_CLAIM_COUNT")
        rows = []
        for ident in recipe["selected_cells"]:
            folder, native = generation_root / "cells" / ident, output / "native" / ident
            completed, failed = (folder / "completed.json").exists(), (folder / "failed.json").exists()
            require(completed != failed, "D2_MISSING_OR_AMBIGUOUS_TERMINAL")
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
        summary = {"schema": "kws-d2-generation-result-v1", "status": "completed",
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
    require(not output.exists(), "D2_ASR_EXPORT_CREATE_ONLY")
    qroot, _qmeta, request, queue_sha = queue.load_queue(Path(stage["queue_root"]))
    definitions = [item for item in request["jobs"] if item["job_id"] == job_id]
    require(len(definitions) == 1, "D2_UNKNOWN_JOB")
    definition = definitions[0]
    result = queue.result_for(qroot, definition, queue_sha)
    require(result["status"] == "completed", "D2_COMPLETED_JOB_REQUIRED")
    recipe_path = qroot / "recipes" / (job_id + ".json")
    recipe = common.read(recipe_path)
    validate_job_recipe(recipe, stage, str(Path(stage_path).absolute()))
    job = qroot / "jobs" / job_id
    native_root = job / "outputs"
    summary = common.read(native_root / "summary.json")
    inventory = common.read(native_root / "inventory.json")
    require(summary.get("schema") == "kws-d2-generation-result-v1" and
            summary.get("status") == "completed" and summary.get("job_id") == job_id and
            summary.get("recipe_sha256") == definition["recipe_sha256"] and
            [row["cell_id"] for row in summary["rows"]] == recipe["selected_cells"] and
            summary["claims"] == recipe["max_calls"] and
            summary["generation_root"] == inventory["generation_root"] == stage["generation_root"],
            "D2_RESULT_SCOPE_CHANGED")
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
    index = {"schema": "kws-d2-asr-export-v1", "stage": {"path": str(stage_path), **common.record(stage_path)},
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
    prep.add_argument("--job-minutes", type=float, default=5)
    execute = sub.add_parser("run")
    for field in ("recipe", "output", "heavy-lock"):
        execute.add_argument("--" + field, type=Path, required=True)
    export = sub.add_parser("export-asr")
    export.add_argument("--stage", type=Path, required=True)
    export.add_argument("--job-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    else:
        result = export_job(args.stage, args.job_id, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
