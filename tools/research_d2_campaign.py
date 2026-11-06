#!/usr/bin/env python3
"""Complete one frozen D2 stage through its existing generation and ASR queues.

Each generated chunk is exported, independently transcribed and replayed into
a machine-weak view before the next chunk starts. No model is loaded here; the
unchanged bounded queues own all model claims, deadlines and the shared slot.
Completed artifacts are authenticated before reuse. Failed/unknown model jobs
are never retried, and an interrupted non-model output is never overwritten.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_batch as batch
import active_asr_selective as selective
import research_d2_generate as generation
import research_queue as queue
import research_queue_supervisor as supervisor
import research_d2_checkpoints as checkpoints
import research_d2_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCHEMA = "kws-d2-campaign-v1"
SCHEMA = "kws-d2-campaign-v2"
COMPLETION = "kws-d2-campaign-chunk-complete-v1"
ASR_JOB = "asr-selective"


def pin(path):
    path = queue.safe_path(path)
    return {"path": str(path), **queue.file_record(path)}


def check_pin(value):
    queue.require(pin(value["path"]) == value, "D2_CAMPAIGN_PIN_CHANGED:" + value["path"])


def read(path):
    return queue.loads(queue.read_bytes(path))


def driver_identity():
    """Fail before TTS if this process cannot perform the audio preparation."""
    batch.research_runtime.check_interpreter(batch.research_runtime.load_manifest())
    try:
        packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile")}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("D2_CAMPAIGN_DRIVER_AUDIO_PACKAGES_REQUIRED:" + str(error)) from error
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).absolute()), "packages": packages}


def load_data(root):
    root = queue.safe_path(root)
    sys.path.insert(0, str(root / "automation"))
    module = importlib.import_module("active_d2_weak_view")
    queue.require(Path(module.__file__).resolve() == root / "automation/active_d2_weak_view.py",
                  "D2_CAMPAIGN_DATA_IMPORT_CHANGED")
    # Generation first imports its frozen producer. Python may retain those
    # shared contract/pool modules when the weak-view adapter is then loaded.
    # Equal bytes are reusable; different cached producer code must not be
    # reported as the newly pinned data repository's executed implementation.
    module_paths = {
        "active_d2_weak_view": "automation/active_d2_weak_view.py",
        "active_asr_weak_view": "automation/active_asr_weak_view.py",
        "d2_pool": "research/auto-tts-voice-clone-20261102/d2_pool.py",
        "bulk_pool": "research/auto-tts-voice-clone-20261102/bulk_pool.py",
        "kws_data_contract": "kws_data_contract/__init__.py",
        "kws_data_contract.identity": "kws_data_contract/identity.py",
        "kws_data_contract.source_registry": "kws_data_contract/source_registry.py",
        "kws_data_contract.reference_text": "kws_data_contract/reference_text.py",
    }
    for name, relative in module_paths.items():
        loaded = sys.modules.get(name)
        queue.require(loaded is not None and queue.file_record(loaded.__file__) == queue.file_record(root / relative),
                      "D2_CAMPAIGN_CACHED_DATA_SOURCE_CHANGED:" + name)
    return module


def paths(root, job_id):
    chunk = Path(root) / "chunks" / job_id
    asr_queue = chunk / "asr-queue"
    return {"chunk": chunk, "export": chunk / "export", "inputs": chunk / "inputs",
            "source_map": chunk / "source-map.json", "asr_request": chunk / "asr-request.json",
            "asr_queue_request": chunk / "asr-queue-request.json", "asr_queue": asr_queue,
            "asr_output": asr_queue / "jobs" / ASR_JOB / "outputs/asr",
            "view": chunk / "weak-view", "complete": chunk / "complete.json"}


def registration_path(stage_path):
    return Path(stage_path).parent / "campaign-driver.json"


def driver_lock(stage_path):
    return Path(stage_path).parent / "campaign-driver.lock"


def declare(args):
    """Freeze the complete allowed prefix once, optionally adopting its files."""
    stage_path, root = queue.safe_path(args.stage), queue.safe_path(args.output)
    driver = driver_identity()
    stage, _prepared, plan, _bulk = generation.load_stage(stage_path)
    history_loss = [pin(path) for path in getattr(args, "history_loss", [])]
    recovery.guard_plan(plan, history_loss)
    qroot, qmeta, request, queue_sha = queue.load_queue(stage["queue_root"])
    queue.require(type(args.max_chunks) is int and 1 <= args.max_chunks <= len(stage["chunks"]),
                  "D2_CAMPAIGN_CHUNK_LIMIT")
    total_ns = queue.minutes_ns(args.total_minutes)
    asr_ns = queue.minutes_ns(args.asr_job_minutes, 999)
    queue.require(type(args.asr_budget_seconds) is int and 1 <= args.asr_budget_seconds <= 600 and
                  args.asr_budget_seconds * 1_000_000_000 < asr_ns and
                  type(args.threads) is int and 1 <= args.threads <= 4 and total_ns >= asr_ns,
                  "D2_CAMPAIGN_ASR_BUDGET")
    runtime = queue.safe_path(args.asr_runtime)
    queue.require(str(runtime / "heavy-model.lock") == stage["heavy_lock"] == qmeta["heavy_lock"],
                  "D2_CAMPAIGN_SHARED_MODEL_SLOT_REQUIRED")
    python = runtime / "venv/bin/python"
    queue.require(python.is_file() and os.access(python, os.X_OK), "D2_CAMPAIGN_ASR_PYTHON_MISSING")
    policy = queue.safe_path(args.policy)
    queue.require(read(policy) == selective.POLICY, "D2_CAMPAIGN_POLICY_CHANGED")
    data_root = queue.safe_path(args.data_root)
    data = load_data(data_root)
    own_names = set(generation.OWN_FILES) | set(selective.source_identity()) | {
        "tools/research_d2_campaign.py", "tools/research_queue_supervisor.py",
        "tools/research_d2_checkpoints.py", "tools/research_d2_recovery.py"}
    own = {name: queue.file_record(ROOT / name) for name in sorted(own_names)}
    producer = {name: queue.file_record(data_root / name) for name in data.PRODUCER_FILES}
    value = {"schema": SCHEMA, "created_at": queue.utc(), "root": str(root), "stage": pin(stage_path),
             "generation_queue": {"path": str(qroot), "queue_sha256": queue_sha},
             "data_root": str(data_root), "data_files": producer, "pipeline_files": own,
             "driver_runtime": driver,
             "asr_runtime": str(runtime), "asr_python": str(python), "policy": pin(policy),
             "fsmn_tokens": pin(args.fsmn_tokens) if args.fsmn_tokens else None,
             "job_ids": [job["job_id"] for job in request["jobs"][:args.max_chunks]],
             "max_chunks": args.max_chunks, "total_budget_minutes": args.total_minutes,
             "asr_job_minutes": args.asr_job_minutes, "asr_budget_seconds": args.asr_budget_seconds,
             "threads": args.threads, "model_concurrency": 1, "automatic_retries": 0,
             "d1_reissued_cells": 0, "human_gold": False,
             "durable_checkpoints": {"schema": checkpoints.SCHEMA, "required": True,
                                     "intent_owner": "one-live-driver-invocation", "history_loss": history_loss}}
    with queue.exclusive(driver_lock(stage_path)):
        registration = registration_path(stage_path)
        queue.require(not registration.exists(), "D2_STAGE_DRIVER_ALREADY_REGISTERED")
        root.mkdir(mode=0o700, parents=False, exist_ok=True)
        queue.require({p.name for p in root.iterdir()} <= {"chunks"}, "D2_NEW_CAMPAIGN_DECLARATION_REQUIRED")
        if (root / "chunks").exists():
            queue.require({p.name for p in (root / "chunks").iterdir()} <= set(value["job_ids"]),
                          "D2_ADOPTED_CHUNK_OUTSIDE_SCOPE")
        queue.write_json(root / "campaign.json", value)
        queue.write_json(registration, {"schema": SCHEMA, "stage": value["stage"],
                                      "campaign": pin(root / "campaign.json")})
    return value


def load(path):
    path = queue.safe_path(path)
    value = read(path)
    queue.require(value["schema"] in (LEGACY_SCHEMA, SCHEMA) and value["model_concurrency"] == 1 and
                  value["automatic_retries"] == value["d1_reissued_cells"] == 0 and
                  value["human_gold"] is False and path == Path(value["root"]) / "campaign.json",
                  "D2_CAMPAIGN_POLICY_CHANGED")
    check_pin(value["stage"])
    registration = read(registration_path(value["stage"]["path"]))
    queue.require(registration == {"schema": value["schema"], "stage": value["stage"], "campaign": pin(path)},
                  "D2_CAMPAIGN_REGISTRATION_CHANGED")
    queue.require(driver_identity() == value["driver_runtime"], "D2_CAMPAIGN_DRIVER_RUNTIME_CHANGED")
    for key, root in (("pipeline_files", ROOT), ("data_files", Path(value["data_root"]))):
        for name, expected in value[key].items():
            queue.require(queue.file_record(root / queue.relative(name)) == expected,
                          "D2_CAMPAIGN_SOURCE_CHANGED:" + name)
    check_pin(value["policy"])
    if value["fsmn_tokens"] is not None:
        check_pin(value["fsmn_tokens"])
    stage, _prepared, plan, _bulk = generation.load_stage(value["stage"]["path"])
    if value["schema"] == SCHEMA:
        policy = value["durable_checkpoints"]
        queue.require(policy == {"schema": checkpoints.SCHEMA, "required": True,
                                 "intent_owner": "one-live-driver-invocation", "history_loss": policy["history_loss"]},
                      "D2_DURABLE_CHECKPOINT_POLICY_CHANGED")
        recovery.guard_plan(plan, policy["history_loss"])
    qroot, qmeta, request, queue_sha = queue.load_queue(stage["queue_root"])
    queue.require(value["generation_queue"] == {"path": str(qroot), "queue_sha256": queue_sha} and
                  value["job_ids"] == [j["job_id"] for j in request["jobs"][:value["max_chunks"]]] and
                  1 <= value["max_chunks"] <= len(stage["chunks"]) and
                  qmeta["heavy_lock"] == str(Path(value["asr_runtime"]) / "heavy-model.lock"),
                  "D2_CAMPAIGN_GENERATION_SCOPE_CHANGED")
    return value, stage, request


def asr_queue_definition(campaign, job_id):
    where = paths(campaign["root"], job_id)
    recipe = where["asr_request"]
    return {"schema": queue.SCHEMA, "queue_id": "d2-" + job_id + "-asr",
            "global_budget_minutes": campaign["asr_job_minutes"], "jobs": [{
                "job_id": ASR_JOB, "recipe": str(recipe), "recipe_sha256": queue.file_record(recipe)["sha256"],
                "argv": [campaign["asr_python"], "-B", str(ROOT / "tools/active_asr_selective.py"),
                         "run", "--request", "{recipe}"], "cwd": str(ROOT), "depends_on": [],
                "timeout_minutes": campaign["asr_job_minutes"],
                "outputs": ["asr/audit/final-summary.json", "asr/audit/file-hashes.json"]}]}


def check_asr_queue(campaign, job_id):
    where = paths(campaign["root"], job_id)
    qroot, qmeta, request, _sha = queue.load_queue(where["asr_queue"])
    queue.require(request == asr_queue_definition(campaign, job_id) and
                  qmeta["heavy_lock"] == str(Path(campaign["asr_runtime"]) / "heavy-model.lock"),
                  "D2_ASR_QUEUE_SCOPE_CHANGED")
    return queue.inspect(qroot)


def budget(campaign):
    """Count persisted actual time, including adopted jobs, with no estimates."""
    generated = queue.inspect(campaign["generation_queue"]["path"])
    selected = set(campaign["job_ids"])
    queue.require(all(state["status"] == "pending" for name, state in generated["jobs"].items()
                      if name not in selected), "D2_GENERATION_OUTSIDE_CAMPAIGN_PREFIX")
    records = [("generation/" + name, state) for name, state in generated["jobs"].items() if name in selected]
    asr = {}
    for job_id in campaign["job_ids"]:
        root = paths(campaign["root"], job_id)["asr_queue"]
        if root.exists():
            report = check_asr_queue(campaign, job_id)
            asr[job_id] = report
            records += [(job_id + "/asr/" + name, state) for name, state in report["jobs"].items()]
    known = sum(state.get("actual_elapsed_ns") or 0 for _, state in records)
    unresolved = [name for name, state in records if state["status"] in ("unknown", "running")]
    failures = [name for name, state in records if state["status"] in ("failed", "timed_out", "unknown")]
    total = queue.minutes_ns(campaign["total_budget_minutes"])
    return {"actual_known_job_nanoseconds": known, "actual_known_job_minutes": known / queue.NS_PER_MINUTE,
            "remaining_nanoseconds": max(0, total - known), "overrun_nanoseconds": max(0, known - total),
            "actual_total_is_complete": not unresolved, "unresolved_jobs": unresolved, "failed_jobs": failures,
            "generation": generated, "asr": asr}


def reserve(report, required_ns):
    queue.require(not report["failed_jobs"] and not report["unresolved_jobs"], "D2_CAMPAIGN_UNRESOLVED_JOB")
    return required_ns <= report["remaining_nanoseconds"] and not report["overrun_nanoseconds"]


def source_options(campaign, exported):
    return {"plan_path": exported["plan"], "plan_sha256": exported["plan_sha256"],
            "generation_root": exported["generation_root"], "native_root": exported["native_root"],
            "job_recipe_path": exported["job_recipe"]["path"],
            "job_recipe_sha256": exported["job_recipe"]["sha256"], "stage_path": campaign["stage"]["path"]}


def ensure_export(campaign, job_id):
    target = paths(campaign["root"], job_id)["export"]
    if not target.exists():
        return generation.export_job(campaign["stage"]["path"], job_id, target)
    queue.require((target / "export.json").is_file(), "D2_PARTIAL_EXPORT_NOT_RETRIED")
    # The exporter revalidates queue receipts and all native inventories. Its
    # deterministic replay writes only to a temporary directory, never source.
    with tempfile.TemporaryDirectory(prefix="d2-export-check-") as folder:
        expected = generation.export_job(campaign["stage"]["path"], job_id, Path(folder) / "export")
        queue.require(read(target / "export.json") == expected and
                      queue.file_record(target / "source-original.zip") == expected["archive"],
                      "D2_EXISTING_EXPORT_CHANGED")
    return expected


def ensure_inputs(campaign, job_id, exported):
    where = paths(campaign["root"], job_id)
    manifest_path = where["inputs"] / "inputs.json"
    count = len(exported["included"])
    if not where["inputs"].exists():
        batch.prepare(Namespace(archive=where["export"] / "source-original.zip",
                                source_sha256=exported["archive"]["sha256"], expected_count=count,
                                output=where["inputs"]))
    queue.require(manifest_path.is_file(), "D2_PARTIAL_INPUTS_NOT_RETRIED")
    manifest = batch.verify_inputs(manifest_path)
    queue.require(manifest["source_archive_sha256"] == exported["archive"]["sha256"] and
                  len(manifest["clips"]) == count, "D2_EXISTING_INPUTS_CHANGED")
    return manifest_path


def ensure_source_map(campaign, job_id, exported, inputs, data):
    where = paths(campaign["root"], job_id)
    expected = data.build_source_map(inputs, queue.file_record(inputs)["sha256"],
        where["export"] / "source-original.zip", **source_options(campaign, exported))
    if where["source_map"].exists():
        queue.require(read(where["source_map"]) == expected, "D2_EXISTING_SOURCE_MAP_CHANGED")
    else:
        queue.write_json(where["source_map"], expected)
    return where["source_map"]


def ensure_asr_request(campaign, job_id, inputs, source_map):
    where = paths(campaign["root"], job_id)
    if not where["asr_request"].exists():
        selective.declare(Namespace(runtime=Path(campaign["asr_runtime"]), inputs=inputs,
            source_map=source_map, output=where["asr_output"], request=where["asr_request"],
            policy=Path(campaign["policy"]["path"]), threads=campaign["threads"],
            budget_seconds=campaign["asr_budget_seconds"]))
    request, _, _, _ = selective.verify_request(where["asr_request"])
    queue.require(request["runtime"] == campaign["asr_runtime"] and request["output"] == str(where["asr_output"]) and
                  request["inputs"]["path"] == str(inputs) and request["source_map"]["path"] == str(source_map) and
                  request["policy"]["path"] == campaign["policy"]["path"] and
                  request["threads"] == campaign["threads"] and request["budget_seconds"] == campaign["asr_budget_seconds"],
                  "D2_EXISTING_ASR_REQUEST_CHANGED")
    return request


def ensure_asr_queue(campaign, job_id):
    where = paths(campaign["root"], job_id)
    expected = asr_queue_definition(campaign, job_id)
    if where["asr_queue_request"].exists():
        queue.require(read(where["asr_queue_request"]) == expected, "D2_EXISTING_ASR_QUEUE_REQUEST_CHANGED")
    else:
        queue.write_json(where["asr_queue_request"], expected)
    if not where["asr_queue"].exists():
        queue.create(where["asr_queue"], where["asr_queue_request"],
                     Path(campaign["generation_queue"]["path"]) / "active-plan.json",
                     Path(campaign["asr_runtime"]) / "heavy-model.lock")
    return check_asr_queue(campaign, job_id)


def ensure_view(campaign, job_id, exported, inputs, data):
    where = paths(campaign["root"], job_id)
    final = where["asr_output"] / "audit/final-summary.json"
    audit = selective.seal_or_verify(where["asr_request"])
    queue.require(audit["state"]["status"] == "complete", "D2_ASR_AUDIT_INCOMPLETE")
    if not where["view"].exists():
        data.create_view(inputs, queue.file_record(inputs)["sha256"], where["export"] / "source-original.zip",
            where["asr_output"], where["view"],
            fsmn_tokens=campaign["fsmn_tokens"]["path"] if campaign["fsmn_tokens"] else None,
            selective_source_map=where["source_map"], selective_policy=campaign["policy"]["path"],
            selective_audit_sha256=queue.file_record(final)["sha256"], **source_options(campaign, exported))
    report = data.verify_view(where["view"])
    owner = report["source_adapter"]["job_owner"]
    queue.require(report["inputs_sha256"] == queue.file_record(inputs)["sha256"] and
                  report["source_archive_sha256"] == exported["archive"]["sha256"] and
                  report["asr_adapter"]["final_audit_sha256"] == queue.file_record(final)["sha256"] and
                  report["source_adapter"]["plan_sha256"] == exported["plan_sha256"] and
                  owner["job_recipe_sha256"] == exported["job_recipe"]["sha256"] and
                  owner["stage_sha256"] == campaign["stage"]["sha256"] and owner["job_id"] == job_id,
                  "D2_EXISTING_WEAK_VIEW_SCOPE_CHANGED")
    return report


def seal_chunk(campaign, job_id, exported, report):
    where = paths(campaign["root"], job_id)
    names = [where["export"] / "export.json", where["export"] / "source-original.zip"]
    if report is not None:
        names += [where["inputs"] / "inputs.json", where["source_map"], where["asr_request"],
                  where["asr_queue"] / "queue.json", where["asr_queue"] / "jobs" / ASR_JOB / "result.json",
                  where["asr_output"] / "audit/final-summary.json", where["view"] / "MANIFEST.json"]
    value = {"schema": COMPLETION, "campaign": pin(Path(campaign["root"]) / "campaign.json"),
             "job_id": job_id, "status": "completed" if report is not None else "completed_no_asr_eligible_return",
             "files": {str(path.relative_to(where["chunk"])): queue.file_record(path) for path in names},
             "native_included": len(exported["included"]), "native_excluded": exported["excluded"],
             "statistics": report["statistics"] if report is not None else None,
             "human_gold": False, "automatic_retries": 0}
    if where["complete"].exists():
        queue.require(read(where["complete"]) == value, "D2_CHUNK_COMPLETION_CHANGED")
    else:
        queue.write_json(where["complete"], value)
    return value


def check_chunk_receipt(campaign, job_id):
    where = paths(campaign["root"], job_id)
    value = read(where["complete"])
    queue.require(value["schema"] == COMPLETION and value["job_id"] == job_id and
                  value["campaign"] == pin(Path(campaign["root"]) / "campaign.json") and
                  value["status"] in ("completed", "completed_no_asr_eligible_return") and
                  value["human_gold"] is False and value["automatic_retries"] == 0,
                  "D2_CHUNK_RECEIPT_SCOPE_CHANGED")
    for name, expected in value["files"].items():
        queue.require(queue.file_record(where["chunk"] / queue.relative(name)) == expected,
                      "D2_CHUNK_RECEIPT_FILE_CHANGED:" + name)
    return value


def compact_budget(report):
    return {key: value for key, value in report.items() if key not in ("generation", "asr")}


def checkpoint_sources(campaign, stage, job_id, phase):
    """Capture static dependencies and the exact phase, never future job output."""
    prepared = Path(stage["prepared"])
    names = [Path(campaign["root"]) / "campaign.json", Path(campaign["policy"]["path"])]
    if campaign["fsmn_tokens"] is not None:
        names.append(Path(campaign["fsmn_tokens"]["path"]))
    names += [Path(row["path"]) for row in campaign["durable_checkpoints"]["history_loss"]]
    for child in sorted(prepared.iterdir()):
        if child.name not in ("queue", "generation", "campaign-driver.lock") and not child.name.endswith(".lock"):
            names += checkpoints.regular_tree(child)
    for key, root in (("pipeline_files", ROOT), ("data_files", Path(campaign["data_root"]))):
        names += [root / queue.relative(name) for name in campaign[key]]
    qroot = Path(campaign["generation_queue"]["path"])
    names += [qroot / name for name in ("queue.json", "request.json", "active-plan.json")]
    names.append(qroot / "recipes" / (job_id + ".json"))
    if phase != "before-generation":
        names += checkpoints.regular_tree(qroot / "jobs" / job_id)
        generation_root = Path(stage.get("generation_root", prepared / "generation"))
        for cell in stage["chunks"][campaign["job_ids"].index(job_id)]:
            directory = generation_root / "cells" / cell
            if directory.exists():
                names += checkpoints.regular_tree(directory)
        for child in generation_root.glob("*"):
            if child.is_file() and not child.name.endswith(".lock"):
                names.append(child)
        names += checkpoints.regular_tree(paths(campaign["root"], job_id)["chunk"])
    return names


def advance(campaign_path, *, max_chunks_this_call=None, progress=None, checkpoint_gate=None):
    """Advance an explicit prefix; completed earlier chunks do not use the cap."""
    campaign, stage, request = load(campaign_path)
    queue.require(campaign["schema"] == SCHEMA and checkpoint_gate is not None,
                  "D2_DURABLE_HANDOFF_REQUIRED_FOR_EXECUTION")
    queue.require(max_chunks_this_call is None or type(max_chunks_this_call) is int and max_chunks_this_call > 0,
                  "D2_POSITIVE_CHUNKS_PER_INVOCATION_REQUIRED")
    data = load_data(campaign["data_root"])
    done_this_call, completed = 0, []
    for job_id in campaign["job_ids"]:
        where = paths(campaign["root"], job_id)
        already_sealed = where["complete"].exists()
        if not already_sealed and max_chunks_this_call is not None and done_this_call >= max_chunks_this_call:
            return {"status": "paused", "stop_reason": "max_chunks_this_call", "completed": completed,
                    "new_completed_chunks": done_this_call, "budget": compact_budget(budget(campaign))}, 0
        report = budget(campaign)
        if report["failed_jobs"]:
            return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(report)}, 1
        if report["unresolved_jobs"]:
            return {"status": "waiting_for_worker", "job_id": job_id, "budget": compact_budget(report)}, 75
        if report["overrun_nanoseconds"]:
            return {"status": "budget_exhausted", "budget": compact_budget(report)}, 3
        state = report["generation"]["jobs"][job_id]
        if progress:
            progress(job_id, "generation")
        if state["status"] == "pending":
            definition = next(item for item in request["jobs"] if item["job_id"] == job_id)
            allowance = min(queue.minutes_ns(definition["timeout_minutes"], 999),
                            report["generation"]["remaining_budget_nanoseconds"])
            # Reserve a full downstream ASR job before consuming new TTS data.
            if allowance <= 0 or not reserve(report, allowance + queue.minutes_ns(campaign["asr_job_minutes"], 999)):
                return {"status": "budget_exhausted", "job_id": job_id, "budget": compact_budget(report)}, 3
            checkpoint_gate.barrier(campaign, stage, job_id, "before-generation",
                                    checkpoint_sources(campaign, stage, job_id, "before-generation"), progress=progress)
            generated, code = queue.run(campaign["generation_queue"]["path"], max_jobs=1)
            if code:
                return {"status": generated["status"], "job_id": job_id,
                        "budget": compact_budget(budget(campaign))}, code
            queue.require(generated["jobs"][job_id]["status"] == "completed", "D2_UNEXPECTED_GENERATION_JOB")
        where["chunk"].mkdir(mode=0o700, parents=True, exist_ok=True)
        if progress:
            progress(job_id, "export_and_source_map")
        exported = ensure_export(campaign, job_id)
        checkpoint_gate.barrier(campaign, stage, job_id, "generation-complete",
                                checkpoint_sources(campaign, stage, job_id, "generation-complete"), progress=progress)
        view = None
        if exported["included"]:
            inputs = ensure_inputs(campaign, job_id, exported)
            mapping = ensure_source_map(campaign, job_id, exported, inputs, data)
            ensure_asr_request(campaign, job_id, inputs, mapping)
            asr = ensure_asr_queue(campaign, job_id)
            if progress:
                progress(job_id, "asr")
            if asr["jobs"][ASR_JOB]["status"] != "completed":
                current = budget(campaign)
                if current["failed_jobs"]:
                    return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(current)}, 1
                if current["unresolved_jobs"]:
                    return {"status": "waiting_for_worker", "job_id": job_id, "budget": compact_budget(current)}, 75
                if not reserve(current, queue.minutes_ns(campaign["asr_job_minutes"], 999)):
                    return {"status": "budget_exhausted", "job_id": job_id, "budget": compact_budget(current)}, 3
                checkpoint_gate.barrier(campaign, stage, job_id, "before-asr",
                                        checkpoint_sources(campaign, stage, job_id, "before-asr"), progress=progress)
                asr, code = queue.run(where["asr_queue"], max_jobs=1)
                if code:
                    return {"status": asr["status"], "job_id": job_id, "budget": compact_budget(budget(campaign))}, code
            if progress:
                progress(job_id, "verify_weak_view")
            view = ensure_view(campaign, job_id, exported, inputs, data)
        elif any(where[name].exists() for name in ("inputs", "source_map", "asr_request", "asr_queue", "view")):
            raise ValueError("D2_EMPTY_EXPORT_HAS_UNEXPECTED_ASR_ARTIFACTS")
        seal_chunk(campaign, job_id, exported, view)
        checkpoint_gate.barrier(campaign, stage, job_id, "chunk-complete",
                                checkpoint_sources(campaign, stage, job_id, "chunk-complete"), progress=progress)
        completed.append(job_id)
        done_this_call += not already_sealed
    final_budget = budget(campaign)
    exceeded = bool(final_budget["overrun_nanoseconds"])
    return {"status": "completed_budget_overrun" if exceeded else "completed", "completed": completed,
            "new_completed_chunks": done_this_call, "budget": compact_budget(final_budget)}, 3 if exceeded else 0


def run(campaign_path, *, max_chunks_this_call=None, wait_seconds=1.0, checkpoint_gate=None):
    campaign, _stage, _request = load(campaign_path)
    queue.require(campaign["schema"] == SCHEMA, "D2_DURABLE_HANDOFF_REQUIRED_FOR_EXECUTION")
    if checkpoint_gate is None:
        checkpoint_gate = checkpoints.Gate()
    queue.require(type(wait_seconds) in (int, float) and .02 <= wait_seconds <= 30,
                  "D2_BOUNDED_WAIT_INTERVAL_REQUIRED")
    root = Path(campaign["root"])
    with queue.exclusive(driver_lock(campaign["stage"]["path"])):
        queue.require(max_chunks_this_call is None or type(max_chunks_this_call) is int and max_chunks_this_call > 0,
                      "D2_POSITIVE_CHUNKS_PER_INVOCATION_REQUIRED")
        sealed_before = {job_id for job_id in campaign["job_ids"]
                         if paths(root, job_id)["complete"].exists()}
        stopped, state_lock = threading.Event(), threading.Lock()
        state = {"schema": SCHEMA, "campaign": pin(campaign_path), "pid": os.getpid(),
                 "started_at": queue.utc(), "state": "starting", "job_id": None}
        def publish(job_id, phase):
            with state_lock:
                state.update(job_id=job_id, state=phase)
        def heartbeat():
            while not stopped.is_set():
                with state_lock:
                    value = {**state, "updated_at": queue.utc()}
                supervisor.replace_heartbeat(root / "heartbeat.json", value)
                stopped.wait(5)
        thread = threading.Thread(target=heartbeat, name="d2-campaign-heartbeat", daemon=True)
        thread.start()
        try:
            while True:
                limit = max_chunks_this_call
                if limit is not None:
                    sealed_now = {job_id for job_id in campaign["job_ids"]
                                  if paths(root, job_id)["complete"].exists()}
                    limit -= len(sealed_now - sealed_before)
                    if limit <= 0:
                        result = {"status": "paused", "stop_reason": "max_chunks_this_call",
                                  "completed": sorted(sealed_now), "budget": compact_budget(budget(campaign))}
                        publish(None, result["status"])
                        return result, 0
                result, code = advance(campaign_path, max_chunks_this_call=limit, progress=publish,
                                       checkpoint_gate=checkpoint_gate)
                publish(result.get("job_id"), result["status"])
                if code != 75:
                    return result, code
                # Do not repeatedly replay large completed evidence while a
                # different bounded owner still holds the model/controller.
                time.sleep(wait_seconds)
                while model_or_controller_active(campaign):
                    time.sleep(wait_seconds)
        except BaseException:
            publish(state.get("job_id"), "stopped_before_completion")
            raise
        finally:
            stopped.set()
            thread.join(timeout=6)
            with state_lock:
                supervisor.replace_heartbeat(root / "heartbeat.json", {**state, "updated_at": queue.utc()})


def model_or_controller_active(campaign):
    if supervisor.lock_active(Path(campaign["asr_runtime"]) / "heavy-model.lock"):
        return True
    roots = [Path(campaign["generation_queue"]["path"])]
    roots += [paths(campaign["root"], job_id)["asr_queue"] for job_id in campaign["job_ids"]]
    return any(supervisor.lock_active(root / "controller.lock") or
               any(supervisor.lock_active(folder / "job.lock") for folder in (root / "jobs").glob("*"))
               for root in roots if root.exists())


def status(campaign_path):
    campaign, _, _ = load(campaign_path)
    completed = []
    for job_id in campaign["job_ids"]:
        if paths(campaign["root"], job_id)["complete"].exists():
            check_chunk_receipt(campaign, job_id)
            completed.append(job_id)
    return {"schema": SCHEMA, "budget": compact_budget(budget(campaign)),
            "driver_active": supervisor.lock_active(driver_lock(campaign["stage"]["path"])),
            "max_chunks": campaign["max_chunks"], "completed_chunks": completed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("declare")
    for name in ("stage", "output", "data-root", "asr-runtime"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--policy", type=Path, default=selective.POLICY_PATH)
    prep.add_argument("--fsmn-tokens", type=Path)
    prep.add_argument("--max-chunks", type=int, required=True)
    prep.add_argument("--total-minutes", type=float, required=True)
    prep.add_argument("--asr-job-minutes", type=float, default=10)
    prep.add_argument("--asr-budget-seconds", type=int, default=570)
    prep.add_argument("--threads", type=int, default=4)
    prep.add_argument("--history-loss", type=Path, action="append", default=[],
                      help="Pin each known immutable runtime-loss exclusion before new model work.")
    for name in ("run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name == "run":
            command.add_argument("--max-chunks-this-call", type=int)
            command.add_argument("--wait-seconds", type=float, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "declare":
            result, code = declare(args), 0
        elif args.command == "run":
            result, code = run(args.campaign, max_chunks_this_call=args.max_chunks_this_call,
                               wait_seconds=args.wait_seconds)
        else:
            result, code = status(args.campaign), 0
    except BlockingIOError:
        result, code = {"status": "another_driver_or_controller_is_active"}, 75
    except (ValueError, KeyError, OSError, TypeError) as error:
        result, code = {"status": "invalid", "reason": str(error)}, 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
