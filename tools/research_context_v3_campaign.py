#!/usr/bin/env python3
"""Close all nine declared context-v3 chunks through generation, ASR and full2601.

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
import research_context_v3_generate as generation
import research_queue as queue
import research_queue_supervisor as supervisor
import research_d2_checkpoints as checkpoints
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import research_context_v3_inputs as projection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kws-context-v3-campaign-v1"
COMPLETION = "kws-context-v3-campaign-chunk-complete-v1"
ASR_JOB = "asr-selective"


def pin(path):
    path = queue.safe_path(path)
    return {"path": str(path), **queue.file_record(path)}


def check_pin(value):
    queue.require(pin(value["path"]) == value, "CONTEXT_V3_CAMPAIGN_PIN_CHANGED:" + value["path"])


def read(path):
    return queue.loads(queue.read_bytes(path))


def driver_identity():
    """Fail before TTS if this process cannot perform the audio preparation."""
    batch.research_runtime.check_interpreter(batch.research_runtime.load_manifest())
    try:
        packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile")}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("CONTEXT_V3_CAMPAIGN_DRIVER_AUDIO_PACKAGES_REQUIRED:" + str(error)) from error
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).absolute()), "packages": packages}


def data_records(root):
    root = queue.safe_path(root)
    return {name: queue.file_record(root / name) for name in projection.DATA_FILES}


def pipeline_records():
    names = set(generation.OWN_FILES) | set(selective.source_identity()) | {
        "tools/research_context_v3_campaign.py", "tools/research_queue_supervisor.py",
        "tools/research_d2_checkpoints.py", "training/research_context_v3_inputs.py",
        "training/research_d1_vocab.py"}
    return {name: queue.file_record(ROOT / name) for name in sorted(names)}


def validate_static(campaign):
    queue.require(campaign["schema"] == SCHEMA and campaign["max_chunks"] == 9 and
        campaign["total_budget_minutes"] == 180 and campaign["asr_job_minutes"] == 10 and
        campaign["asr_budget_seconds"] == 570 and campaign["threads"] == 4 and
        campaign["model_concurrency"] == 1 and campaign["automatic_retries"] == 0 and
        campaign["max_tts_calls"] == 72 and campaign["max_asr_calls"] == 216 and
        campaign["driver_wall_seconds"] == 10800 and campaign["human_gold"] is False and
        campaign["job_ids"] == [f"generate-{i:03d}" for i in range(1, 10)] and
        campaign["durable_checkpoints"] == {"schema": checkpoints.SCHEMA, "required": True,
            "intent_owner": "one-live-driver-invocation", "before_every_model_phase": True,
            "closure_before_next_chunk": True, "pending_intent_restart_allowed": False},
        "CONTEXT_V3_CAMPAIGN_FIXED_POLICY")
    for key in ("stage", "policy", "fsmn_tokens", "vocabulary", "asr_setup"):
        check_pin(campaign[key])
    queue.require(campaign["pipeline_files"] == pipeline_records() and
                  campaign["data_files"] == data_records(campaign["data_root"]),
                  "CONTEXT_V3_CAMPAIGN_SOURCE_CHANGED")


def worker_data(campaign):
    # Called only in a fresh no-model subprocess. The generator's copied
    # producer modules must never masquerade as the separate data checkout.
    validate_static(campaign)
    return projection.load_data_module(Path(campaign["data_root"]) / projection.DATA_FILES[0])

def paths(root, job_id):
    chunk = Path(root) / "chunks" / job_id
    asr_queue = chunk / "asr-queue"
    return {"chunk": chunk, "export": chunk / "export", "inputs": chunk / "inputs",
            "source_map": chunk / "source-map.json", "asr_request": chunk / "asr-request.json",
            "asr_queue_request": chunk / "asr-queue-request.json", "asr_queue": asr_queue,
            "asr_output": asr_queue / "jobs" / ASR_JOB / "outputs/asr",
            "view": chunk / "weak-view", "projection": chunk / "projection", "complete": chunk / "complete.json"}


def registration_path(stage_path):
    return Path(stage_path).parent / "campaign-driver.json"


def driver_lock(stage_path):
    return Path(stage_path).parent / "campaign-driver.lock"


def declare(args):
    stage_path, root = queue.safe_path(args.stage), queue.safe_path(args.output)
    driver = driver_identity()
    stage, _prepared, plan, _bulk = generation.load_stage(stage_path)
    qroot, qmeta, request, queue_sha = queue.load_queue(stage["queue_root"])
    queue.require(len(stage["chunks"]) == 9 and all(len(c) == 8 for c in stage["chunks"]) and
                  len(plan["cells"]) == 72 and stage["allocation"]["aggregate_runner_minutes_ceiling"] == 180,
                  "CONTEXT_V3_CAMPAIGN_EXACT72_NINE_CHUNKS")
    runtime = queue.safe_path(args.asr_runtime)
    queue.require(str(runtime / "heavy-model.lock") == stage["heavy_lock"] == qmeta["heavy_lock"],
                  "CONTEXT_V3_CAMPAIGN_SHARED_MODEL_SLOT_REQUIRED")
    python = runtime / "venv/bin/python"
    queue.require(python.is_file() and os.access(python, os.X_OK), "CONTEXT_V3_CAMPAIGN_ASR_PYTHON_MISSING")
    policy = queue.safe_path(args.policy)
    queue.require(read(policy) == selective.POLICY, "CONTEXT_V3_CAMPAIGN_POLICY_CHANGED")
    vocabulary = read(args.vocabulary)
    projection.validate_vocabulary(vocabulary, Path(args.fsmn_tokens).read_bytes())
    queue.require(queue.file_record(args.vocabulary)["sha256"] == plan["vocabulary"]["file_sha256"] and
                  projection.digest(vocabulary) == plan["vocabulary"]["canonical_sha256"],
                  "CONTEXT_V3_CAMPAIGN_ORIGINAL_VOCABULARY_REQUIRED")
    data_root = queue.safe_path(args.data_root)
    value = {"schema": SCHEMA, "created_at": queue.utc(), "root": str(root), "stage": pin(stage_path),
             "generation_queue": {"path": str(qroot), "queue_sha256": queue_sha},
             "data_root": str(data_root), "data_files": data_records(data_root), "pipeline_files": pipeline_records(),
             "driver_runtime": driver, "asr_runtime": str(runtime), "asr_python": str(python),
             "asr_setup": pin(runtime / "setup.json"), "policy": pin(policy),
             "fsmn_tokens": pin(args.fsmn_tokens), "vocabulary": pin(args.vocabulary),
             "plan_sha256": projection.digest(plan),
             "job_ids": [job["job_id"] for job in request["jobs"]],
             "max_chunks": 9, "total_budget_minutes": 180, "asr_job_minutes": 10,
             "asr_budget_seconds": 570, "threads": 4, "max_tts_calls": 72, "max_asr_calls": 216,
             "driver_wall_seconds": 10800, "model_concurrency": 1, "automatic_retries": 0, "human_gold": False,
             "durable_checkpoints": {"schema": checkpoints.SCHEMA, "required": True,
                 "intent_owner": "one-live-driver-invocation", "before_every_model_phase": True,
                 "closure_before_next_chunk": True, "pending_intent_restart_allowed": False}}
    validate_static(value)
    with queue.exclusive(driver_lock(stage_path)):
        registration = registration_path(stage_path)
        queue.require(not registration.exists() and not root.exists(), "CONTEXT_V3_NEW_CAMPAIGN_DECLARATION_REQUIRED")
        root.mkdir(mode=0o700, parents=False)
        queue.write_json(root / "campaign.json", value)
        queue.write_json(registration, {"schema": SCHEMA, "stage": value["stage"], "campaign": pin(root / "campaign.json")})
    return value

def load(path):
    path = queue.safe_path(path)
    value = read(path)
    validate_static(value)
    queue.require(path == Path(value["root"]) / "campaign.json", "CONTEXT_V3_CAMPAIGN_ROOT_CHANGED")
    queue.require(read(registration_path(value["stage"]["path"])) ==
        {"schema": SCHEMA, "stage": value["stage"], "campaign": pin(path)},
        "CONTEXT_V3_CAMPAIGN_REGISTRATION_CHANGED")
    queue.require(driver_identity() == value["driver_runtime"], "CONTEXT_V3_CAMPAIGN_DRIVER_RUNTIME_CHANGED")
    stage, _prepared, plan, _bulk = generation.load_stage(value["stage"]["path"])
    qroot, qmeta, request, queue_sha = queue.load_queue(stage["queue_root"])
    queue.require(value["generation_queue"] == {"path": str(qroot), "queue_sha256": queue_sha} and
                  value["job_ids"] == [j["job_id"] for j in request["jobs"]] and
                  len(stage["chunks"]) == 9 and all(len(c) == 8 for c in stage["chunks"]) and
                  value["plan_sha256"] == projection.digest(plan) and
                  qmeta["heavy_lock"] == str(Path(value["asr_runtime"]) / "heavy-model.lock"),
                  "CONTEXT_V3_CAMPAIGN_GENERATION_SCOPE_CHANGED")
    return value, stage, request

def asr_queue_definition(campaign, job_id):
    where = paths(campaign["root"], job_id)
    recipe = where["asr_request"]
    return {"schema": queue.SCHEMA, "queue_id": "context-v3-" + job_id + "-asr",
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
                  "CONTEXT_V3_ASR_QUEUE_SCOPE_CHANGED")
    return queue.inspect(qroot, reconcile=True)


def budget(campaign):
    """Count persisted actual time, including adopted jobs, with no estimates."""
    generated = queue.inspect(campaign["generation_queue"]["path"], reconcile=True)
    selected = set(campaign["job_ids"])
    queue.require(all(state["status"] == "pending" for name, state in generated["jobs"].items()
                      if name not in selected), "CONTEXT_V3_GENERATION_OUTSIDE_CAMPAIGN_PREFIX")
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
    stage = read(campaign["stage"]["path"])
    tts_claims = len(list((Path(stage["generation_root"]) / "cells").glob("*/started.json")))
    native_returns = sum(len(list((Path(stage["queue_root"]) / "jobs" / ident / "outputs/native").glob("*/returned.json")))
                         for ident in campaign["job_ids"])
    asr_claims = sum(len(list((paths(campaign["root"], ident)["asr_output"] / "claims").glob("*/*.json")))
                     for ident in campaign["job_ids"])
    asr_results = sum(len(list((paths(campaign["root"], ident)["asr_output"] / "results").glob("*/*.json")))
                      for ident in campaign["job_ids"])
    queue.require(tts_claims <= 72 and native_returns <= tts_claims and asr_results <= asr_claims <= 216,
                  "CONTEXT_V3_ACTUAL_CALL_ROSTER_EXCEEDS_DECLARATION")
    return {"actual_known_job_nanoseconds": known, "actual_known_job_minutes": known / queue.NS_PER_MINUTE,
            "remaining_nanoseconds": max(0, total - known), "overrun_nanoseconds": max(0, known - total),
            "actual_total_is_complete": not unresolved, "unresolved_jobs": unresolved, "failed_jobs": failures,
            "tts_claims_consumed": tts_claims, "native_return_receipts": native_returns,
            "asr_claims_consumed": asr_claims, "asr_results_saved": asr_results,
            "claimed_calls_are_not_assumed_successes": True,
            "generation": generated, "asr": asr}


def reserve(report, required_ns):
    queue.require(not report["failed_jobs"] and not report["unresolved_jobs"], "CONTEXT_V3_CAMPAIGN_UNRESOLVED_JOB")
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
    queue.require((target / "export.json").is_file(), "CONTEXT_V3_PARTIAL_EXPORT_NOT_RETRIED")
    # The exporter revalidates queue receipts and all native inventories. Its
    # deterministic replay writes only to a temporary directory, never source.
    with tempfile.TemporaryDirectory(prefix="context-v3-export-check-") as folder:
        expected = generation.export_job(campaign["stage"]["path"], job_id, Path(folder) / "export")
        queue.require(read(target / "export.json") == expected and
                      queue.file_record(target / "source-original.zip") == expected["archive"],
                      "CONTEXT_V3_EXISTING_EXPORT_CHANGED")
    return expected


def ensure_inputs(campaign, job_id, exported):
    where = paths(campaign["root"], job_id)
    manifest_path = where["inputs"] / "inputs.json"
    count = len(exported["included"])
    if not where["inputs"].exists():
        batch.prepare(Namespace(archive=where["export"] / "source-original.zip",
                                source_sha256=exported["archive"]["sha256"], expected_count=count,
                                output=where["inputs"]))
    queue.require(manifest_path.is_file(), "CONTEXT_V3_PARTIAL_INPUTS_NOT_RETRIED")
    manifest = batch.verify_inputs(manifest_path)
    queue.require(manifest["source_archive_sha256"] == exported["archive"]["sha256"] and
                  len(manifest["clips"]) == count, "CONTEXT_V3_EXISTING_INPUTS_CHANGED")
    return manifest_path


def nonmodel_worker(campaign, job_id, task):
    where = paths(campaign["root"], job_id)
    command = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_context_v3_campaign.py"),
               "worker", "--campaign", str(Path(campaign["root"]) / "campaign.json"),
               "--job-id", job_id, "--task", task]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    queue.require(result.returncode == 0,
                  "CONTEXT_V3_NONMODEL_WORKER_FAILED:" + task + ":" + result.stderr[-2000:] + result.stdout[-2000:])
    value = queue.loads(result.stdout.encode())
    queue.require(value.get("task") == task and value.get("new_model_calls") == 0,
                  "CONTEXT_V3_NONMODEL_WORKER_SCOPE")
    receipt = where["chunk"] / (task + "-replay.json")
    if receipt.exists():
        queue.require(read(receipt) == value, "CONTEXT_V3_NONMODEL_REPLAY_CHANGED")
    else:
        queue.write_json(receipt, value)
    return value


def ensure_source_map(campaign, job_id, exported, inputs):
    nonmodel_worker(campaign, job_id, "source-map")
    return paths(campaign["root"], job_id)["source_map"]

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
                  "CONTEXT_V3_EXISTING_ASR_REQUEST_CHANGED")
    return request


def ensure_asr_queue(campaign, job_id):
    where = paths(campaign["root"], job_id)
    expected = asr_queue_definition(campaign, job_id)
    if where["asr_queue_request"].exists():
        queue.require(read(where["asr_queue_request"]) == expected, "CONTEXT_V3_EXISTING_ASR_QUEUE_REQUEST_CHANGED")
    else:
        queue.write_json(where["asr_queue_request"], expected)
    if not where["asr_queue"].exists():
        queue.create(where["asr_queue"], where["asr_queue_request"],
                     Path(campaign["generation_queue"]["path"]) / "active-plan.json",
                     Path(campaign["asr_runtime"]) / "heavy-model.lock")
    return check_asr_queue(campaign, job_id)


def ensure_view(campaign, job_id, exported, inputs):
    audit = selective.seal_or_verify(paths(campaign["root"], job_id)["asr_request"])
    queue.require(audit["state"]["status"] == "complete", "CONTEXT_V3_ASR_AUDIT_INCOMPLETE")
    nonmodel_worker(campaign, job_id, "weak-view")
    return read(paths(campaign["root"], job_id)["view"] / "report.json")


def execute_nonmodel(campaign_path, job_id, task):
    campaign = read(campaign_path)
    queue.require(Path(campaign_path) == Path(campaign["root"]) / "campaign.json" and job_id in campaign["job_ids"],
                  "CONTEXT_V3_WORKER_DECLARATION_SCOPE")
    data, source = worker_data(campaign)
    where = paths(campaign["root"], job_id)
    exported = read(where["export"] / "export.json")
    inputs = where["inputs"] / "inputs.json"
    options = source_options(campaign, exported)
    if task == "source-map":
        expected = data.build_source_map(inputs, queue.file_record(inputs)["sha256"],
            where["export"] / "source-original.zip", **options)
        if where["source_map"].exists():
            queue.require(read(where["source_map"]) == expected, "CONTEXT_V3_EXISTING_SOURCE_MAP_CHANGED")
        else:
            queue.write_json(where["source_map"], expected)
        return {"task": task, "source_map": pin(where["source_map"]), "new_model_calls": 0}
    if task == "weak-view":
        final = where["asr_output"] / "audit/final-summary.json"
        if not where["view"].exists():
            data.create_view(inputs, queue.file_record(inputs)["sha256"], where["export"] / "source-original.zip",
                where["asr_output"], where["view"], fsmn_tokens=campaign["fsmn_tokens"]["path"],
                selective_source_map=where["source_map"], selective_policy=campaign["policy"]["path"],
                selective_audit_sha256=queue.file_record(final)["sha256"], **options)
        report = data.verify_view(where["view"])
        owner = report["source_adapter"]["job_owner"]
        queue.require(report["inputs_sha256"] == queue.file_record(inputs)["sha256"] and
            report["source_archive_sha256"] == exported["archive"]["sha256"] and
            report["asr_adapter"]["final_audit_sha256"] == queue.file_record(final)["sha256"] and
            report["source_adapter"]["plan_sha256"] == campaign["plan_sha256"] and
            owner["job_recipe_sha256"] == exported["job_recipe"]["sha256"] and
            owner["stage_sha256"] == campaign["stage"]["sha256"] and owner["job_id"] == job_id,
            "CONTEXT_V3_WEAK_VIEW_SCOPE_CHANGED")
        return {"task": task, "manifest": pin(where["view"] / "MANIFEST.json"),
                "statistics": report["statistics"], "new_model_calls": 0}
    queue.require(task == "projection", "CONTEXT_V3_UNKNOWN_NONMODEL_TASK")
    packages = []
    for prior in campaign["job_ids"][:campaign["job_ids"].index(job_id) + 1]:
        prior_where = paths(campaign["root"], prior)
        if prior != job_id:
            check_chunk_receipt(campaign, prior)
            checkpoints.verify_ack(Path(campaign["root"]) / "checkpoints" / prior / "chunk-complete")
        if prior_where["view"].exists():
            packages.append({"path": str(prior_where["view"]),
                             "manifest_sha256": queue.file_record(prior_where["view"] / "MANIFEST.json")["sha256"]})
    rows, report = projection.verified_projection(packages, data, source["files"],
        expected_plan_sha256=campaign["plan_sha256"], vocabulary_path=campaign["vocabulary"]["path"],
        vocabulary_sha256=campaign["vocabulary"]["sha256"], original_tokens_path=campaign["fsmn_tokens"]["path"])
    bodies = {"report.json": queue.canonical(report), "packages.json": queue.canonical(packages),
              "rows.jsonl": b"".join(queue.canonical(row) for row in rows)}
    manifest = {name: queue.sha(raw) for name, raw in bodies.items()}
    if where["projection"].exists():
        queue.require(read(where["projection"] / "MANIFEST.json") == manifest and
            {p.name for p in where["projection"].iterdir()} == set(bodies) | {"MANIFEST.json"} and
            all((where["projection"] / name).read_bytes() == raw for name, raw in bodies.items()),
            "CONTEXT_V3_PROJECTION_CHANGED_OR_PARTIAL")
    else:
        where["projection"].mkdir(mode=0o700)
        for name, raw in bodies.items():
            queue.write_new(where["projection"] / name, raw)
        queue.write_json(where["projection"] / "MANIFEST.json", manifest)
    return {"task": task, "manifest": pin(where["projection"] / "MANIFEST.json"),
            "observed_rows": report["observed_rows"], "eligible_rows": report["eligible_rows"],
            "eligible_rows_by_category": report["eligible_rows_by_category"], "new_model_calls": 0}

def seal_chunk(campaign, job_id, exported, report):
    where = paths(campaign["root"], job_id)
    projected = read(where["projection"] / "report.json") if report is not None else None
    files = {str(path.relative_to(where["chunk"])): queue.file_record(path)
             for path in checkpoints.regular_tree(where["chunk"]) if path != where["complete"]}
    value = {"schema": COMPLETION, "campaign": pin(Path(campaign["root"]) / "campaign.json"),
        "job_id": job_id, "status": "completed" if report is not None else "completed_no_asr_eligible_return",
        "files": files, "generation_result": exported["job_result"],
        "native_included": len(exported["included"]), "native_excluded": exported["excluded"],
        "statistics": report["statistics"] if report is not None else None,
        "cumulative_full2601": projected, "human_gold": False, "automatic_retries": 0}
    if where["complete"].exists():
        queue.require(read(where["complete"]) == value, "CONTEXT_V3_CHUNK_COMPLETION_CHANGED")
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
                  "CONTEXT_V3_CHUNK_RECEIPT_SCOPE_CHANGED")
    check_pin(value["generation_result"])
    actual = {str(path.relative_to(where["chunk"])) for path in checkpoints.regular_tree(where["chunk"])
              if path != where["complete"]}
    queue.require(actual == set(value["files"]), "CONTEXT_V3_CHUNK_FILE_ROSTER_CHANGED")
    for name, expected in value["files"].items():
        queue.require(queue.file_record(where["chunk"] / queue.relative(name)) == expected,
                      "CONTEXT_V3_CHUNK_RECEIPT_FILE_CHANGED:" + name)
    return value


def compact_budget(report):
    return {key: value for key, value in report.items() if key not in ("generation", "asr")}


def checkpoint_sources(campaign, stage, job_id):
    prepared = Path(stage["prepared"])
    names = generation.checkpoint_sources(stage, prepared, job_id)
    names += [Path(campaign["root"]) / "campaign.json", registration_path(campaign["stage"]["path"])]
    names += [Path(campaign[key]["path"]) for key in ("policy", "fsmn_tokens", "vocabulary", "asr_setup")]
    for key, root in (("pipeline_files", ROOT), ("data_files", Path(campaign["data_root"]))):
        names += [root / queue.relative(name) for name in campaign[key]]
    where = paths(campaign["root"], job_id)
    if where["chunk"].exists():
        names += checkpoints.regular_tree(where["chunk"])
    for ack in sorted((Path(campaign["root"]) / "checkpoints").glob("*/*/ack.json")):
        names += [ack, ack.parent / "provider-save.json", ack.parent / "capsule.json", ack.parent / "phase-marker.json"]
    return names


def phase_checkpoint(gate, campaign, stage, job_id, phase, *, progress=None, terminal_state=None):
    prepared = Path(stage["prepared"])
    recipe_path = Path(stage["queue_root"]) / "recipes" / (job_id + ".json")
    recipe = read(recipe_path)
    generation.validate_job_recipe(recipe, stage, campaign["stage"]["path"])
    if phase.startswith("generation") or phase == "before-generation":
        folder = prepared / "checkpoints" / job_id / phase
        scope = generation.checkpoint_scope(campaign["stage"]["path"], stage, recipe_path, recipe, phase)
        if phase != "before-generation":
            state, result = terminal_receipt(Path(stage["queue_root"]), job_id)
            queue.require(state == terminal_state, "CONTEXT_V3_GENERATION_TERMINAL_STATE_CHANGED")
            scope.update(queue_result_status=state, queue_result=result)
    else:
        folder = Path(campaign["root"]) / "checkpoints" / job_id / phase
        where = paths(campaign["root"], job_id)
        scope = {"schema": "kws-context-v3-campaign-phase-v1", "campaign": pin(Path(campaign["root"]) / "campaign.json"),
            "stage": campaign["stage"], "job_id": job_id, "phase": phase,
            "cell_ids": recipe["selected_cells"], "max_tts_calls": 8, "max_asr_calls": 24,
            "automatic_retries": 0, "resume_pending_intent_after_restart": False}
        if where["asr_request"].exists():
            scope.update(asr_request=pin(where["asr_request"]), asr_queue=pin(where["asr_queue"] / "queue.json"))
        if phase.startswith("asr-"):
            actual_state, result = terminal_receipt(where["asr_queue"], ASR_JOB)
            queue.require(terminal_state in (actual_state, "failed_or_unknown"),
                          "CONTEXT_V3_ASR_TERMINAL_STATE_CHANGED")
            terminal_state = actual_state
            scope["asr_result"] = result
        if phase == "chunk-complete":
            scope["completion"] = pin(where["complete"])
        if terminal_state is not None:
            scope["terminal_state"] = terminal_state
    return gate.checkpoint(folder, scope, checkpoint_sources(campaign, stage, job_id),
        intent=phase in ("before-generation", "before-asr"),
        on_progress=(lambda: progress(job_id, "awaiting_durable_save:" + phase)) if progress else None)


def terminal_receipt(root, job_id):
    queue.inspect(root, reconcile=True)
    root, _metadata, request, queue_sha = queue.load_queue(root)
    definitions = [job for job in request["jobs"] if job["job_id"] == job_id]
    queue.require(len(definitions) == 1, "CONTEXT_V3_TERMINAL_JOB_REQUIRED")
    state = queue.result_for(root, definitions[0], queue_sha)
    queue.require(state["status"] in ("completed", "failed", "timed_out", "unknown"),
                  "CONTEXT_V3_AUTHENTIC_TERMINAL_REQUIRED")
    return state["status"], pin(root / "jobs" / job_id / "result.json")


def ensure_wall(deadline, reserve_seconds=0):
    queue.require(deadline is None or time.monotonic() + reserve_seconds < deadline,
                  "CONTEXT_V3_DRIVER_WALL_BUDGET_EXHAUSTED")


def run_generation(campaign, stage, job_id, gate, progress, deadline=None):
    phase_checkpoint(gate, campaign, stage, job_id, "before-generation", progress=progress)
    ensure_wall(deadline, 600)
    folder = Path(stage["prepared"]) / "checkpoints" / job_id / "before-generation"
    old = {key: os.environ.get(key) for key in (generation.INTENT_FOLDER_ENV, generation.INTENT_OWNER_ENV)}
    os.environ.update({generation.INTENT_FOLDER_ENV: str(folder), generation.INTENT_OWNER_ENV: gate.owner})
    try:
        return queue.run(campaign["generation_queue"]["path"], max_jobs=1)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def advance(campaign_path, *, max_chunks_this_call=None, progress=None, checkpoint_gate=None, deadline=None):
    campaign, stage, request = load(campaign_path)
    queue.require(checkpoint_gate is not None, "CONTEXT_V3_DURABLE_HANDOFF_REQUIRED")
    done, completed = 0, []
    for job_id in campaign["job_ids"]:
        ensure_wall(deadline)
        where = paths(campaign["root"], job_id)
        if where["complete"].exists():
            check_chunk_receipt(campaign, job_id)
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "chunk-complete", progress=progress)
            completed.append(job_id)
            continue
        if max_chunks_this_call is not None and done >= max_chunks_this_call:
            return {"status": "paused", "completed": completed, "new_completed_chunks": done,
                    "budget": compact_budget(budget(campaign))}, 0
        current = budget(campaign)
        state = current["generation"]["jobs"][job_id]["status"]
        if state in ("failed", "timed_out", "unknown"):
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "generation-terminal",
                             terminal_state=state, progress=progress)
            return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(current)}, 1
        if current["failed_jobs"]:
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "asr-terminal",
                             terminal_state="failed_or_unknown", progress=progress)
            return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(current)}, 1
        if current["unresolved_jobs"]:
            return {"status": "waiting_for_worker", "job_id": job_id, "budget": compact_budget(current)}, 75
        if current["overrun_nanoseconds"]:
            return {"status": "budget_exhausted", "job_id": job_id, "budget": compact_budget(current)}, 3
        if progress: progress(job_id, "generation")
        if state == "pending":
            queue.require(not where["chunk"].exists(), "CONTEXT_V3_PENDING_GENERATION_WITH_DOWNSTREAM_FILES")
            definition = next(item for item in request["jobs"] if item["job_id"] == job_id)
            generation.validate_job_recipe(read(Path(stage["queue_root"]) / "recipes" / (job_id + ".json")),
                                           stage, campaign["stage"]["path"])
            allowance = min(queue.minutes_ns(definition["timeout_minutes"], 999),
                            current["generation"]["remaining_budget_nanoseconds"])
            if allowance <= 0 or not reserve(current, allowance + queue.minutes_ns(10)):
                return {"status": "budget_exhausted", "job_id": job_id, "budget": compact_budget(current)}, 3
            generated, code = run_generation(campaign, stage, job_id, checkpoint_gate, progress, deadline)
            state = generated["jobs"][job_id]["status"]
            if state == "pending" or state == "running":
                return {"status": generated["status"], "job_id": job_id, "budget": compact_budget(budget(campaign))}, code
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id,
                "generation-complete" if state == "completed" else "generation-terminal",
                terminal_state=state, progress=progress)
            if code or state != "completed":
                return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(budget(campaign))}, 1
        else:
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "generation-complete",
                             terminal_state="completed", progress=progress)
        where["chunk"].mkdir(mode=0o700, parents=True, exist_ok=True)
        if progress: progress(job_id, "export_and_source_map")
        exported = ensure_export(campaign, job_id)
        view = None
        if exported["included"]:
            inputs = ensure_inputs(campaign, job_id, exported)
            mapping = ensure_source_map(campaign, job_id, exported, inputs)
            ensure_asr_request(campaign, job_id, inputs, mapping)
            asr = ensure_asr_queue(campaign, job_id)
            if asr["jobs"][ASR_JOB]["status"] != "completed":
                current = budget(campaign)
                if current["failed_jobs"]:
                    phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "asr-terminal",
                                     terminal_state="failed_or_unknown", progress=progress)
                    return {"status": "failed_or_unknown", "job_id": job_id, "budget": compact_budget(current)}, 1
                if current["unresolved_jobs"]:
                    return {"status": "waiting_for_worker", "job_id": job_id, "budget": compact_budget(current)}, 75
                if not reserve(current, queue.minutes_ns(10)):
                    return {"status": "budget_exhausted", "job_id": job_id, "budget": compact_budget(current)}, 3
                phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "before-asr", progress=progress)
                ensure_wall(deadline, 600)
                if progress: progress(job_id, "asr")
                asr, code = queue.run(where["asr_queue"], max_jobs=1)
                if code:
                    if asr["jobs"][ASR_JOB]["status"] not in ("pending", "running"):
                        phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "asr-terminal",
                                         terminal_state=asr["jobs"][ASR_JOB]["status"], progress=progress)
                    return {"status": asr["status"], "job_id": job_id, "budget": compact_budget(budget(campaign))}, code
            # Preserve the expensive recognizer results even if a later pure
            # replay/projection fails. This never authorizes another model call.
            phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "asr-complete",
                             terminal_state="completed", progress=progress)
            if progress: progress(job_id, "weak_view_and_full2601")
            view = ensure_view(campaign, job_id, exported, inputs)
            nonmodel_worker(campaign, job_id, "projection")
        elif any(where[name].exists() for name in ("inputs", "source_map", "asr_request", "asr_queue", "view", "projection")):
            raise ValueError("CONTEXT_V3_EMPTY_EXPORT_HAS_UNEXPECTED_ASR_ARTIFACTS")
        receipt = seal_chunk(campaign, job_id, exported, view)
        phase_checkpoint(checkpoint_gate, campaign, stage, job_id, "chunk-complete", progress=progress)
        print(json.dumps({"event": "chunk_durably_closed", "job_id": job_id,
            "native_included": receipt["native_included"], "statistics": receipt["statistics"],
            "full2601_observed": receipt["cumulative_full2601"]["observed_rows"] if view else None,
            "full2601_eligible": receipt["cumulative_full2601"]["eligible_rows"] if view else None}), flush=True)
        completed.append(job_id); done += 1
    final = budget(campaign)
    return {"status": "completed", "completed": completed, "new_completed_chunks": done,
            "budget": compact_budget(final)}, 0

def run(campaign_path, *, max_chunks_this_call=None, wait_seconds=1.0, checkpoint_gate=None):
    campaign, _stage, _request = load(campaign_path)
    queue.require(campaign["schema"] == SCHEMA, "CONTEXT_V3_DURABLE_HANDOFF_REQUIRED_FOR_EXECUTION")
    if checkpoint_gate is None:
        announced = set()
        def on_wait(folder, descriptor):
            if str(folder) not in announced:
                print(json.dumps({"event": "awaiting_durable_save", "checkpoint": str(folder),
                                  "capsule": descriptor["capsule"]}, sort_keys=True), flush=True)
                announced.add(str(folder))
        checkpoint_gate = checkpoints.Gate(on_wait=on_wait)
    queue.require(type(wait_seconds) in (int, float) and .02 <= wait_seconds <= 30,
                  "CONTEXT_V3_BOUNDED_WAIT_INTERVAL_REQUIRED")
    root = Path(campaign["root"])
    with queue.exclusive(driver_lock(campaign["stage"]["path"])):
        queue.require(max_chunks_this_call is None or type(max_chunks_this_call) is int and max_chunks_this_call > 0,
                      "CONTEXT_V3_POSITIVE_CHUNKS_PER_INVOCATION_REQUIRED")
        sealed_before = {job_id for job_id in campaign["job_ids"]
                         if paths(root, job_id)["complete"].exists()}
        started_monotonic = time.monotonic()
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
        thread = threading.Thread(target=heartbeat, name="context-v3-campaign-heartbeat", daemon=True)
        thread.start()
        try:
            while True:
                queue.require(time.monotonic() - started_monotonic < campaign["driver_wall_seconds"],
                              "CONTEXT_V3_DRIVER_WALL_BUDGET_EXHAUSTED")
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
                                       checkpoint_gate=checkpoint_gate,
                                       deadline=started_monotonic + campaign["driver_wall_seconds"])
                publish(result.get("job_id"), result["status"])
                if code != 75:
                    return result, code
                # Do not repeatedly replay large completed evidence while a
                # different bounded owner still holds the model/controller.
                time.sleep(wait_seconds)
                while model_or_controller_active(campaign):
                    queue.require(time.monotonic() - started_monotonic < campaign["driver_wall_seconds"],
                                  "CONTEXT_V3_DRIVER_WALL_BUDGET_EXHAUSTED")
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
    for name in ("stage", "output", "data-root", "asr-runtime", "vocabulary", "fsmn-tokens"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--policy", type=Path, default=selective.POLICY_PATH)
    for name in ("run", "status"):
        command = commands.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name == "run":
            command.add_argument("--max-chunks-this-call", type=int)
            command.add_argument("--wait-seconds", type=float, default=1)
    child = commands.add_parser("worker")
    child.add_argument("--campaign", type=Path, required=True)
    child.add_argument("--job-id", required=True)
    child.add_argument("--task", choices=("source-map", "weak-view", "projection"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "declare":
            result, code = declare(args), 0
        elif args.command == "run":
            result, code = run(args.campaign, max_chunks_this_call=args.max_chunks_this_call, wait_seconds=args.wait_seconds)
        elif args.command == "worker":
            result, code = execute_nonmodel(args.campaign, args.job_id, args.task), 0
        else:
            result, code = status(args.campaign), 0
    except BlockingIOError:
        result, code = {"status": "another_driver_or_controller_is_active"}, 75
    except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError) as error:
        result, code = {"status": "invalid", "reason": str(error)}, 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
