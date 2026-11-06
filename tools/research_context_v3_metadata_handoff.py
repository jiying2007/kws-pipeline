#!/usr/bin/env python3
"""Source-bound continuation of the authenticated context-v3 PEAK metadata stop.

The original campaign, four closed weak packages, generation queue, model
workers and forty consumed TTS calls stay unchanged. Only the unclaimed ASR5
and generation/ASR6–9 pairs can run. Each weak package keeps its exact reader.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training")]
import research_queue as q
import research_d2_checkpoints as cp
import research_queue_supervisor as supervisor
import research_context_v3_inputs as original_inputs

SCHEMA = "kws-context-v3-metadata-handoff-v1"
CHUNK = "kws-context-v3-metadata-handoff-chunk-v1"
ASR_JOB = "asr-selective"
ASR_OWNER_ENV = "KWS_CONTEXT_V3_METADATA_ASR_OWNER"
ASR_INTENT_ENV = "KWS_CONTEXT_V3_METADATA_ASR_INTENT"
OLD = [f"generate-{i:03d}" for i in range(1, 5)]
NEW = [f"generate-{i:03d}" for i in range(5, 10)]
GEN = NEW[1:]
ENTRY = "automation/active_context_v3_metadata_weak_view.py"
DATA_FILES = (ENTRY, *original_inputs.DATA_FILES)
CODE = ("tools/research_context_v3_metadata_handoff.py", "training/research_context_v3_handoff_inputs.py",
        "training/research_context_v3_inputs.py", "training/research_d1_vocab.py",
        "tools/research_queue.py", "tools/research_job_lock.py", "tools/research_d2_checkpoints.py",
        "tools/research_queue_supervisor.py")


def read(path):
    return q.loads(q.read_bytes(path, 64 * 1024 ** 2))


def pin(path):
    return cp.pin(path)


def check_pin(value):
    q.require(pin(value["path"]) == value, "V3_HANDOFF_PIN_CHANGED:" + value["path"])


def verify_scope(folder, expected, *, intent=False, owner=None):
    """A genuine save authorizes only its exact declared operation."""
    cp.verify_ack(folder)
    actual = read(Path(folder) / "capsule.json")["context"]
    q.require(type(actual.get("owner_invocation")) is str and bool(actual["owner_invocation"]) and
              {k: v for k, v in actual.items() if k != "owner_invocation"} ==
              {**expected, "launch_intent": intent} and
              (owner is None or actual["owner_invocation"] == owner), "V3_HANDOFF_ACK_SCOPE_CHANGED")
    return actual


def once(path, value):
    if path.exists():
        q.require(read(path) == value, "V3_HANDOFF_EXISTING_VALUE_CHANGED:" + str(path))
    else:
        q.write_json(path, value)


def records(root, names):
    return {name: q.file_record(Path(root) / name) for name in names}


def reader(root, revision):
    names = original_inputs.DATA_FILES if revision == "context-v3-r1" else DATA_FILES
    q.require(revision in ("context-v3-r1", "context-v3-metadata-r2"), "V3_HANDOFF_READER_REVISION")
    root = q.safe_path(root)
    return {"revision": revision, "root": str(root),
            "source_files_sha256": {name: q.file_record(root / name)["sha256"] for name in names}}


def original_module(validator):
    """Execute the original module from its own original path; no ROOT patch."""
    check_pin(validator)
    path = Path(validator["path"])
    q.require(path.name == "research_context_v3_campaign.py" and path.parent.name == "tools",
              "V3_HANDOFF_ORIGINAL_VALIDATOR_PATH")
    name = "_context_v3_original_campaign_for_metadata_handoff"
    previous = sys.modules.get(name)
    if previous is not None:
        q.require(Path(previous.__file__) == path, "V3_HANDOFF_ORIGINAL_MODULE_LOCATION_CHANGED")
        module = previous
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    q.require(module.ROOT == path.parents[1], "V3_HANDOFF_ORIGINAL_ROOT_CHANGED")
    return module


def load_original(validator, parent_path):
    module = original_module(validator)
    parent = read(parent_path)
    dependency_provenance(module, parent)
    parent, stage, request = module.load(parent_path)
    q.require(q.file_record(validator["path"]) == parent["pipeline_files"]["tools/research_context_v3_campaign.py"],
              "V3_HANDOFF_ORIGINAL_VALIDATOR_SOURCE_CHANGED")
    return module, parent, stage, request


def dependency_provenance(module, parent):
    """Shared cached utilities must be byte-identical to the original pins.

    The original module and its path-sensitive generation/ASR imports execute
    from their original checkout. Only the six already imported, unchanged
    utility modules may come from this separately frozen checkout.
    """
    shared = {"research_queue", "research_job_lock", "research_d2_checkpoints",
              "research_queue_supervisor", "research_context_v3_inputs", "research_d1_vocab"}
    observed = {}
    for relative, expected in parent["pipeline_files"].items():
        if not relative.endswith(".py"):
            continue
        name = Path(relative).stem
        loaded = module if name == "research_context_v3_campaign" else sys.modules.get(name)
        if loaded is None:
            continue
        path = q.safe_path(loaded.__file__)
        allowed = [module.ROOT / relative]
        if name in shared:
            allowed.append(ROOT / relative)
        q.require(path in allowed and q.file_record(path) == expected and
                  q.file_record(module.ROOT / relative) == expected,
                  "V3_HANDOFF_ORIGINAL_DEPENDENCY_CHANGED:" + name)
        observed[name] = pin(path)
    q.require(all(name in observed for name in
                  ("research_context_v3_campaign", "research_context_v3_generate", "active_asr_batch",
                   "active_asr_selective", *sorted(shared))), "V3_HANDOFF_ORIGINAL_DEPENDENCY_MISSING")
    return observed


def prefix_scope(value, parent, stage, item):
    job = item["job_id"]
    chunk = Path(parent["root"]) / "chunks" / job
    return {"schema": "kws-context-v3-campaign-phase-v1", "phase": "chunk-complete",
        "campaign": value["parent_campaign"], "stage": parent["stage"], "job_id": job,
        "cell_ids": stage["chunks"][int(job[-3:]) - 1], "max_tts_calls": 8, "max_asr_calls": 24,
        "automatic_retries": 0, "resume_pending_intent_after_restart": False,
        "asr_request": pin(chunk / "asr-request.json"), "asr_queue": pin(chunk / "asr-queue/queue.json"),
        "completion": item["completion"]}


def static(value):
    q.require(value["schema"] == SCHEMA and value["original_completed_chunks"] == OLD and
        value["continuation_chunks"] == NEW and value["new_generation_jobs"] == GEN and
        value["read_only_generation_jobs"] == ["generate-005"] and value["max_new_tts_calls"] == 32 and
        value["max_new_asr_calls"] == 120 and value["max_total_tts_calls"] == 72 and
        value["max_total_asr_calls"] == 216 and value["total_actual_runner_minutes"] == 180 and
        value["new_runner_allocation_minutes"] == 90 and value["driver_wall_seconds"] == 5400 and
        value["worker_seconds"] == 570 and value["job_minutes"] == 10 and value["automatic_retries"] == 0 and
        value["model_concurrency"] == 1 and value["human_gold"] is False and
        value["pending_intent_restart_allowed"] is False and
        value["initial_status_lock_owner"] == "this-declaration-holds-original-driver-lock", "V3_HANDOFF_FIXED_SCOPE_CHANGED")
    for key in ("parent_campaign", "original_validator", "original_stage", "initial_status", "failure_receipt"):
        check_pin(value[key])
    cp.verify_ack(value["failure_checkpoint"])
    failure_scope = read(Path(value["failure_checkpoint"]) / "capsule.json")["context"]
    q.require(failure_scope["failure"] == value["failure_receipt"] and failure_scope["campaign"] == value["parent_campaign"] and
              failure_scope["new_model_calls"] == 0, "V3_HANDOFF_FAILURE_CLOSURE_CHANGED")
    parent = read(value["parent_campaign"]["path"])
    q.require(value["pipeline_files"] == records(ROOT, set(parent["pipeline_files"]) | set(CODE)) and
        value["original_reader"] == reader(parent["data_root"], "context-v3-r1") and
        value["metadata_reader"] == reader(value["metadata_reader"]["root"], "context-v3-metadata-r2") and
        all(value["original_reader"]["source_files_sha256"][name] ==
            value["metadata_reader"]["source_files_sha256"][name] for name in original_inputs.DATA_FILES) and
        value["driver_runtime"] == parent["driver_runtime"] and value["original_stage"] == parent["stage"] and
        value["heavy_lock"] == str(Path(parent["asr_runtime"]) / "heavy-model.lock"),
        "V3_HANDOFF_SOURCE_OR_RUNTIME_CHANGED")
    q.require([item["job_id"] for item in value["original_prefix"]] == OLD and
              list(value["original_generation_results"]) == OLD + ["generate-005"],
              "V3_HANDOFF_ORIGINAL_PREFIX_ROSTER_CHANGED")
    stage = read(parent["stage"]["path"])
    for item in value["original_prefix"]:
        check_pin(item["completion"]); check_pin(item["weak_manifest"])
        verify_scope(item["checkpoint"], prefix_scope(value, parent, stage, item))
    for original in value["original_generation_results"].values():
        check_pin(original)
    q.require(read(value["initial_status"]["path"])["budget"]["tts_claims_consumed"] == 40 and
              read(value["initial_status"]["path"])["budget"]["asr_claims_consumed"] == 71,
              "V3_HANDOFF_INITIAL_MODEL_COUNTS_CHANGED")
    return value


def load(path):
    value = static(read(path))
    q.require(Path(path).absolute() == Path(value["root"]) / "handoff.json" and
              read(value["registration"]) == {"schema": SCHEMA, "parent_campaign": value["parent_campaign"],
                                             "handoff": pin(path)}, "V3_HANDOFF_REGISTRATION_CHANGED")
    old, parent, stage, request = load_original(value["original_validator"], value["parent_campaign"]["path"])
    q.require(old.driver_identity() == value["driver_runtime"], "V3_HANDOFF_DRIVER_RUNTIME_CHANGED")
    for item in value["original_prefix"]:
        old.check_chunk_receipt(parent, item["job_id"])
    return value, old, parent, stage, request


def context(value, parent):
    # Only operational parameters for original export/audio/ASR helpers. This
    # is not a rewritten campaign declaration or a replacement data checkout.
    keys = ("stage", "generation_queue", "asr_runtime", "asr_python", "policy", "fsmn_tokens", "vocabulary",
            "plan_sha256", "threads", "asr_budget_seconds", "asr_job_minutes")
    return {**{key: parent[key] for key in keys}, "root": value["root"]}


def where(value, job):
    root = Path(value["root"]) / "chunks" / job
    queue = root / "asr-queue"
    return {"chunk": root, "export": root / "export", "inputs": root / "inputs/inputs.json",
        "source_map": root / "source-map.json", "request": root / "asr-request.json", "queue": queue,
        "asr": queue / "jobs" / ASR_JOB / "outputs/asr", "view": root / "weak-view",
        "projection": root / "projection", "complete": root / "complete.json"}


def asr_definition(value, parent, job):
    path = where(value, job)
    return {"schema": q.SCHEMA, "queue_id": "context-v3-metadata-" + job + "-asr", "global_budget_minutes": 10,
        "jobs": [{"job_id": ASR_JOB, "recipe": str(path["request"]),
            "recipe_sha256": q.file_record(path["request"])["sha256"],
            "argv": [parent["asr_python"], "-B", str(ROOT / "tools/research_context_v3_metadata_handoff.py"),
                "asr-worker", "--handoff", str(Path(value["root"]) / "handoff.json"),
                "--job-id", job, "--request", "{recipe}"], "cwd": str(ROOT), "depends_on": [],
            "timeout_minutes": 10, "outputs": ["asr/audit/final-summary.json", "asr/audit/file-hashes.json"]}]}


def check_asr_queue(value, parent, job):
    path = where(value, job)
    root, metadata, request, _ = q.load_queue(path["queue"])
    q.require(request == asr_definition(value, parent, job) and metadata["heavy_lock"] == value["heavy_lock"],
              "V3_HANDOFF_ASR_QUEUE_SCOPE_CHANGED")
    return q.inspect(root, reconcile=True)


def ensure_asr_queue(value, parent, job):
    path = where(value, job)
    declaration = path["chunk"] / "asr-queue-request.json"
    once(declaration, asr_definition(value, parent, job))
    if not path["queue"].exists():
        q.create(path["queue"], declaration, Path(parent["generation_queue"]["path"]) / "active-plan.json", value["heavy_lock"])
    return check_asr_queue(value, parent, job)


def create_chunk_layout(root):
    """Create the fixed empty parent directories before the create-only export."""
    root = q.safe_path(root)
    chunks = root / "chunks"
    q.require(root.is_dir() and not chunks.exists(), "V3_HANDOFF_CREATE_ONLY_CHUNK_LAYOUT")
    chunks.mkdir(mode=0o700)
    for job in NEW:
        (chunks / job).mkdir(mode=0o700)


def ensure_chunk_parent(root, job):
    """Restore a missing empty parent; never replace files, links or exports."""
    root = q.safe_path(root)
    q.require(root.is_dir() and job in NEW, "V3_HANDOFF_CHUNK_LAYOUT_SCOPE")
    chunk = q.safe_path(root / "chunks" / job)
    chunk.mkdir(mode=0o700, parents=True, exist_ok=True)
    return chunk


def declare(args):
    root, parent_path, validator = q.safe_path(args.output), q.safe_path(args.campaign), pin(args.original_validator)
    old, parent, stage, request = load_original(validator, parent_path)
    with q.exclusive(old.driver_lock(parent["stage"]["path"])):
        # Inspect under this lock, so the expected active flag is this owner.
        initial = old.status(parent_path)
        q.require(initial["driver_active"] is True and initial["completed_chunks"] == OLD and initial["budget"]["tts_claims_consumed"] == 40 and
            initial["budget"]["native_return_receipts"] == 40 and initial["budget"]["asr_claims_consumed"] == 71 and
            initial["budget"]["asr_results_saved"] == 71 and initial["budget"]["failed_jobs"] ==
            initial["budget"]["unresolved_jobs"] == [], "V3_HANDOFF_EXACT_ORIGINAL_PREFIX_REQUIRED")
        report = q.inspect(parent["generation_queue"]["path"], reconcile=True)
        q.require(all(report["jobs"][job]["status"] == "completed" for job in OLD + ["generate-005"]) and
                  all(report["jobs"][job]["status"] == "pending" for job in GEN), "V3_HANDOFF_ONLY_UNCLAIMED_GENERATION")
        for job in NEW:
            q.require(not (Path(parent["root"]) / "chunks" / job / "asr-request.json").exists() and
                      not (Path(parent["root"]) / "checkpoints" / job / "before-asr").exists(),
                      "V3_HANDOFF_OLD_ASR_INTENT_OR_CLAIM_FORBIDDEN")
        for job in GEN:
            q.require(not (Path(stage["prepared"]) / "checkpoints" / job / "before-generation").exists(),
                      "V3_HANDOFF_OLD_GENERATION_INTENT_FORBIDDEN")
        registration = Path(parent["root"]).parent / "context-v3-metadata-handoff.json"
        q.require(not registration.exists() and not root.exists(), "V3_HANDOFF_CREATE_ONLY_DECLARATION")
        root.mkdir(mode=0o700)
        create_chunk_layout(root)
        q.write_json(root / "initial-status.json", initial)
        prefix = [{"job_id": job, "completion": pin(Path(parent["root"]) / "chunks" / job / "complete.json"),
            "weak_manifest": pin(Path(parent["root"]) / "chunks" / job / "weak-view/MANIFEST.json"),
            "checkpoint": str(Path(parent["root"]) / "checkpoints" / job / "chunk-complete")} for job in OLD]
        value = {"schema": SCHEMA, "root": str(root), "created_at": q.utc(), "registration": str(registration),
            "parent_campaign": pin(parent_path), "original_validator": validator, "original_stage": parent["stage"],
            "initial_status": pin(root / "initial-status.json"), "failure_receipt": pin(args.failure_receipt),
            "initial_status_lock_owner": "this-declaration-holds-original-driver-lock",
            "failure_checkpoint": str(q.safe_path(args.failure_checkpoint)), "original_prefix": prefix,
            "original_generation_results": {job: pin(Path(stage["queue_root"]) / "jobs" / job / "result.json") for job in OLD + ["generate-005"]},
            "original_reader": reader(parent["data_root"], "context-v3-r1"),
            "metadata_reader": reader(args.metadata_data_root, "context-v3-metadata-r2"),
            "pipeline_files": records(ROOT, set(parent["pipeline_files"]) | set(CODE)),
            "driver_runtime": old.driver_identity(), "heavy_lock": stage["heavy_lock"],
            "original_completed_chunks": OLD, "continuation_chunks": NEW, "new_generation_jobs": GEN,
            "read_only_generation_jobs": ["generate-005"], "max_new_tts_calls": 32, "max_new_asr_calls": 120,
            "max_total_tts_calls": 72, "max_total_asr_calls": 216, "total_actual_runner_minutes": 180,
            "new_runner_allocation_minutes": 90, "driver_wall_seconds": 5400, "worker_seconds": 570,
            "job_minutes": 10, "automatic_retries": 0, "model_concurrency": 1,
            "pending_intent_restart_allowed": False, "human_gold": False}
        static(value)
        q.write_json(root / "handoff.json", value)
        q.write_json(registration, {"schema": SCHEMA, "parent_campaign": value["parent_campaign"],
                                   "handoff": pin(root / "handoff.json")})
    return {"handoff": pin(root / "handoff.json"), "new_model_calls": 0, "new_generation_jobs": GEN,
            "asr_only_existing_native_job": "generate-005", "new_runner_allocation_minutes": 90}


def budget(value, old, parent):
    original = old.budget(parent)
    records_ = []
    calls = results = 0
    for job in NEW:
        path = where(value, job)
        if path["queue"].exists():
            report = check_asr_queue(value, parent, job)
            records_ += [(job + "/asr", state) for state in report["jobs"].values()]
            calls += len(list((path["asr"] / "claims").glob("*/*.json")))
            results += len(list((path["asr"] / "results").glob("*/*.json")))
    elapsed = original["actual_known_job_nanoseconds"] + sum(r.get("actual_elapsed_ns") or 0 for _, r in records_)
    unresolved = original["unresolved_jobs"] + [name for name, r in records_ if r["status"] in ("unknown", "running")]
    failed = original["failed_jobs"] + [name for name, r in records_ if r["status"] in ("failed", "timed_out", "unknown")]
    q.require(40 <= original["tts_claims_consumed"] <= 72 and calls <= 120 and results <= calls and
              original["asr_claims_consumed"] == 71 and original["asr_results_saved"] == 71,
              "V3_HANDOFF_CALL_ROSTER_CHANGED")
    additional = elapsed - read(value["initial_status"]["path"])["budget"]["actual_known_job_nanoseconds"]
    q.require(additional >= 0, "V3_HANDOFF_ORIGINAL_TIME_WAS_REMOVED")
    return {"actual_known_job_nanoseconds": elapsed, "actual_known_job_minutes": elapsed / q.NS_PER_MINUTE,
        "new_actual_known_job_nanoseconds": additional,
        "new_remaining_nanoseconds": max(0, q.minutes_ns(90) - additional),
        "new_overrun_nanoseconds": max(0, additional - q.minutes_ns(90)),
        "remaining_nanoseconds": max(0, q.minutes_ns(180) - elapsed), "overrun_nanoseconds": max(0, elapsed - q.minutes_ns(180)),
        "tts_claims_consumed": original["tts_claims_consumed"], "native_return_receipts": original["native_return_receipts"],
        "asr_claims_consumed": 71 + calls, "asr_results_saved": 71 + results,
        "new_tts_claims": original["tts_claims_consumed"] - 40, "new_asr_claims": calls,
        "unresolved_jobs": unresolved, "failed_jobs": failed, "actual_total_is_complete": not unresolved}


def ensure_launch(value, old, parent, deadline):
    current = budget(value, old, parent)
    q.require(not current["unresolved_jobs"] and not current["failed_jobs"] and
              current["remaining_nanoseconds"] >= q.minutes_ns(10) and
              current["new_remaining_nanoseconds"] >= q.minutes_ns(10) and
              time.monotonic() + 600 <= deadline, "V3_HANDOFF_LAUNCH_BUDGET_OR_TERMINAL_STATE")
    return current


def metadata_files(folder):
    folder = Path(folder)
    return [folder / name for name in ("capsule.json", "phase-marker.json", "provider-save.json", "ack.json")]


def sources(value, old, parent, stage, job=None):
    root = Path(value["root"])
    paths = [root / "handoff.json", Path(value["registration"]), Path(value["initial_status"]["path"]),
             Path(value["parent_campaign"]["path"]), Path(value["original_validator"]["path"]),
             Path(value["failure_receipt"]["path"]), *metadata_files(value["failure_checkpoint"])]
    paths += [ROOT / name for name in value["pipeline_files"]]
    paths += [old.ROOT / name for name in parent["pipeline_files"]]
    for kind in ("original_reader", "metadata_reader"):
        paths += [Path(value[kind]["root"]) / name for name in value[kind]["source_files_sha256"]]
    for item in value["original_prefix"]:
        paths += [Path(item["completion"]["path"]), Path(item["weak_manifest"]["path"]), *metadata_files(item["checkpoint"])]
    paths += [Path(p["path"]) for p in value["original_generation_results"].values()]
    if job is not None:
        paths += old.generation.checkpoint_sources(stage, Path(stage["prepared"]), job)
        current = where(value, job)["chunk"]
        if current.exists(): paths += cp.regular_tree(current)
    for ack in sorted((root / "checkpoints").glob("*/*/ack.json")):
        paths += metadata_files(ack.parent)
    for ack in sorted((root / "checkpoints").glob("*/ack.json")):
        paths += metadata_files(ack.parent)
    return paths


def phase_scope(value, parent, job, name):
    root = Path(value["root"])
    path = where(value, job)
    scope = {"schema": SCHEMA, "phase": name, "handoff": pin(root / "handoff.json"),
        "parent_campaign": value["parent_campaign"], "job_id": job, "stage": parent["stage"],
        "reader": value["metadata_reader"], "max_asr_calls": 24, "automatic_retries": 0,
        "pending_intent_restart_allowed": False}
    if path["request"].exists():
        scope.update(asr_request=pin(path["request"]), asr_queue=pin(path["queue"] / "queue.json"))
    return scope


def phase(gate, value, old, parent, stage, job, name, *, terminal_state=None):
    root, path = Path(value["root"]), where(value, job)
    scope = phase_scope(value, parent, job, name)
    if name.startswith("asr-"):
        report = q.inspect(path["queue"], reconcile=True)
        state = report["jobs"][ASR_JOB]["status"]
        q.require(state == terminal_state and state not in ("pending", "running"), "V3_HANDOFF_ASR_TERMINAL_REQUIRED")
        scope.update(terminal_state=state, result=pin(path["queue"] / "jobs" / ASR_JOB / "result.json"))
    if name == "chunk-complete": scope["completion"] = pin(path["complete"])
    return gate.checkpoint(root / "checkpoints" / job / name, scope,
                          sources(value, old, parent, stage, job), intent=name == "before-asr")


def asr_worker(handoff, job, request):
    value = static(read(handoff)); parent = read(value["parent_campaign"]["path"])
    q.require(job in NEW and Path(handoff) == Path(value["root"]) / "handoff.json" and
              str(Path(sys.executable).absolute()) == parent["asr_python"], "V3_HANDOFF_ASR_WORKER_SCOPE")
    old = original_module(value["original_validator"])
    dependency_provenance(old, parent)
    old.validate_static(parent)
    check_asr_queue(value, parent, job)
    folder = Path(value["root"]) / "checkpoints" / job / "before-asr"
    q.require(os.environ.get(ASR_INTENT_ENV) == str(folder) and os.environ.get(ASR_OWNER_ENV),
              "V3_HANDOFF_ASR_LIVE_OWNER_REQUIRED")
    refresh_intent(folder, value, old, parent, read(parent["stage"]["path"]), job,
                   "before-asr", os.environ[ASR_OWNER_ENV])
    path = where(value, job)
    q.require(Path(request).absolute() == path["queue"] / "recipes" / (ASR_JOB + ".json") and
              q.file_record(request) == q.file_record(path["request"]) and
              os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == q.file_record(request)["sha256"] and
              os.environ.get("KWS_RESEARCH_JOB_ID") == ASR_JOB, "V3_HANDOFF_ASR_RECIPE_BINDING")
    return old.selective.run(Path(request))


def queued_attempt(root, value, old, parent, deadline, expected_job, refresh):
    """Keep one live intent while a no-claim shared-slot wait resolves."""
    wait_started = time.monotonic()
    while True:
        ensure_launch(value, old, parent, deadline)
        refresh()
        try:
            report, code = q.run(root, max_jobs=1)
        except BlockingIOError:
            report = q.inspect(root, reconcile=True); report["status"] = "waiting_for_heavy_model_slot"; code = 75
        if report["status"] != "waiting_for_heavy_model_slot": return report, code
        q.require(report["jobs"][expected_job]["status"] == "pending" and
                  not (Path(root) / "jobs" / expected_job / "claim.json").exists(), "V3_HANDOFF_WAIT_ALREADY_CONSUMED")
        q.require(time.monotonic() - wait_started < 60, "V3_HANDOFF_SHARED_SLOT_WAIT_EXHAUSTED")
        time.sleep(.2)


def generation_scope(value, old, parent, stage, job, name, terminal_state=None):
    recipe_path = Path(stage["queue_root"]) / "recipes" / (job + ".json")
    recipe = read(recipe_path)
    old.generation.validate_job_recipe(recipe, stage, parent["stage"]["path"])
    scope = old.generation.checkpoint_scope(parent["stage"]["path"], stage, recipe_path, recipe, name)
    if name != "before-generation":
        report = q.inspect(stage["queue_root"], reconcile=True)
        state = report["jobs"][job]["status"]
        q.require(state == terminal_state and state not in ("pending", "running"), "V3_HANDOFF_GENERATION_TERMINAL_REQUIRED")
        scope.update(queue_result_status=state, queue_result=pin(Path(stage["queue_root"]) / "jobs" / job / "result.json"))
    return scope


def generation_phase(gate, value, old, parent, stage, job, name, terminal_state=None):
    scope = generation_scope(value, old, parent, stage, job, name, terminal_state)
    return gate.checkpoint(Path(stage["prepared"]) / "checkpoints" / job / name, scope,
                          sources(value, old, parent, stage, job), intent=name == "before-generation")


def refresh_intent(folder, value, old, parent, stage, job, name, owner):
    expected = (generation_scope(value, old, parent, stage, job, name) if name == "before-generation" else
                phase_scope(value, parent, job, name))
    verify_scope(folder, expected, intent=True, owner=owner)
    descriptor = cp.verify_capsule(folder, current_sources=True)
    required = [Path(value["root"]) / "handoff.json", Path(value["registration"]),
                *[ROOT / relative for relative in value["pipeline_files"]],
                *[old.ROOT / relative for relative in parent["pipeline_files"]]]
    for kind in ("original_reader", "metadata_reader"):
        required += [Path(value[kind]["root"]) / relative for relative in value[kind]["source_files_sha256"]]
    if name == "before-asr":
        path = where(value, job)
        required += [path["request"], path["queue"] / "queue.json", path["queue"] / "recipes" / (ASR_JOB + ".json")]
    else:
        required += [Path(parent["stage"]["path"]), Path(stage["queue_root"]) / "recipes" / (job + ".json")]
    with zipfile.ZipFile(descriptor["capsule"]["path"]) as archive:
        members = q.loads(archive.read("MANIFEST.json"))["files"]
    retained = {item["path"]: item for item in members.values()}
    q.require(all(retained.get(str(path)) == pin(path) for path in required),
              "V3_HANDOFF_INTENT_SOURCE_MEMBERSHIP")
    return descriptor


def metadata_loader(value):
    item = value["metadata_reader"]
    q.require(item == reader(item["root"], "context-v3-metadata-r2"), "V3_HANDOFF_METADATA_READER_CHANGED")
    path = Path(item["root"]) / ENTRY
    for name, module in list(sys.modules.items()):
        if name in ("active_context_v3_metadata_weak_view", "active_context_v3_weak_view", "active_asr_weak_view",
                    "context_supplement_v3", "context_supplement_v2", "context_supplement", "d2_pool",
                    "bulk_pool", "bulk_runner", "runner", "campaign") or name == "kws_data_contract" or name.startswith("kws_data_contract."):
            origin = getattr(module, "__file__", None)
            q.require(origin and Path(origin).resolve().is_relative_to(Path(item["root"])),
                      "V3_HANDOFF_MIXED_METADATA_MODULES")
    spec = importlib.util.spec_from_file_location("active_context_v3_metadata_weak_view", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    q.require(tuple(module.PRODUCER_FILES) == DATA_FILES and
              module.SCHEMA == "kws-active-context-v3-metadata-research-weak-view-v1" and
              module.SOURCE_ADAPTER == "kws-active-context-v3-metadata-native-source-v1" and
              item == reader(item["root"], "context-v3-metadata-r2"), "V3_HANDOFF_METADATA_READER_CONTRACT")
    return module


def packages(value, upto=None):
    result = [{"path": str(Path(item["weak_manifest"]["path"]).parent),
               "manifest_sha256": item["weak_manifest"]["sha256"], "reader": value["original_reader"]}
              for item in value["original_prefix"]]
    for job in NEW:
        path = where(value, job)
        if path["view"].exists():
            result.append({"path": str(path["view"]), "manifest_sha256": q.file_record(path["view"] / "MANIFEST.json")["sha256"],
                           "reader": value["metadata_reader"]})
        if job == upto: break
    return result


def pure_worker(handoff, job, task):
    # No original generation producer is imported in this fresh process.
    value = static(read(handoff)); parent = read(value["parent_campaign"]["path"])
    q.require(job in NEW and Path(handoff) == Path(value["root"]) / "handoff.json", "V3_HANDOFF_PURE_WORKER_SCOPE")
    path = where(value, job); exported = read(path["export"] / "export.json")
    options = {"plan_path": exported["plan"], "plan_sha256": exported["plan_sha256"],
        "generation_root": exported["generation_root"], "native_root": exported["native_root"],
        "job_recipe_path": exported["job_recipe"]["path"], "job_recipe_sha256": exported["job_recipe"]["sha256"],
        "stage_path": parent["stage"]["path"]}
    if task == "projection":
        import research_context_v3_handoff_inputs as mixed
        roster = packages(value, job)
        rows, report = mixed.aggregate(roster, expected_plan_sha256=parent["plan_sha256"],
            vocabulary_path=parent["vocabulary"]["path"], vocabulary_sha256=parent["vocabulary"]["sha256"],
            original_tokens_path=parent["fsmn_tokens"]["path"])
        bodies = {"packages.json": q.canonical(roster), "report.json": q.canonical(report),
                  "rows.jsonl": b"".join(q.canonical(row) for row in rows)}
        manifest = {name: q.sha(raw) for name, raw in bodies.items()}
        if path["projection"].exists():
            q.require(read(path["projection"] / "MANIFEST.json") == manifest and
                      {p.name for p in path["projection"].iterdir()} == set(bodies) | {"MANIFEST.json"} and
                      all((path["projection"] / name).read_bytes() == raw for name, raw in bodies.items()),
                      "V3_HANDOFF_PROJECTION_CHANGED")
        else:
            path["projection"].mkdir()
            for name, raw in bodies.items(): q.write_new(path["projection"] / name, raw)
            q.write_json(path["projection"] / "MANIFEST.json", manifest)
        return {"task": task, "manifest": pin(path["projection"] / "MANIFEST.json"), "report": report, "new_model_calls": 0}
    data = metadata_loader(value)
    if task == "source-map":
        expected = data.build_source_map(path["inputs"], q.file_record(path["inputs"])["sha256"],
            path["export"] / "source-original.zip", **options)
        once(path["source_map"], expected)
        return {"task": task, "source_map": pin(path["source_map"]), "new_model_calls": 0}
    q.require(task == "weak-view", "V3_HANDOFF_PURE_TASK")
    final = path["asr"] / "audit/final-summary.json"
    if not path["view"].exists():
        data.create_view(path["inputs"], q.file_record(path["inputs"])["sha256"], path["export"] / "source-original.zip",
            path["asr"], path["view"], fsmn_tokens=parent["fsmn_tokens"]["path"],
            selective_source_map=path["source_map"], selective_policy=parent["policy"]["path"],
            selective_audit_sha256=q.file_record(final)["sha256"], **options)
    report = data.verify_view(path["view"])
    owner = report["source_adapter"]["job_owner"]
    q.require(report["inputs_sha256"] == q.file_record(path["inputs"])["sha256"] and
        report["source_archive_sha256"] == exported["archive"]["sha256"] and
        report["asr_adapter"]["final_audit_sha256"] == q.file_record(final)["sha256"] and
        report["source_adapter"]["plan_sha256"] == parent["plan_sha256"] and
        owner["job_recipe_sha256"] == exported["job_recipe"]["sha256"] and
        owner["stage_sha256"] == parent["stage"]["sha256"] and owner["job_id"] == job and
        report["producer"]["source_files_sha256"] == value["metadata_reader"]["source_files_sha256"],
        "V3_HANDOFF_WEAK_VIEW_SCOPE_CHANGED")
    return {"task": task, "manifest": pin(path["view"] / "MANIFEST.json"),
            "statistics": report["statistics"], "new_model_calls": 0}


def pure(value, job, task):
    path = where(value, job)
    command = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_context_v3_metadata_handoff.py"),
               "worker", "--handoff", str(Path(value["root"]) / "handoff.json"), "--job-id", job, "--task", task]
    result = subprocess.run(command, capture_output=True, timeout=300)
    q.require(result.returncode == 0, "V3_HANDOFF_NONMODEL_FAILED:" + task + ":" +
              result.stderr.decode(errors="replace")[-2000:] + result.stdout.decode(errors="replace")[-2000:])
    receipt = q.loads(result.stdout)
    q.require(receipt["task"] == task and receipt["new_model_calls"] == 0, "V3_HANDOFF_PURE_RECEIPT")
    once(path["chunk"] / (task + "-replay.json"), receipt)
    return receipt


def seal_chunk(value, job, exported):
    path = where(value, job)
    q.require(path["projection"].is_dir() and bool(exported["included"]) == path["view"].is_dir(),
              "V3_HANDOFF_COMPLETE_OBSERVED_EXPORT_REQUIRED")
    report = read(path["view"] / "report.json") if exported["included"] else None
    value_ = {"schema": CHUNK, "status": "completed" if report else "completed_no_asr_eligible_return",
        "handoff": pin(Path(value["root"]) / "handoff.json"),
        "job_id": job, "generation_result": exported["job_result"],
        "native_included": len(exported["included"]), "native_excluded": exported["excluded"],
        "asr_result": pin(path["queue"] / "jobs" / ASR_JOB / "result.json") if report else None,
        "reader": value["metadata_reader"], "statistics": report["statistics"] if report else None,
        "cumulative_projection": read(path["projection"] / "report.json"),
        "files": {str(p.relative_to(path["chunk"])): q.file_record(p) for p in cp.regular_tree(path["chunk"]) if p != path["complete"]},
        "human_gold": False, "automatic_retries": 0}
    once(path["complete"], value_)
    return value_


def check_chunk(value, job):
    path = where(value, job); receipt = read(path["complete"])
    q.require(receipt["schema"] == CHUNK and receipt["status"] in ("completed", "completed_no_asr_eligible_return") and receipt["job_id"] == job and
              receipt["handoff"] == pin(Path(value["root"]) / "handoff.json") and
              receipt["reader"] == value["metadata_reader"] and receipt["automatic_retries"] == 0 and receipt["human_gold"] is False,
              "V3_HANDOFF_CHUNK_SCOPE_CHANGED")
    actual = {str(p.relative_to(path["chunk"])): q.file_record(p) for p in cp.regular_tree(path["chunk"]) if p != path["complete"]}
    q.require(actual == receipt["files"], "V3_HANDOFF_CHUNK_FILES_CHANGED")
    check_pin(receipt["generation_result"])
    if receipt["asr_result"] is not None: check_pin(receipt["asr_result"])
    return receipt


def check_chunk_closure(value, parent, job):
    receipt = check_chunk(value, job)
    verify_scope(Path(value["root"]) / "checkpoints" / job / "chunk-complete",
                 {**phase_scope(value, parent, job, "chunk-complete"),
                  "completion": pin(where(value, job)["complete"])})
    return receipt


def final_scope(value):
    root = Path(value["root"])
    return {"schema": SCHEMA, "phase": "handoff-complete", "handoff": pin(root / "handoff.json"),
            "completion": pin(root / "complete.json")}


def persist_exception(gate, value, old, parent, stage, job, current_phase, error):
    """Reconcile a claimed interrupted phase before saving the owner's stop.

    A queue claim is not a successful model return. It is nevertheless consumed
    and its authenticated result, including unknown duration, must survive.
    Pure derivation failures remain separately classified after prior models.
    """
    root, path = Path(value["root"]), where(value, job)
    kind = "generation" if current_phase in ("before-generation", "generation", "generation-complete") else (
           "asr" if current_phase in ("before-asr", "asr", "asr-complete") else None)
    model = None
    if kind:
        queue_root = Path(stage["queue_root"]) if kind == "generation" else path["queue"]
        identifier = job if kind == "generation" else ASR_JOB
        report = q.inspect(queue_root, reconcile=True)
        state = report["jobs"][identifier]["status"]
        folder = queue_root / "jobs" / identifier
        claim = pin(folder / "claim.json") if (folder / "claim.json").exists() else None
        result = pin(folder / "result.json") if (folder / "result.json").exists() else None
        model = {"kind": kind, "queue": str(queue_root), "job_id": identifier, "status": state,
                 "claim": claim, "result": result, "retry_allowed": False}
        if claim is not None and state not in ("pending", "running"):
            q.require(result is not None, "V3_HANDOFF_INTERRUPTED_TERMINAL_RECEIPT_REQUIRED")
            if kind == "generation":
                name = "generation-complete" if state == "completed" else "generation-terminal"
                generation_phase(gate, value, old, parent, stage, job, name, state)
            else:
                name = "asr-complete" if state == "completed" else "asr-terminal"
                phase(gate, value, old, parent, stage, job, name, terminal_state=state)
    classified_model = model is not None and model["claim"] is not None
    failure = root / (job + "-failure.json")
    once(failure, {"schema": SCHEMA, "status": "stopped", "job_id": job, "phase": current_phase,
        "classification": "model_phase_interrupted" if classified_model else "nonmodel_or_unclaimed_phase_failure",
        "model_phase": model, "budget": budget(value, old, parent),
        "error_type": type(error).__name__, "error": str(error), "automatic_retry": False})
    name = "owner-terminal" if classified_model else "nonmodel-terminal"
    return gate.checkpoint(root / "checkpoints" / job / name, {"schema": SCHEMA,
        "phase": name, "handoff": pin(root / "handoff.json"), "failure": pin(failure)},
        sources(value, old, parent, stage, job) + [failure], intent=False)


def run(path, *, checkpoint_gate=None):
    value, old, parent, stage, _request = load(path)
    root, ctx = Path(value["root"]), context(value, parent)
    if checkpoint_gate is None:
        announced = set()
        def on_wait(folder, descriptor):
            if str(folder) not in announced:
                print(json.dumps({"event": "awaiting_durable_save", "checkpoint": str(folder), "capsule": descriptor["capsule"]}), flush=True)
                announced.add(str(folder))
        checkpoint_gate = cp.Gate(on_wait=on_wait)
    gate = checkpoint_gate
    job, current_phase = None, "handoff-declared"
    with q.exclusive(old.driver_lock(parent["stage"]["path"])):
        deadline = time.monotonic() + value["driver_wall_seconds"]
        gate.checkpoint(root / "checkpoints/handoff-declared", {"schema": SCHEMA, "phase": "handoff-declared",
            "handoff": pin(path), "new_model_calls": 0}, sources(value, old, parent, stage), intent=False)
        try:
            for job in NEW:
                current_phase = "validate-prefix"
                for previous in NEW[:NEW.index(job)]:
                    check_chunk_closure(value, parent, previous)
                p = where(value, job)
                if p["complete"].exists():
                    check_chunk_closure(value, parent, job)
                    continue
                generated = q.inspect(stage["queue_root"], reconcile=True)
                state = generated["jobs"][job]["status"]
                if state == "running": return {"status": "running", "job_id": job}, 75
                if state == "pending":
                    q.require(job in GEN, "V3_HANDOFF_GENERATION5_READ_ONLY")
                    current_phase = "before-generation"
                    ensure_launch(value, old, parent, deadline)
                    generation_phase(gate, value, old, parent, stage, job, "before-generation")
                    ensure_launch(value, old, parent, deadline)
                    folder = Path(stage["prepared"]) / "checkpoints" / job / "before-generation"
                    q.require(old.generation.INTENT_OWNER_ENV not in os.environ and old.generation.INTENT_FOLDER_ENV not in os.environ,
                              "V3_HANDOFF_OWNER_ENV_CONFLICT")
                    os.environ.update({old.generation.INTENT_OWNER_ENV: gate.owner, old.generation.INTENT_FOLDER_ENV: str(folder)})
                    try:
                        current_phase = "generation"
                        _generated, _code = queued_attempt(stage["queue_root"], value, old, parent, deadline, job,
                            lambda: refresh_intent(folder, value, old, parent, stage, job, "before-generation", gate.owner))
                    finally:
                        os.environ.pop(old.generation.INTENT_OWNER_ENV, None); os.environ.pop(old.generation.INTENT_FOLDER_ENV, None)
                    generated = q.inspect(stage["queue_root"], reconcile=True); state = generated["jobs"][job]["status"]
                if state in ("pending", "running"): return {"status": state, "job_id": job}, 75
                if state != "completed":
                    generation_phase(gate, value, old, parent, stage, job, "generation-terminal", state)
                    return {"status": state, "job_id": job, "automatic_retry": False}, 1
                current_phase = "generation-complete"
                if job == "generate-005":
                    verify_scope(Path(stage["prepared"]) / "checkpoints" / job / "generation-complete",
                                 generation_scope(value, old, parent, stage, job, "generation-complete", "completed"))
                    check_pin(value["original_generation_results"][job])
                else:
                    generation_phase(gate, value, old, parent, stage, job, "generation-complete", "completed")
                current_phase = "export-and-inputs"
                ensure_chunk_parent(root, job)
                exported = old.ensure_export(ctx, job)
                if not exported["included"]:
                    q.require(not any(p[key].exists() for key in ("inputs", "source_map", "request", "queue", "view")),
                              "V3_HANDOFF_EMPTY_EXPORT_HAS_ASR_STATE")
                    current_phase = "projection"
                    pure(value, job, "projection")
                    receipt = seal_chunk(value, job, exported)
                    phase(gate, value, old, parent, stage, job, "chunk-complete")
                    print(json.dumps({"event": "chunk_durably_closed", "job_id": job,
                        "native_included": 0, "unobserved_selected_cells_retained": True}), flush=True)
                    continue
                inputs = old.ensure_inputs(ctx, job, exported)
                current_phase = "source-map"
                pure(value, job, "source-map")
                old.ensure_asr_request(ctx, job, inputs, p["source_map"])
                asr = ensure_asr_queue(value, parent, job); state = asr["jobs"][ASR_JOB]["status"]
                if state == "running": return {"status": "running", "job_id": job}, 75
                if state == "pending":
                    current_phase = "before-asr"
                    ensure_launch(value, old, parent, deadline)
                    phase(gate, value, old, parent, stage, job, "before-asr")
                    ensure_launch(value, old, parent, deadline)
                    current_phase = "asr"
                    folder = root / "checkpoints" / job / "before-asr"
                    q.require(ASR_OWNER_ENV not in os.environ and ASR_INTENT_ENV not in os.environ,
                              "V3_HANDOFF_ASR_OWNER_ENV_CONFLICT")
                    os.environ.update({ASR_OWNER_ENV: gate.owner, ASR_INTENT_ENV: str(folder)})
                    try:
                        _asr, _code = queued_attempt(p["queue"], value, old, parent, deadline, ASR_JOB,
                            lambda: refresh_intent(folder, value, old, parent, stage, job, "before-asr", gate.owner))
                    finally:
                        os.environ.pop(ASR_OWNER_ENV, None); os.environ.pop(ASR_INTENT_ENV, None)
                    asr = check_asr_queue(value, parent, job); state = asr["jobs"][ASR_JOB]["status"]
                if state in ("pending", "running"): return {"status": state, "job_id": job}, 75
                if state != "completed":
                    phase(gate, value, old, parent, stage, job, "asr-terminal", terminal_state=state)
                    return {"status": state, "job_id": job, "automatic_retry": False}, 1
                current_phase = "asr-complete"
                phase(gate, value, old, parent, stage, job, "asr-complete", terminal_state="completed")
                audit = old.selective.seal_or_verify(p["request"])
                q.require(audit["state"]["status"] == "complete", "V3_HANDOFF_ASR_INCOMPLETE")
                current_phase = "weak-view"
                pure(value, job, "weak-view")
                current_phase = "projection"
                pure(value, job, "projection")
                receipt = seal_chunk(value, job, exported)
                phase(gate, value, old, parent, stage, job, "chunk-complete")
                print(json.dumps({"event": "chunk_durably_closed", "job_id": job, "statistics": receipt["statistics"],
                                  "projection": receipt["cumulative_projection"]}, ensure_ascii=False), flush=True)
            final = {"schema": SCHEMA, "status": "completed", "handoff": pin(path), "completed_chunks": OLD + NEW,
                "budget": budget(value, old, parent), "packages": packages(value), "human_gold": False,
                "automatic_retries": 0, "original_prefix_modified": False}
            q.require(not final["budget"]["failed_jobs"] and not final["budget"]["unresolved_jobs"] and
                      not final["budget"]["overrun_nanoseconds"] and not final["budget"]["new_overrun_nanoseconds"],
                      "V3_HANDOFF_FINAL_BUDGET")
            once(root / "complete.json", final)
            gate.checkpoint(root / "checkpoints/handoff-complete", final_scope(value),
                sources(value, old, parent, stage, NEW[-1]) + [root / "complete.json"], intent=False)
            return final, 0
        except BaseException as error:
            if job is not None:
                persist_exception(gate, value, old, parent, stage, job, current_phase, error)
            raise


def status(path):
    value, old, parent, stage, _request = load(path)
    completed = list(OLD)
    for job in NEW:
        if where(value, job)["complete"].exists():
            check_chunk_closure(value, parent, job)
            completed.append(job)
    result = {"schema": SCHEMA, "handoff": pin(path), "completed_chunks": completed,
        "driver_active": supervisor.lock_active(old.driver_lock(parent["stage"]["path"])),
        "budget": budget(value, old, parent), "packages": packages(value), "new_model_calls": 0}
    final = Path(value["root"]) / "complete.json"
    result["completion"] = pin(final) if final.exists() else None
    result["completion_checkpoint"] = str(Path(value["root"]) / "checkpoints/handoff-complete") if final.exists() else None
    if final.exists():
        receipt = read(final)
        q.require(receipt["schema"] == SCHEMA and receipt["status"] == "completed" and
            receipt["handoff"] == pin(path) and receipt["completed_chunks"] == completed == OLD + NEW and
            receipt["packages"] == result["packages"] and receipt["budget"] == result["budget"] and
            result["budget"]["failed_jobs"] == result["budget"]["unresolved_jobs"] == [] and
            result["budget"]["overrun_nanoseconds"] == result["budget"]["new_overrun_nanoseconds"] == 0,
            "V3_HANDOFF_FINAL_RECEIPT_CHANGED")
        verify_scope(result["completion_checkpoint"], final_scope(value))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("declare")
    for name in ("campaign", "original-validator", "metadata-data-root", "failure-receipt", "failure-checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("run", "status"):
        p = commands.add_parser(name); p.add_argument("--handoff", type=Path, required=True)
    p = commands.add_parser("worker"); p.add_argument("--handoff", type=Path, required=True)
    p.add_argument("--job-id", required=True); p.add_argument("--task", choices=("source-map", "weak-view", "projection"), required=True)
    p = commands.add_parser("asr-worker")
    for name in ("handoff", "request"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "declare": result, code = declare(args), 0
        elif args.command == "run": result, code = run(args.handoff)
        elif args.command == "status": result, code = status(args.handoff), 0
        elif args.command == "worker": result, code = pure_worker(args.handoff, args.job_id, args.task), 0
        else:
            result = asr_worker(args.handoff, args.job_id, args.request)
            code = 0 if result["state"]["status"] == "complete" else 1
    except BlockingIOError:
        result, code = {"status": "another_owner_or_model_active"}, 75
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        result, code = {"status": "invalid", "reason": str(error)}, 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return code


if __name__ == "__main__":
    raise SystemExit(main())
