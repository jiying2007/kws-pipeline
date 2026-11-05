#!/usr/bin/env python3
"""One bounded continuation of a sealed, timed-out native D1 observation job.

The original 64 and 128 returned observations are reused without another call.
The consumed native claim without a return remains prediction-unknown. Only
the exact 62 never-claimed inputs may receive calls, under a new 180 s budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training"), str(ROOT)]
import run_d1_zipformer_extension as ext

previous, io, read, require = ext.previous, ext.io, ext.read, ext.require
SCHEMA = "kws-d1-zipformer-continuation-job-v4"
RESULT_SCHEMA = "kws-d1-zipformer-continuation-result-v4"
SOURCE_PATH = "tools/run_d1_zipformer_continuation.py"
QUEUE_OUTPUTS = ("summary.json", "observations.jsonl", "unavailable-predictions.jsonl", "MANIFEST.json")
NEW_LIMIT_NS = 180_000_000_000


class NoML:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in previous.ML_MODULES:
            raise RuntimeError("CONTINUATION_READ_ONLY_FORBIDS_ML:" + fullname)


def readonly_guard():
    require(not any(name in sys.modules for name in previous.ML_MODULES), "CONTINUATION_READ_ONLY_ML_ALREADY_IMPORTED")
    previous.research_no_network.set_offline_environment()
    return previous.research_no_network.install()


def source_records():
    return {**ext.source_records(), SOURCE_PATH: previous.training.sha_file(ROOT / SOURCE_PATH)}


def pin(path):
    return {"path": str(io.safe_path(path)), **io.file_record(path)}


def pinned(value):
    require(type(value) is dict and set(value) == {"path", "sha256"} and
        type(value["path"]) is str and Path(value["path"]).is_absolute() and
        type(value["sha256"]) is str and io.SHA.fullmatch(value["sha256"]), "CONTINUATION_EVIDENCE_PIN")
    path = io.safe_path(value["path"])
    require(io.file_record(path)["sha256"] == value["sha256"], "CONTINUATION_EVIDENCE_CHANGED")
    return path


def validate_job(job):
    require(type(job) is dict and set(job) == {"schema", "failed_native", "partial_inventory",
        "partial_accounting", "pending_source_ids", "source_files", "timeout_seconds", "maximum_attempts"}
        and job["schema"] == SCHEMA, "CONTINUATION_JOB_SCHEMA")
    failed = job["failed_native"]
    require(type(failed) is dict and set(failed) == {"queue_root", "job_id", "receipt_sha256"} and
        type(failed["queue_root"]) is str and Path(failed["queue_root"]).is_absolute() and
        type(failed["job_id"]) is str and io.ID.fullmatch(failed["job_id"]) and
        type(failed["receipt_sha256"]) is str and io.SHA.fullmatch(failed["receipt_sha256"]),
        "CONTINUATION_REAL_FAILED_BINDING")
    ids = job["pending_source_ids"]
    require(type(ids) is list and len(ids) == len(set(ids)) == 62 and
        all(type(value) is str and previous.development.FAMILY_ID.fullmatch(value) for value in ids),
        "CONTINUATION_EXACT_62_DECLARED_INPUTS")
    require(type(job["timeout_seconds"]) is int and job["timeout_seconds"] == 180 and
        type(job["maximum_attempts"]) is int and job["maximum_attempts"] == 1 and
        job["source_files"] == source_records(), "CONTINUATION_FIXED_SOURCE_BUDGET_ATTEMPT")
    for key in ("partial_inventory", "partial_accounting"):
        pinned(job[key])
    return job


def failed_terminal(job):
    binding = job["failed_native"]
    root, queue, request, queue_sha = io.load_queue(binding["queue_root"])
    require(len(request["jobs"]) == 1 and request["jobs"][0]["job_id"] == binding["job_id"],
        "CONTINUATION_ORIGINAL_SINGLE_JOB")
    definition = request["jobs"][0]
    result = io.result_for(root, definition, queue_sha)
    require(result["status"] == "timed_out" and result["exit_code"] == -9 and
        result["receipt_sha256"] == binding["receipt_sha256"] and result["actual_duration_known"] is True and
        result["automatic_retry"] is False and result["allowed_runner_nanoseconds"] == 120_000_000_000 and
        io.minutes_ns(request["global_budget_minutes"]) == 120_000_000_000 and
        result["actual_elapsed_ns"] >= 120_000_000_000 and
        result["deadline_overrun_nanoseconds"] == result["actual_elapsed_ns"] - 120_000_000_000,
        "CONTINUATION_ACTUAL_120_SECOND_TIMEOUT_REQUIRED")
    folder = root / "jobs" / binding["job_id"]
    require(result["argv"] == io.expanded_argv(root, definition) == read(folder / "command-started.json")["argv"]
        and result["cwd"] == definition["cwd"], "CONTINUATION_ORIGINAL_COMMAND_BINDING")
    recipe_path = root / "recipes" / (binding["job_id"] + ".json")
    recipe = ext.validate_job(read(recipe_path))
    require(recipe["timeout_seconds"] == 120 and previous.training.sha_file(recipe_path) == result["recipe_sha256"],
        "CONTINUATION_ORIGINAL_RECIPE_CHANGED")
    return {"root": root, "queue": queue, "terminal": result, "recipe": recipe,
            "recipe_path": recipe_path, "output": folder / "outputs"}


def verify_inventory(path, output, terminal_path):
    inventory = read(path)
    require(inventory["schema"] == "d1-native-partial-output-inventory-v1" and
        inventory["root"] == str(output) and inventory["terminal"] == pin(terminal_path),
        "CONTINUATION_PARTIAL_INVENTORY_BINDING")
    files = inventory["files"]
    require(type(files) is dict and type(inventory["file_count"]) is int and set(files) == {p.relative_to(output).as_posix()
        for p in output.rglob("*") if p.is_file()} and inventory["file_count"] == len(files) and
        not (output / "summary.json").exists() and not (output / "MANIFEST.json").exists(),
        "CONTINUATION_PARTIAL_EXACT_ROSTER")
    for name, value in files.items():
        require(io.file_record(output / io.relative(name)) == value, "CONTINUATION_PARTIAL_BYTES_CHANGED:" + name)
    require(type(inventory["total_bytes"]) is int and inventory["total_bytes"] == sum(value["bytes"] for value in files.values()) and
        all(type(inventory[key]) is int and inventory[key] == 0 for key in ("ml_imports", "new_model_calls")) and
        inventory["verification_scope"] == "byte_inventory_and_roster_only",
        "CONTINUATION_PARTIAL_BYTE_COUNT")
    return inventory


def partition_partial(output, rows, identity_sha, declared_pending):
    """Authenticate the actual sequential producer prefix, never select by labels."""
    require(len(rows) == 191 and len({row["source_id"] for row in rows}) == 191, "CONTINUATION_ORIGINAL_191")
    claims = {path.name for path in (output / "new/claims").iterdir()}
    returned = {path.name for path in (output / "new/native").iterdir()}
    require(claims == {row["source_id"] + ".json" for row in rows[:129]} and
        returned == {row["source_id"] + ".json" for row in rows[:128]}, "CONTINUATION_129_CLAIMS_128_RETURNS_PREFIX")
    observations, seconds = [], 0.0
    for index, row in enumerate(rows[:129]):
        path = output / "new/claims" / (row["source_id"] + ".json")
        claim = read(path)
        require(claim.get("schema") == 1 and claim.get("attempt") == 1 and
            claim.get("identity_sha256") == identity_sha and claim.get("input") == previous.input_binding(row),
            "CONTINUATION_ORIGINAL_CELL_CLAIM")
        if index < 128:
            result_path = output / "new/native" / path.name
            value = previous.native.cached_result(result_path, identity_sha, previous.input_binding(row))
            require(value is not None and type(value["inference_seconds"]) in (int, float) and
                math.isfinite(value["inference_seconds"]) and value["inference_seconds"] >= 0,
                "CONTINUATION_ORIGINAL_NATIVE_RETURN")
            seconds += value["inference_seconds"]
            observations.append(previous.observed_row(row, value, previous.training.sha_file(result_path)))
    actual = [io.loads(line) for line in io.read_bytes(output / "new/observations.jsonl", 64 * 1024 * 1024).splitlines()]
    require(observations == actual, "CONTINUATION_ORIGINAL_OBSERVATIONS_CHANGED")
    pending = rows[129:]
    require([row["source_id"] for row in pending] == declared_pending and
        not set(declared_pending) & {row["source_id"] for row in rows[:129]}, "CONTINUATION_PENDING_NOT_EXACT_UNCLAIMED_COMPLEMENT")
    return {"returned_rows": rows[:128], "observations": observations, "unknown_rows": rows[128:129],
            "pending_rows": pending, "original_inference_seconds": seconds}


def _verify_origin(job, deadline):
    origin = failed_terminal(job)
    output, recipe = origin["output"], origin["recipe"]
    inventory_path = pinned(job["partial_inventory"])
    verify_inventory(inventory_path, output, output.parent / "result.json")
    old = ext.original_native(recipe, deadline)
    prepared, dependency, rows, projected_sha, closed = ext.full_rows(recipe, old, deadline)
    new_rows = ext.join_unchanged(old["rows"], rows)
    identity = ext.identity_for(recipe, origin["recipe_path"], dependency, projected_sha, rows, old, closed)
    require(read(output / "identity.json") == identity and read(output / "terminal-coverage.json") == closed and
        io.read_bytes(output / "request.json") == io.read_bytes(origin["recipe_path"]), "CONTINUATION_ORIGINAL_IDENTITY_CHANGED")
    require([io.loads(line) for line in io.read_bytes(output / "unobservable.jsonl").splitlines()] == closed["unobservable"],
        "CONTINUATION_GENERATION_UNKNOWN_CHANGED")
    copies = read(output / "coverage-evidence.json")
    require([item["original"] for item in copies] == closed["evidence_files"], "CONTINUATION_COVERAGE_BINDINGS")
    for item in copies:
        require(io.file_record(output / io.relative(item["retained"])) ==
            {key: item["original"][key] for key in ("bytes", "sha256")}, "CONTINUATION_COVERAGE_RETAINED_BYTES")
    for name in set(read(old["output"] / "MANIFEST.json")) | {"MANIFEST.json"}:
        require(io.file_record(output / "reused/original-output" / io.relative(name)) ==
            io.file_record(old["output"] / name), "CONTINUATION_PREVIOUS64_BYTES_CHANGED")
    for name, expected in recipe["source_files"].items():
        require(previous.training.sha_file(output / "execution-source" / io.relative(name)) == expected,
            "CONTINUATION_ORIGINAL_EXECUTION_SOURCE_CHANGED")
    profile = read(ROOT / previous.PROFILE_PATH)
    require(previous.training.sha_file(ROOT / previous.PROFILE_PATH) == recipe["profile_sha256"],
        "CONTINUATION_FIXED_PROFILE_CHANGED")
    for asset in profile["assets"].values():
        previous.native.checked_bytes(previous.native.bound_path(io.safe_path(recipe["models"]), asset["filename"]),
            asset["sha256"], asset["bytes"])
    keyword_bytes = ("\n".join(" ".join(row["tokens"]) + " @kw" + str(row["id"])
        for row in profile["keywords"]) + "\n").encode()
    require(io.read_bytes(output / "keywords.txt") == keyword_bytes == io.read_bytes(old["output"] / "keywords.txt"),
        "CONTINUATION_KEYWORDS_CHANGED")
    partitions = partition_partial(output, new_rows, ext.digest(identity), job["pending_source_ids"])
    accounting = read(pinned(job["partial_accounting"]))
    require(accounting["schema"] == "d1-native-partial-accounting-v1" and
        accounting["terminal"] == pin(output.parent / "result.json") and accounting["output_inventory"] == pin(inventory_path) and
        accounting["identity"] == pin(output / "identity.json") and accounting["recipe"] == pin(output / "request.json") and
        accounting["projected_rows"] == pin(prepared / "projected-rows.jsonl") and
        len(accounting["previous64_source_ids"]) == 64 and
        set(accounting["previous64_source_ids"]) == {row["source_id"] for row in old["rows"]} and
        accounting["planned_new_source_ids"] == [row["source_id"] for row in new_rows] and
        accounting["returned_source_ids"] == [row["source_id"] for row in partitions["returned_rows"]] and
        accounting["native_consumed_unknown_source_ids"] == [row["source_id"] for row in partitions["unknown_rows"]] and
        accounting["never_claimed_source_ids"] == job["pending_source_ids"] and
        accounting["generation_consumed_unknown_source_ids"] == closed["consumed_unknown_source_ids"],
        "CONTINUATION_PARTIAL_ACCOUNTING_CHANGED")
    require(all(type(accounting[key]) is int and accounting[key] == value for key, value in
        (("claimed_count", 129), ("returned_count", 128), ("unknown_count", 1), ("never_claimed_count", 62),
         ("ml_imports", 0), ("new_model_calls", 0))) and accounting["complete_native_execution"] is False,
        "CONTINUATION_PARTIAL_COUNTS_AND_FLAGS")
    require(time.monotonic() < deadline and not any(name in sys.modules for name in previous.ML_MODULES),
        "CONTINUATION_VERIFY_DEADLINE_OR_ML")
    return {**origin, **partitions, "old": old, "prepared": prepared, "rows": rows, "closed": closed,
            "identity_sha256": ext.digest(identity), "profile": profile, "keyword_bytes": keyword_bytes}


def verify_origin(job, deadline):
    guard = NoML()
    require(not any(name in sys.modules for name in previous.ML_MODULES), "CONTINUATION_VERIFY_ML_ALREADY_IMPORTED")
    sys.meta_path.insert(0, guard)
    try:
        return _verify_origin(job, deadline)
    finally:
        sys.meta_path.remove(guard)


def registration_path(job):
    binding = job["failed_native"]
    return io.safe_path(binding["queue_root"]).parent / ("native-continuation-" + binding["receipt_sha256"] + ".json")


def registration_value(job, queue, recipe_path, output):
    return {"schema": "kws-d1-native-single-continuation-registration-v1", "maximum_attempts": 1,
        "failed_native": job["failed_native"], "partial_inventory": job["partial_inventory"],
        "partial_accounting": job["partial_accounting"], "pending_source_ids": job["pending_source_ids"],
        "recipe_sha256": previous.training.sha_file(recipe_path), "queue": queue, "output": str(output),
        "source_files": job["source_files"], "new_budget_nanoseconds": NEW_LIMIT_NS, "reset_allowed": False}


def register(job, queue, recipe_path, output):
    path = registration_path(job)
    require(not path.exists(), "CONTINUATION_ALREADY_REGISTERED_NO_SECOND_ATTEMPT")
    io.write_json(path, registration_value(job, queue, recipe_path, output))
    return pin(path)


def current_queue_binding(path, output, job, lease):
    previous.training.queue_binding(path, lease)
    require(os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(output) and
        os.environ.get("KWS_RESEARCH_RECIPE") == str(path), "CONTINUATION_QUEUE_PATH_BINDING")
    root, queue, request, queue_sha = io.load_queue(output.parents[2])
    job_id = os.environ.get("KWS_RESEARCH_JOB_ID")
    require(len(request["jobs"]) == 1 and request["jobs"][0]["job_id"] == job_id and
        output == root / "jobs" / job_id / "outputs" and queue["heavy_lock"] == str(lease.path),
        "CONTINUATION_SINGLE_QUEUE_AND_HEAVY_LEASE")
    definition, claim = request["jobs"][0], read(output.parent / "claim.json")
    require(definition["depends_on"] == [] and tuple(definition["outputs"]) == QUEUE_OUTPUTS and
        io.minutes_ns(request["global_budget_minutes"]) == NEW_LIMIT_NS and
        io.minutes_ns(definition["timeout_minutes"], 999) == NEW_LIMIT_NS and
        claim["schema"] == "kws-local-research-job-claim-v1" and claim["attempt"] == 1 and claim["automatic_retry"] is False and
        all(claim.get(key) == value for key, value in io.job_binding(definition, queue_sha).items()) and
        definition["recipe_sha256"] == previous.training.sha_file(path) and
        0 < claim["allowed_runner_nanoseconds"] <= NEW_LIMIT_NS and
        claim["deadline_monotonic_ns"] == claim["started_monotonic_ns"] + claim["allowed_runner_nanoseconds"] and
        claim["dependencies_sha256"] == previous.training.sha_file(output.parent / "dependencies.json"),
        "CONTINUATION_180_SECOND_QUEUE_CLAIM")
    return {"job_id": job_id, "queue_sha256": queue_sha, "claim_sha256": previous.training.sha_file(output.parent / "claim.json"),
        "allowed_runner_nanoseconds": claim["allowed_runner_nanoseconds"], "deadline_monotonic_ns": claim["deadline_monotonic_ns"],
        "inherited_heavy_lock": lease.inherited, "supervised": lease.supervised, "pid": os.getpid(), "process_group": os.getpgrp()}


def unknown_predictions(proof):
    return [{"schema": "kws-d1-native-unavailable-prediction-v1", "prediction_status": "consumed-unknown",
        "source_id": row["source_id"], "original_projected_row": row, "label_binding_sha256": previous.row_identity(row),
        "original_claim": pin(proof["output"] / "new/claims" / (row["source_id"] + ".json")),
        "original_timeout_receipt_sha256": proof["terminal"]["receipt_sha256"], "new_native_call": False,
        "prediction_available": False, "included_in_prediction_denominator": False} for row in proof["unknown_rows"]]


def merged_predictions(proof, new_values, output, identity_sha):
    groups = (("original64", proof["old"]["observations"], proof["output"] / "reused/original-output",
               proof["old"]["identity_sha256"]),
              ("original-timeout-returned", proof["observations"], proof["output"] / "new", proof["identity_sha256"]),
              ("continuation-new", new_values, output / "new", identity_sha))
    by_id, rows = {}, {row["source_id"]: row for row in proof["rows"]}
    for origin, values, native_root, native_identity in groups:
        for value in values:
            name = value["source_id"]
            require(name not in by_id, "CONTINUATION_DUPLICATE_PREDICTION")
            native_path = native_root / "native" / (name + ".json")
            native = previous.native.cached_result(native_path, native_identity, previous.input_binding(rows[name]))
            require(native is not None and previous.observed_row(rows[name], native, previous.training.sha_file(native_path)) == value,
                "CONTINUATION_REUSED_OBSERVATION_CHANGED")
            by_id[name] = {"schema": "kws-d1-zipformer-continuation-observation-v4", "source_id": name,
                "origin": origin, "new_native_call": origin == "continuation-new", "prediction_available": True,
                "native_result": pin(native_path), "native_identity_sha256": native_identity,
                "original_observation_sha256": ext.digest(value), "observation": value}
    unknown = {row["source_id"] for row in proof["unknown_rows"]}
    require(len(by_id) == 254 and set(by_id) == set(rows) - unknown and len(unknown) == 1,
        "CONTINUATION_EXACT_AVAILABLE_PREDICTION_ROSTER")
    return [by_id[row["source_id"]] for row in proof["rows"] if row["source_id"] in by_id]


def label_counts(rows):
    return {split: {str(index + 1): sum(row["split"] == split and row["label_status"] == "machine_candidate" and
        row.get("generation_outcome") != "failed-native" and not row["quarantined"] and row["keyword_mask"][index]
        for row in rows) for index in range(2)} for split in ("train", "development")}


def counts(proof, observations):
    def reduced(values):
        return {split: previous.summarize([item["observation"] for item in values
            if item["observation"]["split"] == split]) for split in ("train", "development")}
    return {"prediction_available": reduced(observations),
        "previous64": reduced([row for row in observations if row["origin"] == "original64"]),
        "additional": reduced([row for row in observations if row["origin"] != "original64"]),
        "label_eligible_keyword_clips": label_counts(proof["rows"]),
        "native_unknown_label_eligible_keyword_clips": label_counts(proof["unknown_rows"]),
        "native_unknown_by_split": dict(Counter(row["split"] for row in proof["unknown_rows"]))}


def identity_for(job, path, proof):
    return {"schema": RESULT_SCHEMA, "recipe_sha256": previous.training.sha_file(path),
        "failed_native": job["failed_native"], "partial_inventory": job["partial_inventory"],
        "partial_accounting": job["partial_accounting"], "original_native_identity_sha256": proof["identity_sha256"],
        "pending_inputs": [previous.input_binding(row) for row in proof["pending_rows"]],
        "pending_label_bindings": [previous.row_identity(row) for row in proof["pending_rows"]],
        "source_files": job["source_files"], "profile_sha256": proof["recipe"]["profile_sha256"]}


def run(path, output, lock):
    path, output = io.safe_path(path), io.safe_path(output)
    job = validate_job(read(path))
    with previous.heavy_model_slot(lock) as lease:
        queue = current_queue_binding(path, output, job, lease)
        origin = failed_terminal(job)
        require(origin["queue"]["heavy_lock"] == str(lease.path), "CONTINUATION_SAME_GLOBAL_HEAVY_LOCK")
        output = previous.training.output_directory(output)
        started = time.monotonic()
        deadline = min(started + 180, queue["deadline_monotonic_ns"] / 1e9)
        registration = register(job, queue, path, output)
        io.write_json(output / "invocation.json", {"schema": RESULT_SCHEMA, "queue": queue,
            "registration": registration, "automatic_retry": False})
        try:
            require(not any(name in sys.modules for name in previous.ML_MODULES), "CONTINUATION_ML_BEFORE_GUARD")
            previous.research_no_network.set_offline_environment()
            io.write_json(output / "network-guard.json", previous.research_no_network.install())
            proof = verify_origin(job, deadline)
            # Runtime package metadata is checked in the actual core process.
            # The pure-stdlib verifier/auditor also works with -I -S -B.
            previous.native.check_profile(proof["profile"], io.safe_path(proof["recipe"]["models"]))
            identity = identity_for(job, path, proof)
            identity_sha = ext.digest(identity)
            io.write_json(output / "identity.json", identity)
            io.write_new(output / "keywords.txt", proof["keyword_bytes"])
            previous.training.copy_once(path, output / "request.json")
            for key in ("partial_inventory", "partial_accounting"):
                previous.training.copy_once(pinned(job[key]), output / (key + ".json"))
            for name, expected in job["source_files"].items():
                require(previous.training.copy_once(ROOT / name, output / "execution-source" / name) == expected,
                    "CONTINUATION_EXECUTION_SOURCE_CHANGED")
            for name in ("claims", "native"):
                (output / "new" / name).mkdir(parents=True)
            io.write_new(output / "unavailable-predictions.jsonl", b"".join(io.canonical(row) for row in unknown_predictions(proof)))
            require(time.monotonic() < deadline, "CONTINUATION_SOFT_DEADLINE")
            new_values, seconds, load_seconds = ext.new_native_rows(proof["pending_rows"], proof["prepared"], output / "new",
                identity_sha, proof["profile"], io.safe_path(proof["recipe"]["models"]), output / "keywords.txt", deadline)
            observations = merged_predictions(proof, new_values, output, identity_sha)
            io.write_new(output / "observations.jsonl", b"".join(io.canonical(row) for row in observations))
            require(time.monotonic() < deadline, "CONTINUATION_SOFT_DEADLINE")
            summary = {"schema": RESULT_SCHEMA, "status": "completed", "identity_sha256": identity_sha,
                "queue": queue, "registration": registration, "new_native_calls": len(new_values),
                "reused_previous64": 64, "reused_original_timeout_returns": 128, "native_predictions_available": len(observations),
                "native_consumed_unknown": 1, "generation_consumed_unknown": proof["closed"]["consumed_unknown_cells"],
                "legal_pcm_inputs": len(proof["rows"]), "planned_cells": 256, "complete_native_prediction_coverage": False,
                "counts": counts(proof, observations), "original_timeout_actual_nanoseconds": proof["terminal"]["actual_elapsed_ns"],
                "new_declared_budget_nanoseconds": NEW_LIMIT_NS,
                "combined_declared_upper_nanoseconds": proof["terminal"]["actual_elapsed_ns"] + NEW_LIMIT_NS,
                "new_native_inference_seconds": seconds, "model_load_seconds": load_seconds,
                "wall_seconds": time.monotonic() - started, "optimizer_steps": 0, "asr_or_tts_calls": 0,
                "automatic_retry": False, "maximum_attempts": 1, "thresholds_tuned": False, "human_gold": False,
                "qualification_approved": False, "quality_winner": None,
                "scope": "A separately budgeted continuation. Original timeout remains timed_out. Missing native prediction is not a negative or false negative; source-generation unknown remains a separate absence. Original partial evidence is referenced by its full verified immutable inventory, not rewritten or duplicated."}
            io.write_json(output / "summary.json", summary)
            io.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): io.file_record(p)
                for p in sorted(output.rglob("*")) if p.is_file()})
            return summary
        except BaseException as error:
            io.write_json(output / "failure.json", {"schema": RESULT_SCHEMA, "status": "failed",
                "error_type": type(error).__name__, "error": str(error), "wall_seconds": time.monotonic() - started,
                "new_claim_files": len(list((output / "new/claims").glob("*.json"))),
                "new_native_files": len(list((output / "new/native").glob("*.json"))),
                "registration_consumed": True, "unresolved_claims_are_consumed": True, "automatic_retry": False})
            raise


def _audit_finished(path, output):
    path, output = io.safe_path(path), io.safe_path(output)
    job = validate_job(read(path))
    proof = verify_origin(job, time.monotonic() + 180)
    manifest = read(output / "MANIFEST.json")
    require(set(manifest) == {p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file() and p != output / "MANIFEST.json"}, "CONTINUATION_FINAL_OUTPUT_ROSTER")
    for name, expected in manifest.items():
        require(io.file_record(output / io.relative(name)) == expected, "CONTINUATION_FINAL_OUTPUT_CHANGED")
    for key in ("partial_inventory", "partial_accounting"):
        require(io.file_record(output / (key + ".json")) == io.file_record(pinned(job[key])),
            "CONTINUATION_FINAL_RETAINED_PARTIAL_PIN")
    identity = identity_for(job, path, proof)
    require(read(output / "identity.json") == identity, "CONTINUATION_FINAL_IDENTITY_CHANGED")
    values, seconds = previous.readback_native(output / "new", proof["pending_rows"], ext.digest(identity))
    observations = merged_predictions(proof, values, output, ext.digest(identity))
    require([io.loads(line) for line in io.read_bytes(output / "observations.jsonl", 64 * 1024 * 1024).splitlines()] == observations and
        [io.loads(line) for line in io.read_bytes(output / "unavailable-predictions.jsonl", 64 * 1024 * 1024).splitlines()] ==
        unknown_predictions(proof), "CONTINUATION_FINAL_PREDICTION_OR_UNKNOWN_CHANGED")
    summary = read(output / "summary.json")
    require(summary["schema"] == RESULT_SCHEMA and summary["status"] == "completed" and
        summary["identity_sha256"] == ext.digest(identity) and summary["counts"] == counts(proof, observations) and
        summary["new_native_calls"] == 62 and summary["native_predictions_available"] == 254 and
        summary["native_consumed_unknown"] == 1 and summary["generation_consumed_unknown"] == proof["closed"]["consumed_unknown_cells"] and
        summary["new_native_inference_seconds"] == seconds and summary["new_declared_budget_nanoseconds"] == NEW_LIMIT_NS and
        summary["original_timeout_actual_nanoseconds"] == proof["terminal"]["actual_elapsed_ns"] and
        summary["combined_declared_upper_nanoseconds"] == proof["terminal"]["actual_elapsed_ns"] + NEW_LIMIT_NS,
        "CONTINUATION_FINAL_SUMMARY_CHANGED")
    root, queue, request, queue_sha = io.load_queue(output.parents[2])
    require(len(request["jobs"]) == 1 and output == root / "jobs" / request["jobs"][0]["job_id"] / "outputs" and
        request["jobs"][0]["recipe_sha256"] == previous.training.sha_file(path) and
        io.minutes_ns(request["global_budget_minutes"]) == NEW_LIMIT_NS, "CONTINUATION_COMPLETED_QUEUE_BINDING")
    terminal = io.result_for(root, request["jobs"][0], queue_sha)
    require(terminal["status"] == "completed" and terminal["exit_code"] == 0 and
        summary["queue"]["queue_sha256"] == queue_sha and summary["queue"]["claim_sha256"] == terminal["claim_sha256"] and
        read(registration_path(job)) == registration_value(job, summary["queue"], path, output) and
        summary["registration"] == pin(registration_path(job)), "CONTINUATION_FINAL_SINGLE_REGISTRATION")
    return {"status": "verified", "new_model_calls": 0, "ml_imports": 0, "native_predictions_available": 254,
        "native_consumed_unknown": 1, "generation_consumed_unknown": proof["closed"]["consumed_unknown_cells"],
        "original_timeout_actual_nanoseconds": proof["terminal"]["actual_elapsed_ns"],
        "continuation_actual_nanoseconds": terminal["actual_elapsed_ns"],
        "combined_actual_nanoseconds": proof["terminal"]["actual_elapsed_ns"] + terminal["actual_elapsed_ns"],
        "manifest_sha256": previous.training.sha_file(output / "MANIFEST.json"),
        "summary_sha256": previous.training.sha_file(output / "summary.json")}


def audit_finished(path, output):
    network = readonly_guard()
    guard = NoML()
    sys.meta_path.insert(0, guard)
    try:
        result = _audit_finished(path, output)
        require(not any(name in sys.modules for name in previous.ML_MODULES), "CONTINUATION_AUDIT_ML_IMPORTED")
        return {**result, "network_guard": network}
    finally:
        sys.meta_path.remove(guard)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "run", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--recipe", type=Path, required=True)
        if name != "verify":
            command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--heavy-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        network = readonly_guard()
        job = validate_job(read(args.recipe))
        started = time.monotonic()
        proof = verify_origin(job, started + 180)
        result = {"status": "verified", "pending_native_calls": len(proof["pending_rows"]),
            "native_consumed_unknown": len(proof["unknown_rows"]), "reused_returned": 192,
            "registration_path": str(registration_path(job)), "registration_exists": registration_path(job).exists(),
            "original_actual_ns": proof["terminal"]["actual_elapsed_ns"], "new_budget_ns": NEW_LIMIT_NS,
            "combined_upper_ns": proof["terminal"]["actual_elapsed_ns"] + NEW_LIMIT_NS,
            "wall_seconds": time.monotonic() - started, "new_model_calls": 0, "ml_imports": 0, "network_guard": network}
    elif args.command == "audit":
        result = audit_finished(args.recipe, args.output)
    else:
        result = run(args.recipe, args.output, args.heavy_lock)
    print(io.canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
