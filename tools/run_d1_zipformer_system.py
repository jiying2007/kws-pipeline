#!/usr/bin/env python3
"""Fixed native Zipformer observations of the complete sealed D1 prefix.

This is a separate bounded queue job, after the real preparation has completed.
Every retained PCM row is observed, including technical failures and unknown
labels. Only known projected keyword masks enter machine-weak event counts.
No CTC token score, threshold search, training, or formal qualification is done.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "training")]
import research_d1_evaluate as development
import research_d1_train as training
import research_no_network
import run_zipformer_system as native
from research_d1_recipe import row_identity
from research_d1_vocab import digest, require
from research_job_lock import heavy_model_slot

SCHEMA = "kws-d1-zipformer-system-job-v1"
RESULT_SCHEMA = "kws-d1-zipformer-system-result-v1"
SOURCE_PATH = "tools/run_d1_zipformer_system.py"
PROFILE_PATH = "configs/research/zipformer-system-20261105.lock.json"
PROFILE_SHA256 = "d7ac85aeac952ec6e35a39851d29edde1a14355599f888016e5e5b325d0f37c7"
QUEUE_OUTPUTS = ("summary.json", "observations.jsonl", "MANIFEST.json")
CODE_PATHS = tuple(dict.fromkeys((*training.CODE_PATHS, "training/research_d1_evaluate.py",
    SOURCE_PATH, "tools/run_zipformer_system.py", "tools/research_no_network.py", PROFILE_PATH)))
ML_MODULES = ("numpy", "torch", "onnxruntime", "sherpa_onnx", "qwen_tts", "qwen_asr", "whisper")


def source_records():
    return {name: training.sha_file(ROOT / name) for name in CODE_PATHS}


def validate_job(value):
    require(type(value) is dict and set(value) == {"schema", "completed_preparation", "plan_sha256",
        "source_ids", "held_out_source_family_ids", "models", "profile_sha256", "source_files",
        "timeout_seconds"} and value["schema"] == SCHEMA, "D1_ZIPFORMER_JOB_SCHEMA")
    dep = value["completed_preparation"]
    require(type(dep) is dict and set(dep) == {"queue_root", "job_id", "receipt_sha256", "manifest_sha256"}
        and type(dep["queue_root"]) is str and Path(dep["queue_root"]).is_absolute()
        and type(dep["job_id"]) is str and training.io_contract.ID.fullmatch(dep["job_id"]),
        "D1_ZIPFORMER_REAL_PREPARATION_BINDING")
    for sha in (dep["receipt_sha256"], dep["manifest_sha256"], value["plan_sha256"]):
        require(type(sha) is str and training.io_contract.SHA.fullmatch(sha), "D1_ZIPFORMER_SHA256")
    ids, families = value["source_ids"], value["held_out_source_family_ids"]
    require(type(ids) is list and len(ids) == 64 and ids == sorted(set(ids)) and
        all(type(name) is str and development.FAMILY_ID.fullmatch(name) for name in ids),
        "D1_ZIPFORMER_COMPLETE_64_ROSTER")
    require(type(families) is list and len(families) == 2 and families == sorted(set(families)) and
        all(type(name) is str and development.FAMILY_ID.fullmatch(name) for name in families),
        "D1_ZIPFORMER_HELD_OUT_FAMILIES")
    seconds = value["timeout_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 120,
        "D1_ZIPFORMER_MAXIMUM_120_SECONDS")
    require(type(value["models"]) is str and Path(value["models"]).is_absolute() and
        value["profile_sha256"] == PROFILE_SHA256 and value["source_files"] == source_records(),
        "D1_ZIPFORMER_FIXED_SOURCE_AND_PROFILE")
    return value


def completed_preparation(binding):
    """Read the actual old queue terminal; never invent an inter-queue dependency."""
    io = training.io_contract
    root, _, request, queue_sha = io.load_queue(binding["queue_root"])
    jobs = [job for job in request["jobs"] if job["job_id"] == binding["job_id"]]
    require(len(jobs) == 1, "D1_ZIPFORMER_PREPARATION_JOB_MISSING")
    result = io.result_for(root, jobs[0], queue_sha, check_active=False)
    require(result["status"] == "completed" and result["exit_code"] == 0 and
        result["receipt_sha256"] == binding["receipt_sha256"] and result["actual_duration_known"] is True and
        "MANIFEST.json" in result["outputs"] and
        result["outputs"]["MANIFEST.json"]["sha256"] == binding["manifest_sha256"],
        "D1_ZIPFORMER_PREPARATION_NOT_COMPLETE_OR_CHANGED")
    prepared = root / "jobs" / binding["job_id"] / "outputs"
    require(io.file_record(prepared / "MANIFEST.json") == result["outputs"]["MANIFEST.json"],
        "D1_ZIPFORMER_PREPARATION_MANIFEST_CHANGED")
    return prepared, {**binding, "queue_sha256": queue_sha, "claim_sha256": result["claim_sha256"],
        "recipe_sha256": result["recipe_sha256"], "actual_elapsed_ns": result["actual_elapsed_ns"]}


def verified_rows(job, prepared, record, vocabulary, train_rows):
    raw = training.io_contract.read_bytes(prepared / "projected-rows.jsonl", 64 * 1024 * 1024)
    rows = [training.io_contract.loads(line) for line in raw.splitlines()]
    require(len(rows) == 64 and sorted(row["source_id"] for row in rows) == job["source_ids"] and
        record["aggregate"]["plan_sha256"] == job["plan_sha256"] and
        all(row["global_source_key"] == [job["plan_sha256"], row["source_id"]] for row in rows),
        "D1_ZIPFORMER_PREPARED_GLOBAL_ROSTER_CHANGED")
    # This replays label projection, quarantine, source ancestry separation and
    # the exact training cohort. It deliberately returns *all* train/dev rows.
    development.development_rows(rows, train_rows, record, vocabulary, job["held_out_source_family_ids"])
    require(Counter(row["split"] for row in rows) == {"train": 48, "development": 16},
        "D1_ZIPFORMER_48_TRAIN_16_DEVELOPMENT_REQUIRED")
    for row in rows:
        development.read_pcm(prepared, row)
    return rows, hashlib.sha256(raw).hexdigest()


def input_binding(row):
    return {key: row[key] for key in ("source_id", "global_source_key", "file_sha256", "pcm_sha256", "samples")}


def observed_row(row, result, native_sha256):
    """Count each known keyword separately; unknown is never a negative."""
    blocked = ("failed-native" if row.get("generation_outcome") == "failed-native" else
        "uncertain-label" if row["label_status"] != "machine_candidate" else
        "quarantined-label" if row["quarantined"] else None)
    detected = {event["keyword_id"] for event in result["events"]}
    keywords = []
    for index in range(2):
        known = blocked is None and row["keyword_mask"][index]
        if known:
            require(type(row["keyword_targets"][index]) is int and row["keyword_targets"][index] in (0, 1)
                and row["keyword_weight"][index] > 0, "D1_ZIPFORMER_KNOWN_KEYWORD_LABEL")
        keywords.append({"keyword_id": index + 1, "known": known,
            "target": row["keyword_targets"][index] if known else None,
            "confidence_weight": row["keyword_weight"][index] if known else 0.0,
            "detected": index + 1 in detected,
            "excluded_reason": None if known else (blocked or "keyword-mask-unknown")})
    return {"schema": "kws-d1-zipformer-observation-v1", **input_binding(row),
        "split": row["split"], "source_family_id": row["source_family_id"], "speaker_id": row["speaker_id"],
        "source_role": row["source_role"], "label_binding_sha256": row_identity(row),
        "label_status": row["label_status"], "generation_outcome": row.get("generation_outcome"),
        "quarantined": row["quarantined"], "native_result_sha256": native_sha256,
        "events": result["events"], "keyword_observations": keywords,
        "ctc_eligibility_required": False, "human_gold": False, "qualification_eligible": False}


def summarize(rows):
    keywords = {}
    for keyword in (1, 2):
        items = [row["keyword_observations"][keyword - 1] for row in rows]
        known = [item for item in items if item["known"]]
        keywords[str(keyword)] = {
            "known_machine_weak_clips": len(known),
            "weak_positive_clips": sum(item["target"] == 1 for item in known),
            "weak_negative_clips": sum(item["target"] == 0 for item in known),
            "triggered_positive_clips": sum(item["target"] == 1 and item["detected"] for item in known),
            "triggered_negative_clips": sum(item["target"] == 0 and item["detected"] for item in known),
            "no_trigger_positive_clips": sum(item["target"] == 1 and not item["detected"] for item in known),
            "no_trigger_negative_clips": sum(item["target"] == 0 and not item["detected"] for item in known),
            "excluded_clips_by_reason": dict(Counter(item["excluded_reason"] for item in items if not item["known"])),
        }
    return {"raw_observed_clips": len(rows), "raw_native_events": sum(len(row["events"]) for row in rows),
        "raw_clips_with_events": sum(bool(row["events"]) for row in rows), "keywords": keywords,
        "formal_qualification_denominator": 0}


def current_queue_binding(path, output, job, lease):
    io = training.io_contract
    training.queue_binding(path, lease)
    require(os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(output) and
        os.environ.get("KWS_RESEARCH_RECIPE") == str(path), "D1_ZIPFORMER_QUEUE_PATH_BINDING")
    root, queue, request, queue_sha = io.load_queue(output.parents[2])
    job_id = os.environ.get("KWS_RESEARCH_JOB_ID")
    definitions = [item for item in request["jobs"] if item["job_id"] == job_id]
    require(len(definitions) == 1 and output == root / "jobs" / job_id / "outputs" and
        queue["heavy_lock"] == str(lease.path), "D1_ZIPFORMER_QUEUE_AND_LEASE_IDENTITY")
    definition = definitions[0]
    claim = training.read_json(output.parent / "claim.json")
    require(len(request["jobs"]) == 1 and definition["depends_on"] == [] and
        io.minutes_ns(request["global_budget_minutes"]) <= 120 * 10**9 and
        claim["schema"] == "kws-local-research-job-claim-v1" and claim["attempt"] == 1 and
        claim["automatic_retry"] is False and
        all(claim.get(key) == value for key, value in io.job_binding(definition, queue_sha).items()) and
        definition["recipe_sha256"] == training.sha_file(path) and
        io.minutes_ns(definition["timeout_minutes"], 999) <= int(job["timeout_seconds"] * 1e9) and
        0 < claim["allowed_runner_nanoseconds"] <= int(job["timeout_seconds"] * 1e9) and
        claim["deadline_monotonic_ns"] == claim["started_monotonic_ns"] + claim["allowed_runner_nanoseconds"] and
        claim["dependencies_sha256"] == training.sha_file(output.parent / "dependencies.json"),
        "D1_ZIPFORMER_QUEUE_DEADLINE_OR_CLAIM")
    return {"job_id": job_id, "queue_sha256": queue_sha,
        "claim_sha256": training.sha_file(output.parent / "claim.json"),
        "allowed_runner_nanoseconds": claim["allowed_runner_nanoseconds"],
        "deadline_monotonic_ns": claim["deadline_monotonic_ns"],
        "inherited_heavy_lock": lease.inherited, "supervised": lease.supervised,
        "pid": os.getpid(), "process_group": os.getpgrp()}


def run_calls(rows, prepared, output, identity_sha256, engine, convert, deadline):
    observations = []
    for row in rows:
        require(time.monotonic() < deadline, "D1_ZIPFORMER_SOFT_DEADLINE")
        item = input_binding(row)
        pcm = development.read_pcm(prepared, row)
        samples = convert(pcm)
        require(len(samples) == row["samples"], "D1_ZIPFORMER_NATIVE_INPUT_SAMPLES")
        claim_path = output / "claims" / (row["source_id"] + ".json")
        result_path = output / "native" / claim_path.name
        # Any already consumed cell blocks this attempt. No quality or crash retry.
        require(not claim_path.exists() and not result_path.exists(), "D1_ZIPFORMER_CELL_ALREADY_CONSUMED")
        native.write_new_json(claim_path, {"schema": 1, "identity_sha256": identity_sha256,
            "input": item, "attempt": 1, "started_unix_seconds": time.time()})
        started = time.monotonic()
        result = {"schema": 1, "identity_sha256": identity_sha256, "input": item, "status": "completed",
            "claim_sha256": training.sha_file(claim_path),
            **native.decode_samples(engine, samples, 160, deadline)}
        result["inference_seconds"] = time.monotonic() - started
        result["result_sha256"] = native.sha256(native.canonical(result))
        native.write_new_json(result_path, result)
        native.cached_result(result_path, identity_sha256, item)
        observation = observed_row(row, result, training.sha_file(result_path))
        observations.append(observation)
        training.append_json(output / "observations.jsonl", observation)
    return observations


def load_bound_rows(job):
    prepared, dependency = completed_preparation(job["completed_preparation"])
    record, vocabulary, train_rows = training.verify_prepared(prepared, job["completed_preparation"]["manifest_sha256"])
    require(record["request_sha256"] == dependency["recipe_sha256"], "D1_ZIPFORMER_PREPARATION_RECIPE_BINDING")
    rows, projected_sha = verified_rows(job, prepared, record, vocabulary, train_rows)
    return prepared, dependency, rows, projected_sha


def readback_native(output, rows, identity_sha256):
    """Reconstruct observations solely from the retained claims/native bytes."""
    expected = {row["source_id"] + ".json" for row in rows}
    for name in ("claims", "native"):
        require({path.name for path in (output / name).iterdir()} == expected,
            "D1_ZIPFORMER_NATIVE_ROSTER_CHANGED")
    observations, seconds = [], 0.0
    for row in rows:
        path = output / "native" / (row["source_id"] + ".json")
        result = native.cached_result(path, identity_sha256, input_binding(row))
        require(result is not None and type(result["inference_seconds"]) in (int, float) and
            math.isfinite(result["inference_seconds"]) and result["inference_seconds"] >= 0,
            "D1_ZIPFORMER_NATIVE_RESULT_INCOMPLETE")
        seconds += result["inference_seconds"]
        observations.append(observed_row(row, result, training.sha_file(path)))
    actual = [training.io_contract.loads(line) for line in
        training.io_contract.read_bytes(output / "observations.jsonl", 64 * 1024 * 1024).splitlines()]
    require(observations == actual, "D1_ZIPFORMER_NATIVE_PROJECTION_CHANGED")
    return observations, seconds


def audit_finished(path, output):
    """Read an already sealed result, without locks, ML imports or file writes."""
    path, output = Path(path), Path(output)
    job = validate_job(training.read_json(path))
    _, dependency, rows, projected_sha = load_bound_rows(job)
    manifest = training.read_json(output / "MANIFEST.json")
    require(set(manifest) == {p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file() and p != output / "MANIFEST.json"}, "D1_ZIPFORMER_OUTPUT_ROSTER_CHANGED")
    for name, expected in manifest.items():
        require(training.io_contract.file_record(output / training.io_contract.relative(name)) == expected,
            "D1_ZIPFORMER_OUTPUT_BYTES_CHANGED:" + name)
    identity = training.read_json(output / "identity.json")
    require(identity == {"schema": RESULT_SCHEMA, "recipe_sha256": training.sha_file(path),
        "completed_preparation": dependency, "projected_rows_sha256": projected_sha,
        "row_bindings_sha256": digest([row_identity(row) for row in rows]),
        "profile_sha256": job["profile_sha256"], "source_files": job["source_files"],
        "inputs": [input_binding(row) for row in rows]}, "D1_ZIPFORMER_OUTPUT_IDENTITY_CHANGED")
    observations, seconds = readback_native(output, rows, digest(identity))
    summary = training.read_json(output / "summary.json")
    require(summary["schema"] == RESULT_SCHEMA and summary["status"] == "completed" and
        summary["identity_sha256"] == digest(identity) and summary["new_native_calls"] == len(rows) and
        summary["native_inference_seconds"] == seconds and summary["by_split"] == {
            split: summarize([row for row in observations if row["split"] == split])
            for split in ("train", "development")}, "D1_ZIPFORMER_SUMMARY_PROJECTION_CHANGED")
    return {"status": "verified", "raw_observed_clips": len(rows), "new_model_calls": 0,
        "manifest_sha256": training.sha_file(output / "MANIFEST.json"), "summary_sha256": training.sha_file(output / "summary.json")}


def run(path, output, lock):
    path, output = training.io_contract.safe_path(path), training.io_contract.safe_path(output)
    job = validate_job(training.read_json(path))
    with heavy_model_slot(lock) as lease:
        queue = current_queue_binding(path, output, job, lease)
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        deadline = min(started + job["timeout_seconds"], queue["deadline_monotonic_ns"] / 1e9)
        native.write_new_json(output / "invocation.json", {"schema": RESULT_SCHEMA, "status": "claimed",
            "recipe_sha256": training.sha_file(path), "queue": queue, "automatic_retry": False})
        observations = []
        try:
            require(not any(name in sys.modules for name in ML_MODULES), "D1_ZIPFORMER_ML_IMPORTED_BEFORE_GUARD")
            research_no_network.set_offline_environment()
            guard = research_no_network.install()
            native.write_new_json(output / "network-guard.json", guard)
            prepared, dependency, rows, projected_sha = load_bound_rows(job)
            profile = training.read_json(ROOT / PROFILE_PATH)
            require(training.sha_file(ROOT / PROFILE_PATH) == job["profile_sha256"], "D1_ZIPFORMER_PROFILE_CHANGED")
            models = training.io_contract.safe_path(job["models"])
            native.check_profile(profile, models)
            identity = {"schema": RESULT_SCHEMA, "recipe_sha256": training.sha_file(path),
                "completed_preparation": dependency, "projected_rows_sha256": projected_sha,
                "row_bindings_sha256": digest([row_identity(row) for row in rows]),
                "profile_sha256": job["profile_sha256"], "source_files": job["source_files"],
                "inputs": [input_binding(row) for row in rows]}
            identity_sha = digest(identity)
            native.write_new_json(output / "identity.json", identity)
            for name in ("claims", "native"):
                (output / name).mkdir()
            keywords = "\n".join(" ".join(row["tokens"]) + " @kw" + str(row["id"]) for row in profile["keywords"]) + "\n"
            training.io_contract.write_new(output / "keywords.txt", keywords.encode())
            training.copy_once(path, output / "request.json")
            for name, expected in job["source_files"].items():
                require(training.copy_once(ROOT / name, output / "execution-source" / name) == expected,
                    "D1_ZIPFORMER_EXECUTION_SOURCE_CHANGED")
            require(time.monotonic() < deadline, "D1_ZIPFORMER_SOFT_DEADLINE")
            import numpy as np
            loading = time.monotonic()
            engine = native.make_engine(profile, models, output / "keywords.txt")
            load_seconds = time.monotonic() - loading
            require(time.monotonic() < deadline, "D1_ZIPFORMER_SOFT_DEADLINE")
            observations = run_calls(rows, prepared, output, identity_sha, engine,
                lambda pcm: np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768), deadline)
            observations, inference_seconds = readback_native(output, rows, identity_sha)
            require(time.monotonic() < deadline, "D1_ZIPFORMER_SOFT_DEADLINE")
            end = resource.getrusage(resource.RUSAGE_SELF)
            summary = {"schema": RESULT_SCHEMA, "status": "completed", "identity_sha256": identity_sha,
                "job_recipe_sha256": training.sha_file(path), "projected_rows_sha256": projected_sha,
                "preparation_dependency": dependency, "queue": queue,
                "new_native_calls": len(observations), "optimizer_steps": 0, "asr_or_tts_calls": 0,
                "by_split": {split: summarize([row for row in observations if row["split"] == split])
                    for split in ("train", "development")},
                "raw_observed_clips": len(observations), "added_padding_samples": 0,
                "native_inference_seconds": inference_seconds, "model_load_seconds": load_seconds,
                "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": end.ru_utime - usage.ru_utime, "system_cpu_seconds": end.ru_stime - usage.ru_stime,
                "peak_rss_kib": end.ru_maxrss, "observations_sha256": training.sha_file(output / "observations.jsonl"),
                "human_gold": False, "independent_human_test": False, "qualification_approved": False,
                "thresholds_tuned": False, "quality_winner": None, "automatic_retry": False,
                "scope": "Native keyword-event observations on synthetic source-group train/dev with machine-weak labels.",
                "comparison_boundary": "No shared CTC token score or causal single-variable comparison with the three trained CTC candidates.",
                "timing_boundary": "Native timestamps and supplied PCM coordinates, not acoustic completion or device latency."}
            native.write_new_json(output / "summary.json", summary)
            training.io_contract.write_json(output / "MANIFEST.json", {
                p.relative_to(output).as_posix(): training.io_contract.file_record(p)
                for p in sorted(output.rglob("*")) if p.is_file()})
            return summary
        except BaseException as error:
            native.write_new_json(output / "failure.json", {"schema": RESULT_SCHEMA, "status": "failed",
                "wall_seconds": time.monotonic() - started, "error_type": type(error).__name__, "error": str(error),
                "recorded_native_files": len(list((output / "native").glob("*.json"))),
                "recorded_claim_files": len(list((output / "claims").glob("*.json"))),
                "unresolved_claims_are_consumed": True, "automatic_retry": False})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "output", "heavy-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.recipe, args.output, args.heavy_lock), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
