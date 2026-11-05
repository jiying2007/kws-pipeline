#!/usr/bin/env python3
"""Close 256 planned D1 cells, reusing the exact previous 64 native results.

Only additional cells with retained legal PCM receive new native calls. The old identity,
claims, native bytes and observation objects are retained unchanged. Previous
PCM, label identity, masks and weights must match the full preparation exactly.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training"), str(ROOT)]
import run_d1_zipformer_system as previous
import research_d1_native_coverage as coverage

SCHEMA = "kws-d1-zipformer-extension-job-v2"
RESULT_SCHEMA = "kws-d1-zipformer-extension-result-v2"
SOURCE_PATH = "tools/run_d1_zipformer_extension.py"
ADDED_SOURCE_PATHS = (SOURCE_PATH, "tools/research_d1_native_coverage.py", "tools/research_d1_previous_native.py")
QUEUE_OUTPUTS = ("summary.json", "observations.jsonl", "MANIFEST.json")
require = previous.require
io = previous.training.io_contract
read = previous.training.read_json
digest = previous.digest


def source_records():
    return {**previous.source_records(), **{name: previous.training.sha_file(ROOT / name) for name in ADDED_SOURCE_PATHS}}


def validate_binding(value):
    require(type(value) is dict and set(value) == {"queue_root", "job_id", "receipt_sha256", "manifest_sha256"}
        and type(value["queue_root"]) is str and Path(value["queue_root"]).is_absolute()
        and type(value["job_id"]) is str and io.ID.fullmatch(value["job_id"]), "D1_NATIVE_V2_COMPLETED_BINDING")
    require(all(type(value[key]) is str and io.SHA.fullmatch(value[key])
        for key in ("receipt_sha256", "manifest_sha256")), "D1_NATIVE_V2_COMPLETED_HASHES")


def validate_job(job):
    require(type(job) is dict and set(job) == {"schema", "completed_preparation", "previous_native",
        "previous_source_root", "previous_source_files", "final_prior_evidence", "plan_sha256", "source_ids",
        "held_out_source_family_ids", "models", "profile_sha256", "source_files", "timeout_seconds"} and job["schema"] == SCHEMA,
        "D1_NATIVE_V2_JOB_SCHEMA")
    validate_binding(job["completed_preparation"])
    validate_binding(job["previous_native"])
    old_files = job["previous_source_files"]
    require(type(old_files) is dict and set(old_files) == set(previous.CODE_PATHS)
        and all(type(value) is str and io.SHA.fullmatch(value) for value in old_files.values()),
        "D1_NATIVE_V2_ORIGINAL_SOURCE_PINS")
    final_prior = job["final_prior_evidence"]
    require(type(final_prior) is dict and set(final_prior) == {"path", "sha256"}
        and type(final_prior["path"]) is str and Path(final_prior["path"]).is_absolute()
        and type(final_prior["sha256"]) is str and io.SHA.fullmatch(final_prior["sha256"]),
        "D1_NATIVE_V2_ACTUAL_FINAL_CLOSURE_PIN")
    require(type(job["plan_sha256"]) is str and io.SHA.fullmatch(job["plan_sha256"]), "D1_NATIVE_V2_PLAN")
    ids, families = job["source_ids"], job["held_out_source_family_ids"]
    require(type(ids) is list and len(ids) == 256 and ids == sorted(set(ids)) and
        all(type(name) is str and previous.development.FAMILY_ID.fullmatch(name) for name in ids),
        "D1_NATIVE_V2_FULL_256_ROSTER")
    require(type(families) is list and len(families) == 2 and families == sorted(set(families)) and
        all(type(name) is str and previous.development.FAMILY_ID.fullmatch(name) for name in families),
        "D1_NATIVE_V2_HELD_OUT_FAMILIES")
    require(all(type(job[key]) is str and Path(job[key]).is_absolute()
        for key in ("models", "previous_source_root")), "D1_NATIVE_V2_ABSOLUTE_ASSETS_AND_SOURCE")
    seconds = job["timeout_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 120,
        "D1_NATIVE_V2_MAXIMUM_120_SECONDS")
    require(job["profile_sha256"] == previous.PROFILE_SHA256 and job["source_files"] == source_records(),
        "D1_NATIVE_V2_SOURCE_AND_PROFILE_CHANGED")
    return job


def old_source_readback(job, action, *, deadline, queue_root=None):
    """Never mix the original queue/prepare module globals with the new ones."""
    require(time.monotonic() < deadline, "D1_NATIVE_V2_OLD_HELPER_DEADLINE")
    request = {"source_root": job["previous_source_root"], "source_files": job["previous_source_files"],
        "previous_native": job["previous_native"], "action": action, "queue_root": queue_root}
    completed = subprocess.run([sys.executable, "-I", "-S", "-B",
        str(ROOT / "tools/research_d1_previous_native.py")], input=io.canonical(request),
        capture_output=True, timeout=max(.001, deadline - time.monotonic()), start_new_session=False)
    require(completed.returncode == 0, "D1_NATIVE_V2_OLD_HELPER_FAILED:" + completed.stderr[-4096:].decode(errors="replace"))
    require(len(completed.stdout) <= 64 * 1024 * 1024, "D1_NATIVE_V2_OLD_HELPER_BOUNDED_OUTPUT")
    value = io.loads(completed.stdout)
    require(value["ml_imports"] == value["new_model_calls"] == 0 and value["helper_pgid"] == os.getpgrp()
        and value["verified_source_files"] == job["previous_source_files"], "D1_NATIVE_V2_OLD_HELPER_CONTRACT")
    return value


def original_native(job, deadline=None):
    deadline = deadline if deadline is not None else time.monotonic() + job["timeout_seconds"]
    old = old_source_readback(job, "native", deadline=deadline)
    recipe = old["recipe"]
    require(recipe["models"] == job["models"] and recipe["profile_sha256"] == job["profile_sha256"]
        and recipe["plan_sha256"] == job["plan_sha256"] and recipe["source_files"] == job["previous_source_files"]
        and recipe["held_out_source_family_ids"] == job["held_out_source_family_ids"], "D1_NATIVE_V2_PREVIOUS_SYSTEM_CHANGED")
    for key in ("root", "output", "recipe_path"):
        old[key] = io.safe_path(old[key])
    return old


def full_rows(job, old, deadline):
    prepared, dependency = previous.completed_preparation(job["completed_preparation"])
    record, vocabulary, train_rows = previous.training.verify_prepared(prepared,
        job["completed_preparation"]["manifest_sha256"])
    require(record["request_sha256"] == dependency["recipe_sha256"], "D1_NATIVE_V2_PREPARATION_RECIPE")
    raw = io.read_bytes(prepared / "projected-rows.jsonl", 64 * 1024 * 1024)
    rows = [io.loads(line) for line in raw.splitlines()]
    require(64 <= len(rows) <= 256 and record["aggregate"]["plan_sha256"] == job["plan_sha256"] and
        all(row["global_source_key"] == [job["plan_sha256"], row["source_id"]] for row in rows),
        "D1_NATIVE_V2_PREPARED_GLOBAL_ROSTER")
    previous.development.development_rows(rows, train_rows, record, vocabulary, job["held_out_source_family_ids"])
    loader, _, sources = previous.training.load_data_modules(prepared / "data-source/automation/active_asr_weak_view.py")
    require(sources["files"] == record["data_source"]["files"], "D1_NATIVE_V2_RETAINED_DATA_SOURCE")
    declarations = [{key: value[key] for key in ("path", "manifest_sha256")} for value in record["aggregate"]["parents"]]
    raw_rows, aggregate = previous.training.aggregate_batches(declarations, loader, expected_plan_sha256=job["plan_sha256"])
    projected = previous.training.project_rows(raw_rows, vocabulary)
    require(aggregate == record["aggregate"] and [previous.row_identity(row) for row in projected] ==
        [previous.row_identity(row) for row in rows], "D1_NATIVE_V2_REPLAYED_PROJECTED_ROWS_CHANGED")
    closed = coverage.bind(job, record, rows, [row["source_id"] for row in old["rows"]],
        lambda root: old_source_readback(job, "queue", deadline=deadline, queue_root=root))
    for row in rows:
        previous.development.read_pcm(prepared, row)
    return prepared, dependency, rows, hashlib.sha256(raw).hexdigest(), closed


def join_unchanged(old_rows, rows):
    old = {row["source_id"]: row for row in old_rows}
    full = {row["source_id"]: row for row in rows}
    require(len(old) == len(old_rows) == 64 and len(full) == len(rows) and 64 <= len(rows) <= 256 and set(old) <= set(full),
        "D1_NATIVE_V2_PREVIOUS_GLOBAL_ROSTER")
    for name, original in old.items():
        current = full[name]
        for key in ("global_source_key", "file_sha256", "pcm_sha256", "samples", "source_family_id",
                    "speaker_id", "source_role", "split", "keyword_targets", "keyword_mask", "keyword_weight"):
            require(current[key] == original[key], "D1_NATIVE_V2_PREVIOUS_ROW_DRIFT:" + key)
        # This includes full labels, original known masks/weights, ASR/native
        # evidence, source identity and projection; only file location may differ.
        require(previous.row_identity(current) == previous.row_identity(original),
            "D1_NATIVE_V2_PREVIOUS_LABEL_IDENTITY_CHANGED")
    new = [row for row in rows if row["source_id"] not in old]
    require(len(new) == len(rows) - 64, "D1_NATIVE_V2_EXACT_NEW_OBSERVABLE_ROSTER")
    return new


def retain_previous(output, old):
    manifest = read(old["output"] / "MANIFEST.json")
    target = output / "reused/original-output"
    for name in sorted(set(manifest) | {"MANIFEST.json"}):
        expected = previous.training.sha_file(old["output"] / io.relative(name))
        require(previous.training.copy_once(old["output"] / name, target / name) == expected,
            "D1_NATIVE_V2_RETAINED_PREVIOUS_BYTES_CHANGED")
    name = old["terminal"]["job_id"]
    for relative in ("queue.json", "request.json", "active-plan.json", "recipes/" + name + ".json",
        "jobs/" + name + "/result.json", "jobs/" + name + "/claim.json",
        "jobs/" + name + "/stdout.log", "jobs/" + name + "/stderr.log"):
        previous.training.copy_once(old["root"] / relative, output / "reused/original-queue" / relative)


def merged_observations(rows, old, new_observations, output, identity_sha256):
    old_items = {row["source_id"]: row for row in old["observations"]}
    new_items = {row["source_id"]: row for row in new_observations}
    require(len(old_items) == 64 and len(new_items) == len(new_observations) == len(rows) - 64 and not set(old_items) & set(new_items)
        and set(old_items) | set(new_items) == {row["source_id"] for row in rows}, "D1_NATIVE_V2_MERGED_ROSTER")
    result = []
    for row in rows:
        reused = row["source_id"] in old_items
        observation = (old_items if reused else new_items)[row["source_id"]]
        relative = ("reused/original-output" if reused else "new") + "/native/" + row["source_id"] + ".json"
        native_path = output / relative
        native_identity = old["identity_sha256"] if reused else identity_sha256
        native_value = previous.native.cached_result(native_path, native_identity, previous.input_binding(row))
        require(native_value is not None and previous.observed_row(row, native_value,
            previous.training.sha_file(native_path)) == observation, "D1_NATIVE_V2_ORIGINAL_OBSERVATION_CHANGED")
        result.append({"schema": "kws-d1-zipformer-extension-observation-v2", "source_id": row["source_id"],
            "global_source_key": row["global_source_key"], "origin": "reused-previous64" if reused else "new-observable",
            "new_native_call": not reused, "native_identity_sha256": native_identity,
            "native_result": {"path": relative, **io.file_record(native_path)},
            "original_observation_sha256": digest(observation), "observation": observation,
            "previous_terminal_sha256": old["terminal"]["receipt_sha256"] if reused else None})
    return result


def grouped_counts(rows):
    def split_counts(values):
        return {split: previous.summarize([row["observation"] for row in values
            if row["observation"]["split"] == split]) for split in ("train", "development")}
    return {"all_observable": split_counts(rows), "previous64": split_counts([row for row in rows if not row["new_native_call"]]),
        "new_observable": split_counts([row for row in rows if row["new_native_call"]])}


def identity_for(job, path, dependency, projected_sha, rows, old, closed):
    return {"schema": RESULT_SCHEMA, "recipe_sha256": previous.training.sha_file(path),
        "completed_preparation": dependency, "projected_rows_sha256": projected_sha,
        "row_bindings_sha256": digest([previous.row_identity(row) for row in rows]),
        "previous_native": job["previous_native"], "previous_identity_sha256": old["identity_sha256"],
        "previous_source_root": job["previous_source_root"], "previous_source_files": old["recipe"]["source_files"],
        "final_prior_evidence": job["final_prior_evidence"], "terminal_coverage_sha256": digest(closed),
        "profile_sha256": job["profile_sha256"], "source_files": job["source_files"],
        "inputs": [previous.input_binding(row) for row in rows]}


def retain_coverage(output, closed):
    previous.native.write_new_json(output / "terminal-coverage.json", closed)
    io.write_new(output / "unobservable.jsonl", b"".join(io.canonical(row) for row in closed["unobservable"]))
    copies = []
    for index, binding in enumerate(closed["evidence_files"]):
        relative = "coverage-evidence/" + format(index, "04d") + ".json"
        require(previous.training.copy_once(binding["path"], output / relative) == binding["sha256"]
            and io.file_record(output / relative) == {key: binding[key] for key in ("bytes", "sha256")},
            "D1_NATIVE_V2_RETAINED_COVERAGE_CHANGED")
        copies.append({"original": binding, "retained": relative})
    previous.native.write_new_json(output / "coverage-evidence.json", copies)


def new_native_rows(rows, prepared, output, identity_sha, profile, models, keywords, deadline):
    # A fully excluded suffix still has complete generation terminals. It must
    # not load NumPy or an engine, nor turn omitted PCM into an empty waveform.
    if not rows:
        io.write_new(output / "observations.jsonl", b"")
        return [], 0.0, 0.0
    require(time.monotonic() < deadline, "D1_NATIVE_V2_SOFT_DEADLINE")
    import numpy as np
    loading = time.monotonic()
    engine = previous.native.make_engine(profile, models, keywords)
    load_seconds = time.monotonic() - loading
    previous.run_calls(rows, prepared, output, identity_sha, engine,
        lambda pcm: np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768), deadline)
    observations, inference_seconds = previous.readback_native(output, rows, identity_sha)
    return observations, inference_seconds, load_seconds


def run(path, output, lock):
    path, output = io.safe_path(path), io.safe_path(output)
    job = validate_job(read(path))
    with previous.heavy_model_slot(lock) as lease:
        queue = previous.current_queue_binding(path, output, job, lease)
        output = previous.training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        deadline = min(started + job["timeout_seconds"], queue["deadline_monotonic_ns"] / 1e9)
        previous.native.write_new_json(output / "invocation.json", {"schema": RESULT_SCHEMA, "status": "claimed",
            "recipe_sha256": previous.training.sha_file(path), "queue": queue, "automatic_retry": False})
        try:
            require(not any(name in sys.modules for name in previous.ML_MODULES), "D1_NATIVE_V2_ML_BEFORE_GUARD")
            previous.research_no_network.set_offline_environment()
            guard = previous.research_no_network.install()
            previous.native.write_new_json(output / "network-guard.json", guard)
            old = original_native(job, deadline)
            prepared, dependency, rows, projected_sha, closed = full_rows(job, old, deadline)
            new_rows = join_unchanged(old["rows"], rows)
            profile = read(ROOT / previous.PROFILE_PATH)
            models = io.safe_path(job["models"])
            previous.native.check_profile(profile, models)
            identity = identity_for(job, path, dependency, projected_sha, rows, old, closed)
            identity_sha = digest(identity)
            previous.native.write_new_json(output / "identity.json", identity)
            retain_previous(output, old)
            retain_coverage(output, closed)
            new_root = output / "new"
            for name in ("claims", "native"):
                (new_root / name).mkdir(parents=True)
            keyword_raw = (old["output"] / "keywords.txt").read_bytes()
            expected = "\n".join(" ".join(row["tokens"]) + " @kw" + str(row["id"]) for row in profile["keywords"]) + "\n"
            require(keyword_raw == expected.encode(), "D1_NATIVE_V2_KEYWORDS_CHANGED")
            io.write_new(output / "keywords.txt", keyword_raw)
            previous.training.copy_once(path, output / "request.json")
            for name, expected_sha in job["source_files"].items():
                require(previous.training.copy_once(ROOT / name, output / "execution-source" / name) == expected_sha,
                    "D1_NATIVE_V2_EXECUTION_SOURCE_CHANGED")
            require(time.monotonic() < deadline, "D1_NATIVE_V2_SOFT_DEADLINE")
            new_observations, inference_seconds, load_seconds = new_native_rows(new_rows, prepared, new_root,
                identity_sha, profile, models, output / "keywords.txt", deadline)
            observations = merged_observations(rows, old, new_observations, output, identity_sha)
            io.write_new(output / "observations.jsonl", b"".join(io.canonical(row) for row in observations))
            require(time.monotonic() < deadline, "D1_NATIVE_V2_SOFT_DEADLINE")
            end = resource.getrusage(resource.RUSAGE_SELF)
            summary = {"schema": RESULT_SCHEMA, "status": "completed", "identity_sha256": identity_sha,
                "recipe_sha256": previous.training.sha_file(path), "completed_preparation": dependency,
                "previous_native": job["previous_native"], "previous_label_identities_unchanged": True,
                "queue": queue, "new_native_calls": len(new_rows), "reused_native_calls": 64,
                "planned_cells": 256, "raw_observed_clips": len(rows), "unobservable_cells": closed["excluded_cells"],
                "unobservable_by_split": closed["unobservable_by_split"], "terminal_coverage_sha256": digest(closed),
                "counts": grouped_counts(observations), "new_native_inference_seconds": inference_seconds,
                "previous_native_inference_seconds": old["native_inference_seconds"],
                "previous_actual_queue_elapsed_ns": old["terminal"]["actual_elapsed_ns"],
                "model_load_seconds": load_seconds, "wall_seconds": time.monotonic() - started,
                "user_cpu_seconds": end.ru_utime - usage.ru_utime, "system_cpu_seconds": end.ru_stime - usage.ru_stime,
                "peak_rss_kib": end.ru_maxrss, "observations_sha256": previous.training.sha_file(output / "observations.jsonl"),
                "added_padding_samples": 0, "thresholds_tuned": False, "automatic_retry": False,
                "optimizer_steps": 0, "asr_or_tts_calls": 0, "human_gold": False,
                "qualification_approved": False, "quality_winner": None,
                "scope": "Complete 256 generation terminals; native events only for legal retained PCM. Original 16 development rows remain separate from additional observed development rows.",
                "comparison_boundary": "Machine-weak keyword events, not the three CTC candidates' token scores or a device qualification."}
            previous.native.write_new_json(output / "summary.json", summary)
            io.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): io.file_record(p)
                for p in sorted(output.rglob("*")) if p.is_file()})
            return summary
        except BaseException as error:
            previous.native.write_new_json(output / "failure.json", {"schema": RESULT_SCHEMA, "status": "failed",
                "wall_seconds": time.monotonic() - started, "error_type": type(error).__name__, "error": str(error),
                "new_claim_files": len(list((output / "new/claims").glob("*.json"))),
                "new_native_files": len(list((output / "new/native").glob("*.json"))),
                "unresolved_claims_are_consumed": True, "automatic_retry": False})
            raise


def audit_finished(path, output):
    """Pure readback. Corrupt old evidence is never repaired by another call."""
    path, output = Path(path), Path(output)
    job = validate_job(read(path))
    deadline = time.monotonic() + job["timeout_seconds"]
    old = original_native(job, deadline)
    _, dependency, rows, projected_sha, closed = full_rows(job, old, deadline)
    new_rows = join_unchanged(old["rows"], rows)
    manifest = read(output / "MANIFEST.json")
    require(set(manifest) == {p.relative_to(output).as_posix() for p in output.rglob("*")
        if p.is_file() and p != output / "MANIFEST.json"}, "D1_NATIVE_V2_OUTPUT_ROSTER")
    for name, expected in manifest.items():
        require(io.file_record(output / io.relative(name)) == expected, "D1_NATIVE_V2_OUTPUT_CHANGED:" + name)
    old_manifest = read(old["output"] / "MANIFEST.json")
    for name in set(old_manifest) | {"MANIFEST.json"}:
        require((output / "reused/original-output" / name).read_bytes() == (old["output"] / name).read_bytes(),
            "D1_NATIVE_V2_PREVIOUS_BYTES_CHANGED")
    identity = identity_for(job, path, dependency, projected_sha, rows, old, closed)
    require(read(output / "identity.json") == identity, "D1_NATIVE_V2_IDENTITY_CHANGED")
    require(read(output / "terminal-coverage.json") == closed and
        [io.loads(line) for line in io.read_bytes(output / "unobservable.jsonl", 64 * 1024 * 1024).splitlines()] ==
        closed["unobservable"], "D1_NATIVE_V2_COVERAGE_CHANGED")
    copies = read(output / "coverage-evidence.json")
    require([item["original"] for item in copies] == closed["evidence_files"], "D1_NATIVE_V2_COVERAGE_COPIES")
    for item in copies:
        require(io.file_record(output / io.relative(item["retained"])) ==
            {key: item["original"][key] for key in ("bytes", "sha256")}, "D1_NATIVE_V2_COVERAGE_COPY_CHANGED")
    observed_new, seconds = previous.readback_native(output / "new", new_rows, digest(identity))
    merged = merged_observations(rows, old, observed_new, output, digest(identity))
    actual = [io.loads(line) for line in io.read_bytes(output / "observations.jsonl", 64 * 1024 * 1024).splitlines()]
    require(merged == actual, "D1_NATIVE_V2_MERGED_OBSERVATIONS_CHANGED")
    summary = read(output / "summary.json")
    require(summary["schema"] == RESULT_SCHEMA and summary["status"] == "completed" and
        summary["identity_sha256"] == digest(identity) and summary["new_native_calls"] == len(new_rows) and
        summary["reused_native_calls"] == 64 and summary["raw_observed_clips"] == len(rows) and
        summary["planned_cells"] == 256 and summary["unobservable_cells"] == closed["excluded_cells"] and
        summary["terminal_coverage_sha256"] == digest(closed) and summary["unobservable_by_split"] == closed["unobservable_by_split"] and
        summary["counts"] == grouped_counts(merged) and summary["new_native_inference_seconds"] == seconds,
        "D1_NATIVE_V2_SUMMARY_CHANGED")
    return {"status": "verified", "new_model_calls": 0, "reused_native_calls": 64,
        "raw_observed_clips": len(rows), "planned_cells": 256, "unobservable_cells": closed["excluded_cells"],
        "manifest_sha256": previous.training.sha_file(output / "MANIFEST.json"),
        "summary_sha256": previous.training.sha_file(output / "summary.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "output", "heavy-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.recipe, args.output, args.heavy_lock), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
