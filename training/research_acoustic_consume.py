#!/usr/bin/env python3
"""Consume a closed acoustic observer in one fixed, positive-only C1 smoke.

The original variants and ASR observations are read-only. This module adds a
separate weighted full-text projection and one ordinary research-queue worker.
It has no generator, recognizer, scheduler, retry, or quality-selection loop.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_train as training
import research_d1_fit_diagnostic as diagnostic
import research_d2_consume as historical
import research_context_v3_consume as core
from research_context_v3_handoff_inputs import GlobalSources
from research_d1_recipe import BATCH_SIZE, SEED, row_identity
from research_d1_vocab import digest, fulltext_targets, require, validate_vocabulary

PREP = "kws-acoustic-consumption-preparation-v1"
ROW = "kws-acoustic-consumer-fulltext-v1"
JOB = "kws-acoustic-c1-consumption-job-v1"
RESULT = "kws-acoustic-c1-consumption-result-v1"
JOB_ID = "acoustic-positive-ingestion"
CANDIDATE = "c1-ctc2601"
STEPS, SECONDS, ROWS = 25, 120, 124
ENGINES = ("qwen06", "sensevoice", "whisper")
KINDS = {"onset-offset": 28, "digital-level": 28,
         "measured-rir-convolution": 28, "recorded-noise-mix": 40}
LABEL_FIELDS = ("ctc_mask", "ctc_weight", "full_text", "target_ids",
                "keyword_targets", "keyword_mask", "keyword_weight")
CODE = tuple(dict.fromkeys((*core.CODE, "training/research_context_v2_inputs.py",
                           "training/research_acoustic_consume.py")))
QUEUE_OUTPUTS = ("summary.json", "consumption.jsonl", "before.json", "after.json",
                 "development.json", "final.pt", "MANIFEST.json")

# This runs the original frozen observer in a fresh stdlib-only process. All
# outputs must already exist before calling its save-or-verify helpers; no
# consumed/unknown job can be reconciled or retried by this consumer.
OBSERVER_PROGRAM = r'''
import fcntl, hashlib, json, os, stat, sys
from pathlib import Path
source, stage_path, stage_sha, observer_sha = map(str, sys.argv[1:])
source, stage_path = Path(source), Path(stage_path)
def need(ok, why):
    if not ok:
        raise ValueError(why)
def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
observer = source / "tools/research_acoustic_asr.py"
need(file_sha(stage_path) == stage_sha and file_sha(observer) == observer_sha,
     "ACOUSTIC_CONSUMER_FROZEN_OBSERVER_CHANGED")
sys.path[:0] = [str(source), str(source / "tools"), str(source / "training")]
import research_acoustic_asr as a
q = a.queue
fd = os.open(stage_path.parent / "driver.lock",
             os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    before = os.fstat(fd)
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
         "ACOUSTIC_CONSUMER_ORIGINAL_DRIVER_LOCK")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage, plan, request, _ = a.load_stage(stage_path)
    need(stage["source_hashes"]["tools/research_acoustic_asr.py"]["sha256"] == observer_sha,
         "ACOUSTIC_CONSUMER_OBSERVER_SOURCE_PIN")
    report = q.inspect(stage_path.parent / "queue", reconcile=False)
    ids = [a.job_id(i) for i in range(a.JOBS)]
    need(len(ids) == 16 and set(report["jobs"]) == set(ids) and
         report["status"] == "completed" and report["actual_total_is_complete"] is True and
         not report["unresolved_jobs"] and not report["budget_overrun_nanoseconds"] and
         all(report["jobs"][name]["status"] == "completed" for name in ids),
         "ACOUSTIC_CONSUMER_COMPLETE_QUEUE_REQUIRED")
    target = stage_path.parent / "validation"
    need(target.is_dir() and {p.name for p in target.iterdir()} ==
         {"MANIFEST.json", "summary.json", "rows.jsonl"},
         "ACOUSTIC_CONSUMER_EXISTING_AGGREGATE_REQUIRED")
    for i, name in enumerate(ids):
        output = Path(a.read(stage["asr_requests"][i]["path"])["output"])
        need((output / "audit/final-summary.json").is_file() and
             (stage_path.parent / "chunks" / name / "validation.json").is_file(),
             "ACOUSTIC_CONSUMER_SEALED_NATIVE_VIEW_REQUIRED")
    # This replays the exact original parent, split, asset, PCM and label joins.
    variants, bindings = a.verify_variants(
        Path(plan["variant_manifest"]["path"]).parent, plan["variant_manifest"]["sha256"])
    need(bindings == plan["bindings"] and
         [r["sample_id"] for r in variants] == plan["ordered_sample_ids"],
         "ACOUSTIC_CONSUMER_ORIGINAL_BINDINGS_CHANGED")
    prerequisite = a.verify_prerequisites(stage)
    summary, aggregate = a.aggregate(stage, plan)
    need(summary["observed_rows"] == 124 and summary["native_asr_calls"] == 372 and
         summary["noise_ctc_masked_rows_unchanged"] == 40 and
         summary["frozen_variant_rows_changed"] == 0 and summary["human_gold"] is False,
         "ACOUSTIC_CONSUMER_FULL124_OBSERVATION_REQUIRED")
    final = stage_path.parent / "checkpoints/final/campaign-complete"
    saved = a.phase_metadata(final)
    scope = a.read(final / "capsule.json")["context"]
    expected = {"schema": a.SCHEMA, "phase": "campaign-complete", "stage": a.pin(stage_path),
        "aggregate": aggregate, "maximum_asr_calls": 372, "source_rows_changed": 0,
        "new_tts_calls": 0, "new_optimizer_steps": 0, "launch_intent": False}
    need({k: v for k, v in scope.items() if k != "owner_invocation"} == expected,
         "ACOUSTIC_CONSUMER_FINAL_SAVE_SCOPE_CHANGED")
    observations = [q.loads(line) for line in q.read_bytes(target / "rows.jsonl").splitlines()]
    need([r["sample_id"] for r in observations] == plan["ordered_sample_ids"],
         "ACOUSTIC_CONSUMER_OBSERVATION_ROSTER")
    need("torch" not in sys.modules, "ACOUSTIC_CONSUMER_OBSERVER_IMPORTED_MODEL_RUNTIME")
    closure = {"stage": a.pin(stage_path), "observer": a.pin(observer), "prerequisite": prerequisite,
        "heavy_lock": stage["heavy_lock"],
        "variant_manifest": plan["variant_manifest"], "aggregate": aggregate,
        "aggregate_summary": a.pin(target / "summary.json"), "aggregate_rows": a.pin(target / "rows.jsonl"),
        "final_save": saved, "summary": summary, "new_model_calls": 0}
    items = [{"binding": bindings[r["sample_id"]], "observation": r} for r in observations]
    print(json.dumps({"closure": closure, "items": items}, ensure_ascii=False, sort_keys=True))
finally:
    after = os.fstat(fd)
    need((before.st_dev, before.st_ino, before.st_size) ==
         (after.st_dev, after.st_ino, after.st_size), "ACOUSTIC_CONSUMER_DRIVER_LOCK_CHANGED")
    os.close(fd)
'''


def source_records():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def validate_request(value):
    require(type(value) is dict and set(value) == {"schema", "stage", "stage_sha256", "observer",
            "observer_sha256", "anchor", "anchor_manifest_sha256"} and value["schema"] == PREP,
            "ACOUSTIC_CONSUMER_PREPARATION_REQUEST")
    for name in ("stage", "observer", "anchor"):
        require(type(value[name]) is str and Path(value[name]).is_absolute(),
                "ACOUSTIC_CONSUMER_ABSOLUTE_INPUT_REQUIRED")
        training.io_contract.safe_path(value[name])
    for name in ("stage_sha256", "observer_sha256", "anchor_manifest_sha256"):
        require(type(value[name]) is str and training.io_contract.SHA.fullmatch(value[name]),
                "ACOUSTIC_CONSUMER_INPUT_HASH_REQUIRED")
    require(Path(value["observer"]).name == "research_acoustic_asr.py" and
            Path(value["observer"]).parent.name == "tools", "ACOUSTIC_CONSUMER_OBSERVER_PATH")
    return value


def closed_observer(request):
    request = validate_request(request)
    require(training.sha_file(request["stage"]) == request["stage_sha256"] and
            training.sha_file(request["observer"]) == request["observer_sha256"],
            "ACOUSTIC_CONSUMER_INPUT_CHANGED")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", OBSERVER_PROGRAM,
        str(Path(request["observer"]).parents[1]), request["stage"], request["stage_sha256"],
        request["observer_sha256"]], stdin=subprocess.DEVNULL, capture_output=True, timeout=90)
    require(result.returncode == 0, "ACOUSTIC_CONSUMER_OBSERVER_NOT_CLOSED:" +
            result.stderr.decode(errors="replace")[-2000:])
    require(len(result.stdout) <= 32 * 1024 * 1024, "ACOUSTIC_CONSUMER_OBSERVER_RECEIPT_BOUND")
    value = training.io_contract.loads(result.stdout)
    require(value["closure"]["stage"]["sha256"] == training.sha_file(request["stage"]) == request["stage_sha256"] and
            value["closure"]["observer"]["sha256"] == training.sha_file(request["observer"]) ==
            request["observer_sha256"], "ACOUSTIC_CONSUMER_INPUT_CHANGED_DURING_REPLAY")
    return value


def project_row(binding, observation, vocabulary, package_sha):
    """Add consumer masks, keeping inherited labels and native evidence intact."""
    row, parent = binding["variant_row"], binding["parent_row"]
    require(observation["sample_id"] == row["sample_id"] and
            observation["variant_row_sha256"] == training.io_contract.sha(training.io_contract.canonical(row)) and
            observation["parent_row_sha256"] == training.io_contract.sha(training.io_contract.canonical(parent)) and
            observation["inherited_variant_labels"] == {k: row[k] for k in LABEL_FIELDS},
            "ACOUSTIC_CONSUMER_OBSERVATION_BINDING")
    require(observation["parent_key"] == row["parent_key"] and
            observation["parent_file_sha256"] == row["parent_file_sha256"] and
            observation["variant_file_sha256"] == row["file_sha256"] and
            observation["original_pcm16_sha256"] == row["pcm_sha256"] and
            observation["inherited_parent_full_text"] == observation["normalized_parent_full_text"] ==
            parent["full_text"] and all(observation[k] == row[k] for k in
            ("source_family_id", "speaker_id", "source_role", "split", "kind", "recipe")),
            "ACOUSTIC_CONSUMER_OBSERVATION_SOURCE_CHANGED")
    require(row["split"] in ("train", "development") and
            row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
            all(row[k] == parent[k] for k in ("source_family_id", "speaker_id", "source_role", "split")) and
            row["parent_key"] == parent["global_source_key"] and
            row["parent_file_sha256"] == parent["file_sha256"], "ACOUSTIC_CONSUMER_PARENT_SPLIT_JOIN")
    require(row["training_admitted"] is False and row["human_gold"] is False and
            row["independent_test"] is False and row["new_independent_speaker"] is False and
            parent["generation_category"] == "full-positive" and parent["ctc_mask"] is True and
            parent["ctc_weight"] == .5 and row["vocabulary_sha256"] == digest(vocabulary),
            "ACOUSTIC_CONSUMER_ORIGINAL_SCOPE")
    votes = observation["observations"]
    require(set(votes) == set(ENGINES), "ACOUSTIC_CONSUMER_THREE_ASR_REQUIRED")
    agreed = observation["conclusion"] == "unanimous-parent-fulltext-match"
    if agreed:
        require(all(v["decoder_terminated"] is True and v["quality_flags"] == [] and
                    v["usable_complete_text"] is True and bool(v["normalized_full_text"]) and
                    v["normalized_full_text"] == observation["normalized_parent_full_text"]
                    for v in votes.values()), "ACOUSTIC_CONSUMER_FALSE_FULLTEXT_AGREEMENT")
    target = fulltext_targets(parent["full_text"], vocabulary)
    require(target["supported"] and target["target_ids"] == parent["target_ids"],
            "ACOUSTIC_CONSUMER_PARENT_VOCABULARY_CHANGED")
    require(type(row["samples"]) is int and row["samples"] > 0 and row["sample_rate_hz"] == 16000,
            "ACOUSTIC_CONSUMER_PCM_GEOMETRY")
    frames = max(0, 1 + (row["samples"] - 400) // 320)
    ids = target["target_ids"]
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
    noise = row["kind"] == "recorded-noise-mix"
    require(row["kind"] in KINDS and row["ctc_mask"] is (not noise) and
            row["ctc_weight"] == (0 if noise else parent["ctc_weight"]) and
            row["target_ids"] == (None if noise else ids) and
            row["full_text"] == (None if noise else parent["full_text"]),
            "ACOUSTIC_CONSUMER_INHERITED_CTC_CHANGED")
    eligible = not noise and agreed and frames >= minimum
    reason = ("untranscribed_additive_background" if noise else
              "acoustic_asr_" + observation["conclusion"] if not agreed else
              "unpadded_c1_ctc_geometry_infeasible" if frames < minimum else "verified_parent_fulltext")
    return {**copy.deepcopy(row), "consumer": ROW, "source_id": row["sample_id"], "item_id": row["sample_id"],
        "source_package_sha256": package_sha, "global_source_key": [package_sha, row["sample_id"]],
        "parent_global_source_key": list(row["parent_key"]), "parent_source_sha256": parent["parent_source_sha256"],
        "reference_pcm_sha256": parent["reference_pcm_sha256"], "reference_wav_sha256": parent["reference_wav_sha256"],
        "audio": binding["variant_wav"]["path"], "audio_path": binding["variant_wav"]["path"],
        "generation_category": "full-positive", "label_status": "machine_candidate" if eligible else "uncertain",
        "label_source": "acoustic-three-asr-confirmed-parent-weak-label" if eligible else "acoustic-observation-quarantined",
        "inherited_variant_labels": copy.deepcopy(observation["inherited_variant_labels"]),
        "asr_observation": copy.deepcopy(observation), "parent_observed_row": copy.deepcopy(parent),
        "full_text": parent["full_text"] if eligible else None, "target_ids": list(ids) if eligible else None,
        "ctc_mask": eligible, "ctc_weight": parent["ctc_weight"] if eligible else 0.0,
        "keyword_targets": list(row["keyword_targets"]) if eligible else [None, None],
        "keyword_mask": list(row["keyword_mask"]) if eligible else [False, False],
        "keyword_weight": list(row["keyword_weight"]) if eligible else [0.0, 0.0],
        "expected_c1_frames": frames, "ctc_minimum_frames": minimum, "ctc_reason": reason,
        "quarantined": not eligible, "keyword_loss_used": False,
        "training_admitted": eligible and row["split"] == "train",
        "original_rows_modified": False, "new_independent_evidence": False}


def project_rows(observed, vocabulary, prior_sources, original_tokens):
    validate_vocabulary(vocabulary, original_tokens)
    require(set(observed) == {"closure", "items"} and len(observed["items"]) == ROWS,
            "ACOUSTIC_CONSUMER_FULL_ROSTER")
    package_sha = observed["closure"]["variant_manifest"]["sha256"]
    require(type(package_sha) is str and training.io_contract.SHA.fullmatch(package_sha),
            "ACOUSTIC_CONSUMER_PACKAGE_HASH")
    roles = GlobalSources()
    alias_fields = ("source_family_id", "speaker_id", "pcm_sha256", "wav_sha256",
                    "parent_source_sha256", "reference_pcm_sha256", "reference_wav_sha256")
    for previous in prior_sources:
        # Repeated ancestors within a split are valid. Distinct derived samples
        # are checked separately; an ancestor is not another independent sample.
        roles.add({"role": previous["role"], **{k: previous.get(k) for k in alias_fields}})
        if previous.get("original_pcm_sha256"):
            roles.add({"role": previous["role"], "pcm_sha256": previous["original_pcm_sha256"]})
    rows, ids, pcm_ids = [], set(), set()
    for item in observed["items"]:
        binding, observation = item["binding"], item["observation"]
        variant, parent = binding["variant_row"], binding["parent_row"]
        require(variant["sample_id"] not in ids and variant["pcm_sha256"] not in pcm_ids,
                "ACOUSTIC_CONSUMER_DUPLICATE_VARIANT")
        ids.add(variant["sample_id"]); pcm_ids.add(variant["pcm_sha256"])
        for source in (parent, variant):
            roles.add({"role": source["source_role"], **{k: source.get(k) for k in alias_fields},
                       "wav_sha256": source["file_sha256"]})
            if source.get("original_pcm_sha256"):
                roles.add({"role": source["source_role"], "pcm_sha256": source["original_pcm_sha256"]})
        asset = binding.get("acoustic_asset")
        require((asset is not None) == (variant["kind"] in ("measured-rir-convolution", "recorded-noise-mix")),
                "ACOUSTIC_CONSUMER_ASSET_ROSTER")
        if asset is not None:
            require(asset["split"] == variant["split"] == variant["recipe"]["asset_split"] and
                    asset["sha256"] == variant["recipe"]["asset_sha256"] and
                    asset["source_group"] == variant["recipe"]["asset_group"],
                    "ACOUSTIC_CONSUMER_ASSET_SPLIT")
            roles.add({"role": variant["source_role"], "wav_sha256": asset["sha256"],
                       "source_family_id": "acoustic-asset-group:" + asset["source_group"]})
        rows.append(project_row(binding, observation, vocabulary, package_sha))
    require(dict(Counter(row["kind"] for row in rows)) == KINDS and
            dict(Counter(row["split"] for row in rows)) == {"train": 100, "development": 24} and
            sum(row["inherited_variant_labels"]["ctc_mask"] for row in rows) == 84,
            "ACOUSTIC_CONSUMER_ORIGINAL_POPULATION_CHANGED")
    return rows


def selected_rows(rows):
    train = [r for r in rows if r["split"] == "train" and r["ctc_mask"]]
    dev = [r for r in rows if r["split"] == "development" and r["ctc_mask"]]
    require(1 <= len(train) <= 60 and len(dev) <= 24, "ACOUSTIC_CONSUMER_NO_BOUNDED_TRAIN_COHORT")
    require(all(r["source_role"] == "train-candidate" and r["generation_category"] == "full-positive" and
                r["training_admitted"] is True and
                r["ctc_weight"] == .5 and r["target_ids"] and not r["quarantined"] for r in train),
            "ACOUSTIC_CONSUMER_POSITIVE_TRAIN_ONLY")
    require(all(r["source_role"] == "development-only" and r["training_admitted"] is False for r in dev),
            "ACOUSTIC_CONSUMER_DEVELOPMENT_ADMISSION")
    return train, dev


def probe_recipe(rows, base):
    require(rows and all(r["split"] == "train" and r["source_role"] == "train-candidate" and
                        r["ctc_mask"] and r["generation_category"] == "full-positive" for r in rows),
            "ACOUSTIC_CONSUMER_DEVELOPMENT_OR_UNKNOWN_DRAW")
    recipe = diagnostic.tiny_recipe(rows, base, STEPS)
    recipe["schema"] = JOB
    return recipe


def prepare(request, output):
    request = validate_request(request)
    destination = Path(output).resolve()
    require(all(not destination.is_relative_to(source.resolve()) for source in
                (Path(request["stage"]).parent, Path(request["observer"]).parents[1],
                 Path(request["anchor"]), ROOT)), "ACOUSTIC_CONSUMER_OUTPUT_INSIDE_FROZEN_INPUT")
    original = historical.anchor(request["anchor"], request["anchor_manifest_sha256"])
    core.verify_original_implementation(original, request["anchor"])
    observed = closed_observer(request)
    rows = project_rows(observed, original["vocabulary"], original["prior_sources"],
                        (Path(request["anchor"]) / "official/tokens.txt").read_bytes())
    train, dev = selected_rows(rows)
    output = training.output_directory(output)
    for row in rows:
        name = "audio/" + row["source_id"] + ".wav"
        require(training.io_contract.ID.fullmatch(row["source_id"]) and
                training.copy_once(row["audio_path"], output / name) == row["file_sha256"],
                "ACOUSTIC_CONSUMER_AUDIO_COPY_CHANGED")
        row["audio"] = row["audio_path"] = name
    for source, target in ((Path(request["anchor"]) / "initial" / (CANDIDATE + ".pt"), "initial.pt"),
                           (Path(request["anchor"]) / "official/tokens.txt", "tokens.txt")):
        training.copy_once(source, output / target)
    require(training.sha_file(output / "initial.pt") == original["initial"]["file_sha256"],
            "ACOUSTIC_CONSUMER_INITIAL_COPY_CHANGED")
    validate_vocabulary(original["vocabulary"], (output / "tokens.txt").read_bytes())
    code = source_records()
    for name, expected in code.items():
        require(training.copy_once(ROOT / name, output / "execution-source" / name) == expected,
                "ACOUSTIC_CONSUMER_SOURCE_COPY_CHANGED")
    for name, value in (("request.json", request), ("anchor.json", original),
                        ("observer-replay.json", observed), ("vocabulary.json", original["vocabulary"])):
        training.io_contract.write_json(output / name, value)
    training.io_contract.write_new(output / "projected-rows.jsonl",
                                  b"".join(training.canonical(row) + b"\n" for row in rows))
    record = {"schema": PREP, "status": "completed", "source_hashes": code,
        "initial_checkpoint": original["initial"], "observer_closure": observed["closure"],
        "row_bindings": [row_identity(r) for r in rows],
        "train_source_ids": [r["source_id"] for r in train], "development_source_ids": [r["source_id"] for r in dev],
        "projection_reasons": dict(Counter(r["ctc_reason"] for r in rows)),
        "eligible_by_split": dict(Counter(r["split"] for r in rows if r["ctc_mask"])),
        "positive_only": True, "ordinary_negative_rows": 0, "source_rows_modified": 0,
        "human_gold": False, "new_model_calls": 0, "quality_acceptance_claim": False}
    training.io_contract.write_json(output / "preparation.json", record)
    core.manifest(output)
    return {"prepared": str(output), "manifest_sha256": training.sha_file(output / "MANIFEST.json"),
            "eligible_by_split": record["eligible_by_split"], "projection_reasons": record["projection_reasons"],
            "new_model_calls": 0}


def verify_prepared(root, expected_sha):
    root = Path(root)
    historical.verify_files(root, expected_sha)
    record = training.read_json(root / "preparation.json")
    request = validate_request(training.read_json(root / "request.json"))
    require(record["schema"] == PREP and record["status"] == "completed" and
            record["source_hashes"] == source_records() and
            all(training.sha_file(root / "execution-source" / name) == sha
                for name, sha in record["source_hashes"].items()), "ACOUSTIC_CONSUMER_PREPARED_SOURCE_CHANGED")
    original = historical.anchor(request["anchor"], request["anchor_manifest_sha256"])
    core.verify_original_implementation(original, request["anchor"])
    observed = closed_observer(request)
    require(original == training.read_json(root / "anchor.json") and
            observed == training.read_json(root / "observer-replay.json") and
            observed["closure"] == record["observer_closure"], "ACOUSTIC_CONSUMER_REPLAY_CHANGED")
    vocabulary = training.read_json(root / "vocabulary.json")
    require(vocabulary == original["vocabulary"] and training.sha_file(root / "tokens.txt") ==
            vocabulary["base_tokens_sha256"] and training.sha_file(root / "initial.pt") ==
            original["initial"]["file_sha256"] == record["initial_checkpoint"]["file_sha256"],
            "ACOUSTIC_CONSUMER_ANCHOR_CHANGED")
    projected = project_rows(observed, vocabulary, original["prior_sources"], (root / "tokens.txt").read_bytes())
    rows = historical.read_lines(root / "projected-rows.jsonl")
    require([row_identity(r) for r in projected] == record["row_bindings"] ==
            [row_identity(r) for r in rows], "ACOUSTIC_CONSUMER_PROJECTED_LABELS_CHANGED")
    for row in rows:
        require(row["audio"] == row["audio_path"] == "audio/" + row["source_id"] + ".wav" and
                training.sha_file(root / row["audio"]) == row["file_sha256"],
                "ACOUSTIC_CONSUMER_RETAINED_AUDIO_CHANGED")
    train, dev = selected_rows(rows)
    require([r["source_id"] for r in train] == record["train_source_ids"] and
            [r["source_id"] for r in dev] == record["development_source_ids"] and
            record["projection_reasons"] == dict(Counter(r["ctc_reason"] for r in rows)) and
            record["eligible_by_split"] == dict(Counter(r["split"] for r in rows if r["ctc_mask"])) and
            record["positive_only"] is True and record["ordinary_negative_rows"] == record["source_rows_modified"] ==
            record["new_model_calls"] == 0 and record["human_gold"] is False and
            record["quality_acceptance_claim"] is False, "ACOUSTIC_CONSUMER_PREPARATION_SUMMARY_CHANGED")
    return record, vocabulary, rows, train, dev, original["base_recipe"]


def validate_job(job):
    require(type(job) is dict and set(job) == {"schema", "prepared", "prepared_manifest_sha256", "candidate",
        "optimizer_steps", "timeout_seconds", "train_binding_sha256", "development_binding_sha256",
        "source_hashes", "positive_only", "quality_acceptance_claim"} and job["schema"] == JOB and
        job["candidate"] == CANDIDATE and job["optimizer_steps"] == STEPS and
        job["timeout_seconds"] == SECONDS and Path(job["prepared"]).is_absolute() and
        job["source_hashes"] == source_records() and job["positive_only"] is True and
        job["quality_acceptance_claim"] is False, "ACOUSTIC_CONSUMER_FIXED_JOB_CHANGED")
    for key in ("prepared_manifest_sha256", "train_binding_sha256", "development_binding_sha256"):
        require(type(job[key]) is str and training.io_contract.SHA.fullmatch(job[key]),
                "ACOUSTIC_CONSUMER_JOB_HASH")
    return job


def declare(prepared, manifest_sha, output):
    require(not Path(output).resolve().is_relative_to(Path(prepared).resolve()),
            "ACOUSTIC_CONSUMER_DECLARATION_OUTSIDE_PREPARED")
    _, _, _, train, dev, _ = verify_prepared(prepared, manifest_sha)
    job = validate_job({"schema": JOB, "prepared": str(Path(prepared).absolute()),
        "prepared_manifest_sha256": manifest_sha, "candidate": CANDIDATE, "optimizer_steps": STEPS,
        "timeout_seconds": SECONDS, "train_binding_sha256": digest([row_identity(r) for r in train]),
        "development_binding_sha256": digest([row_identity(r) for r in dev]), "source_hashes": source_records(),
        "positive_only": True, "quality_acceptance_claim": False})
    training.io_contract.write_new(Path(output), training.io_contract.canonical(job))
    return {"recipe": str(output), "recipe_sha256": training.sha_file(output), "optimizer_steps": STEPS,
            "sample_exposures": STEPS * BATCH_SIZE, "train_rows": len(train), "development_rows": len(dev),
            "new_model_calls": 0}


def queue_request(recipe, lock):
    job = validate_job(training.read_json(recipe))
    record, _, _, _, _, _ = verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    require(str(Path(lock).absolute()) == record["observer_closure"]["heavy_lock"],
            "ACOUSTIC_CONSUMER_CANONICAL_HEAVY_LOCK")
    return {"schema": "kws-local-research-queue-v1", "queue_id": JOB_ID,
        "global_budget_minutes": 10, "jobs": [{"job_id": JOB_ID, "depends_on": [], "timeout_minutes": 10,
        "cwd": str(ROOT), "recipe": str(Path(recipe).absolute()), "recipe_sha256": training.sha_file(recipe),
        "argv": [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "run",
                 "--recipe", "{recipe}", "--output", "{output_dir}", "--heavy-lock", str(Path(lock).absolute())],
        "outputs": list(QUEUE_OUTPUTS)}]}


def consumed_summary(ledger):
    families, categories, words, losses = Counter(), Counter(), Counter(), Counter()
    for event in ledger:
        for draw in event["draws"]:
            families[draw["source_family_id"]] += 1
            categories[draw["generation_category"]] += 1
            losses[draw["generation_category"]] += draw["weighted_batch_contribution"]
            for word, target, mask in zip(("你好小窝", "小窝小窝"), draw["keyword_targets"], draw["keyword_mask"]):
                if target == 1 and mask:
                    words[word] += 1
    return {"optimizer_steps": len(ledger), "sample_exposures": sum(categories.values()),
            "actual_exposures_by_family": dict(families), "actual_exposures_by_category": dict(categories),
            "actual_positive_exposures_by_word": dict(words),
            "actual_weighted_batch_loss_sum_by_category": dict(losses)}


def run(job_path, output, lock):
    job = validate_job(training.read_json(job_path))
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(job_path, lease)
        require(os.environ.get("KWS_RESEARCH_JOB_ID") == JOB_ID and
                os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(job_path).absolute()) and
                os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(Path(output).absolute()),
                "ACOUSTIC_CONSUMER_QUEUE_WORKER_SCOPE")
        output = training.output_directory(output)
        start, deadline = time.monotonic(), time.monotonic() + SECONDS
        status = {"schema": RESULT, "status": "running", "candidate": CANDIDATE, "started_at": training.utc(),
            "job_recipe_sha256": training.sha_file(job_path), "positive_only": True, "ordinary_negative_rows": 0,
            "quality_acceptance_claim": False, "tiny_fit_passed": None, "generalization_passed": None,
            "development_optimizer_draws": 0, "development_selection_or_tuning": False, "human_gold": False,
            "asr_or_tts_calls": 0, "automatic_retry": False, "source_rows_modified": 0,
            "scope": "fixed positive-only acoustic ingestion smoke, not a quality experiment"}
        training.save_json(output / "status.json", status)
        try:
            with core.worker_deadline(SECONDS):
                manifest = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(manifest)
                training.research_runtime.check_packages(manifest, "core")
                prepared = Path(job["prepared"])
                record, vocabulary, all_rows, train, dev, base = verify_prepared(prepared, job["prepared_manifest_sha256"])
                require(str(Path(lock).absolute()) == record["observer_closure"]["heavy_lock"],
                        "ACOUSTIC_CONSUMER_CANONICAL_HEAVY_LOCK")
                require(digest([row_identity(r) for r in train]) == job["train_binding_sha256"] and
                        digest([row_identity(r) for r in dev]) == job["development_binding_sha256"],
                        "ACOUSTIC_CONSUMER_JOB_ROWS_CHANGED")
                torch, reference = training.torch_runtime()
                from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
                features, measured = core.feature_rows(prepared, train, torch)
                dev_features, dev_measured = core.feature_rows(prepared, dev, torch)
                torch.manual_seed(SEED)
                model, description = training.build_fresh(CANDIDATE, vocabulary, reference)
                initial, initial_sha = load_checkpoint_snapshot(prepared / "initial.pt")
                original = training.read_json(prepared / "anchor.json")
                require(initial_sha == record["initial_checkpoint"]["file_sha256"] and
                        initial["schema"] == "kws-d1-initial-checkpoint-v2" and initial["candidate"] == CANDIDATE and
                        initial["optimizer_steps"] == 0 and initial["description"] == description and
                        initial["vocabulary_sha256"] == digest(vocabulary) and initial["source_hashes"] ==
                        original["source_hashes"], "ACOUSTIC_CONSUMER_EXACT_ZERO_STEP_INITIAL_CHANGED")
                model.load_state_dict(initial["state_dict"], strict=True)
                require(state_identity(model.state_dict()) == initial["float_state_identity"],
                        "ACOUSTIC_CONSUMER_INITIAL_LOAD_CHANGED")
                for source, target in ((prepared / "initial.pt", "initial.pt"),
                                       (prepared / "projected-rows.jsonl", "all-rows.jsonl"),
                                       (job_path, "request.json")):
                    training.copy_once(source, output / target)
                for name, value in (("train-rows.json", train), ("development-rows.json", dev),
                                    ("feature-observations.json", measured + dev_measured)):
                    training.io_contract.write_json(output / name, value)
                recipe = probe_recipe(train, base)
                training.io_contract.write_json(output / "exposure-recipe.json", recipe)
                before = diagnostic.observe(model, features, train, "before", output, deadline)
                spec = recipe["optimizers"]["fresh-rnn64-or-c1"]
                optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
                model.train()
                consumed = training.train_batches(model, features, train, recipe, optimizer, output=output, deadline=deadline)
                gradients = historical.ledger_readback(historical.read_lines(output / "consumption.jsonl"), recipe, train)
                names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
                changed, delta = diagnostic.parameter_changes(initial["state_dict"], model.state_dict(), names)
                identity = state_identity(model.state_dict())
                after = diagnostic.observe(model, features, train, "after", output, deadline)
                development = diagnostic.observe(model, dev_features, dev, "development", output, deadline)
                require(state_identity(model.state_dict()) == identity, "ACOUSTIC_CONSUMER_READBACK_MUTATED_MODEL")
                atomic_save_checkpoint({"schema": RESULT, "candidate": CANDIDATE, "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "float_state_identity": identity,
                    "job_recipe_sha256": training.sha_file(job_path), "optimizer_steps": STEPS,
                    "initial_checkpoint_sha256": initial_sha, "vocabulary_sha256": digest(vocabulary)}, output / "final.pt")
                final, final_sha = load_checkpoint_snapshot(output / "final.pt")
                require(state_identity(final["state_dict"]) == identity and consumed["optimizer_steps"] == STEPS and
                        consumed["sample_exposures"] == STEPS * BATCH_SIZE, "ACOUSTIC_CONSUMER_TRAINING_READBACK")
                status.update(consumed, status="completed", finished_at=training.utc(), wall_seconds=time.monotonic()-start,
                    prepared_manifest_sha256=job["prepared_manifest_sha256"], train_binding_sha256=job["train_binding_sha256"],
                    development_binding_sha256=job["development_binding_sha256"], vocabulary_sha256=digest(vocabulary),
                    initial_checkpoint_sha256=initial_sha, initial_float_state_sha256=initial["float_state_identity"]["sha256"],
                    final_checkpoint_sha256=final_sha, final_float_state_sha256=identity["sha256"],
                    trainable_parameter_names=names, changed_parameter_names=changed, parameter_delta_l2=delta,
                    optimizer_readback=core.optimizer_readback(final["optimizer_state"], final["state_dict"], names, spec),
                    gradients=gradients, train_rows=len(train), development_rows=len(dev), all_observed_rows=len(all_rows),
                    training_model_forwards=STEPS*BATCH_SIZE, train_diagnostic_model_forwards=2*len(train),
                    development_forward_calls=len(dev), development_readback_passes=1, model_constructions=1,
                    before=core.observed_summary(before), after=core.observed_summary(after),
                    development=core.observed_summary(development),
                    consumption_sha256=training.sha_file(output / "consumption.jsonl"), consumption_verified=True,
                    projection_reasons=record["projection_reasons"],
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                training.save_json(output / "status.json", status)
                training.io_contract.write_json(output / "summary.json", status)
                core.manifest(output)
                return status
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error),
                          wall_seconds=time.monotonic()-start)
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected_sha):
    """Read saved tensors and recompute CTC scores, with no model construction."""
    output = Path(output)
    historical.verify_files(output, expected_sha)
    job = validate_job(training.read_json(output / "request.json"))
    summary = training.read_json(output / "summary.json")
    record, vocabulary, all_rows, train, dev, base = verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    require(summary["schema"] == RESULT and summary["status"] == "completed" and
            summary["candidate"] == CANDIDATE and summary["optimizer_steps"] == STEPS and
            summary["sample_exposures"] == STEPS*BATCH_SIZE and summary["all_observed_rows"] == ROWS and
            summary["positive_only"] is True and summary["ordinary_negative_rows"] == summary["asr_or_tts_calls"] ==
            summary["source_rows_modified"] == summary["development_optimizer_draws"] == 0 and
            summary["human_gold"] is False and summary["automatic_retry"] is False and
            summary["quality_acceptance_claim"] is False and summary["development_selection_or_tuning"] is False and
            summary["tiny_fit_passed"] is summary["generalization_passed"] is None and
            summary["train_rows"] == len(train) and summary["development_rows"] == len(dev) and
            summary["model_constructions"] == 1 and summary["training_model_forwards"] == STEPS*BATCH_SIZE and
            summary["train_diagnostic_model_forwards"] == 2*len(train) and
            summary["development_forward_calls"] == len(dev) and summary["development_readback_passes"] == 1 and
            summary["job_recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["train_binding_sha256"] == job["train_binding_sha256"] == digest([row_identity(r) for r in train]) and
            summary["development_binding_sha256"] == job["development_binding_sha256"] == digest([row_identity(r) for r in dev]) and
            summary["prepared_manifest_sha256"] == job["prepared_manifest_sha256"] and
            summary["vocabulary_sha256"] == digest(vocabulary) and
            summary["projection_reasons"] == record["projection_reasons"] and summary["consumption_verified"] is True and
            training.read_json(output / "train-rows.json") == train and
            training.read_json(output / "development-rows.json") == dev and
            historical.read_lines(output / "all-rows.jsonl") == all_rows, "ACOUSTIC_CONSUMER_RESULT_SCOPE_CHANGED")
    recipe = probe_recipe(train, base)
    ledger = historical.read_lines(output / "consumption.jsonl")
    require(training.read_json(output / "exposure-recipe.json") == recipe and
            historical.ledger_readback(ledger, recipe, train) == summary["gradients"] and
            training.sha_file(output / "consumption.jsonl") == summary["consumption_sha256"] and
            all(summary[key] == value for key, value in consumed_summary(ledger).items()),
            "ACOUSTIC_CONSUMER_LEDGER_CHANGED")
    torch, _ = training.torch_runtime()
    from training_state import load_checkpoint_snapshot, state_identity
    initial, initial_sha = load_checkpoint_snapshot(output / "initial.pt")
    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
    require(initial_sha == summary["initial_checkpoint_sha256"] == record["initial_checkpoint"]["file_sha256"] and
            state_identity(initial["state_dict"]) == initial["float_state_identity"] and
            initial["float_state_identity"]["sha256"] == summary["initial_float_state_sha256"] and
            final_sha == summary["final_checkpoint_sha256"] and final["schema"] == RESULT and
            final["candidate"] == CANDIDATE and final["optimizer_steps"] == STEPS and
            final["vocabulary_sha256"] == digest(vocabulary) and final["initial_checkpoint_sha256"] == initial_sha and
            final["job_recipe_sha256"] == summary["job_recipe_sha256"] and
            state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"],
            "ACOUSTIC_CONSUMER_CHECKPOINT_CHANGED")
    changed, delta = diagnostic.parameter_changes(initial["state_dict"], final["state_dict"], summary["trainable_parameter_names"])
    require(changed == summary["changed_parameter_names"] and delta == summary["parameter_delta_l2"] and
            core.optimizer_readback(final["optimizer_state"], final["state_dict"], summary["trainable_parameter_names"],
                recipe["optimizers"]["fresh-rnn64-or-c1"]) == summary["optimizer_readback"],
            "ACOUSTIC_CONSUMER_OPTIMIZER_READBACK_CHANGED")
    feature_count = audit_numerics(output, Path(job["prepared"]), train, dev, summary, torch)
    return {"schema": "kws-acoustic-consumption-readback-v1", "status": "verified",
        "manifest_sha256": expected_sha, "optimizer_steps_replayed": STEPS, "new_optimizer_steps": 0,
        "new_model_forwards": 0, "model_constructions": 0, "train_rows": len(train), "development_rows": len(dev),
        "development_optimizer_draws": 0, "feature_rows_replayed": feature_count,
        "consumption_verified": True, "quality_acceptance_claim": False}


def audit_numerics(output, prepared, train, dev, summary, torch):
    """Replay retained PCM and saved logits; this path never constructs a model."""
    feature_rows = training.read_json(output / "feature-observations.json")
    features = {r["source_id"]: r for r in feature_rows}
    require(len(features) == len(feature_rows) and set(features) == {r["source_id"] for r in train + dev},
            "ACOUSTIC_CONSUMER_FEATURE_ROSTER")
    _, recomputed_features = core.feature_rows(prepared, train + dev, torch)
    require(recomputed_features == feature_rows, "ACOUSTIC_CONSUMER_FEATURE_PCM_REPLAY_CHANGED")
    for phase, selected in (("before", train), ("after", train), ("development", dev)):
        observations = training.read_json(output / (phase + ".json"))
        require(len(observations) == len(selected), "ACOUSTIC_CONSUMER_NUMERIC_ROSTER")
        for row, item in zip(selected, observations):
            numeric = item["numeric_artifact"]
            path = output / training.io_contract.relative(numeric["path"])
            require(training.io_contract.file_record(path) == {k: numeric[k] for k in ("bytes", "sha256")},
                    "ACOUSTIC_CONSUMER_NUMERIC_HASH")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            feature = features[row["source_id"]]
            require(item["source_id"] == saved["source_id"] == row["source_id"] and
                    item["label_binding_sha256"] == row_identity(row) and item["pcm_sha256"] == row["pcm_sha256"] and
                    all(item[key] == row[key] for key in ("target_ids", "ctc_mask", "ctc_weight")) and
                    training.tensor_sha(saved["feature"]) == saved["feature_sha256"] == feature["feature_sha256"] and
                    feature["split"] == row["split"] and feature["pcm_sha256"] == row["pcm_sha256"] and
                    tuple(saved["feature"].shape) == (feature["frames"], 32) and
                    feature["frames"] == row["expected_c1_frames"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                    all(item[k] == v for k, v in diagnostic.score(saved["logits"], row).items()),
                    "ACOUSTIC_CONSUMER_SAVED_NUMERIC_CHANGED")
        require(core.observed_summary(observations) == summary[phase], "ACOUSTIC_CONSUMER_NUMERIC_SUMMARY")
    return len(feature_rows)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--request", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ("verify", "declare"):
        command = sub.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        if name == "declare":
            command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("queue-request")
    command.add_argument("--recipe", type=Path, required=True)
    command.add_argument("--heavy-lock", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--recipe", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--heavy-lock", type=Path, required=True)
    command = sub.add_parser("audit")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--manifest-sha256", required=True)
    args = cli.parse_args(argv)
    if args.command == "prepare":
        result = prepare(training.read_json(args.request), args.output)
    elif args.command == "verify":
        record, _, rows, train, dev, _ = verify_prepared(args.prepared, args.manifest_sha256)
        result = {"status": "verified", "observed_rows": len(rows), "train_rows": len(train),
                  "development_rows": len(dev), "projection_reasons": record["projection_reasons"], "new_model_calls": 0}
    elif args.command == "declare":
        result = declare(args.prepared, args.manifest_sha256, args.output)
    elif args.command == "queue-request":
        result = queue_request(args.recipe, args.heavy_lock)
    elif args.command == "run":
        result = run(args.recipe, args.output, args.heavy_lock)
    else:
        result = audit(args.output, args.manifest_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
