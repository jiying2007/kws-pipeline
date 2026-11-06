#!/usr/bin/env python3
"""Fixed C1/2601 source augmentation, with a saved full52 reference model.

Only the roster adapter is new. The prior known-zero-prefix loss, 8000-forward
training loop, frontend, optimizer accounting and numerical observations are
reused. Metadata projection preserves the complete observed weak transcript;
generation intent is a diagnostic, never supervision or an admission filter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_context52_known_silence as prior
import research_d1_inputs as d1_inputs
import research_d1_recipe as d1_recipe
from research_d1_vocab import validate_vocabulary

context, silence, diagnostic = prior.context, prior.silence, prior.diagnostic
onset, training, fit, evaluation = prior.onset, prior.training, prior.fit, prior.evaluation
io, require = prior.io, prior.require
SOURCE = "training/research_c1_source_augmentation.py"
CODE = tuple(dict.fromkeys((*prior.CODE, SOURCE)))
SCHEMA = "kws-c1-source-augmentation-v1"
REQUEST = "kws-c1-source-augmentation-request-v1"
ACCOUNTING = "kws-newsource64-terminal-accounting-v1"
POLICY = "observed-D1-weak-fulltext-C1-only-v1"
BASELINE_SHA = "92d1d0b3d98e39bb9d5739d5bce80dc4a93bd7eb2e6a97f6fd55308012b19857"
BASELINE_RECIPE_SHA = "d5d10948fd2381fc1fa1c465953e01dcd72ba0e1356469cb3c2a98491f81affb"
BASELINE_MANIFEST_SHA = "3dc3f3f8159fff515cc2a20ff68ed33ca99b388d984a50da1a81bf96cf2c5fdb"
NEW_PLAN_SHA = "8ff19752b173cd8097ad8a612bd3329d44ae679419e8c54d30a69e73ad88ebd9"
MODES = ("baseline-evaluation", "augmentation2000")
OFFSETS = (0, 200, 500)
STEPS, DRAWS, BATCH = 2000, 8000, 4


def pin(path):
    return onset.pin(path)


def checked(value):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"} and
        pin(value["path"]) == value, "AUGMENTATION_PIN_CHANGED")
    return Path(value["path"])


def contract(mode):
    require(mode in MODES, "AUGMENTATION_MODE")
    learn = mode == "augmentation2000"
    return {"schema": SCHEMA, "mode": mode, "candidate": "c1-ctc2601", "projection_policy": POLICY,
        "optimizer_steps": STEPS if learn else 0, "batch_size": BATCH, "training_forward_limit": DRAWS if learn else 0,
        "final_forward_limit_maximum": 402 if learn else 192, "worker_seconds": 840 if learn else 270,
        "queue_minutes": 15 if learn else 5,
        "threads": 4, "interop_threads": 1, "maximum_model_jobs_concurrently": 1, "model_construction_limit": 1,
        "seed": 20261105, "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01},
        "fixed_ctc_normalizer": 1.0, "gradient_clip_max_norm": 5.0, "auxiliary_scale": 1.0,
        "offsets_ms": list(OFFSETS), "known_zero_frames": {"0": 0, "200": 9, "500": 24},
        "asr_or_tts_calls": 0, "automatic_retry": False, "human_gold": False, "independent_test": False,
        "previously_exposed_development": True, "development_optimizer_draws": 0,
        "generation_intent_is_supervision": False, "single_variable_data_size_ablation": False,
        "threshold_selection": False, "checkpoint_selection": False, "quality_winner": None,
        "generalization_passed": None, "training_logits_retained": False,
        "full_float32_final_features_and_logits_retained": True,
        "complete_optimization_numerically_replayed": False}


def project_row(row, vocabulary):
    """C1-only geometry; no class/intent gate and no transcript substitution."""
    require(row.get("consumer") == "d1-verified-full-transcript-v2" and
        row.get("purpose") == "technical-research-only" and row.get("human_gold") is False and
        row.get("independent_test") is False and row["split"] in ("train", "development") and
        row["source_role"] == ("train-candidate" if row["split"] == "train" else "development-only") and
        row["global_source_key"] == [row["source_plan_sha256"], row["source_id"]], "AUGMENTATION_ROW_SCOPE")
    confirmed = row["label_status"] == "machine_candidate"
    require(row["label_status"] in ("machine_candidate", "uncertain") and
        type(row["ctc_mask"]) is bool and row["ctc_mask"] == confirmed and
        type(row["ctc_weight"]) in (int, float) and math.isfinite(row["ctc_weight"]) and
        (0 < row["ctc_weight"] <= 1 if confirmed else row["ctc_weight"] == 0), "AUGMENTATION_WEAK_MASK_WEIGHT")
    require(not confirmed or type(row["full_text"]) is str and bool(row["full_text"]), "AUGMENTATION_FULL_TEXT_REQUIRED")
    require(all(type(row[k]) is list and len(row[k]) == 2 for k in
        ("keyword_targets", "keyword_mask", "keyword_weight")), "AUGMENTATION_KEYWORD_SHAPE")
    for text, target, mask, weight in zip(fit.KEYWORDS, row["keyword_targets"], row["keyword_mask"], row["keyword_weight"]):
        require(type(mask) is bool and type(weight) in (int, float) and math.isfinite(weight) and
            ((confirmed and type(target) is int and target == int(text in row["full_text"]) and 0 < weight <= 1)
             if mask else target is None and weight == 0), "AUGMENTATION_UNKNOWN_NOT_NEGATIVE")
    require(type(row["samples"]) is int and row["samples"] >= 400, "AUGMENTATION_PCM_GEOMETRY")
    outcome = row.get("generation_outcome")
    require(outcome in (None, "completed", "failed-native") and
        (outcome != "failed-native" or not confirmed and row["full_text"] is None), "AUGMENTATION_FAILED_NATIVE_LABEL")
    target = fit.fulltext_targets(row["full_text"] if confirmed else None, vocabulary)
    frames, minimum = onset.geometry(row["samples"], 0)["frames"], None
    if target["supported"]:
        ids = target["target_ids"]
        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if frames < minimum:
            target = {**target, "supported": False, "reason": "unpadded_c1_ctc_geometry_infeasible", "target_ids": None}
    category, intended = row["generation_category"], row.get("intended_keyword_id")
    require(category in ("full-positive", "ordinary-negative", "partial-negative", "near-negative") and
        (intended is None or type(intended) is int and intended in (1, 2)), "AUGMENTATION_GENERATION_METADATA")
    observed_positive = any(mask and value == 1 for value, mask in zip(row["keyword_targets"], row["keyword_mask"]))
    conflict = confirmed and (observed_positive != (category == "full-positive") or
        category == "full-positive" and intended is not None and row["keyword_targets"][intended - 1] != 1)
    eligible = confirmed and target["supported"]
    return {**row, "consumer": POLICY, "vocabulary_sha256": fit.digest(vocabulary),
        "observed_label": {key: row[key] for key in
            ("full_text", "ctc_mask", "ctc_weight", "keyword_targets", "keyword_mask", "keyword_weight")},
        "ctc_mask": eligible, "ctc_weight": row["ctc_weight"] if eligible else 0.0,
        "target_ids": target["target_ids"] if eligible else None, "ctc_reason": target["reason"],
        "oov_characters": target["oov_characters"], "expected_c1_frames": frames, "ctc_minimum_frames": minimum,
        "intent_label_conflict_diagnostic": conflict, "intent_conflict_used_for_filtering": False,
        "shared_three_model_geometry_used": False, "quarantined": not eligible, "keyword_loss_used": False}


def exposure_plan(rows):
    require(rows and all(r["split"] == "train" and r["ctc_mask"] for r in rows), "AUGMENTATION_TRAIN_ONLY")
    n = len(rows)
    return [{"row_index": i % n, "row_visit_index": i // n, "source_id": rows[i % n]["source_id"],
        "ctc_weight": rows[i % n]["ctc_weight"], "offset_ms": OFFSETS[(i // n) % 3],
        "known_zero_prefix_frames": (0, 9, 24)[(i // n) % 3]} for i in range(DRAWS)]


def role_check(old_rows, new_rows):
    require(len({r["source_id"] for r in old_rows + new_rows}) == len(old_rows) + len(new_rows),
        "AUGMENTATION_DUPLICATE_OUTPUT_SOURCE_ID")
    old = set().union(*(evaluation.identity_keys(r) for r in old_rows))
    new = set().union(*(evaluation.identity_keys(r) for r in new_rows)) if new_rows else set()
    require(not old & new, "AUGMENTATION_NEW_SOURCE_OVERLAPS_OLD_CONTEXT")
    split = {name: set().union(*(evaluation.identity_keys(r) for r in old_rows + new_rows if r["split"] == name))
        for name in ("train", "development")}
    require(not split["train"] & split["development"], "AUGMENTATION_SOURCE_ROLE_LEAK")


def validate_accounting(value, cells, rows):
    require(type(value) is dict and value.get("schema") == ACCOUNTING and value.get("terminal") is True and
        value.get("plan_sha256") == NEW_PLAN_SHA and value.get("selected_cell_ids") == [c["cell_id"] for c in cells] and
        len(value.get("entries", [])) == 64 and value.get("source_pins"), "AUGMENTATION_TERMINAL_ACCOUNTING")
    for source in value["source_pins"]:
        checked(source)
    seen = {r["source_id"] for r in rows}
    by_id = {r["source_id"]: r for r in rows}
    for cell, item in zip(cells, value["entries"]):
        require(item.get("cell_id") == cell["cell_id"] and
            item.get("tts_status") in ("completed", "failed", "unknown", "not_started") and
            item.get("asr_status") in ("completed", "partial", "not_run") and
            type(item.get("weak_row_present")) is bool and item["weak_row_present"] == (cell["cell_id"] in seen) and
            type(item.get("evidence")) is list and item["evidence"], "AUGMENTATION_CELL_ACCOUNTING")
        for evidence in item["evidence"]:
            checked(evidence)
        if item["weak_row_present"]:
            row = by_id[item["cell_id"]]
            require(item["tts_status"] == ("failed" if row.get("generation_outcome") == "failed-native" else "completed") and
                item["asr_status"] in ("completed", "partial"), "AUGMENTATION_ACCOUNTED_ROW_STATE")
    return value


def assemble(request, *, require_terminal=True):
    require(type(request) is dict and set(request) == {"schema", "baseline_output", "baseline_audit", "new_plan",
        "packages", "view_loader", "terminal_accounting", "global_budget"} and request["schema"] == REQUEST, "AUGMENTATION_REQUEST")
    budget = training.read_json(checked(request["global_budget"]))
    jobs = {j["id"]: j for j in budget["jobs"]}
    require(budget["schema"] == "kws-balanced64-global-budget-v1" and budget["global_runner_minutes_ceiling"] == 370 and
        budget["heavy_model_concurrency"] == 1 and budget["threads"] == 4 and budget["interop_threads"] == 1 and
        jobs["c1-baseline-new64"]["queue_wall_seconds"] == 300 and
        jobs["c1-baseline-new64"]["maximum_new_inference_forwards"] == 192 and
        jobs["c1-augmentation-fixed2000"]["queue_wall_seconds"] == 900 and
        jobs["c1-augmentation-fixed2000"]["maximum_optimizer_steps"] == STEPS and
        jobs["c1-augmentation-fixed2000"]["maximum_training_forwards"] == DRAWS and
        jobs["c1-augmentation-fixed2000"]["maximum_final_inference_forwards"] == 402,
        "AUGMENTATION_ROOT_BUDGET_CHANGED")
    output = io.safe_path(request["baseline_output"])
    onset.verify_manifest(output, BASELINE_MANIFEST_SHA)
    base = training.read_json(output / "request.json")
    summary = training.read_json(output / "summary.json")
    prior.completed_queue(output, summary)
    prior.terminal_counts(summary)
    require(training.sha_file(output / "request.json") == BASELINE_RECIPE_SHA and
        training.sha_file(output / "final.pt") == summary["final_checkpoint_sha256"] == BASELINE_SHA and
        base["initial_checkpoint"]["sha256"] == silence.INITIAL_SHA == summary["initial_checkpoint_sha256"] and
        base["initial_binding"]["float_state_sha256"] == summary["initial_float_state_sha256"],
        "AUGMENTATION_FIXED_BASELINE_CHANGED")
    audit = training.read_json(checked(request["baseline_audit"]))
    require(audit["status"] == "verified" and audit["manifest_sha256"] == BASELINE_MANIFEST_SHA and
        audit["saved_training_forwards"] == DRAWS and audit["saved_optimizer_steps"] == STEPS and
        audit["saved_final_conditions"] == 210 and audit["new_model_forwards"] == 0, "AUGMENTATION_BASELINE_AUDIT")
    original_tokens = checked(base["tokens_file"]).read_bytes()
    vocabulary = validate_vocabulary(training.read_json(checked(base["vocabulary_file"])), original_tokens)
    require(fit.digest(vocabulary) == base["vocabulary_sha256"], "AUGMENTATION_VOCABULARY_CHANGED")
    checked(base["initial_checkpoint"])
    plan = training.read_json(checked(request["new_plan"]))
    require(fit.digest(plan) == NEW_PLAN_SHA and len(plan["cells"]) == 256, "AUGMENTATION_ORIGINAL_NEW_PLAN")
    cells = plan["cells"][:64]
    by_id = {c["cell_id"]: c for c in cells}
    require(len(by_id) == 64 and Counter(c["role"] for c in cells) ==
        {"train-candidate": 48, "development-only": 16}, "AUGMENTATION_FIXED_FIRST64")
    loader, _, loader_sources = d1_inputs.load_data_modules(checked(request["view_loader"]))
    raw, aggregate = d1_inputs.aggregate_batches(request["packages"], loader, expected_plan_sha256=NEW_PLAN_SHA)
    require(raw and {r["source_id"] for r in raw} <= set(by_id), "AUGMENTATION_OUTSIDE_FIRST64")
    raw.sort(key=lambda row: list(by_id).index(row["source_id"]))
    for row in raw:
        cell = by_id[row["source_id"]]
        require(row["source_role"] == cell["role"] and row["source_family_id"] == cell["source_family_id"] and
            row["generation_category"] == cell["generation_category"] and
            row["intended_keyword_id"] == cell["intended_keyword_id"], "AUGMENTATION_ORIGINAL_CELL_ROLE")
    new = [project_row(row, vocabulary) for row in raw]
    old = base["train"] + base["development"] + base["excluded_original_rows"]
    role_check(old, new)
    require(len(base["train"]) == 52 and len(base["development"]) == 18 and len(base["excluded_original_rows"]) == 2,
        "AUGMENTATION_ORIGINAL_CONTEXT_ROSTER")
    accounting = None
    if request["terminal_accounting"] is not None:
        accounting = validate_accounting(training.read_json(checked(request["terminal_accounting"])), cells, new)
    require(not require_terminal or accounting is not None, "AUGMENTATION_UPSTREAM_TERMINAL_REQUIRED")
    train_new = [r for r in new if r["split"] == "train" and r["ctc_mask"]]
    dev_new = [r for r in new if r["split"] == "development"]
    require(not require_terminal or train_new, "AUGMENTATION_NO_NEW_TRAIN_SUPERVISION")
    train = base["train"] + train_new
    development = base["development"] + dev_new
    require(len(train) <= 100 and len(development) <= 34, "AUGMENTATION_ROSTER_BOUND")
    require(budget["shared_heavy_lock"] == base["heavy_lock"], "AUGMENTATION_ROOT_LOCK_CHANGED")
    # Keep the adopted old row dictionaries byte-equivalent in the recipe. File
    # relocation is explicit and happens only in the new execution output.
    audio = {r["source_id"]: pin(Path(base["prepared"]) / r["audio"]) for r in old}
    audio.update({r["source_id"]: pin(r["audio_path"]) for r in new})
    require(all(audio[r["source_id"]]["sha256"] == r["file_sha256"] for r in old + new), "AUGMENTATION_AUDIO_PIN")
    return {"base": base, "base_summary": summary, "vocabulary": vocabulary, "new": new,
        "train": train, "development": development, "audio": audio, "aggregate": aggregate,
        "loader_sources": loader_sources, "accounting": accounting,
        "missing_weak_rows": [c["cell_id"] for c in cells if c["cell_id"] not in {r["source_id"] for r in new}]}


def declare(request, mode):
    data = assemble(request)
    base, train, dev = data["base"], data["train"], data["development"]
    learn = mode == "augmentation2000"
    unevaluable_train = [r for r in data["new"] if r["split"] == "train" and not r["ctc_mask"]]
    rows = train + unevaluable_train + dev if learn else data["new"]
    draws = exposure_plan(train) if learn else []
    job = {**contract(mode), "inputs": request, "job_id": "c1-newsource64-" + mode,
        "train": train, "development": dev, "evaluation_rows": rows, "all_new_rows": data["new"],
        "excluded_original_rows": base["excluded_original_rows"], "missing_weak_rows": data["missing_weak_rows"],
        "new_zero_loss_rows": [r["source_id"] for r in data["new"] if not r["ctc_mask"]],
        "aggregate": data["aggregate"], "loader_sources": data["loader_sources"], "audio_files": data["audio"],
        "initial_checkpoint": base["initial_checkpoint"], "initial_binding": base["initial_binding"],
        "initial_source_hashes": base["initial_source_hashes"], "model_description": base["model_description"],
        "canonical_parameter_names": base["canonical_parameter_names"], "vocabulary_sha256": base["vocabulary_sha256"],
        "vocabulary_file": base["vocabulary_file"], "tokens_file": base["tokens_file"], "heavy_lock": base["heavy_lock"],
        "baseline_checkpoint": pin(Path(request["baseline_output"]) / "final.pt"),
        "baseline_float_state_sha256": data["base_summary"]["final_float_state_sha256"],
        "source_hashes": {name: training.sha_file(ROOT / name) for name in CODE},
        "exposure_plan": {"draws": draws}, "exposure_plan_sha256": fit.digest(draws),
        "exposures_by_source": dict(Counter(d["source_id"] for d in draws)),
        "exposures_by_offset": {str(ms): sum(d["offset_ms"] == ms for d in draws) for ms in OFFSETS},
        "train_binding_sha256": fit.digest([fit.row_identity(r) for r in train]),
        "development_binding_sha256": fit.digest([fit.row_identity(r) for r in dev]),
        "evaluation_binding_sha256": fit.digest([fit.row_identity(r) for r in rows]),
        "final_forward_limit": len(rows) * 3,
        "final_logits_payload_bytes": sum(onset.geometry(r["samples"], ms)["frames"] * 2601 * 4 for r in rows for ms in OFFSETS),
        "final_features_payload_bytes": sum(onset.geometry(r["samples"], ms)["frames"] * 32 * 4 for r in rows for ms in OFFSETS)}
    return job


def validate_contract(job):
    fixed = contract(job.get("mode"))
    require(io.canonical({k: job.get(k) for k in fixed}) == io.canonical(fixed) and
        job["job_id"] == "c1-newsource64-" + job["mode"] and
        0 < job["final_forward_limit"] <= job["final_forward_limit_maximum"], "AUGMENTATION_FIXED_CONTRACT")


def validate_recipe(job):
    validate_contract(job)
    require(io.canonical(declare(job["inputs"], job["mode"])) == io.canonical(job), "AUGMENTATION_RECIPE_CHANGED")
    return training.read_json(checked(job["vocabulary_file"]))


def observed_score(row, model_pin, pcm, ms, feature, logits, baseline, baseline_feature, history, artifact, vocabulary):
    if row["ctc_mask"]:
        return {**diagnostic.observation(row, model_pin, pcm, ms, feature, logits, baseline,
            baseline_feature, history, artifact, vocabulary), "evaluation_status": "evaluated"}
    # Save the complete numerical result even when no full weak target exists.
    # No absent label is turned into a negative or a synthetic CTC target.
    greedy = evaluation.collapse_ctc(logits[:, 0].argmax(-1).tolist())
    zero_frames = onset.geometry(row["samples"], ms)["fully_digital_silence_frames"]
    emissions = onset.emission_locations(logits[:, 0].argmax(-1).tolist(), ms)
    return {"source_id": row["source_id"], "source_family_id": row["source_family_id"], "split": row["split"],
        "offset_ms": ms, "label_binding_sha256": fit.row_identity(row), "evaluation_status": "not-evaluable",
        "reason": row.get("ctc_reason", "no_confirmed_full_transcript"), "ctc_weight": row["ctc_weight"],
        **silence.prefix_observation(logits, zero_frames),
        "numeric_artifact": artifact, "model": model_pin, "full_text": row["full_text"],
        "observed_label": row.get("observed_label"), "pcm_sha256": row["pcm_sha256"],
        "greedy_token_ids": greedy, "keyword_targets": row["keyword_targets"], "keyword_mask": row["keyword_mask"],
        "keyword_weight": row["keyword_weight"], "keywords": diagnostic.keyword_observations(greedy, row, vocabulary),
        "emissions_entirely_in_added_silence": sum(e["window_entirely_in_added_silence"] for e in emissions),
        "feature_sha256": training.tensor_sha(feature), "logits_sha256": training.tensor_sha(logits)}


def observe(job, output, bank, pcms, vocabulary, checkpoint_sha, torch, *, model=None, end=None, status=None):
    rows, observed, calls = job["evaluation_rows"], [], []
    saved_rows = context.lines(output / "observations.jsonl") if model is None else None
    require(saved_rows is None or len(saved_rows) == job["final_forward_limit"], "AUGMENTATION_SAVED_COUNT")
    model_pin = {"model_id": job["job_id"], "checkpoint": {"sha256": checkpoint_sha}}
    history = sum((k - 1) * d for k, d in zip(job["model_description"]["kernels"], job["model_description"]["dilations"]))
    with torch.inference_mode():
        for index, row in enumerate(rows):
            baseline = None
            for ms in OFFSETS:
                feature = bank[index, ms]
                name = "numeric/%s/ms-%03d/%s.pt" % (row["split"], ms, row["source_id"])
                call = prior.call_record("final", len(observed), row, ms, training.tensor_sha(feature))
                if model is not None:
                    require(time.monotonic() < end and status["final_forwards_started"] < job["final_forward_limit"], "AUGMENTATION_FINAL_BUDGET")
                    training.append_json(output / "forward-calls.jsonl", {**call, "phase": "started"})
                    status["final_forwards_started"] += 1
                    logits = model(feature.unsqueeze(0))
                    diagnostic.check_logits(logits, feature, torch)
                    status["final_forwards_completed"] += 1
                    training.append_json(output / "forward-calls.jsonl", {**call, "phase": "completed", "logits_sha256": training.tensor_sha(logits)})
                    artifact = {"path": name, **evaluation.save_numeric(output / name, row["source_id"], feature, logits)}
                else:
                    item = saved_rows[len(observed)]
                    artifact = item["numeric_artifact"]
                    require(artifact == {"path": name, **io.file_record(output / name)}, "AUGMENTATION_SAVED_NUMERIC_PIN")
                    saved = torch.load(output / name, weights_only=True, map_location="cpu")
                    logits = saved["logits"]
                    diagnostic.check_logits(logits, feature, torch)
                    require(saved["source_id"] == row["source_id"] and saved["feature"].dtype == torch.float32 and
                        saved["feature"].equal(feature) and saved["feature_sha256"] == training.tensor_sha(feature) and
                        saved["logits_sha256"] == training.tensor_sha(logits), "AUGMENTATION_SAVED_NUMERIC_CHANGED")
                if ms == 0:
                    baseline = logits
                item = observed_score(row, model_pin, pcms[index], ms, feature, logits, baseline,
                    bank[index, 0], history, artifact, vocabulary)
                observed.append(item)
                calls.extend(({**call, "phase": "started"}, {**call, "phase": "completed", "logits_sha256": training.tensor_sha(logits)}))
                if model is not None:
                    training.append_json(output / "observations.jsonl", item)
                else:
                    require(saved_rows[len(observed) - 1] == item, "AUGMENTATION_SAVED_METRICS_CHANGED")
    return observed, calls


def conditions(observations, job):
    require([(r["source_id"], r["split"], r["offset_ms"]) for r in observations] ==
        [(r["source_id"], r["split"], ms) for r in job["evaluation_rows"] for ms in OFFSETS], "AUGMENTATION_EVAL_ROSTER")
    new_ids = {r["source_id"] for r in job["all_new_rows"]}
    result = {}
    for cohort in ("old-train", "old-development", "new-train", "new-development"):
        origin, split = cohort.split("-", 1)
        result[cohort] = {}
        for ms in OFFSETS:
            group = [r for r in observations if r["split"] == split and r["offset_ms"] == ms and
                (r["source_id"] in new_ids) == (origin == "new")]
            scored = [r for r in group if r.get("evaluation_status") == "evaluated"]
            keywords = []
            for i in range(2):
                known = [r["keywords"][i] for r in group if r["keywords"][i]["known"]]
                positives, negatives = [r for r in known if r["reference"] == 1], [r for r in known if r["reference"] == 0]
                keywords.append({"keyword_id": i + 1, "known_positive_rows": len(positives),
                    "matched_positive_rows": sum(r["complete_sequence_present"] for r in positives),
                    "known_negative_rows": len(negatives), "false_positive_rows": sum(r["complete_sequence_present"] for r in negatives),
                    "unknown_rows": len(group) - len(known)})
            # Full-text loss/edit denominators exclude only explicit unsupported
            # targets; keyword known masks are independently preserved above.
            detail = fit.summarize(scored) if scored else {}
            result[cohort][str(ms)] = {**detail, **evaluation.summarize(group), "keywords": keywords,
                "known_zero_frames": sum(r["known_zero_prefix_frames"] for r in group),
                "known_zero_nonblank_frames": sum(r["known_zero_nonblank_frames"] for r in group),
                "emissions_entirely_in_added_silence": sum(r["emissions_entirely_in_added_silence"] for r in group)}
    return result


def relocated_rows(job):
    return [{**r, "audio": "audio/" + r["source_id"] + ".wav", "audio_path": "audio/" + r["source_id"] + ".wav"}
        for r in job["evaluation_rows"]]


def validate_terminal_counts(status, job):
    require(status["model_constructions"] == 1 and status["actual_optimizer_steps"] == job["optimizer_steps"] and
        status["training_forwards_started"] == status["training_forwards_completed"] == job["training_forward_limit"] and
        status["final_forwards_started"] == status["final_forwards_completed"] == job["final_forward_limit"], "AUGMENTATION_TERMINAL_COUNTS")


@contextmanager
def deadline(seconds):
    require(type(seconds) is int and 0 < seconds <= 840 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
        "AUGMENTATION_DEADLINE_SCOPE")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise TimeoutError("AUGMENTATION_FIXED_WORKER_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(recipe_path, output, lock):
    job = training.read_json(recipe_path)
    validate_contract(job)
    require(str(io.safe_path(lock)) == job["heavy_lock"], "AUGMENTATION_CANONICAL_LOCK")
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        require(os.environ.get("KWS_RESEARCH_JOB_ID") == job["job_id"] and
            os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(recipe_path).absolute()) and
            os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(Path(output).absolute()), "AUGMENTATION_QUEUE_SCOPE")
        output = training.output_directory(output)
        started = time.monotonic()
        status = {**contract(job["mode"]), "status": "running", "started_at": training.utc(),
            "job_id": job["job_id"], "recipe_sha256": training.sha_file(recipe_path), "model_constructions": 0, "actual_optimizer_steps": 0,
            "training_forwards_started": 0, "training_forwards_completed": 0,
            "final_forwards_started": 0, "final_forwards_completed": 0, "final_forward_limit": job["final_forward_limit"]}
        training.save_json(output / "status.json", status)
        model = optimizer = None
        try:
            with deadline(job["worker_seconds"]) as end:
                vocabulary = validate_recipe(job)
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                torch, reference = training.torch_runtime()
                from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
                training.copy_once(recipe_path, output / "request.json")
                for name in CODE:
                    training.copy_once(ROOT / name, output / "execution-source" / name)
                for key, name in (("initial_checkpoint", "initial.pt"), ("vocabulary_file", "vocabulary.json"), ("tokens_file", "tokens.txt")):
                    training.copy_once(checked(job[key]), output / name)
                rows = relocated_rows(job)
                for row in rows:
                    training.copy_once(checked(job["audio_files"][row["source_id"]]), output / row["audio"])
                for name, value in (("all-new-rows.json", job["all_new_rows"]), ("train-rows.json", job["train"]),
                    ("development-rows.json", job["development"]), ("evaluation-rows.json", rows)):
                    io.write_json(output / name, value)
                initial = prior.initial_state(output / "initial.pt", job)
                bank, evidence, pcms = prior.feature_bank(output, rows, torch)
                io.write_json(output / "feature-bindings.json", evidence)
                torch.manual_seed(job["seed"])
                model, description = training.build_fresh(onset.CANDIDATE, vocabulary, reference)
                status["model_constructions"] = 1
                require(description == job["model_description"], "AUGMENTATION_ARCHITECTURE_CHANGED")
                if job["mode"] == "augmentation2000":
                    model.load_state_dict(initial["state_dict"], strict=True)
                    require(state_identity(model.state_dict()) == initial["float_state_identity"], "AUGMENTATION_INITIAL_STATE_MISMATCH")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                    forward = prior.TrainingCalls(model, bank, evidence, job["train"], job, output, torch)
                    updates = prior.OptimizerCalls(optimizer, forward, output)
                    silence.train(forward, bank, job["train"], job, updates, output, end, status, torch)
                    gradients = prior.ledger_readback(context.lines(output / "consumption.jsonl"), job, job["train"])
                    require(forward.count == DRAWS and updates.count == STEPS, "AUGMENTATION_ACCOUNTED_TRAINING_COUNTS")
                    identity = state_identity(model.state_dict())
                    names = [name for name, _ in model.named_parameters()]
                    require(names == job["canonical_parameter_names"] == list(model.state_dict()), "AUGMENTATION_PARAMETER_ORDER")
                    payload = {"schema": SCHEMA, "candidate": onset.CANDIDATE, "state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "float_state_identity": identity,
                        "job_recipe_sha256": status["recipe_sha256"], "optimizer_steps": STEPS, "training_model_forwards": DRAWS,
                        "initial_checkpoint_sha256": silence.INITIAL_SHA, "vocabulary_sha256": job["vocabulary_sha256"],
                        "train_binding_sha256": job["train_binding_sha256"], "source_hashes": job["source_hashes"]}
                    atomic_save_checkpoint(payload, output / "final.pt")
                    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
                    require(state_identity(final["state_dict"]) == identity, "AUGMENTATION_FINAL_READBACK")
                    optim = prior.optimizer_readback(final["optimizer_state"], final["state_dict"], names, job["canonical_parameter_names"], torch)
                    status.update(gradients=gradients, optimizer_readback=optim)
                else:
                    training.copy_once(checked(job["baseline_checkpoint"]), output / "final.pt")
                    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
                    identity = state_identity(final["state_dict"])
                    require(final_sha == BASELINE_SHA and final["schema"] == prior.SCHEMA and
                        final["optimizer_steps"] == STEPS and final["training_model_forwards"] == DRAWS and
                        final["initial_checkpoint_sha256"] == silence.INITIAL_SHA and
                        final["vocabulary_sha256"] == job["vocabulary_sha256"] and
                        identity == final["float_state_identity"] and identity["sha256"] == job["baseline_float_state_sha256"],
                        "AUGMENTATION_BASELINE_STATE_CHANGED")
                    model.load_state_dict(final["state_dict"], strict=True)
                    require(state_identity(model.state_dict()) == identity, "AUGMENTATION_BASELINE_LOAD_CHANGED")
                model.eval()
                observed, calls = observe(job, output, bank, pcms, vocabulary, final_sha, torch, model=model, end=end, status=status)
                require(state_identity(model.state_dict()) == identity, "AUGMENTATION_EVAL_MUTATED_STATE")
                if job["mode"] == "augmentation2000":
                    prior.verify_calls(output, job, evidence, calls)
                else:
                    require(context.lines(output / "forward-calls.jsonl") == calls, "AUGMENTATION_BASELINE_FORWARD_LEDGER")
                validate_terminal_counts(status, job)
                require(time.monotonic() < end, "AUGMENTATION_DEADLINE")
                status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, final_checkpoint_sha256=final_sha,
                    initial_float_state_sha256=initial["float_state_identity"]["sha256"], final_float_state_sha256=identity["sha256"],
                    final_observation_state_unchanged=True, conditions=conditions(observed, job))
                training.save_json(output / "status.json", status)
                io.write_json(output / "summary.json", status)
                context.manifest(output)
                return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), error_type=type(error).__name__, error=str(error),
                wall_seconds=time.monotonic() - started, counts_are_lower_bounds=True,
                possible_incomplete_call="retain original started claims; no retry")
            if model is not None and optimizer is not None and not (output / "final.pt").exists():
                try:
                    atomic_save_checkpoint({"schema": SCHEMA, "status": "partial-failure-not-final", "state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "counts_are_lower_bounds": True,
                        "recorded_optimizer_steps": status["actual_optimizer_steps"]}, output / "partial.pt")
                except BaseException as saved_error:
                    status["partial_checkpoint_error"] = str(saved_error)
            training.save_json(output / "status.json", status)
            if not (output / "MANIFEST.json").exists():
                context.manifest(output)
            raise


def audit(output, expected):
    output = io.safe_path(output)
    onset.verify_manifest(output, expected)
    job, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    receipt = training.read_json(output.parent / "result.json")
    require(summary["status"] == "completed" and training.read_json(output / "status.json") == summary and
        receipt["schema"] == io.RECEIPT_SCHEMA and receipt["status"] == "completed" and receipt["exit_code"] == 0 and
        receipt["job_id"] == job["job_id"] == output.parent.name and
        receipt["queue_sha256"] == training.sha_file(output.parents[2] / "queue.json") and
        receipt["recipe_sha256"] == summary["recipe_sha256"] == training.sha_file(output / "request.json") and
        receipt["outputs"]["MANIFEST.json"] == io.file_record(output / "MANIFEST.json"), "AUGMENTATION_COMPLETED_QUEUE")
    with training.heavy_model_slot(job["heavy_lock"]):
        with deadline(240):
            vocabulary = validate_recipe(job)
            validate_contract(summary)
            validate_terminal_counts(summary, job)
            require({n: training.sha_file(output / "execution-source" / n) for n in CODE} == job["source_hashes"], "AUGMENTATION_SAVED_SOURCE")
            runtime = training.research_runtime.load_manifest()
            training.research_runtime.check_interpreter(runtime)
            training.research_runtime.check_packages(runtime, "core")
            torch, _ = training.torch_runtime()
            from training_state import load_checkpoint_snapshot, state_identity
            initial = prior.initial_state(output / "initial.pt", job)
            final, final_sha = load_checkpoint_snapshot(output / "final.pt")
            identity = state_identity(final["state_dict"])
            require(identity == final["float_state_identity"] and identity["sha256"] == summary["final_float_state_sha256"] and
                final_sha == summary["final_checkpoint_sha256"] and summary["final_observation_state_unchanged"] is True and
                initial["float_state_identity"]["sha256"] == summary["initial_float_state_sha256"], "AUGMENTATION_SAVED_STATE")
            if job["mode"] == "augmentation2000":
                require(final["schema"] == SCHEMA and final["candidate"] == onset.CANDIDATE and
                    final["optimizer_steps"] == STEPS and final["training_model_forwards"] == DRAWS and
                    final["job_recipe_sha256"] == summary["recipe_sha256"] and final["initial_checkpoint_sha256"] == silence.INITIAL_SHA and
                    final["vocabulary_sha256"] == job["vocabulary_sha256"] and final["train_binding_sha256"] == job["train_binding_sha256"] and
                    final["source_hashes"] == job["source_hashes"], "AUGMENTATION_FINAL_CHECKPOINT_CONTRACT")
                require(prior.optimizer_readback(final["optimizer_state"], final["state_dict"], list(final["state_dict"]),
                    job["canonical_parameter_names"], torch) == summary["optimizer_readback"] and
                    prior.ledger_readback(context.lines(output / "consumption.jsonl"), job, job["train"]) == summary["gradients"],
                    "AUGMENTATION_SAVED_OPTIMIZER_OR_CONSUMPTION")
            else:
                require(final_sha == BASELINE_SHA, "AUGMENTATION_SAVED_BASELINE_CHANGED")
            rows = relocated_rows(job)
            require(training.read_json(output / "evaluation-rows.json") == rows and
                training.read_json(output / "all-new-rows.json") == job["all_new_rows"] and
                training.read_json(output / "train-rows.json") == job["train"] and
                training.read_json(output / "development-rows.json") == job["development"], "AUGMENTATION_SAVED_ROWS")
            bank, evidence, pcms = prior.feature_bank(output, rows, torch)
            require(evidence == training.read_json(output / "feature-bindings.json"), "AUGMENTATION_SAVED_FEATURES")
            observed, calls = observe(job, output, bank, pcms, vocabulary, final_sha, torch)
            if job["mode"] == "augmentation2000":
                prior.verify_calls(output, job, evidence, calls)
            else:
                require(context.lines(output / "forward-calls.jsonl") == calls, "AUGMENTATION_BASELINE_CALLS")
            require(conditions(observed, job) == summary["conditions"], "AUGMENTATION_SAVED_SUMMARY")
            return {"schema": SCHEMA, "status": "verified", "mode": job["mode"], "manifest_sha256": expected,
                "saved_training_forwards": job["training_forward_limit"], "saved_optimizer_steps": job["optimizer_steps"],
                "saved_final_conditions": job["final_forward_limit"], "new_model_constructions": 0,
                "new_model_forwards": 0, "new_optimizer_steps": 0, "complete_optimization_numerically_replayed": False}


def queue_definition(recipe, python):
    job = training.read_json(recipe)
    validate_contract(job)
    outputs = ["summary.json", "observations.jsonl", "forward-calls.jsonl", "final.pt", "MANIFEST.json"]
    if job["mode"] == "augmentation2000":
        outputs += ["consumption.jsonl", "optimizer-calls.jsonl"]
    return {"schema": io.SCHEMA, "queue_id": job["job_id"], "global_budget_minutes": job["queue_minutes"], "jobs": [{
        "job_id": job["job_id"], "recipe": str(Path(recipe).absolute()), "recipe_sha256": training.sha_file(recipe),
        "argv": [python, "-I", "-B", str(ROOT / SOURCE), "run", "--recipe", "{recipe}",
                 "--output", "{output_dir}", "--heavy-lock", job["heavy_lock"]],
        "cwd": str(ROOT), "depends_on": [], "timeout_minutes": job["queue_minutes"], "outputs": outputs}]}


def completed_evidence(value):
    require(set(value) == {"output", "manifest_sha256", "audit"}, "AUGMENTATION_COMPARISON_INPUT")
    output = io.safe_path(value["output"])
    onset.verify_manifest(output, value["manifest_sha256"])
    job, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    validate_contract(job)
    validate_contract(summary)
    validate_terminal_counts(summary, job)
    result = training.read_json(output.parent / "result.json")
    audit_record = training.read_json(checked(value["audit"]))
    require(summary["status"] == "completed" and training.read_json(output / "status.json") == summary and
        result["status"] == "completed" and result["exit_code"] == 0 and result["job_id"] == job["job_id"] == output.parent.name and
        result["recipe_sha256"] == summary["recipe_sha256"] == training.sha_file(output / "request.json") and
        result["queue_sha256"] == training.sha_file(output.parents[2] / "queue.json") and
        result["outputs"]["MANIFEST.json"] == io.file_record(output / "MANIFEST.json") and
        audit_record["schema"] == SCHEMA and audit_record["status"] == "verified" and
        audit_record["manifest_sha256"] == value["manifest_sha256"] and audit_record["mode"] == job["mode"] and
        audit_record["saved_final_conditions"] == job["final_forward_limit"] and
        audit_record["saved_training_forwards"] == job["training_forward_limit"] and
        audit_record["saved_optimizer_steps"] == job["optimizer_steps"] and
        audit_record["new_model_forwards"] == audit_record["new_model_constructions"] == audit_record["new_optimizer_steps"] == 0,
        "AUGMENTATION_COMPARE_ACTUAL_COMPLETION_AND_AUDIT")
    return output, job, summary


def compare(request):
    """Join actual saved rows; reference old baseline210 without new inference."""
    require(set(request) == {"baseline_evaluation", "augmentation"}, "AUGMENTATION_COMPARE_REQUEST")
    baseline, baseline_job, baseline_summary = completed_evidence(request["baseline_evaluation"])
    augmentation, job, summary = completed_evidence(request["augmentation"])
    require(baseline_job["mode"] == "baseline-evaluation" and job["mode"] == "augmentation2000" and
        io.canonical(baseline_job["inputs"]) == io.canonical(job["inputs"]) and
        baseline_summary["initial_float_state_sha256"] == summary["initial_float_state_sha256"] ==
            job["initial_binding"]["float_state_sha256"], "AUGMENTATION_MATCHED_INPUTS_AND_INITIAL")
    old = Path(job["inputs"]["baseline_output"])
    onset.verify_manifest(old, BASELINE_MANIFEST_SHA)
    baseline_rows = context.lines(old / "observations.jsonl") + context.lines(baseline / "observations.jsonl")
    actual_rows = context.lines(augmentation / "observations.jsonl")
    key = lambda item: (item["source_id"], item["split"], item["offset_ms"])
    previous = {key(item): item for item in baseline_rows}
    actual = {key(item): item for item in actual_rows}
    require(len(previous) == len(baseline_rows) == len(actual) == len(actual_rows) and set(previous) == set(actual),
        "AUGMENTATION_PAIRED_COMPLETE_DATA_COVERAGE")
    previous_features = {key(item): item for path in (old, baseline)
        for item in training.read_json(path / "feature-bindings.json")}
    actual_features = {key(item): item for item in training.read_json(augmentation / "feature-bindings.json")}
    require(previous_features == actual_features, "AUGMENTATION_PAIR_INPUT_FEATURES_CHANGED")
    new_ids = {r["source_id"] for r in job["all_new_rows"]}
    pairs = []
    ordered_baseline = []
    for item in actual_rows:
        original = previous[key(item)]
        require(original["label_binding_sha256"] == item["label_binding_sha256"], "AUGMENTATION_PAIR_LABEL_CHANGED")
        # Historical observation schema predates the explicit evaluated flag.
        original = {**original, "evaluation_status": original.get("evaluation_status", "evaluated")}
        ordered_baseline.append(original)
        paired = {"source_id": item["source_id"], "split": item["split"], "offset_ms": item["offset_ms"],
            "cohort": ("new-" if item["source_id"] in new_ids else "old-") + item["split"],
            "label_binding_sha256": item["label_binding_sha256"], "baseline": original, "augmentation": item}
        if original["evaluation_status"] == item["evaluation_status"] == "evaluated":
            paired["augmentation_minus_baseline"] = {k: item[k] - original[k] for k in
                ("ctc_loss_target_length_normalized", "ctc_loss_confidence_weighted", "edit_numerator", "deletions")}
        pairs.append(paired)
    return {"schema": "kws-c1-source-augmentation-saved-comparison-v1", "status": "completed-saved-only",
        "inputs": request, "old_baseline_manifest": pin(old / "MANIFEST.json"), "old_baseline_conditions_reused": 210,
        "new_baseline_conditions": baseline_job["final_forward_limit"], "augmentation_conditions": job["final_forward_limit"],
        "initial_float_state_sha256": summary["initial_float_state_sha256"], "all_pairs": pairs,
        "baseline_conditions": conditions(ordered_baseline, job), "augmentation_conditions": conditions(actual_rows, job),
        "new_model_forwards": 0, "new_optimizer_steps": 0, "human_gold": False, "independent_test": False,
        "previously_exposed_development": True, "single_variable_data_size_ablation": False,
        "source_text_weak_weight_and_per_item_exposure_change_together": True,
        "offsets_are_repeated_conditions_not_independent_rows": True, "quality_winner": None, "generalization_passed": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, keys in (("probe", ("request", "output")), ("compare", ("request", "output")), ("declare", ("request", "mode", "output")),
        ("queue-request", ("recipe", "python", "output")), ("run", ("recipe", "output", "heavy-lock")),
        ("audit", ("output", "manifest-sha256", "receipt"))):
        command = sub.add_parser(name)
        for key in keys:
            command.add_argument("--" + key, required=True, **({"choices": MODES} if key == "mode" else {}))
    args = parser.parse_args(argv)
    if args.command == "probe":
        data = assemble(training.read_json(args.request), require_terminal=False)
        result = {"schema": SCHEMA, "status": "metadata-probe", "new_model_calls": 0, "optimizer_steps": 0,
            "execution_authorized": False, "new_rows": data["new"], "train_rows": len(data["train"]),
            "development_rows": len(data["development"]), "aggregate": data["aggregate"],
            "missing_first64_weak_rows": data["missing_weak_rows"], "loader_sources": data["loader_sources"],
            "original52_adopted_unchanged": data["train"][:52] == data["base"]["train"],
            "projection_reasons": dict(Counter(r["ctc_reason"] for r in data["new"]))}
        io.write_json(Path(args.output), result)
        result = {k: v for k, v in result.items() if k not in ("new_rows", "loader_sources", "missing_first64_weak_rows")}
    elif args.command == "compare":
        result = compare(training.read_json(args.request))
        io.write_json(Path(args.output), result)
        result = {k: v for k, v in result.items() if k not in ("all_pairs", "baseline_conditions", "augmentation_conditions")}
    elif args.command == "declare":
        job = declare(training.read_json(args.request), args.mode)
        io.write_json(Path(args.output), job)
        result = {"recipe": pin(args.output), "new_model_calls": 0, "training_forward_limit": job["training_forward_limit"],
            "final_forward_limit": job["final_forward_limit"], "mode": args.mode}
    elif args.command == "queue-request":
        io.write_json(Path(args.output), queue_definition(args.recipe, args.python))
        result = {"queue_request": pin(args.output), "new_model_calls": 0}
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        result = audit(args.output, args.manifest_sha256)
        io.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
