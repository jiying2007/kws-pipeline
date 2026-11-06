#!/usr/bin/env python3
"""Bounded D1 memorization diagnostic using an existing sealed preparation.

The original vocabulary, features, masks, initialization, optimizer and CTC
consumer are reused. Four or eight known training rows are declared before a
new job starts. This is a training-set learnability check, not a development
comparison, a new weak-label decision, or permission to repeat a failed job.
Import, --help and declare do not import Torch or construct a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_evaluate as evaluation
import research_d1_train as training
from research_d1_recipe import BATCH_SIZE, KEYWORDS, SEED, row_identity
from research_d1_vocab import OUTPUTS, digest, fulltext_targets, require
from research_job_lock import heavy_model_slot

SCHEMA = "kws-d1-tiny-fit-job-v1"
RESULT_SCHEMA = "kws-d1-tiny-fit-result-v1"
SOURCE_PATHS = ("training/research_d1_fit_diagnostic.py", "training/research_d1_evaluate.py")
BUCKETS = ("keyword-1", "keyword-2", "ordinary-negative", "partial-negative")
QUEUE_OUTPUTS = ("summary.json", "consumption.jsonl", "before.json", "after.json", "final.pt", "MANIFEST.json")


def diagnostic_sources():
    return {name: training.sha_file(ROOT / name) for name in SOURCE_PATHS}


def validate_job(value):
    require(type(value) is dict and set(value) == {
        "schema", "prepared", "prepared_manifest_sha256", "candidate", "source_ids",
        "row_binding_sha256", "optimizer_steps", "timeout_seconds", "diagnostic_source_hashes"
    } and value["schema"] == SCHEMA, "D1_FIT_JOB_SCHEMA")
    require(type(value["prepared"]) is str and Path(value["prepared"]).is_absolute() and
            type(value["prepared_manifest_sha256"]) is str and
            training.io_contract.SHA.fullmatch(value["prepared_manifest_sha256"]) and
            value["candidate"] in training.CANDIDATES, "D1_FIT_PREPARED_OR_CANDIDATE")
    ids = value["source_ids"]
    require(type(ids) is list and len(ids) in (4, 8) and
            all(type(name) is str and training.io_contract.ID.fullmatch(name) for name in ids) and
            len(set(ids)) == len(ids), "D1_FIT_FOUR_OR_EIGHT_DISTINCT_SOURCES")
    require(type(value["row_binding_sha256"]) is str and
            training.io_contract.SHA.fullmatch(value["row_binding_sha256"]), "D1_FIT_LABEL_BINDING")
    require(type(value["optimizer_steps"]) is int and 1 <= value["optimizer_steps"] <= 2000,
            "D1_FIT_FIXED_STEP_BUDGET")
    seconds = value["timeout_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 600,
            "D1_FIT_FIXED_TIME_BUDGET")
    sources = value["diagnostic_source_hashes"]
    require(type(sources) is dict and set(sources) == set(SOURCE_PATHS) and
            all(type(sha) is str and training.io_contract.SHA.fullmatch(sha) for sha in sources.values()),
            "D1_FIT_DIAGNOSTIC_SOURCE_BINDING")
    return value


def selection_bucket(row, vocabulary):
    """Only actual confirmed train labels can enter the memorization cohort."""
    require(row["split"] == "train" and row["source_role"] == "train-candidate" and
            row["label_status"] == "machine_candidate" and row["ctc_mask"] is True and
            type(row["ctc_weight"]) in (int, float) and 0 < row["ctc_weight"] <= 1 and
            row["quarantined"] is False and row.get("generation_outcome") != "failed-native" and
            row["human_gold"] is False and row["independent_test"] is False,
            "D1_FIT_KNOWN_TRAINING_ROWS_ONLY")
    projection = fulltext_targets(row["full_text"], vocabulary)
    require(projection["supported"] and row["target_ids"] == projection["target_ids"] and
            row["vocabulary_sha256"] == digest(vocabulary), "D1_FIT_FULLTEXT_TARGET_CHANGED")
    require(row["keyword_mask"] == [True, True] and
            all(type(mask) is bool for mask in row["keyword_mask"]) and
            row["keyword_targets"] == [int(word in row["full_text"]) for word in KEYWORDS],
            "D1_FIT_KEYWORD_MASK_CHANGED")
    # project_row has already checked keyword observations against the complete
    # transcript. Generation categories only stratify these observed labels.
    category = row["generation_category"]
    if category == "full-positive" and row["keyword_targets"] in ([1, 0], [0, 1]):
        return "keyword-1" if row["keyword_targets"] == [1, 0] else "keyword-2"
    if category in BUCKETS[2:] and row["keyword_targets"] == [0, 0]:
        return category
    return None


def choose_rows(rows, vocabulary, count):
    """Metadata-only deterministic stratification; never inspect predictions."""
    require(type(count) is int and count in (4, 8), "D1_FIT_FOUR_OR_EIGHT_DISTINCT_SOURCES")
    pools = {bucket: [] for bucket in BUCKETS}
    for row in rows:
        bucket = selection_bucket(row, vocabulary)
        if bucket is not None:
            pools[bucket].append(row)
    chosen, used, families = [], set(), Counter()
    for bucket in BUCKETS * (count // len(BUCKETS)):
        eligible = [row for row in pools[bucket] if row["source_id"] not in used]
        require(eligible, "D1_FIT_MISSING_KNOWN_POOL:" + bucket)
        row = min(eligible, key=lambda item: (families[item["source_family_id"]], item["source_id"]))
        chosen.append(row)
        used.add(row["source_id"])
        families[row["source_family_id"]] += 1
    return chosen


def verify_selection(job, rows, vocabulary):
    require(len({row["source_id"] for row in rows}) == len(rows), "D1_FIT_DUPLICATE_COHORT_SOURCE")
    by_id = {row["source_id"]: row for row in rows}
    require(set(job["source_ids"]) <= set(by_id), "D1_FIT_SELECTION_OUTSIDE_TRAIN_COHORT")
    selected = [by_id[name] for name in job["source_ids"]]
    require([selection_bucket(row, vocabulary) for row in selected] ==
            list(BUCKETS) * (len(selected) // len(BUCKETS)), "D1_FIT_SELECTION_STRATA_CHANGED")
    require(digest([row_identity(row) for row in selected]) == job["row_binding_sha256"],
            "D1_FIT_SELECTED_LABELS_CHANGED")
    return selected


def prepared_rows(job):
    prepared = training.io_contract.safe_path(job["prepared"])
    record, vocabulary, rows = training.verify_prepared(prepared, job["prepared_manifest_sha256"])
    projected = [training.io_contract.loads(line) for line in
                 training.io_contract.read_bytes(prepared / "projected-rows.jsonl", 64 * 1024 * 1024).splitlines()]
    families = sorted({row["source_family_id"] for row in projected if row["split"] == "development"})
    # Reuse the existing complete projection/source-role verifier. Development
    # metadata is checked for leakage; it never chooses the training subset.
    evaluation.development_rows(projected, rows, record, vocabulary, families)
    return prepared, record, vocabulary, rows


def declare(prepared, manifest_sha256, candidate, count, steps, seconds, output):
    partial = {"prepared": str(training.io_contract.safe_path(prepared)),
               "prepared_manifest_sha256": manifest_sha256}
    _, _, vocabulary, rows = prepared_rows(partial)
    selected = choose_rows(rows, vocabulary, count)
    job = validate_job({"schema": SCHEMA, **partial, "candidate": candidate,
        "source_ids": [row["source_id"] for row in selected],
        "row_binding_sha256": digest([row_identity(row) for row in selected]),
        "optimizer_steps": steps, "timeout_seconds": seconds,
        "diagnostic_source_hashes": diagnostic_sources()})
    training.io_contract.write_json(output, job)
    return {"recipe": str(output), "recipe_sha256": training.sha_file(output),
            "selected_train_rows": count, "optimizer_steps_declared": steps,
            "model_constructions": 0, "model_forward_calls": 0}


def tiny_recipe(rows, base, steps):
    """Change only exposure schedule; retain the saved CTC/optimizer settings."""
    draws = [{"row_index": i % len(rows), "source_id": rows[i % len(rows)]["source_id"],
              "ctc_weight": rows[i % len(rows)]["ctc_weight"]}
             for i in range(steps * BATCH_SIZE)]
    return {"schema": SCHEMA, "batch_size": BATCH_SIZE, "seed": SEED,
            "exposure_plan": {"draws": draws}, "exposure_plan_sha256": digest(draws),
            "optimizer_steps": steps, "total_exposures": len(draws),
            "fixed_ctc_normalizer": base["fixed_ctc_normalizer"],
            "gradient_clip_max_norm": base["gradient_clip_max_norm"],
            "optimizers": base["optimizers"], "keyword_loss_used": False}


def score(logits, row):
    """Saved-logit diagnostics, sharing the actual D1 CTC and greedy scorer."""
    import torch
    require(row["ctc_mask"] is True and 0 < row["ctc_weight"] <= 1, "D1_FIT_CANNOT_SCORE_UNKNOWN_LABEL")
    result = evaluation.score_logits(logits, row)
    values = logits.detach()[:, 0]
    probabilities = values.softmax(dim=-1)
    predicted = values.argmax(dim=-1)
    target_tokens = sorted(set(row["target_ids"]))
    other = torch.ones(OUTPUTS, dtype=torch.bool)
    other[[0, *target_tokens]] = False
    token_stats = []
    for token in target_tokens:
        # Competition rank matches diagnose_acoustic_alignment.token_best_ranks:
        # equal scores share a rank; this is not evidence of ordered alignment.
        ranks = 1 + (values > values[:, token:token + 1]).sum(dim=-1)
        token_stats.append({"token_id": token, "best_competition_rank": int(ranks.min()),
            "top3_frames": int((ranks <= 3).sum()), "argmax_frames": int((predicted == token).sum()),
            "maximum_probability": float(probabilities[:, token].max()),
            "maximum_logit_margin_over_blank": float((values[:, token] - values[:, 0]).max())})
    return {**result, "frames": len(values), "blank_argmax_frames": int((predicted == 0).sum()),
            "blank_argmax_fraction": float((predicted == 0).float().mean()),
            "blank_probability_mean": float(probabilities[:, 0].mean()),
            "target_probability_mass_mean": float(probabilities[:, target_tokens].sum(dim=-1).mean()),
            "other_nonblank_probability_mass_mean": float(probabilities[:, other].sum(dim=-1).mean()),
            "target_tokens": token_stats, "target_rank_is_forced_alignment": False,
            "feature_target_geometry_feasible": True}


def summarize(rows):
    scored = [{**row, "evaluation_status": "evaluated"} for row in rows]
    summary = evaluation.summarize(scored)
    predictions = {tuple(row["greedy_token_ids"]) for row in rows}
    frames = sum(row["frames"] for row in rows)
    return {**summary, "empty_outputs": sum(not row["greedy_token_ids"] for row in rows),
            "single_token_outputs": sum(len(row["greedy_token_ids"]) == 1 for row in rows),
            "distinct_predictions": len(predictions),
            "constant_single_token_prediction": len(predictions) == 1 and len(next(iter(predictions), ())) == 1,
            "blank_argmax_fraction": sum(row["blank_argmax_frames"] for row in rows) / frames,
            "blank_probability_mean": sum(row["blank_probability_mean"] * row["frames"] for row in rows) / frames,
            "mean_target_ctc_nll": sum(row["ctc_nll"] for row in rows) / len(rows)}


def observe(model, features, rows, phase, output, deadline):
    import torch
    observed = []
    model.eval()
    with torch.inference_mode():
        for row, feature in zip(rows, features):
            require(time.monotonic() < deadline, "D1_FIT_SOFT_DEADLINE")
            logits = model(feature.unsqueeze(0))
            name = "numeric/" + phase + "/" + row["source_id"] + ".pt"
            observed.append({"source_id": row["source_id"], "pcm_sha256": row["pcm_sha256"],
                "label_binding_sha256": row_identity(row), "target_ids": row["target_ids"],
                "ctc_mask": row["ctc_mask"], "ctc_weight": row["ctc_weight"],
                **score(logits, row), "numeric_artifact": {"path": name,
                    **evaluation.save_numeric(output / name, row["source_id"], feature, logits)}})
    training.io_contract.write_json(output / (phase + ".json"), observed)
    return observed


def gradient_observation(ledger, recipe, rows):
    require(len(ledger) == recipe["optimizer_steps"], "D1_FIT_INCOMPLETE_STEP_LEDGER")
    for index, event in enumerate(ledger):
        expected = recipe["exposure_plan"]["draws"][index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        require(event["optimizer_step"] == index + 1 and event["draw_offset"] == index * BATCH_SIZE and
                len(event["draws"]) == len(expected), "D1_FIT_STEP_LEDGER_CHANGED")
        for observed, draw in zip(event["draws"], expected):
            row = rows[draw["row_index"]]
            require(observed["source_id"] == row["source_id"] and
                    observed["label_binding_sha256"] == row_identity(row) and
                    observed["target_ids"] == row["target_ids"] and observed["ctc_mask"] is True and
                    observed["ctc_weight"] == row["ctc_weight"] and observed["keyword_loss_used"] is False,
                    "D1_FIT_CONSUMED_LABEL_CHANGED")
        require(type(event["gradient_norm_before_clip"]) in (int, float) and
                math.isfinite(event["gradient_norm_before_clip"]) and event["gradient_norm_before_clip"] >= 0 and
                math.isfinite(event["loss"]), "D1_FIT_NONFINITE_GRADIENT_OR_LOSS")
    norms = [event["gradient_norm_before_clip"] for event in ledger]
    return {"recorded_optimizer_steps": len(ledger), "nonzero_gradient_steps": sum(value > 0 for value in norms),
            "zero_gradient_steps": sum(value == 0 for value in norms),
            "gradient_norm_min": min(norms), "gradient_norm_max": max(norms),
            "all_selected_parameter_gradients_present_and_finite": True}


def parameter_changes(initial, final, names):
    require(set(initial) == set(final) and names and set(names) <= set(initial), "D1_FIT_PARAMETER_ROSTER_CHANGED")
    changed = [name for name in names if training.tensor_sha(final[name]) != training.tensor_sha(initial[name])]
    delta = math.sqrt(sum(float((final[name].double() - initial[name].double()).square().sum()) for name in names))
    require(math.isfinite(delta), "D1_FIT_NONFINITE_PARAMETER_CHANGE")
    require(all(training.tensor_sha(final[name]) == training.tensor_sha(initial[name])
                for name in initial if name not in names), "D1_FIT_FROZEN_PARAMETER_CHANGED")
    return changed, delta


def run(job_path, output, lock):
    job = validate_job(training.read_json(job_path))
    require(job["diagnostic_source_hashes"] == diagnostic_sources(), "D1_FIT_SOURCE_CHANGED")
    with heavy_model_slot(lock) as lease:
        training.queue_binding(job_path, lease)
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        deadline = started + job["timeout_seconds"]
        status = {"schema": RESULT_SCHEMA, "status": "running", "started_at": training.utc(),
            "job_recipe_sha256": training.sha_file(job_path), "candidate": job["candidate"],
            "optimizer_steps": 0, "sample_exposures": 0, "human_gold": False,
            "independent_human_test": False, "development_forward_calls": 0, "asr_or_tts_calls": 0,
            "quality_winner": None, "automatic_retry": False, "keyword_loss_used": False,
            "tiny_fit_passed": None, "scope": "memorization of predeclared known training rows only"}
        training.save_json(output / "status.json", status)
        try:
            runtime = training.research_runtime.load_manifest()
            training.research_runtime.check_interpreter(runtime)
            training.research_runtime.check_packages(runtime, "core")
            prepared, record, vocabulary, all_rows = prepared_rows(job)
            rows = verify_selection(job, all_rows, vocabulary)
            base = training.read_json(prepared / "recipes/0.json")
            recipe = tiny_recipe(rows, base, job["optimizer_steps"])
            torch, reference = training.torch_runtime()
            from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
            torch.manual_seed(SEED)
            if job["candidate"] == "fsmn-head-ctc2601":
                model, _, description = training.build_extended_fsmn(prepared / "official", vocabulary)
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(name.startswith(training.HEAD))
                cache_kind, optimizer_key = "fsmn", "extended-fsmn-head"
            else:
                model, description = training.build_fresh(job["candidate"], vocabulary, reference)
                cache_kind, optimizer_key = "fresh", "fresh-rnn64-or-c1"
            initial, initial_sha = load_checkpoint_snapshot(prepared / "initial" / (job["candidate"] + ".pt"))
            require(initial_sha == record["initial_checkpoints"][job["candidate"]]["file_sha256"] and
                    initial["candidate"] == job["candidate"] and initial["vocabulary_sha256"] == digest(vocabulary) and
                    initial["optimizer_steps"] == 0 and initial["description"] == description, "D1_FIT_INITIAL_CHANGED")
            model.load_state_dict(initial["state_dict"], strict=True)
            require(state_identity(model.state_dict()) == initial["float_state_identity"], "D1_FIT_INITIAL_LOAD_CHANGED")
            caches = torch.load(prepared / "features.pt", weights_only=True, map_location="cpu")[cache_kind]
            observations = training.read_json(prepared / "feature-observations.json")
            require(len(caches) == len(observations) == len(all_rows), "D1_FIT_FEATURE_COHORT_CHANGED")
            by_id = {row["source_id"]: index for index, row in enumerate(all_rows)}
            features = []
            for row in rows:
                index = by_id[row["source_id"]]
                feature, observed = caches[index], observations[index]
                require(observed["source_id"] == row["source_id"] and observed["pcm_sha256"] == row["pcm_sha256"] and
                        training.tensor_sha(feature) == observed[cache_kind + "_feature_sha256"], "D1_FIT_FEATURE_CHANGED")
                features.append(feature)
            training.io_contract.write_json(output / "exposure-recipe.json", recipe)
            training.io_contract.write_json(output / "selected-rows.json", rows)
            before = observe(model, features, rows, "before", output, deadline)
            trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
            frozen = {name: training.tensor_sha(value) for name, value in model.state_dict().items() if name not in trainable}
            optimizer_spec = recipe["optimizers"][optimizer_key]
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=optimizer_spec["lr"],
                                          weight_decay=optimizer_spec["weight_decay"])
            model.train()
            status.update(optimizer_loop_entered=True, initial_checkpoint_sha256=initial_sha,
                          prepared_manifest_sha256=job["prepared_manifest_sha256"],
                          vocabulary_sha256=digest(vocabulary), row_binding_sha256=job["row_binding_sha256"])
            training.save_json(output / "status.json", status)
            consumed = training.train_batches(model, features, rows, recipe, optimizer, output=output, deadline=deadline)
            ledger = [training.io_contract.loads(line) for line in (output / "consumption.jsonl").read_bytes().splitlines()]
            gradients = gradient_observation(ledger, recipe, rows)
            require(consumed["optimizer_steps"] == job["optimizer_steps"] and
                    consumed["sample_exposures"] == job["optimizer_steps"] * BATCH_SIZE, "D1_FIT_BUDGET_NOT_CONSUMED")
            require(all(training.tensor_sha(model.state_dict()[name]) == sha for name, sha in frozen.items()),
                    "D1_FIT_FROZEN_PARAMETER_CHANGED")
            changed, delta = parameter_changes(initial["state_dict"], model.state_dict(), list(trainable))
            final_identity = state_identity(model.state_dict())
            after = observe(model, features, rows, "after", output, deadline)
            require(state_identity(model.state_dict()) == final_identity, "D1_FIT_OBSERVATION_CHANGED_MODEL")
            payload = {"schema": RESULT_SCHEMA, "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "float_state_identity": final_identity, "candidate": job["candidate"], "job_recipe_sha256": training.sha_file(job_path),
                "initial_checkpoint_sha256": initial_sha, "prepared_manifest_sha256": job["prepared_manifest_sha256"],
                "vocabulary_sha256": digest(vocabulary), "optimizer_steps": consumed["optimizer_steps"],
                "sample_exposures": consumed["sample_exposures"], "human_gold": False, "quality_winner": None}
            atomic_save_checkpoint(payload, output / "final.pt")
            retained, checkpoint_sha = load_checkpoint_snapshot(output / "final.pt")
            require(state_identity(retained["state_dict"]) == final_identity, "D1_FIT_CHECKPOINT_READBACK_CHANGED")
            training.copy_once(job_path, output / "request.json")
            for name, sha in job["diagnostic_source_hashes"].items():
                require(training.copy_once(ROOT / name, output / "execution-source" / name) == sha, "D1_FIT_SOURCE_CHANGED")
            require(time.monotonic() < deadline, "D1_FIT_SOFT_DEADLINE")
            end = resource.getrusage(resource.RUSAGE_SELF)
            status.update(consumed, status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                user_cpu_seconds=end.ru_utime - usage.ru_utime, system_cpu_seconds=end.ru_stime - usage.ru_stime,
                peak_rss_kib=end.ru_maxrss, before=summarize(before), after=summarize(after), gradients=gradients,
                trainable_parameter_names=list(trainable), changed_parameter_names=changed, parameter_delta_l2=delta,
                frozen_parameters_unchanged=True, initial_float_state_sha256=initial["float_state_identity"]["sha256"],
                final_float_state_sha256=final_identity["sha256"], final_checkpoint_sha256=checkpoint_sha,
                selected_training_rows=len(rows), training_model_forwards=consumed["sample_exposures"],
                diagnostic_model_forwards=2 * len(rows), diagnostic_source_hashes=job["diagnostic_source_hashes"],
                consumption_sha256=training.sha_file(output / "consumption.jsonl"),
                tiny_fit_passed=all(row["exact_token_match"] for row in after) and bool(changed) and
                                gradients["nonzero_gradient_steps"] > 0,
                completion_is_not_memorization_or_generalization=True)
            training.save_json(output / "status.json", status)
            training.save_json(output / "summary.json", status)
            training.io_contract.write_json(output / "MANIFEST.json", {
                path.relative_to(output).as_posix(): training.io_contract.file_record(path)
                for path in sorted(output.rglob("*")) if path.is_file()})
            return status
        except BaseException as error:
            ledger = output / "consumption.jsonl"
            if ledger.exists():
                # A crashed/incomplete append is not an additional completed step.
                complete_lines = ledger.read_bytes().splitlines(keepends=True)
                events = [training.io_contract.loads(line) for line in complete_lines if line.endswith(b"\n")]
                status.update(optimizer_steps=len(events), sample_exposures=sum(len(row["draws"]) for row in events),
                              consumption_sha256=training.sha_file(ledger))
            status.update(status="failed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                          error_type=type(error).__name__, error=str(error), recorded_counts_are_lower_bounds=True)
            training.save_json(output / "status.json", status)
            raise


def audit(output, manifest_sha256):
    """Recompute saved-logit statistics; never construct or call a model."""
    output = training.io_contract.safe_path(output)
    raw = training.io_contract.read_bytes(output / "MANIFEST.json")
    require(hashlib.sha256(raw).hexdigest() == manifest_sha256, "D1_FIT_MANIFEST_CHANGED")
    manifest = training.io_contract.loads(raw)
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*")
              if path.is_file() and path != output / "MANIFEST.json"}
    require(set(manifest) == actual, "D1_FIT_OUTPUT_ROSTER_CHANGED")
    for name, expected in manifest.items():
        require(training.io_contract.file_record(output / training.io_contract.relative(name)) == expected,
                "D1_FIT_OUTPUT_CHANGED:" + name)
    job = validate_job(training.read_json(output / "request.json"))
    require(job["diagnostic_source_hashes"] == diagnostic_sources(), "D1_FIT_SOURCE_CHANGED")
    summary = training.read_json(output / "summary.json")
    require(summary["schema"] == RESULT_SCHEMA and summary["status"] == "completed" and
            summary["job_recipe_sha256"] == training.sha_file(output / "request.json"), "D1_FIT_SUMMARY_BINDING")
    prepared, _, vocabulary, all_rows = prepared_rows(job)
    rows = verify_selection(job, all_rows, vocabulary)
    require(training.read_json(output / "selected-rows.json") == rows, "D1_FIT_SAVED_ROWS_CHANGED")
    recipe = tiny_recipe(rows, training.read_json(prepared / "recipes/0.json"), job["optimizer_steps"])
    require(training.read_json(output / "exposure-recipe.json") == recipe, "D1_FIT_EXPOSURE_RECIPE_CHANGED")
    ledger = [training.io_contract.loads(line) for line in (output / "consumption.jsonl").read_bytes().splitlines()]
    require(gradient_observation(ledger, recipe, rows) == summary["gradients"], "D1_FIT_GRADIENT_SUMMARY_CHANGED")
    runtime = training.research_runtime.load_manifest()
    training.research_runtime.check_interpreter(runtime)
    training.research_runtime.check_packages(runtime, "core")
    torch, _ = training.torch_runtime()
    from training_state import load_checkpoint_snapshot, state_identity
    initial, initial_sha = load_checkpoint_snapshot(prepared / "initial" / (job["candidate"] + ".pt"))
    final, final_sha = load_checkpoint_snapshot(output / "final.pt")
    require(final_sha == summary["final_checkpoint_sha256"] and initial_sha == summary["initial_checkpoint_sha256"] and
            final["job_recipe_sha256"] == summary["job_recipe_sha256"] and final["schema"] == RESULT_SCHEMA and
            final["candidate"] == job["candidate"] and final["prepared_manifest_sha256"] == job["prepared_manifest_sha256"] and
            final["optimizer_steps"] == summary["optimizer_steps"] == job["optimizer_steps"] and
            final["sample_exposures"] == summary["sample_exposures"] == BATCH_SIZE * job["optimizer_steps"] and
            state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"],
            "D1_FIT_CHECKPOINT_BINDING_CHANGED")
    names = [name for name in initial["state_dict"]
             if job["candidate"] != "fsmn-head-ctc2601" or name.startswith(training.HEAD)]
    require(names == summary["trainable_parameter_names"], "D1_FIT_TRAINABLE_PARAMETER_ROSTER_CHANGED")
    changed, delta = parameter_changes(initial["state_dict"], final["state_dict"], names)
    require(changed == summary["changed_parameter_names"] and delta == summary["parameter_delta_l2"],
            "D1_FIT_PARAMETER_CHANGE_SUMMARY_CHANGED")
    for phase in ("before", "after"):
        observations = training.read_json(output / (phase + ".json"))
        require(len(observations) == len(rows), "D1_FIT_OBSERVATION_ROSTER_CHANGED")
        for row, observation in zip(rows, observations):
            require(observation["source_id"] == row["source_id"] and
                    observation["label_binding_sha256"] == row_identity(row), "D1_FIT_OBSERVATION_LABEL_CHANGED")
            numeric = observation["numeric_artifact"]
            path = output / training.io_contract.relative(numeric["path"])
            require(training.io_contract.file_record(path) == {key: numeric[key] for key in ("bytes", "sha256")},
                    "D1_FIT_NUMERIC_CHANGED")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            require(saved["source_id"] == row["source_id"] and
                    training.tensor_sha(saved["feature"]) == saved["feature_sha256"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"], "D1_FIT_NUMERIC_IDENTITY_CHANGED")
            require(all(observation[key] == value for key, value in score(saved["logits"], row).items()),
                    "D1_FIT_SAVED_LOGIT_STATISTICS_CHANGED")
        require(summarize(observations) == summary[phase], "D1_FIT_PHASE_SUMMARY_CHANGED")
    require(summary["tiny_fit_passed"] == (summary["after"]["exact_token_matches"] == len(rows) and
            bool(changed) and summary["gradients"]["nonzero_gradient_steps"] > 0), "D1_FIT_VERDICT_CHANGED")
    return {"status": "verified", "manifest_sha256": manifest_sha256,
            "tiny_fit_passed": summary["tiny_fit_passed"], "new_model_forwards": 0, "optimizer_steps": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("declare")
    prep.add_argument("--prepared", type=Path, required=True)
    prep.add_argument("--prepared-manifest-sha256", required=True)
    prep.add_argument("--candidate", choices=training.CANDIDATES, required=True)
    prep.add_argument("--count", type=int, choices=(4, 8), default=4)
    prep.add_argument("--steps", type=int, default=400)
    prep.add_argument("--timeout-seconds", type=float, default=180)
    prep.add_argument("--output", type=Path, required=True)
    run_command = sub.add_parser("run")
    run_command.add_argument("--recipe", type=Path, required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--heavy-lock", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        result = declare(args.prepared, args.prepared_manifest_sha256, args.candidate, args.count,
                         args.steps, args.timeout_seconds, args.output)
    elif args.command == "run":
        result = run(args.recipe, args.output, args.heavy_lock)
    else:
        result = audit(args.output, args.manifest_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
