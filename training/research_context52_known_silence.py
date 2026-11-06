#!/usr/bin/env python3
"""One fixed full-context52 learning stage, followed by train/dev readback.

The original known-silence numerical loop is reused unchanged. This adapter
changes its declared roster, adds durable call accounting, and retains all
210 final float32 observations. Declaration imports no tensor runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_c1_context_development as diagnostic

context, silence = diagnostic.context, diagnostic.silence
onset, training, fit, evaluation = diagnostic.onset, diagnostic.training, diagnostic.fit, diagnostic.evaluation
io, require = diagnostic.io, diagnostic.require
SOURCE = "training/research_context52_known_silence.py"
PLAN = "configs/research/context52-known-silence-20261106.json"
PLAN_SHA = "87ef781b8c95bccb6158633a4c0d8d506b8e5351be38cdf38415bd18c163bda4"
DECISION_SHA = "08b37c8f71112265c6b20a4aa8827aa516c966deb3de736bfb41220ec555cdbe"
CODE = (*diagnostic.CODE, SOURCE, PLAN)
SCHEMA = "kws-context52-known-silence-stage-v1"
JOB_ID = "context52-known-silence-2000"
STEPS, BATCH, TRAIN, FINAL = 2000, 4, 8000, 210
OUTPUTS = ("summary.json", "consumption.jsonl", "forward-calls.jsonl", "optimizer-calls.jsonl",
           "observations.jsonl", "final.pt", "MANIFEST.json")


def plan():
    require(training.sha_file(ROOT / PLAN) == PLAN_SHA, "CONTEXT52_PRIOR_FIXED_PLAN_CHANGED")
    return training.read_json(ROOT / PLAN)


def fixed_contract():
    declared = plan()
    keys = ("candidate", "training_rows", "training_families", "training_strata", "development_rows",
        "development_families", "optimizer_steps", "batch_size", "training_forward_limit", "final_forward_limit",
        "final_train_forward_calls", "final_development_forward_calls", "before_forward_calls", "offsets_ms",
        "worker_seconds", "queue_minutes", "optimizer", "gradient_clip_max_norm", "fixed_ctc_normalizer", "seed",
        "auxiliary_scale", "known_zero_frames", "maximum_model_jobs_concurrently", "asr_or_tts_calls",
        "automatic_retry", "threshold_selection", "development_optimizer_draws", "development_selection_or_tuning",
        "human_gold", "independent_test", "previously_exposed_development", "single_variable_data_size_ablation",
        "quality_winner", "generalization_passed", "initial_checkpoint_sha256", "vocabulary_sha256")
    return {**{key: declared[key] for key in keys}, "model_construction_limit": 1,
        "full_float32_final_features_and_logits_retained": True, "training_logits_retained": False,
        "training_ledger_replays_complete_optimization": False, "reset_each_forward": True,
        "labels_changed": False, "metrics_choose_training_settings": False,
        "scope": "fixed broadened-roster research training and exposed development readback"}


def validate_contract(job):
    require(type(job) is dict and job.get("schema") == SCHEMA and
        io.canonical({k: job.get(k) for k in fixed_contract()}) == io.canonical(fixed_contract()) and
        all(type(x) is int for x in job["offsets_ms"]), "CONTEXT52_FIXED_CONTRACT_CHANGED")


def sources():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def strata(rows):
    return dict(Counter("keyword-%d" % row["intended_keyword_id"]
        if row["intended_keyword_id"] is not None else "ordinary-negative" for row in rows))


def validate_roster(all_rows, dev, vocabulary, old_rows):
    """Keep original split, complete targets, weak weights and unknown masks."""
    rows = [row for row in all_rows if row["split"] == "train" and row["ctc_mask"]]
    excluded = [row for row in all_rows if not row["ctc_mask"]]
    require(len(all_rows) == len({r["source_id"] for r in all_rows}) == 72 and len(rows) == 52 and
        len({r["source_family_id"] for r in rows}) == 18 and strata(rows) == fixed_contract()["training_strata"] and
        len(dev) == 18 and len({r["source_family_id"] for r in dev}) == 6 and
        {fit.row_identity(r) for r in dev} == {fit.row_identity(r) for r in all_rows if r["split"] == "development"} and
        len(excluded) == 2 and all(r["split"] == "train" and r["generation_outcome"] == "failed-native" and
            r["ctc_weight"] == 0 and r["full_text"] is None and r["keyword_mask"] == [False, False]
            and r["keyword_targets"] == [None, None] and r["keyword_weight"] == [0, 0] for r in excluded),
        "CONTEXT52_FULL_SEALED_ROSTER_REQUIRED")
    original_identities = set().union(*(evaluation.identity_keys(r) for r in old_rows))
    for row in rows + dev:
        expected_role = "train-candidate" if row["split"] == "train" else "development-only"
        projection = fit.fulltext_targets(row["full_text"], vocabulary)
        require(row["source_role"] == expected_role and row["ctc_mask"] is True and
            type(row["ctc_weight"]) in (int, float) and math.isfinite(row["ctc_weight"]) and 0 < row["ctc_weight"] <= 1 and
            row["generation_outcome"] == "completed" and row["quarantined"] is False and
            row["human_gold"] is False and row["independent_test"] is False and projection["supported"] and
            projection["target_ids"] == row["target_ids"] and row["vocabulary_sha256"] == fit.digest(vocabulary) and
            row["expected_c1_frames"] == onset.geometry(row["samples"], 0)["frames"] and
            row["ctc_minimum_frames"] == len(row["target_ids"]) + sum(a == b for a, b in zip(row["target_ids"], row["target_ids"][1:])) and
            row["expected_c1_frames"] >= row["ctc_minimum_frames"], "CONTEXT52_ORIGINAL_FULL_TEXT_REQUIRED")
        require(len(row["keyword_targets"]) == len(row["keyword_mask"]) == len(row["keyword_weight"]) == 2,
            "CONTEXT52_KEYWORD_MASK_SHAPE")
        for text, target, mask, weight in zip(fit.KEYWORDS, row["keyword_targets"], row["keyword_mask"], row["keyword_weight"]):
            require(type(mask) is bool and type(weight) in (int, float) and math.isfinite(weight) and
                ((type(target) is int and target == int(text in row["full_text"]) and 0 < weight <= 1)
                 if mask else target is None and weight == 0), "CONTEXT52_UNKNOWN_IS_NOT_NEGATIVE")
        require(not evaluation.identity_keys(row) & original_identities, "CONTEXT52_ORIGINAL_TINY_SOURCE_OVERLAP")
    diagnostic.validate_rows(dev, vocabulary, rows)
    return rows, excluded


def exposure_plan(rows):
    require(len(rows) == len({r["source_id"] for r in rows}) == 52 and
        all(r["split"] == "train" and r["source_role"] == "train-candidate" for r in rows),
        "CONTEXT52_TRAIN_ONLY_EXPOSURE_REQUIRED")
    return [{"row_index": i % 52, "row_visit_index": i // 52, "source_id": rows[i % 52]["source_id"],
        "ctc_weight": rows[i % 52]["ctc_weight"], "offset_ms": onset.OFFSETS[(i // 52) % 3],
        "known_zero_prefix_frames": (0, 9, 24)[(i // 52) % 3]} for i in range(TRAIN)]


def validate_prerequisite(summary, audit):
    diagnostic.validate_contract(summary)
    require(summary["status"] == "completed" and audit["status"] == "verified" and
        summary["started_forward_calls"] == summary["completed_forward_calls"] == 108 and summary["model_constructions"] == 2 and
        summary["optimizer_steps"] == summary["training_forwards"] == 0 and summary["model_states_unchanged"] is True and
        audit["schema"] == diagnostic.SCHEMA and audit["saved_conditions_verified"] == 108 and
        audit["new_model_constructions"] == audit["new_model_forwards"] == audit["new_optimizer_steps"] == 0,
        "CONTEXT52_COMPLETED_PAIRED_DIAGNOSTIC_REQUIRED")


def assemble(request):
    require(type(request) is dict and set(request) == {"diagnostic_output", "diagnostic_manifest_sha256",
        "diagnostic_audit", "prior_decision"}, "CONTEXT52_INPUT_REQUEST")
    declared = plan()
    decision = context.checked(request["prior_decision"])
    require(training.sha_file(decision) == DECISION_SHA, "CONTEXT52_PRIOR_DECISION_CHANGED")
    root, prior, summary, lock = diagnostic.completed_output(request["diagnostic_output"],
        request["diagnostic_manifest_sha256"], request["diagnostic_audit"])
    audit = training.read_json(context.checked(request["diagnostic_audit"]))
    required = declared["current_diagnostic_prerequisite"]
    require(root.parent.name == required["job_id"] and training.sha_file(root / "request.json") == required["recipe_sha256"],
        "CONTEXT52_ORIGINAL_PAIRED_DIAGNOSTIC_REQUIRED")
    validate_prerequisite(summary, audit)
    # This authenticates old manifests and frozen verifiers in isolated stdlib
    # processes. Numerical observations never choose a stage setting or row.
    vocabulary = diagnostic.validate_recipe(prior)
    prepared = io.safe_path(prior["prepared"])
    require(prior["prepared_manifest_sha256"] == declared["context_prepared_manifest_sha256"] and
        prior["vocabulary_sha256"] == declared["vocabulary_sha256"] and lock == prior["heavy_lock"],
        "CONTEXT52_FIXED_INPUT_LINEAGE_CHANGED")
    all_rows = context.lines(prepared / "projected-rows.jsonl")
    old_rows = training.read_json(Path(prior["inputs"]["correction_output"]) / "selected-rows.json")
    corrected_summary = training.read_json(Path(prior["inputs"]["correction_output"]) / "summary.json")
    rows, excluded = validate_roster(all_rows, prior["development"], vocabulary, old_rows)
    record, anchor = training.read_json(prepared / "preparation.json"), training.read_json(prepared / "anchor.json")
    initial = onset.pin(prepared / "initial.pt")
    require(initial["sha256"] == silence.INITIAL_SHA == record["initial_checkpoint"]["file_sha256"] and
        anchor["base_recipe"]["optimizers"]["fresh-rnn64-or-c1"] == declared["optimizer"] and
        anchor["base_recipe"]["seed"] == declared["seed"] and anchor["base_recipe"]["gradient_clip_max_norm"] == 5.0 and
        anchor["base_recipe"]["fixed_ctc_normalizer"] == 1.0, "CONTEXT52_ZERO_INITIAL_OR_LOSS_CHANGED")
    draws = exposure_plan(rows)
    dev = prior["development"]
    job = {"schema": SCHEMA, **fixed_contract(), "inputs": request, "predeclared_plan": onset.pin(ROOT / PLAN),
        "prepared": str(prepared), "prepared_manifest_sha256": prior["prepared_manifest_sha256"],
        "initial_checkpoint": initial, "initial_binding": record["initial_checkpoint"],
        "initial_source_hashes": anchor["source_hashes"], "model_description": prior["model_description"],
        "canonical_parameter_names": corrected_summary["parameter_names"],
        "train": rows, "development": dev, "excluded_original_rows": excluded,
        "train_binding_sha256": fit.digest([fit.row_identity(r) for r in rows]),
        "development_binding_sha256": fit.digest([fit.row_identity(r) for r in dev]),
        "exposure_plan": {"draws": draws}, "exposure_plan_sha256": fit.digest(draws),
        "exposures_by_offset": {str(ms): sum(d["offset_ms"] == ms for d in draws) for ms in onset.OFFSETS},
        "exposures_by_source": dict(Counter(d["source_id"] for d in draws)),
        "vocabulary_file": prior["vocabulary_file"], "tokens_file": prior["tokens_file"],
        "frozen_input_verifiers": prior["frozen_input_verifiers"], "heavy_lock": lock, "source_hashes": sources(),
        "final_logits_payload_bytes": sum(onset.geometry(r["samples"], ms)["frames"] * 2601 * 4 for r in rows + dev for ms in onset.OFFSETS),
        "final_features_payload_bytes": sum(onset.geometry(r["samples"], ms)["frames"] * 32 * 4 for r in rows + dev for ms in onset.OFFSETS)}
    return job, vocabulary


def validate_recipe(job):
    validate_contract(job)
    rebuilt, vocabulary = assemble(job["inputs"])
    require(io.canonical(job) == io.canonical(rebuilt), "CONTEXT52_RECIPE_OR_INPUT_CHANGED")
    return vocabulary


def feature_bank(prepared, rows, torch):
    bank, evidence, pcms = {}, [], []
    for index, row in enumerate(rows):
        pcm = evaluation.read_pcm(prepared, row)
        pcms.append(pcm)
        for ms in onset.OFFSETS:
            feature = diagnostic.features(pcm, row, ms, torch)
            shape = onset.geometry(row["samples"], ms)
            if ms == 0:
                baseline = feature
            require(feature.dtype == torch.float32 and feature[shape["shift_frames"]:].equal(baseline),
                "CONTEXT52_ORIGINAL_FEATURE_SUFFIX_CHANGED")
            bank[index, ms] = feature
            evidence.append({"source_id": row["source_id"], "split": row["split"], "offset_ms": ms,
                "label_binding_sha256": fit.row_identity(row), "pcm_sha256": row["pcm_sha256"],
                "feature_sha256": training.tensor_sha(feature), "baseline_feature_sha256": training.tensor_sha(baseline), **shape})
    return bank, evidence, pcms


def call_record(kind, ordinal, row, ms, feature_sha):
    return {"kind": kind, "ordinal": ordinal, "source_id": row["source_id"], "split": row["split"],
        "source_family_id": row["source_family_id"], "label_binding_sha256": fit.row_identity(row),
        "offset_ms": ms, "feature_sha256": feature_sha,
        "logits_shape": [onset.geometry(row["samples"], ms)["frames"], 1, 2601], "logits_dtype": "float32"}


class TrainingCalls:
    """Delegate the original loop unchanged; write intent before each call."""
    def __init__(self, model, bank, evidence, rows, job, output, torch):
        self.model, self.bank, self.rows, self.job = model, bank, rows, job
        self.output, self.torch, self.count = output, torch, 0
        self.features = {(r["source_id"], r["offset_ms"]): r["feature_sha256"] for r in evidence}

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, feature):
        require(self.count < TRAIN, "CONTEXT52_TRAIN_FORWARD_LIMIT")
        draw = self.job["exposure_plan"]["draws"][self.count]
        row, ms = self.rows[draw["row_index"]], draw["offset_ms"]
        expected = self.bank[draw["row_index"], ms]
        require(feature.equal(expected.unsqueeze(0)), "CONTEXT52_TRAIN_FORWARD_INPUT_CHANGED")
        call = call_record("training", self.count, row, ms, self.features[row["source_id"], ms])
        training.append_json(self.output / "forward-calls.jsonl", {**call, "phase": "started"})
        logits = self.model(feature)
        diagnostic.check_logits(logits, expected, self.torch)
        training.append_json(self.output / "forward-calls.jsonl", {**call, "phase": "completed"})
        self.count += 1
        return logits


class OptimizerCalls:
    def __init__(self, optimizer, forward, output):
        self.optimizer, self.forward, self.output, self.count = optimizer, forward, output, 0

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self):
        require(self.count < STEPS and self.forward.count == (self.count + 1) * BATCH,
            "CONTEXT52_OPTIMIZER_CALL_ORDER")
        call = {"optimizer_step": self.count + 1, "completed_training_forwards": self.forward.count}
        training.append_json(self.output / "optimizer-calls.jsonl", {**call, "phase": "started"})
        self.optimizer.step()
        training.append_json(self.output / "optimizer-calls.jsonl", {**call, "phase": "completed"})
        self.count += 1


def ledger_readback(events, job, rows):
    result = silence.ledger_readback(events, job, rows)
    for step, event in enumerate(events):
        for item, draw in zip(event["draws"], job["exposure_plan"]["draws"][step * BATCH:(step + 1) * BATCH]):
            row = rows[draw["row_index"]]
            require(all(type(item[key]) is type(row[key]) and item[key] == row[key]
                for key in ("full_text", "pcm_sha256", "source_family_id", "ctc_weight")), "CONTEXT52_DRAW_SOURCE_CHANGED")
            require(math.isfinite(item["total_batch_contribution"]) and math.isclose(item["total_batch_contribution"],
                item["weighted_batch_contribution"] + item["aux_batch_contribution"], rel_tol=1e-6, abs_tol=1e-7),
                "CONTEXT52_EACH_DRAW_TOTAL_CHANGED")
    return {**result, "draw_contributions_checked": TRAIN, "per_draw_total_equals_primary_plus_auxiliary": True,
        "complete_optimization_numerically_replayed": False}


def verify_calls(output, job, evidence, final_calls):
    feature_hashes = {(r["source_id"], r["offset_ms"]): r["feature_sha256"] for r in evidence}
    calls = []
    for ordinal, draw in enumerate(job["exposure_plan"]["draws"]):
        row, ms = job["train"][draw["row_index"]], draw["offset_ms"]
        call = call_record("training", ordinal, row, ms, feature_hashes[row["source_id"], ms])
        calls.extend(({**call, "phase": "started"}, {**call, "phase": "completed"}))
    require(calls + final_calls == context.lines(output / "forward-calls.jsonl"), "CONTEXT52_FORWARD_LEDGER_CHANGED")
    expected = []
    for step in range(1, STEPS + 1):
        call = {"optimizer_step": step, "completed_training_forwards": step * BATCH}
        expected.extend(({**call, "phase": "started"}, {**call, "phase": "completed"}))
    require(expected == context.lines(output / "optimizer-calls.jsonl"), "CONTEXT52_OPTIMIZER_LEDGER_CHANGED")


def optimizer_readback(saved, parameters, names, canonical_names, torch):
    expected = {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01,
        "amsgrad": False, "maximize": False, "foreach": None, "capturable": False, "differentiable": False,
        "fused": None, "decoupled_weight_decay": True, "params": list(range(len(names)))}
    require(set(saved) == {"state", "param_groups"} and saved["param_groups"] == [expected] and
        set(saved["state"]) == set(expected["params"]) and names == canonical_names == list(parameters),
        "CONTEXT52_ADAMW_PARAMETER_MAP")
    moments = []
    for index, name in enumerate(names):
        state, parameter = saved["state"][index], parameters[name]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and isinstance(state["step"], torch.Tensor) and
            state["step"].dtype == torch.float32 and tuple(state["step"].shape) == () and float(state["step"]) == STEPS,
            "CONTEXT52_ADAMW_STEPS")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            require(value.shape == parameter.shape and value.dtype == parameter.dtype == torch.float32 and
                bool(value.isfinite().all()) and (key != "exp_avg_sq" or bool((value >= 0).all())), "CONTEXT52_ADAMW_MOMENTS")
        moments.append({"parameter": name, "step": STEPS,
            **{key + "_sha256": training.tensor_sha(state[key]) for key in ("exp_avg", "exp_avg_sq")}})
    return {"name": "AdamW", "parameter_names": names, "steps_per_parameter": STEPS, "moments": moments}


def conditions(observed, job):
    rows = job["train"] + job["development"]
    require([(r["source_id"], r["split"], r["offset_ms"]) for r in observed] ==
        [(r["source_id"], r["split"], ms) for r in rows for ms in onset.OFFSETS] and len(observed) == FINAL,
        "CONTEXT52_ALL_FINAL_CONDITIONS_REQUIRED")
    result = {}
    for split in ("train", "development"):
        result[split] = {}
        for ms in onset.OFFSETS:
            group = [r for r in observed if r["split"] == split and r["offset_ms"] == ms]
            words = []
            for i in range(2):
                known = [r["keywords"][i] for r in group if r["keywords"][i]["known"]]
                positive = [r for r in known if r["reference"] == 1]
                negative = [r for r in known if r["reference"] == 0]
                words.append({"keyword_id": i + 1, "known_positive_rows": len(positive),
                    "matched_positive_rows": sum(r["complete_sequence_present"] for r in positive),
                    "known_negative_rows": len(negative), "false_positive_rows": sum(r["complete_sequence_present"] for r in negative),
                    "unknown_rows": len(group) - len(known)})
            result[split][str(ms)] = {**fit.summarize(group), "keywords": words,
                "known_zero_frames": sum(r["known_zero_prefix_frames"] for r in group),
                "known_zero_nonblank_frames": sum(r["known_zero_nonblank_frames"] for r in group),
                "emissions_entirely_in_added_silence": sum(r["emissions_entirely_in_added_silence"] for r in group)}
    return result


def observe(job, output, bank, pcms, vocabulary, final_sha, torch, *, model=None, end=None, status=None):
    """Execute once, or audit saved tensors with no model construction/forward."""
    rows, observed, calls = job["train"] + job["development"], [], []
    saved_rows = context.lines(output / "observations.jsonl") if model is None else None
    require(saved_rows is None or len(saved_rows) == FINAL, "CONTEXT52_SAVED_OBSERVATION_COUNT")
    model_pin = {"model_id": JOB_ID, "checkpoint": {"sha256": final_sha}}
    history = sum((k - 1) * d for k, d in zip(job["model_description"]["kernels"], job["model_description"]["dilations"]))
    with torch.inference_mode():
        for index, row in enumerate(rows):
            baseline = None
            for ms in onset.OFFSETS:
                feature = bank[index, ms]
                name = "numeric/%s/ms-%03d/%s.pt" % (row["split"], ms, row["source_id"])
                call = call_record("final", len(observed), row, ms, training.tensor_sha(feature))
                if model is not None:
                    require(time.monotonic() < end and status["final_forwards_started"] < FINAL, "CONTEXT52_FINAL_FORWARD_BUDGET")
                    training.append_json(output / "forward-calls.jsonl", {**call, "phase": "started"})
                    status["final_forwards_started"] += 1
                    logits = model(feature.unsqueeze(0))
                    status["final_forwards_completed"] += 1
                    diagnostic.check_logits(logits, feature, torch)
                    training.append_json(output / "forward-calls.jsonl", {**call, "phase": "completed", "logits_sha256": training.tensor_sha(logits)})
                    artifact = {"path": name, **evaluation.save_numeric(output / name, row["source_id"], feature, logits)}
                else:
                    item = saved_rows[len(observed)]
                    artifact = item["numeric_artifact"]
                    require(artifact == {"path": name, **io.file_record(output / name)}, "CONTEXT52_SAVED_NUMERIC_PIN")
                    saved = torch.load(output / name, weights_only=True, map_location="cpu")
                    logits = saved["logits"]
                    diagnostic.check_logits(logits, feature, torch)
                    require(saved["source_id"] == row["source_id"] and saved["feature"].dtype == torch.float32 and
                        saved["feature"].equal(feature) and saved["feature_sha256"] == training.tensor_sha(feature) and
                        saved["logits_sha256"] == training.tensor_sha(logits), "CONTEXT52_SAVED_NUMERIC_CHANGED")
                if ms == 0:
                    baseline = logits
                item = diagnostic.observation(row, model_pin, pcms[index], ms, feature, logits, baseline,
                    bank[index, 0], history, artifact, vocabulary)
                observed.append(item)
                calls.extend(({**call, "phase": "started"}, {**call, "phase": "completed", "logits_sha256": training.tensor_sha(logits)}))
                if model is not None:
                    training.append_json(output / "observations.jsonl", item)
                else:
                    require(saved_rows[len(observed) - 1] == item, "CONTEXT52_SAVED_METRICS_CHANGED")
    return observed, calls


def initial_state(path, job):
    from training_state import load_checkpoint_snapshot, state_identity
    initial, sha = load_checkpoint_snapshot(path)
    require(sha == silence.INITIAL_SHA == job["initial_checkpoint"]["sha256"] and
        initial["schema"] == "kws-d1-initial-checkpoint-v2" and initial["candidate"] == onset.CANDIDATE and
        initial["optimizer_steps"] == 0 and initial["vocabulary_sha256"] == job["vocabulary_sha256"] and
        initial["source_hashes"] == job["initial_source_hashes"] and initial["description"] == job["model_description"] and
        list(initial["state_dict"]) == job["canonical_parameter_names"] and
        state_identity(initial["state_dict"]) == initial["float_state_identity"] and
        initial["float_state_identity"]["sha256"] == job["initial_binding"]["float_state_sha256"], "CONTEXT52_ZERO_INITIAL_CHANGED")
    return initial


def terminal_counts(status):
    require(status["model_constructions"] == 1 and status["actual_optimizer_steps"] == STEPS and
        status["training_forwards_started"] == status["training_forwards_completed"] == TRAIN and
        status["final_forwards_started"] == status["final_forwards_completed"] == FINAL,
        "CONTEXT52_TERMINAL_COUNTS")


def completed_queue(output, summary):
    require(summary["status"] == "completed" and training.read_json(output / "status.json") == summary,
        "CONTEXT52_FAILED_OR_DIVERGENT_TERMINAL_STATUS")
    receipt = training.read_json(output.parent / "result.json")
    require(receipt["schema"] == io.RECEIPT_SCHEMA and receipt["status"] == "completed" and receipt["exit_code"] == 0 and
        receipt["job_id"] == JOB_ID == output.parent.name and
        receipt["queue_sha256"] == training.sha_file(output.parents[2] / "queue.json") and
        receipt["recipe_sha256"] == summary["recipe_sha256"] == training.sha_file(output / "request.json") and
        receipt["outputs"]["MANIFEST.json"] == io.file_record(output / "MANIFEST.json"),
        "CONTEXT52_COMPLETED_QUEUE_RECEIPT_REQUIRED")


def run(recipe_path, output, lock):
    job = training.read_json(recipe_path)
    validate_contract(job)
    require(str(io.safe_path(lock)) == job["heavy_lock"], "CONTEXT52_CANONICAL_LOCK_REQUIRED")
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        require(os.environ.get("KWS_RESEARCH_JOB_ID") == JOB_ID and
            os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(recipe_path).absolute()) and
            os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(Path(output).absolute()), "CONTEXT52_QUEUE_SCOPE")
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        status = {"schema": SCHEMA, **fixed_contract(), "status": "running", "started_at": training.utc(),
            "recipe_sha256": training.sha_file(recipe_path), "model_constructions": 0, "actual_optimizer_steps": 0,
            "training_forwards_started": 0, "training_forwards_completed": 0,
            "final_forwards_started": 0, "final_forwards_completed": 0}
        model = optimizer = None
        training.save_json(output / "status.json", status)
        try:
            with diagnostic.deadline() as end:
                vocabulary = validate_recipe(job)
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                torch, reference = training.torch_runtime()
                from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
                rows, dev, prepared = job["train"], job["development"], Path(job["prepared"])
                training.copy_once(recipe_path, output / "request.json")
                for name in CODE:
                    training.copy_once(ROOT / name, output / "execution-source" / name)
                for name in ("initial.pt", "vocabulary.json", "tokens.txt", "projected-rows.jsonl"):
                    training.copy_once(prepared / name, output / name)
                for row in rows + dev:
                    training.copy_once(prepared / row["audio"], output / row["audio"])
                for name, value in (("train-rows.json", rows), ("development-rows.json", dev),
                                    ("excluded-original-rows.json", job["excluded_original_rows"])):
                    io.write_json(output / name, value)
                initial = initial_state(output / "initial.pt", job)
                bank, evidence, pcms = feature_bank(output, rows + dev, torch)
                io.write_json(output / "feature-bindings.json", evidence)
                torch.manual_seed(job["seed"])
                model, description = training.build_fresh(onset.CANDIDATE, vocabulary, reference)
                status["model_constructions"] = 1
                require(description == job["model_description"], "CONTEXT52_ARCHITECTURE_CHANGED")
                model.load_state_dict(initial["state_dict"], strict=True)
                require(state_identity(model.state_dict()) == initial["float_state_identity"], "CONTEXT52_INITIAL_LOAD_CHANGED")
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                forward = TrainingCalls(model, bank, evidence, rows, job, output, torch)
                updates = OptimizerCalls(optimizer, forward, output)
                silence.train(forward, bank, rows, job, updates, output, end, status, torch)
                gradients = ledger_readback(context.lines(output / "consumption.jsonl"), job, rows)
                require(forward.count == TRAIN and updates.count == STEPS, "CONTEXT52_ACCOUNTED_TRAINING_COUNTS")
                identity = state_identity(model.state_dict())
                names = [name for name, _ in model.named_parameters()]
                require(names == job["canonical_parameter_names"] == list(model.state_dict()), "CONTEXT52_CANONICAL_PARAMETER_ORDER")
                changed, delta = fit.parameter_changes(initial["state_dict"], model.state_dict(), names)
                payload = {"schema": SCHEMA, "candidate": onset.CANDIDATE, "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "float_state_identity": identity,
                    "job_recipe_sha256": status["recipe_sha256"], "optimizer_steps": STEPS, "training_model_forwards": TRAIN,
                    "initial_checkpoint_sha256": silence.INITIAL_SHA, "vocabulary_sha256": job["vocabulary_sha256"],
                    "train_binding_sha256": job["train_binding_sha256"], "source_hashes": job["source_hashes"]}
                # Preserve the actual trained checkpoint even if final readback times out.
                atomic_save_checkpoint(payload, output / "final.pt")
                final, final_sha = load_checkpoint_snapshot(output / "final.pt")
                require(state_identity(final["state_dict"]) == identity, "CONTEXT52_FINAL_CHECKPOINT_READBACK")
                optim = optimizer_readback(final["optimizer_state"], final["state_dict"], names, job["canonical_parameter_names"], torch)
                model.eval()
                observed, calls = observe(job, output, bank, pcms, vocabulary, final_sha, torch, model=model, end=end, status=status)
                require(state_identity(model.state_dict()) == identity, "CONTEXT52_READBACK_MUTATED_MODEL")
                verify_calls(output, job, evidence, calls)
                terminal_counts(status)
                validate_recipe(job)
                require(time.monotonic() < end, "CONTEXT52_TERMINAL_DEADLINE")
                now = resource.getrusage(resource.RUSAGE_SELF)
                status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                    user_cpu_seconds=now.ru_utime - usage.ru_utime, system_cpu_seconds=now.ru_stime - usage.ru_stime,
                    peak_rss_kib=now.ru_maxrss, final_checkpoint_sha256=final_sha,
                    initial_float_state_sha256=initial["float_state_identity"]["sha256"], final_float_state_sha256=identity["sha256"],
                    parameter_names=names, changed_parameter_names=changed, parameter_delta_l2=delta,
                    gradients=gradients, optimizer_readback=optim, conditions=conditions(observed, job),
                    exposure_plan_sha256=job["exposure_plan_sha256"], exposures_by_offset=job["exposures_by_offset"],
                    train_binding_sha256=job["train_binding_sha256"], development_binding_sha256=job["development_binding_sha256"],
                    final_observation_state_unchanged=True)
                training.save_json(output / "status.json", status)
                io.write_json(output / "summary.json", status)
                context.manifest(output)
                return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), error_type=type(error).__name__, error=str(error),
                wall_seconds=time.monotonic() - started, counts_are_lower_bounds=True,
                possible_incomplete_call="retain started claims; do not retry or infer unrecorded completion")
            if model is not None and optimizer is not None and not (output / "final.pt").exists():
                try:
                    atomic_save_checkpoint({"schema": SCHEMA, "status": "partial-failure-not-final",
                        "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "job_recipe_sha256": status["recipe_sha256"], "counts_are_lower_bounds": True,
                        "recorded_optimizer_steps": status["actual_optimizer_steps"]}, output / "partial.pt")
                except BaseException as save_error:
                    status["partial_checkpoint_error"] = type(save_error).__name__ + ":" + str(save_error)
            training.save_json(output / "status.json", status)
            if not (output / "MANIFEST.json").exists():
                context.manifest(output)
            raise


def audit(output, expected):
    """Authenticate saved input/call/loss/optimizer evidence and all 210 tensors."""
    output = io.safe_path(output)
    onset.verify_manifest(output, expected)
    job, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    completed_queue(output, summary)
    with training.heavy_model_slot(job["heavy_lock"]):
        vocabulary = validate_recipe(job)
        validate_contract(summary)
        terminal_counts(summary)
        require(summary["status"] == "completed" and summary["recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["final_observation_state_unchanged"] is True and training.read_json(output / "train-rows.json") == job["train"] and
            training.read_json(output / "development-rows.json") == job["development"] and
            training.read_json(output / "excluded-original-rows.json") == job["excluded_original_rows"] and
            {n: training.sha_file(output / "execution-source" / n) for n in CODE} == job["source_hashes"], "CONTEXT52_SAVED_BINDINGS")
        runtime = training.research_runtime.load_manifest()
        training.research_runtime.check_interpreter(runtime)
        training.research_runtime.check_packages(runtime, "core")
        torch, _ = training.torch_runtime()
        from training_state import load_checkpoint_snapshot, state_identity
        initial = initial_state(output / "initial.pt", job)
        final, final_sha = load_checkpoint_snapshot(output / "final.pt")
        require(final_sha == summary["final_checkpoint_sha256"] and final["schema"] == SCHEMA and
            final["candidate"] == onset.CANDIDATE and final["optimizer_steps"] == STEPS and final["training_model_forwards"] == TRAIN and
            final["job_recipe_sha256"] == summary["recipe_sha256"] and final["initial_checkpoint_sha256"] == silence.INITIAL_SHA and
            final["vocabulary_sha256"] == job["vocabulary_sha256"] and final["train_binding_sha256"] == job["train_binding_sha256"] and
            final["source_hashes"] == job["source_hashes"] and state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"], "CONTEXT52_SAVED_FINAL_CHANGED")
        names = summary["parameter_names"]
        optim = optimizer_readback(final["optimizer_state"], final["state_dict"], names, job["canonical_parameter_names"], torch)
        changed, delta = fit.parameter_changes(initial["state_dict"], final["state_dict"], names)
        require(optim == summary["optimizer_readback"] and changed == summary["changed_parameter_names"] and
            delta == summary["parameter_delta_l2"] and initial["float_state_identity"]["sha256"] == summary["initial_float_state_sha256"] and
            ledger_readback(context.lines(output / "consumption.jsonl"), job, job["train"]) == summary["gradients"],
            "CONTEXT52_SAVED_OPTIMIZER_OR_LOSS_CHANGED")
        bank, evidence, pcms = feature_bank(output, job["train"] + job["development"], torch)
        require(evidence == training.read_json(output / "feature-bindings.json"), "CONTEXT52_SAVED_FEATURES_CHANGED")
        observed, calls = observe(job, output, bank, pcms, vocabulary, final_sha, torch)
        verify_calls(output, job, evidence, calls)
        require(conditions(observed, job) == summary["conditions"] and
            all(summary[k] == job[k] for k in ("exposure_plan_sha256", "exposures_by_offset", "train_binding_sha256", "development_binding_sha256")),
            "CONTEXT52_SAVED_SUMMARY_CHANGED")
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected,
            "saved_training_forwards": TRAIN, "saved_optimizer_steps": STEPS, "saved_final_conditions": FINAL,
            "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0,
            "complete_optimization_numerically_replayed": False, "quality_winner": None, "generalization_passed": None,
            "independent_test": False}


def queue_definition(recipe, python):
    job = training.read_json(recipe)
    validate_contract(job)
    return {"schema": io.SCHEMA, "queue_id": JOB_ID, "global_budget_minutes": 10, "jobs": [{
        "job_id": JOB_ID, "recipe": str(Path(recipe).absolute()), "recipe_sha256": training.sha_file(recipe),
        "argv": [python, "-I", "-B", str(ROOT / SOURCE), "run", "--recipe", "{recipe}",
                 "--output", "{output_dir}", "--heavy-lock", job["heavy_lock"]],
        "cwd": str(ROOT), "depends_on": [], "timeout_minutes": 10, "outputs": list(OUTPUTS)}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = (("declare", ("diagnostic-output", "diagnostic-manifest-sha256", "diagnostic-audit", "prior-decision", "output")),
        ("run", ("recipe", "output", "heavy-lock")), ("audit", ("output", "manifest-sha256", "receipt")),
        ("queue-request", ("recipe", "python", "output")))
    for name, keys in commands:
        command = sub.add_parser(name)
        for key in keys:
            command.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        request = {"diagnostic_output": str(io.safe_path(args.diagnostic_output)),
            "diagnostic_manifest_sha256": args.diagnostic_manifest_sha256,
            "diagnostic_audit": onset.pin(args.diagnostic_audit), "prior_decision": onset.pin(args.prior_decision)}
        job, _ = assemble(request)
        io.write_json(Path(args.output), job)
        result = {"recipe": args.output, "sha256": training.sha_file(args.output), "new_model_calls": 0,
            "training_forward_limit": TRAIN, "final_forward_limit": FINAL, "exposures_by_offset": job["exposures_by_offset"],
            "final_logits_payload_bytes": job["final_logits_payload_bytes"], "final_features_payload_bytes": job["final_features_payload_bytes"]}
    elif args.command == "queue-request":
        io.write_json(Path(args.output), queue_definition(args.recipe, args.python))
        result = {"queue_request": args.output, "sha256": training.sha_file(args.output), "new_model_calls": 0}
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        result = audit(args.output, args.manifest_sha256)
        io.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
