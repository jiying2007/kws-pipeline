#!/usr/bin/env python3
"""One fixed 3:1 source-allocation comparison against the saved mixed C1 arm.

Preparation is metadata-only. The numerical loop, losses, initial state,
observation and audit are shared through explicit ExecutionPolicy dependencies.
There is no ratio/step sweep and no new data or baseline model execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_c1_source_augmentation as engine

io, require, training, fit = engine.io, engine.require, engine.training, engine.fit
SOURCE = "training/research_c1_source_allocation.py"
CODE = (*engine.CODE, SOURCE)
SCHEMA = "kws-c1-fixed-source-allocation-v1"
REQUEST = "kws-c1-fixed-source-allocation-request-v1"
DECISION = "kws-c1-allocation-decision-v1"
JOB_ID = "c1-old3-new1-fixed2000"
SAMPLER = "old3-new1-fair-cycle-v1"
CONTROL_RECIPE_SHA = "75b5743222f96dde75fca0191159cb74a4a53d55531e625d4a4d255b46f2b0cd"
CONTROL_MANIFEST_SHA = "06dc3b01a63500a9a88f149cdc73ba50493849f620668f909354b856456c8ea3"
CONTROL_CHECKPOINT_SHA = "13cdff7365f5c34e0e8f09d73f32d73d90b462a447245506d7acac20695d8f92"
CONTROL_COMPARISON_SHA = "ec9b488266bcab8480b08dd88835fdd998954c028b6903f0b89008099ac93844"
OLD_ROWS, NEW_ROWS, DEVELOPMENT_ROWS, UNKNOWN_ROWS = 52, 44, 34, 4
OLD_DRAWS, NEW_DRAWS = 6000, 2000


def contract(mode="augmentation2000"):
    require(mode == "augmentation2000", "ALLOCATION_TRAINING_ARM_ONLY")
    return {**engine.contract(mode), "schema": SCHEMA, "sampler": SAMPLER,
        "source_allocation": {"old_per_batch": 3, "new_per_batch": 1,
            "old_draws": OLD_DRAWS, "new_draws": NEW_DRAWS},
        "train_rows": OLD_ROWS + NEW_ROWS, "development_rows": DEVELOPMENT_ROWS,
        "retained_unknown_train_rows": UNKNOWN_ROWS, "maximum_invocations": 1,
        "execution_authorized": True, "single_variable_sampling_policy_ablation": True,
        "marginal_source_ratio_effect_isolated": False,
        "sampling_policy_changes": ["cohort_draw_quota", "within_batch_cohort_mix", "row_visit_and_offset_order"]}


def validate_contract(job):
    fixed = contract(job.get("mode"))
    require(io.canonical({k: job.get(k) for k in fixed}) == io.canonical(fixed) and
        job.get("job_id") == JOB_ID and job.get("final_forward_limit") == 402,
        "ALLOCATION_FIXED_CONTRACT")


def validate_roster(control):
    """Check actual source cohorts, full evaluation coverage and zero-loss rows."""
    rows, dev, new = control["train"], control["development"], control["all_new_rows"]
    require(len(rows) == OLD_ROWS + NEW_ROWS and len(dev) == DEVELOPMENT_ROWS and len(new) == 64,
        "ALLOCATION_FIXED_ROSTER_COUNTS")
    new_ids = {r["source_id"] for r in new}
    require(len(new_ids) == len(new), "ALLOCATION_DUPLICATE_NEW_SOURCE")
    require(all(r["split"] == "train" and r["source_role"] == "train-candidate" and
        r["ctc_mask"] is True and type(r["ctc_weight"]) in (int, float) and
        math.isfinite(r["ctc_weight"]) and 0 < r["ctc_weight"] <= 1 and
        r["target_ids"] is not None for r in rows), "ALLOCATION_TRAIN_ONLY_KNOWN")
    require(all(r["split"] == "development" and r["source_role"] == "development-only" for r in dev),
        "ALLOCATION_DEVELOPMENT_ROLE")
    old = [r for r in rows + dev if r["source_id"] not in new_ids] + control["excluded_original_rows"]
    engine.role_check(old, new)
    require(sum(r["source_id"] in new_ids for r in rows) == NEW_ROWS and
        sum(r["source_id"] in new_ids for r in dev) == 16, "ALLOCATION_SOURCE_COHORT_COUNTS")
    by_id = {r["source_id"]: r for r in new}
    require(all(r == by_id[r["source_id"]] for r in rows + dev if r["source_id"] in new_ids),
        "ALLOCATION_SOURCE_IDENTITY_CHANGED")
    unknown = [r for r in new if r["split"] == "train" and not r["ctc_mask"]]
    require(len(unknown) == UNKNOWN_ROWS and all(r["ctc_mask"] is False and
        r["ctc_weight"] == 0 and r["target_ids"] is None for r in unknown) and
        [r["source_id"] for r in unknown] == control["new_zero_loss_rows"] and
        control["evaluation_rows"] == rows + unknown + dev and
        len(control["evaluation_rows"]) == 134, "ALLOCATION_UNKNOWN_OR_EVALUATION_CHANGED")
    require(len({r["source_id"] for r in rows + unknown + dev}) == 134,
        "ALLOCATION_DUPLICATE_EVALUATION_SOURCE")
    return new_ids


def exposure_plan(rows, new_ids):
    """Three old visits then one new visit; each source cycles its own offsets."""
    require(len(rows) == OLD_ROWS + NEW_ROWS and len({r["source_id"] for r in rows}) == len(rows) and
        all(r["split"] == "train" and r["source_role"] == "train-candidate" and r["ctc_mask"] is True and
            type(r["ctc_weight"]) in (int, float) and math.isfinite(r["ctc_weight"]) and
            0 < r["ctc_weight"] <= 1 and r["target_ids"] is not None for r in rows),
        "ALLOCATION_TRAIN_ONLY_KNOWN")
    pools = {False: [i for i, row in enumerate(rows) if row["source_id"] not in new_ids],
        True: [i for i, row in enumerate(rows) if row["source_id"] in new_ids]}
    require(len(pools[False]) == OLD_ROWS and len(pools[True]) == NEW_ROWS,
        "ALLOCATION_SOURCE_COHORT_COUNTS")
    cursors, visits, draws = {False: 0, True: 0}, Counter(), []
    for _ in range(engine.STEPS):
        for is_new in (False, False, False, True):
            pool = pools[is_new]
            index = pool[cursors[is_new] % len(pool)]
            visit = visits[index]
            draws.append({"row_index": index, "row_visit_index": visit, "source_id": rows[index]["source_id"],
                "ctc_weight": rows[index]["ctc_weight"], "offset_ms": engine.OFFSETS[visit % 3],
                "known_zero_prefix_frames": (0, 9, 24)[visit % 3]})
            cursors[is_new] += 1
            visits[index] += 1
    require(cursors == {False: OLD_DRAWS, True: NEW_DRAWS} and len(draws) == engine.DRAWS,
        "ALLOCATION_FIXED_EXPOSURE_COUNT")
    return draws


def validate_budget(budget, heavy_lock):
    require(type(budget) is dict and budget.get("schema") == "kws-posterior-allocation-continuation-stage-v1" and
        type(budget.get("global_runner_minutes_ceiling")) is int and
        0 < budget["global_runner_minutes_ceiling"] <= 350 and
        budget.get("python_version") == "3.12.16" and budget.get("heavy_model_concurrency") == 1 and
        budget.get("threads") == 4 and budget.get("interop_threads") == 1 and
        budget.get("shared_heavy_lock") == heavy_lock and
        budget.get("model_execution_requires_frozen_followup_decision") is True and
        all(budget.get(k) == 0 for k in ("new_tts_calls", "new_asr_calls", "unknown_model_retries",
            "paid_model_api_calls", "external_codex_or_work_jobs")), "ALLOCATION_STAGE_BUDGET")
    reservations = budget.get("reservations_runner_minutes", {})
    require(type(reservations) is dict and reservations and
        all(type(v) is int and v >= 0 for v in reservations.values()) and
        sum(reservations.values()) <= budget["global_runner_minutes_ceiling"] and
        reservations.get("optional_single_allocation_training_and_audit", 0) >= 19,
        "ALLOCATION_AGGREGATE_BUDGET")
    phases = [p for p in budget.get("phases", []) if p.get("id") == "source-allocation-candidate"]
    required = {"new_training_forwards_max": 8000, "new_optimizer_steps_max": 2000,
        "new_final_forwards_max": 402, "model_constructions_max": 1,
        "worker_seconds_max": 840, "queue_seconds_max": 900}
    require(len(phases) == 1 and all(type(phases[0].get(k)) is int and phases[0][k] == v
        for k, v in required.items()), "ALLOCATION_PHASE_BUDGET")


def validate_decision(value, request):
    fields = {"schema", "decision", "posterior_summary", "control_completed_evidence", "rationale",
        "budget", "one_model_invocation", "no_threshold_or_loss_tuning"}
    require(type(value) is dict and set(value) == fields and value["schema"] == DECISION and
        value["decision"] == "run-single-fixed-allocation-3to1" and
        io.canonical(value["control_completed_evidence"]) == io.canonical(request["control"]) and
        value["budget"] == request["budget"] and value["one_model_invocation"] is True and
        value["no_threshold_or_loss_tuning"] is True and
        type(value["rationale"]) is str and bool(value["rationale"].strip()),
        "ALLOCATION_POSTERIOR_DECISION")
    # Validate actual completion and source bindings, not the operator's
    # scientific interpretation or a synthetic approval score.
    summary = training.read_json(engine.checked(value["posterior_summary"]))
    require(summary.get("schema") == "kws-c1-posterior-audit-v1" and summary.get("status") == "completed" and
        all(type(summary.get(k)) is int and summary[k] == 0 for k in
            ("new_model_constructions", "new_model_forwards", "new_optimizer_steps")),
        "ALLOCATION_POSTERIOR_NOT_COMPLETED_SAVED_ONLY")
    inputs = summary.get("inputs", {})
    require(inputs.get("comparison", {}).get("sha256") == CONTROL_COMPARISON_SHA and
        inputs.get("augmentation", {}).get("manifest_sha256") == request["control"]["manifest_sha256"] ==
        CONTROL_MANIFEST_SHA, "ALLOCATION_POSTERIOR_CONTROL_BINDING")
    engine.checked(inputs["comparison"])
    workload = summary.get("workload", {})
    require(all(type(workload.get(k)) is int and workload[k] == v for k, v in
        {"new_numeric_conditions": 570, "reused_conditions": 210,
         "unscored_conditions": 24, "paired_model_conditions": 804}.items()),
        "ALLOCATION_POSTERIOR_INCOMPLETE_WORKLOAD")


def control_evidence(value):
    output, control, summary = engine.completed_evidence(value)
    require(value["manifest_sha256"] == CONTROL_MANIFEST_SHA and
        training.sha_file(output / "request.json") == CONTROL_RECIPE_SHA and
        summary["final_checkpoint_sha256"] == CONTROL_CHECKPOINT_SHA and
        control["mode"] == "augmentation2000", "ALLOCATION_EXACT_CONTROL")
    # This is the code saved by the original completed run, not current code
    # retroactively substituted for that execution.
    require({name: training.sha_file(output / "execution-source" / name)
        for name in control["source_hashes"]} == control["source_hashes"],
        "ALLOCATION_CONTROL_SAVED_SOURCE")
    validate_roster(control)
    return output, control, summary


def source_hashes(control):
    current = {name: training.sha_file(ROOT / name) for name in CODE}
    require(set(control["source_hashes"]) == set(engine.CODE) and
        all(current[name] == sha for name, sha in control["source_hashes"].items() if name != engine.SOURCE),
        "ALLOCATION_NUMERICAL_DEPENDENCY_CHANGED")
    return current


def declare(request, *, allow_pending=False):
    require(type(request) is dict and set(request) == {"schema", "control", "budget", "decision"} and
        request["schema"] == REQUEST, "ALLOCATION_REQUEST")
    output, control, _ = control_evidence(request["control"])
    validate_budget(training.read_json(engine.checked(request["budget"])), control["heavy_lock"])
    if request["decision"] is None:
        require(allow_pending, "ALLOCATION_POSTERIOR_DECISION_REQUIRED")
    else:
        validate_decision(training.read_json(engine.checked(request["decision"])), request)
    for name in ("initial_checkpoint", "vocabulary_file", "tokens_file"):
        engine.checked(control[name])
    for record in control["audio_files"].values():
        engine.checked(record)
    new_ids = validate_roster(control)
    draws = exposure_plan(control["train"], new_ids)
    job = {**copy.deepcopy(control), **contract(), "job_id": JOB_ID,
        "execution_authorized": request["decision"] is not None,
        "inputs": copy.deepcopy(request), "control_original_inputs": copy.deepcopy(control["inputs"]),
        "control_recipe": engine.pin(output / "request.json"),
        "control_checkpoint": engine.pin(output / "final.pt"),
        "control_source_hashes": copy.deepcopy(control["source_hashes"]),
        "source_hashes": source_hashes(control),
        "exposure_plan": {"draws": draws}, "exposure_plan_sha256": fit.digest(draws),
        "exposures_by_source": dict(Counter(d["source_id"] for d in draws)),
        "exposures_by_offset": {str(ms): sum(d["offset_ms"] == ms for d in draws) for ms in engine.OFFSETS}}
    return job


def validate_recipe(job):
    validate_contract(job)
    require(io.canonical(declare(job["inputs"])) == io.canonical(job), "ALLOCATION_RECIPE_CHANGED")
    return training.read_json(engine.checked(job["vocabulary_file"]))


def execution_policy():
    return engine.ExecutionPolicy(SCHEMA, SOURCE, CODE, contract, validate_contract, validate_recipe)


def compare(request):
    require(type(request) is dict and set(request) == {"control", "allocation"}, "ALLOCATION_COMPARE_REQUEST")
    control_output, control_job, control_summary = control_evidence(request["control"])
    output, job, summary = engine.completed_evidence(request["allocation"], policy=execution_policy())
    validate_recipe(job)
    require(io.canonical(job["inputs"]["control"]) == io.canonical(request["control"]) and
        control_summary["initial_float_state_sha256"] == summary["initial_float_state_sha256"] ==
        job["initial_binding"]["float_state_sha256"] and
        all(control_job[k] == job[k] for k in ("train", "development", "evaluation_rows", "all_new_rows",
            "excluded_original_rows", "new_zero_loss_rows", "initial_checkpoint", "initial_binding",
            "vocabulary_file", "tokens_file", "model_description", "canonical_parameter_names", "optimizer",
            "fixed_ctc_normalizer", "auxiliary_scale", "gradient_clip_max_norm", "batch_size", "optimizer_steps",
            "training_forward_limit", "final_forward_limit", "seed", "offsets_ms")),
        "ALLOCATION_COMPARISON_NOT_SAMPLER_ONLY")
    original = engine.context.lines(control_output / "observations.jsonl")
    actual = engine.context.lines(output / "observations.jsonl")
    key = lambda r: (r["source_id"], r["split"], r["offset_ms"])
    require(len(original) == len(actual) == 402 and [key(r) for r in original] == [key(r) for r in actual] and
        len({key(r) for r in original}) == 402 and
        training.read_json(control_output / "feature-bindings.json") ==
        training.read_json(output / "feature-bindings.json"), "ALLOCATION_PAIRED_INPUTS_CHANGED")
    pairs = []
    for before, after in zip(original, actual):
        require(before["label_binding_sha256"] == after["label_binding_sha256"], "ALLOCATION_LABEL_CHANGED")
        pair = {"source_id": after["source_id"], "split": after["split"], "offset_ms": after["offset_ms"],
            "control": before, "allocation": after}
        if before["evaluation_status"] == after["evaluation_status"] == "evaluated":
            pair["allocation_minus_control"] = {k: after[k] - before[k] for k in
                ("ctc_loss_target_length_normalized", "ctc_loss_confidence_weighted", "edit_numerator", "deletions")}
        pairs.append(pair)
    return {"schema": "kws-c1-fixed-source-allocation-comparison-v1", "status": "completed-saved-only",
        "inputs": request, "all_pairs": pairs, "conditions": 402, "control_conditions": engine.conditions(original, job),
        "allocation_conditions": engine.conditions(actual, job), "new_model_forwards": 0, "new_optimizer_steps": 0,
        "single_variable_sampling_policy_ablation": True, "marginal_source_ratio_effect_isolated": False,
        "sampling_policy_changes": ["cohort_draw_quota", "within_batch_cohort_mix", "row_visit_and_offset_order"],
        "human_gold": False, "independent_test": False,
        "previously_exposed_development": True, "offsets_are_repeated_conditions_not_independent_rows": True,
        "quality_winner": None, "generalization_passed": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fields in (("preview", ("request", "output")), ("declare", ("request", "output")),
        ("queue-request", ("recipe", "python", "output")), ("run", ("recipe", "output", "heavy-lock")),
        ("audit", ("output", "manifest-sha256", "receipt")), ("compare", ("request", "output"))):
        command = sub.add_parser(name)
        for field in fields:
            command.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    if args.command in ("preview", "declare"):
        job = declare(training.read_json(args.request), allow_pending=args.command == "preview")
        io.write_json(Path(args.output), job)
        result = {"recipe": engine.pin(args.output), "new_model_calls": 0,
            "execution_authorized": job["execution_authorized"], "source_allocation": job["source_allocation"]}
    elif args.command == "queue-request":
        validate_recipe(training.read_json(args.recipe))
        io.write_json(Path(args.output), engine.queue_definition(args.recipe, args.python, policy=execution_policy()))
        result = {"queue_request": engine.pin(args.output), "new_model_calls": 0}
    elif args.command == "run":
        result = engine.run(Path(args.recipe), Path(args.output), Path(args.heavy_lock), policy=execution_policy())
    elif args.command == "audit":
        result = engine.audit(args.output, args.manifest_sha256, policy=execution_policy())
        io.write_json(Path(args.receipt), result)
    else:
        result = compare(training.read_json(args.request))
        io.write_json(Path(args.output), result)
        result = {k: v for k, v in result.items()
            if k not in ("all_pairs", "control_conditions", "allocation_conditions")}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
