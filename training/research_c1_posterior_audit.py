#!/usr/bin/env python3
"""Saved balanced64 posterior adapter: 570 new analyses, 210 reuse, 24 unknown.

The unchanged saved-CTC kernel owns all numerical CTC/FB/Viterbi computation.
This module binds the current saved inputs and retains unknown masks. It never
constructs a model, runs inference, trains, or substitutes generation intent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import resource
import signal
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_saved_ctc_diagnostic as kernel

training, fit, io, require = kernel.training, kernel.fit, kernel.io, kernel.require
SOURCE = "training/research_c1_posterior_audit.py"
PLAN = "configs/research/c1-saved-posterior-20261106.json"
TEST = "tests/test_research_c1_posterior_audit.py"
DOC = "docs/research-c1-posterior-audit.md"
PLAN_SHA = "37d0da96f014c57f8d22a7be904097f05a9b20fe7d8bf6ff4b0efb6809b5466e"
CODE = tuple(dict.fromkeys((*kernel.CODE, SOURCE, PLAN, TEST, DOC)))
SCHEMA = "kws-c1-posterior-audit-v1"
JOB_ID = "c1-balanced64-saved-posterior570"
OUTPUTS = ("summary.json", "status.json", "public-summary.json", "conditions.jsonl",
           "coverage.json", "analysis-calls.jsonl", "MANIFEST.json")
METRICS = ("q_weighted_raw_class_probability", "q_weighted_logit_margin_over_blank",
    "q_weighted_logit_margin_over_other_nonblank", "q_weighted_rank_all", "q_weighted_rank_nonblank",
    "q_mass_blank_argmax", "q_mass_target_argmax", "q_mass_other_argmax", "q_mass_rank3",
    "q_mass_in_fully_added_zero_frames", "conditional_width80_frames", "conditional_temporal_entropy_nats")
BOUNDARIES = {"new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0,
    "optimizer_steps": 0, "backward_calls": 0, "asr_or_tts_calls": 0, "automatic_retry": False,
    "human_gold": False, "independent_test": False, "previously_exposed_development": True,
    "quality_winner": None, "generalization_passed": None, "decoder_replay_performed": False,
    "threshold_selection": False, "pairwise_preferences_only": True,
    "conditional_alignment_is_not_acoustic_truth": True, "original_logits_duplicated": False,
    "offsets_are_repeated_conditions_not_independent_rows": True}


def equal(a, b, why):
    kernel.equal(a, b, "C1_POSTERIOR_" + why)


def plan():
    require(training.sha_file(ROOT / PLAN) == PLAN_SHA, "C1_POSTERIOR_FIXED_PLAN_CHANGED")
    return training.read_json(ROOT / PLAN)


def sources(contract):
    current = {name: training.sha_file(ROOT / name) for name in CODE}
    equal({name: current[name] for name in kernel.CODE}, contract["kernel_source_hashes"], "KERNEL_BYTES_CHANGED")
    return current


def key(item):
    return item["source_id"], item["split"], item["offset_ms"]


def pin(path):
    return kernel.pin(path)


def pinned_json(path, expected=None):
    record = pin(path)
    require(expected is None or record["sha256"] == expected, "C1_POSTERIOR_METADATA_CHANGED")
    return record, training.read_json(path)


def manifest_metadata(output, expected):
    """Read manifest bytes only; do not open any tensor during declaration."""
    output = io.safe_path(output)
    record, manifest = pinned_json(output / "MANIFEST.json", expected)
    return output, record, manifest


def metadata_file(output, manifest, name):
    record = pin(output / name)
    equal({k: record[k] for k in ("bytes", "sha256")}, manifest[name], "MANIFEST_METADATA:" + name)
    return record


def artifact_from_manifest(output, manifest, artifact):
    require(set(artifact) == {"path", "bytes", "sha256"}, "C1_POSTERIOR_ARTIFACT_FIELDS")
    relative = io.relative(artifact["path"])
    equal({k: artifact[k] for k in ("bytes", "sha256")}, manifest[str(relative)], "ARTIFACT_MANIFEST")
    path = io.safe_path(output / relative)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == artifact["bytes"],
        "C1_POSTERIOR_REGULAR_ARTIFACT_SIZE")
    return {**artifact, "path": str(path)}


def terminal_metadata(output, manifest_pin, recipe_pin, summary):
    equal(summary, training.read_json(output / "status.json"), "ORIGIN_STATUS_DIVERGED")
    result_pin, result = pinned_json(output.parent / "result.json")
    queue_pin = pin(output.parents[2] / "queue.json")
    claim_pin = pin(output.parent / "claim.json")
    require(summary["status"] == result["status"] == "completed" and result["exit_code"] == 0 and
        result["job_id"] == output.parent.name and result["recipe_sha256"] == recipe_pin["sha256"] and
        summary["recipe_sha256"] == recipe_pin["sha256"] and result["queue_sha256"] == queue_pin["sha256"] and
        result["claim_sha256"] == claim_pin["sha256"] and
        result["outputs"]["MANIFEST.json"] == {k: manifest_pin[k] for k in ("bytes", "sha256")},
        "C1_POSTERIOR_COMPLETED_ORIGIN_REQUIRED")
    return {"queue_result": result_pin, "queue": queue_pin, "claim": claim_pin}


def new_origin(value, kind, contract):
    require(set(value) == {"output", "manifest_sha256", "audit"} and
        value["manifest_sha256"] == contract["manifest_sha256"][kind], "C1_POSTERIOR_ORIGIN_SCOPE")
    output, manifest_pin, manifest = manifest_metadata(value["output"], value["manifest_sha256"])
    names = ("request.json", "summary.json", "status.json", "evaluation-rows.json", "observations.jsonl",
             "forward-calls.jsonl", "vocabulary.json")
    pins = {name: metadata_file(output, manifest, name) for name in names}
    require(pins["request.json"]["sha256"] == contract["recipe_sha256"][kind], "C1_POSTERIOR_ORIGINAL_RECIPE")
    request, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    receipts = terminal_metadata(output, manifest_pin, pins["request.json"], summary)
    audit_pin, audit = pinned_json(value["audit"]["path"])
    equal(audit_pin, value["audit"], "ORIGINAL_AUDIT_PIN")
    expected_count = contract["origin_conditions"][kind]
    require(audit["status"] == "verified" and audit["manifest_sha256"] == manifest_pin["sha256"] and
        audit["saved_final_conditions"] == expected_count and
        audit["new_model_constructions"] == audit["new_model_forwards"] == audit["new_optimizer_steps"] == 0 and
        audit["saved_training_forwards"] == request["training_forward_limit"] == (8000 if kind == "augmentation" else 0) and
        audit["saved_optimizer_steps"] == request["optimizer_steps"] == (2000 if kind == "augmentation" else 0) and
        summary["final_checkpoint_sha256"] == contract["checkpoint_sha256"][kind], "C1_POSTERIOR_ORIGIN_AUDIT_OR_MODEL")
    rows = training.read_json(output / "evaluation-rows.json")
    equal([fit.row_identity(r) for r in rows], [fit.row_identity(r) for r in request["evaluation_rows"]], "ORIGINAL_ROWS")
    observations = kernel.lines(output / "observations.jsonl")
    require(len(observations) == expected_count, "C1_POSTERIOR_ORIGIN_CONDITION_COUNT")
    equal([key(r) for r in observations], [(r["source_id"], r["split"], ms) for r in rows for ms in contract["offsets_ms"]],
        "ORIGINAL_CONDITION_ORDER")
    calls = [r for r in kernel.lines(output / "forward-calls.jsonl") if r.get("kind") == "final" and r["phase"] == "completed"]
    require(len(calls) == expected_count and len({key(c) for c in calls}) == expected_count and
        {key(c) for c in calls} == {key(r) for r in observations}, "C1_POSTERIOR_COMPLETED_FORWARD_BINDINGS")
    vocabulary = training.read_json(output / "vocabulary.json")
    require(fit.digest(vocabulary) == contract["vocabulary_sha256"] and vocabulary["blank_id"] == 0 and
        vocabulary["output_count"] == 2601, "C1_POSTERIOR_VOCABULARY")
    meta = {"output": str(output), "manifest": manifest_pin, "manifest_sha256": manifest_pin["sha256"],
        "files": pins, "audit": audit_pin, **receipts}
    return {"meta": meta, "output": output, "manifest": manifest, "rows": rows,
        "observations": {key(r): r for r in observations}, "calls": {key(c): c for c in calls}, "vocabulary": vocabulary}


def original_reuse(output, audit_path, contract):
    output, manifest_pin, manifest = manifest_metadata(output, contract["legacy_manifest_sha256"])
    names = ("request.json", "summary.json", "status.json", "conditions.jsonl", "analysis-calls.jsonl")
    pins = {name: metadata_file(output, manifest, name) for name in names}
    require(pins["request.json"]["sha256"] == contract["legacy_recipe_sha256"] and
        pins["conditions.jsonl"]["sha256"] == contract["legacy_conditions_sha256"], "C1_POSTERIOR_REUSE_BYTES_CHANGED")
    request, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    receipts = terminal_metadata(output, manifest_pin, pins["request.json"], summary)
    audit_pin, audit = pinned_json(audit_path, contract["legacy_audit_sha256"])
    require(audit["status"] == "verified" and audit["manifest_sha256"] == manifest_pin["sha256"] and
        audit["saved_conditions_verified"] == 264 and audit["new_model_constructions"] == audit["new_model_forwards"] ==
        audit["optimizer_steps"] == 0, "C1_POSTERIOR_REUSE_AUDIT")
    results = [r for r in kernel.lines(output / "conditions.jsonl") if r["origin"] == "context52"]
    records = [r for r in request["records"] if r["origin"] == "context52"]
    require(len(results) == len(records) == 210 and len({key(r) for r in results}) == 210, "C1_POSTERIOR_REUSE_COUNT")
    return {"meta": {"output": str(output), "manifest": manifest_pin, "manifest_sha256": manifest_pin["sha256"],
        "files": pins, "audit": audit_pin, **receipts}, "output": output, "manifest": manifest,
        "results": {key(r): r for r in results}, "records": {(r["row"]["source_id"], r["row"]["split"], r["offset_ms"]): r for r in records}}


def validate_unknown(row, item):
    require(row["ctc_mask"] is False and (row["full_text"] is None or type(row["full_text"]) is str) and row["target_ids"] is None and
        type(row["ctc_weight"]) in (int, float) and row["ctc_weight"] == 0 and
        row["keyword_targets"] == [None, None] and all(m is False for m in row["keyword_mask"]) and
        row["keyword_weight"] == [0, 0] and row["human_gold"] is False and row["independent_test"] is False,
        "C1_POSTERIOR_UNKNOWN_ORIGINAL_LABELS")
    require(item["source_id"] == row["source_id"] and item["split"] == row["split"] and
        item["source_family_id"] == row["source_family_id"] and
        item["label_binding_sha256"] == fit.row_identity(row) and item["ctc_weight"] == 0 and
        item["full_text"] == row["full_text"] and item["keyword_targets"] == [None, None] and
        all(m is False for m in item["keyword_mask"]) and item["keyword_weight"] == [0, 0],
        "C1_POSTERIOR_UNKNOWN_OBSERVATION")
    # Deliberately do not fill legacy omissions of ctc_mask/frames/target_ids.
    require(item.get("ctc_mask", False) is False and item.get("target_ids") is None,
        "C1_POSTERIOR_UNKNOWN_NOT_FILLED")


def base_record(row, observation, pair, origin):
    return {"origin": origin, "comparison_side": "augmentation" if origin == "augmentation" else "baseline",
        "cohort": pair["cohort"], "row": row, "offset_ms": pair["offset_ms"],
        "label_binding_sha256": pair["label_binding_sha256"], "saved_greedy": observation["greedy_token_ids"],
        "saved_keyword_hits": [k["complete_sequence_present"] for k in observation["keywords"]]}


def new_record(row, observation, pair, origin, data, contract):
    shape = kernel.validate_observation(row, observation, origin, contract)
    call = data["calls"][key(observation)]
    require(call["label_binding_sha256"] == fit.row_identity(row) and call["logits_dtype"] == "float32" and
        call["logits_shape"] == [shape["frames"], 1, 2601], "C1_POSTERIOR_FORWARD_TENSOR_BINDING")
    hypothesis = kernel.variants(row["target_ids"], observation["greedy_token_ids"], row["keyword_targets"],
        row["keyword_mask"], contract["keywords"])
    columns = [0, *sorted(set(row["target_ids"]) | set(observation["greedy_token_ids"]))]
    return {**base_record(row, observation, pair, origin), **shape, "analysis_mode": "computed",
        "checkpoint_sha256": observation["checkpoint_sha256"],
        "numeric_artifact": artifact_from_manifest(data["output"], data["manifest"], observation["numeric_artifact"]),
        "feature_sha256": call["feature_sha256"], "logits_sha256": call["logits_sha256"],
        "class_ids": columns, "saved_original_ctc_nll": observation["ctc_nll"], "hypotheses": hypothesis,
        "derived_array_payload_bytes": 8 * (4 * shape["frames"] * len(columns) + len(row["target_ids"]) * shape["frames"] + 2 * shape["frames"])}


def validate_reuse(row, observation, previous, result, contract):
    kernel.validate_observation(row, observation, "baseline-old", contract)
    require(fit.row_identity(previous["row"]) == fit.row_identity(row) == result["label_binding_sha256"] and
        previous["origin"] == result["origin"] == "context52" and
        previous["condition_binding_sha256"] == result["condition_binding_sha256"] ==
        fit.digest({k: v for k, v in previous.items() if k != "condition_binding_sha256"}) and
        result["checkpoint_sha256"] == observation["checkpoint_sha256"] and result["original_weak_weight"] == row["ctc_weight"] and
        key(result) == key(observation), "C1_POSTERIOR_REUSE_SOURCE_LABEL_MODEL")
    equal(previous["saved_greedy"], observation["greedy_token_ids"], "REUSE_GREEDY")
    equal(previous["saved_keyword_hits"], [k["complete_sequence_present"] for k in observation["keywords"]], "REUSE_KEYWORD_HITS")
    equal([p["token_id"] for p in result["original_target_positions"]], row["target_ids"], "REUSE_TARGET_POSITIONS")
    equal(previous["hypotheses"], kernel.variants(row["target_ids"], observation["greedy_token_ids"], row["keyword_targets"],
        row["keyword_mask"], contract["keywords"]), "REUSE_HYPOTHESES")
    require(previous["numeric_artifact"]["sha256"] == observation["numeric_artifact"]["sha256"] and
        previous["numeric_artifact"]["bytes"] == observation["numeric_artifact"]["bytes"], "C1_POSTERIOR_REUSE_NUMERIC_BINDING")


def assemble(inputs):
    require(set(inputs) == {"comparison", "legacy_output", "legacy_audit"}, "C1_POSTERIOR_INPUT_FIELDS")
    contract = plan()
    comparison_pin, comparison = pinned_json(inputs["comparison"], contract["comparison_sha256"])
    require(comparison["status"] == "completed-saved-only" and comparison["new_model_forwards"] == 0 and
        comparison["human_gold"] is False and comparison["independent_test"] is False and
        len(comparison["all_pairs"]) == 402, "C1_POSTERIOR_COMPARISON_SCOPE")
    augmentation = new_origin(comparison["inputs"]["augmentation"], "augmentation", contract)
    baseline = new_origin(comparison["inputs"]["baseline_evaluation"], "baseline-new", contract)
    legacy = original_reuse(inputs["legacy_output"], inputs["legacy_audit"], contract)
    equal(augmentation["vocabulary"], baseline["vocabulary"], "MATCHED_VOCABULARY")
    vocabulary = augmentation["vocabulary"]
    for word, kid in (("你好小窝", "1"), ("小窝小窝", "2")):
        equal([vocabulary["token_to_id"][x] for x in word], contract["keywords"][kid], "KEYWORD_IDS")
    rows = {r["source_id"]: r for r in augmentation["rows"]}
    require(len(rows) == len(augmentation["rows"]) == 134 and
        Counter((r["split"], r["ctc_mask"]) for r in rows.values()) == {("train", True): 96, ("development", True): 34, ("train", False): 4},
        "C1_POSTERIOR_ROSTER")
    for row in baseline["rows"]:
        require(fit.row_identity(row) == fit.row_identity(rows[row["source_id"]]), "C1_POSTERIOR_PAIRED_ROWS")
    for row in rows.values():
        if row["ctc_mask"]:
            projected = fit.fulltext_targets(row["full_text"], vocabulary)
            require(projected["supported"] and projected["target_ids"] == row["target_ids"], "C1_POSTERIOR_OBSERVED_TEXT_PROJECTION")
    pairs = comparison["all_pairs"]
    require(len({key(p) for p in pairs}) == 402 and {key(p) for p in pairs} == set(augmentation["observations"]),
        "C1_POSTERIOR_PAIR_COVERAGE")
    new_ids = {r["source_id"] for r in baseline["rows"]}
    computed, reused, unknown, coverage = [], [], [], []
    for origin, side, selected, data in (("augmentation", "augmentation", pairs, augmentation),
            ("baseline-new", "baseline", [p for p in pairs if p["source_id"] in new_ids], baseline),
            ("baseline-old", "baseline", [p for p in pairs if p["source_id"] not in new_ids], None)):
        for pair in selected:
            row, item = rows[pair["source_id"]], pair[side]
            require(pair["cohort"] == ("new-" if row["source_id"] in new_ids else "old-") + row["split"] and
                pair["label_binding_sha256"] == fit.row_identity(row) and key(pair) == key(item), "C1_POSTERIOR_PAIR_BINDING")
            if data is not None:
                equal(item, data["observations"][key(item)], "COMPARISON_ORIGINAL_OBSERVATION")
            if row["ctc_mask"] is False:
                validate_unknown(row, item)
                record = {**base_record(row, item, pair, origin), "analysis_mode": "unscored-target-free",
                    "numeric_artifact": artifact_from_manifest(data["output"], data["manifest"], item["numeric_artifact"]),
                    "saved_observation": item}
                unknown.append(record)
            elif data is not None:
                record = new_record(row, item, pair, origin, data, contract)
                computed.append(record)
            else:
                prior, result = legacy["records"][key(item)], legacy["results"][key(item)]
                validate_reuse(row, item, prior, result, contract)
                record = {**base_record(row, item, pair, origin), "analysis_mode": "reused",
                    "checkpoint_sha256": item["checkpoint_sha256"], "numeric_artifact": prior["numeric_artifact"],
                    "legacy_result": result, "legacy_result_sha256": fit.digest(result),
                    "legacy_derived_artifact": artifact_from_manifest(legacy["output"], legacy["manifest"], result["derived_artifact"])}
                reused.append(record)
            coverage.append({"comparison_side": side, "cohort": pair["cohort"], "source_id": row["source_id"],
                "split": row["split"], "offset_ms": pair["offset_ms"], "analysis_mode": record["analysis_mode"],
                "label_binding_sha256": pair["label_binding_sha256"], "ctc_mask": row["ctc_mask"],
                "ctc_weight": row["ctc_weight"], "keyword_targets": row["keyword_targets"],
                "keyword_mask": row["keyword_mask"], "keyword_weight": row["keyword_weight"]})
    for ordinal, record in enumerate(computed + reused + unknown):
        record["ordinal"] = ordinal
        record["condition_binding_sha256"] = fit.digest(record)
    workload = {"new_numeric_conditions": len(computed), "reused_conditions": len(reused), "unscored_conditions": len(unknown),
        "paired_model_conditions": len(coverage), "new": kernel.workload(computed)}
    equal(workload, contract["workload"], "FIXED_WORKLOAD")
    require(len({(r["comparison_side"], r["source_id"], r["split"], r["offset_ms"]) for r in coverage}) == 804,
        "C1_POSTERIOR_EACH_MODEL_CONDITION_ONCE")
    runtime = training.research_runtime.load_manifest()
    training.research_runtime.check_interpreter(runtime)
    return {"schema": SCHEMA, "contract": contract, "inputs": inputs,
        "evidence": {"comparison": comparison_pin, "augmentation": augmentation["meta"],
            "baseline_evaluation": baseline["meta"], "legacy": legacy["meta"]},
        "records": computed, "reused_records": reused, "unknown_records": unknown, "coverage": coverage,
        "workload": workload, "source_hashes": sources(contract), "runtime": runtime}


def validate_recipe(job):
    require(type(job) is dict and job.get("schema") == SCHEMA, "C1_POSTERIOR_RECIPE_SCHEMA")
    equal(job.get("contract"), plan(), "CONTRACT_CHANGED")
    equal(job, assemble(job["inputs"]), "RECIPE_OR_SOURCE_CHANGED")
    return job


def wrap(record, result, artifact):
    return {"record_ordinal": record["ordinal"], "condition_binding_sha256": record["condition_binding_sha256"],
        "analysis_mode": record["analysis_mode"], "comparison_side": record["comparison_side"], "cohort": record["cohort"],
        "source_id": record["row"]["source_id"], "split": record["row"]["split"], "offset_ms": record["offset_ms"],
        "label_binding_sha256": record["label_binding_sha256"], "saved_keyword_hits": record["saved_keyword_hits"],
        "keyword_targets": record["row"]["keyword_targets"], "keyword_mask": record["row"]["keyword_mask"],
        "keyword_weight": record["row"]["keyword_weight"], "ctc_weight": record["row"]["ctc_weight"],
        "derived_artifact": artifact, "derived_path_basis": "current-output" if record["analysis_mode"] == "computed" else "absolute-legacy-output",
        "result": result}


def aggregate(results, coverage):
    """Whitelist aggregates only: no row IDs, weak transcripts, or local paths."""
    conditions = kernel.aggregate([{**r["result"], "origin": r["comparison_side"] + "/" + r["cohort"]} for r in results])
    hit_groups = []
    for kid in (1, 2):
        for hit in (False, True):
            selected = [r for r in results if r["keyword_mask"][kid-1] and r["keyword_targets"][kid-1] == 1 and r["saved_keyword_hits"][kid-1] == hit]
            for item in kernel.aggregate([{**r["result"], "origin": r["comparison_side"] + "/" + r["cohort"]} for r in selected]):
                hit_groups.append({**item, "group_keyword_id": kid, "saved_greedy_keyword_hit": hit,
                    "keyword_positions": [p for p in item["keyword_positions"] if p["keyword_id"] == kid]})
    groups = defaultdict(list)
    for r in results:
        groups[(r["comparison_side"], r["cohort"], r["offset_ms"])].append(r)
    controls = []
    for (side, cohort, offset), members in sorted(groups.items()):
        for kid in (1, 2):
            index = kid - 1
            positives = [r for r in members if r["keyword_mask"][index] and r["keyword_targets"][index] == 1]
            negatives = [r for r in members if r["keyword_mask"][index] and r["keyword_targets"][index] == 0]
            controls.append({"comparison_side": side, "cohort": cohort, "offset_ms": offset, "keyword_id": kid,
                "known_positive_rows": len(positives), "positive_greedy_hits": sum(r["saved_keyword_hits"][index] for r in positives),
                "known_negative_rows": len(negatives), "negative_greedy_hits": sum(r["saved_keyword_hits"][index] for r in negatives)})
    by_key = {(r["comparison_side"], *key(r)): r for r in results}
    paired = defaultdict(list)
    for r in results:
        if r["comparison_side"] == "augmentation":
            b = by_key[("baseline", *key(r))]
            equal(r["label_binding_sha256"], b["label_binding_sha256"], "PAIRED_AGGREGATE_LABEL")
            paired[(r["cohort"], r["offset_ms"])].append((b, r))
    paired_groups = []
    for (cohort, offset), pairs in sorted(paired.items()):
        positions = []
        for kid in (1, 2):
            for position in range(1, 5):
                differences, support = [], {metric: [] for metric in METRICS}
                for b, a in pairs:
                    get = lambda r: {(h["keyword_id"], h["occurrence"], h["target_position"]): h for h in r["result"]["logical_hypotheses"] if h["kind"] == "single-keyword-token-deletion" and h["keyword_id"] == kid and h["keyword_position"] == position}
                    bh, ah = get(b), get(a)
                    equal(sorted(bh), sorted(ah), "PAIRED_MENTIONS")
                    for mention, h in ah.items():
                        differences.append(h["original_minus_candidate_log_probability"] - bh[mention]["original_minus_candidate_log_probability"])
                        target_position = h["target_position"]
                        for metric in METRICS:
                            support[metric].append(a["result"]["original_target_positions"][target_position][metric] - b["result"]["original_target_positions"][target_position][metric])
                positions.append({"keyword_id": kid, "keyword_position": position, "reference_mentions": len(differences),
                    "augmentation_minus_baseline_deletion_delta": kernel.scalar_stats(differences),
                    **{k: kernel.scalar_stats(v) for k, v in support.items()}})
        paired_groups.append({"cohort": cohort, "offset_ms": offset, "paired_rows": len(pairs), "keyword_positions": positions})
    counts = Counter((r["comparison_side"], r["cohort"], r["offset_ms"], r["analysis_mode"]) for r in coverage)
    return {"conditions": conditions, "keyword_hit_groups": hit_groups, "keyword_controls": controls, "paired_changes": paired_groups,
        "coverage_counts": [{"comparison_side": s, "cohort": c, "offset_ms": o, "analysis_mode": m, "conditions": n}
            for (s, c, o, m), n in sorted(counts.items())]}


@contextmanager
def deadline(seconds):
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "C1_POSTERIOR_EXISTING_TIMER")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError("C1_POSTERIOR_FIXED_600_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def limits(contract):
    return kernel.computation_limits(contract)


def expected_counts(contract):
    return {**limits(contract), "conditions_started": 570, "conditions_completed": 570,
        "conditions_reused": 210, "conditions_unscored": 24}


def check_budget(contract, end):
    require(time.monotonic() < end and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= contract["max_rss_kib"],
        "C1_POSTERIOR_TIME_OR_MEMORY_LIMIT")


def run(recipe_path, output, lock):
    contract = plan()
    require(str(io.safe_path(lock)) == contract["heavy_lock"], "C1_POSTERIOR_SHARED_LOCK")
    start, counts = time.monotonic(), Counter()
    output = training.output_directory(output)
    with deadline(contract["worker_seconds"]) as end, training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        io.write_json(output / "analysis-claim.json", {"schema": SCHEMA, "recipe_sha256": training.sha_file(recipe_path),
            "started_at": training.utc(), **BOUNDARIES})
        try:
            job = validate_recipe(training.read_json(recipe_path))
            training.copy_once(recipe_path, output / "request.json")
            for name in CODE:
                training.copy_once(ROOT / name, output / "execution-source" / name)
            torch = kernel.numerical_runtime(contract)
            maxima = limits(contract)
            def progress(name):
                check_budget(contract, end)
                require(name in maxima and counts[name] < maxima[name], "C1_POSTERIOR_NUMERIC_CALL_LIMIT")
                counts[name] += 1
            results = []
            with kernel.no_model_or_optimizer(torch), torch.inference_mode():
                for record in job["records"]:
                    check_budget(contract, end)
                    claim = {"record_ordinal": record["ordinal"], "condition_binding_sha256": record["condition_binding_sha256"]}
                    training.append_json(output / "analysis-calls.jsonl", {**claim, "phase": "started", "monotonic_ns": time.monotonic_ns()})
                    counts["conditions_started"] += 1
                    kernel.check_pin(record["numeric_artifact"])
                    saved = torch.load(record["numeric_artifact"]["path"], weights_only=True, map_location="cpu")
                    kernel.verify_numeric(saved, record, torch)
                    arrays, result = kernel.numerical_condition(saved, record, contract, torch, progress)
                    name = "derived/%03d.pt" % record["ordinal"]
                    artifact = {"path": name, **kernel.save_arrays(output / name, arrays, record, torch)}
                    wrapped = wrap(record, result, artifact)
                    training.append_json(output / "conditions.jsonl", wrapped)
                    training.append_json(output / "analysis-calls.jsonl", {**claim, "phase": "completed", "monotonic_ns": time.monotonic_ns(),
                        "result_sha256": fit.digest(wrapped), "derived_artifact": artifact})
                    counts["conditions_completed"] += 1
                    results.append(wrapped)
                    del saved, arrays
                for record in job["reused_records"]:
                    check_budget(contract, end)
                    kernel.check_pin(record["legacy_derived_artifact"])
                    wrapped = wrap(record, record["legacy_result"], record["legacy_derived_artifact"])
                    training.append_json(output / "conditions.jsonl", wrapped)
                    training.append_json(output / "analysis-calls.jsonl", {"record_ordinal": record["ordinal"], "phase": "reused",
                        "condition_binding_sha256": record["condition_binding_sha256"], "result_sha256": fit.digest(wrapped)})
                    counts["conditions_reused"] += 1
                    results.append(wrapped)
                for record in job["unknown_records"]:
                    training.append_json(output / "analysis-calls.jsonl", {"record_ordinal": record["ordinal"], "phase": "unscored-target-free",
                        "condition_binding_sha256": record["condition_binding_sha256"]})
                    counts["conditions_unscored"] += 1
            equal(dict(counts), expected_counts(contract), "COMPLETION_COUNTS")
            io.write_json(output / "coverage.json", job["coverage"])
            public = {"schema": SCHEMA, "status": "completed", "workload": job["workload"], **BOUNDARIES,
                "primary_offset_ms": 0, "aggregates": aggregate(results, job["coverage"])}
            io.write_json(output / "public-summary.json", public)
            summary = {**public, "recipe_sha256": training.sha_file(recipe_path), "source_hashes": job["source_hashes"],
                "inputs": job["evidence"], "actual_counts": dict(counts), "tensor_bodies_read": 570,
                "reused_tensor_bodies_read": 0, "reused_derived_byte_hash_checks": 210,
                "derived_fullprecision_arrays_retained": True, "wall_seconds": time.monotonic() - start,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            check_budget(contract, end)
            io.write_json(output / "summary.json", summary)
            io.write_json(output / "status.json", summary)
            io.write_json(output / "MANIFEST.json", kernel.make_manifest(output))
            return {k: v for k, v in summary.items() if k not in ("aggregates", "source_hashes", "inputs")}
        except BaseException as exc:
            failed = {"schema": SCHEMA, "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "actual_counts": dict(counts), **BOUNDARIES}
            io.write_json(output / "failure.json", failed)
            if not (output / "status.json").exists():
                io.write_json(output / "status.json", failed)
            raise


def queue_definition(recipe, python):
    job = training.read_json(recipe)
    equal(job["contract"], plan(), "QUEUE_CONTRACT")
    return {"schema": io.SCHEMA, "queue_id": JOB_ID, "global_budget_minutes": 10, "jobs": [{"job_id": JOB_ID,
        "recipe": str(io.safe_path(recipe)), "recipe_sha256": training.sha_file(recipe),
        "argv": [python, "-I", "-B", str(ROOT / SOURCE), "run", "--recipe", "{recipe}", "--output", "{output_dir}",
            "--heavy-lock", job["contract"]["heavy_lock"]], "cwd": str(ROOT), "depends_on": [],
        "timeout_minutes": 10, "outputs": list(OUTPUTS)}]}


def completed_output(output, expected):
    kernel.onset.verify_manifest(output, expected)
    summary = training.read_json(output / "summary.json")
    manifest_pin, recipe_pin = pin(output / "MANIFEST.json"), pin(output / "request.json")
    terminal_metadata(output, manifest_pin, recipe_pin, summary)
    require(summary["schema"] == SCHEMA and summary["status"] == "completed" and
        not (output / "failure.json").exists(), "C1_POSTERIOR_SUCCESS_REQUIRED")
    return summary


def audit(output, expected):
    """Recompute only the 570 new conditions; never run the previous 264 job."""
    contract, start, counts = plan(), time.monotonic(), Counter()
    output = io.safe_path(output)
    with deadline(contract["audit_seconds"]) as end, training.heavy_model_slot(contract["heavy_lock"]):
        summary = completed_output(output, expected)
        job = validate_recipe(training.read_json(output / "request.json"))
        equal({k: summary[k] for k in BOUNDARIES}, BOUNDARIES, "SAVED_BOUNDARIES")
        equal(summary["inputs"], job["evidence"], "SAVED_INPUTS")
        equal(summary["workload"], job["workload"], "SAVED_WORKLOAD")
        equal(summary["actual_counts"], expected_counts(contract), "SAVED_COUNTS")
        equal(summary["source_hashes"], job["source_hashes"], "SAVED_SOURCE_BINDING")
        equal(job["source_hashes"], {n: training.sha_file(output / "execution-source" / n) for n in CODE}, "EXECUTION_SOURCE_BYTES")
        require(type(summary["wall_seconds"]) in (int, float) and 0 <= summary["wall_seconds"] <= contract["worker_seconds"] and
            type(summary["peak_rss_kib"]) is int and 0 < summary["peak_rss_kib"] <= contract["max_rss_kib"] and
            summary["tensor_bodies_read"] == 570 and summary["reused_tensor_bodies_read"] == 0 and
            summary["reused_derived_byte_hash_checks"] == 210, "C1_POSTERIOR_SAVED_BUDGET")
        claim = training.read_json(output / "analysis-claim.json")
        require(claim["schema"] == SCHEMA and claim["recipe_sha256"] == summary["recipe_sha256"], "C1_POSTERIOR_ANALYSIS_CLAIM")
        equal({k: claim[k] for k in BOUNDARIES}, BOUNDARIES, "CLAIM_BOUNDARIES")
        recorded, ledger = kernel.lines(output / "conditions.jsonl"), kernel.lines(output / "analysis-calls.jsonl")
        require(len(recorded) == 780 and len(ledger) == 1374, "C1_POSTERIOR_SAVED_LEDGER_COUNTS")
        equal(training.read_json(output / "coverage.json"), job["coverage"], "SAVED_UNKNOWN_COVERAGE")
        torch = kernel.numerical_runtime(contract)
        maxima = limits(contract)
        def progress(name):
            check_budget(contract, end)
            require(name in maxima and counts[name] < maxima[name], "C1_POSTERIOR_AUDIT_CALL_LIMIT")
            counts[name] += 1
        with kernel.no_model_or_optimizer(torch), torch.inference_mode():
            for index, record in enumerate(job["records"]):
                check_budget(contract, end)
                counts["conditions_started"] += 1
                kernel.check_pin(record["numeric_artifact"])
                saved = torch.load(record["numeric_artifact"]["path"], weights_only=True, map_location="cpu")
                kernel.verify_numeric(saved, record, torch)
                arrays, result = kernel.numerical_condition(saved, record, contract, torch, progress)
                name = "derived/%03d.pt" % record["ordinal"]
                artifact = {"path": name, **io.file_record(output / name)}
                retained = torch.load(output / name, weights_only=True, map_location="cpu")
                require(set(retained) == {"schema", "condition_binding_sha256", "original_numeric_sha256", "class_ids", "arrays", "array_hashes"} and
                    retained["schema"] == kernel.NUMERIC_SCHEMA and retained["condition_binding_sha256"] == record["condition_binding_sha256"] and
                    retained["original_numeric_sha256"] == record["numeric_artifact"]["sha256"] and
                    retained["class_ids"] == record["class_ids"] and set(retained["arrays"]) == set(kernel.ARRAY_KEYS),
                    "C1_POSTERIOR_DERIVED_SCHEMA")
                hashes = {k: training.tensor_sha(v) for k, v in arrays.items()}
                equal(retained["array_hashes"], hashes, "DERIVED_HASH_RECORD")
                equal({k: training.tensor_sha(v) for k, v in retained["arrays"].items()}, hashes, "DERIVED_ARRAY_VALUES")
                wrapped = wrap(record, result, artifact)
                equal(wrapped, recorded[index], "NUMERICAL_RESULT")
                before, after = ledger[2 * index:2 * index + 2]
                require(before["phase"] == "started" and after["phase"] == "completed" and
                    before["record_ordinal"] == after["record_ordinal"] == record["ordinal"] and
                    before["condition_binding_sha256"] == after["condition_binding_sha256"] == record["condition_binding_sha256"] and
                    before["monotonic_ns"] <= after["monotonic_ns"] and after["derived_artifact"] == artifact and
                    after["result_sha256"] == fit.digest(wrapped), "C1_POSTERIOR_CALL_PAIR")
                counts["conditions_completed"] += 1
                del saved, arrays, retained
            for index, record in enumerate(job["reused_records"]):
                check_budget(contract, end)
                kernel.check_pin(record["legacy_derived_artifact"])
                wrapped = wrap(record, record["legacy_result"], record["legacy_derived_artifact"])
                equal(wrapped, recorded[570 + index], "REUSED_RESULT_BYTES")
                equal(ledger[1140 + index], {"record_ordinal": record["ordinal"], "phase": "reused",
                    "condition_binding_sha256": record["condition_binding_sha256"], "result_sha256": fit.digest(wrapped)}, "REUSE_LEDGER")
                counts["conditions_reused"] += 1
            for index, record in enumerate(job["unknown_records"]):
                equal(ledger[1350 + index], {"record_ordinal": record["ordinal"], "phase": "unscored-target-free",
                    "condition_binding_sha256": record["condition_binding_sha256"]}, "UNKNOWN_LEDGER")
                counts["conditions_unscored"] += 1
        equal(dict(counts), expected_counts(contract), "AUDIT_COUNTS")
        public = {"schema": SCHEMA, "status": "completed", "workload": job["workload"], **BOUNDARIES,
            "primary_offset_ms": 0, "aggregates": aggregate(recorded, job["coverage"])}
        equal(public, training.read_json(output / "public-summary.json"), "PUBLIC_AGGREGATE")
        equal(summary["aggregates"], public["aggregates"], "PRIVATE_AGGREGATE")
        check_budget(contract, end)
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected,
            "recipe_sha256": summary["recipe_sha256"], "inputs": job["evidence"], "actual_counts": dict(counts),
            "new_saved_conditions_numerically_verified": 570, "reused_conditions_source_verified": 210,
            "target_free_conditions_verified_without_targets": 24, "original_tensor_bodies_read": 570,
            "new_derived_tensor_bodies_read": 570, "legacy_tensor_bodies_read": 0,
            "wall_seconds": time.monotonic() - start, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            **BOUNDARIES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    specs = (("declare", ("comparison", "legacy-output", "legacy-audit", "output")),
        ("queue-request", ("recipe", "python", "output")), ("run", ("recipe", "output", "heavy-lock")),
        ("audit", ("output", "manifest-sha256", "receipt")))
    for name, options in specs:
        command = sub.add_parser(name)
        for option in options:
            command.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        inputs = {k: str(io.safe_path(getattr(args, k))) for k in ("comparison", "legacy_output", "legacy_audit")}
        job = assemble(inputs)
        io.write_json(Path(args.output), job)
        result = {"schema": SCHEMA, "status": "declared", "recipe": pin(args.output), "workload": job["workload"],
            "torch_imported": "torch" in sys.modules, "tensor_bodies_read": 0, **BOUNDARIES}
    elif args.command == "queue-request":
        io.write_json(Path(args.output), queue_definition(args.recipe, args.python))
        result = {"queue_request": pin(args.output), **BOUNDARIES}
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        require(not Path(args.receipt).exists() and not Path(args.receipt).is_symlink(), "C1_POSTERIOR_AUDIT_RECEIPT_EXISTS")
        result = audit(args.output, args.manifest_sha256)
        io.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
