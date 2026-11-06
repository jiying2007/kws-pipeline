#!/usr/bin/env python3
"""Fixed saved-tensor fulltext CTC analysis; no model or optimizer execution.

Declaration uses only stdlib metadata and immutable source-byte verification.
The numerical worker reuses existing CTC FB/Viterbi and temporal-weight helpers.
Deletion alternatives are pairwise hypotheses, never replacements for labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_context52_known_silence as origin

training, fit, evaluation = origin.training, origin.fit, origin.evaluation
io, require, onset = origin.io, origin.require, origin.onset
SOURCE = "training/research_saved_ctc_diagnostic.py"
PLAN = "configs/research/saved-ctc-diagnostic-20261106.json"
PLAN_SHA = "6ab71bb187ba1c6e96ae1c2c381f48c53ee1c5d662170e0940f7a0c44185018b"
TEST = "tests/test_research_saved_ctc_diagnostic.py"
DOC = "docs/research-saved-ctc-diagnostic.md"
EXTRA_CODE = ("tools/diagnose_keyword_ctc_sequence_competition.py",
    "tools/diagnose_decoder_search_path_decomposition.py", "tools/diagnose_sequence_margin_runtime_gap.py",
    "training/sequence_margin.py", "training/objective_contract.py")
CODE = tuple(dict.fromkeys((*origin.CODE, *EXTRA_CODE, SOURCE, PLAN, TEST, DOC)))
SCHEMA = "kws-saved-ctc-diagnostic-job-v1"
NUMERIC_SCHEMA = "kws-saved-ctc-diagnostic-numeric-v1"
JOB_ID = "saved-ctc-context52-corrected264"
OUTPUTS = ("summary.json", "status.json", "conditions.jsonl", "analysis-calls.jsonl", "MANIFEST.json")
KINDS = ("context52", "known-silence-corrected")
ARRAY_KEYS = ("class_log_probs", "temporal_weights", "rank_all", "rank_nonblank",
              "other_nonblank_max_log_probability", "argmax", "full_log_normalizer")


def plan():
    require(training.sha_file(ROOT / PLAN) == PLAN_SHA, "SAVED_CTC_FIXED_PLAN_CHANGED")
    return training.read_json(ROOT / PLAN)


def sources():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def equal(left, right, reason):
    require(io.canonical(left) == io.canonical(right), reason)


def pin(path):
    path = io.safe_path(path)
    return {"path": str(path), **io.file_record(path)}


def check_pin(value):
    equal(pin(value["path"]), value, "SAVED_CTC_PIN_CHANGED")


def lines(path):
    return [io.loads(raw) for raw in io.read_bytes(path, 64 * 1024 * 1024).splitlines()]


def variants(target, greedy, keyword_targets, keyword_mask, keywords):
    """All original keyword positions; exact tuple dedup keeps logical provenance."""
    require(type(target) is list and target and all(type(x) is int and x > 0 for x in target),
        "SAVED_CTC_NONEMPTY_ORIGINAL_TARGET")
    require(type(greedy) is list and all(type(x) is int and x > 0 for x in greedy), "SAVED_CTC_GREEDY_TARGET")
    require(len(keyword_targets) == len(keyword_mask) == 2 and all(type(m) is bool for m in keyword_mask),
        "SAVED_CTC_KNOWN_MASKS")
    unique, logical, index = [], [], {}
    def add(kind, sequence, **metadata):
        sequence = tuple(sequence)
        if sequence not in index:
            index[sequence] = len(unique)
            unique.append(list(sequence))
        logical.append({"kind": kind, "hypothesis_index": index[sequence], **metadata})
    add("unchanged-fulltext", target)
    add("saved-greedy", greedy)
    mentions = []
    for kid in (1, 2):
        word = keywords[str(kid)]
        starts = [i for i in range(len(target) - len(word) + 1) if target[i:i + len(word)] == word]
        positive = keyword_mask[kid - 1] and keyword_targets[kid - 1] == 1
        require(bool(starts) == bool(positive), "SAVED_CTC_KEYWORD_LABEL_OR_OCCURRENCE")
        require(keyword_mask[kid - 1] or keyword_targets[kid - 1] is None, "SAVED_CTC_UNKNOWN_NOT_NEGATIVE")
        for occurrence, start in enumerate(starts, 1):
            mention = {"keyword_id": kid, "occurrence": occurrence, "reference_start": start,
                "reference_stop": start + len(word)}
            mentions.append(mention)
            for offset in range(len(word)):
                position = start + offset
                add("single-keyword-token-deletion", target[:position] + target[position + 1:],
                    **mention, target_position=position, keyword_position=offset + 1, deleted_token_id=target[position])
    return {"unique_targets": unique, "logical_hypotheses": logical, "keyword_mentions": mentions}


def validate_observation(row, item, kind, contract):
    require(row["ctc_mask"] is True and type(row["ctc_weight"]) in (int, float) and
        math.isfinite(row["ctc_weight"]) and 0 < row["ctc_weight"] <= 1 and row["full_text"] and
        row["split"] in ("train", "development") and row["human_gold"] is False and row["independent_test"] is False,
        "SAVED_CTC_ORIGINAL_WEAK_ROW")
    for key in ("source_id", "source_family_id", "speaker_id", "split", "source_role", "ctc_mask", "ctc_weight",
                "keyword_targets", "keyword_mask", "keyword_weight", "target_ids"):
        equal(item[key], row[key], "SAVED_CTC_ROW_FIELD:" + key)
    require(item["target_text"] == row["full_text"] and item["label_binding_sha256"] == fit.row_identity(row) and
        item["original_pcm_sha256"] == row["pcm_sha256"] and item["checkpoint_sha256"] == contract["checkpoint_sha256"][kind],
        "SAVED_CTC_LABEL_PCM_OR_MODEL_CHANGED")
    require(type(item["offset_ms"]) is int and item["offset_ms"] in contract["offsets_ms"] and
        type(item["frames"]) is int and 0 < item["frames"] <= contract["max_frames"], "SAVED_CTC_FRAME_SCOPE")
    shape = onset.geometry(row["samples"], item["offset_ms"])
    require(item["frames"] == shape["frames"] and item["known_zero_prefix_frames"] == shape["fully_digital_silence_frames"],
        "SAVED_CTC_ORIGINAL_GEOMETRY")
    return shape


def original_inputs(kind, output, audit_path, contract):
    output = io.safe_path(output)
    expected = contract["manifest_sha256"][kind]
    manifest = onset.verify_manifest(output, expected)  # opaque byte hashes only
    request = training.read_json(output / "request.json")
    summary = training.read_json(output / "summary.json")
    audit = training.read_json(audit_path)
    result = training.read_json(output.parent / "result.json")
    queue = output.parents[2] / "queue.json"
    require(training.sha_file(output / "request.json") == contract["recipe_sha256"][kind] and
        summary["status"] == "completed" and result["status"] == "completed" and result["exit_code"] == 0 and
        result["recipe_sha256"] == contract["recipe_sha256"][kind] and result["queue_sha256"] == training.sha_file(queue) and
        result["outputs"]["MANIFEST.json"] == io.file_record(output / "MANIFEST.json") and
        result["claim_sha256"] == training.sha_file(output.parent / "claim.json") and
        audit["status"] == "verified" and audit["manifest_sha256"] == expected and
        audit["new_model_constructions"] == audit["new_model_forwards"] == audit["new_optimizer_steps"] == 0,
        "SAVED_CTC_COMPLETED_AUTHENTIC_ORIGIN_REQUIRED")
    equal(summary, training.read_json(output / "status.json"), "SAVED_CTC_ORIGIN_STATUS_DIVERGED")
    driver = Path(audit_path).parent / "driver/audit/receipt.json"
    receipt = training.read_json(driver)
    require(receipt["returncode"] == 0, "SAVED_CTC_ORIGIN_AUDIT_COMMAND_FAILED")
    for name in ("command", "stdout", "stderr"):
        check_pin(receipt[name])
    equal(training.read_json(receipt["stdout"]["path"]), audit, "SAVED_CTC_ORIGIN_AUDIT_STDOUT")
    rows = request["train"] + request["development"] if kind == "context52" else request["development"]
    if kind == "context52":
        equal(training.read_json(output / "train-rows.json"), request["train"], "SAVED_CTC_TRAIN_ROSTER")
        require(audit["saved_training_forwards"] == 8000 and audit["saved_optimizer_steps"] == 2000 and
            audit["saved_final_conditions"] == 210, "SAVED_CTC_CURRENT_AUDIT_SCOPE")
    else:
        require(audit["saved_conditions_verified"] == 108, "SAVED_CTC_PAIRED_AUDIT_SCOPE")
    equal(training.read_json(output / "development-rows.json"), request["development"], "SAVED_CTC_DEV_ROSTER")
    observations = lines(output / "observations.jsonl")
    selected = observations if kind == "context52" else [x for x in observations if x["model_id"] == kind]
    require(len(observations) == (210 if kind == "context52" else 108) and
        len(selected) == contract["conditions_by_origin"][kind], "SAVED_CTC_ORIGINAL_CONDITION_COUNT")
    expected_model = origin.JOB_ID if kind == "context52" else kind
    require(all(x["model_id"] == expected_model for x in selected), "SAVED_CTC_MODEL_EXCLUSION")
    expected_order = [(r["source_id"], ms) for r in rows for ms in contract["offsets_ms"]]
    equal([(x["source_id"], x["offset_ms"]) for x in selected], expected_order, "SAVED_CTC_ORIGINAL_ORDER")
    calls = [c for c in lines(output / "forward-calls.jsonl") if c["phase"] == "completed" and
        (c.get("kind") == "final" if kind == "context52" else c.get("model_id") == kind)]
    require(len(calls) == len(selected), "SAVED_CTC_ORIGINAL_COMPLETED_CALLS")
    by_call = {(c["source_id"], c["offset_ms"]): c for c in calls}
    vocabulary = training.read_json(output / "vocabulary.json")
    require(fit.digest(vocabulary) == contract["vocabulary_sha256"] and vocabulary["blank_id"] == 0 and
        vocabulary["output_count"] == contract["vocabulary_outputs"], "SAVED_CTC_VOCABULARY")
    for text, kid in (("你好小窝", "1"), ("小窝小窝", "2")):
        equal([vocabulary["token_to_id"][c] for c in text], contract["keywords"][kid], "SAVED_CTC_KEYWORD_IDS")
    records = []
    for row in rows:
        projection = fit.fulltext_targets(row["full_text"], vocabulary)
        require(projection["supported"] and projection["target_ids"] == row["target_ids"], "SAVED_CTC_FULLTEXT_PROJECTION")
    by_source = {r["source_id"]: r for r in rows}
    for item in selected:
        row = by_source[item["source_id"]]
        shape = validate_observation(row, item, kind, contract)
        call = by_call[row["source_id"], item["offset_ms"]]
        require(call["label_binding_sha256"] == fit.row_identity(row), "SAVED_CTC_CALL_LABEL_BINDING")
        artifact = item["numeric_artifact"]
        relative = io.relative(artifact["path"])
        equal({k: artifact[k] for k in ("bytes", "sha256")}, manifest[str(relative)], "SAVED_CTC_NUMERIC_MANIFEST")
        hypothesis = variants(row["target_ids"], item["greedy_token_ids"], row["keyword_targets"], row["keyword_mask"], contract["keywords"])
        columns = [0, *sorted(set(row["target_ids"]) | set(item["greedy_token_ids"]))]
        length, frames = len(row["target_ids"]), item["frames"]
        records.append({"origin": kind, "model_id": expected_model, "checkpoint_sha256": item["checkpoint_sha256"],
            "row": row, "label_binding_sha256": fit.row_identity(row), "offset_ms": item["offset_ms"], **shape,
            "numeric_artifact": {"path": str(output / relative), **manifest[str(relative)]},
            "feature_sha256": call["feature_sha256"], "logits_sha256": call["logits_sha256"],
            "saved_greedy": item["greedy_token_ids"], "saved_original_ctc_nll": item["ctc_nll"],
            "saved_keyword_hits": [x["complete_sequence_present"] for x in item["keywords"]],
            "class_ids": columns, "hypotheses": hypothesis,
            "derived_array_payload_bytes": 8 * (4 * frames * len(columns) + length * frames + 2 * frames)})
    meta = {"output": str(output), "manifest": pin(output / "MANIFEST.json"), "recipe": pin(output / "request.json"),
        "summary": pin(output / "summary.json"), "queue_result": pin(output.parent / "result.json"), "queue": pin(queue),
        "audit": pin(audit_path), "audit_driver_receipt": pin(driver), "vocabulary": pin(output / "vocabulary.json"),
        "observations": pin(output / "observations.jsonl")}
    return meta, records, vocabulary


def workload(records):
    count = Counter()
    for r in records:
        hyp = r["hypotheses"]
        count.update({"saved_conditions": 1, "saved_serialized_numeric_bytes": r["numeric_artifact"]["bytes"],
            "saved_frames": r["frames"], "original_target_positions": len(r["row"]["target_ids"]),
            "keyword_reference_mentions": len(hyp["keyword_mentions"]),
            "single_token_deletion_contrasts": sum(x["kind"] == "single-keyword-token-deletion" for x in hyp["logical_hypotheses"]),
            "logical_sequence_scores": len(hyp["logical_hypotheses"]), "unique_sequence_scores": len(hyp["unique_targets"]),
            "empty_greedy_sequences": int(not r["saved_greedy"]), "derived_array_payload_bytes": r["derived_array_payload_bytes"]})
    return dict(count)


def validate_workload(value, contract):
    require(set(value) == {"saved_conditions", "saved_serialized_numeric_bytes", "saved_frames", "original_target_positions",
        "keyword_reference_mentions", "single_token_deletion_contrasts", "logical_sequence_scores", "unique_sequence_scores",
        "empty_greedy_sequences", "derived_array_payload_bytes"}, "SAVED_CTC_WORKLOAD_KEYS")
    for key in value:
        if key != "derived_array_payload_bytes":
            equal(value[key], contract[key], "SAVED_CTC_FIXED_WORKLOAD:" + key)
    require(type(value["derived_array_payload_bytes"]) is int and 0 < value["derived_array_payload_bytes"] <= 64 * 1024 * 1024,
        "SAVED_CTC_DERIVED_PAYLOAD_LIMIT")


def assemble(inputs):
    require(set(inputs) == {"context_output", "context_audit", "paired_output", "paired_audit"}, "SAVED_CTC_INPUT_KEYS")
    contract, origins, records, vocabulary = plan(), {}, [], None
    for kind, prefix in (("context52", "context"), ("known-silence-corrected", "paired")):
        meta, selected, vocab = original_inputs(kind, inputs[prefix + "_output"], inputs[prefix + "_audit"], contract)
        origins[kind] = meta
        require(vocabulary is None or vocab == vocabulary, "SAVED_CTC_PAIRED_VOCABULARY")
        vocabulary = vocab
        records.extend(selected)
    current = {(r["row"]["source_id"], r["offset_ms"]): r for r in records if r["origin"] == "context52"}
    for r in records:
        if r["origin"] == "known-silence-corrected":
            other = current[r["row"]["source_id"], r["offset_ms"]]
            equal(r["row"], other["row"], "SAVED_CTC_PAIRED_ORIGINAL_ROW")
            require(r["feature_sha256"] == other["feature_sha256"], "SAVED_CTC_PAIRED_FEATURE_IDENTITY")
    for ordinal, r in enumerate(records):
        r["ordinal"] = ordinal
        r["condition_binding_sha256"] = fit.digest(r)
    counts = workload(records)
    validate_workload(counts, contract)
    runtime = training.research_runtime.load_manifest()
    training.research_runtime.check_interpreter(runtime)
    return {"schema": SCHEMA, "contract": contract, "plan": pin(ROOT / PLAN), "inputs": inputs,
        "origins": origins, "records": records, "workload": counts, "source_hashes": sources(),
        "runtime": runtime, "records_sha256": fit.digest(records)}


def validate_recipe(job):
    require(type(job) is dict and job.get("schema") == SCHEMA, "SAVED_CTC_JOB_SCHEMA")
    equal(job.get("contract"), plan(), "SAVED_CTC_FIXED_CONTRACT_CHANGED")
    equal(job, assemble(job["inputs"]), "SAVED_CTC_RECIPE_OR_SOURCE_CHANGED")
    return job


def close(actual, expected, contract, kind="double_parity"):
    require(type(actual) in (int, float) and type(expected) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and
        math.isclose(actual, expected, abs_tol=contract[kind + "_abs_tolerance"], rel_tol=contract[kind + "_rel_tolerance"]),
        "SAVED_CTC_NUMERICAL_PARITY:" + kind)


@contextmanager
def no_model_or_optimizer(torch):
    """A local guard, not a process sandbox; the worker has no inference route."""
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("SAVED_CTC_MODEL_BACKWARD_OR_OPTIMIZER_FORBIDDEN")
    targets = [(torch.nn.Module, "__init__"), (torch.nn.Module, "_call_impl"),
        (torch.optim.Optimizer, "__init__"), (torch.optim.Optimizer, "step"),
        (torch.Tensor, "backward"), (torch.autograd, "backward"), (torch.autograd, "grad")]
    originals = [(obj, name, getattr(obj, name)) for obj, name in targets]
    try:
        for obj, name, _ in originals:
            setattr(obj, name, forbidden)
        yield
    finally:
        for obj, name, value in originals:
            setattr(obj, name, value)


def numerical_runtime(contract):
    manifest = training.research_runtime.load_manifest()
    training.research_runtime.check_interpreter(manifest)
    training.research_runtime.check_packages(manifest, "core")
    import torch
    torch.set_num_threads(contract["cpu_threads"])
    if torch.get_num_interop_threads() != contract["interop_threads"]:
        torch.set_num_interop_threads(contract["interop_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    return torch


def verify_numeric(saved, record, torch):
    require(set(saved) == {"schema", "source_id", "feature", "logits", "feature_sha256", "logits_sha256"} and
        saved["schema"] == "kws-d1-development-numeric-evidence-v2" and saved["source_id"] == record["row"]["source_id"],
        "SAVED_CTC_ORIGINAL_NUMERIC_SCHEMA")
    for key, shape in (("feature", (record["frames"], 32)), ("logits", (record["frames"], 1, 2601))):
        tensor = saved[key]
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float32 and tuple(tensor.shape) == shape and
            bool(tensor.isfinite().all()) and not tensor.requires_grad, "SAVED_CTC_ORIGINAL_NUMERIC_TYPE")
        require(training.tensor_sha(tensor) == saved[key + "_sha256"] == record[key + "_sha256"],
            "SAVED_CTC_ORIGINAL_TENSOR_CHANGED")


def compute_arrays(logits, target, greedy, contract, torch, progress=lambda _key: None):
    """No new DP: reuse both existing exact CTC helpers on original fulltext."""
    from diagnose_keyword_ctc_sequence_competition import ctc_forward_backward
    from sequence_margin import _ctc_token_state_temporal_weights
    values = logits.detach()[:, 0].to(torch.float64)
    require(bool(values.isfinite().all()), "SAVED_CTC_FINITE_LOGITS")
    argmax = values.argmax(-1)
    require(evaluation.collapse_ctc(argmax.tolist()) == greedy, "SAVED_CTC_SAVED_GREEDY_CHANGED")
    class_ids = [0, *sorted(set(target) | set(greedy))]
    mapping = {token: index for index, token in enumerate(class_ids)}
    full_log_probs = values.log_softmax(-1)
    compact = full_log_probs[:, class_ids].contiguous()  # never renormalize this substochastic subset
    mapped = tuple(mapping[t] for t in target)
    progress("fulltext_fb_started")
    reference_logp, positions, viterbi = ctc_forward_backward(compact.tolist(), mapped, blank=0)
    progress("fulltext_fb_completed")
    progress("temporal_weights_started")
    temporal = _ctc_token_state_temporal_weights(compact, mapped, blank=0).contiguous()
    progress("temporal_weights_completed")
    require(tuple(temporal.shape) == (len(target), len(values)) and temporal.dtype == torch.float64 and
        bool(temporal.isfinite().all()) and bool((temporal >= 0).all()), "SAVED_CTC_TEMPORAL_ARRAY")
    frames = torch.arange(len(values), dtype=torch.float64)
    for i, position in enumerate(positions):
        q = temporal[i]
        close(float(q.sum()), 1.0, contract, "temporal_summary")
        close(float((q * frames).sum()), position["expected_frame"], contract, "temporal_summary")
        positive = q > 0
        close(float(-(q[positive] * q[positive].log()).sum()), position["temporal_entropy_nats"], contract, "temporal_summary")
        require(position["state_occupancy_frames"] >= 1.0 - 1e-8, "SAVED_CTC_OCCUPANCY_IS_CONDITIONAL")
    rank_all, rank_nonblank, other = [], [], []
    for token in class_ids:
        rank_all.append(1 + (values > values[:, token:token + 1]).sum(-1))
        rank_nonblank.append(1 + (values[:, 1:] > values[:, token:token + 1]).sum(-1))
        competitors = full_log_probs[:, 1:].clone()
        if token:
            competitors[:, token - 1] = -math.inf
        other.append(competitors.max(-1).values)
    arrays = {"class_log_probs": compact, "temporal_weights": temporal,
        "rank_all": torch.stack(rank_all, dim=1), "rank_nonblank": torch.stack(rank_nonblank, dim=1),
        "other_nonblank_max_log_probability": torch.stack(other, dim=1), "argmax": argmax,
        "full_log_normalizer": values.logsumexp(-1)}
    require(all(bool(value.isfinite().all()) for value in arrays.values()), "SAVED_CTC_DERIVED_FINITE")
    return arrays, class_ids, reference_logp, positions, viterbi


def sequence_score(log_probs, sequence, torch):
    if not sequence:
        return float(log_probs[:, 0].sum())
    return -float(torch.nn.functional.ctc_loss(log_probs.unsqueeze(1), torch.tensor(sequence, dtype=torch.long),
        torch.tensor([len(log_probs)], dtype=torch.long), torch.tensor([len(sequence)], dtype=torch.long),
        blank=0, reduction="sum", zero_infinity=False))


def numerical_condition(saved, record, contract, torch, progress=lambda _key: None):
    target, greedy = record["row"]["target_ids"], record["saved_greedy"]
    arrays, class_ids, reference, positions, viterbi = compute_arrays(saved["logits"], target, greedy, contract, torch, progress)
    require(class_ids == record["class_ids"], "SAVED_CTC_CLASS_COLUMN_MAP")
    mapping = {token: index for index, token in enumerate(class_ids)}
    scores = []
    for sequence in record["hypotheses"]["unique_targets"]:
        progress("unique_score_started")
        score = sequence_score(arrays["class_log_probs"], [mapping[t] for t in sequence], torch)
        require(math.isfinite(score), "SAVED_CTC_FINITE_SEQUENCE_SCORE")
        scores.append(score)
        progress("unique_score_completed")
    close(reference, scores[0], contract)
    close(-reference, record["saved_original_ctc_nll"], contract, "saved_float32_nll")
    class_prob = arrays["class_log_probs"].exp()
    token_records = []
    for index, token in enumerate(target):
        column, q = mapping[token], arrays["temporal_weights"][index]
        blank = arrays["argmax"] == 0
        target_top1 = arrays["argmax"] == token
        rank = arrays["rank_all"][:, column]
        rank_nb = arrays["rank_nonblank"][:, column]
        margin_blank = arrays["class_log_probs"][:, column] - arrays["class_log_probs"][:, 0]
        margin_other = arrays["class_log_probs"][:, column] - arrays["other_nonblank_max_log_probability"][:, column]
        entry = {"target_position": index, "token_id": token, "class_column": column,
            "conditional_state_occupancy_frames": positions[index]["state_occupancy_frames"],
            "conditional_expected_frame": positions[index]["expected_frame"],
            "conditional_q10_frame": positions[index]["q10_frame"], "conditional_q50_frame": positions[index]["q50_frame"],
            "conditional_q90_frame": positions[index]["q90_frame"], "conditional_width80_frames": positions[index]["width80_frames"],
            "conditional_temporal_entropy_nats": positions[index]["temporal_entropy_nats"],
            "raw_class_probability_global_max": float(class_prob[:, column].max()),
            "raw_best_competition_rank_global": int(rank.min()), "raw_target_argmax_frames": int(target_top1.sum()),
            "raw_maximum_logit_margin_over_blank": float(margin_blank.max()),
            "q_weighted_raw_class_probability": float((q * class_prob[:, column]).sum()),
            "q_weighted_rank_all": float((q * rank).sum()), "q_weighted_rank_nonblank": float((q * rank_nb).sum()),
            "q_mass_target_argmax": float(q[target_top1].sum()), "q_mass_blank_argmax": float(q[blank].sum()),
            "q_mass_other_argmax": float(q[~(target_top1 | blank)].sum()),
            "q_mass_rank1": float(q[rank <= 1].sum()), "q_mass_rank3": float(q[rank <= 3].sum()),
            "q_weighted_logit_margin_over_blank": float((q * margin_blank).sum()),
            "q_weighted_logit_margin_over_other_nonblank": float((q * margin_other).sum()),
            "q_mass_in_fully_added_zero_frames": float(q[:record["fully_digital_silence_frames"]].sum()),
            "viterbi_first_frame": viterbi[index]["first_frame"], "viterbi_last_frame": viterbi[index]["last_frame"],
            "viterbi_occupied_frames": viterbi[index]["frames"],
            "viterbi_blank_gap_before": viterbi[index]["first_frame"] - viterbi[index - 1]["last_frame"] - 1 if index else None,
            "viterbi_blank_gap_after": viterbi[index + 1]["first_frame"] - viterbi[index]["last_frame"] - 1 if index + 1 < len(target) else None,
            "viterbi_original_window_start_seconds": (viterbi[index]["first_frame"] * 320 - record["added_samples"]) / 16000,
            "viterbi_original_window_end_seconds": (viterbi[index]["last_frame"] * 320 + 400 - record["added_samples"]) / 16000}
        first, stop = viterbi[index]["first_frame"], viterbi[index]["last_frame"] + 1
        entry["actual_argmax_blank_frames_on_viterbi_token_interval"] = int(blank[first:stop].sum())
        token_records.append(entry)
    logical = [{**item, "log_probability": scores[item["hypothesis_index"]],
        "original_minus_candidate_log_probability": reference - scores[item["hypothesis_index"]],
        "candidate_target_length": len(record["hypotheses"]["unique_targets"][item["hypothesis_index"]])}
        for item in record["hypotheses"]["logical_hypotheses"]]
    result = {"ordinal": record["ordinal"], "condition_binding_sha256": record["condition_binding_sha256"],
        "origin": record["origin"], "split": record["row"]["split"], "source_id": record["row"]["source_id"],
        "offset_ms": record["offset_ms"], "source_family_id": record["row"]["source_family_id"],
        "label_binding_sha256": record["label_binding_sha256"], "checkpoint_sha256": record["checkpoint_sha256"],
        "original_weak_weight": record["row"]["ctc_weight"], "original_target_length": len(target),
        "reference_log_probability": reference, "unique_log_probabilities": scores, "logical_hypotheses": logical,
        "keyword_mentions": record["hypotheses"]["keyword_mentions"], "original_target_positions": token_records,
        "pairwise_preference_only": True, "conditional_alignment_is_not_acoustic_truth": True,
        "viterbi_is_one_optimal_path": True, "word_timestamps_claimed": False,
        "raw_zero_probability_may_be_float64_underflow": True}
    return arrays, result


def scalar_stats(values):
    return {"count": len(values), "minimum": min(values), "median": statistics.median(values),
        "maximum": max(values), "mean": sum(values) / len(values)} if values else {"count": 0}


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["origin"], record["split"], record["offset_ms"])].append(record)
    output = []
    for (model, split, offset), items in sorted(groups.items()):
        positions = []
        for kid in (1, 2):
            for position in range(1, 5):
                pairs = [(row, hyp) for row in items for hyp in row["logical_hypotheses"]
                    if hyp["kind"] == "single-keyword-token-deletion" and hyp["keyword_id"] == kid and hyp["keyword_position"] == position]
                support = [row["original_target_positions"][hyp["target_position"]] for row, hyp in pairs]
                metric_keys = ("q_weighted_raw_class_probability", "q_weighted_logit_margin_over_blank",
                    "q_weighted_logit_margin_over_other_nonblank", "q_weighted_rank_all", "q_weighted_rank_nonblank",
                    "q_mass_blank_argmax", "q_mass_target_argmax", "q_mass_other_argmax", "q_mass_rank3",
                    "q_mass_in_fully_added_zero_frames", "conditional_width80_frames", "conditional_temporal_entropy_nats")
                positions.append({"keyword_id": kid, "keyword_position": position, "reference_mentions": len(pairs),
                    "source_rows": len({row["source_id"] for row, _ in pairs}),
                    "original_minus_single_deletion_log_probability": scalar_stats([h["original_minus_candidate_log_probability"] for _, h in pairs]),
                    **{key: scalar_stats([s[key] for s in support]) for key in metric_keys}})
        output.append({"origin": model, "split": split, "offset_ms": offset, "observed_rows": len(items),
            "reference_log_probability": scalar_stats([r["reference_log_probability"] for r in items]),
            "original_minus_saved_greedy_log_probability": scalar_stats([r["logical_hypotheses"][1]["original_minus_candidate_log_probability"] for r in items]),
            "keyword_positions": positions})
    return output


@contextmanager
def deadline(seconds):
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "SAVED_CTC_EXISTING_TIMER")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError("SAVED_CTC_FIXED_540_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def save_arrays(path, arrays, record, torch):
    payload = {"schema": NUMERIC_SCHEMA, "condition_binding_sha256": record["condition_binding_sha256"],
        "original_numeric_sha256": record["numeric_artifact"]["sha256"], "class_ids": record["class_ids"],
        "arrays": arrays, "array_hashes": {name: training.tensor_sha(value) for name, value in arrays.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    io.sync_dir(path.parent)
    return io.file_record(path)


def make_manifest(output):
    return {p.relative_to(output).as_posix(): io.file_record(p, durable=True)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}


def computation_limits(contract):
    return {"fulltext_fb_started": contract["fulltext_forward_backward_calls"],
        "fulltext_fb_completed": contract["fulltext_forward_backward_calls"],
        "temporal_weights_started": contract["temporal_weight_calls"],
        "temporal_weights_completed": contract["temporal_weight_calls"],
        "unique_score_started": contract["unique_sequence_scores"],
        "unique_score_completed": contract["unique_sequence_scores"]}


def terminal_metadata(job, summary, recipe_sha):
    equal(summary["actual_counts"], {**computation_limits(job["contract"]),
        "conditions_started": 264, "conditions_completed": 264}, "SAVED_CTC_TERMINAL_COUNTS")
    equal(summary["source_hashes"], job["source_hashes"], "SAVED_CTC_TERMINAL_SOURCE")
    equal(summary["workload"], job["workload"], "SAVED_CTC_TERMINAL_WORKLOAD")
    require(summary["schema"] == SCHEMA and summary["recipe_sha256"] == recipe_sha and
        type(summary["wall_seconds"]) in (int, float) and 0 <= summary["wall_seconds"] <= job["contract"]["worker_seconds"] and
        type(summary["peak_rss_kib"]) is int and 0 < summary["peak_rss_kib"] <= job["contract"]["max_rss_kib"],
        "SAVED_CTC_TERMINAL_BUDGET_OR_RECIPE")
    fixed = {"new_model_constructions": 0, "new_model_forwards": 0, "optimizer_steps": 0, "backward_calls": 0,
        "asr_or_tts_calls": 0, "automatic_retry": False, "human_gold": False, "independent_test": False,
        "previously_exposed_development": True, "quality_winner": None, "generalization_passed": None,
        "pairwise_preferences_only": True, "conditional_alignment_is_not_acoustic_truth": True,
        "original_logits_duplicated": False, "derived_fullprecision_arrays_retained": True}
    equal({key: summary[key] for key in fixed}, fixed, "SAVED_CTC_TERMINAL_BOUNDARIES")


def run(recipe_path, output, lock):
    contract = plan()
    require(str(io.safe_path(lock)) == contract["heavy_lock"], "SAVED_CTC_SHARED_LOCK_PATH")
    start = time.monotonic()
    counts = Counter()
    output = training.output_directory(output)
    with deadline(contract["worker_seconds"]) as end, training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        io.write_json(output / "analysis-claim.json", {"schema": SCHEMA, "recipe_sha256": training.sha_file(recipe_path),
            "started_at": training.utc(), "automatic_retry": False, "new_model_forwards": 0})
        try:
            job = validate_recipe(training.read_json(recipe_path))
            training.copy_once(recipe_path, output / "request.json")
            for name in CODE:
                training.copy_once(ROOT / name, output / "execution-source" / name)
            torch = numerical_runtime(contract)
            limits = computation_limits(contract)
            def progress(key):
                require(key in limits and counts[key] < limits[key] and time.monotonic() < end, "SAVED_CTC_COMPUTATION_LIMIT")
                counts[key] += 1
            results = []
            with no_model_or_optimizer(torch), torch.inference_mode():
                for record in job["records"]:
                    require(time.monotonic() < end and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= contract["max_rss_kib"],
                        "SAVED_CTC_TIME_OR_MEMORY_LIMIT")
                    claim = {"ordinal": record["ordinal"], "condition_binding_sha256": record["condition_binding_sha256"]}
                    training.append_json(output / "analysis-calls.jsonl", {**claim, "phase": "started", "monotonic_ns": time.monotonic_ns()})
                    counts["conditions_started"] += 1
                    check_pin(record["numeric_artifact"])
                    saved = torch.load(record["numeric_artifact"]["path"], weights_only=True, map_location="cpu")
                    verify_numeric(saved, record, torch)
                    arrays, result = numerical_condition(saved, record, contract, torch, progress)
                    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= contract["max_rss_kib"],
                        "SAVED_CTC_MEMORY_LIMIT")
                    path = "derived/%03d.pt" % record["ordinal"]
                    artifact = {"path": path, **save_arrays(output / path, arrays, record, torch)}
                    result["derived_artifact"] = artifact
                    training.append_json(output / "conditions.jsonl", result)
                    training.append_json(output / "analysis-calls.jsonl", {**claim, "phase": "completed", "monotonic_ns": time.monotonic_ns(),
                        "derived_artifact": artifact, "result_sha256": fit.digest(result)})
                    counts["conditions_completed"] += 1
                    results.append(result)
                    del saved, arrays
            equal(dict(counts), {**limits, "conditions_started": 264, "conditions_completed": 264}, "SAVED_CTC_COMPLETION_COUNTS")
            summary = {"schema": SCHEMA, "status": "completed", "recipe_sha256": training.sha_file(recipe_path),
                "source_hashes": job["source_hashes"], "workload": job["workload"], "actual_counts": dict(counts),
                "conditions": aggregate(results), "new_model_constructions": 0, "new_model_forwards": 0,
                "optimizer_steps": 0, "backward_calls": 0, "asr_or_tts_calls": 0, "automatic_retry": False,
                "wall_seconds": time.monotonic() - start, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "human_gold": False, "independent_test": False, "previously_exposed_development": True,
                "quality_winner": None, "generalization_passed": None, "pairwise_preferences_only": True,
                "conditional_alignment_is_not_acoustic_truth": True, "original_logits_duplicated": False,
                "derived_fullprecision_arrays_retained": True}
            terminal_metadata(job, summary, training.sha_file(recipe_path))
            io.write_json(output / "summary.json", summary)
            io.write_json(output / "status.json", summary)
            io.write_json(output / "MANIFEST.json", make_manifest(output))
            return summary
        except BaseException as exc:
            failure = {"schema": SCHEMA, "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "counts": dict(counts), "automatic_retry": False, "new_model_forwards": 0, "optimizer_steps": 0}
            io.write_json(output / "failure.json", failure)
            if not (output / "status.json").exists():
                io.write_json(output / "status.json", failure)
            raise


def completed_queue(output, summary):
    equal(summary, training.read_json(output / "status.json"), "SAVED_CTC_TERMINAL_STATUS_DIVERGED")
    require(summary["status"] == "completed" and not (output / "failure.json").exists(), "SAVED_CTC_FAILED_TERMINAL")
    result = training.read_json(output.parent / "result.json")
    require(result["status"] == "completed" and result["exit_code"] == 0 and result["job_id"] == JOB_ID and
        result["recipe_sha256"] == training.sha_file(output / "request.json") and
        result["queue_sha256"] == training.sha_file(output.parents[2] / "queue.json") and
        result["claim_sha256"] == training.sha_file(output.parent / "claim.json") and
        result["outputs"]["MANIFEST.json"] == io.file_record(output / "MANIFEST.json"), "SAVED_CTC_COMPLETED_QUEUE_REQUIRED")


def audit(output, expected):
    """Recompute from original saved tensors; no model, optimizer or new labels."""
    with deadline(plan()["worker_seconds"]):
        return audit_within_deadline(output, expected)


def audit_within_deadline(output, expected):
    output = io.safe_path(output)
    onset.verify_manifest(output, expected)
    summary = training.read_json(output / "summary.json")
    completed_queue(output, summary)
    job = validate_recipe(training.read_json(output / "request.json"))
    contract = job["contract"]
    terminal_metadata(job, summary, training.sha_file(output / "request.json"))
    claim = training.read_json(output / "analysis-claim.json")
    require(claim["schema"] == SCHEMA and claim["recipe_sha256"] == summary["recipe_sha256"] and
        claim["automatic_retry"] is False and type(claim["new_model_forwards"]) is int and claim["new_model_forwards"] == 0,
        "SAVED_CTC_ANALYSIS_CLAIM")
    with training.heavy_model_slot(contract["heavy_lock"]):
        torch = numerical_runtime(contract)
        recorded, claims = lines(output / "conditions.jsonl"), lines(output / "analysis-calls.jsonl")
        require(len(recorded) == 264 and len(claims) == 528, "SAVED_CTC_SAVED_LEDGER_COUNT")
        with no_model_or_optimizer(torch), torch.inference_mode():
            for record, expected_result in zip(job["records"], recorded):
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= contract["max_rss_kib"], "SAVED_CTC_AUDIT_MEMORY_LIMIT")
                source = torch.load(record["numeric_artifact"]["path"], weights_only=True, map_location="cpu")
                verify_numeric(source, record, torch)
                arrays, result = numerical_condition(source, record, contract, torch)
                artifact = expected_result["derived_artifact"]
                name = "derived/%03d.pt" % record["ordinal"]
                equal(artifact, {"path": name, **io.file_record(output / name)}, "SAVED_CTC_DERIVED_FILE_PIN")
                retained = torch.load(output / name, weights_only=True, map_location="cpu")
                require(retained["schema"] == NUMERIC_SCHEMA and retained["condition_binding_sha256"] == record["condition_binding_sha256"] and
                    retained["original_numeric_sha256"] == record["numeric_artifact"]["sha256"] and retained["class_ids"] == record["class_ids"] and
                    set(retained["arrays"]) == set(ARRAY_KEYS), "SAVED_CTC_DERIVED_SCHEMA")
                expected_hashes = {k: training.tensor_sha(v) for k, v in arrays.items()}
                equal(retained["array_hashes"], expected_hashes, "SAVED_CTC_DERIVED_VALUES_CHANGED")
                equal({k: training.tensor_sha(v) for k, v in retained["arrays"].items()}, expected_hashes, "SAVED_CTC_DERIVED_ARRAY_BYTES")
                result["derived_artifact"] = artifact
                equal(result, expected_result, "SAVED_CTC_DERIVED_SCALARS_CHANGED")
                before, after = claims[record["ordinal"] * 2:record["ordinal"] * 2 + 2]
                require(before["phase"] == "started" and after["phase"] == "completed" and
                    before["ordinal"] == after["ordinal"] == record["ordinal"] and
                    before["condition_binding_sha256"] == after["condition_binding_sha256"] == record["condition_binding_sha256"] and
                    before["monotonic_ns"] <= after["monotonic_ns"] and after["derived_artifact"] == artifact and
                    after["result_sha256"] == fit.digest(result), "SAVED_CTC_ANALYSIS_CLAIM_PAIRS")
                del source, retained, arrays
        equal(aggregate(recorded), summary["conditions"], "SAVED_CTC_AGGREGATE_CHANGED")
        equal(job["source_hashes"], {n: training.sha_file(output / "execution-source" / n) for n in CODE}, "SAVED_CTC_EXECUTION_SOURCE_CHANGED")
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected,
            "saved_conditions_verified": 264, "single_token_deletion_contrasts_verified": 876,
            "logical_sequence_scores_verified": 1404, "unique_sequence_scores_verified": 1382,
            "new_model_constructions": 0, "new_model_forwards": 0, "optimizer_steps": 0,
            "conditional_alignment_is_not_acoustic_truth": True, "pairwise_preferences_only": True, "independent_test": False}


def queue_definition(recipe, python):
    job = training.read_json(recipe)
    equal(job["contract"], plan(), "SAVED_CTC_QUEUE_CONTRACT")
    return {"schema": io.SCHEMA, "queue_id": JOB_ID, "global_budget_minutes": 10, "jobs": [{"job_id": JOB_ID,
        "recipe": str(io.safe_path(recipe)), "recipe_sha256": training.sha_file(recipe),
        "argv": [python, "-I", "-B", str(ROOT / SOURCE), "run", "--recipe", "{recipe}", "--output", "{output_dir}",
            "--heavy-lock", job["contract"]["heavy_lock"]], "cwd": str(ROOT), "depends_on": [],
        "timeout_minutes": 10, "outputs": list(OUTPUTS)}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    specs = (("declare", ("context-output", "context-audit", "paired-output", "paired-audit", "output")),
        ("queue-request", ("recipe", "python", "output")), ("run", ("recipe", "output", "heavy-lock")),
        ("audit", ("output", "manifest-sha256", "receipt")))
    for name, options in specs:
        command = sub.add_parser(name)
        for option in options:
            command.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        inputs = {name: str(io.safe_path(getattr(args, name))) for name in ("context_output", "context_audit", "paired_output", "paired_audit")}
        job = assemble(inputs)
        io.write_json(Path(args.output), job)
        result = {"schema": SCHEMA, "status": "declared", "recipe": pin(args.output), "workload": job["workload"],
            "new_model_forwards": 0, "torch_imported": "torch" in sys.modules}
    elif args.command == "queue-request":
        io.write_json(Path(args.output), queue_definition(args.recipe, args.python))
        result = {"queue_request": pin(args.output), "new_model_forwards": 0}
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        require(not Path(args.receipt).exists() and not Path(args.receipt).is_symlink(), "SAVED_CTC_AUDIT_RECEIPT_ALREADY_EXISTS")
        result = audit(args.output, args.manifest_sha256)
        io.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
