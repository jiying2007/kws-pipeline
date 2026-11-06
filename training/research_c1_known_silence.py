#!/usr/bin/env python3
"""One fixed C1 correction: cyclic digital prefixes plus known-silence CTC.

The original eight training rows, zero-step checkpoint, full-text CTC and AdamW
settings are retained. This is a combined input/objective correction, not a
single-variable ablation. No model is imported by declare or metadata checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_c1_onset_diagnostic as onset

training, fit, evaluation = onset.training, onset.fit, onset.evaluation
io, require = onset.io, onset.require
SOURCE = "training/research_c1_known_silence.py"
CODE = (*onset.CODE, SOURCE)
SCHEMA = "kws-c1-known-silence-correction-v1"
INITIAL_SHA = "da7671fe71fef9ecd6822c747e3fe153b9ee563a6784c18eded90d3354486ecf"
STEPS, BATCH, FORWARDS, SECONDS = 2000, 4, 8000, 540
OUTPUTS = ("summary.json", "consumption.jsonl", "observations.jsonl", "final.pt", "MANIFEST.json")


def fixed_contract():
    return {"candidate": "c1-ctc2601", "optimizer_steps": STEPS, "batch_size": BATCH,
        "training_forward_limit": FORWARDS, "final_forward_limit": 24, "before_forward_calls": 0,
        "development_forward_calls": 0, "worker_seconds": SECONDS, "queue_minutes": 10,
        "offsets_ms": [0, 200, 500], "auxiliary_scale": 1.0,
        "auxiliary_definition": "empty-target CTC over only fully added-zero windows, divided by that window count",
        "auxiliary_confidence": 1.0, "auxiliary_uses_speech_weak_weight": False,
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
        "gradient_clip_max_norm": 5.0, "fixed_ctc_normalizer": 1.0, "seed": 20261105,
        "known_silence_source": "added PCM16 zeros only; no VAD or amplitude decision on original audio",
        "minimum_exact_per_condition": 5, "maximum_token_edits_per_condition": 7,
        "strict_tiny_fit_exact_conditions": 24, "required_known_zero_nonblank_frames": 0,
        "launch_requires_entire_acoustic124_campaign_terminal": True,
        "automatic_retry": False, "asr_or_tts_calls": 0, "labels_changed": False,
        "human_gold": False, "quality_winner": None, "single_variable_ablation": False,
        "scope": "fixed known-training-row input/objective correction, not independent generalization"}


def sources():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def validate_contract(job):
    require(type(job) is dict and job.get("schema") == SCHEMA and
        all(type(job.get(k)) is type(v) and job[k] == v for k, v in fixed_contract().items()) and
        all(type(x) is int for x in job["offsets_ms"]), "SILENCE_FIXED_CONTRACT_CHANGED")


def exposure_plan(rows):
    require(len(rows) == 8 and len({r["source_id"] for r in rows}) == 8, "SILENCE_EIGHT_ORIGINAL_ROWS")
    return [{"row_index": i % 8, "row_visit_index": i // 8, "source_id": rows[i % 8]["source_id"],
        "ctc_weight": rows[i % 8]["ctc_weight"], "offset_ms": onset.OFFSETS[(i // 8) % 3],
        "known_zero_prefix_frames": (0, 9, 24)[(i // 8) % 3]} for i in range(FORWARDS)]


def reference_inputs(path, expected, audit_path):
    reference = io.safe_path(path)
    onset.verify_manifest(reference, expected)
    prior_audit = training.read_json(audit_path)
    require(prior_audit["status"] == "verified" and prior_audit["manifest_sha256"] == expected and
        prior_audit["new_model_constructions"] == prior_audit["new_model_forwards"] == prior_audit["optimizer_steps"] == 0,
        "SILENCE_VERIFIED_ONSET_REQUIRED")
    onset_recipe = training.read_json(reference / "request.json")
    onset.validate_recipe(onset_recipe)
    benchmark = training.read_json(reference / "summary.json")
    require(benchmark["status"] == "completed" and benchmark["completed_forward_calls"] == 16 and
        benchmark["baseline_replays"] == 8 and benchmark["model_state_unchanged"] is True and
        benchmark["recipe_sha256"] == training.sha_file(reference / "request.json"), "SILENCE_COMPLETED_REFERENCE_REQUIRED")
    original = onset.origin_inputs(onset_recipe["origin"], onset_recipe["origin_manifest_sha256"], onset_recipe["prior_audit"]["path"])
    origin, prepared, control, record, vocabulary, rows, control_summary, _, lock = original
    initial = prepared / "initial/c1-ctc2601.pt"
    require(training.sha_file(initial) == INITIAL_SHA == control_summary["initial_checkpoint_sha256"], "SILENCE_SAME_ZERO_STEP_INITIAL_REQUIRED")
    base = training.read_json(prepared / "recipes/0.json")
    require(base["optimizers"]["fresh-rnn64-or-c1"] == fixed_contract()["optimizer"] and
        base["fixed_ctc_normalizer"] == 1.0 and base["gradient_clip_max_norm"] == 5.0 and base["seed"] == 20261105,
        "SILENCE_ORIGINAL_OPTIMIZER_OR_LOSS_CHANGED")
    return reference, original, benchmark, initial


def make_recipe(path, expected, audit_path):
    reference, original, benchmark, initial = reference_inputs(path, expected, audit_path)
    _, prepared, control, _, vocabulary, rows, _, _, lock = original
    draws = exposure_plan(rows)
    return {"schema": SCHEMA, **fixed_contract(), "reference_output": str(reference),
        "reference_manifest_sha256": expected, "reference_audit": onset.pin(audit_path),
        "reference_conditions": benchmark["conditions"], "initial_checkpoint": onset.pin(initial),
        "prepared": str(prepared), "prepared_manifest_sha256": control["prepared_manifest_sha256"],
        "source_ids": [r["source_id"] for r in rows], "row_binding_sha256": control["row_binding_sha256"],
        "vocabulary_sha256": fit.digest(vocabulary), "heavy_lock": lock, "source_hashes": sources(),
        "exposure_plan": {"draws": draws}, "exposure_plan_sha256": fit.digest(draws),
        "exposures_by_offset": {str(k): sum(d["offset_ms"] == k for d in draws) for k in onset.OFFSETS}}


def validate_recipe(job):
    validate_contract(job)
    require(job == make_recipe(job["reference_output"], job["reference_manifest_sha256"], job["reference_audit"]["path"]),
        "SILENCE_RECIPE_OR_INPUT_CHANGED")


def prefix_blank_ctc(logits, count):
    """An empty CTC target has exactly one path: blank at every known-zero frame.

    Mean NLL gives equal per-condition weight to 200/500 ms. This is a soft
    penalty, not a guarantee of blank output. No speech-label weight is used.
    """
    require(logits.ndim == 3 and tuple(logits.shape[1:]) == (1, 2601) and
        type(count) is int and count in (0, 9, 24) and count <= len(logits), "SILENCE_BLANK_PREFIX_GEOMETRY")
    return -logits[:count, 0].log_softmax(-1)[:, 0].mean() if count else logits.sum() * 0


def prefix_observation(logits, count):
    if not count:
        return {"known_zero_prefix_frames": 0, "known_zero_nonblank_frames": 0,
            "prefix_blank_probability_mean": None, "prefix_blank_probability_minimum": None,
            "prefix_blank_nll_mean": 0.0}
    values = logits[:count, 0]
    blank = values.softmax(-1)[:, 0]
    return {"known_zero_prefix_frames": count, "known_zero_nonblank_frames": int((values.argmax(-1) != 0).sum()),
        "prefix_blank_probability_mean": float(blank.mean()), "prefix_blank_probability_minimum": float(blank.min()),
        "prefix_blank_nll_mean": float(prefix_blank_ctc(logits, count))}


def decisions(observations):
    conditions = onset.summarize(observations)
    padded = [r for r in observations if r["offset_ms"]]
    require(len(padded) == 16 and all(r["known_zero_prefix_frames"] == {200: 9, 500: 24}[r["offset_ms"]] for r in padded),
        "SILENCE_FINAL_MASK_ROSTER_CHANGED")
    quiet = all(r["known_zero_nonblank_frames"] == 0 for r in padded)
    retention = all(c["exact_token_matches"] >= 5 and c["edit_numerator"] <= 7 for c in conditions.values())
    for offset in onset.OFFSETS:
        group = [r for r in observations if r["offset_ms"] == offset]
        conditions[str(offset)].update(known_zero_frames=sum(r["known_zero_prefix_frames"] for r in group),
            known_zero_nonblank_frames=sum(r["known_zero_nonblank_frames"] for r in group))
    return {"conditions": conditions, "known_silence_boundary_passed": quiet,
        "original_unpadded_accuracy_restored_in_each_condition": retention,
        "minimum_mechanism_gate_passed": quiet and retention,
        "strict_tiny_fit_passed": quiet and sum(r["exact_token_match"] for r in observations) == 24,
        "quality_winner": None, "generalization_passed": None}


@contextmanager
def worker_deadline():
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "SILENCE_EXISTING_DEADLINE")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise TimeoutError("SILENCE_FIXED_540_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, SECONDS)
    try:
        yield time.monotonic() + SECONDS
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def feature_bank(prepared, rows, torch):
    bank, evidence, pcms = {}, [], []
    original_features = {r["source_id"]: r for r in training.read_json(prepared / "feature-observations.json")}
    for index, row in enumerate(rows):
        pcm = evaluation.read_pcm(prepared, row)
        pcms.append(pcm)
        baseline_sha = None
        for offset in onset.OFFSETS:
            padded, shape = onset.prepend_pcm(pcm, offset)
            derived = {**row, "expected_output_frames": {**row["expected_output_frames"], "c1": shape["frames"]}}
            feature = evaluation.native_features(padded, derived, "c1-ctc2601", torch)
            suffix_sha = training.tensor_sha(feature[shape["shift_frames"]:])
            if not offset:
                baseline_sha = suffix_sha
                require(original_features[row["source_id"]]["pcm_sha256"] == row["pcm_sha256"] and
                    baseline_sha == original_features[row["source_id"]]["fresh_feature_sha256"],
                    "SILENCE_ORIGINAL_SAVED_FEATURE_CHANGED")
            require(suffix_sha == baseline_sha and shape["fully_digital_silence_frames"] == {0: 0, 200: 9, 500: 24}[offset] and
                bool((feature[:shape["fully_digital_silence_frames"]] == 0).all()), "SILENCE_KNOWN_ZERO_OR_FEATURE_SUFFIX_CHANGED")
            bank[index, offset] = feature
            evidence.append({"source_id": row["source_id"], "offset_ms": offset, **shape,
                "pcm_sha256": row["pcm_sha256"], "feature_sha256": training.tensor_sha(feature),
                "original_feature_suffix_sha256": suffix_sha, "original_feature_sha256": baseline_sha,
                "only_added_pcm_used_for_blank_mask": True})
    return bank, evidence, pcms


def ledger_readback(events, job, rows):
    gradients = fit.gradient_observation(events, job, rows)
    for step, event in enumerate(events):
        expected = job["exposure_plan"]["draws"][step * BATCH:(step + 1) * BATCH]
        primary, auxiliary, total = 0.0, 0.0, 0.0
        for item, draw in zip(event["draws"], expected):
            require(item["offset_ms"] == draw["offset_ms"] and item["row_visit_index"] == draw["row_visit_index"] and
                item["known_zero_prefix_frames"] == draw["known_zero_prefix_frames"] and item["auxiliary_scale"] == 1.0 and
                math.isfinite(item["aux_blank_nll_mean"]) and item["aux_blank_nll_mean"] >= 0 and
                (draw["offset_ms"] != 0 or item["aux_blank_nll_mean"] == 0), "SILENCE_CONSUMED_PREFIX_CHANGED")
            ctc = item["raw_length_normalized_ctc"] * item["ctc_weight"] / job["fixed_ctc_normalizer"] / BATCH
            aux = item["aux_blank_nll_mean"] / BATCH
            require(math.isclose(item["weighted_batch_contribution"], ctc, rel_tol=1e-6, abs_tol=1e-7) and
                math.isclose(item["aux_batch_contribution"], aux, rel_tol=1e-6, abs_tol=1e-7), "SILENCE_LOSS_SCALE_CHANGED")
            primary += ctc; auxiliary += aux; total += item["total_batch_contribution"]
        require(math.isclose(event["fulltext_ctc_batch_loss"], primary, rel_tol=1e-6, abs_tol=1e-6) and
            math.isclose(event["aux_batch_loss"], auxiliary, rel_tol=1e-6, abs_tol=1e-6) and
            math.isclose(event["loss"], total, rel_tol=1e-6, abs_tol=1e-6) and
            math.isclose(event["loss"], primary + auxiliary, rel_tol=1e-6, abs_tol=1e-6), "SILENCE_BATCH_LOSS_CHANGED")
    return gradients


def train(model, bank, rows, job, optimizer, output, end, status, torch):
    """Original batch/clip/AdamW operations, with explicit separate blank terms."""
    parameters = list(model.parameters())
    draws = job["exposure_plan"]["draws"]
    model.train()
    for begin in range(0, FORWARDS, BATCH):
        require(time.monotonic() < end, "SILENCE_TRAINING_DEADLINE")
        optimizer.zero_grad(set_to_none=True)
        terms, recorded = [], []
        for draw in draws[begin:begin + BATCH]:
            row = rows[draw["row_index"]]
            status["training_forwards_started"] += 1
            logits = model(bank[draw["row_index"], draw["offset_ms"]].unsqueeze(0))
            status["training_forwards_completed"] += 1
            primary, raw = training.fulltext_ctc(logits, row["target_ids"], row["ctc_weight"], row["ctc_mask"],
                normalizer=job["fixed_ctc_normalizer"])
            auxiliary = prefix_blank_ctc(logits, draw["known_zero_prefix_frames"])
            term = primary + job["auxiliary_scale"] * auxiliary
            require(bool(term.isfinite()), "SILENCE_NONFINITE_LOSS")
            terms.append(term)
            recorded.append({"source_id": row["source_id"], "label_binding_sha256": fit.row_identity(row),
                "target_ids": row["target_ids"], "full_text": row["full_text"], "pcm_sha256": row["pcm_sha256"],
                "source_family_id": row["source_family_id"], "ctc_mask": row["ctc_mask"], "ctc_weight": row["ctc_weight"],
                "keyword_loss_used": False, "offset_ms": draw["offset_ms"], "row_visit_index": draw["row_visit_index"],
                "known_zero_prefix_frames": draw["known_zero_prefix_frames"], "raw_length_normalized_ctc": raw.item(),
                "weighted_batch_contribution": primary.detach().item() / BATCH,
                "aux_blank_nll_mean": auxiliary.detach().item(), "auxiliary_scale": 1.0,
                "aux_batch_contribution": auxiliary.detach().item() / BATCH,
                "total_batch_contribution": term.detach().item() / BATCH})
        loss = torch.stack(terms).mean()
        loss.backward()
        require(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in parameters), "SILENCE_GRADIENT_INVALID")
        norm = torch.nn.utils.clip_grad_norm_(parameters, job["gradient_clip_max_norm"])
        require(bool(norm.isfinite()), "SILENCE_GRADIENT_NORM_INVALID")
        optimizer.step()
        status["actual_optimizer_steps"] += 1
        training.append_json(output / "consumption.jsonl", {"optimizer_step": begin // BATCH + 1,
            "draw_offset": begin, "loss": loss.item(), "gradient_norm_before_clip": norm.item(), "draws": recorded,
            "fulltext_ctc_batch_loss": sum(r["weighted_batch_contribution"] for r in recorded),
            "aux_batch_loss": sum(r["aux_batch_contribution"] for r in recorded)})
        require(time.monotonic() < end, "SILENCE_TRAINING_DEADLINE")


def observe(model, bank, rows, pcms, history, output, end, status, torch):
    model.eval()
    result = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            baseline = None
            for offset in onset.OFFSETS:
                require(time.monotonic() < end and status["final_forwards_started"] < 24, "SILENCE_FINAL_BUDGET")
                feature = bank[index, offset]
                status["final_forwards_started"] += 1
                logits = model(feature.unsqueeze(0))
                status["final_forwards_completed"] += 1
                if not offset:
                    baseline = logits
                name = "numeric/ms-%03d/%s.pt" % (offset, row["source_id"])
                artifact = {"path": name, **evaluation.save_numeric(output / name, row["source_id"], feature, logits)}
                item = {**onset.observation(row, pcms[index], offset, feature, logits, baseline, bank[index, 0], history, artifact),
                    "new_model_forward": True, **prefix_observation(logits, {0: 0, 200: 9, 500: 24}[offset])}
                result.append(item)
                training.append_json(output / "observations.jsonl", item)
    return result


def run(recipe_path, output, lock):
    job = training.read_json(recipe_path)
    validate_contract(job)
    require(str(io.safe_path(lock)) == job["heavy_lock"], "SILENCE_CANONICAL_LOCK_REQUIRED")
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        status = {"schema": SCHEMA, **fixed_contract(), "status": "running", "started_at": training.utc(),
            "recipe_sha256": training.sha_file(recipe_path), "model_constructions": 0,
            "training_forwards_started": 0, "training_forwards_completed": 0,
            "final_forwards_started": 0, "final_forwards_completed": 0, "actual_optimizer_steps": 0}
        training.save_json(output / "status.json", status)
        try:
            with worker_deadline() as end:
                validate_recipe(job)
                _, original, benchmark, initial_path = reference_inputs(job["reference_output"], job["reference_manifest_sha256"], job["reference_audit"]["path"])
                _, prepared, _, record, vocabulary, rows, _, _, _ = original
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                torch, reference_model = training.torch_runtime()
                from training_state import load_checkpoint_snapshot, state_identity, atomic_save_checkpoint
                initial, initial_sha = load_checkpoint_snapshot(initial_path)
                require(initial_sha == INITIAL_SHA == job["initial_checkpoint"]["sha256"] and
                    initial["optimizer_steps"] == 0 and initial["candidate"] == "c1-ctc2601" and
                    initial["vocabulary_sha256"] == job["vocabulary_sha256"] and
                    initial["source_hashes"] == record["source_hashes"] and
                    state_identity(initial["state_dict"]) == initial["float_state_identity"], "SILENCE_INITIAL_CHANGED")
                bank, features, pcms = feature_bank(prepared, rows, torch)
                torch.manual_seed(job["seed"])
                model, description = training.build_fresh("c1-ctc2601", vocabulary, reference_model)
                status["model_constructions"] = 1
                require(description == initial["description"], "SILENCE_ARCHITECTURE_CHANGED")
                model.load_state_dict(initial["state_dict"], strict=True)
                require(state_identity(model.state_dict()) == initial["float_state_identity"], "SILENCE_INITIAL_LOAD_CHANGED")
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                training.copy_once(recipe_path, output / "request.json")
                training.copy_once(initial_path, output / "initial.pt")
                io.write_json(output / "selected-rows.json", rows)
                io.write_json(output / "feature-bindings.json", features)
                io.write_json(output / "reference-summary.json", benchmark)
                training.copy_once(prepared / "vocabulary.json", output / "vocabulary.json")
                for name in CODE:
                    training.copy_once(ROOT / name, output / "execution-source" / name)
                train(model, bank, rows, job, optimizer, output, end, status, torch)
                events = [io.loads(line) for line in io.read_bytes(output / "consumption.jsonl").splitlines()]
                gradients = ledger_readback(events, job, rows)
                final_identity = state_identity(model.state_dict())
                names = [name for name, _ in model.named_parameters()]
                changed, delta = fit.parameter_changes(initial["state_dict"], model.state_dict(), names)
                history = sum((k - 1) * d for k, d in zip(description["kernels"], description["dilations"]))
                observations = observe(model, bank, rows, pcms, history, output, end, status, torch)
                require(state_identity(model.state_dict()) == final_identity, "SILENCE_FINAL_OBSERVATION_MUTATED_MODEL")
                payload = {"schema": SCHEMA, "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "float_state_identity": final_identity, "optimizer_steps": STEPS, "training_model_forwards": FORWARDS,
                    "initial_checkpoint_sha256": initial_sha, "job_recipe_sha256": status["recipe_sha256"],
                    "vocabulary_sha256": job["vocabulary_sha256"], "candidate": "c1-ctc2601"}
                atomic_save_checkpoint(payload, output / "final.pt")
                restored, final_sha = load_checkpoint_snapshot(output / "final.pt")
                require(state_identity(restored["state_dict"]) == final_identity and
                    status["training_forwards_started"] == status["training_forwards_completed"] == FORWARDS and
                    status["final_forwards_started"] == status["final_forwards_completed"] == 24 and
                    status["actual_optimizer_steps"] == STEPS and time.monotonic() < end,
                    "SILENCE_FINAL_COUNTS_OR_CHECKPOINT_CHANGED")
                validate_recipe(job)
                now = resource.getrusage(resource.RUSAGE_SELF)
                status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                    user_cpu_seconds=now.ru_utime-usage.ru_utime, system_cpu_seconds=now.ru_stime-usage.ru_stime,
                    peak_rss_kib=now.ru_maxrss, initial_checkpoint_sha256=initial_sha, final_checkpoint_sha256=final_sha,
                    initial_float_state_sha256=initial["float_state_identity"]["sha256"], final_float_state_sha256=final_identity["sha256"],
                    parameter_names=names, changed_parameter_names=changed, parameter_delta_l2=delta, gradients=gradients,
                    exposure_plan_sha256=job["exposure_plan_sha256"], exposures_by_offset=job["exposures_by_offset"],
                    reference_conditions=benchmark["conditions"], **decisions(observations))
                training.save_json(output / "status.json", status)
                io.write_json(output / "summary.json", status)
                io.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): io.file_record(p)
                    for p in sorted(output.rglob("*")) if p.is_file()})
                return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), error_type=type(error).__name__, error=str(error),
                wall_seconds=time.monotonic()-started, counts_are_lower_bounds=True)
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected):
    """Authenticate saved gradients/optimizer and rescore all saved tensors."""
    output = io.safe_path(output)
    onset.verify_manifest(output, expected)
    job, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    with training.heavy_model_slot(job["heavy_lock"]):
        validate_recipe(job)
        validate_contract(summary)
        require(summary["status"] == "completed" and summary["recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["training_forwards_started"] == summary["training_forwards_completed"] == FORWARDS and
            summary["final_forwards_started"] == summary["final_forwards_completed"] == 24 and
            summary["actual_optimizer_steps"] == STEPS, "SILENCE_SAVED_COUNTS_CHANGED")
        _, original, benchmark, _ = reference_inputs(job["reference_output"], job["reference_manifest_sha256"], job["reference_audit"]["path"])
        origin, prepared, _, _, _, rows, control_summary, _, _ = original
        require(training.read_json(output / "selected-rows.json") == rows and
            {name: training.sha_file(output / "execution-source" / name) for name in CODE} == job["source_hashes"], "SILENCE_SAVED_SOURCES_CHANGED")
        runtime = training.research_runtime.load_manifest()
        training.research_runtime.check_interpreter(runtime)
        training.research_runtime.check_packages(runtime, "core")
        torch, _ = training.torch_runtime()
        from training_state import load_checkpoint_snapshot, state_identity
        initial, initial_sha = load_checkpoint_snapshot(output / "initial.pt")
        final, final_sha = load_checkpoint_snapshot(output / "final.pt")
        control, control_sha = load_checkpoint_snapshot(origin / "final.pt")
        require(initial_sha == INITIAL_SHA and control_sha == control_summary["final_checkpoint_sha256"] and
            final_sha == summary["final_checkpoint_sha256"] and final["schema"] == SCHEMA and
            final["initial_checkpoint_sha256"] == INITIAL_SHA and final["job_recipe_sha256"] == summary["recipe_sha256"] and
            final["optimizer_steps"] == STEPS and final["training_model_forwards"] == FORWARDS and
            final["candidate"] == "c1-ctc2601" and final["vocabulary_sha256"] == job["vocabulary_sha256"] and
            state_identity(final["state_dict"]) == final["float_state_identity"] and
            final["float_state_identity"]["sha256"] == summary["final_float_state_sha256"], "SILENCE_SAVED_CHECKPOINT_CHANGED")
        actual, prior = final["optimizer_state"], control["optimizer_state"]
        require(actual["param_groups"] == prior["param_groups"] and set(actual["state"]) == set(prior["state"]), "SILENCE_OPTIMIZER_SETTINGS_CHANGED")
        params = actual["param_groups"][0]["params"]
        require(len(params) == len(summary["parameter_names"]) and
            summary["parameter_names"] == control_summary["trainable_parameter_names"], "SILENCE_OPTIMIZER_PARAMETER_MAP")
        for index, name in zip(params, summary["parameter_names"]):
            state = actual["state"][index]
            require(set(state) == set(prior["state"][index]) and state["step"].item() == STEPS and
                all(state[k].shape == final["state_dict"][name].shape and bool(state[k].isfinite().all())
                    for k in ("exp_avg", "exp_avg_sq")), "SILENCE_OPTIMIZER_STEPS_OR_MOMENTS")
        changed, delta = fit.parameter_changes(initial["state_dict"], final["state_dict"], summary["parameter_names"])
        require(changed == summary["changed_parameter_names"] and delta == summary["parameter_delta_l2"], "SILENCE_PARAMETER_DELTA_CHANGED")
        events = [io.loads(line) for line in io.read_bytes(output / "consumption.jsonl").splitlines()]
        require(ledger_readback(events, job, rows) == summary["gradients"], "SILENCE_SAVED_GRADIENT_LEDGER_CHANGED")
        bank, feature_evidence, pcms = feature_bank(prepared, rows, torch)
        require(feature_evidence == training.read_json(output / "feature-bindings.json"), "SILENCE_SAVED_FEATURES_CHANGED")
        observations = [io.loads(line) for line in io.read_bytes(output / "observations.jsonl").splitlines()]
        require(len(observations) == 24, "SILENCE_FINAL_OBSERVATION_COUNT")
        description = initial["description"]
        history = sum((k-1)*d for k,d in zip(description["kernels"], description["dilations"]))
        for index, row in enumerate(rows):
            baseline = None
            for ordinal, offset in enumerate(onset.OFFSETS):
                item = observations[index*3+ordinal]
                artifact = item["numeric_artifact"]
                path = output / io.relative(artifact["path"])
                require(artifact["path"] == "numeric/ms-%03d/%s.pt" % (offset,row["source_id"]) and
                    io.file_record(path) == {k: artifact[k] for k in ("bytes","sha256")}, "SILENCE_SAVED_NUMERIC_PIN")
                saved = torch.load(path, weights_only=True, map_location="cpu")
                feature = bank[index,offset]
                require(saved["source_id"] == row["source_id"] and training.tensor_sha(feature) == saved["feature_sha256"] and
                    training.tensor_sha(saved["feature"]) == saved["feature_sha256"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"], "SILENCE_SAVED_NUMERIC_CHANGED")
                if not offset:
                    baseline = saved["logits"]
                rebuilt = {**onset.observation(row, pcms[index], offset, feature, saved["logits"], baseline, bank[index,0],history,artifact),
                    "new_model_forward": True, **prefix_observation(saved["logits"], {0:0,200:9,500:24}[offset])}
                require(rebuilt == item, "SILENCE_SAVED_OBSERVATION_CHANGED")
        require(all(summary[k] == v for k,v in decisions(observations).items()) and
            summary["reference_conditions"] == benchmark["conditions"], "SILENCE_SAVED_DECISION_CHANGED")
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected,
            "saved_optimizer_steps": STEPS, "saved_training_forwards": FORWARDS, "saved_final_conditions": 24,
            "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0,
            "minimum_mechanism_gate_passed": summary["minimum_mechanism_gate_passed"],
            "strict_tiny_fit_passed": summary["strict_tiny_fit_passed"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    declare=sub.add_parser("declare")
    for name in ("reference-output","reference-manifest-sha256","reference-audit","output"):
        declare.add_argument("--"+name,required=True)
    worker=sub.add_parser("run")
    for name in ("recipe","output","heavy-lock"):
        worker.add_argument("--"+name,required=True)
    check=sub.add_parser("audit")
    for name in ("output","manifest-sha256","receipt"):
        check.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    if args.command=="declare":
        job=make_recipe(args.reference_output,args.reference_manifest_sha256,args.reference_audit)
        io.write_json(Path(args.output),job)
        result={"recipe":args.output,"sha256":training.sha_file(args.output),"new_model_calls":0,
            "exposures_by_offset":job["exposures_by_offset"],"training_forward_limit":FORWARDS,"final_forward_limit":24}
    elif args.command=="run":
        result=run(Path(args.recipe),Path(args.output),Path(args.heavy_lock))
    else:
        result=audit(args.output,args.manifest_sha256)
        io.write_json(Path(args.receipt),result)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
