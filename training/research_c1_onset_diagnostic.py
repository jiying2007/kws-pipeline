#!/usr/bin/env python3
"""Fixed-checkpoint onset sensitivity, reusing the original D1 CPU evaluator.

Zero milliseconds replays saved logits. 200/500 ms prepend PCM16 digital zeros
before the unchanged frontend. Sixteen new reset-sequence forwards, no optimizer,
no ASR/TTS, no label change, and no model selection. Declare imports no Torch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_fit_diagnostic as fit

training, evaluation = fit.training, fit.evaluation
io, require = training.io_contract, fit.require
SOURCE = "training/research_c1_onset_diagnostic.py"
SCHEMA = "kws-c1-onset-diagnostic-v1"
OFFSETS = (0, 200, 500)
RATE, HOP, FRAME = 16000, 320, 400
CANDIDATE = "c1-ctc2601"
CODE = (*training.CODE_PATHS, *fit.SOURCE_PATHS, SOURCE)
OUTPUTS = ("summary.json", "observations.jsonl", "forward-calls.jsonl", "MANIFEST.json")


def sources():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def pin(path):
    return {"path": str(io.safe_path(path)), **io.file_record(path)}


def fixed_contract():
    return {"offsets_ms": list(OFFSETS), "baseline": "reuse-saved-after-logits",
        "order": "original-eight-row-order; offsets-0-200-500",
        "selected_training_rows": 8, "observation_count": 24, "new_forward_limit": 16,
        "worker_seconds": 240, "queue_minutes": 5, "optimizer_steps": 0,
        "asr_or_tts_calls": 0, "automatic_retry": False, "reset_each_condition": True,
        "frontend_normalization": "logmel; feature-bin mean within each frame; no across-time normalization",
        "original_feature_suffix_must_be_bitwise_equal": True,
        "human_gold": False, "quality_winner": None, "labels_changed": False,
        "scope": "known-training-row fixed-checkpoint onset sensitivity only"}


def validate_contract(job):
    require(type(job) is dict and job.get("schema") == SCHEMA and
            all(type(job.get(k)) is type(v) and job[k] == v for k, v in fixed_contract().items()),
            "ONSET_FIXED_CONTRACT_CHANGED")
    require(all(type(value) is int for value in job["offsets_ms"]), "ONSET_TYPED_OFFSETS_REQUIRED")


def geometry(samples, offset_ms):
    require(type(samples) is int and samples >= FRAME and type(offset_ms) is int and
            offset_ms in OFFSETS, "ONSET_PCM_GEOMETRY")
    added = offset_ms * RATE // 1000
    require(added % HOP == 0, "ONSET_HOP_ALIGNED_OFFSET_REQUIRED")
    return {"added_samples": added, "shift_frames": added // HOP,
        "frames": (samples + added - FRAME) // HOP + 1,
        "fully_digital_silence_frames": max(0, (added - FRAME) // HOP + 1)}


def prepend_pcm(pcm, offset_ms):
    require(type(pcm) is bytes and len(pcm) % 2 == 0, "ONSET_PCM16_BYTES_REQUIRED")
    shape = geometry(len(pcm) // 2, offset_ms)
    return b"\0\0" * shape["added_samples"] + pcm, shape


def emission_locations(frame_ids, offset_ms):
    """Greedy CTC emissions; window coordinates are not phoneme alignment."""
    require(type(offset_ms) is int and offset_ms in OFFSETS, "ONSET_OFFSET_CHANGED")
    previous, result = None, []
    for frame, token in enumerate(frame_ids):
        require(type(token) is int and 0 <= token < 2601, "ONSET_FRAME_TOKEN")
        if token and token != previous:
            start, end = frame * HOP, frame * HOP + FRAME
            result.append({"token_id": token, "frame_index": frame,
                "window_start_seconds": start / RATE, "window_end_seconds": end / RATE,
                "original_window_start_seconds": start / RATE - offset_ms / 1000,
                "original_window_end_seconds": end / RATE - offset_ms / 1000,
                "window_entirely_in_added_silence": end <= offset_ms * RATE // 1000})
        previous = token
    return result


def verify_manifest(root, expected):
    raw = io.read_bytes(root / "MANIFEST.json")
    require(hashlib.sha256(raw).hexdigest() == expected, "ONSET_MANIFEST_CHANGED")
    manifest = io.loads(raw)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != root / "MANIFEST.json"}
    require(type(manifest) is dict and set(manifest) == actual, "ONSET_FILE_ROSTER_CHANGED")
    for name, record in manifest.items():
        require(io.file_record(root / io.relative(name)) == record, "ONSET_FILE_CHANGED:" + name)
    return manifest


def origin_inputs(origin, expected, prior_audit):
    """Authenticate the already executed eight-row 2000-step run; no tensors."""
    origin = io.safe_path(origin)
    manifest = verify_manifest(origin, expected)
    audit = training.read_json(prior_audit)
    require(audit["status"] == "verified" and audit["manifest_sha256"] == expected and
            audit["new_model_forwards"] == audit["optimizer_steps"] == 0, "ONSET_PRIOR_AUDIT_REQUIRED")
    receipt = training.read_json(origin.parent / "result.json")
    queue = origin.parents[2] / "queue.json"
    require(receipt["schema"] == "kws-local-research-job-result-v1" and
            receipt["status"] == "completed" and receipt["exit_code"] == 0 and
            receipt["queue_sha256"] == training.sha_file(queue) and
            receipt["outputs"]["MANIFEST.json"] == io.file_record(origin / "MANIFEST.json"),
            "ONSET_ORIGINAL_QUEUE_RECEIPT_CHANGED")
    original = fit.validate_job(training.read_json(origin / "request.json"))
    require(original["candidate"] == CANDIDATE and original["optimizer_steps"] == 2000 and
            len(original["source_ids"]) == 8 and original["diagnostic_source_hashes"] == fit.diagnostic_sources() and
            receipt["recipe_sha256"] == training.sha_file(origin / "request.json"), "ONSET_ORIGINAL_JOB_CHANGED")
    prepared, record, vocabulary, cohort = fit.prepared_rows(original)
    rows = fit.verify_selection(original, cohort, vocabulary)
    require(rows == training.read_json(origin / "selected-rows.json"), "ONSET_ORIGINAL_ROWS_CHANGED")
    summary = training.read_json(origin / "summary.json")
    require(summary["schema"] == fit.RESULT_SCHEMA and summary["status"] == "completed" and
            summary["candidate"] == CANDIDATE and summary["optimizer_steps"] == 2000 and
            summary["selected_training_rows"] == 8 and
            summary["final_checkpoint_sha256"] == manifest["final.pt"]["sha256"] and
            summary["job_recipe_sha256"] == receipt["recipe_sha256"] and
            summary["prepared_manifest_sha256"] == original["prepared_manifest_sha256"] and
            summary["row_binding_sha256"] == original["row_binding_sha256"], "ONSET_ORIGINAL_SUMMARY_CHANGED")
    after = training.read_json(origin / "after.json")
    require(len(after) == 8 and all(observation["source_id"] == row["source_id"] and
            observation["label_binding_sha256"] == fit.row_identity(row)
            for observation, row in zip(after, rows)), "ONSET_ORIGINAL_LOGIT_ROSTER")
    return origin, prepared, original, record, vocabulary, rows, summary, after, training.read_json(queue)["heavy_lock"]


def make_recipe(origin, expected, prior_audit):
    root, prepared, original, _, vocabulary, rows, _, _, lock = origin_inputs(origin, expected, prior_audit)
    return {"schema": SCHEMA, **fixed_contract(), "origin": str(root), "origin_manifest_sha256": expected,
        "prior_audit": pin(prior_audit), "checkpoint": pin(root / "final.pt"),
        "prepared": str(prepared), "prepared_manifest_sha256": original["prepared_manifest_sha256"],
        "source_ids": [row["source_id"] for row in rows], "row_binding_sha256": original["row_binding_sha256"],
        "vocabulary_sha256": fit.digest(vocabulary), "heavy_lock": lock, "source_hashes": sources()}


def validate_recipe(job):
    validate_contract(job)
    require(job == make_recipe(job["origin"], job["origin_manifest_sha256"], job["prior_audit"]["path"]),
            "ONSET_RECIPE_OR_INPUT_CHANGED")


@contextmanager
def deadline(seconds):
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "ONSET_EXISTING_DEADLINE")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise TimeoutError("ONSET_240_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield time.monotonic() + seconds
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def observation(row, pcm, offset, feature, logits, baseline, baseline_feature, history_frames, artifact):
    shaped = geometry(len(pcm) // 2, offset)
    padded, _ = prepend_pcm(pcm, offset)
    probabilities = logits[:, 0].softmax(dim=-1)
    emissions = emission_locations(logits[:, 0].argmax(dim=-1).tolist(), offset)
    for event in emissions:
        event["posterior_probability"] = float(probabilities[event["frame_index"], event["token_id"]])
    aligned = logits[shaped["shift_frames"]:]
    require(aligned.shape == baseline.shape, "ONSET_ALIGNED_LOGIT_GEOMETRY")
    aligned_feature = feature[shaped["shift_frames"]:]
    require(aligned_feature.shape == baseline_feature.shape and aligned_feature.equal(baseline_feature),
            "ONSET_ALIGNED_ORIGINAL_FEATURE_CHANGED")
    feature_delta = (aligned_feature - baseline_feature).abs()
    delta = (aligned - baseline).abs()
    tail = delta[history_frames:]
    return {"source_id": row["source_id"], "label_binding_sha256": fit.row_identity(row),
        "target_text": row["full_text"], "target_ids": row["target_ids"], "ctc_mask": row["ctc_mask"],
        "ctc_weight": row["ctc_weight"], "original_pcm_sha256": row["pcm_sha256"], "offset_ms": offset,
        "derived_pcm_sha256": hashlib.sha256(padded).hexdigest(), **shaped,
        "aligned_original_feature_sha256": training.tensor_sha(aligned_feature),
        "baseline_original_feature_sha256": training.tensor_sha(baseline_feature),
        "aligned_original_feature_bitwise_equal": True,
        "aligned_original_feature_maximum_absolute_delta": float(feature_delta.max()),
        "new_model_forward": offset != 0, "sequence_reset": True, "numeric_artifact": artifact,
        **fit.score(logits, row), "emissions": emissions,
        "emissions_entirely_in_added_silence": sum(e["window_entirely_in_added_silence"] for e in emissions),
        "aligned_maximum_absolute_logit_delta": float(delta.max()),
        "aligned_argmax_match_fraction": float((aligned.argmax(-1) == baseline.argmax(-1)).float().mean()),
        "receptive_history_frames": history_frames,
        "after_history_maximum_absolute_logit_delta": float(tail.max()) if len(tail) else None,
        "window_coordinates_are_phoneme_alignment": False}


def summarize(observations):
    require(len(observations) == 24 and len({r["source_id"] for r in observations}) == 8 and
            [(r["source_id"], r["offset_ms"]) for r in observations] ==
            [(observations[i]["source_id"], offset) for i in range(0, 24, 3) for offset in OFFSETS],
            "ONSET_OBSERVATION_ORDER_OR_COUNT")
    return {str(offset): {**fit.summarize([r for r in observations if r["offset_ms"] == offset]),
        "all_original_feature_suffixes_bitwise_equal": all(r["aligned_original_feature_bitwise_equal"]
            for r in observations if r["offset_ms"] == offset),
        "maximum_aligned_original_feature_delta": max(r["aligned_original_feature_maximum_absolute_delta"]
            for r in observations if r["offset_ms"] == offset),
        "emissions_entirely_in_added_silence": sum(r["emissions_entirely_in_added_silence"]
            for r in observations if r["offset_ms"] == offset)} for offset in OFFSETS}


def run(recipe_path, output, lock):
    job = training.read_json(recipe_path)
    validate_contract(job)
    require(str(io.safe_path(lock)) == job["heavy_lock"], "ONSET_CANONICAL_HEAVY_LOCK_CHANGED")
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        status = {"schema": SCHEMA, "status": "running", **fixed_contract(), "model_constructions": 0,
            "started_forward_calls": 0, "completed_forward_calls": 0, "baseline_replays": 0,
            "started_at": training.utc(), "recipe_sha256": training.sha_file(recipe_path)}
        training.save_json(output / "status.json", status)
        try:
            with deadline(job["worker_seconds"]) as end:
                validate_recipe(job)
                root, prepared, original, record, vocabulary, rows, summary, saved_after, _ = origin_inputs(
                    job["origin"], job["origin_manifest_sha256"], job["prior_audit"]["path"])
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                torch, reference = training.torch_runtime()
                from training_state import load_checkpoint_snapshot, state_identity
                checkpoint, checkpoint_sha = load_checkpoint_snapshot(root / "final.pt")
                require(checkpoint_sha == job["checkpoint"]["sha256"] and checkpoint["schema"] == fit.RESULT_SCHEMA and
                    checkpoint["candidate"] == CANDIDATE and checkpoint["optimizer_steps"] == 2000 and
                    checkpoint["prepared_manifest_sha256"] == original["prepared_manifest_sha256"] and
                    checkpoint["job_recipe_sha256"] == summary["job_recipe_sha256"] and
                    state_identity(checkpoint["state_dict"]) == checkpoint["float_state_identity"] and
                    checkpoint["float_state_identity"]["sha256"] == summary["final_float_state_sha256"],
                    "ONSET_FIXED_CHECKPOINT_CHANGED")
                torch.manual_seed(fit.SEED)
                model, description = training.build_fresh(CANDIDATE, vocabulary, reference)
                status["model_constructions"] = 1
                require(description == record["initial_checkpoints"][CANDIDATE]["description"], "ONSET_ARCHITECTURE_CHANGED")
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval().requires_grad_(False)
                require(state_identity(model.state_dict()) == checkpoint["float_state_identity"], "ONSET_MODEL_LOAD_CHANGED")
                history = sum((k - 1) * d for k, d in zip(description["kernels"], description["dilations"]))
                for name in CODE:
                    training.copy_once(ROOT / name, output / "execution-source" / name)
                training.copy_once(recipe_path, output / "request.json")
                training.copy_once(root / "final.pt", output / "fixed-checkpoint.pt")
                training.copy_once(root / "selected-rows.json", output / "selected-rows.json")
                training.copy_once(prepared / "vocabulary.json", output / "vocabulary.json")
                observed = []
                with torch.inference_mode():
                    for row, original_observation in zip(rows, saved_after):
                        pcm = evaluation.read_pcm(prepared, row)
                        original_numeric = root / io.relative(original_observation["numeric_artifact"]["path"])
                        saved = torch.load(original_numeric, weights_only=True, map_location="cpu")
                        feature0 = evaluation.native_features(pcm, row, CANDIDATE, torch)
                        require(saved["source_id"] == row["source_id"] and
                            training.tensor_sha(saved["feature"]) == saved["feature_sha256"] and
                            training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                            torch.equal(feature0, saved["feature"]) and
                            all(original_observation[k] == v for k, v in fit.score(saved["logits"], row).items()),
                            "ONSET_BASELINE_REPLAY_CHANGED")
                        for offset in OFFSETS:
                            require(time.monotonic() < end, "ONSET_MONOTONIC_DEADLINE")
                            padded, shape = prepend_pcm(pcm, offset)
                            feature_row = {**row, "expected_output_frames": {**row["expected_output_frames"], "c1": shape["frames"]}}
                            feature = feature0 if not offset else evaluation.native_features(padded, feature_row, CANDIDATE, torch)
                            require(torch.equal(feature[shape["shift_frames"]:], feature0), "ONSET_ORIGINAL_FEATURE_SUFFIX_CHANGED")
                            name = "numeric/ms-%03d/%s.pt" % (offset, row["source_id"])
                            if offset:
                                require(status["started_forward_calls"] < job["new_forward_limit"], "ONSET_FORWARD_BUDGET")
                                call = {"source_id": row["source_id"], "offset_ms": offset, "feature_sha256": training.tensor_sha(feature)}
                                training.append_json(output / "forward-calls.jsonl", {**call, "phase": "started"})
                                status["started_forward_calls"] += 1
                                logits = model(feature.unsqueeze(0))
                                status["completed_forward_calls"] += 1
                                training.append_json(output / "forward-calls.jsonl", {**call, "phase": "completed"})
                                artifact = evaluation.save_numeric(output / name, row["source_id"], feature, logits)
                            else:
                                logits = saved["logits"]
                                training.copy_once(original_numeric, output / name)
                                artifact = io.file_record(output / name)
                                status["baseline_replays"] += 1
                            item = observation(row, pcm, offset, feature, logits, saved["logits"], feature0, history, {"path": name, **artifact})
                            observed.append(item)
                            training.append_json(output / "observations.jsonl", item)
                require(status["started_forward_calls"] == status["completed_forward_calls"] == 16 and
                    status["baseline_replays"] == 8 and sources() == job["source_hashes"] and
                    state_identity(model.state_dict()) == checkpoint["float_state_identity"], "ONSET_FINAL_STATE_OR_COUNTS_CHANGED")
                validate_recipe(job)
                now = resource.getrusage(resource.RUSAGE_SELF)
                status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                    user_cpu_seconds=now.ru_utime - usage.ru_utime, system_cpu_seconds=now.ru_stime - usage.ru_stime,
                    peak_rss_kib=now.ru_maxrss, checkpoint_sha256=checkpoint_sha, model_state_unchanged=True,
                    conditions=summarize(observed), receptive_history_frames=history,
                    interpretation="Descriptive paired onset diagnostic. No onset-trimming, gain rule, label decision, model selection or generalization conclusion.")
                training.save_json(output / "status.json", status)
                io.write_json(output / "summary.json", status)
                io.write_json(output / "MANIFEST.json", {p.relative_to(output).as_posix(): io.file_record(p)
                    for p in sorted(output.rglob("*")) if p.is_file()})
                return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), error_type=type(error).__name__, error=str(error),
                wall_seconds=time.monotonic() - started, counts_are_lower_bounds=True)
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected):
    """Recompute every saved condition without constructing/calling a model."""
    output = io.safe_path(output)
    verify_manifest(output, expected)
    job = training.read_json(output / "request.json")
    with training.heavy_model_slot(job["heavy_lock"]):
        validate_recipe(job)
        root, prepared, _, record, _, rows, original_summary, original_after, _ = origin_inputs(
            job["origin"], job["origin_manifest_sha256"], job["prior_audit"]["path"])
        summary = training.read_json(output / "summary.json")
        require(summary["status"] == "completed" and summary["model_constructions"] == 1 and
            summary["started_forward_calls"] == summary["completed_forward_calls"] == 16 and
            summary["baseline_replays"] == 8 and summary["model_state_unchanged"] is True and
            summary["recipe_sha256"] == training.sha_file(output / "request.json") and
            training.sha_file(output / "fixed-checkpoint.pt") == summary["checkpoint_sha256"] == job["checkpoint"]["sha256"],
            "ONSET_SAVED_SUMMARY_BINDING")
        validate_contract(summary)
        require(training.read_json(output / "selected-rows.json") == rows and
            {name: training.sha_file(output / "execution-source" / name) for name in CODE} == job["source_hashes"],
            "ONSET_RETAINED_INPUT_OR_SOURCE_CHANGED")
        runtime = training.research_runtime.load_manifest()
        training.research_runtime.check_interpreter(runtime)
        training.research_runtime.check_packages(runtime, "core")
        torch, _ = training.torch_runtime()
        saved_rows = [io.loads(line) for line in (output / "observations.jsonl").read_bytes().splitlines()]
        require(len(saved_rows) == 24, "ONSET_SAVED_OBSERVATION_COUNT")
        replayed, calls = [], []
        description = record["initial_checkpoints"][CANDIDATE]["description"]
        history = sum((k - 1) * d for k, d in zip(description["kernels"], description["dilations"]))
        for index, (row, original) in enumerate(zip(rows, original_after)):
            pcm = evaluation.read_pcm(prepared, row)
            baseline_path = root / io.relative(original["numeric_artifact"]["path"])
            baseline_saved = torch.load(baseline_path, weights_only=True, map_location="cpu")
            for ordinal, offset in enumerate(OFFSETS):
                item = saved_rows[index * 3 + ordinal]
                artifact = item["numeric_artifact"]
                require(artifact["path"] == "numeric/ms-%03d/%s.pt" % (offset, row["source_id"]) and
                    io.file_record(output / io.relative(artifact["path"])) == {k: artifact[k] for k in ("bytes", "sha256")},
                    "ONSET_SAVED_NUMERIC_PIN")
                saved = torch.load(output / artifact["path"], weights_only=True, map_location="cpu")
                padded, shape = prepend_pcm(pcm, offset)
                feature_row = {**row, "expected_output_frames": {**row["expected_output_frames"], "c1": shape["frames"]}}
                feature = evaluation.native_features(padded, feature_row, CANDIDATE, torch)
                require(saved["source_id"] == row["source_id"] and torch.equal(feature, saved["feature"]) and
                    training.tensor_sha(feature) == saved["feature_sha256"] and
                    training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                    (offset != 0 or io.file_record(output / artifact["path"]) == io.file_record(baseline_path)),
                    "ONSET_SAVED_NUMERIC_CHANGED")
                actual = observation(row, pcm, offset, feature, saved["logits"], baseline_saved["logits"],
                    baseline_saved["feature"], history, artifact)
                require(actual == item, "ONSET_SAVED_METRICS_CHANGED")
                replayed.append(actual)
                if offset:
                    calls.extend({"source_id": row["source_id"], "offset_ms": offset,
                        "feature_sha256": saved["feature_sha256"], "phase": phase} for phase in ("started", "completed"))
        require([io.loads(line) for line in (output / "forward-calls.jsonl").read_bytes().splitlines()] == calls and
            summarize(replayed) == summary["conditions"] and
            fit.summarize([r for r in replayed if r["offset_ms"] == 0]) == original_summary["after"],
            "ONSET_CALL_LEDGER_OR_SUMMARY_CHANGED")
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected,
            "new_model_forwards": 0, "new_model_constructions": 0, "optimizer_steps": 0,
            "saved_conditions_verified": 24, "new_forward_receipts_verified": 16}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    declaration = sub.add_parser("declare")
    for name in ("origin", "origin-manifest-sha256", "prior-audit", "output"):
        declaration.add_argument("--" + name, required=True)
    worker = sub.add_parser("run")
    for name in ("recipe", "output", "heavy-lock"):
        worker.add_argument("--" + name, required=True)
    check = sub.add_parser("audit")
    for name in ("output", "manifest-sha256", "receipt"):
        check.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        result = make_recipe(args.origin, args.origin_manifest_sha256, args.prior_audit)
        io.write_json(Path(args.output), result)
        result = {"recipe": args.output, "sha256": training.sha_file(args.output), "model_calls": 0,
            "checkpoint_sha256": result["checkpoint"]["sha256"], **fixed_contract()}
    elif args.command == "run":
        result = run(Path(args.recipe), Path(args.output), Path(args.heavy_lock))
    else:
        result = audit(args.output, args.manifest_sha256)
        io.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
