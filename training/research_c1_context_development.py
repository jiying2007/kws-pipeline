#!/usr/bin/env python3
"""One read-only comparison of two fixed C1 models on the exposed context dev18.

The frozen context consumer supplies every development row, without filtering.
Both checkpoints see 0/200/500 ms digital prefixes once: 108 forwards, no
optimizer, decoder threshold, ASR/TTS, retry or independent-test claim.
Declaration and contract checks import no tensor runtime.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_context_v3_consume as context
import research_c1_known_silence as silence

onset, training, fit, evaluation = silence.onset, silence.training, silence.fit, silence.evaluation
io, require = onset.io, onset.require
SCHEMA = "kws-c1-context-development-diagnostic-v1"
SOURCE = "training/research_c1_context_development.py"
CODE = tuple(dict.fromkeys((*context.CODE, *silence.CODE, SOURCE)))
MODELS = ("original-tiny-fit", "known-silence-corrected")
JOB_ID = "c1-context-development18"
SECONDS, LIMIT = 540, 108
OUTPUTS = ("summary.json", "observations.jsonl", "forward-calls.jsonl", "development-rows.json", "MANIFEST.json")


def fixed_contract():
    return {"candidate": onset.CANDIDATE, "model_order": list(MODELS), "development_rows": 18,
        "offsets_ms": list(onset.OFFSETS), "forward_limit": LIMIT, "model_construction_limit": 2,
        "worker_seconds": SECONDS, "queue_minutes": 10, "optimizer_steps": 0, "training_forwards": 0,
        "asr_or_tts_calls": 0, "automatic_retry": False, "reset_each_condition": True,
        "row_order": "unchanged frozen context72 consumer development order",
        "labels_changed": False, "full_float32_logits_retained": True, "human_gold": False,
        "previously_exposed_development": True, "independent_test": False,
        "development_selection_or_tuning": False, "threshold_selection": False,
        "quality_winner": None, "generalization_passed": None,
        "keyword_measure": "complete contiguous token sequence in greedy CTC, not a wake-event decoder",
        "known_silence": "fully added PCM16-zero windows only; no decisions on original quiet speech",
        "scope": "paired descriptive development diagnostic, not independent final evaluation"}


def validate_contract(job):
    require(type(job) is dict and job.get("schema") == SCHEMA and
        all(type(job.get(k)) is type(v) and job[k] == v for k, v in fixed_contract().items()) and
        all(type(x) is int for x in job["offsets_ms"]), "CONTEXT_DIAGNOSTIC_FIXED_CONTRACT_CHANGED")


def sources():
    return {name: training.sha_file(ROOT / name) for name in CODE}


def completed_output(path, expected, audit_pin):
    """Read saved completion/audit identities; never rerun the prior worker."""
    root = io.safe_path(path)
    onset.verify_manifest(root, expected)
    audit = training.read_json(context.checked(audit_pin))
    require(audit["status"] == "verified" and audit["manifest_sha256"] == expected and
        audit["new_model_forwards"] == audit["new_optimizer_steps"] == 0,
        "CONTEXT_DIAGNOSTIC_PRIOR_AUDIT_REQUIRED")
    receipt = training.read_json(root.parent / "result.json")
    queue = root.parents[2] / "queue.json"
    require(receipt["schema"] == io.RECEIPT_SCHEMA and receipt["status"] == "completed" and
        receipt["exit_code"] == 0 and receipt["job_id"] == root.parent.name and
        receipt["queue_sha256"] == training.sha_file(queue) and
        receipt["recipe_sha256"] == training.sha_file(root / "request.json") and
        receipt["outputs"]["MANIFEST.json"] == io.file_record(root / "MANIFEST.json"),
        "CONTEXT_DIAGNOSTIC_COMPLETED_QUEUE_REQUIRED")
    job, summary = training.read_json(root / "request.json"), training.read_json(root / "summary.json")
    require(summary["status"] == "completed" and
        summary.get("job_recipe_sha256", summary.get("recipe_sha256")) == receipt["recipe_sha256"],
        "CONTEXT_DIAGNOSTIC_PRIOR_SUMMARY_CHANGED")
    return root, job, summary, training.read_json(queue)["heavy_lock"]


def validate_rows(rows, vocabulary, trained_rows):
    require(len(rows) == 18 and len({r["source_id"] for r in rows}) == 18,
        "CONTEXT_DIAGNOSTIC_ALL_EIGHTEEN_REQUIRED")
    trained = set().union(*(evaluation.identity_keys(r) for r in trained_rows))
    for row in rows:
        require(io.ID.fullmatch(row["source_id"]) and row["split"] == "development" and
            row["source_role"] == "development-only" and row["ctc_mask"] is True and
            type(row["ctc_weight"]) in (int, float) and math.isfinite(row["ctc_weight"]) and
            0 < row["ctc_weight"] <= 1 and row["quarantined"] is False and
            row["generation_outcome"] == "completed" and row["human_gold"] is False and
            row["independent_test"] is False, "CONTEXT_DIAGNOSTIC_DEVELOPMENT_LABEL_SCOPE")
        projection = fit.fulltext_targets(row["full_text"], vocabulary)
        require(projection["supported"] and projection["target_ids"] == row["target_ids"] and
            row["vocabulary_sha256"] == fit.digest(vocabulary) and
            row["expected_c1_frames"] == onset.geometry(row["samples"], 0)["frames"] and
            row["expected_c1_frames"] >= row["ctc_minimum_frames"],
            "CONTEXT_DIAGNOSTIC_COMPLETE_TRANSCRIPT_REQUIRED")
        require(len(row["keyword_targets"]) == len(row["keyword_mask"]) == len(row["keyword_weight"]) == 2,
            "CONTEXT_DIAGNOSTIC_KEYWORD_MASK_SHAPE")
        for target, mask, weight, text in zip(row["keyword_targets"], row["keyword_mask"],
                                            row["keyword_weight"], fit.KEYWORDS):
            require(type(mask) is bool and type(weight) in (int, float) and math.isfinite(weight) and
                ((type(target) is int and target == int(text in row["full_text"]) and 0 < weight <= 1)
                 if mask else target is None and weight == 0), "CONTEXT_DIAGNOSTIC_UNKNOWN_IS_NOT_NEGATIVE")
        require(not evaluation.identity_keys(row) & trained, "CONTEXT_DIAGNOSTIC_TRAIN_SOURCE_OVERLAP")


FROZEN_METADATA = r'''
import json, sys
from pathlib import Path
kind, source, output = sys.argv[1:]
sys.path.insert(0, str(Path(source) / "training"))
if kind == "context":
    import research_context_v3_consume as m
    job = m.training.read_json(Path(output) / "request.json")
    m.validate_job(job)
    record, vocabulary, all_rows, train, dev, _ = m.verify_prepared(job["prepared"], job["prepared_manifest_sha256"])
    result = {"record": record, "vocabulary": vocabulary, "observed_rows": len(all_rows), "train": train, "dev": dev}
else:
    import research_c1_known_silence as m
    job = m.training.read_json(Path(output) / "request.json")
    m.validate_recipe(job)
    _, original, _, _ = m.reference_inputs(job["reference_output"], job["reference_manifest_sha256"], job["reference_audit"]["path"])
    origin, _, control, record, vocabulary, rows, summary, _, lock = original
    result = {"origin": str(origin), "control": control, "record": record, "vocabulary": vocabulary,
              "rows": rows, "summary": summary, "heavy_lock": lock}
if "torch" in sys.modules:
    raise ValueError("CONTEXT_DIAGNOSTIC_METADATA_IMPORTED_TORCH")
print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
'''


def frozen_sources(kind, source, expected):
    """Preserve original verifiers; require unchanged numerical source/runtime."""
    require(set(expected) == set(context.CODE if kind == "context" else silence.CODE) and
        {name: training.sha_file(source / name) for name in expected} == expected,
        "CONTEXT_DIAGNOSTIC_FROZEN_VERIFIER_CHANGED")
    for name, sha in expected.items():
        if sha == training.sha_file(ROOT / name):
            continue
        if name == "configs/runtime/research-runtime.json":
            old, current = training.read_json(source / name), training.read_json(ROOT / name)
            old.pop("validation", None); current.pop("validation", None)
            require(old == current, "CONTEXT_DIAGNOSTIC_NUMERIC_RUNTIME_CHANGED")
        else:
            # This saved helper seals/replays metadata. Its original bytes are
            # executed unchanged; no algorithm or projection source may drift.
            require(kind == "context" and name == "tools/research_d2_checkpoints.py",
                "CONTEXT_DIAGNOSTIC_NUMERIC_OR_READER_SOURCE_CHANGED:" + name)


def frozen_inputs(kind, output, job):
    source = (io.safe_path(job["prepared"]) if kind == "context" else output) / "execution-source"
    if kind == "context":
        onset.verify_manifest(source.parent, job["prepared_manifest_sha256"])
    frozen_sources(kind, source, job["source_hashes"])
    child = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", FROZEN_METADATA,
        kind, str(source), str(output)], cwd=source, capture_output=True, timeout=120)
    require(child.returncode == 0, "CONTEXT_DIAGNOSTIC_FROZEN_METADATA_FAILED:" + kind + ":" +
        child.stderr.decode("utf-8", errors="replace")[-1500:])
    frozen_sources(kind, source, job["source_hashes"])
    require(len(child.stdout) <= 16 * 1024 * 1024, "CONTEXT_DIAGNOSTIC_METADATA_SIZE")
    return io.loads(child.stdout)


def assemble(request):
    require(type(request) is dict and set(request) == {"context_output", "context_manifest_sha256", "context_audit",
        "correction_output", "correction_manifest_sha256", "correction_audit"}, "CONTEXT_DIAGNOSTIC_INPUT_REQUEST")
    croot, cjob, csummary, clock = completed_output(request["context_output"],
        request["context_manifest_sha256"], request["context_audit"])
    prepared = io.safe_path(cjob["prepared"])
    c = frozen_inputs("context", croot, cjob)
    record, vocabulary, train, dev = c["record"], c["vocabulary"], c["train"], c["dev"]
    require(c["observed_rows"] == 72 and dev == training.read_json(croot / "development-rows.json") and
        len(dev) == csummary["development_rows"] == 18 and csummary["development_optimizer_draws"] == 0 and
        fit.digest([fit.row_identity(r) for r in dev]) == cjob["development_binding_sha256"] and
        fit.digest([fit.row_identity(r) for r in train]) == cjob["train_binding_sha256"] and
        record["selection"]["sha256"] == cjob["selection_sha256"], "CONTEXT_DIAGNOSTIC_FROZEN_DEVELOPMENT_CHANGED")
    sroot, sjob, ssummary, slock = completed_output(request["correction_output"],
        request["correction_manifest_sha256"], request["correction_audit"])
    s = frozen_inputs("correction", sroot, sjob)
    origin, control, old_record, old_vocab = Path(s["origin"]), s["control"], s["record"], s["vocabulary"]
    trained_rows, old_summary, lock = s["rows"], s["summary"], s["heavy_lock"]
    require(vocabulary == old_vocab and record["initial_checkpoint"]["file_sha256"] == silence.INITIAL_SHA and
        clock == slock == lock == sjob["heavy_lock"] and
        ssummary["actual_optimizer_steps"] == 2000 and
        ssummary["training_forwards_started"] == ssummary["training_forwards_completed"] == 8000 and
        ssummary["final_forwards_started"] == ssummary["final_forwards_completed"] == 24 and
        training.read_json(sroot / "selected-rows.json") == trained_rows,
        "CONTEXT_DIAGNOSTIC_SAME_FIXED_ORIGIN_REQUIRED")
    validate_rows(dev, vocabulary, trained_rows + train)
    models = []
    for ident, folder, schema, summary, recipe_sha in (
            (MODELS[0], origin, fit.RESULT_SCHEMA, old_summary, training.sha_file(origin / "request.json")),
            (MODELS[1], sroot, silence.SCHEMA, ssummary, training.sha_file(sroot / "request.json"))):
        checkpoint = onset.pin(folder / "final.pt")
        require(checkpoint["sha256"] == summary["final_checkpoint_sha256"] and
            summary["initial_checkpoint_sha256"] == silence.INITIAL_SHA, "CONTEXT_DIAGNOSTIC_CHECKPOINT_PIN")
        models.append({"model_id": ident, "checkpoint": checkpoint, "checkpoint_schema": schema,
            "training_recipe_sha256": recipe_sha, "float_state_sha256": summary["final_float_state_sha256"]})
    job = {"schema": SCHEMA, **fixed_contract(), "inputs": request, "prepared": str(prepared),
        "prepared_manifest_sha256": cjob["prepared_manifest_sha256"], "selection_sha256": cjob["selection_sha256"],
        "development": dev, "development_binding_sha256": cjob["development_binding_sha256"],
        "original_training_source_ids": control["source_ids"], "original_training_binding_sha256": control["row_binding_sha256"],
        "models": models, "model_description": old_record["initial_checkpoints"][onset.CANDIDATE]["description"],
        "frozen_input_verifiers": {"context": cjob["source_hashes"], "correction": sjob["source_hashes"]},
        "vocabulary_sha256": fit.digest(vocabulary), "vocabulary_file": onset.pin(prepared / "vocabulary.json"),
        "tokens_file": onset.pin(prepared / "tokens.txt"), "heavy_lock": lock, "source_hashes": sources(),
        "logits_payload_bytes": 2 * sum(onset.geometry(r["samples"], ms)["frames"] * 2601 * 4
            for r in dev for ms in onset.OFFSETS)}
    return job, vocabulary


def validate_recipe(job):
    validate_contract(job)
    rebuilt, vocabulary = assemble(job["inputs"])
    require(job == rebuilt, "CONTEXT_DIAGNOSTIC_RECIPE_OR_INPUT_CHANGED")
    return vocabulary


def keyword_observations(hypothesis, row, vocabulary):
    result = []
    for i, text in enumerate(fit.KEYWORDS):
        target = fit.fulltext_targets(text, vocabulary)["target_ids"]
        found = any(hypothesis[j:j + len(target)] == target for j in range(len(hypothesis) - len(target) + 1))
        result.append({"keyword_id": i + 1, "text": text, "token_ids": target,
            "known": row["keyword_mask"][i], "reference": row["keyword_targets"][i],
            "complete_sequence_present": found})
    return result


def observation(row, model, pcm, offset, feature, logits, baseline, baseline_feature, history, artifact, vocabulary):
    item = onset.observation(row, pcm, offset, feature, logits, baseline, baseline_feature, history, artifact)
    return {**item, "new_model_forward": True, "model_id": model["model_id"],
        "checkpoint_sha256": model["checkpoint"]["sha256"], "split": row["split"], "source_role": row["source_role"],
        "source_family_id": row["source_family_id"], "speaker_id": row["speaker_id"],
        "keyword_targets": row["keyword_targets"], "keyword_mask": row["keyword_mask"],
        "keyword_weight": row["keyword_weight"], "full_float32_logits_retained": True,
        "keywords": keyword_observations(item["greedy_token_ids"], row, vocabulary),
        **silence.prefix_observation(logits, item["fully_digital_silence_frames"])}


def summarize(observations, job):
    expected = [(model, row["source_id"], ms) for model in MODELS for row in job["development"] for ms in onset.OFFSETS]
    require([(x["model_id"], x["source_id"], x["offset_ms"]) for x in observations] == expected and len(expected) == LIMIT,
        "CONTEXT_DIAGNOSTIC_ALL_CONDITIONS_REQUIRED")
    result = {}
    for model in MODELS:
        result[model] = {}
        for ms in onset.OFFSETS:
            group = [x for x in observations if x["model_id"] == model and x["offset_ms"] == ms]
            words = []
            for i in range(2):
                known = [x["keywords"][i] for x in group if x["keywords"][i]["known"]]
                positive = [x for x in known if x["reference"] == 1]
                negative = [x for x in known if x["reference"] == 0]
                words.append({"keyword_id": i + 1, "known_positive_rows": len(positive),
                    "matched_positive_rows": sum(x["complete_sequence_present"] for x in positive),
                    "known_negative_rows": len(negative),
                    "false_positive_rows": sum(x["complete_sequence_present"] for x in negative),
                    "unknown_rows": len(group) - len(known)})
            result[model][str(ms)] = {**fit.summarize(group), "keywords": words,
                "known_zero_frames": sum(x["known_zero_prefix_frames"] for x in group),
                "known_zero_nonblank_frames": sum(x["known_zero_nonblank_frames"] for x in group),
                "emissions_entirely_in_added_silence": sum(x["emissions_entirely_in_added_silence"] for x in group)}
    return result


@contextmanager
def deadline():
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "CONTEXT_DIAGNOSTIC_EXISTING_TIMER")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise TimeoutError("CONTEXT_DIAGNOSTIC_FIXED_540_SECOND_DEADLINE")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, SECONDS)
    try:
        yield time.monotonic() + SECONDS
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def features(pcm, row, offset, torch):
    padded, shape = onset.prepend_pcm(pcm, offset)
    feature = evaluation.native_features(padded, {**row, "expected_output_frames": {"c1": shape["frames"]}}, onset.CANDIDATE, torch)
    require(bool((feature[:shape["fully_digital_silence_frames"]] == 0).all()), "CONTEXT_DIAGNOSTIC_KNOWN_ZERO_FEATURES")
    return feature


def check_logits(logits, feature, torch):
    require(logits.dtype == torch.float32 and tuple(logits.shape) == (len(feature), 1, 2601) and
        bool(logits.isfinite().all()), "CONTEXT_DIAGNOSTIC_FULL_FLOAT32_LOGITS_REQUIRED")


def checkpoint_state(path, model, job):
    from training_state import load_checkpoint_snapshot, state_identity
    saved, sha = load_checkpoint_snapshot(path)
    require(sha == model["checkpoint"]["sha256"] and saved["schema"] == model["checkpoint_schema"] and
        saved["candidate"] == onset.CANDIDATE and saved["optimizer_steps"] == 2000 and
        saved["initial_checkpoint_sha256"] == silence.INITIAL_SHA and saved["vocabulary_sha256"] == job["vocabulary_sha256"] and
        saved["job_recipe_sha256"] == model["training_recipe_sha256"] and
        state_identity(saved["state_dict"]) == saved["float_state_identity"] and
        saved["float_state_identity"]["sha256"] == model["float_state_sha256"], "CONTEXT_DIAGNOSTIC_FIXED_STATE_CHANGED")
    return saved


def run(recipe_path, output, lock):
    job = training.read_json(recipe_path)
    validate_contract(job)
    require(str(io.safe_path(lock)) == job["heavy_lock"], "CONTEXT_DIAGNOSTIC_CANONICAL_LOCK_REQUIRED")
    with training.heavy_model_slot(lock) as lease:
        training.queue_binding(recipe_path, lease)
        require(os.environ.get("KWS_RESEARCH_JOB_ID") == JOB_ID and
            os.environ.get("KWS_RESEARCH_RECIPE") == str(Path(recipe_path).absolute()) and
            os.environ.get("KWS_RESEARCH_OUTPUT_DIR") == str(Path(output).absolute()), "CONTEXT_DIAGNOSTIC_QUEUE_SCOPE")
        output = training.output_directory(output)
        started = time.monotonic()
        status = {"schema": SCHEMA, **fixed_contract(), "status": "running", "started_at": training.utc(),
            "recipe_sha256": training.sha_file(recipe_path), "model_constructions": 0,
            "started_forward_calls": 0, "completed_forward_calls": 0}
        training.save_json(output / "status.json", status)
        try:
            with deadline() as end:
                vocabulary = validate_recipe(job)
                runtime = training.research_runtime.load_manifest()
                training.research_runtime.check_interpreter(runtime)
                training.research_runtime.check_packages(runtime, "core")
                torch, reference = training.torch_runtime()
                from training_state import state_identity
                rows, prepared = job["development"], Path(job["prepared"])
                training.copy_once(recipe_path, output / "request.json")
                io.write_json(output / "development-rows.json", rows)
                for name in CODE:
                    training.copy_once(ROOT / name, output / "execution-source" / name)
                for name in ("vocabulary.json", "tokens.txt"):
                    training.copy_once(prepared / name, output / name)
                for row in rows:
                    training.copy_once(prepared / row["audio"], output / row["audio"])
                observed, shared_features = [], {}
                history = sum((k - 1) * d for k, d in zip(job["model_description"]["kernels"], job["model_description"]["dilations"]))
                with torch.inference_mode():
                    for model_pin in job["models"]:
                        model_id = model_pin["model_id"]
                        saved = checkpoint_state(model_pin["checkpoint"]["path"], model_pin, job)
                        training.copy_once(model_pin["checkpoint"]["path"], output / "checkpoints" / (model_id + ".pt"))
                        torch.manual_seed(fit.SEED)
                        model, description = training.build_fresh(onset.CANDIDATE, vocabulary, reference)
                        status["model_constructions"] += 1
                        require(description == job["model_description"], "CONTEXT_DIAGNOSTIC_ARCHITECTURE_CHANGED")
                        model.load_state_dict(saved["state_dict"], strict=True)
                        model.eval().requires_grad_(False)
                        require(state_identity(model.state_dict()) == saved["float_state_identity"], "CONTEXT_DIAGNOSTIC_LOADED_STATE_CHANGED")
                        for row in rows:
                            pcm = evaluation.read_pcm(output, row)
                            baseline = baseline_feature = None
                            for offset in onset.OFFSETS:
                                require(time.monotonic() < end and status["started_forward_calls"] < LIMIT, "CONTEXT_DIAGNOSTIC_FORWARD_BUDGET")
                                feature = features(pcm, row, offset, torch)
                                feature_sha = training.tensor_sha(feature)
                                require(shared_features.setdefault((row["source_id"], offset), feature_sha) == feature_sha,
                                    "CONTEXT_DIAGNOSTIC_PAIRED_FEATURES_CHANGED")
                                call = {"model_id": model_id, "checkpoint_sha256": model_pin["checkpoint"]["sha256"],
                                    "source_id": row["source_id"], "offset_ms": offset,
                                    "label_binding_sha256": fit.row_identity(row), "feature_sha256": feature_sha}
                                training.append_json(output / "forward-calls.jsonl", {**call, "phase": "started"})
                                status["started_forward_calls"] += 1
                                logits = model(feature.unsqueeze(0))
                                status["completed_forward_calls"] += 1
                                check_logits(logits, feature, torch)
                                training.append_json(output / "forward-calls.jsonl", {**call, "phase": "completed", "logits_sha256": training.tensor_sha(logits)})
                                if not offset:
                                    baseline, baseline_feature = logits, feature
                                name = "numeric/%s/ms-%03d/%s.pt" % (model_id, offset, row["source_id"])
                                artifact = {"path": name, **evaluation.save_numeric(output / name, row["source_id"], feature, logits)}
                                item = observation(row, model_pin, pcm, offset, feature, logits, baseline, baseline_feature, history, artifact, vocabulary)
                                observed.append(item)
                                training.append_json(output / "observations.jsonl", item)
                        require(state_identity(model.state_dict()) == saved["float_state_identity"], "CONTEXT_DIAGNOSTIC_EVALUATION_CHANGED_MODEL")
                        del model, saved
                require(status["model_constructions"] == 2 and status["started_forward_calls"] == status["completed_forward_calls"] == LIMIT,
                    "CONTEXT_DIAGNOSTIC_TERMINAL_COUNTS")
                validate_recipe(job)
                require(time.monotonic() < end, "CONTEXT_DIAGNOSTIC_TERMINAL_DEADLINE")
                status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                    model_states_unchanged=True, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    conditions=summarize(observed, job))
                training.save_json(output / "status.json", status)
                io.write_json(output / "summary.json", status)
                context.manifest(output)
                return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), error_type=type(error).__name__, error=str(error),
                wall_seconds=time.monotonic() - started, counts_are_lower_bounds=True)
            training.save_json(output / "status.json", status)
            raise


def audit(output, expected):
    """Rescore full retained tensors; no model construction, forward or optimizer."""
    output = io.safe_path(output)
    onset.verify_manifest(output, expected)
    job, summary = training.read_json(output / "request.json"), training.read_json(output / "summary.json")
    with training.heavy_model_slot(job["heavy_lock"]):
        vocabulary = validate_recipe(job)
        validate_contract(summary)
        require(summary["status"] == "completed" and summary["recipe_sha256"] == training.sha_file(output / "request.json") and
            summary["model_constructions"] == 2 and summary["started_forward_calls"] == summary["completed_forward_calls"] == LIMIT and
            summary["model_states_unchanged"] is True and training.read_json(output / "development-rows.json") == job["development"] and
            {name: training.sha_file(output / "execution-source" / name) for name in CODE} == job["source_hashes"],
            "CONTEXT_DIAGNOSTIC_SAVED_BINDINGS")
        runtime = training.research_runtime.load_manifest()
        training.research_runtime.check_interpreter(runtime)
        training.research_runtime.check_packages(runtime, "core")
        torch, _ = training.torch_runtime()
        observations = context.lines(output / "observations.jsonl")
        require(len(observations) == LIMIT, "CONTEXT_DIAGNOSTIC_SAVED_COUNT")
        history = sum((k - 1) * d for k, d in zip(job["model_description"]["kernels"], job["model_description"]["dilations"]))
        replayed, calls, shared_features = [], [], {}
        for model_pin in job["models"]:
            checkpoint_state(output / "checkpoints" / (model_pin["model_id"] + ".pt"), model_pin, job)
            for row in job["development"]:
                pcm = evaluation.read_pcm(output, row)
                baseline = baseline_feature = None
                for offset in onset.OFFSETS:
                    item = observations[len(replayed)]
                    artifact = item["numeric_artifact"]
                    name = "numeric/%s/ms-%03d/%s.pt" % (model_pin["model_id"], offset, row["source_id"])
                    require(artifact == {"path": name, **io.file_record(output / name)}, "CONTEXT_DIAGNOSTIC_SAVED_TENSOR_PIN")
                    saved = torch.load(output / name, weights_only=True, map_location="cpu")
                    feature = features(pcm, row, offset, torch)
                    check_logits(saved["logits"], feature, torch)
                    require(saved["source_id"] == row["source_id"] and feature.equal(saved["feature"]) and
                        training.tensor_sha(feature) == saved["feature_sha256"] and
                        training.tensor_sha(saved["logits"]) == saved["logits_sha256"] and
                        shared_features.setdefault((row["source_id"], offset), saved["feature_sha256"]) == saved["feature_sha256"],
                        "CONTEXT_DIAGNOSTIC_SAVED_NUMERIC_CHANGED")
                    if not offset:
                        baseline, baseline_feature = saved["logits"], feature
                    actual = observation(row, model_pin, pcm, offset, feature, saved["logits"], baseline, baseline_feature, history, artifact, vocabulary)
                    require(actual == item, "CONTEXT_DIAGNOSTIC_SAVED_METRICS_CHANGED")
                    replayed.append(actual)
                    call = {"model_id": model_pin["model_id"], "checkpoint_sha256": model_pin["checkpoint"]["sha256"],
                        "source_id": row["source_id"], "offset_ms": offset, "label_binding_sha256": fit.row_identity(row),
                        "feature_sha256": saved["feature_sha256"]}
                    calls.extend(({**call, "phase": "started"}, {**call, "phase": "completed", "logits_sha256": saved["logits_sha256"]}))
        require(calls == context.lines(output / "forward-calls.jsonl") and summarize(replayed, job) == summary["conditions"],
            "CONTEXT_DIAGNOSTIC_SAVED_LEDGER_OR_SUMMARY_CHANGED")
        return {"schema": SCHEMA, "status": "verified", "manifest_sha256": expected, "saved_conditions_verified": LIMIT,
            "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0, "independent_test": False}


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
    declare = sub.add_parser("declare")
    for name in ("context-output", "context-manifest-sha256", "context-audit", "correction-output",
                 "correction-manifest-sha256", "correction-audit", "output"):
        declare.add_argument("--" + name, required=True)
    for command, names in (("run", ("recipe", "output", "heavy-lock")),
                           ("audit", ("output", "manifest-sha256", "receipt")),
                           ("queue-request", ("recipe", "python", "output"))):
        subparser = sub.add_parser(command)
        for name in names:
            subparser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "declare":
        request = {k: str(io.safe_path(getattr(args, k))) if k.endswith("_output") else getattr(args, k)
                   for k in ("context_output", "context_manifest_sha256", "correction_output", "correction_manifest_sha256")}
        request.update(context_audit=onset.pin(args.context_audit), correction_audit=onset.pin(args.correction_audit))
        job, _ = assemble(request)
        io.write_json(Path(args.output), job)
        result = {"recipe": args.output, "sha256": training.sha_file(args.output), "new_model_calls": 0,
            "development_rows": len(job["development"]), "forward_limit": LIMIT,
            "checkpoint_sha256": [m["checkpoint"]["sha256"] for m in job["models"]], "logits_payload_bytes": job["logits_payload_bytes"]}
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
