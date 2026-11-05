#!/usr/bin/env python3
"""Read-only D1 development CTC diagnostics under the existing bounded queue.

The two held-out source families are declared before execution. Each job reads
one completed training dependency and the same sealed preparation; it does not
update model state, optimize a threshold, or select a candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_d1_train as training
from research_d1_recipe import project_row, row_identity
from research_d1_vocab import OUTPUTS, digest, require
from research_job_lock import heavy_model_slot

SCHEMA = "kws-d1-development-evaluation-job-v2"
RESULT_SCHEMA = "kws-d1-development-evaluation-result-v2"
FAMILY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SOURCE_PATH = "training/research_d1_evaluate.py"
QUEUE_OUTPUTS = ("summary.json", "observations.jsonl", "MANIFEST.json")


def validate_job(value):
    require(type(value) is dict and set(value) == {
        "schema", "prepared_from_dependency", "training_from_dependency", "candidate", "near_percent",
        "held_out_source_family_ids", "evaluator_source_sha256", "timeout_seconds"} and value["schema"] == SCHEMA,
        "D1_DEV_JOB_SCHEMA")
    for key in ("prepared_from_dependency", "training_from_dependency"):
        require(type(value[key]) is str and training.io_contract.ID.fullmatch(value[key]), "D1_DEV_DEPENDENCY_ID")
    require(value["prepared_from_dependency"] != value["training_from_dependency"] and
            value["candidate"] in training.CANDIDATES and type(value["near_percent"]) is int and
            value["near_percent"] in (0, 5, 10), "D1_DEV_CANDIDATE_OR_DEPENDENCIES")
    families = value["held_out_source_family_ids"]
    require(type(families) is list and len(families) == 2 and
            all(type(name) is str and FAMILY_ID.fullmatch(name) for name in families) and
            families == sorted(set(families)), "D1_DEV_TWO_PREDECLARED_FAMILIES")
    require(type(value["evaluator_source_sha256"]) is str and
            training.io_contract.SHA.fullmatch(value["evaluator_source_sha256"]), "D1_DEV_SOURCE_PIN")
    seconds = value["timeout_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 999 * 60,
            "D1_DEV_EXPLICIT_TIMEOUT")
    return value


def completed_training(job, *, dependencies_path=None):
    path = dependencies_path or os.environ.get("KWS_RESEARCH_DEPENDENCIES")
    require(path is not None, "D1_DEV_COMPLETED_TRAINING_DEPENDENCY_REQUIRED")
    dependencies = training.read_json(path)
    name = job["training_from_dependency"]
    require(name in dependencies, "D1_DEV_TRAINING_NOT_A_DECLARED_DEPENDENCY")
    dependency = dependencies[name]
    raw = training.io_contract.read_bytes(dependency["receipt"])
    receipt = training.io_contract.loads(raw)
    require(hashlib.sha256(raw).hexdigest() == dependency["receipt_sha256"] and
            receipt["schema"] == "kws-local-research-job-result-v1" and receipt["status"] == "completed" and
            receipt["exit_code"] == 0 and receipt["job_id"] == name and
            receipt["outputs"] == dependency["outputs"], "D1_DEV_TRAINING_RECEIPT_CHANGED")
    output = training.io_contract.safe_path(dependency["outputs_root"])
    for filename in ("final.pt", "summary.json", "consumption.jsonl"):
        require(filename in dependency["outputs"] and
                training.io_contract.file_record(output / filename) == dependency["outputs"][filename],
                "D1_DEV_TRAINING_OUTPUT_CHANGED:" + filename)
    summary = training.read_json(output / "summary.json")
    require(summary["schema"] == "kws-d1-training-result-v2" and summary["status"] == "completed" and
            summary["actual_training_executed"] is True and summary["candidate"] == job["candidate"] and
            summary["near_percent"] == job["near_percent"] and
            summary["job_recipe_sha256"] == receipt["recipe_sha256"] and
            summary["final_checkpoint_sha256"] == dependency["outputs"]["final.pt"]["sha256"] and
            summary["consumption_sha256"] == dependency["outputs"]["consumption.jsonl"]["sha256"],
            "D1_DEV_COMPLETED_TRAINING_SUMMARY")
    return output, summary, {"job_id": name, "receipt_sha256": dependency["receipt_sha256"],
                            "outputs": dependency["outputs"]}


def identity_keys(row):
    """Repeat the retained source-role relation, without inferring speakers."""
    pairs = [("family", row["source_family_id"]), ("speaker", row["speaker_id"])]
    pairs += [("pcm", row.get(key)) for key in ("pcm_sha256", "original_pcm_sha256", "reference_pcm_sha256")]
    pairs += [("file", row.get(key)) for key in
              ("file_sha256", "original_wav_sha256", "reference_wav_sha256", "parent_source_sha256")]
    return {(kind, value) for kind, value in pairs if value is not None}


def development_rows(rows, train_rows, record, vocabulary, families):
    require(type(rows) is list and len(rows) == record["observed_rows"] and
            len({tuple(row["global_source_key"]) for row in rows}) == len(rows) and
            dict(Counter(row["split"] for row in rows)) == record["projected_rows_by_split"],
            "D1_DEV_PREPARED_ROW_ROSTER")
    by_role = {"train": set(), "development": set()}
    for row in rows:
        require(row["consumer"] == "d1-objective-projection-v2" and
                row["vocabulary_sha256"] == digest(vocabulary) and row["split"] in by_role,
                "D1_DEV_PROJECTED_ROW_SCOPE")
        # Replay the already fixed objective projection from its saved original
        # labels, including quarantine and failed-native zero-loss protection.
        original = {**row, "consumer": "d1-verified-full-transcript-v2",
                    "ctc_mask": row["label_ctc_mask"], "ctc_weight": row["label_ctc_weight"],
                    "keyword_targets": row["label_keyword_targets"], "keyword_mask": row["label_keyword_mask"],
                    "keyword_weight": row["label_keyword_weight"]}
        require(project_row(original, vocabulary) == row, "D1_DEV_PROJECTED_LABEL_CHANGED")
        by_role[row["split"]].update(identity_keys(row))
    require(not by_role["train"] & by_role["development"], "D1_DEV_SOURCE_ROLE_LEAK")
    all_train = {row["source_id"]: row for row in rows if row["split"] == "train" and row["ctc_mask"]}
    require(len(all_train) == len(train_rows) and
            all(all_train.get(row["source_id"]) == row for row in train_rows), "D1_DEV_TRAIN_COHORT_CHANGED")
    result = sorted((row for row in rows if row["split"] == "development"),
                    key=lambda row: (row["source_family_id"], row["source_id"]))
    require({row["source_family_id"] for row in result} == set(families), "D1_DEV_HELD_OUT_ROSTER_CHANGED")
    return result


def collapse_ctc(frame_ids):
    result, previous = [], None
    for value in frame_ids:
        require(type(value) is int and 0 <= value < OUTPUTS, "D1_DEV_GREEDY_TOKEN_ID")
        if value != previous and value != 0:
            result.append(value)
        previous = value
    return result


def token_errors(reference, hypothesis):
    """Levenshtein S/D/I; equal-cost paths prefer substitution, delete, insert."""
    require(reference and all(type(value) is int and 0 < value < OUTPUTS
                              for value in [*reference, *hypothesis]), "D1_DEV_EDIT_TARGETS")
    # Each cell is (substitutions, deletions, insertions); only one DP row is kept.
    previous = [(0, 0, index) for index in range(len(hypothesis) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, observed in enumerate(hypothesis, 1):
            diagonal = previous[j - 1]
            candidates = [(diagonal[0] + int(expected != observed), diagonal[1], diagonal[2]),
                          (previous[j][0], previous[j][1] + 1, previous[j][2]),
                          (current[-1][0], current[-1][1], current[-1][2] + 1)]
            current.append(min(candidates, key=sum))
        previous = current
    substitutions, deletions, insertions = previous[-1]
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions,
            "edit_numerator": substitutions + deletions + insertions,
            "target_token_denominator": len(reference), "hypothesis_tokens": len(hypothesis),
            "exact_token_match": reference == hypothesis}


def score_logits(logits, row):
    weighted, normalized = training.fulltext_ctc(logits, row["target_ids"], row["ctc_weight"], True)
    require(bool(logits.isfinite().all()), "D1_DEV_NONFINITE_LOGITS")
    hypothesis = collapse_ctc(logits[:, 0].argmax(dim=-1).tolist())
    return {"greedy_token_ids": hypothesis, **token_errors(row["target_ids"], hypothesis),
            "ctc_loss_target_length_normalized": normalized.item(),
            "ctc_nll": normalized.double().item() * len(row["target_ids"]),
            "ctc_loss_confidence_weighted": weighted.item()}


def summarize(observations):
    scored = [row for row in observations if row["evaluation_status"] == "evaluated"]
    numerator = sum(row["edit_numerator"] for row in scored)
    denominator = sum(row["target_token_denominator"] for row in scored)
    matches = sum(row["exact_token_match"] for row in scored)
    return {"observed_clips": len(observations), "evaluable_clips": len(scored),
            "not_evaluable_reasons": dict(Counter(row["reason"] for row in observations
                                                 if row["evaluation_status"] != "evaluated")),
            "edit_numerator": numerator, "target_token_denominator": denominator,
            "token_edit_rate": numerator / denominator if denominator else None,
            "substitutions": sum(row["substitutions"] for row in scored),
            "deletions": sum(row["deletions"] for row in scored),
            "insertions": sum(row["insertions"] for row in scored),
            "exact_token_matches": matches, "exact_token_match_denominator": len(scored),
            "exact_token_match_rate": matches / len(scored) if scored else None,
            "mean_ctc_loss_target_length_normalized":
                sum(row["ctc_loss_target_length_normalized"] for row in scored) / len(scored) if scored else None,
            "mean_ctc_loss_confidence_weighted":
                sum(row["ctc_loss_confidence_weighted"] for row in scored) / len(scored) if scored else None,
            "confidence_weight_sum": sum(row["ctc_weight"] for row in scored)}


def read_pcm(prepared, row):
    require(row["audio"] == row["audio_path"], "D1_DEV_AUDIO_PATH_BINDING")
    relative = training.io_contract.relative(row["audio"])
    raw = training.io_contract.read_bytes(prepared / relative, 2 * 1024 * 1024)
    require(hashlib.sha256(raw).hexdigest() == row["file_sha256"], "D1_DEV_WAV_CHANGED")
    with wave.open(io.BytesIO(raw), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes(), reader.getcomptype()) ==
                (1, 2, 16000, row["samples"], "NONE"), "D1_DEV_PCM_FORMAT")
        pcm = reader.readframes(reader.getnframes())
    require(len(pcm) == row["samples"] * 2 and hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"],
            "D1_DEV_PCM_CHANGED")
    return pcm


def native_features(pcm, row, candidate, torch, processor=None):
    import numpy as np
    waveform = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768))
    if candidate == "fsmn-head-ctc2601":
        require(processor is not None, "D1_DEV_FSMN_FRONTEND_REQUIRED")
        data = [{"key": row["source_id"], "label": row["target_ids"], "sample_rate": 16000,
                 "wav": waveform.unsqueeze(0)}]
        fbank = processor["compute_fbank"](data, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
        context = processor["context_expansion"](fbank, left=2, right=2)
        feature = next(processor["frame_skip"](context, skip_rate=3))["feat"].contiguous()
        frames, width = row["expected_output_frames"]["fsmn"], 400
    else:
        from frontend import features
        feature = features(waveform, feature_dim=32, frontend="logmel").contiguous()
        frames, width = row["expected_output_frames"]["rnn64" if candidate == "rnn64-ctc2601" else "c1"], 32
    require(tuple(feature.shape) == (frames, width) and bool(torch.isfinite(feature).all()) and
            frames >= row["ctc_minimum_frames"], "D1_DEV_ACTUAL_UNPADDED_GEOMETRY")
    return feature


def save_numeric(path, source_id, feature, logits):
    """Create one bounded per-clip tensor artifact and verify its readback."""
    import torch
    expected = {"feature_sha256": training.tensor_sha(feature), "logits_sha256": training.tensor_sha(logits)}
    payload = {"schema": "kws-d1-development-numeric-evidence-v2", "source_id": source_id,
               "feature": feature, "logits": logits, **expected}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    retained = torch.load(path, weights_only=True, map_location="cpu")
    require(retained["schema"] == payload["schema"] and retained["source_id"] == source_id and
            all(retained[name] == value for name, value in expected.items()) and
            training.tensor_sha(retained["feature"]) == expected["feature_sha256"] and
            training.tensor_sha(retained["logits"]) == expected["logits_sha256"], "D1_DEV_NUMERIC_READBACK_CHANGED")
    return training.io_contract.file_record(path)


def checkpoint_binding(checkpoint, checkpoint_sha, job, summary, record, prepared_sha, recipe):
    require(checkpoint["schema"] == "kws-d1-trained-checkpoint-v2" and
            checkpoint["candidate"] == job["candidate"] and checkpoint["near_percent"] == job["near_percent"] and
            checkpoint_sha == summary["final_checkpoint_sha256"] and
            checkpoint["prepared_manifest_sha256"] == summary["prepared_manifest_sha256"] == prepared_sha and
            checkpoint["vocabulary_sha256"] == summary["vocabulary_sha256"] == record["vocabulary_sha256"] and
            checkpoint["source_hashes"] == record["source_hashes"] and
            checkpoint["job_recipe_sha256"] == summary["job_recipe_sha256"] and
            checkpoint["initial_checkpoint_sha256"] == summary["initial_checkpoint_sha256"] ==
                record["initial_checkpoints"][job["candidate"]]["file_sha256"] and
            checkpoint["exposure_plan_sha256"] == summary["exposure_plan_sha256"] == recipe["exposure_plan_sha256"] and
            checkpoint["optimizer_steps"] == summary["optimizer_steps"] == recipe["optimizer_steps"] == 150 and
            checkpoint["sample_exposures"] == summary["sample_exposures"] == recipe["total_exposures"] == 600 and
            checkpoint["seed"] == training.SEED and checkpoint["human_gold"] is False and
            checkpoint["quality_winner"] is None, "D1_DEV_CHECKPOINT_TRAINING_BINDING")


def evaluate(job_path, output, lock):
    job = validate_job(training.read_json(job_path))
    require(training.sha_file(__file__) == job["evaluator_source_sha256"], "D1_DEV_EXECUTION_SOURCE_CHANGED")
    with heavy_model_slot(lock) as lease:
        training.queue_binding(job_path, lease)
        output = training.output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        deadline = started + job["timeout_seconds"]
        status = {"schema": RESULT_SCHEMA, "status": "running", "started_at": training.utc(),
                  "job_recipe_sha256": training.sha_file(job_path), "candidate": job["candidate"],
                  "near_percent": job["near_percent"], "optimizer_steps": 0, "model_forward_calls": 0,
                  "initial_models_constructed": 0, "human_gold": False, "independent_human_test": False,
                  "quality_winner": None, "thresholds_tuned": False, "automatic_retry": False,
                  "formal_effectiveness_acceptance": False,
                  "weak_label_use": "predeclared-development-engineering-diagnostics-only"}
        training.save_json(output / "status.json", status)
        observations = []
        try:
            runtime = training.research_runtime.load_manifest()
            training.research_runtime.check_interpreter(runtime)
            training.research_runtime.check_packages(runtime, "core")
            prepared, prepared_sha, preparation_dependency = training.resolve_prepared(job)
            record, vocabulary, train_rows = training.verify_prepared(prepared, prepared_sha)
            all_rows = [training.io_contract.loads(line) for line in
                        training.io_contract.read_bytes(prepared / "projected-rows.jsonl", 64 * 1024 * 1024).splitlines()]
            rows = development_rows(all_rows, train_rows, record, vocabulary, job["held_out_source_family_ids"])
            trained, summary, dependency = completed_training(job)
            recipe = training.read_json(prepared / "recipes" / (str(job["near_percent"]) + ".json"))
            require(summary["exposure_recipe_sha256"] == training.sha_file(prepared / "recipes" / (str(job["near_percent"]) + ".json")) and
                    recipe["near_percent_of_all_draws"] == job["near_percent"] and
                    recipe["cohort_sha256"] == record["cohort_sha256"], "D1_DEV_TRAINING_RECIPE_CHANGED")
            torch, reference = training.torch_runtime()
            from training_state import load_checkpoint_snapshot, state_identity
            checkpoint, checkpoint_sha = load_checkpoint_snapshot(trained / "final.pt")
            checkpoint_binding(checkpoint, checkpoint_sha, job, summary, record, prepared_sha, recipe)
            identity = state_identity(checkpoint["state_dict"])
            require(identity == checkpoint["float_state_identity"] and
                    identity["sha256"] == summary["final_float_state_sha256"], "D1_DEV_CHECKPOINT_FLOAT_STATE_CHANGED")
            require(time.monotonic() < deadline, "D1_DEV_SOFT_DEADLINE")
            model = processor = None
            if any(row["ctc_mask"] for row in rows):
                if job["candidate"] == "fsmn-head-ctc2601":
                    model, processor, description = training.build_extended_fsmn(prepared / "official", vocabulary)
                else:
                    model, description = training.build_fresh(job["candidate"], vocabulary, reference)
                require(description == record["initial_checkpoints"][job["candidate"]]["description"] == summary["description"],
                        "D1_DEV_MODEL_DESCRIPTION_CHANGED")
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                require(state_identity(model.state_dict()) == identity, "D1_DEV_MODEL_LOAD_CHANGED")
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                status["initial_models_constructed"] = 1
            status.update(prepared_manifest_sha256=prepared_sha, preparation_dependency=preparation_dependency,
                training_dependency=dependency, checkpoint_sha256=checkpoint_sha,
                checkpoint_float_state_sha256=identity["sha256"], evaluator_source_sha256=job["evaluator_source_sha256"],
                preparation_source_hashes=record["source_hashes"], vocabulary_sha256=digest(vocabulary),
                held_out_source_family_ids=job["held_out_source_family_ids"],
                development_roster_sha256=digest([row_identity(row) for row in rows]),
                evaluable_roster_sha256=digest([row_identity(row) for row in rows if row["ctc_mask"]]),
                development_scope="two held-out synthetic reference-clone families; machine-weak full transcripts")
            training.save_json(output / "status.json", status)
            numeric = []
            with torch.inference_mode():
                for row in rows:
                    require(time.monotonic() < deadline, "D1_DEV_SOFT_DEADLINE")
                    pcm = read_pcm(prepared, row)
                    item = {key: row[key] for key in (
                        "source_id", "global_source_key", "item_id", "source_family_id", "speaker_id", "source_role",
                        "file_sha256", "pcm_sha256", "full_text", "target_ids", "ctc_mask", "ctc_weight",
                        "label_status", "label_reason", "ctc_reason", "oov_characters", "generation_category")}
                    item.update(label_binding_sha256=row_identity(row),
                                generation_outcome=row.get("generation_outcome"), samples=row["samples"],
                                evaluation_status="not_evaluable", reason=row["ctc_reason"])
                    if row["ctc_mask"]:
                        feature = native_features(pcm, row, job["candidate"], torch, processor)
                        require(time.monotonic() < deadline, "D1_DEV_SOFT_DEADLINE")
                        logits = model(feature.unsqueeze(0))
                        status["model_forward_calls"] += 1
                        require(tuple(logits.shape) == (len(feature), 1, OUTPUTS), "D1_DEV_LOGIT_GEOMETRY")
                        item.update(evaluation_status="evaluated", reason=None, feature_sha256=training.tensor_sha(feature),
                                    feature_shape=list(feature.shape), logits_sha256=training.tensor_sha(logits),
                                    **score_logits(logits, row))
                        name = "numeric/" + row["source_id"] + ".pt"
                        item["numeric_artifact"] = {"path": name,
                            **save_numeric(output / name, row["source_id"], feature, logits)}
                        numeric.append(item["numeric_artifact"])
                    observations.append(item)
                    training.append_json(output / "observations.jsonl", item)
                    require(time.monotonic() < deadline, "D1_DEV_SOFT_DEADLINE")
            require(model is None or state_identity(model.state_dict()) == identity, "D1_DEV_FROZEN_CHECKPOINT_CHANGED")
            require(training.sha_file(trained / "final.pt") == checkpoint_sha, "D1_DEV_CHECKPOINT_FILE_CHANGED")
            training.copy_once(job_path, output / "request.json")
            require(training.copy_once(ROOT / SOURCE_PATH, output / "execution-source" / SOURCE_PATH) == job["evaluator_source_sha256"],
                    "D1_DEV_SOURCE_CHANGED_DURING_EXECUTION")
            require(time.monotonic() < deadline, "D1_DEV_SOFT_DEADLINE")
            end = resource.getrusage(resource.RUSAGE_SELF)
            status.update(status="completed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                user_cpu_seconds=end.ru_utime - usage.ru_utime, system_cpu_seconds=end.ru_stime - usage.ru_stime,
                peak_rss_kib=end.ru_maxrss, checkpoint_state_unchanged=True, aggregate=summarize(observations),
                by_source_family={name: summarize([row for row in observations if row["source_family_id"] == name])
                                  for name in job["held_out_source_family_ids"]},
                observations_sha256=training.sha_file(output / "observations.jsonl"),
                numeric_artifact_roster_sha256=digest(numeric),
                metric_scope="unweighted token edits and clip exact matches; confidence affects only CTC loss diagnostic")
            training.save_json(output / "status.json", status)
            training.save_json(output / "summary.json", status)
            training.io_contract.write_json(output / "MANIFEST.json", {
                path.relative_to(output).as_posix(): training.io_contract.file_record(path)
                for path in sorted(output.rglob("*")) if path.is_file()})
            return status
        except BaseException as error:
            status.update(status="failed", finished_at=training.utc(), wall_seconds=time.monotonic() - started,
                          recorded_clip_observations=len(observations), recorded_forwards_are_lower_bound=True,
                          error_type=type(error).__name__, error=str(error))
            training.save_json(output / "status.json", status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heavy-lock", required=True, type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.recipe, args.output, args.heavy_lock)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
