#!/usr/bin/env python3
"""Prepare and run new D1 CTC candidates under the durable local queue.

Preparation freezes the verified row projections, two native feature geometries,
initial checkpoints and three fixed-exposure recipes. A training job consumes
exactly one candidate/ratio with a fresh optimizer. No automatic resume, quality
selection, model export, remote API, or historical result backfill is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
import research_runtime
import research_queue as io_contract
from research_job_lock import heavy_model_slot
from research_d1_inputs import aggregate_batches, load_data_modules
from research_d1_models import build_extended_fsmn, build_fresh, fulltext_ctc
from research_d1_recipe import BATCH_SIZE, SEED, available_recipes, cohort_rows, project_rows, row_identity
from research_d1_vocab import canonical, digest, make_vocabulary, require, validate_vocabulary
from research_weak_train import append_json, save_json, sha_file, tensor_sha, torch_runtime

PREP_SCHEMA = "kws-d1-preparation-request-v2"
JOB_SCHEMA = "kws-d1-training-job-v2"
CANDIDATES = ("rnn64-ctc2601", "c1-ctc2601", "fsmn-head-ctc2601")
HEAD = "backbone.out_linear2."
CODE_PATHS = (
    "training/research_d1_train.py", "training/research_d1_vocab.py", "training/research_d1_inputs.py",
    "training/research_d1_models.py", "training/research_d1_recipe.py",
    "training/research_weak_train.py", "training/research_weak_recipe.py", "training/research_fsmn_weak.py",
    "training/model.py", "training/frontend.py", "training/frontend_spec.py", "training/training_state.py",
    "research/candidate_encoders/contract.py", "research/candidate_encoders/fulltext_contract.py",
    "research/candidate_encoders/torch_models.py", "configs/runtime/fsmn-fulltext-sources.json",
    "configs/runtime/research-runtime.json", "tools/research_runtime.py", "tools/research_job_lock.py",
    "tools/research_queue.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return io_contract.loads(io_contract.read_bytes(path, 64 * 1024 * 1024))


def output_directory(path):
    root = io_contract.safe_path(path)
    if root.exists():
        require(root.is_dir() and not any(root.iterdir()), "D1_OUTPUT_MUST_BE_NEW_OR_EMPTY_QUEUE_DIRECTORY")
    else:
        root.mkdir(parents=True)
    return root


def copy_once(origin, target):
    raw = io_contract.read_bytes(origin, 256 * 1024 * 1024)
    target.parent.mkdir(parents=True, exist_ok=True)
    io_contract.write_new(target, raw)
    return hashlib.sha256(raw).hexdigest()


def validate_prepare_request(value):
    require(type(value) is dict and set(value) == {
        "schema", "packages", "plan_sha256", "view_loader", "original_tokens", "official_fsmn_sources"} and
        value["schema"] == PREP_SCHEMA, "D1_PREPARATION_REQUEST_SCHEMA")
    require(type(value["packages"]) is list and value["packages"], "D1_PREPARATION_PACKAGES")
    for key in ("view_loader", "original_tokens", "official_fsmn_sources"):
        require(type(value[key]) is str and Path(value[key]).is_absolute(), "D1_ABSOLUTE_DECLARED_SOURCE_PATH")
    return value


def validate_job(value):
    base = {"schema", "candidate", "near_percent", "timeout_seconds"}
    require(type(value) is dict and set(value) in (base | {"prepared", "prepared_manifest_sha256"},
                                                  base | {"prepared_from_dependency"}) and
        value["schema"] == JOB_SCHEMA and value["candidate"] in CANDIDATES and
        type(value["near_percent"]) is int and value["near_percent"] in (0, 5, 10), "D1_TRAIN_JOB_SCHEMA")
    if "prepared" in value:
        require(type(value["prepared"]) is str and Path(value["prepared"]).is_absolute() and
                type(value["prepared_manifest_sha256"]) is str and
                io_contract.SHA.fullmatch(value["prepared_manifest_sha256"]), "D1_PREPARED_IDENTITY_REQUIRED")
    else:
        require(type(value["prepared_from_dependency"]) is str and io_contract.ID.fullmatch(value["prepared_from_dependency"]),
                "D1_PREPARED_DEPENDENCY_ID")
    require(type(value["timeout_seconds"]) in (int, float) and math.isfinite(value["timeout_seconds"]) and
            0 < value["timeout_seconds"] <= 999 * 60, "D1_DECLARED_JOB_TIMEOUT")
    return value


def resolve_prepared(job, *, dependencies_path=None):
    if "prepared" in job:
        return Path(job["prepared"]), job["prepared_manifest_sha256"], None
    path = dependencies_path or os.environ.get("KWS_RESEARCH_DEPENDENCIES")
    require(path is not None, "D1_COMPLETED_PREPARATION_DEPENDENCY_REQUIRED")
    dependencies = read_json(path)
    name = job["prepared_from_dependency"]
    require(name in dependencies, "D1_PREPARATION_NOT_A_DECLARED_DEPENDENCY")
    dep = dependencies[name]
    raw = io_contract.read_bytes(dep["receipt"])
    receipt = io_contract.loads(raw)
    require(hashlib.sha256(raw).hexdigest() == dep["receipt_sha256"] and receipt["status"] == "completed" and
            receipt["job_id"] == name and receipt["outputs"] == dep["outputs"] and "MANIFEST.json" in dep["outputs"],
            "D1_PREPARATION_RECEIPT_CHANGED")
    root = io_contract.safe_path(dep["outputs_root"])
    record = dep["outputs"]["MANIFEST.json"]
    require(io_contract.file_record(root / "MANIFEST.json") == record, "D1_DEPENDENCY_MANIFEST_CHANGED")
    return root, record["sha256"], {"job_id": name, "receipt_sha256": dep["receipt_sha256"]}


def queue_binding(path, lease):
    require(lease.supervised and os.environ.get("KWS_RESEARCH_RECIPE_SHA256") == sha_file(path),
            "D1_HASH_BOUND_DURABLE_QUEUE_REQUIRED")


def source_records():
    return {name: sha_file(ROOT / name) for name in CODE_PATHS}


def prepare(request_path, output, lock):
    request = validate_prepare_request(read_json(request_path))
    with heavy_model_slot(lock) as lease:
        queue_binding(request_path, lease)
        runtime = research_runtime.load_manifest()
        research_runtime.check_interpreter(runtime)
        research_runtime.check_packages(runtime, "core")
        output = output_directory(output)
        started = time.monotonic()
        status = {"schema": "kws-d1-preparation-status-v2", "status": "running", "started_at": utc(),
                  "request_sha256": sha_file(request_path), "optimizer_steps": 0}
        save_json(output / "status.json", status)
        try:
            code = source_records()
            torch, reference = torch_runtime()
            import numpy as np
            from frontend import features
            from research_fsmn_weak import load_upstream, source_lock, verify_sources
            from training_state import atomic_save_checkpoint, state_identity
            loader, sampler, data_source = load_data_modules(request["view_loader"])
            verified, aggregate = aggregate_batches(request["packages"], loader,
                expected_plan_sha256=request["plan_sha256"])
            original_tokens = io_contract.read_bytes(request["original_tokens"])
            vocabulary = make_vocabulary(original_tokens)
            projected = project_rows(verified, vocabulary)
            recipes, availability = available_recipes(projected, vocabulary, sampler)
            save_json(output / "arm-availability.json", availability)
            require(recipes, "D1_NO_AVAILABLE_TRAINING_ARMS")
            cohort = cohort_rows(projected)
            for row in projected:
                name = "audio/" + row["source_id"] + ".wav"
                require(copy_once(row["audio_path"], output / name) == row["file_sha256"], "D1_AUDIO_CHANGED_AFTER_REPLAY")
                row["audio"] = row["audio_path"] = name
            # Cohort entries reference the same dictionaries; their binding
            # deliberately excludes location while retaining PCM/file hashes.
            require([row_identity(row) for row in cohort] == [item["label_binding_sha256"] for item in recipes["0"]["items"]],
                    "D1_ROW_IDENTITY_CHANGED_DURING_COPY")
            official = Path(request["official_fsmn_sources"])
            verify_sources(official)
            for item in source_lock()["files"]:
                require(copy_once(official / item["path"], output / "official" / item["path"]) == item["sha256"],
                        "D1_OFFICIAL_SOURCE_CHANGED_DURING_COPY")
            require((output / "official/tokens.txt").read_bytes() == original_tokens, "D1_ORIGINAL_VOCABULARY_DIFFERS")
            _, _, processor = load_upstream(output / "official", torch)
            caches, observations = {"fresh": [], "fsmn": []}, []
            for row in cohort:
                with wave.open(str(output / row["audio"]), "rb") as reader:
                    require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes(), reader.getcomptype()) ==
                            (1, 2, 16000, row["samples"], "NONE"), "D1_TRAINING_PCM_FORMAT")
                    pcm = reader.readframes(reader.getnframes())
                require(hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"], "D1_TRAINING_PCM_CHANGED")
                waveform = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768))
                fresh = features(waveform, feature_dim=32, frontend="logmel").contiguous()
                data = [{"key": row["source_id"], "label": row["target_ids"], "sample_rate": 16000,
                         "wav": waveform.unsqueeze(0)}]
                fbank = processor["compute_fbank"](data, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
                context = processor["context_expansion"](fbank, left=2, right=2)
                fsmn = next(processor["frame_skip"](context, skip_rate=3))["feat"].contiguous()
                require(tuple(fresh.shape) == (row["expected_output_frames"]["rnn64"], 32) and
                        tuple(fsmn.shape) == (row["expected_output_frames"]["fsmn"], 400) and
                        bool(torch.isfinite(fresh).all()) and bool(torch.isfinite(fsmn).all()) and
                        min(len(fresh), len(fsmn)) >= row["ctc_minimum_frames"], "D1_ACTUAL_UNPADDED_GEOMETRY")
                caches["fresh"].append(fresh)
                caches["fsmn"].append(fsmn)
                observations.append({"source_id": row["source_id"], "pcm_sha256": row["pcm_sha256"],
                    "fresh_frames": len(fresh), "fsmn_frames": len(fsmn), "ctc_minimum_frames": row["ctc_minimum_frames"],
                    "fresh_feature_sha256": tensor_sha(fresh), "fsmn_feature_sha256": tensor_sha(fsmn)})
            atomic_save_checkpoint(caches, output / "features.pt")
            initial = {}
            for candidate in CANDIDATES:
                torch.manual_seed(SEED)
                if candidate == "fsmn-head-ctc2601":
                    model, _, description = build_extended_fsmn(output / "official", vocabulary)
                else:
                    model, description = build_fresh(candidate, vocabulary, reference)
                identity = state_identity(model.state_dict())
                path = output / "initial" / (candidate + ".pt")
                payload = {"schema": "kws-d1-initial-checkpoint-v2", "candidate": candidate,
                           "vocabulary_sha256": digest(vocabulary), "state_dict": model.state_dict(),
                           "float_state_identity": identity, "description": description, "seed": SEED,
                           "optimizer_steps": 0, "source_hashes": code}
                atomic_save_checkpoint(payload, path)
                initial[candidate] = {"file_sha256": sha_file(path), "float_state_sha256": identity["sha256"],
                                      "description": description}
                del model
            for name, expected in code.items():
                require(copy_once(ROOT / name, output / "execution-source" / name) == expected,
                        "D1_EXECUTION_SOURCE_CHANGED_DURING_PREPARATION")
            for name, expected in data_source["files"].items():
                require(copy_once(Path(data_source["root"]) / name, output / "data-source" / name) == expected,
                        "D1_DATA_SOURCE_CHANGED_DURING_PREPARATION")
            io_contract.write_new(output / "request.json", io_contract.read_bytes(request_path))
            io_contract.write_new(output / "projected-rows.jsonl", b"".join(canonical(row) + b"\n" for row in projected))
            io_contract.write_new(output / "train-rows.jsonl", b"".join(canonical(row) + b"\n" for row in cohort))
            save_json(output / "vocabulary.json", vocabulary)
            save_json(output / "aggregate.json", aggregate)
            save_json(output / "feature-observations.json", observations)
            for percent, recipe in recipes.items():
                save_json(output / "recipes" / (percent + ".json"), recipe)
            status.update(status="completed", finished_at=utc(), wall_seconds=time.monotonic() - started)
            save_json(output / "status.json", status)
            preparation = {"schema": "kws-d1-training-preparation-v2", "status": "completed",
                "request_sha256": status["request_sha256"], "source_hashes": code, "data_source": data_source,
                "vocabulary_sha256": digest(vocabulary), "cohort_sha256": recipes["0"]["cohort_sha256"],
                "cohort_items": len(cohort), "observed_rows": len(projected), "aggregate": aggregate,
                "projected_rows_by_split": dict(Counter(row["split"] for row in projected)),
                "projection_reasons": dict(Counter(row["ctc_reason"] for row in projected)),
                "arm_availability": availability,
                "keyword_eligible_train_items": len(cohort_rows(projected, objective="keyword")),
                "initial_checkpoints": initial, "optimizer_steps": 0, "asr_or_tts_calls": 0,
                "initial_models_constructed": 3, "model_forward_calls": 0,
                "human_gold": False, "quality_winner": None, "wall_seconds": status["wall_seconds"]}
            save_json(output / "preparation.json", preparation)
            manifest = {path.relative_to(output).as_posix(): io_contract.file_record(path)
                        for path in sorted(output.rglob("*")) if path.is_file()}
            io_contract.write_json(output / "MANIFEST.json", manifest)
            return {"status": "completed", "prepared": str(output), "manifest_sha256": sha_file(output / "MANIFEST.json"),
                    "cohort_items": len(cohort), "optimizer_steps": 0}
        except BaseException as error:
            status.update(status="failed", finished_at=utc(), error_type=type(error).__name__, error=str(error),
                          wall_seconds=time.monotonic() - started)
            save_json(output / "status.json", status)
            raise


def verify_prepared(root, manifest_sha256):
    root = io_contract.safe_path(root)
    raw = io_contract.read_bytes(root / "MANIFEST.json")
    require(hashlib.sha256(raw).hexdigest() == manifest_sha256, "D1_PREPARED_MANIFEST_CHANGED")
    manifest = io_contract.loads(raw)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != root / "MANIFEST.json"}
    require(type(manifest) is dict and actual == set(manifest), "D1_PREPARED_FILE_ROSTER_CHANGED")
    for name, expected in manifest.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
                "D1_UNSAFE_PREPARED_MEMBER")
        require(io_contract.file_record(root / path) == expected, "D1_PREPARED_FILE_CHANGED:" + name)
    record = read_json(root / "preparation.json")
    require(record["schema"] == "kws-d1-training-preparation-v2" and record["status"] == "completed" and
            record["optimizer_steps"] == 0 and record["source_hashes"] == source_records(),
            "D1_PREPARED_EXECUTION_SOURCE_CHANGED")
    vocabulary = validate_vocabulary(read_json(root / "vocabulary.json"), io_contract.read_bytes(root / "official/tokens.txt"))
    require(record["vocabulary_sha256"] == digest(vocabulary), "D1_PREPARED_VOCABULARY_CHANGED")
    rows = [io_contract.loads(line) for line in io_contract.read_bytes(root / "train-rows.jsonl", 64 * 1024 * 1024).splitlines()]
    require(len(rows) == record["cohort_items"] and len({tuple(row["global_source_key"]) for row in rows}) == len(rows),
            "D1_PREPARED_COHORT_ROSTER")
    return record, vocabulary, rows


def train_batches(model, features, rows, recipe, optimizer, *, output, deadline):
    """The actual weighted loss consumer; each saved draw is consumed once."""
    import torch
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    steps, exposures = 0, 0
    losses = defaultdict(float)
    families, categories, positive_words = Counter(), Counter(), Counter()
    draws = recipe["exposure_plan"]["draws"]
    for begin in range(0, len(draws), BATCH_SIZE):
        require(time.monotonic() < deadline, "D1_SOFT_DEADLINE")
        batch = draws[begin:begin + BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        terms, observed = [], []
        for draw in batch:
            index = draw["row_index"]
            row = rows[index]
            require(row["source_id"] == draw["source_id"] and row["split"] == "train" and row["ctc_mask"] is True and
                    draw["ctc_weight"] == row["ctc_weight"], "D1_ACTUAL_DRAW_BINDING")
            logits = model(features[index].unsqueeze(0))
            term, raw = fulltext_ctc(logits, row["target_ids"], draw["ctc_weight"], row["ctc_mask"],
                                     normalizer=recipe["fixed_ctc_normalizer"])
            terms.append(term)
            contribution = term.detach().item() / len(batch)
            losses[row["generation_category"]] += contribution
            observed.append({"source_id": row["source_id"], "global_source_key": row["global_source_key"],
                "local_item_id": row["item_id"], "source_family_id": row["source_family_id"],
                "pcm_sha256": row["pcm_sha256"], "full_text": row["full_text"], "target_ids": row["target_ids"],
                "ctc_mask": row["ctc_mask"], "ctc_weight": draw["ctc_weight"],
                "keyword_targets": row["keyword_targets"], "keyword_mask": row["keyword_mask"],
                "keyword_weight": row["keyword_weight"], "keyword_loss_used": False,
                "label_binding_sha256": row_identity(row), "raw_length_normalized_ctc": raw.item(),
                "weighted_batch_contribution": contribution, "generation_category": row["generation_category"]})
            families[row["source_family_id"]] += 1
            categories[row["generation_category"]] += 1
            for keyword, target, mask in zip(("你好小窝", "小窝小窝"), row["keyword_targets"], row["keyword_mask"]):
                if target == 1 and mask:
                    positive_words[keyword] += 1
        loss = torch.stack(terms).mean()
        require(bool(torch.isfinite(loss)), "D1_NONFINITE_BATCH_LOSS")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected), "D1_NONFINITE_OR_MISSING_GRADIENT")
        norm = torch.nn.utils.clip_grad_norm_(selected, recipe["gradient_clip_max_norm"])
        require(bool(torch.isfinite(norm)), "D1_NONFINITE_GRADIENT_NORM")
        optimizer.step()
        steps += 1
        exposures += len(batch)
        append_json(output / "consumption.jsonl", {"optimizer_step": steps, "draw_offset": begin,
                    "loss": loss.item(), "gradient_norm_before_clip": norm.item(), "draws": observed})
        require(time.monotonic() < deadline, "D1_SOFT_DEADLINE")
    return {"optimizer_steps": steps, "sample_exposures": exposures,
            "actual_weighted_batch_loss_sum_by_category": dict(losses),
            "actual_exposures_by_family": dict(families), "actual_exposures_by_category": dict(categories),
            "actual_positive_exposures_by_word": dict(positive_words)}


def run(job_path, output, lock):
    job = validate_job(read_json(job_path))
    with heavy_model_slot(lock) as lease:
        queue_binding(job_path, lease)
        output = output_directory(output)
        started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
        deadline = started + job["timeout_seconds"]
        status = {"schema": "kws-d1-training-result-v2", "status": "running", "started_at": utc(),
            "job_recipe_sha256": sha_file(job_path), "candidate": job["candidate"], "near_percent": job["near_percent"],
            "optimizer_steps": 0, "sample_exposures": 0, "human_gold": False, "quality_winner": None,
            "keyword_loss_used": False, "actual_training_executed": False}
        save_json(output / "status.json", status)
        try:
            runtime = research_runtime.load_manifest()
            research_runtime.check_interpreter(runtime)
            research_runtime.check_packages(runtime, "core")
            prepared, prepared_manifest_sha, preparation_dependency = resolve_prepared(job)
            record, vocabulary, rows = verify_prepared(prepared, prepared_manifest_sha)
            require(job["near_percent"] in record["arm_availability"]["available"], "D1_REQUESTED_ARM_BLOCKED_BY_MISSING_POOL")
            recipe = read_json(prepared / "recipes" / (str(job["near_percent"]) + ".json"))
            require(recipe["cohort_sha256"] == record["cohort_sha256"] and recipe["near_percent_of_all_draws"] == job["near_percent"] and
                    [row_identity(row) for row in rows] == [item["label_binding_sha256"] for item in recipe["items"]],
                    "D1_RECIPE_ROW_BINDING")
            require(time.monotonic() < deadline, "D1_SOFT_DEADLINE")
            torch, reference = torch_runtime()
            from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
            torch.manual_seed(SEED)
            if job["candidate"] == "fsmn-head-ctc2601":
                model, _, description = build_extended_fsmn(prepared / "official", vocabulary)
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(name.startswith(HEAD))
                optimizer_spec = recipe["optimizers"]["extended-fsmn-head"]
                cache_kind = "fsmn"
            else:
                model, description = build_fresh(job["candidate"], vocabulary, reference)
                optimizer_spec = recipe["optimizers"]["fresh-rnn64-or-c1"]
                cache_kind = "fresh"
            initial_path = prepared / "initial" / (job["candidate"] + ".pt")
            initial, initial_sha = load_checkpoint_snapshot(initial_path)
            require(initial_sha == record["initial_checkpoints"][job["candidate"]]["file_sha256"] and
                    initial["candidate"] == job["candidate"] and initial["vocabulary_sha256"] == digest(vocabulary) and
                    initial["optimizer_steps"] == 0 and initial["description"] == description, "D1_INITIAL_CHECKPOINT_BINDING")
            model.load_state_dict(initial["state_dict"], strict=True)
            require(state_identity(model.state_dict()) == initial["float_state_identity"], "D1_INITIAL_STATE_CHANGED")
            frozen_before = {name: tensor_sha(value) for name, value in model.state_dict().items()
                             if job["candidate"] == "fsmn-head-ctc2601" and not name.startswith(HEAD)}
            feature_cache = torch.load(prepared / "features.pt", weights_only=True, map_location="cpu")[cache_kind]
            observations = read_json(prepared / "feature-observations.json")
            require(len(feature_cache) == len(rows) == len(observations), "D1_FEATURE_CACHE_ROSTER")
            for row, feature, observed in zip(rows, feature_cache, observations):
                require(row["source_id"] == observed["source_id"] and
                        tensor_sha(feature) == observed[cache_kind + "_feature_sha256"], "D1_FEATURE_ROW_BINDING")
            selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(selected, lr=optimizer_spec["lr"], weight_decay=optimizer_spec["weight_decay"])
            model.train()
            status.update(prepared_manifest_sha256=prepared_manifest_sha, preparation_dependency=preparation_dependency,
                vocabulary_sha256=digest(vocabulary),
                exposure_recipe_sha256=sha_file(prepared / "recipes" / (str(job["near_percent"]) + ".json")),
                exposure_plan_sha256=recipe["exposure_plan_sha256"], initial_checkpoint_sha256=initial_sha,
                initial_float_state_sha256=initial["float_state_identity"]["sha256"],
                trainable_parameter_names=trainable_names, description=description)
            require(time.monotonic() < deadline, "D1_SOFT_DEADLINE")
            status.update(optimizer_loop_entered=True, actual_training_executed=None)
            save_json(output / "status.json", status)
            consumed = train_batches(model, feature_cache, rows, recipe, optimizer, output=output, deadline=deadline)
            require(consumed["optimizer_steps"] == recipe["optimizer_steps"] and
                    consumed["sample_exposures"] == recipe["total_exposures"] and
                    consumed["actual_exposures_by_family"] == recipe["exposures_by_family"] and
                    consumed["actual_exposures_by_category"] == recipe["exposures_by_generation_category"] and
                    consumed["actual_positive_exposures_by_word"] == recipe["positive_exposures_by_word"],
                    "D1_EXPOSURE_BUDGET_NOT_CONSUMED")
            require(all(tensor_sha(model.state_dict()[name]) == expected for name, expected in frozen_before.items()),
                    "D1_FROZEN_FSMN_BACKBONE_CHANGED")
            identity = state_identity(model.state_dict())
            payload = {"schema": "kws-d1-trained-checkpoint-v2", "candidate": job["candidate"],
                "near_percent": job["near_percent"], "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "float_state_identity": identity, "vocabulary_sha256": digest(vocabulary),
                "prepared_manifest_sha256": prepared_manifest_sha, "preparation_dependency": preparation_dependency,
                "initial_checkpoint_sha256": initial_sha,
                "job_recipe_sha256": sha_file(job_path), "exposure_plan_sha256": recipe["exposure_plan_sha256"],
                "source_hashes": record["source_hashes"], "optimizer_steps": consumed["optimizer_steps"],
                "sample_exposures": consumed["sample_exposures"], "seed": SEED, "human_gold": False, "quality_winner": None}
            atomic_save_checkpoint(payload, output / "final.pt")
            reloaded, checkpoint_sha = load_checkpoint_snapshot(output / "final.pt")
            require(state_identity(reloaded["state_dict"]) == identity, "D1_CHECKPOINT_READBACK_CHANGED")
            require(time.monotonic() < deadline, "D1_SOFT_DEADLINE")
            end = resource.getrusage(resource.RUSAGE_SELF)
            status.update(consumed, status="completed", actual_training_executed=True, finished_at=utc(),
                wall_seconds=time.monotonic() - started, user_cpu_seconds=end.ru_utime - usage.ru_utime,
                system_cpu_seconds=end.ru_stime - usage.ru_stime, peak_rss_kib=end.ru_maxrss,
                final_checkpoint_sha256=checkpoint_sha, final_float_state_sha256=identity["sha256"],
                consumption_sha256=sha_file(output / "consumption.jsonl"), frozen_fsmn_backbone_unchanged=bool(frozen_before))
            save_json(output / "status.json", status)
            save_json(output / "summary.json", status)
            return status
        except BaseException as error:
            # A partial ledger is evidence of actual completed optimizer steps,
            # never permission to continue or reset the job automatically.
            ledger = output / "consumption.jsonl"
            if ledger.exists():
                events = [io_contract.loads(line) for line in ledger.read_bytes().splitlines()]
                status.update(optimizer_steps=len(events), sample_exposures=sum(len(row["draws"]) for row in events),
                              actual_training_executed=True if events else None, consumption_sha256=sha_file(ledger))
            status.update(status="failed", finished_at=utc(), error_type=type(error).__name__, error=str(error),
                          wall_seconds=time.monotonic() - started, automatic_retry=False,
                          recorded_optimizer_steps_are_lower_bound=True)
            save_json(output / "status.json", status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("--recipe", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--heavy-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (prepare if args.command == "prepare" else run)(args.recipe, args.output, args.heavy_lock)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
