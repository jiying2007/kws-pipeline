#!/usr/bin/env python3
"""Bounded D0 weak-label consumption diagnostics, separate from product training.

The main RNN64/C1 runs use identical full-text CTC inputs and fixed exposures.
A separately recorded frozen-C1 keyword-head probe consumes keyword masks.
There is no development split, model selection, quality retry or model export.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "training"))
import research_runtime
from research_job_lock import heavy_model_slot
from research_weak_recipe import (CHAR_IDS, TOKEN_NAMES, canonical, digest, make_recipe, require)


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_read(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def tensor_sha(value):
    import torch
    tensor = value.detach().cpu().contiguous()
    header = canonical({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return hashlib.sha256(header + b"\n" + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def torch_runtime():
    # Import the existing reference's deterministic CPU contract before Torch.
    # This intentionally keeps both architectures at one thread (within <=4).
    import model as reference
    import torch
    torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            "shared deterministic one-thread CPU contract required")
    require(not torch.backends.mkldnn.enabled, "shared MKLDNN setting changed")
    return torch, reference.TinyStreamingRNN


def code_inventory():
    paths = ["training/research_weak_train.py", "training/research_weak_recipe.py", "training/model.py",
             "training/frontend.py", "training/frontend_spec.py", "training/training_state.py",
             "research/candidate_encoders/contract.py", "research/candidate_encoders/torch_models.py",
             "configs/runtime/research-runtime.json", "tools/research_runtime.py", "tools/research_job_lock.py"]
    return {path: sha_file(ROOT / path) for path in paths}


def prepare(args):
    manifest = research_runtime.load_manifest()
    research_runtime.check_interpreter(manifest)
    research_runtime.check_packages(manifest, "core")
    view = args.view.resolve()
    require(sha_file(view / "MANIFEST.json") == args.view_manifest_sha256, "view manifest differs from the fixed snapshot")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    loader_path = args.view_loader.resolve()
    spec = importlib.util.spec_from_file_location("d0_verified_view_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    rows = list(loader.load_training_rows(view, "rnn-four-token"))
    recipe = make_recipe(rows, expected_items=40)
    require(sha_file(view / "MANIFEST.json") == args.view_manifest_sha256, "view changed during verification")
    torch, _ = torch_runtime()
    import numpy as np
    from frontend import features
    tensors, observations = [], []
    for row in rows:
        audio = Path(row["audio_path"])
        require(sha_file(audio) == row["file_sha256"], "training audio file differs")
        with wave.open(str(audio), "rb") as stream:
            require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000),
                    "training derivative must be mono PCM16 at 16 kHz")
            require(stream.getnframes() == row["samples"], "training sample count differs")
            pcm = stream.readframes(stream.getnframes())
        require(hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"], "training PCM differs")
        signal_values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
        feature = features(torch.from_numpy(signal_values), feature_dim=32, frontend="logmel").contiguous()
        needed = len(row["target_ids"]) + sum(a == b for a, b in zip(row["target_ids"], row["target_ids"][1:]))
        require(feature.shape[0] >= needed and feature.shape[1] == 32 and bool(torch.isfinite(feature).all()),
                "full-text CTC geometry is infeasible or features are nonfinite")
        tensors.append(feature)
        observations.append({"item_id": row["item_id"], "source_id": row["source_id"],
                             "samples": row["samples"], "feature_frames": feature.shape[0],
                             "ctc_minimum_frames": needed, "feature_sha256": tensor_sha(feature),
                             "full_target_ids": row["target_ids"], "pcm_sha256": row["pcm_sha256"],
                             "label_binding_sha256": recipe["items"][len(observations)]["label_binding_sha256"]})
    cache = output / "features.pt"
    with cache.open("xb") as stream:
        torch.save({"features": tensors}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    source_rows = output / "source-rows.jsonl"
    with source_rows.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    save_json(output / "recipe.json", recipe)
    save_json(output / "feature-observations.json", observations)
    data_root = loader_path.parents[1]
    data_sources = {str(loader_path.relative_to(data_root)): sha_file(loader_path)}
    for name in ("identity.py", "reference_text.py", "source_registry.py"):
        path = data_root / "kws_data_contract" / name
        data_sources[str(path.relative_to(data_root))] = sha_file(path)
    for name, sha in data_sources.items():
        target = output / "data-source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (data_root / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sha, "data loader changed during snapshot")
        target.write_bytes(raw)
    record = {"schema": "d0-weak-training-preparation-v1", "status": "complete", "created_at": utc(),
              "view": str(view), "view_manifest_sha256": args.view_manifest_sha256,
              "consumer_manifest_sha256": sha_file(view / "rnn-four-token-train.jsonl"),
              "view_report_sha256": sha_file(view / "report.json"), "view_loader": str(loader_path),
              "data_sources": data_sources, "recipe_sha256": sha_file(output / "recipe.json"),
              "source_rows_sha256": sha_file(source_rows), "feature_cache_sha256": sha_file(cache),
              "feature_observations_sha256": sha_file(output / "feature-observations.json"),
              "source_code": code_inventory(), "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__,
                          "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                          "mkldnn_enabled": torch.backends.mkldnn.enabled},
              "items": len(rows), "ctc_feasible_items": len(observations),
              "source_families": dict(Counter(row["source_family_id"] for row in rows)),
              "label_tiers": dict(Counter(row["label_tier"] for row in rows)),
              "keyword_mask_true_cells": sum(sum(row["keyword_mask"]) for row in rows),
              "keyword_mask_false_cells": sum(2 - sum(row["keyword_mask"]) for row in rows),
              "input_samples": sum(row["samples"] for row in rows),
              "feature_frames": sum(row["feature_frames"] for row in observations),
              "model_constructions": 0, "optimizer_steps": 0, "human_gold": False,
              "independent_development_items": 0, "quality_winner": None}
    save_json(output / "preparation.json", record)
    print(json.dumps({"status": "complete", "items": len(rows), "ctc_feasible_items": len(rows),
                      "recipe_sha256": record["recipe_sha256"], "output": str(output)}), flush=True)
    return 0


def weighted_ctc(logits, targets, weight, mask, *, normalizer):
    import torch
    if not mask:
        return logits.sum() * 0.0, logits.sum().detach() * 0.0
    require(0 <= weight <= 1 and normalizer > 0 and len(targets) > 0 and all(1 <= x <= 4 for x in targets),
            "invalid explicit full-text weighted CTC target")
    target = torch.tensor(targets, dtype=torch.long)
    raw = torch.nn.functional.ctc_loss(logits.log_softmax(2), target,
        torch.tensor([logits.shape[0]], dtype=torch.long), torch.tensor([len(targets)], dtype=torch.long),
        blank=0, reduction="none", zero_infinity=False)[0] / len(targets)
    return raw * (weight / normalizer), raw.detach()


def masked_keyword_loss(logits, targets, weights, masks, *, normalizer):
    import torch
    require(logits.shape == targets.shape == weights.shape == masks.shape and logits.shape[-1] == 2,
            "keyword loss shapes differ")
    require(normalizer > 0 and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()), "invalid keyword weights")
    safe_targets = torch.where(masks, targets, torch.zeros_like(targets))
    require(bool(((safe_targets == 0) | (safe_targets == 1)).all()), "known keyword targets must be binary")
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    effective = weights * masks.to(weights.dtype)
    return (raw * effective).mean() / normalizer, raw.detach()


def verified_prepared(root):
    record = json_read(root / "preparation.json")
    require(record["status"] == "complete" and record["source_code"] == code_inventory(), "training code or preparation differs from fixed recipe")
    for name, key in (("recipe.json", "recipe_sha256"), ("source-rows.jsonl", "source_rows_sha256"),
                      ("features.pt", "feature_cache_sha256"), ("feature-observations.json", "feature_observations_sha256")):
        require(sha_file(root / name) == record[key], "prepared artifact changed: " + name)
    rows = [json.loads(line) for line in (root / "source-rows.jsonl").read_text().splitlines()]
    recipe = json_read(root / "recipe.json")
    require(recipe == make_recipe(rows), "recipe differs from the observed fixed snapshot")
    for row in rows:
        require(sha_file(row["audio_path"]) == row["file_sha256"], "bound training WAV changed")
    return record, rows, recipe


def build_main(kind, recipe, reference):
    from research.candidate_encoders.contract import rnn64_spec, causal_ds_tcn_spec
    from research.candidate_encoders.torch_models import build_encoder
    if kind == "rnn64":
        spec = rnn64_spec()
        return build_encoder(spec, rnn_model=reference(32, 64, 5)), spec.to_dict()
    config = recipe["c1"]
    spec = causal_ds_tcn_spec(hidden_dim=config["hidden_dim"], kernels=tuple(config["kernels"]), dilations=tuple(config["dilations"]))
    return build_encoder(spec), spec.to_dict()


def worker(args):
    prepared, output = args.prepared.resolve(), args.output.resolve()
    record, rows, recipe = verified_prepared(prepared)
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    started = time.monotonic()
    status = {"schema": "d0-weak-training-run-v1", "kind": args.kind, "status": "running", "stage": "load",
              "started_at": utc(), "optimizer_steps": 0, "training_sample_exposures": 0,
              "recipe_sha256": record["recipe_sha256"], "preparation_sha256": sha_file(prepared / "preparation.json"),
              "source_revision": record["source_revision"], "source_code": record["source_code"],
              "human_gold": False, "independent_development_items": 0, "quality_winner": None,
              "training_set_readback_is_generalization": False, "production_qualified": False}
    save_json(status_path, status)
    args.heavy_lock.parent.mkdir(parents=True, exist_ok=True)
    with heavy_model_slot(args.heavy_lock):
        torch, reference = torch_runtime()
        from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
        manifest = research_runtime.load_manifest()
        research_runtime.check_interpreter(manifest)
        research_runtime.check_packages(manifest, "core")
        random.seed(recipe["seed"])
        torch.manual_seed(recipe["seed"])
        cached = torch.load(prepared / "features.pt", map_location="cpu", weights_only=True)["features"]
        observations = json_read(prepared / "feature-observations.json")
        require(len(cached) == len(rows) and all(tensor_sha(x) == item["feature_sha256"] for x, item in zip(cached, observations)),
                "cached feature tensors differ from their source bindings")
        status.update(runtime={"python": sys.version.split()[0], "torch": str(torch.__version__),
                               "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
                      deadline_seconds=args.seconds, actual_feature_frames=sum(x.shape[0] for x in cached))
        is_head = args.kind == "c1-keyword-head"
        frozen_encoder = None
        try:
            if is_head:
                encoder_checkpoint = args.encoder_checkpoint.resolve()
                payload, checkpoint_sha = load_checkpoint_snapshot(encoder_checkpoint)
                require(payload["kind"] == "c1-ctc" and payload["recipe_sha256"] == record["recipe_sha256"] and payload["completed_epochs"] == recipe["epochs"],
                        "keyword probe must use the fixed final C1 checkpoint")
                frozen_encoder, spec = build_main("c1-ctc", recipe, reference)
                frozen_encoder.load_state_dict(payload["state_dict"], strict=True)
                frozen_encoder.eval()
                for parameter in frozen_encoder.parameters():
                    parameter.requires_grad_(False)
                frozen_identity = state_identity(frozen_encoder.state_dict())
                require(frozen_identity == payload["float_state_identity"], "frozen encoder changed on load")
                embeddings = []
                for feature in cached:
                    captured = []
                    hook = frozen_encoder.out_proj.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
                    with torch.no_grad():
                        frozen_encoder(feature.unsqueeze(0))
                    hook.remove()
                    require(len(captured) == 1 and tuple(captured[0].shape) == (1, feature.shape[0], recipe["c1"]["hidden_dim"]),
                            "unexpected C1 pre-CTC hidden representation")
                    embeddings.append(captured[0][0].mean(0).contiguous())
                torch.manual_seed(recipe["head_seed"])
                model = torch.nn.Linear(recipe["c1"]["hidden_dim"], 2)
                spec = {"kind": "frozen-c1-keyword-head", "encoder": spec,
                        "head": {"type": "Linear", "input_dim": recipe["c1"]["hidden_dim"], "output_dim": 2},
                        "pooling": "mean over unpadded encoder frames"}
                status.update(encoder_checkpoint_sha256=checkpoint_sha, frozen_encoder_state_sha256=frozen_identity["sha256"],
                              encoder_feature_exposures=len(rows), probe_pooling="mean over unpadded C1 hidden frames",
                              embedding_sha256=[tensor_sha(value) for value in embeddings])
                epochs = recipe["head_probe_epochs"]
            else:
                model, spec = build_main(args.kind, recipe, reference)
                epochs = recipe["epochs"]
            optimizer_spec = recipe["optimizer"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_spec["lr"], betas=tuple(optimizer_spec["betas"]),
                                          eps=optimizer_spec["eps"], weight_decay=optimizer_spec["weight_decay"], foreach=False)
            initial_state = state_identity(model.state_dict())
            status.update(stage="training", parameter_count=sum(p.numel() for p in model.parameters()),
                          architecture=spec, objective="masked-weighted-keyword-BCE" if is_head else "full-text-weighted-CTC",
                          initial_state_sha256=initial_state["sha256"], optimizer=optimizer_spec,
                          main_comparison_member=not is_head)

            def checkpoint(epoch):
                identity = state_identity(model.state_dict())
                value = {"schema": "d0-research-checkpoint-v1", "kind": args.kind, "state_dict": model.state_dict(),
                         "float_state_identity": identity, "optimizer_state_dict": optimizer.state_dict(),
                         "completed_epochs": epoch, "optimizer_steps": status["optimizer_steps"],
                         "training_sample_exposures": status["training_sample_exposures"],
                         "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
                         "recipe_sha256": record["recipe_sha256"], "source_rows_sha256": record["source_rows_sha256"],
                         "feature_cache_sha256": record["feature_cache_sha256"], "architecture": spec,
                         "runtime": status["runtime"], "source_revision": record["source_revision"],
                         "source_code": record["source_code"], "human_gold": False, "quality_winner": None,
                         "encoder_checkpoint_sha256": status.get("encoder_checkpoint_sha256"), "resume_supported": False}
                path = output / "checkpoints" / f"epoch-{epoch:03d}.pt"
                atomic_save_checkpoint(value, path)
                return {"path": str(path), "sha256": sha_file(path), "float_state_sha256": identity["sha256"]}

            def batch_loss(indices):
                contributions = []
                if is_head:
                    logits = model(torch.stack([embeddings[index] for index in indices]))
                    masks = torch.tensor([rows[index]["keyword_mask"] for index in indices], dtype=torch.bool)
                    targets = torch.tensor([[target if target is not None else 0 for target in rows[index]["keyword_targets"]]
                                            for index in indices], dtype=torch.float32)
                    weights = torch.tensor([recipe["items"][index]["effective_keyword_weight"] for index in indices], dtype=torch.float32)
                    loss, raw = masked_keyword_loss(logits, targets, weights, masks, normalizer=recipe["fixed_keyword_normalizer"])
                    for offset, index in enumerate(indices):
                        contributions.append({"row_index": index, "raw_loss": raw[offset].mean().item(),
                            "weighted_contribution": (raw[offset] * weights[offset] * masks[offset]).sum().item() /
                                (2 * len(indices) * recipe["fixed_keyword_normalizer"])})
                else:
                    terms = []
                    for index in indices:
                        logits = model(cached[index].unsqueeze(0))
                        term, raw = weighted_ctc(logits, rows[index]["target_ids"], recipe["items"][index]["effective_ctc_weight"],
                                                rows[index]["ctc_mask"], normalizer=recipe["fixed_ctc_normalizer"])
                        terms.append(term)
                        contributions.append({"row_index": index, "raw_loss": raw.item(), "weighted_contribution": term.detach().item() / len(indices)})
                    loss = torch.stack(terms).mean()
                require(bool(torch.isfinite(loss)), "nonfinite weak objective; no retry")
                return loss, contributions

            def readback(label):
                model.eval()
                loss_sum, by_class = 0.0, defaultdict(float)
                with torch.no_grad():
                    for index in range(len(rows)):
                        loss, contributions = batch_loss([index])
                        loss_sum += loss.item()
                        by_class[recipe["items"][index]["class"]] += contributions[0]["weighted_contribution"] / len(rows)
                result = {"scope": "same-training-set-loss-readback-only", "label": label, "items": len(rows),
                          "loss": loss_sum / len(rows), "weighted_contribution_by_class": dict(by_class),
                          "independent_development": False, "generalization_metric": False}
                append_json(output / "readback.jsonl", result)
                model.train()
                return result

            status["initial_checkpoint"] = checkpoint(0)
            status["initial_training_set_readback"] = readback("before-fixed-training")
            save_json(status_path, status)
            exposures = Counter()
            for epoch in range(epochs):
                order = recipe["epoch_row_order"][epoch]
                epoch_losses, frame_exposures = [], 0
                for start in range(0, len(order), recipe["batch_size"]):
                    require(time.monotonic() - started < args.seconds, "training item wall-clock budget exhausted")
                    indices = order[start:start + recipe["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    loss, contributions = batch_loss(indices)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_max_norm"], error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "nonfinite updated parameter; no retry")
                    status["optimizer_steps"] += 1
                    status["training_sample_exposures"] += len(indices)
                    epoch_losses.append(loss.detach().item())
                    draws = []
                    for contribution in contributions:
                        index = contribution["row_index"]
                        row, policy = rows[index], recipe["items"][index]
                        exposures[row["source_id"]] += 1
                        frame_exposures += cached[index].shape[0]
                        draws.append({**contribution, "item_id": row["item_id"], "source_id": row["source_id"],
                                      "source_family_id": row["source_family_id"], "speaker_id": row["speaker_id"],
                                      "original_wav_sha256": row["original_wav_sha256"], "pcm_sha256": row["pcm_sha256"],
                                      "feature_sha256": observations[index]["feature_sha256"],
                                      "label_binding_sha256": policy["label_binding_sha256"],
                                      "ctc_mask": row["ctc_mask"], "asr_ctc_weight": row["ctc_weight"],
                                      "effective_ctc_weight": policy["effective_ctc_weight"],
                                      "keyword_targets": row["keyword_targets"], "keyword_mask": row["keyword_mask"],
                                      "asr_keyword_weight": row["keyword_weight"], "effective_keyword_weight": policy["effective_keyword_weight"],
                                      "class": policy["class"], "class_multiplier": policy["class_multiplier"]})
                    append_json(output / "consumption.jsonl", {"epoch": epoch + 1, "optimizer_step": status["optimizer_steps"],
                        "loss": loss.detach().item(), "gradient_norm_before_clip": norm.item(), "draws": draws})
                    save_json(status_path, status)
                saved = checkpoint(epoch + 1)
                append_json(output / "epochs.jsonl", {"epoch": epoch + 1, "mean_step_loss": sum(epoch_losses) / len(epoch_losses),
                            "optimizer_steps": status["optimizer_steps"], "training_sample_exposures": status["training_sample_exposures"],
                            "feature_frame_exposures_this_epoch": frame_exposures, "checkpoint": saved})
                status["final_checkpoint"] = saved
            status["final_training_set_readback"] = readback("after-fixed-final-epoch")
            if frozen_encoder is not None:
                require(state_identity(frozen_encoder.state_dict()) == frozen_identity, "keyword probe changed its frozen encoder")
                status["frozen_encoder_unchanged"] = True
            status.update(status="complete", stage="complete", finished_at=utc(), elapsed_seconds=round(time.monotonic() - started, 6),
                          sample_exposures_by_source=dict(exposures), unique_training_sources=len(exposures),
                          training_set_readback_forward_exposures=2 * len(rows),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          final_state_sha256=state_identity(model.state_dict())["sha256"])
            require(status["initial_state_sha256"] != status["final_state_sha256"], "optimizer did not change model state")
            save_json(status_path, status)
            print(json.dumps({key: status[key] for key in ("kind", "status", "optimizer_steps", "training_sample_exposures", "elapsed_seconds", "max_rss_kib")}), flush=True)
        except BaseException as error:
            status.update(status="failed", stage=status.get("stage", "unknown"), error_type=type(error).__name__, error=str(error),
                          finished_at=utc(), elapsed_seconds=round(time.monotonic() - started, 6),
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save_json(status_path, status)
            raise
    return 0


def stage_command(command, *, log, lease, seconds):
    """Keep supervised stages in the queue group and inherit its model slot.

    These stage workers do not launch child processes. Standalone runs own a
    stage group; supervised runs let the outer queue bound the complete group.
    """
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=not lease.supervised, env=lease.child_environment(), pass_fds=lease.pass_fds)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        if lease.supervised:
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return 124
    except BaseException:
        if lease.supervised:
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def run(args):
    args.heavy_lock.parent.mkdir(parents=True, exist_ok=True)
    with heavy_model_slot(args.heavy_lock) as lease:
        return run_with_slot(args, lease)


def run_with_slot(args, lease):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    prepared, output = args.prepared.resolve(), args.output.resolve()
    record, _, recipe = verified_prepared(prepared)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "d0-weak-training-controller-v1", "started_at": utc(), "status": "running", "runs": {},
               "recipe_sha256": record["recipe_sha256"], "per_item_seconds_cap": args.seconds,
               "heavy_model_concurrency": 1, "independent_development_items": 0, "quality_winner": None}
    save_json(output / "summary.json", summary)
    for kind in ("rnn64", "c1-ctc", "c1-keyword-head"):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--prepared", str(prepared),
                   "--output", str(output / kind), "--kind", kind, "--seconds", str(args.seconds), "--heavy-lock", str(args.heavy_lock.resolve())]
        if kind == "c1-keyword-head":
            command += ["--encoder-checkpoint", str(output / "c1-ctc/checkpoints" / f"epoch-{recipe['epochs']:03d}.pt")]
        with (output / (kind + ".log")).open("x") as log:
            started = time.monotonic()
            code = stage_command(command, log=log, lease=lease, seconds=args.seconds)
            status_path = output / kind / "status.json"
            if code != 0 and status_path.is_file():
                interrupted = json_read(status_path)
                if interrupted["status"] == "running":
                    interrupted.update(status="budget-exhausted" if code == 124 else "failed",
                                       controller_exit_code=code, finished_at=utc(),
                                       error="worker stopped before its completion receipt; no automatic retry")
                    save_json(status_path, interrupted)
            summary["runs"][kind] = {"exit_code": code, "wall_seconds": round(time.monotonic() - started, 6)}
        save_json(output / "summary.json", summary)
        print(json.dumps({"kind": kind, **summary["runs"][kind]}), flush=True)
        if code != 0:
            break
    summary.update(status="complete" if len(summary["runs"]) == 3 and all(row["exit_code"] == 0 for row in summary["runs"].values()) else "incomplete", finished_at=utc())
    if summary["status"] == "complete":
        left, right = (json_read(output / kind / "status.json") for kind in ("rnn64", "c1-ctc"))
        for key in ("recipe_sha256", "optimizer_steps", "training_sample_exposures", "sample_exposures_by_source", "optimizer"):
            require(left[key] == right[key], "main-pair exposure or optimizer mismatch: " + key)
        summary["main_pair_same_recipe_exposure_optimizer"] = True
        summary["main_pair_model_selection_performed"] = False
    save_json(output / "summary.json", summary)
    return 0 if summary["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--view", type=Path, required=True)
    prep.add_argument("--view-manifest-sha256", required=True)
    prep.add_argument("--view-loader", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--heavy-lock", type=Path, required=True)
        command.add_argument("--seconds", type=int, default=600)
        if name == "worker":
            command.add_argument("--kind", choices=("rnn64", "c1-ctc", "c1-keyword-head"), required=True)
            command.add_argument("--encoder-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.command != "prepare" and not 1 <= args.seconds <= 600:
        parser.error("each initial training item is capped at 600 seconds")
    return {"prepare": prepare, "run": run, "worker": worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
