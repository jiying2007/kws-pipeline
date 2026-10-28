"""Explicit bounded CPU fitting, durable research checkpoints and actual traces.

No Torch import, data access or optimization on import. CLI has no admission
bypass, no automatic resume, no protected split and no qualification result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

from .data import REUSE_PURPOSE, WINDOW_POLICY, load_development, validate_window

SCHEMA = "whole-source-ctc-direct-diagnostic-v1"
SOURCE_PATHS = (
    "research/direct_fit/__init__.py", "research/direct_fit/data.py",
    "research/direct_fit/runner.py", "research/direct_fit/compare.py", "research/direct_fit/registry_data.py",
    "research/direct_kws/model.py", "research/direct_kws/events.py",
    "research/candidate_encoders/contract.py", "research/candidate_encoders/torch_models.py",
    "research/model_comparison/runner.py", "research/model_comparison/compare.py",
    "training/model.py", "training/frontend.py", "training/frontend_spec.py",
    "training/train_ctc.py", "training/training_state.py", "training/training_admission.py",
    "tools/corpus_identity.py", "tools/source_identity.py", "tools/kws_vocab.py", "training/audit_dataset.py",
    "tools/speech_label_admission.py", "tools/speech_asr_consensus.py",
    "tools/asr_source_resolver.py", "tools/asr_native_contracts.py", "eval/score_events.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_hashes(repo):
    return {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def atomic_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name,
                                         delete=False) as stream:
            temporary = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an exact integer in {low}..{high}")
    return value


def validate_config(value):
    from research.direct_kws.model import DirectKWSSpec
    from research.direct_kws.events import DirectEventHead
    fields = {"schema", "development_only", "manifests", "tokens", "evidence_dir",
        "label_reuse_purpose", "coverage_mode", "direct_spec", "window_frames",
        "seed", "epochs", "max_paired_steps", "max_seconds", "learning_rate",
        "direct_event", "ctc_event", "chunk_frames", "input_mode", "registry_inputs"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("diagnostic config missing or extra fields")
    if value["schema"] != SCHEMA or value["development_only"] is not True:
        raise ValueError("explicit non-promotable development diagnostic required")
    from .registry_data import PURPOSE, validate_inputs
    if value["input_mode"] == "reviewed-manifests":
        if value["label_reuse_purpose"] != REUSE_PURPOSE or value["registry_inputs"] is not None:
            raise ValueError("reviewed direct label-fact reuse purpose must be explicit")
    elif value["input_mode"] == "registry-exposed-fit":
        if (value["label_reuse_purpose"] != PURPOSE or value["manifests"] != []
                or value["evidence_dir"] is not None or value["coverage_mode"] != "incomplete-coverage-fit-only"):
            raise ValueError("registry exposed-fit must retain incomplete-coverage and separate input purpose")
        validate_inputs(value["registry_inputs"])
    else:
        raise ValueError("unknown diagnostic input_mode")
    if value["coverage_mode"] not in ("all-three-classes", "incomplete-coverage-fit-only"):
        raise ValueError("coverage_mode must be explicit")
    if (type(value["manifests"]) is not list or not (0 if value["input_mode"] == "registry-exposed-fit" else 1) <= len(value["manifests"]) <= 4
            or any(type(p) is not str or not p for p in value["manifests"])):
        raise ValueError("one to four explicit manifests required")
    if type(value["tokens"]) is not str or not value["tokens"]:
        raise ValueError("token file required")
    if value["evidence_dir"] is not None and (type(value["evidence_dir"]) is not str or
                                               not value["evidence_dir"]):
        raise ValueError("evidence_dir must be a directory path or null")
    spec = DirectKWSSpec.from_dict(value["direct_spec"])
    if spec.backbone_spec().parameter_elements > 65536 or spec.backbone_spec().cache_elements > 16384:
        raise ValueError("topology exceeds bounded CPU diagnostic allocation")
    validate_window(spec, value["window_frames"])
    integer(value["seed"], "seed", 0, 2**31-1)
    integer(value["epochs"], "epochs", 1, 100)
    integer(value["max_paired_steps"], "max_paired_steps", 1, 100000)
    integer(value["chunk_frames"], "chunk_frames", 1, 256)
    for field, upper in (("learning_rate", 1.0), ("max_seconds", 86400.0)):
        number = value[field]
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= upper:
            raise ValueError(f"{field} out of bounded finite range")
    de = value["direct_event"]
    if type(de) is not dict or set(de) != {"thresholds", "min_consecutive_frames", "cooldown_frames"}:
        raise ValueError("direct event settings must be explicit")
    DirectEventHead(**de)
    ce = value["ctc_event"]
    if type(ce) is not dict or set(ce) != {"thresholds", "max_frames", "cooldown_frames"}:
        raise ValueError("CTC event settings must be explicit")
    integer(ce["max_frames"], "CTC max_frames", 1, 1000)
    integer(ce["cooldown_frames"], "CTC cooldown_frames", 0, 1000)
    DirectEventHead(ce["thresholds"], min_consecutive_frames=1, cooldown_frames=ce["cooldown_frames"])
    if ce["cooldown_frames"] != de["cooldown_frames"]:
        raise ValueError("paired event policies must share the declared refractory duration")
    return spec


def paired_models(spec, seed, torch):
    from research.candidate_encoders.torch_models import build_encoder
    from research.direct_kws.model import build_classifier
    torch.manual_seed(seed)
    ctc = build_encoder(spec.backbone_spec())
    torch.manual_seed(seed)
    direct = build_classifier(spec)
    for name, tensor in ctc.state_dict().items():
        if name.startswith(("in_proj.", "blocks.")) and not torch.equal(tensor, direct.state_dict()[name]):
            raise ValueError("shared backbone initialization is not identical")
    return {"ctc": ctc, "direct": direct}


def checkpoint(model, optimizer, generator, *, arm, config, receipt, sources,
               initial_identity, steps, row_count, schedule_hash, status, torch):
    from training_state import capture_completion_state, state_identity
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    completion = capture_completion_state(model, optimizer, generator,
        completed_epochs=steps // row_count, optimizer_steps=steps)
    # The helper records tensor/RNG state correctly at any endpoint; replace its
    # epoch-only policy when we stop inside an epoch, never claim that epoch done.
    completion.update(policy="diagnostic-cpu-endpoint-state-v1", epoch_offset=steps % row_count,
                      schedule_policy="saved-precomputed-row-order-v1")
    return {"schema": SCHEMA, "model_family": "direct-kws" if arm == "direct" else "ctc",
        "arm": arm, "objective": "terminal-window-cross-entropy" if arm == "direct" else "ctc",
        "product_export_compatible": False, "promotion_allowed": False,
        "resume_supported": False, "run_status": status, "config": config,
        "direct_spec": config["direct_spec"], "data_sha256": receipt["data_sha256"],
        "data_evidence_scope": receipt["schema"], "source_hashes": sources,
        "initial_float_state_identity": initial_identity, "float_state_identity": state_identity(state),
        "state_dict": state, "completion_state": completion, "schedule_sha256": schedule_hash,
        "runtime": {"torch_version": str(torch.__version__), "device": "cpu", "dtype": "float32"}}


def load_model(path, repo, torch):
    from training_state import load_checkpoint_snapshot, state_identity
    from research.direct_kws.model import DirectKWSSpec, build_classifier
    from research.candidate_encoders.torch_models import build_encoder
    saved, raw_sha = load_checkpoint_snapshot(path)
    if (saved.get("schema") != SCHEMA or saved.get("product_export_compatible") is not False
            or saved.get("promotion_allowed") is not False or saved.get("arm") not in ("ctc", "direct")):
        raise ValueError("unsupported/non-diagnostic checkpoint")
    arm = saved["arm"]
    if (saved.get("model_family") != ("direct-kws" if arm == "direct" else "ctc") or
            saved.get("objective") != ("terminal-window-cross-entropy" if arm == "direct" else "ctc")):
        raise ValueError("checkpoint objective/class family mismatch")
    if saved.get("source_hashes") != source_hashes(repo):
        raise ValueError("checkpoint/source mismatch")
    if saved.get("float_state_identity") != state_identity(saved["state_dict"]):
        raise ValueError("checkpoint tensor identity mismatch")
    spec = DirectKWSSpec.from_dict(saved["direct_spec"])
    validate_config(saved["config"])
    if saved["direct_spec"] != saved["config"]["direct_spec"]:
        raise ValueError("checkpoint/config topology mismatch")
    model = build_classifier(spec) if arm == "direct" else build_encoder(spec.backbone_spec())
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved, raw_sha


def trace_model(model, items, recording_names, *, arm, config, torch):
    from research.direct_kws.events import DirectEventHead
    from research.model_comparison.runner import GreedyKeywordDecoder
    rows, events = [], []
    if len(items) != len(recording_names):
        raise ValueError("inference recording count mismatch")
    model.eval()
    with torch.no_grad():
        for (acoustic, _, _), recording in zip(items, recording_names):
            if arm == "direct":
                decoder = DirectEventHead(**config["direct_event"])
            elif arm == "ctc":
                settings = config["ctc_event"]
                words = [{"id": i, "tokens": tokens, "threshold": settings["thresholds"][i-1]}
                         for i, tokens in ((1, [1,2,3,4]), (2, [3,4,3,4]))]
                decoder = GreedyKeywordDecoder(words, max_frames=settings["max_frames"],
                                               cooldown_frames=settings["cooldown_frames"])
            else:
                raise ValueError("unknown inference arm")
            state = model.init_state(1, device=acoustic.device, dtype=acoustic.dtype)
            frame = 0
            for chunk in acoustic.split(config["chunk_frames"]):
                logits, state = model.forward_chunk(chunk.unsqueeze(0), state)
                for logit, probs in zip(logits[:, 0].tolist(), logits[:, 0].softmax(-1).tolist()):
                    if not all(math.isfinite(x) for x in logit + probs):
                        raise ValueError("non-finite inference trace")
                    if arm == "direct":
                        hits = decoder.accept(probs)
                    else:
                        token = max(range(5), key=probs.__getitem__)
                        hits = decoder.accept(token, probs[token])
                    rows.append({"recording": recording, "frame": frame,
                        "input_end_sample": 400 + frame * 320, "logits": logit,
                        "probabilities": probs})
                    events.extend({"recording": recording, **event} for event in hits)
                    frame += 1
    return rows, events


def jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fit_pair(items, receipt, config, output, repo, rt, *, clock=time.monotonic):
    """Internal tensor core. The only CLI caller obtains items via live admission.

    A software test may supply artificial tensors; its receipt must say so.
    Optimizer steps are counted per arm; max_paired_steps bounds complete pairs.
    Wall cap is checked between pairs and is not a hard process timeout.
    """
    from training_state import state_identity, atomic_save_checkpoint
    from research.direct_kws.model import window_cross_entropy
    from .compare import compare_events
    torch = rt[0]
    spec = validate_config(config)
    if not items or len(items) > 10000 or len(items) * config["epochs"] > 100000:
        raise ValueError("invalid row/exposure count")
    for acoustic, target, label in items:
        if (tuple(acoustic.shape) != (config["window_frames"], 32) or acoustic.dtype != torch.float32
                or acoustic.device.type != "cpu" or not torch.isfinite(acoustic).all()
                or target.ndim != 1 or target.dtype != torch.long or target.device.type != "cpu"
                or type(label) is not int or label not in (0,1,2)):
            raise ValueError("invalid training tensor/actual label")
        if bool(((target < 1) | (target > 4)).any()) or (
            len(target) + int((target[1:] == target[:-1]).sum()) > len(acoustic)):
            raise ValueError("invalid/un-alignable CTC target")
    if len(receipt["recordings"]) != len(items):
        raise ValueError("data receipt row count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(repo)
    random.seed(config["seed"])
    models = paired_models(spec, config["seed"], torch)
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=1e-4) for arm, model in models.items()}
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    schedule = [i for _ in range(config["epochs"]) for i in torch.randperm(len(items), generator=generator).tolist()]
    schedule_hash = digest(schedule)
    initial = {arm: state_identity(model.state_dict()) for arm, model in models.items()}
    backbone = state_identity({k: v for k,v in models["ctc"].state_dict().items()
                                if k.startswith(("in_proj.", "blocks."))})
    atomic_json(output / "config.json", config)
    atomic_json(output / "data-receipt.json", receipt)
    atomic_json(output / "schedule.json", {"row_indices": schedule, "sha256": schedule_hash})
    steps = {"ctc": 0, "direct": 0}
    class_updates = {arm: {str(i): 0 for i in range(3)} for arm in models}
    status, error, start = "running", None, clock()
    checkpoint_files = {}
    def save(arm, suffix):
        path = output / f"{arm}.{suffix}.pt"
        value = checkpoint(models[arm], optimizers[arm], generator, arm=arm,
            config=config, receipt=receipt, sources=sources, initial_identity=initial[arm],
            steps=steps[arm], row_count=len(items), schedule_hash=schedule_hash,
            status=status, torch=torch)
        atomic_save_checkpoint(value, path)
        checkpoint_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        for arm in models:
            save(arm, "initial")
        with (output / "optimization-trace.jsonl").open("x", encoding="utf-8") as log:
            for pair_index, row_index in enumerate(schedule):
                if pair_index >= config["max_paired_steps"]:
                    status = "paired-step-cap"
                    break
                if clock() - start >= config["max_seconds"]:
                    status = "wall-time-cap"
                    break
                acoustic, target, label = items[row_index]
                for arm, model in models.items():
                    model.train()
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(acoustic.unsqueeze(0))
                    if arm == "direct":
                        loss = window_cross_entropy(logits, torch.tensor([label], dtype=torch.long))
                    else:
                        loss = torch.nn.functional.ctc_loss(logits.log_softmax(-1), target,
                            torch.tensor([len(acoustic)]), torch.tensor([len(target)]),
                            blank=0, reduction="mean", zero_infinity=False)
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps[arm] += 1
                    class_updates[arm][str(label)] += 1
                    # Validate immediately; never serialize a poisoned endpoint.
                    state_identity(model.state_dict())
                    log.write(json.dumps({"arm": arm, "optimizer_step": steps[arm],
                        "pair_index": pair_index, "row_index": row_index, "class_target": label,
                        "loss": float(loss.detach()), "gradient_norm": float(grad_norm)}, allow_nan=False) + "\n")
                    log.flush()
                os.fsync(log.fileno())
            else:
                status = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        save_errors = {}
        for arm in models:
            try:
                save(arm, "final")
            except (Exception, KeyboardInterrupt) as exc:
                save_errors[arm] = f"{type(exc).__name__}: {exc}"
        if save_errors:
            status = "failed"
        run = {"schema": SCHEMA, "status": status, "error": error,
            "checkpoint_save_errors": save_errors, "optimizer_steps": steps,
            "paired_steps": min(steps.values()), "class_update_counts": class_updates,
            "elapsed_seconds": clock() - start,
            "backbone_initial_identity": backbone, "checkpoints": checkpoint_files,
            "data_sha256": receipt["data_sha256"], "schedule_sha256": schedule_hash,
            "source_hashes": sources, "qualification_allowed": False, "winner": None,
            "wall_cap_scope": "checked between paired updates; no hard timeout",
            "resume_supported": False, "trace_status": "not-started"}
        atomic_json(output / "run.json", run)
    if status == "failed":
        raise RuntimeError("diagnostic failed; inspect retained run.json and initial/final checkpoints")
    # Inference explicitly reloads persisted endpoint bytes, not live models.
    try:
        all_events, trace_files = {}, {}
        names = [row["recording"] for row in receipt["recordings"]]
        duration = (400 + (config["window_frames"] - 1) * 320) / 16000
        references = [{"recording": name, "duration_s": duration, "annotation_status": "complete",
                       "expected_keywords": [item[2]] if item[2] else []}
                      for name, item in zip(names, items)]
        jsonl(output / "clip-references.jsonl", references)
        for arm in models:
            model, saved, raw_sha = load_model(output / f"{arm}.final.pt", repo, torch)
            if raw_sha != checkpoint_files[f"{arm}.final.pt"] or saved["arm"] != arm:
                raise ValueError("persisted endpoint bytes or arm changed before trace replay")
            if saved["data_sha256"] != receipt["data_sha256"]:
                raise ValueError("reloaded checkpoint data mismatch")
            traces, events = trace_model(model, items, names, arm=arm, config=config, torch=torch)
            for suffix, rows in (("frame-trace", traces), ("events", events)):
                path = output / f"{arm}.{suffix}.jsonl"
                jsonl(path, rows)
                trace_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            all_events[arm] = events
        atomic_json(output / "comparison.json", compare_events(references, all_events, mode="clip-presence"))
        artifact_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in output.iterdir() if path.is_file() and path.name != "run.json"}
        if any(artifact_hashes.get(name) != saved_hash for name, saved_hash in checkpoint_files.items()):
            raise ValueError("persisted checkpoint bytes changed during trace replay")
        run.update(trace_status="completed", traces=trace_files, artifacts=artifact_hashes)
    except (Exception, KeyboardInterrupt) as exc:
        run.update(trace_status="failed", trace_error=f"{type(exc).__name__}: {exc}")
        atomic_json(output / "run.json", run)
        raise
    atomic_json(output / "run.json", run)
    return run


def main():
    from research.model_comparison.runner import runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.config.stat().st_size > 65536:
        raise ValueError("config exceeds 64KiB")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    repo = Path(__file__).resolve().parents[2]
    rt = runtime(repo)
    if config["input_mode"] == "registry-exposed-fit":
        from .registry_data import load_registry_development
        items, receipt = load_registry_development(config, rt, repo)
    else:
        items, receipt = load_development(config, rt)
    if args.preflight_only:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "data-receipt.json", receipt)
        atomic_json(args.output / "preflight.json", {"optimizer_steps": 0,
            "unique_sources": len(items), "coverage": receipt["coverage"],
            "data_sha256": receipt["data_sha256"], "qualification_allowed": False})
    else:
        fit_pair(items, receipt, config, args.output, repo, rt)


if __name__ == "__main__":
    main()
