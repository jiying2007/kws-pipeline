#!/usr/bin/env python3
"""Execute a bounded weak-MIL online proposal experiment, never qualification.

The only optimizer inputs are the original 30 weak train-role recordings.
No new audio, ASR, phonetic labels, event endpoints or formal admission is made.
Actual model execution belongs on the authorized public CPU research runner.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "research" / "english63_weak_pilot"
for directory in (HERE, OLD, ROOT / "training"):
    sys.path.insert(0, str(directory))
import online_contract as contract
from online_contract import require
import english63_weak_train as pilot
import integrated_pcm_encoder as original
import pcm_frontend_parity as pcm
import english_research_label_bridge as bridge

NUMERIC_TOLERANCE = 2e-4
KINDS = contract.KINDS
REPLAY_SCHEMA = "english63-online-saved-weight-replay-v1"
CODE_FILES = (
    "research/english63_online_pilot/online_pilot.py",
    "research/english63_online_pilot/online_contract.py",
    "research/english63_online_pilot/PREDECLARATION.md",
    "research/english63_weak_pilot/english63_weak_train.py",
    "research/english63_weak_pilot/integrated_pcm_encoder.py",
    "research/english63_weak_pilot/pcm_frontend_parity.py",
    "research/english63_weak_pilot/streaming_parity.py",
    "research/english63_weak_pilot/RESULT.json",
    "training/english_research_label_bridge.py", "training/audit_dataset.py",
    "tools/corpus_identity.py", "tools/source_identity.py",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    with path.open("xb") as out:
        out.write(json_bytes(value))


def code_identity():
    return {name: digest((ROOT / name).read_bytes()) for name in CODE_FILES}


def configure_runtime():
    import numpy as np
    import torch
    require(np.__version__ == "2.3.5" and torch.__version__ == "2.10.0+cpu",
            "ONLINE_PINNED_NUMPY_TORCH_REQUIRED")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    return {"numpy": np.__version__, "torch": torch.__version__,
            "python": sys.version.split()[0],
            "device": "cpu", "threads": 1, "mkldnn": False}


def source_audit(source_zip, rows):
    """Reuse the actual project WAV/lineage/PCM/split auditor before training."""
    import audit_dataset
    with tempfile.TemporaryDirectory(prefix="en63-audit-") as temp:
        root = Path(temp)
        records = []
        with zipfile.ZipFile(source_zip) as archive:
            for row in rows:
                cell = row["cell_id"]
                raw = archive.read("english-seven-voice-actual/capture/cells/" +
                                   cell + "/audio-16k.wav")
                require(digest(raw) == row["wav_sha256"], "ONLINE_AUDIT_WAV_HASH")
                relative = cell + ".wav"
                (root / relative).write_bytes(raw)
                records.append({"path": relative, "source_path": relative,
                                "wav_sha256": row["wav_sha256"],
                                "source_wav_sha256": row["wav_sha256"],
                                "source_pcm_sha256": row["pcm_sha256"],
                                "source_family_id": row["source_family_id"],
                                "source_id": cell, "action": row["action"],
                                "human_gold": False, "training_admitted": False})
        manifests = {}
        for name, subset in (
                ("all_source", records),
                ("weak_train", [r for r in records if r["action"] == contract.TRAIN_ACTION]),
                ("exposed_development", [r for r in records if r["action"] == contract.DEV_ACTION])):
            path = root / (name + ".jsonl")
            path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in subset))
            manifests[name] = path
        shared = {"fail_within_split": True, "require_lineage": True,
                  "require_metadata": ("source_family_id",)}
        inventory = audit_dataset.audit_splits([("all_source", manifests["all_source"])], **shared)
        splits = audit_dataset.audit_splits(
            [(name, manifests[name]) for name in ("weak_train", "exposed_development")], **shared)
        require(inventory["clean"] is True and splits["clean"] is True
                and inventory["splits"]["all_source"]["examples"] == 63
                and splits["splits"]["weak_train"]["examples"] == 30
                and splits["splits"]["exposed_development"]["examples"] == 12,
                "ONLINE_DATASET_AUDIT_FAILED")
        # Paths are temporary and not a second source identity. Preserve the
        # actual auditor's observed coverage, including unknown human identity.
        return {"executed": "training/audit_dataset.py:audit_splits",
                "inventory_clean": True, "train_development_clean": True,
                "all_source_rows": 63, "weak_train_rows": 30,
                "exposed_development_rows": 12, "quarantine_excluded_from_training": 21,
                "clean_scope": splits["clean_scope"],
                "identity_disjointness": splits["identity_disjointness"],
                "formal_label_admission": False}


def load_data(source_zip, review_json):
    import numpy as np
    saved = json.loads((OLD / "RESULT.json").read_bytes())
    # Both existing loaders remain unmodified and enforce original byte identities.
    report = bridge.read_verified_report(review_json)
    rows = bridge.diagnostic_plan(report)["rows"]
    train = contract.select_training_rows(rows)
    sources, features, bank, mean, std = original.read_verified_audio(source_zip, review_json, saved)
    by_id = {row["cell_id"]: row for row in rows}
    require(set(by_id) == set(sources) and sum(len(v[2]) for v in sources.values()) == 2101760,
            "ONLINE_SOURCE_INVENTORY")
    for cell, (_, action, _) in sources.items():
        require(by_id[cell]["action"] == action, "ONLINE_ACTION_MISMATCH")
    normed = {cell: ((value-mean)/std).astype(np.float32) for cell, value in features.items()}
    audit = source_audit(source_zip, rows)
    return {"saved": saved, "rows": rows, "train": train, "by_id": by_id,
            "sources": sources, "features": normed, "bank": bank,
            "mean": mean, "std": std, "audit": audit}


def encoder_states(model, kind, x):
    from torch.nn import functional as F
    if kind == "gru":
        return model.rnn(x)[0]
    require(kind == "dsc_tcn", "ONLINE_MODEL_KIND")
    return model.blocks(F.relu(model.proj(x.transpose(1, 2)))).transpose(1, 2)


def frame_logits(model, kind, x):
    """Causal finite-window readout, identical during training and inference."""
    import torch
    from torch.nn import functional as F
    states = encoder_states(model, kind, x).transpose(1, 2)
    sums = F.avg_pool1d(F.pad(states, (contract.WINDOW-1, 0)),
                        contract.WINDOW, stride=1) * contract.WINDOW
    count = torch.arange(1, x.shape[1]+1, dtype=x.dtype, device=x.device)
    count = count.clamp(max=contract.WINDOW).view(1, 1, -1)
    maximum = F.max_pool1d(F.pad(states, (contract.WINDOW-1, 0), value=-float("inf")),
                          contract.WINDOW, stride=1)
    return model.classifier(torch.cat((sums/count, maximum), dim=1).transpose(1, 2))


def bag_logits(logits, lengths):
    """Weak clip MIL on target-vs-background margins; no invented endpoints."""
    import torch
    frames = torch.arange(logits.shape[1], device=logits.device).view(1, -1)
    mask = (frames >= contract.MIN_FRAMES-1) & (frames < lengths.view(-1, 1))
    require(bool((mask.sum(dim=1) > 0).all()), "ONLINE_MIL_TOO_SHORT")
    margins = logits[:, :, 1:] - logits[:, :, :1]
    scaled = (margins/contract.MIL_TEMPERATURE).masked_fill(~mask.unsqueeze(-1), -float("inf"))
    pooled = contract.MIL_TEMPERATURE * (
        torch.logsumexp(scaled, dim=1) - mask.sum(dim=1).log().unsqueeze(-1))
    return torch.cat((torch.zeros_like(pooled[:, :1]), pooled), dim=-1)


class IncrementalModel:
    """Finite tensor carry and hidden state; never accumulates whole-stream maxima."""
    def __init__(self, model, kind):
        import torch
        self.model, self.kind = model, kind
        self.hidden = None
        self.history = None
        self.pool = deque(maxlen=contract.WINDOW)
        self.frames = 0
        self.maximum_pool_frames = 0
        if kind == "dsc_tcn":
            self.history = [torch.zeros((1, pilot.HIDDEN, 2*b.dilation)) for b in model.blocks]

    def consume(self, feature):
        import torch
        from torch.nn import functional as F
        x = torch.as_tensor(feature, dtype=torch.float32).reshape(1, 1, pilot.FREQ_BANDS)
        with torch.no_grad():
            if self.kind == "gru":
                state, self.hidden = self.model.rnn(x, self.hidden)
            else:
                state = F.relu(self.model.proj(x.transpose(1, 2)))
                for i, block in enumerate(self.model.blocks):
                    joined = torch.cat((self.history[i], state), dim=-1)
                    state = state + 0.5*block.pw(F.relu(block.dw(joined)))
                    self.history[i] = joined[:, :, -2*block.dilation:].clone()
                state = state.transpose(1, 2)
            self.pool.append(state[0, 0].clone())
            current = torch.stack(tuple(self.pool))
            pooled = torch.cat((current.mean(dim=0), current.max(dim=0).values))
            output = self.model.classifier(pooled)
        self.frames += 1
        self.maximum_pool_frames = max(self.maximum_pool_frames, len(self.pool))
        require(len(self.pool) <= contract.WINDOW and bool(torch.isfinite(output).all()),
                "ONLINE_STATE_OR_LOGITS")
        return output


def numeric_self_test():
    """Runner-only meaningful gradient, finite-history and prefix-equivalence check."""
    import torch
    from torch.nn import functional as F
    results = {}
    for i, kind in enumerate(KINDS):
        torch.manual_seed(91700+i)
        model = pilot.create_model(kind)
        x = torch.randn(2, 241, pilot.FREQ_BANDS) * 0.3
        lengths = torch.tensor([241, 137])
        full = frame_logits(model, kind, x)
        loss = F.cross_entropy(bag_logits(full, lengths), torch.tensor([1, 0]))
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()), "ONLINE_MIL_GRADIENT")
        model.eval()
        state = IncrementalModel(model, kind)
        with torch.no_grad():
            streamed = torch.stack([state.consume(frame) for frame in x[0]])
            prefix = frame_logits(model, kind, x[:1, :103])[0]
        error = float((streamed-full[0].detach()).abs().max())
        prefix_error = float((prefix-full[0, :103].detach()).abs().max())
        require(error <= NUMERIC_TOLERANCE and prefix_error <= NUMERIC_TOLERANCE
                and len(state.pool) == contract.WINDOW, "ONLINE_NUMERIC_SELF_TEST")
        results[kind] = {"gradient_finite": True, "prefix_max_abs_diff": prefix_error,
                         "batch_stream_max_abs_diff": error, "retained_pool_frames": len(state.pool)}
    return results


def chunk_plan(length, plan):
    pattern = (1, 159, 161, 3199, 320, 7) if plan == "mixed" else (plan,)
    offset, index = 0, 0
    while offset < length:
        n = min(pattern[index % len(pattern)], length-offset)
        yield offset, offset+n
        offset += n
        index += 1


def replay_stream(model, kind, arrays, data, plan):
    import numpy as np
    import torch
    framer = contract.BoundedFrames()
    encoder = IncrementalModel(model, kind)
    decoder = contract.ProposalDecoder()
    trace = hashlib.sha256()
    events = []
    for samples in arrays:
        for begin, end in chunk_plan(len(samples), plan):
            for number, window in framer.push(samples[begin:end]):
                # One frame per numeric call makes output independent of caller chunks.
                feature = pcm.logmel(np.asarray([window], dtype=np.float32), data["bank"])[0]
                normalized = ((feature-data["mean"])/data["std"]).astype(np.float32)
                logits = encoder.consume(normalized)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                trace.update(probabilities.astype("<f4").tobytes())
                event = decoder.consume(probabilities.tolist())
                require(number == encoder.frames == decoder.frame, "ONLINE_FRAME_CLOCK_DRIFT")
                if event is not None:
                    events.append(event)
    before = (encoder.frames, len(events), decoder.frame, framer.samples, len(framer.tail))
    frames = framer.finish()
    require(before == (encoder.frames, len(events), decoder.frame, framer.samples, len(framer.tail))
            and frames == encoder.frames, "ONLINE_EOF_EMISSION")
    return {"frames": frames, "actual_pcm_samples": framer.samples,
            "trace_sha256": trace.hexdigest(), "proposals": events,
            "maximum_pool_frames": encoder.maximum_pool_frames,
            "maximum_pcm_buffer_samples": framer.maximum_buffer_samples,
            "pending_pcm_samples": len(framer.tail), "eof_proposals": 0}


def compare_replays(left, right):
    require(all(left[key] == right[key] for key in
                ("frames", "actual_pcm_samples", "trace_sha256", "proposals",
                 "maximum_pool_frames", "pending_pcm_samples", "eof_proposals")),
            "ONLINE_CHUNK_REPLAY_DRIFT")


def evaluate(model, kind, data):
    """Only observed proposals/weak clip agreement; no acoustic error-rate score."""
    import torch
    clip_rows, events = [], []
    max_buffer = 0
    for cell in sorted(data["sources"]):
        samples = data["sources"][cell][2]
        first = replay_stream(model, kind, [samples], data, 160)
        second = replay_stream(model, kind, [samples], data, "mixed")
        compare_replays(first, second)
        max_buffer = max(max_buffer, second["maximum_pcm_buffer_samples"])
        row = data["by_id"][cell]
        feature = torch.from_numpy(data["features"][cell]).unsqueeze(0)
        with torch.no_grad():
            logits = frame_logits(model, kind, feature)
            weak = int(bag_logits(logits, torch.tensor([feature.shape[1]])).argmax(dim=-1))
        clip_rows.append({"cell_id": cell, "action": row["action"],
                          "requested_weak_intent_id": row["weak_wholeclip_class"],
                          "weak_bag_prediction": weak, "frames": first["frames"],
                          "trace_sha256": first["trace_sha256"],
                          "proposal_count": len(first["proposals"]),
                          "prediction_is_acoustic_truth": False})
        events.extend({"kind": kind, "stream": "isolated_original_clip",
                       "cell_id": cell, **event} for event in first["proposals"])
    sequence = [data["sources"][cell][2] for cell in sorted(data["sources"])]
    first = replay_stream(model, kind, sequence, data, 160)
    second = replay_stream(model, kind, sequence, data, "mixed")
    compare_replays(first, second)
    require(first["frames"] > 1200 and first["maximum_pool_frames"] == contract.WINDOW,
            "ONLINE_LONG_STREAM_NOT_EXERCISED")
    events.extend({"kind": kind, "stream": "artificial_sorted_source_concatenation",
                   "cell_id": None, **event} for event in first["proposals"])
    observations = {"isolated_clips": clip_rows, "clip_chunk_comparisons": 63,
                    "continuous_chunk_comparisons": 1,
                    "continuous_frames": first["frames"],
                    "continuous_trace_sha256": first["trace_sha256"],
                    "continuous_proposal_count": len(first["proposals"]),
                    "source_order": sorted(data["sources"]),
                    "continuous_input_is_artificial_concatenation": True,
                    "source_hour_multiplier": 1,
                    "continuous_state_resets": 0,
                    "maximum_pool_frames": first["maximum_pool_frames"],
                    "maximum_pcm_buffer_samples": max(max_buffer, second["maximum_pcm_buffer_samples"]),
                    "state_memory_is_not_process_rss": True,
                    "eof_proposals": 0, "events_are_research_proposals_only": True}
    return observations, events


def load_new_weights(path, kind, sha):
    import numpy as np
    import torch
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 500000
            and digest(path.read_bytes()) == sha, "ONLINE_NEW_WEIGHT_IDENTITY")
    model = pilot.create_model(kind)
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        expected = model.state_dict()
        require(set(archive.files) == set(expected), "ONLINE_NEW_WEIGHT_TENSORS")
        state = {}
        for name, value in expected.items():
            array = archive[name]
            require(array.dtype == np.float32 and tuple(array.shape) == tuple(value.shape)
                    and bool(np.isfinite(array).all()), "ONLINE_NEW_WEIGHT_VALUES")
            state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def train_one(kind, data, output):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(contract.SEED + (0 if kind == "gru" else 101))
    model, _, warm_sha = original.load_models(data["saved"])[kind]
    train = contract.select_training_rows(data["rows"])
    x, lengths = pilot.pad_tensor(train, data["features"])
    labels = torch.tensor([r["weak_wholeclip_class"] for r in train], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract.LEARNING_RATE, weight_decay=0.01)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    losses = []
    start = time.monotonic()
    model.train()
    for step in range(contract.STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(bag_logits(frame_logits(model, kind, x), lengths), labels)
        require(bool(torch.isfinite(loss)), "ONLINE_TRAIN_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step+1) % 12 == 0:
            print(json.dumps({"kind": kind, "optimizer_step": step+1,
                              "weak_mil_loss": losses[-1], "train_rows": 30,
                              "formal_training_admitted": False}), flush=True)
    model.eval()
    changed = sum(not torch.equal(initial[key], value) for key, value in model.state_dict().items())
    require(changed > 0, "ONLINE_NO_OPTIMIZER_EFFECT")
    weight = output / (kind + "-weights.npz")
    weight_sha, shapes = pilot.save_safe_npz(weight, model)
    require(weight_sha != warm_sha, "ONLINE_UNCHANGED_CHECKPOINT")
    elapsed = time.monotonic()-start
    observations, events = evaluate(model, kind, data)
    result = {"kind": kind, "optimizer_updates": contract.STEPS, "train_rows": 30,
              "train_nearphone_rows": 0, "train_development_rows": 0,
              "train_cell_ids": [r["cell_id"] for r in train],
              "seed": contract.SEED+(0 if kind == "gru" else 101),
              "weight_file": weight.name, "weights_sha256": weight_sha,
              "warm_start_sha256": warm_sha, "tensor_shapes": shapes,
              "changed_state_tensors": changed,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "weak_mil_loss_initial": losses[0], "weak_mil_loss_last": losses[-1],
              "weak_mil_loss_history": losses, "training_wall_seconds": elapsed,
              "observations": observations, "model_promotion_allowed": False}
    return result, events


def event_bytes(events):
    return b"".join((json.dumps(event, sort_keys=True, allow_nan=False)+"\n").encode()
                    for event in events)


def verify_saved(output, data, runtime):
    summary_path = output / "SUMMARY.json"
    require(output.is_dir() and not output.is_symlink() and summary_path.is_file()
            and not summary_path.is_symlink(), "ONLINE_REPLAY_PATH")
    summary = json.loads(summary_path.read_bytes())
    contract.validate_summary(summary)
    require(summary["code_sha256"] == code_identity()
            and summary["source_tts_zip_sha256"] == pcm.TTS_SHA256
            and summary["source_review_json_sha256"] == pcm.REVIEW_SHA256
            and summary["runtime"] == runtime and summary["dataset_audit"] == data["audit"],
            "ONLINE_REPLAY_SOURCE_OR_RUNTIME")
    models, events = [], []
    for record in summary["models"]:
        kind = record["kind"]
        require(record["train_cell_ids"] == [r["cell_id"] for r in data["train"]],
                "ONLINE_REPLAY_TRAIN_ROSTER")
        model = load_new_weights(output / record["weight_file"], kind, record["weights_sha256"])
        observations, seen = evaluate(model, kind, data)
        require(observations == record["observations"], "ONLINE_SAVED_PREDICTION_DRIFT")
        events.extend(seen)
        models.append({"kind": kind, "weights_sha256": record["weights_sha256"],
                       "clip_comparisons": 63, "continuous_comparisons": 1,
                       "saved_predictions_reproduced": True})
    raw = event_bytes(events)
    require(digest(raw) == summary["events_sha256"]
            and (output / "events.jsonl").read_bytes() == raw, "ONLINE_SAVED_EVENTS_DRIFT")
    return {"schema": REPLAY_SCHEMA, "summary_sha256": digest(summary_path.read_bytes()),
            "events_sha256": digest(raw), "models": models, "optimizer_updates": 0,
            "new_audio_asr_calls": 0, "formal_training_admitted": False,
            "model_promotion_allowed": False, "status": "REPRODUCED_SAVED_WEAK_PROPOSALS"}


def run(source_zip, review_json, output, runtime):
    require(output.is_absolute() and output.parent.is_dir() and not output.exists()
            and not output.parent.is_symlink(), "ONLINE_OUTPUT_CREATE_ONLY")
    data = load_data(source_zip, review_json)
    self_test = numeric_self_test()
    output.mkdir(mode=0o700)
    try:
        models, events = [], []
        for kind in KINDS:
            result, proposals = train_one(kind, data, output)
            models.append(result)
            events.extend(proposals)
        raw = event_bytes(events)
        (output / "events.jsonl").write_bytes(raw)
        summary = {"schema": contract.POLICY, "status": contract.STATUS,
                   "config": contract.fixed_config(), "code_sha256": code_identity(),
                   "source_tts_zip_sha256": pcm.TTS_SHA256,
                   "source_review_json_sha256": pcm.REVIEW_SHA256,
                   "source_audio_clips": 63, "source_pcm_samples": 2101760,
                   "role_counts": contract.COUNTS, "dataset_audit": data["audit"],
                   "runtime": runtime, "numeric_self_test": self_test,
                   "events_sha256": digest(raw), "models": models,
                   "formal_training_admitted": False, "human_gold": False,
                   "model_promotion_allowed": False, "model_selection_made": False,
                   "new_tts_asr_ipa_calls": 0, "threshold_selection": False,
                   **{key: "NOT_RUN" for key in contract.UNMEASURED}}
        contract.validate_summary(summary)
        write_json(output / "SUMMARY.json", summary)
        replay = verify_saved(output, data, runtime)
        write_json(output / "REPLAY.json", replay)
        return {"status": contract.STATUS, "output_files": sorted(p.name for p in output.iterdir()),
                "summary_sha256": digest((output / "SUMMARY.json").read_bytes()),
                "model_promotion_allowed": False}
    except Exception as exc:
        write_json(output / "FAILED.json", {"schema": contract.POLICY,
                   "status": "FAILED_PARTIAL_NOT_A_SUCCESS_RESULT", "error": str(exc),
                   "model_promotion_allowed": False, "formal_training_admitted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-zip", required=True, type=Path)
    parser.add_argument("--review-json", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-result-dir", type=Path)
    args = parser.parse_args()
    try:
        runtime = configure_runtime()
        if args.verify_result_dir is not None:
            data = load_data(args.tts_zip, args.review_json)
            require(not (args.verify_result_dir / "FAILED.json").exists(), "ONLINE_FAILED_RESULT")
            receipt = verify_saved(args.verify_result_dir, data, runtime)
            require(json.loads((args.verify_result_dir / "REPLAY.json").read_bytes()) == receipt,
                    "ONLINE_REPLAY_RECEIPT_DRIFT")
            result = receipt
        else:
            result = run(args.tts_zip, args.review_json, args.output_dir, runtime)
    except Exception as exc:
        print("ONLINE_PILOT_FAILED: " + str(exc), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
