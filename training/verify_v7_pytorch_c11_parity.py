"""PyTorch-to-C11 frame-logit and greedy-sequence parity over TRAIN+DEV only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import numpy as np
import torch
import english_kws_dual_track_v7_train as trainer


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def run(snapshot: Path, candidate: str, model_dir: Path, c11_runner: Path,
        output: Path, atol: float, rtol: float):
    plan = trainer.load_plan()
    manifest, snapshot_info = trainer.verify_training_snapshot(snapshot, candidate, plan)
    candidate_plan = plan["candidates"][candidate]
    candidate_id = candidate_plan["keyword_id"]
    pack_path = model_dir / "candidate.kwv7"
    pt_path = model_dir / "candidate.pt"
    npz_path = model_dir / "candidate.npz"
    for path in (pack_path, pt_path, npz_path, c11_runner):
        if not path.is_file() or path.is_symlink():
            raise ValueError("REGULAR_PARITY_INPUT:" + path.name)

    pack_raw = pack_path.read_bytes()
    if len(pack_raw) != plan["pack"]["total_bytes"] or pack_raw[:8] != b"KWSV7C01":
        raise ValueError("EXACT_CANDIDATE_PACK_GEOMETRY")
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model = trainer.build_model(torch)
    model.load_state_dict(state, strict=True)
    model.eval()
    with np.load(npz_path, allow_pickle=False) as saved:
        bank = saved["mel_filters"].astype(np.float32, copy=True)
        mean = saved["train_mean"].astype(np.float32, copy=True)
        scale = saved["train_scale"].astype(np.float32, copy=True)
        for key, tensor in state.items():
            if key not in saved.files or not np.array_equal(saved[key], tensor.detach().cpu().numpy()):
                raise ValueError("NPZ_PT_WEIGHT_MISMATCH:" + key)
    if bank.shape != (257, 40) or mean.shape != (40,) or scale.shape != (40,):
        raise ValueError("EXACT_FRONTEND_ARRAY_GEOMETRY")

    rows = manifest["rows"]
    if not rows or any(row["role"] not in ("train", "development") for row in rows):
        raise ValueError("TRAIN_DEV_ONLY_PARITY_ROWS")
    if any(row["role"] == "train" and row["training_eligible"] is not True for row in rows):
        raise ValueError("TRAIN_ROWS_MUST_BE_OPTIMIZER_ELIGIBLE")
    if any(row["role"] == "development" and row["training_class"] is not None for row in rows):
        raise ValueError("DEVELOPMENT_LABELS_NOT_OPTIMIZER_VISIBLE")

    details = []
    frames_total = 0
    max_abs = 0.0
    mean_abs_sum = 0.0
    scalar_count = 0
    tokens_equal = True
    events_equal = True
    numerical_equal = True
    with tempfile.TemporaryDirectory(prefix="v7-c11-parity-") as temp:
        temp = Path(temp)
        for index, row in enumerate(rows):
            pcm = trainer.read_pcm(row, snapshot)
            features = trainer.extract_features(pcm, bank, torch, np)
            normalized = ((features - mean) / scale).astype(np.float32)
            input_tensor = torch.from_numpy(normalized).unsqueeze(0)
            with torch.no_grad():
                reference = model(input_tensor).cpu().numpy()[0].astype(np.float32, copy=False)

            pcm_path = temp / f"row-{index}.pcm"
            logits_path = temp / f"row-{index}.logits.f32"
            pcm_path.write_bytes(pcm)
            completed = subprocess.run(
                [c11_runner.as_posix(), pack_path.as_posix(), pcm_path.as_posix(),
                 str(candidate_id), logits_path.as_posix()],
                check=False, capture_output=True, text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError("C11_RUNNER_FAILED:" + completed.stderr[-1000:])
            c11 = np.fromfile(logits_path, dtype="<f4")
            if c11.size % 4:
                raise ValueError("C11_LOGIT_FILE_GEOMETRY")
            c11 = c11.reshape((-1, 4))
            if c11.shape != reference.shape:
                raise ValueError("PYTORCH_C11_FRAME_COUNT_MISMATCH:" + row["id"])
            if not np.isfinite(c11).all() or not np.isfinite(reference).all():
                raise ValueError("NONFINITE_PARITY_LOGITS:" + row["id"])

            delta = np.abs(c11 - reference)
            row_max = float(delta.max(initial=0.0))
            row_mean = float(delta.mean()) if delta.size else 0.0
            row_numerical_match = bool(np.allclose(c11, reference, rtol=rtol, atol=atol))
            numerical_equal = numerical_equal and row_numerical_match
            max_abs = max(max_abs, row_max)
            mean_abs_sum += float(delta.sum())
            scalar_count += int(delta.size)
            frames_total += int(c11.shape[0])

            ref_tokens, ref_events = trainer.greedy_sequence(torch.from_numpy(reference).unsqueeze(0), torch)
            c_tokens, c_events = trainer.greedy_sequence(torch.from_numpy(c11).unsqueeze(0), torch)
            token_match = ref_tokens == c_tokens
            event_match = ref_events == c_events
            tokens_equal = tokens_equal and token_match
            events_equal = events_equal and event_match
            details.append({
                "row_id": row["id"],
                "role": row["role"],
                "source_group_id": row["source_group_id"],
                "frames": int(c11.shape[0]),
                "max_abs_logit_error": row_max,
                "mean_abs_logit_error": row_mean,
                "numerical_tolerance_pass": row_numerical_match,
                "token_sequence_equal": token_match,
                "full_sequence_event_frames_equal": event_match,
            })

    mean_abs = mean_abs_sum / scalar_count if scalar_count else 0.0
    numerical_ok = numerical_equal
    passed = numerical_ok and tokens_equal and events_equal
    report = {
        "schema": "english-kws-dual-track-v7-pytorch-c11-parity-v1",
        "status": "PASS" if passed else "FAIL",
        "candidate": candidate,
        "candidate_id": candidate_plan["candidate_id"],
        "keyword_id": candidate_id,
        "training_snapshot_sha256": snapshot_info["snapshot_sha256"],
        "source_candidate_manifest_sha256": snapshot_info["source_candidate_manifest_sha256"],
        "model_pack_sha256": sha(pack_raw),
        "pytorch_state_sha256": sha(pt_path.read_bytes()),
        "npz_sha256": sha(npz_path.read_bytes()),
        "c11_runner_sha256": sha(c11_runner.read_bytes()),
        "rows_compared": len(details),
        "frames_compared": frames_total,
        "max_abs_logit_error": max_abs,
        "mean_abs_logit_error": mean_abs,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "token_sequences_equal": tokens_equal,
        "full_sequence_event_frames_equal": events_equal,
        "opened_roles": ["train", "development"],
        "calibration_audio_opened": False,
        "locked_test_audio_opened": False,
        "product_promotion_allowed": False,
        "rows": details,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(report) + b"\n")
    print(json.dumps({key: report[key] for key in (
        "status", "candidate", "rows_compared", "frames_compared",
        "max_abs_logit_error", "mean_abs_logit_error",
        "token_sequences_equal", "full_sequence_event_frames_equal",
    )}, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", choices=("VOSEN", "MERO"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--c11-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    args = parser.parse_args()
    run(args.snapshot, args.candidate, args.model_dir, args.c11_runner,
        args.output, args.atol, args.rtol)


if __name__ == "__main__":
    main()
