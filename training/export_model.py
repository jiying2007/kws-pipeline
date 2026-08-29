#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import corpus_digest  # noqa: E402
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

from frontend_spec import FRONTEND_IDS

MODEL_VERSION = 2
MODEL_HEADER_BYTES = 72
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FRONTEND_SPEC_VERSION = 2
MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
SOURCE_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def q8(tensor: torch.Tensor, name: str) -> tuple[bytes, float, dict]:
    tensor = finite_tensor(tensor, name)
    maximum = max(float(tensor.abs().max()), 1.0e-8)
    scale = maximum / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    dequantized = quantized.float() * scale
    error = dequantized - tensor
    rmse = math.sqrt(float(error.square().mean()))
    signal_rms = math.sqrt(float(tensor.square().mean()))
    stats = {
        "scale": scale,
        "max_abs_error": float(error.abs().max()),
        "rmse": rmse,
        "signal_rms": signal_rms,
        "snr_db": 20.0 * math.log10(signal_rms / max(rmse, 1.0e-12))
        if signal_rms > 0.0
        else 0.0,
    }
    if not all(math.isfinite(float(value)) for value in stats.values()):
        raise ValueError(f"{name}: non-finite quantization statistics")
    return quantized.numpy().tobytes(), scale, stats


def float_bytes(tensor: torch.Tensor, name: str) -> bytes:
    return finite_tensor(tensor, name).numpy().tobytes()


def require_shape(state_dict: dict, name: str, shape: tuple[int, ...]) -> None:
    if name not in state_dict:
        raise ValueError(f"checkpoint missing {name}")
    actual = tuple(int(value) for value in state_dict[name].shape)
    if actual != shape:
        raise ValueError(f"{name} shape {actual} does not match expected {shape}")


def align4(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def checkpoint_sha(value, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"checkpoint {label} must be lowercase SHA256 hex")
    return value


def training_environment(checkpoint: dict) -> dict:
    value = checkpoint.get("training_environment")
    if value is None:
        return {"recorded": False}
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("checkpoint training_environment must be schema_version 1")
    required_text = (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "torch_version",
    )
    result: dict = {"recorded": True, "schema_version": 1}
    for key in required_text:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"checkpoint training_environment.{key} is invalid")
        result[key] = item
    for key in ("cuda_version", "repository_sha", "training_image_digest"):
        item = value.get(key)
        if item is not None and not isinstance(item, str):
            raise ValueError(f"checkpoint training_environment.{key} is invalid")
        result[key] = item
    if result["repository_sha"] is not None and SOURCE_SHA_RE.fullmatch(result["repository_sha"]) is None:
        raise ValueError("checkpoint training_environment.repository_sha is invalid")
    if result["training_image_digest"] is not None and IMAGE_DIGEST_RE.fullmatch(result["training_image_digest"]) is None:
        raise ValueError("checkpoint training_environment.training_image_digest is invalid")
    result["cudnn_version"] = value.get("cudnn_version")
    result["torch_num_threads"] = int(value.get("torch_num_threads", 0))
    result["torch_num_interop_threads"] = int(value.get("torch_num_interop_threads", 0))
    result["container_declared"] = bool(value.get("container_declared", False))
    for key in ("requirements_lock_sha256", "dockerfile_sha256"):
        item = value.get(key)
        result[key] = None if item is None else checkpoint_sha(item, f"training_environment.{key}")
    code = value.get("training_code_sha256")
    if not isinstance(code, dict) or not code:
        raise ValueError("checkpoint training_environment.training_code_sha256 is missing")
    normalized_code = {}
    for name, digest in code.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint training code path is invalid")
        normalized_code[name] = checkpoint_sha(digest, f"training_code_sha256.{name}")
    result["training_code_sha256"] = dict(sorted(normalized_code.items()))
    return result


def normalize_training_corpus(checkpoint: dict) -> dict:
    value = checkpoint.get("training_corpus_identity")
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("checkpoint training_corpus_identity must be schema_version 1")
    recordings = value.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("checkpoint training_corpus_identity.recordings must be non-empty")
    normalized: list[dict] = []
    for index, row in enumerate(recordings):
        if not isinstance(row, dict):
            raise ValueError(f"training corpus recording[{index}] must be an object")
        item = {
            "recording": str(row.get("recording", "")),
            "manifest": str(row.get("manifest", "")),
            "path": str(row.get("path", "")),
            "file_sha256": checkpoint_sha(row.get("file_sha256"), f"training corpus[{index}].file_sha256"),
            "pcm_sha256": checkpoint_sha(row.get("pcm_sha256"), f"training corpus[{index}].pcm_sha256"),
            "frames": int(row.get("frames", -1)),
            "duration_s": float(row.get("duration_s", -1.0)),
        }
        if not item["recording"] or not item["manifest"] or not item["path"] or item["frames"] <= 0 or not math.isfinite(item["duration_s"]) or item["duration_s"] <= 0.0:
            raise ValueError(f"training corpus recording[{index}] contains invalid identity")
        for field in IDENTITY_FIELDS:
            if field in row:
                field_value = row[field]
                if not isinstance(field_value, str) or not field_value:
                    raise ValueError(f"training corpus recording[{index}].{field} is invalid")
                item[field] = field_value
        normalized.append(item)
    digest = corpus_digest(normalized)
    if value.get("corpus_sha256") != digest:
        raise ValueError("checkpoint training corpus canonical digest is inconsistent")
    return {"schema_version": 1, "corpus_sha256": digest, "recordings": normalized}


def training_metadata(checkpoint: dict) -> dict:
    manifests = checkpoint.get("training_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("checkpoint training_manifests must be a non-empty list")
    normalized_manifests: list[dict] = []
    for index, item in enumerate(manifests):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"checkpoint training_manifests[{index}] is invalid")
        normalized_manifests.append(
            {
                "name": item["name"],
                "sha256": checkpoint_sha(
                    item.get("sha256"), f"training_manifests[{index}].sha256"
                ),
            }
        )
    result = {
        "manifests": normalized_manifests,
        "corpus_identity": normalize_training_corpus(checkpoint),
        "examples": int(checkpoint["training_examples"]),
        "seed": int(checkpoint["seed"]),
        "epochs": int(checkpoint["epochs"]),
        "batch_size": int(checkpoint["batch_size"]),
        "learning_rate": float(checkpoint["learning_rate"]),
        "optimizer": str(checkpoint["optimizer"]),
        "weight_decay": float(checkpoint["weight_decay"]),
        "grad_clip_norm": float(checkpoint["grad_clip_norm"]),
        "environment": training_environment(checkpoint),
    }
    if result["examples"] <= 0 or result["epochs"] <= 0 or result["batch_size"] <= 0:
        raise ValueError("checkpoint training counts must be positive")
    for key in ("learning_rate", "weight_decay", "grad_clip_norm"):
        if not math.isfinite(result[key]) or result[key] < 0.0:
            raise ValueError(f"checkpoint {key} must be finite and non-negative")
    if not result["optimizer"]:
        raise ValueError("checkpoint optimizer must be non-empty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path, help="default: OUTPUT.provenance.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    checkpoint_vocab_size = int(checkpoint["vocab_size"])
    checkpoint_fingerprint = int(checkpoint.get("vocab_fingerprint", -1))
    checkpoint_tokens_sha256 = checkpoint_sha(checkpoint.get("tokens_sha256"), "tokens_sha256")
    frontend_spec_version = int(checkpoint.get("frontend_spec_version", -1))
    frontend_name = str(checkpoint.get("frontend_name", ""))
    frontend_kind = int(checkpoint.get("frontend_kind", -1))
    frame_length = int(checkpoint["frame_length_samples"])
    frame_hop = int(checkpoint["frame_hop_samples"])
    training = training_metadata(checkpoint)

    if not 1 <= feature_dim <= MAX_FEATURE_DIM:
        raise ValueError(f"feature_dim must be 1..{MAX_FEATURE_DIM}")
    if not 1 <= hidden_dim <= MAX_HIDDEN_DIM:
        raise ValueError(f"hidden_dim must be 1..{MAX_HIDDEN_DIM}")
    if not 2 <= checkpoint_vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be 2..{MAX_VOCAB_SIZE}")
    if frontend_spec_version != FRONTEND_SPEC_VERSION:
        raise ValueError(f"checkpoint frontend_spec_version={frontend_spec_version} does not match required {FRONTEND_SPEC_VERSION}")
    if frontend_name not in FRONTEND_IDS or FRONTEND_IDS[frontend_name] != frontend_kind:
        raise ValueError("checkpoint frontend name/kind identity is invalid")
    if frame_length != FRAME_LENGTH_SAMPLES or frame_hop != FRAME_HOP_SAMPLES:
        raise ValueError(f"ABI v2 requires frame_length/frame_hop {FRAME_LENGTH_SAMPLES}/{FRAME_HOP_SAMPLES}")

    require_shape(state_dict, "in_proj.weight", (hidden_dim, feature_dim))
    require_shape(state_dict, "in_proj.bias", (hidden_dim,))
    require_shape(state_dict, "rec_proj.weight", (hidden_dim, hidden_dim))
    require_shape(state_dict, "out_proj.weight", (checkpoint_vocab_size, hidden_dim))
    require_shape(state_dict, "out_proj.bias", (checkpoint_vocab_size,))

    token_map = load_tokens(args.tokens)
    token_vocab_size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    tokens_sha256 = sha256_file(args.tokens)
    if token_vocab_size != checkpoint_vocab_size:
        raise ValueError(f"token vocabulary size {token_vocab_size} does not match checkpoint vocab_size {checkpoint_vocab_size}")
    if checkpoint_fingerprint != fingerprint:
        raise ValueError("checkpoint vocabulary fingerprint does not match --tokens")

    wx, sx, wx_stats = q8(state_dict["in_proj.weight"], "in_proj.weight")
    wh, sh, wh_stats = q8(state_dict["rec_proj.weight"], "rec_proj.weight")
    wo, so, wo_stats = q8(state_dict["out_proj.weight"], "out_proj.weight")
    bh = float_bytes(state_dict["in_proj.bias"], "in_proj.bias")
    bo = float_bytes(state_dict["out_proj.bias"], "out_proj.bias")

    buffer = bytearray(b"\x00" * MODEL_HEADER_BYTES)
    offsets: list[int] = []
    for block in (wx, wh, bh, wo, bo):
        align4(buffer)
        offsets.append(len(buffer))
        buffer += block
    total = len(buffer)
    header = struct.pack(
        "<4sHHHHHHIIIfffQIIIIII",
        b"KWSP",
        MODEL_VERSION,
        MODEL_HEADER_BYTES,
        feature_dim,
        hidden_dim,
        checkpoint_vocab_size,
        frontend_kind,
        SAMPLE_RATE_HZ,
        frame_length,
        frame_hop,
        sx,
        sh,
        so,
        fingerprint,
        *offsets,
        total,
    )
    assert len(header) == MODEL_HEADER_BYTES
    buffer[:MODEL_HEADER_BYTES] = header
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(buffer)

    provenance_path = args.provenance or pathlib.Path(str(args.output) + ".provenance.json")
    provenance = {
        "schema_version": 3,
        "model": {
            "name": args.output.name,
            "sha256": sha256_file(args.output),
            "bytes": total,
            "abi": MODEL_VERSION,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": checkpoint_vocab_size,
            "vocab_fingerprint": f"0x{fingerprint:016x}",
            "frontend_spec_version": frontend_spec_version,
            "frontend_name": frontend_name,
            "frontend_kind": frontend_kind,
        },
        "checkpoint": {"name": args.checkpoint.name, "sha256": sha256_file(args.checkpoint)},
        "tokens": {
            "name": args.tokens.name,
            "sha256": tokens_sha256,
            "checkpoint_sha256": checkpoint_tokens_sha256,
            "byte_identical_to_training": tokens_sha256 == checkpoint_tokens_sha256,
        },
        "training": training,
        "quantization": {
            "scheme": "symmetric-int8-per-matrix",
            "in_proj": wx_stats,
            "rec_proj": wh_stats,
            "out_proj": wo_stats,
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {total} bytes, frontend={frontend_name}, "
        f"vocab_fingerprint=0x{fingerprint:016x}; corpus={training['corpus_identity']['corpus_sha256']}; provenance={provenance_path}"
    )


if __name__ == "__main__":
    main()
