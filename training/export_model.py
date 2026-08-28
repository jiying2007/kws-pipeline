#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

MODEL_VERSION = 2
MODEL_HEADER_BYTES = 72
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FRONTEND_SPEC_VERSION = 1
MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512


def finite_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def q8(tensor: torch.Tensor, name: str) -> tuple[bytes, float]:
    tensor = finite_tensor(tensor, name)
    maximum = max(float(tensor.abs().max()), 1.0e-8)
    scale = maximum / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized.numpy().tobytes(), scale


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    checkpoint_vocab_size = int(checkpoint["vocab_size"])
    checkpoint_fingerprint = int(checkpoint.get("vocab_fingerprint", -1))
    frontend_spec_version = int(checkpoint.get("frontend_spec_version", -1))
    frame_length = int(checkpoint["frame_length_samples"])
    frame_hop = int(checkpoint["frame_hop_samples"])

    if not 1 <= feature_dim <= MAX_FEATURE_DIM:
        raise ValueError(f"feature_dim must be 1..{MAX_FEATURE_DIM}")
    if not 1 <= hidden_dim <= MAX_HIDDEN_DIM:
        raise ValueError(f"hidden_dim must be 1..{MAX_HIDDEN_DIM}")
    if not 2 <= checkpoint_vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be 2..{MAX_VOCAB_SIZE}")
    if frontend_spec_version != FRONTEND_SPEC_VERSION:
        raise ValueError(
            f"checkpoint frontend_spec_version={frontend_spec_version} does not match "
            f"required {FRONTEND_SPEC_VERSION}"
        )
    if frame_length != FRAME_LENGTH_SAMPLES or frame_hop != FRAME_HOP_SAMPLES:
        raise ValueError(
            f"ABI v2 requires frame_length/frame_hop "
            f"{FRAME_LENGTH_SAMPLES}/{FRAME_HOP_SAMPLES}"
        )

    require_shape(state_dict, "in_proj.weight", (hidden_dim, feature_dim))
    require_shape(state_dict, "in_proj.bias", (hidden_dim,))
    require_shape(state_dict, "rec_proj.weight", (hidden_dim, hidden_dim))
    require_shape(state_dict, "out_proj.weight", (checkpoint_vocab_size, hidden_dim))
    require_shape(state_dict, "out_proj.bias", (checkpoint_vocab_size,))

    token_map = load_tokens(args.tokens)
    token_vocab_size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    if token_vocab_size != checkpoint_vocab_size:
        raise ValueError(
            f"token vocabulary size {token_vocab_size} does not match "
            f"checkpoint vocab_size {checkpoint_vocab_size}"
        )
    if checkpoint_fingerprint != fingerprint:
        raise ValueError(
            "checkpoint vocabulary fingerprint does not match --tokens; "
            "refusing to bind weights to a different token-to-ID mapping"
        )

    wx, sx = q8(state_dict["in_proj.weight"], "in_proj.weight")
    wh, sh = q8(state_dict["rec_proj.weight"], "rec_proj.weight")
    wo, so = q8(state_dict["out_proj.weight"], "out_proj.weight")
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
        0,
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
    print(
        f"wrote {args.output}: {total} bytes, "
        f"vocab_fingerprint=0x{fingerprint:016x}"
    )


if __name__ == "__main__":
    main()
