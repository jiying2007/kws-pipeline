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


def q8(tensor: torch.Tensor) -> tuple[bytes, float]:
    tensor = tensor.detach().cpu().float().contiguous()
    maximum = max(float(tensor.abs().max()), 1.0e-8)
    scale = maximum / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized.numpy().tobytes(), scale


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
    token_map = load_tokens(args.tokens)
    token_vocab_size = vocab_size(token_map)
    checkpoint_vocab_size = int(checkpoint["vocab_size"])
    if token_vocab_size != checkpoint_vocab_size:
        raise ValueError(
            f"token vocabulary size {token_vocab_size} does not match "
            f"checkpoint vocab_size {checkpoint_vocab_size}"
        )
    fingerprint = vocab_fingerprint(token_map)

    wx, sx = q8(state_dict["in_proj.weight"])
    wh, sh = q8(state_dict["rec_proj.weight"])
    wo, so = q8(state_dict["out_proj.weight"])
    bh = (
        state_dict["in_proj.bias"]
        .detach()
        .cpu()
        .float()
        .contiguous()
        .numpy()
        .tobytes()
    )
    bo = (
        state_dict["out_proj.bias"]
        .detach()
        .cpu()
        .float()
        .contiguous()
        .numpy()
        .tobytes()
    )

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
        int(checkpoint["feature_dim"]),
        int(checkpoint["hidden_dim"]),
        checkpoint_vocab_size,
        0,
        16000,
        int(checkpoint["frame_length_samples"]),
        int(checkpoint["frame_hop_samples"]),
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
