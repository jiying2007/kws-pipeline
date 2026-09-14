#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import struct

import torch

from export_model import (
    FRONTEND_IDS,
    FRAME_HOP_SAMPLES,
    FRAME_LENGTH_SAMPLES,
    FRONTEND_SPEC_VERSION,
    MAX_FEATURE_DIM,
    MAX_HIDDEN_DIM,
    MAX_VOCAB_SIZE,
    align4,
    checkpoint_sha,
    float_bytes,
    q8,
    require_shape,
    sha256_file,
    training_metadata,
)
from gru_model import ARCHITECTURE
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size

MAGIC = b"KWG1"
FORMAT_VERSION = 1
HEADER_BYTES = 80
SAMPLE_RATE_HZ = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description="Export development-only Tiny-GRU weights.")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if str(checkpoint.get("architecture", "")) != ARCHITECTURE:
        raise ValueError("checkpoint is not a Tiny-GRU v1 experiment")
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
        raise ValueError("frontend spec version mismatch")
    if frontend_name not in FRONTEND_IDS or FRONTEND_IDS[frontend_name] != frontend_kind:
        raise ValueError("frontend identity mismatch")
    if frame_length != FRAME_LENGTH_SAMPLES or frame_hop != FRAME_HOP_SAMPLES:
        raise ValueError("experimental GRU requires the shipping frame contract")

    require_shape(state_dict, "gru.weight_ih", (3 * hidden_dim, feature_dim))
    require_shape(state_dict, "gru.weight_hh", (3 * hidden_dim, hidden_dim))
    require_shape(state_dict, "gru.bias_ih", (3 * hidden_dim,))
    require_shape(state_dict, "gru.bias_hh", (3 * hidden_dim,))
    require_shape(state_dict, "out_proj.weight", (checkpoint_vocab_size, hidden_dim))
    require_shape(state_dict, "out_proj.bias", (checkpoint_vocab_size,))

    token_map = load_tokens(args.tokens)
    if vocab_size(token_map) != checkpoint_vocab_size:
        raise ValueError("token vocabulary size differs from checkpoint")
    fingerprint = vocab_fingerprint(token_map)
    if fingerprint != checkpoint_fingerprint:
        raise ValueError("token vocabulary fingerprint differs from checkpoint")
    if sha256_file(args.tokens) != checkpoint_tokens_sha256:
        raise ValueError("token vocabulary bytes differ from training")

    wih, sx, wih_stats = q8(state_dict["gru.weight_ih"], "gru.weight_ih")
    whh, sh, whh_stats = q8(state_dict["gru.weight_hh"], "gru.weight_hh")
    wo, so, wo_stats = q8(state_dict["out_proj.weight"], "out_proj.weight")
    bih = float_bytes(state_dict["gru.bias_ih"], "gru.bias_ih")
    bhh = float_bytes(state_dict["gru.bias_hh"], "gru.bias_hh")
    bo = float_bytes(state_dict["out_proj.bias"], "out_proj.bias")

    buffer = bytearray(b"\x00" * HEADER_BYTES)
    offsets: list[int] = []
    for block in (wih, whh, bih, bhh, wo, bo):
        align4(buffer)
        offsets.append(len(buffer))
        buffer += block
    total = len(buffer)
    flags = 0
    header = struct.pack(
        "<4sHHHHHHIIIfffQI7I",
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
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
        flags,
        *offsets,
        total,
    )
    if len(header) != HEADER_BYTES:
        raise AssertionError(f"experimental GRU header size drifted: {len(header)}")
    buffer[:HEADER_BYTES] = header
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(buffer)

    provenance_path = args.provenance or pathlib.Path(str(args.output) + ".provenance.json")
    provenance = {
        "schema_version": 1,
        "evidence_class": "development-only-model-v2-gru",
        "experimental": True,
        "architecture": ARCHITECTURE,
        "model": {
            "name": args.output.name,
            "sha256": sha256_file(args.output),
            "bytes": total,
            "format": FORMAT_VERSION,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": checkpoint_vocab_size,
            "vocab_fingerprint": f"0x{fingerprint:016x}",
            "frontend_name": frontend_name,
            "frontend_kind": frontend_kind,
        },
        "checkpoint": {"name": args.checkpoint.name, "sha256": sha256_file(args.checkpoint)},
        "tokens": {
            "name": args.tokens.name,
            "sha256": sha256_file(args.tokens),
            "byte_identical_to_training": True,
        },
        "training": training,
        "quantization": {
            "scheme": "symmetric-int8-per-matrix",
            "gru_input": wih_stats,
            "gru_recurrent": whh_stats,
            "out_proj": wo_stats,
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"exported experimental {ARCHITECTURE} model {args.output}: bytes={total}")


if __name__ == "__main__":
    main()
