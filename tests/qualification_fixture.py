from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import training_corpus_identity  # noqa: E402


def fnv1a64_token_fingerprint(tokens: list[str]) -> int:
    value = 14695981039346656037
    prime = 1099511628211
    for token_id, token in enumerate(tokens):
        for byte in f"{token_id}\t{token}\n".encode("utf-8"):
            value ^= byte
            value = (value * prime) & 0xFFFFFFFFFFFFFFFF
    return value


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tokens(path: pathlib.Path) -> tuple[list[str], int]:
    tokens = ["<blk>", "ni3", "hao3", "xiao3", "wo1"]
    path.write_text(
        "\n".join(f"{token} {index}" for index, token in enumerate(tokens)) + "\n",
        encoding="utf-8",
    )
    return tokens, fnv1a64_token_fingerprint(tokens)


def write_model(path: pathlib.Path, fingerprint: int, frontend_kind: int = 0) -> int:
    feature_dim = 32
    hidden_dim = 4
    vocab_size = 5
    wx_off = 72
    wh_off = wx_off + feature_dim * hidden_dim
    bh_off = wh_off + hidden_dim * hidden_dim
    wo_off = bh_off + hidden_dim * 4
    bo_off = wo_off + vocab_size * hidden_dim
    total = bo_off + vocab_size * 4
    blob = bytearray(total)
    header = struct.pack(
        "<4sHHHHHHIIIfffQIIIIII",
        b"KWSP",
        2,
        72,
        feature_dim,
        hidden_dim,
        vocab_size,
        frontend_kind,
        16000,
        400,
        320,
        0.01,
        0.01,
        0.01,
        fingerprint,
        wx_off,
        wh_off,
        bh_off,
        wo_off,
        bo_off,
        total,
    )
    blob[:72] = header
    path.write_bytes(blob)
    return total


def write_model_provenance(
    path: pathlib.Path,
    model: pathlib.Path,
    export_tokens: pathlib.Path,
    training_tokens: pathlib.Path,
    checkpoint: pathlib.Path,
    training_manifests: list[pathlib.Path],
    fingerprint: int,
    frontend_kind: int = 0,
) -> str:
    checkpoint_hash = sha256_file(checkpoint)
    export_token_hash = sha256_file(export_tokens)
    training_token_hash = sha256_file(training_tokens)
    quant = {
        "scale": 0.01,
        "max_abs_error": 0.004,
        "rmse": 0.001,
        "signal_rms": 0.25,
        "snr_db": 47.0,
    }
    frontend_name = "logmel" if frontend_kind == 0 else "pcen-lite"
    corpus = training_corpus_identity(training_manifests)
    write_json(
        path,
        {
            "schema_version": 3,
            "model": {
                "name": model.name,
                "sha256": sha256_file(model),
                "bytes": model.stat().st_size,
                "abi": 2,
                "feature_dim": 32,
                "hidden_dim": 4,
                "vocab_size": 5,
                "vocab_fingerprint": f"0x{fingerprint:016x}",
                "frontend_spec_version": 2,
                "frontend_name": frontend_name,
                "frontend_kind": frontend_kind,
            },
            "checkpoint": {"name": checkpoint.name, "sha256": checkpoint_hash},
            "tokens": {
                "name": export_tokens.name,
                "sha256": export_token_hash,
                "checkpoint_sha256": training_token_hash,
                "byte_identical_to_training": export_token_hash == training_token_hash,
            },
            "training": {
                "manifests": [
                    {"name": manifest.name, "sha256": sha256_file(manifest)}
                    for manifest in training_manifests
                ],
                "corpus_identity": corpus,
                "examples": len(corpus["recordings"]),
                "seed": 1337,
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 0.001,
                "optimizer": "AdamW",
                "weight_decay": 0.0001,
                "grad_clip_norm": 5.0,
            },
            "quantization": {
                "scheme": "symmetric-int8-per-matrix",
                "in_proj": dict(quant),
                "rec_proj": dict(quant),
                "out_proj": dict(quant),
            },
        },
    )
    return checkpoint_hash


def write_pack(path: pathlib.Path, fingerprint: int) -> int:
    token_ids = [1, 2] + [0] * 14
    record = struct.pack("<IfHBBBBH16H", 1, 0.99, 2, 0, 0, 0, 0, 0, *token_ids)
    total = 24 + len(record)
    header = struct.pack("<4sHHHHIQ", b"KWKP", 3, 24, 1, 5, total, fingerprint)
    path.write_bytes(header + record)
    return total


def write_wav(path: pathlib.Path, seconds: int = 1) -> None:
    samples = 16000 * seconds
    frames = bytearray()
    for index in range(samples):
        sample = 2000 if ((index // 16) & 1) else -2000
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(frames)


def write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
