from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
        "\n".join(f"{token} {index}" for index, token in enumerate(tokens))
        + "\n",
        encoding="utf-8",
    )
    return tokens, fnv1a64_token_fingerprint(tokens)


def write_model(path: pathlib.Path, fingerprint: int) -> int:
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
        0,
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


def write_pack(path: pathlib.Path, fingerprint: int) -> int:
    token_ids = [1, 2] + [0] * 14
    record = struct.pack("<IfHH16H", 1, 0.99, 2, 0, *token_ids)
    total = 24 + len(record)
    header = struct.pack("<4sHHHHIQ", b"KWKP", 2, 24, 1, 5, total, fingerprint)
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
