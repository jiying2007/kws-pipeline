#!/usr/bin/env python3
from __future__ import annotations

import pathlib

MAX_VOCAB_SIZE = 512
FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211


def load_tokens(path: pathlib.Path) -> dict[str, int]:
    out: dict[str, int] = {}
    used_ids: set[int] = set()
    next_id = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            token, token_id = parts[0], next_id
        elif len(parts) == 2:
            token, token_id = parts[0], int(parts[1])
        else:
            raise ValueError(f"{path}:{line_no}: invalid token line")
        if token in out:
            raise ValueError(f"{path}:{line_no}: duplicate token {token}")
        if token_id < 0 or token_id >= MAX_VOCAB_SIZE:
            raise ValueError(
                f"{path}:{line_no}: token id must be 0..{MAX_VOCAB_SIZE - 1}"
            )
        if token_id in used_ids:
            raise ValueError(f"{path}:{line_no}: duplicate token id {token_id}")
        out[token] = token_id
        used_ids.add(token_id)
        next_id = max(next_id, token_id + 1)
    if out.get("<blk>") != 0:
        raise ValueError("tokens file must map <blk> to id 0")
    if not out:
        raise ValueError("tokens file is empty")
    return out


def vocab_size(token_map: dict[str, int]) -> int:
    return max(token_map.values()) + 1


def vocab_fingerprint(token_map: dict[str, int]) -> int:
    """Stable FNV-1a/64 over canonical ID<TAB>token lines sorted by ID."""
    by_id = sorted((token_id, token) for token, token_id in token_map.items())
    value = FNV1A64_OFFSET
    for token_id, token in by_id:
        encoded = f"{token_id}\t{token}\n".encode("utf-8")
        for byte in encoded:
            value ^= byte
            value = (value * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value
