#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import wave

SAMPLE_RATE_HZ = 16000
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pcm16_wav(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        frames = reader.getnframes()
        pcm = reader.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError(f"{path}: truncated PCM payload")
    return {
        "file_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "frames": frames,
        "duration_s": frames / float(SAMPLE_RATE_HZ),
    }


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact(path: pathlib.Path) -> dict:
    resolved = path.resolve(strict=True)
    stats = inspect_pcm16_wav(resolved)
    return {"path": str(resolved), **stats}


def corpus_digest(rows: list[dict]) -> str:
    normalized = []
    for row in rows:
        item = {
            "recording": row.get("recording"),
            "path": row.get("path"),
            "file_sha256": row["file_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "frames": row["frames"],
        }
        for field in IDENTITY_FIELDS:
            if field in row:
                item[field] = row[field]
        normalized.append(item)
    return canonical_hash(normalized)
