#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
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


def identity_bundle(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("corpus identity requires at least one recording")
    return {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(rows),
        "recordings": rows,
    }


def _metadata(row: dict, label: str) -> dict:
    result = {}
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}: {field} must be non-empty text")
            result[field] = value.strip()
    return result


def training_manifest_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            audio = value.get("audio", value.get("path"))
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError(f"{path}:{line_no}: audio/path must be non-empty")
            rows.append({"audio": audio.strip(), "metadata": _metadata(value, f"{path}:{line_no}")})
    else:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
            audio = raw.split("\t", 1)[0].strip()
            if not audio:
                raise ValueError(f"{path}:{line_no}: empty WAV path")
            rows.append({"audio": audio, "metadata": {}})
    return rows


def training_corpus_identity(manifests: list[pathlib.Path]) -> dict:
    identities: list[dict] = []
    for manifest_index, manifest in enumerate(manifests):
        root = manifest.parent
        for row_index, row in enumerate(training_manifest_rows(manifest), 1):
            raw_path = row["audio"]
            path = pathlib.Path(raw_path)
            resolved = (path if path.is_absolute() else root / path).resolve(strict=True)
            identities.append(
                {
                    "recording": f"manifest-{manifest_index}:{row_index}",
                    "manifest": manifest.name,
                    "path": raw_path,
                    **inspect_pcm16_wav(resolved),
                    **row["metadata"],
                }
            )
    return identity_bundle(identities)


def evaluation_corpus_identity(references: pathlib.Path, audio_root: pathlib.Path) -> dict:
    identities: list[dict] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(references.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{references}:{line_no}: expected JSON object")
        recording = row.get("recording")
        raw_path = row.get("audio_path", row.get("path"))
        if not isinstance(recording, str) or not recording or recording in seen:
            raise ValueError(f"{references}:{line_no}: recording must be unique non-empty text")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{references}:{line_no}: path/audio_path must be non-empty")
        path = pathlib.Path(raw_path.strip())
        resolved = (path if path.is_absolute() else audio_root / path).resolve(strict=True)
        measured = inspect_pcm16_wav(resolved)
        declared_duration = row.get("duration_s")
        if isinstance(declared_duration, bool) or not isinstance(declared_duration, (int, float)):
            raise ValueError(f"{references}:{line_no}: duration_s must be numeric")
        if not math.isfinite(float(declared_duration)) or not math.isclose(
            float(declared_duration), measured["duration_s"], rel_tol=0.0, abs_tol=1.0 / SAMPLE_RATE_HZ
        ):
            raise ValueError(
                f"{references}:{line_no}: duration_s={declared_duration} does not match WAV duration {measured['duration_s']}"
            )
        identities.append(
            {
                "recording": recording,
                "path": raw_path.strip(),
                **measured,
                **_metadata(row, f"{references}:{line_no}"),
            }
        )
        seen.add(recording)
    return identity_bundle(identities)
