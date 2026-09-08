#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import wave


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_wav(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
            raise ValueError(f"{path}: raw capture must be uncompressed PCM16 WAV")
        frames = reader.getnframes()
        rate = reader.getframerate()
        channels = reader.getnchannels()
    if frames <= 0 or rate <= 0 or channels <= 0:
        raise ValueError(f"{path}: invalid WAV geometry")
    return {
        "duration_s": frames / float(rate),
        "capture": {
            "sample_rate_hz": rate,
            "channels": channels,
            "sample_format": "pcm_s16le",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    if draft.get("schema_version") != 1:
        raise ValueError("draft schema_version must be 1")
    if draft.get("corpus_role") != "fresh-held-out-qualification":
        raise ValueError("draft corpus_role must be fresh-held-out-qualification")
    recordings = draft.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("draft recordings must be a non-empty list")

    sealed = {key: value for key, value in draft.items() if key != "recordings"}
    sealed_rows = []
    seen_recordings: set[str] = set()
    seen_hashes: set[str] = set()
    for index, source in enumerate(recordings):
        if not isinstance(source, dict):
            raise ValueError(f"recordings[{index}] must be an object")
        row = dict(source)
        recording = str(row.get("recording", "")).strip()
        if not recording or recording in seen_recordings:
            raise ValueError("recording IDs must be non-empty and unique")
        seen_recordings.add(recording)
        rel = pathlib.Path(str(row.get("input_path", "")))
        if not str(rel):
            raise ValueError(f"{recording}: input_path is required")
        audio = rel if rel.is_absolute() else args.audio_root / rel
        audio = audio.resolve(strict=True)
        identity = inspect_wav(audio)
        digest = sha256_file(audio)
        if digest in seen_hashes:
            raise ValueError(f"{recording}: duplicate raw audio content")
        seen_hashes.add(digest)
        row["input_sha256"] = digest
        row["input_bytes"] = audio.stat().st_size
        row["duration_s"] = identity["duration_s"]
        row["capture"] = identity["capture"]
        sealed_rows.append(row)
    sealed["recordings"] = sealed_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"sealed {len(sealed_rows)} recording(s) into {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
