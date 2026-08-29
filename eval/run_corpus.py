#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import corpus_digest, inspect_pcm16_wav  # noqa: E402

IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_references(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        recording = str(row.get("recording", ""))
        audio_path = row.get("audio_path") or row.get("path")
        if not recording or recording in seen:
            raise ValueError(f"{path}:{line_no}: recording must be non-empty and unique")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"{path}:{line_no}: path is required for corpus execution")
        row["_execution_path"] = audio_path.strip()
        seen.add(recording)
        rows.append(row)
    if not rows:
        raise ValueError("reference corpus is empty")
    return rows


def audio_identity(row: dict, audio: pathlib.Path) -> dict:
    measured = inspect_pcm16_wav(audio.resolve(strict=True))
    item = {
        "recording": str(row["recording"]),
        "path": str(row["_execution_path"]),
        **measured,
    }
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{row['recording']}: {field} must be non-empty text")
            item[field] = value.strip()
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path)
    parser.add_argument("--corpus-identity", type=pathlib.Path)
    args = parser.parse_args()

    rows = load_references(args.references)
    output_lines: list[str] = []
    identities: list[dict] = []
    for row in rows:
        recording = str(row["recording"])
        audio = pathlib.Path(str(row["_execution_path"]))
        if not audio.is_absolute():
            audio = args.audio_root / audio
        identities.append(audio_identity(row, audio))
        completed = subprocess.run(
            [
                str(args.runner),
                str(args.model),
                str(args.keywords),
                str(audio),
                recording,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                detection = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"runner emitted invalid JSON for {recording}:{line_no}: {exc}"
                ) from exc
            if not isinstance(detection, dict):
                raise ValueError(
                    f"runner emitted non-object JSON for {recording}:{line_no}"
                )
            if detection.get("recording") != recording:
                raise ValueError(
                    f"runner recording mismatch: expected {recording}, got {detection.get('recording')}"
                )
            output_lines.append(json.dumps(detection, ensure_ascii=False, allow_nan=False))

    corpus_identity = {
        "schema_version": 1,
        "corpus_sha256": corpus_digest(identities),
        "recordings": identities,
    }
    args.detections.parent.mkdir(parents=True, exist_ok=True)
    args.detections.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
    )

    if args.corpus_identity:
        args.corpus_identity.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_identity.write_text(
            json.dumps(corpus_identity, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if args.provenance:
        provenance = {
            "schema_version": 2,
            "runner_sha256": sha256_file(args.runner),
            "model_sha256": sha256_file(args.model),
            "keyword_pack_sha256": sha256_file(args.keywords),
            "references_sha256": sha256_file(args.references),
            "detections_sha256": sha256_file(args.detections),
            "audio_corpus_sha256": corpus_identity["corpus_sha256"],
            "audio_files": identities,
            "recordings": len(rows),
            "detections": len(output_lines),
        }
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        f"processed {len(rows)} recording(s), emitted {len(output_lines)} detection(s), "
        f"corpus={corpus_identity['corpus_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
