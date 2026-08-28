#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


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
        audio_path = row.get("path")
        if not recording or recording in seen:
            raise ValueError(f"{path}:{line_no}: recording must be non-empty and unique")
        if not audio_path:
            raise ValueError(f"{path}:{line_no}: path is required for corpus execution")
        seen.add(recording)
        rows.append(row)
    if not rows:
        raise ValueError("reference corpus is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path)
    args = parser.parse_args()

    rows = load_references(args.references)
    output_lines: list[str] = []
    for row in rows:
        recording = str(row["recording"])
        audio = pathlib.Path(str(row["path"]))
        if not audio.is_absolute():
            audio = args.audio_root / audio
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
                    f"runner recording mismatch: expected {recording}, "
                    f"got {detection.get('recording')}"
                )
            output_lines.append(
                json.dumps(detection, ensure_ascii=False, allow_nan=False)
            )

    args.detections.parent.mkdir(parents=True, exist_ok=True)
    args.detections.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )

    if args.provenance:
        provenance = {
            "schema_version": 1,
            "runner_sha256": sha256_file(args.runner),
            "model_sha256": sha256_file(args.model),
            "keyword_pack_sha256": sha256_file(args.keywords),
            "references_sha256": sha256_file(args.references),
            "detections_sha256": sha256_file(args.detections),
            "recordings": len(rows),
            "detections": len(output_lines),
        }
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    print(
        f"processed {len(rows)} recording(s), emitted {len(output_lines)} detection(s)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
