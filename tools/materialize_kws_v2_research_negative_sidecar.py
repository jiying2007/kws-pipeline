#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import wave

SPLITS = ("train", "calibration", "test")
MANIFEST_CLASS = "speech-like-synthetic-recording-v1"
INDEX_CLASS = "kws-v2-research-negative-request-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def inspect_wav(path: pathlib.Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16000
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"research negative must be mono 16-kHz PCM16: {path}")
        frames = int(reader.getnframes())
    if frames <= 0:
        raise ValueError(f"research negative WAV is empty: {path}")
    return frames, frames / 16000.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize generated ordinary-speech negatives into empty-target research sidecars."
    )
    parser.add_argument("--generated-root", required=True, type=pathlib.Path)
    parser.add_argument("--request-index", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    generated = args.generated_root.resolve()
    request_index = args.request_index.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("research negative sidecar output-dir must be empty")
    output.mkdir(parents=True, exist_ok=True)

    index_rows = load_jsonl(request_index)
    requested: dict[str, dict] = {}
    for row in index_rows:
        if int(row.get("schema_version", 0)) != 1 or row.get("evidence_class") != INDEX_CLASS:
            raise ValueError("research negative request index identity mismatch")
        split = str(row.get("split", ""))
        source_id = str(row.get("source_id", ""))
        if split not in SPLITS or not source_id or source_id in requested:
            raise ValueError(f"invalid research negative request row: {row}")
        if row.get("target_policy") != "empty-target-nonwake":
            raise ValueError("research negative sidecar requires empty-target policy")
        if row.get("protected_evidence_used") is not False:
            raise ValueError("research negative request unexpectedly used protected evidence")
        requested[source_id] = row

    generated_rows: dict[str, tuple[dict, pathlib.Path]] = {}
    for group in ("train", "generalization-search"):
        manifest = generated / group / "manifest.jsonl"
        if not manifest.is_file():
            raise ValueError(f"generated research negative manifest missing: {manifest}")
        for row in load_jsonl(manifest):
            if (
                int(row.get("schema_version", 0)) != 1
                or row.get("evidence_class") != MANIFEST_CLASS
            ):
                raise ValueError(f"generated manifest identity mismatch: {manifest}")
            source_id = str(row.get("source_id", ""))
            if not source_id or source_id in generated_rows:
                raise ValueError(f"duplicate/empty generated source_id: {source_id}")
            generated_rows[source_id] = (row, manifest.parent)

    if set(requested) != set(generated_rows):
        missing = sorted(set(requested) - set(generated_rows))
        extra = sorted(set(generated_rows) - set(requested))
        raise ValueError(
            f"generated research negatives do not match request index; missing={len(missing)} extra={len(extra)}"
        )

    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    voice_owners: dict[str, str] = {}
    source_hashes: set[str] = set()
    for source_id in sorted(requested):
        request = requested[source_id]
        generated_row, manifest_root = generated_rows[source_id]
        if str(generated_row.get("voice_id", "")) != str(request["voice_id"]):
            raise ValueError(f"voice identity drifted for {source_id}")
        if str(generated_row.get("text", "")) != str(request["text"]):
            raise ValueError(f"text drifted for {source_id}")
        audio_raw = generated_row.get("audio")
        if not isinstance(audio_raw, str) or not audio_raw:
            raise ValueError(f"generated audio path missing for {source_id}")
        source_audio = (manifest_root / audio_raw).resolve()
        if not source_audio.is_file():
            raise ValueError(f"generated audio file missing for {source_id}: {source_audio}")
        expected_sha = str(generated_row.get("file_sha256", ""))
        if sha256_file(source_audio) != expected_sha:
            raise ValueError(f"generated WAV hash mismatch for {source_id}")
        if expected_sha in source_hashes:
            raise ValueError("research negative sidecar contains duplicate WAV bytes")
        source_hashes.add(expected_sha)

        split = str(request["split"])
        voice_id = str(request["voice_id"])
        previous = voice_owners.get(voice_id)
        if previous is not None and previous != split:
            raise ValueError(f"voice_id crossed research splits: {voice_id} {previous}/{split}")
        voice_owners[voice_id] = split

        split_dir = output / split
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        target = audio_dir / f"{len(split_rows[split]):06d}-{expected_sha[:12]}.wav"
        shutil.copyfile(source_audio, target)
        if sha256_file(target) != expected_sha:
            raise ValueError("research negative WAV changed while copying")
        frames, duration_s = inspect_wav(target)
        split_rows[split].append(
            {
                "source_id": source_id,
                "voice_id": voice_id,
                "category": str(request["category"]),
                "utterance_id": str(request["utterance_id"]),
                "text": str(request["text"]),
                "path": str(target.resolve()),
                "file_sha256": expected_sha,
                "frames": frames,
                "duration_s": duration_s,
                "provider_name": str(generated_row.get("provider_name", "")),
                "provider_version": str(generated_row.get("provider_version", "")),
                "license_id": str(generated_row.get("license_id", "")),
                "generation_config_sha256": str(generated_row.get("generation_config_sha256", "")),
            }
        )

    receipts: dict[str, dict] = {}
    for split in SPLITS:
        rows = split_rows[split]
        if not rows:
            raise ValueError(f"research negative split is empty: {split}")
        split_dir = output / split
        tsv = split_dir / "negative-empty-target.tsv"
        tsv.write_text(
            "".join(f"{row['path']}\t\n" for row in rows),
            encoding="utf-8",
        )
        refs = split_dir / "negative.references.jsonl"
        refs.write_text(
            "".join(
                json.dumps(
                    {
                        "recording": f"research-negative-{split}-{index:06d}",
                        "path": row["path"],
                        "duration_s": row["duration_s"],
                        "expected": [],
                        "source_id": row["source_id"],
                        "session_id": f"tts-{row['voice_id']}",
                        "domain": {
                            "kind": "research-ordinary-speech-negative",
                            "category": row["category"],
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for index, row in enumerate(rows)
            ),
            encoding="utf-8",
        )
        index_path = split_dir / "index.jsonl"
        index_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        receipts[split] = {
            "recordings": len(rows),
            "audio_hours": sum(float(row["duration_s"]) for row in rows) / 3600.0,
            "tsv": str(tsv),
            "tsv_sha256": sha256_file(tsv),
            "references": str(refs),
            "references_sha256": sha256_file(refs),
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "voice_ids": sorted({str(row["voice_id"]) for row in rows}),
            "categories": sorted({str(row["category"]) for row in rows}),
        }

    receipt = {
        "schema_version": 1,
        "evidence_class": "kws-v2-research-negative-sidecar-v1",
        "evidence_scope": "research-only",
        "target_policy": "empty-target-nonwake",
        "canonical_kws_vocabulary_required": False,
        "qualification_split_consumed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "request_index_sha256": sha256_file(request_index),
        "splits": receipts,
        "total_recordings": sum(item["recordings"] for item in receipts.values()),
        "unique_wav_sha256": len(source_hashes),
    }
    target = output / "sidecar-receipt.json"
    target.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"research-negative-sidecar: recordings={receipt['total_recordings']} "
        f"wav={receipt['unique_wav_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
