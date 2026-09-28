#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_real_human_corpus import (  # noqa: E402
    FORBIDDEN_FIELDS,
    IDENTITY_FIELDS,
    canonical_sha,
    inspect_wav,
    require_number,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--public-summary", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("corpus_role") != "development-feedback":
        raise ValueError("corpus_role must be development-feedback")
    dataset_id = str(manifest.get("dataset_id", "")).strip()
    if (
        len(dataset_id) < 8
        or len(dataset_id) > 64
        or not dataset_id[0].isalnum()
        or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in dataset_id)
    ):
        raise ValueError("dataset_id has invalid development identity syntax")

    rows = manifest.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recordings must be a non-empty list")

    seen_recordings: set[str] = set()
    seen_hashes: set[str] = set()
    speakers: set[str] = set()
    expected_by_keyword: Counter[int] = Counter()
    negative_hours = 0.0
    normalized: list[dict] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"recordings[{index}] must be an object")
        bad = FORBIDDEN_FIELDS.intersection(row)
        if bad:
            raise ValueError(
                f"recordings[{index}] contains forbidden PII field(s): {sorted(bad)}"
            )
        for field in IDENTITY_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"recordings[{index}].{field} must be non-empty pseudonymous text"
                )
        if row.get("consent_scope") != "product-kws-development":
            raise ValueError(f"recordings[{index}] lacks development consent scope")
        if row.get("retention_class") != "restricted-raw-audio":
            raise ValueError(
                f"recordings[{index}] retention_class must keep raw audio restricted"
            )

        recording = str(row.get("recording", "")).strip()
        if not recording or recording in seen_recordings:
            raise ValueError("recording IDs must be non-empty and unique")
        seen_recordings.add(recording)

        rel = pathlib.Path(str(row.get("input_path", "")))
        audio = rel if rel.is_absolute() else args.audio_root / rel
        audio = audio.resolve(strict=True)
        digest = sha256_file(audio)
        if digest != str(row.get("input_sha256", "")):
            raise ValueError(f"{recording}: input SHA256 mismatch")
        if audio.stat().st_size != int(row.get("input_bytes", 0)):
            raise ValueError(f"{recording}: input byte count mismatch")
        if digest in seen_hashes:
            raise ValueError(f"{recording}: duplicate raw audio content hash")
        seen_hashes.add(digest)

        wav = inspect_wav(audio)
        capture = row.get("capture")
        if not isinstance(capture, dict):
            raise ValueError(f"{recording}: capture metadata is required")
        for field in ("sample_rate_hz", "channels", "sample_format"):
            if capture.get(field) != wav[field]:
                raise ValueError(f"{recording}: declared capture {field} differs from WAV")
        duration = require_number(row.get("duration_s"), f"{recording}.duration_s")
        if abs(duration - wav["duration_s"]) > max(1e-6, 1.0 / wav["sample_rate_hz"]):
            raise ValueError(f"{recording}: duration differs from WAV")

        distance = require_number(row.get("distance_m"), f"{recording}.distance_m")
        azimuth = require_number(row.get("azimuth_deg"), f"{recording}.azimuth_deg")
        if distance < 0.0 or distance > 10.0 or azimuth < 0.0 or azimuth >= 360.0:
            raise ValueError(f"{recording}: distance/azimuth out of range")
        tags = row.get("tags")
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag for tag in tags)
            or len(tags) != len(set(tags))
        ):
            raise ValueError(f"{recording}: tags must be unique non-empty strings")
        if ("rear" in tags) != (135.0 <= azimuth <= 225.0):
            raise ValueError(
                f"{recording}: rear tag must match azimuth 135..225 degrees"
            )

        expected = row.get("expected")
        if not isinstance(expected, list):
            raise ValueError(f"{recording}: expected must be a list")
        if expected:
            speakers.add(str(row["speaker_id"]))
        else:
            negative_hours += duration / 3600.0
        for event_index, event in enumerate(expected):
            if not isinstance(event, dict):
                raise ValueError(
                    f"{recording}: expected[{event_index}] must be an object"
                )
            keyword_id = int(event.get("keyword_id", 0))
            if keyword_id not in (1, 2):
                raise ValueError(
                    f"{recording}: only shipping keyword IDs 1/2 are allowed"
                )
            start_s = require_number(event.get("start_s"), "start_s")
            end_s = require_number(event.get("end_s"), "end_s")
            if start_s < 0.0 or end_s <= start_s or end_s > duration:
                raise ValueError(
                    f"{recording}: expected event must have positive duration within recording"
                )
            expected_by_keyword[keyword_id] += 1

        normalized.append(
            {key: row[key] for key in sorted(row) if key != "input_path"}
        )

    for keyword_id in (1, 2):
        if expected_by_keyword[keyword_id] <= 0:
            raise ValueError(
                f"development corpus has no expected wake for keyword {keyword_id}"
            )
    if negative_hours <= 0.0:
        raise ValueError("development corpus has no negative exposure")

    identity = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "corpus_role": "development-feedback",
        "recordings": sorted(normalized, key=lambda item: item["recording"]),
    }
    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "corpus_role": "development-feedback",
        "corpus_sha256": canonical_sha(identity),
        "manifest_sha256": sha256_file(args.manifest),
        "recordings": len(rows),
        "positive_speakers": len(speakers),
        "expected_by_keyword": {
            str(keyword_id): expected_by_keyword[keyword_id]
            for keyword_id in (1, 2)
        },
        "negative_audio_hours": negative_hours,
        "feedback_allowed": True,
        "repeatable": True,
        "qualification_authority": False,
        "shipping_authority": False,
        "raw_audio_public_upload_allowed": False,
        "post_afe_audio_public_upload_allowed": False,
        "raw_audio_hashes_verified": True,
        "pii_fields_rejected": True,
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
