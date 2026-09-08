#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import wave
from collections import Counter, defaultdict

FORBIDDEN_FIELDS = {"name", "full_name", "email", "phone", "address", "id_number"}
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_number(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def inspect_wav(path: pathlib.Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
            raise ValueError(f"{path}: capture must be uncompressed PCM16 WAV")
        frames = reader.getnframes()
        rate = reader.getframerate()
        channels = reader.getnchannels()
    if frames <= 0 or rate <= 0 or channels <= 0:
        raise ValueError(f"{path}: invalid WAV geometry")
    return {
        "sample_rate_hz": rate,
        "channels": channels,
        "sample_format": "pcm_s16le",
        "duration_s": frames / float(rate),
    }


def in_slice(row: dict, rule: dict) -> bool:
    if "tag" in rule:
        return str(rule["tag"]) in set(row["tags"])
    field = str(rule["field"])
    value = require_number(row[field], field)
    if "min" in rule and value < float(rule["min"]):
        return False
    if "min_exclusive" in rule and value <= float(rule["min_exclusive"]):
        return False
    if "max" in rule and value > float(rule["max"]):
        return False
    if "max_exclusive" in rule and value >= float(rule["max_exclusive"]):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--public-summary", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or policy.get("schema_version") != 1:
        raise ValueError("manifest/policy schema_version must be 1")
    if manifest.get("corpus_role") != "fresh-held-out-qualification":
        raise ValueError("corpus_role must be fresh-held-out-qualification")
    if manifest.get("deployment_tag") != policy.get("deployment_tag"):
        raise ValueError("manifest deployment_tag differs from qualification policy")
    qualification_id = str(manifest.get("qualification_id", "")).strip()
    if len(qualification_id) < 8:
        raise ValueError("qualification_id must be stable and at least 8 characters")

    forbidden = set(policy.get("identity_policy", {}).get("forbidden_fields", [])) | FORBIDDEN_FIELDS
    recordings = manifest.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("recordings must be a non-empty list")

    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    positive_speakers: set[str] = set()
    speaker_sessions: dict[str, set[str]] = defaultdict(set)
    expected_by_keyword: Counter[int] = Counter()
    negative_hours = 0.0
    negative_tag_hours: defaultdict[str, float] = defaultdict(float)
    normalized: list[dict] = []

    for index, row in enumerate(recordings):
        if not isinstance(row, dict):
            raise ValueError(f"recordings[{index}] must be an object")
        bad = forbidden.intersection(row)
        if bad:
            raise ValueError(f"recordings[{index}] contains forbidden PII field(s): {sorted(bad)}")
        for field in IDENTITY_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"recordings[{index}].{field} must be non-empty pseudonymous text")
        if row.get("consent_scope") != "product-kws-qualification":
            raise ValueError(f"recordings[{index}] lacks qualification consent scope")
        if row.get("retention_class") != "restricted-raw-audio":
            raise ValueError(f"recordings[{index}] retention_class must keep raw audio restricted")

        recording = str(row.get("recording", "")).strip()
        if not recording or recording in seen_names:
            raise ValueError("recording IDs must be non-empty and unique")
        seen_names.add(recording)
        rel = pathlib.Path(str(row.get("input_path", "")))
        if rel.is_absolute():
            audio = rel
        else:
            audio = args.audio_root / rel
        audio = audio.resolve(strict=True)
        actual_sha = sha256_file(audio)
        actual_bytes = audio.stat().st_size
        if actual_sha != str(row.get("input_sha256", "")):
            raise ValueError(f"{recording}: input SHA256 mismatch")
        if actual_bytes != int(row.get("input_bytes", 0)):
            raise ValueError(f"{recording}: input byte count mismatch")
        if actual_sha in seen_hashes:
            raise ValueError(f"{recording}: duplicate raw audio content hash")
        seen_hashes.add(actual_sha)

        wav = inspect_wav(audio)
        capture = row.get("capture")
        if not isinstance(capture, dict):
            raise ValueError(f"{recording}: capture metadata is required")
        for field in ("sample_rate_hz", "channels", "sample_format"):
            if capture.get(field) != wav[field]:
                raise ValueError(f"{recording}: declared capture {field} differs from WAV")
        declared_duration = require_number(row.get("duration_s"), f"{recording}.duration_s")
        if abs(declared_duration - wav["duration_s"]) > max(1e-6, 1.0 / wav["sample_rate_hz"]):
            raise ValueError(f"{recording}: duration differs from WAV")

        distance = require_number(row.get("distance_m"), f"{recording}.distance_m")
        azimuth = require_number(row.get("azimuth_deg"), f"{recording}.azimuth_deg")
        if distance < 0.0 or distance > 10.0 or azimuth < 0.0 or azimuth >= 360.0:
            raise ValueError(f"{recording}: distance/azimuth out of range")
        tags = row.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError(f"{recording}: tags must be non-empty strings")
        if len(tags) != len(set(tags)):
            raise ValueError(f"{recording}: duplicate tags")
        is_rear = 135.0 <= azimuth <= 225.0
        if ("rear" in tags) != is_rear:
            raise ValueError(f"{recording}: rear tag must match azimuth 135..225 degrees")

        expected = row.get("expected")
        if not isinstance(expected, list):
            raise ValueError(f"{recording}: expected must be a list")
        for event_index, event in enumerate(expected):
            if not isinstance(event, dict):
                raise ValueError(f"{recording}: expected[{event_index}] must be an object")
            keyword_id = int(event.get("keyword_id", 0))
            if keyword_id not in (1, 2):
                raise ValueError(f"{recording}: only shipping keyword IDs 1/2 are allowed")
            start_s = require_number(event.get("start_s"), "start_s")
            end_s = require_number(event.get("end_s"), "end_s")
            if start_s < 0.0 or end_s < start_s or end_s > declared_duration:
                raise ValueError(f"{recording}: expected event lies outside recording")
            expected_by_keyword[keyword_id] += 1
        if expected:
            speaker = str(row["speaker_id"])
            positive_speakers.add(speaker)
            speaker_sessions[speaker].add(str(row["session_id"]))
        else:
            hours = declared_duration / 3600.0
            negative_hours += hours
            for tag in tags:
                negative_tag_hours[tag] += hours

        identity_row = {key: row[key] for key in sorted(row) if key != "input_path"}
        normalized.append(identity_row)

    minimums = policy["minimums"]
    if len(positive_speakers) < int(minimums["speakers"]):
        raise ValueError("insufficient distinct positive speakers")
    min_sessions = int(minimums["sessions_per_speaker"])
    short = sorted(speaker for speaker in positive_speakers if len(speaker_sessions[speaker]) < min_sessions)
    if short:
        raise ValueError(f"positive speakers missing required session count: {short[:8]}")
    expected_total = sum(expected_by_keyword.values())
    if expected_total < int(minimums["expected_wakes_total"]):
        raise ValueError("insufficient total expected wakes")
    for keyword_id in (1, 2):
        if expected_by_keyword[keyword_id] < int(minimums["expected_wakes_per_keyword"]):
            raise ValueError(f"insufficient expected wakes for keyword {keyword_id}")
    if negative_hours + 1e-12 < float(minimums["negative_audio_hours"]):
        raise ValueError("insufficient negative audio hours")

    slice_counts: dict[str, dict[str, int]] = {}
    for rule in policy.get("critical_positive_slices", []):
        counts = Counter()
        for row in recordings:
            if not row.get("expected") or not in_slice(row, rule):
                continue
            for event in row["expected"]:
                counts[int(event["keyword_id"])] += 1
        minimum = int(rule["min_expected_per_keyword"])
        for keyword_id in (1, 2):
            if counts[keyword_id] < minimum:
                raise ValueError(f"slice {rule['name']} lacks keyword {keyword_id} coverage")
        slice_counts[str(rule["name"])] = {str(k): counts[k] for k in (1, 2)}

    for tag, minimum_hours in policy.get("critical_negative_exposure_hours", {}).items():
        if negative_tag_hours[str(tag)] + 1e-12 < float(minimum_hours):
            raise ValueError(f"negative exposure tag {tag} is below minimum hours")

    identity = {
        "schema_version": 1,
        "qualification_id": qualification_id,
        "deployment_tag": manifest["deployment_tag"],
        "recordings": sorted(normalized, key=lambda item: item["recording"]),
    }
    summary = {
        "schema_version": 1,
        "qualification_id": qualification_id,
        "deployment_tag": manifest["deployment_tag"],
        "corpus_role": manifest["corpus_role"],
        "corpus_sha256": canonical_sha(identity),
        "manifest_sha256": sha256_file(args.manifest),
        "recordings": len(recordings),
        "positive_speakers": len(positive_speakers),
        "expected_wakes": expected_total,
        "expected_by_keyword": {str(k): expected_by_keyword[k] for k in (1, 2)},
        "negative_audio_hours": negative_hours,
        "critical_positive_slice_counts": slice_counts,
        "critical_negative_hours": {key: negative_tag_hours[key] for key in sorted(policy.get("critical_negative_exposure_hours", {}))},
        "raw_audio_public_upload_allowed": False,
        "raw_audio_hashes_verified": True,
        "pii_fields_rejected": True,
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
