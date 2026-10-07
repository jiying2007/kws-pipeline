#!/usr/bin/env python3
"""Read-only structural validation of restricted development audio (stdlib only).

Design source: feat/restricted-development-dataset-iteration-v1 at
064493c5bd71c4de40aa9d6569254724ee1ada46. This standalone hardening does not
import, relax, or invoke the qualification validator, sealer, AFE, or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import stat
import struct
import sys

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_AUDIO_BYTES = 512 * 1024 * 1024
IDENTITY_FIELDS = ("speaker_id", "session_id", "source_id", "room_id", "device_id")
TAGS = {"quiet", "household", "speech", "music", "traffic", "noise", "near", "far", "rear"}
ROW_KEYS = {
    "recording", "input_path", "input_sha256", "input_bytes", "duration_s",
    *IDENTITY_FIELDS, "distance_m", "azimuth_deg", "tags", "expected", "capture",
    "consent_scope", "retention_class",
}


class Invalid(ValueError):
    """Contains only fixed labels and array positions, never input values."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def closed_object(value: object, required: set[str], label: str,
                  optional: set[str] | None = None) -> dict:
    require(type(value) is dict, f"{label}: expected object")
    require(required <= value.keys() <= required | (optional or set()),
            f"{label}: missing or unknown fields")
    return value


def integer(value: object, low: int, high: int, label: str) -> int:
    require(type(value) is int and low <= value <= high,
            f"{label}: expected bounded integer")
    return value


def number(value: object, low: float, high: float, label: str) -> float:
    require(type(value) in (int, float), f"{label}: expected finite number")
    # Compare bounds before converting large JSON integers to float.
    require(low <= value <= high, f"{label}: number outside allowed range")
    result = float(value)
    require(math.isfinite(result), f"{label}: expected finite number")
    return result


def identifier(value: object, label: str, minimum: int = 1) -> str:
    require(type(value) is str and minimum <= len(value) <= 64
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is not None,
            f"{label}: expected bounded opaque identifier")
    return value


def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "manifest: duplicate JSON object key")
        result[key] = value
    return result


def finite_float(raw: str) -> float:
    value = float(raw)
    require(math.isfinite(value), "manifest: nonfinite JSON number")
    return value


def reject_constant(_raw: str) -> None:
    raise Invalid("manifest: nonfinite JSON number")


def absolute_path(path: pathlib.Path) -> pathlib.Path:
    require(".." not in path.parts, "filesystem: parent traversal is forbidden")
    return pathlib.Path(os.path.abspath(path))


def open_directory(path: pathlib.Path) -> int:
    """Pin every directory component without following any symlinks (POSIX)."""
    require(hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd,
            "filesystem: POSIX no-follow directory access is required")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_file(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open a regular, single-link file relative to the pinned audio root."""
    fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        result = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=fd)
    finally:
        os.close(fd)
    try:
        info = os.fstat(result)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                "filesystem: input must be a regular, single-link file")
        return result
    except BaseException:
        os.close(result)
        raise


def snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink)


def read_manifest(path: pathlib.Path) -> dict:
    parent_fd = open_directory(path.parent)
    try:
        fd = open_file(parent_fd, (path.name,))
    finally:
        os.close(parent_fd)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        require(0 < before.st_size <= MAX_MANIFEST_BYTES,
                "manifest: size outside allowed range")
        raw = source.read(MAX_MANIFEST_BYTES + 1)
        require(len(raw) == before.st_size and
                snapshot(before) == snapshot(os.fstat(source.fileno())),
                "manifest: changed during validation")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_float=finite_float, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise Invalid("manifest: invalid UTF-8 JSON") from exc


def audio_parts(value: object, label: str) -> tuple[str, ...]:
    require(type(value) is str and 1 <= len(value) <= 512,
            f"{label}: expected bounded relative WAV path")
    parts = value.split("/")
    require(1 <= len(parts) <= 8 and value.endswith(".wav") and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", part) is not None
        for part in parts), f"{label}: expected canonical root-relative WAV path")
    return tuple(parts)


def inspect_audio(root_fd: int, parts: tuple[str, ...], row: dict, label: str) -> float:
    """Verify exact bytes, every PCM payload byte, and canonical WAV geometry."""
    with os.fdopen(open_file(root_fd, parts), "rb") as source:
        before = os.fstat(source.fileno())
        require(before.st_size == row["input_bytes"], f"{label}: byte count mismatch")
        header = source.read(44)
        require(len(header) == 44, f"{label}: truncated WAV header")
        (riff, riff_bytes, wave, fmt, fmt_bytes, encoding, channels, rate,
         byte_rate, alignment, bits, data, data_bytes) = struct.unpack(
             "<4sI4s4sIHHIIHH4sI", header)
        require((riff, wave, fmt, fmt_bytes, encoding, bits, data) ==
                (b"RIFF", b"WAVE", b"fmt ", 16, 1, 16, b"data"),
                f"{label}: expected canonical PCM16 WAV without extra chunks")
        require(8000 <= rate <= 192000 and 1 <= channels <= 16
                and alignment == channels * 2 and byte_rate == rate * alignment,
                f"{label}: invalid WAV capture geometry")
        require(data_bytes > 0 and data_bytes % alignment == 0
                and riff_bytes + 8 == before.st_size == 44 + data_bytes,
                f"{label}: invalid WAV payload size")
        digest = hashlib.sha256(header)
        remaining = data_bytes
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            require(bool(chunk), f"{label}: truncated WAV payload")
            digest.update(chunk)
            remaining -= len(chunk)
        require(not source.read(1), f"{label}: trailing WAV bytes")
        require(snapshot(before) == snapshot(os.fstat(source.fileno())),
                f"{label}: audio changed during validation")
        require(digest.hexdigest() == row["input_sha256"], f"{label}: SHA256 mismatch")
    capture = row["capture"]
    require(capture["sample_rate_hz"] == rate and capture["channels"] == channels,
            f"{label}: declared capture differs from WAV")
    duration = data_bytes / alignment / rate
    require(math.isclose(row["duration_s"], duration, rel_tol=0.0, abs_tol=1e-9),
            f"{label}: declared duration differs from WAV")
    return duration


def validate(manifest: object, root_fd: int) -> dict:
    manifest = closed_object(manifest, {"schema_version", "dataset_id", "corpus_role",
                                        "recordings"}, "manifest")
    integer(manifest["schema_version"], 1, 1, "schema_version")
    identifier(manifest["dataset_id"], "dataset_id", minimum=8)
    require(manifest["corpus_role"] == "development-feedback", "corpus_role: development only")
    rows = manifest["recordings"]
    require(type(rows) is list and 2 <= len(rows) <= 10000,
            "recordings: expected 2..10000 entries")
    recordings, hashes, paths, speakers = set(), set(), set(), set()
    counts = {1: 0, 2: 0}
    negative_seconds = 0.0
    for index, item in enumerate(rows):
        label = f"recordings[{index}]"
        row = closed_object(item, ROW_KEYS, label, {"snr_db"})
        recording = identifier(row["recording"], f"{label}.recording")
        require(recording not in recordings, f"{label}: duplicate recording ID")
        recordings.add(recording)
        for field in IDENTITY_FIELDS:
            identifier(row[field], f"{label}.{field}")
        require(row["consent_scope"] == "product-kws-development",
                f"{label}: development consent declaration required")
        require(row["retention_class"] == "restricted-raw-audio",
                f"{label}: restricted retention declaration required")
        digest = row["input_sha256"]
        require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"{label}: invalid SHA256 syntax")
        require(digest not in hashes, f"{label}: duplicate audio content hash")
        hashes.add(digest)
        integer(row["input_bytes"], 46, MAX_AUDIO_BYTES, f"{label}.input_bytes")
        duration = number(row["duration_s"], 0.0, 3600.0, f"{label}.duration_s")
        require(duration > 0, f"{label}: duration must be positive")
        number(row["distance_m"], 0, 10, f"{label}.distance_m")
        azimuth = number(row["azimuth_deg"], 0, 360, f"{label}.azimuth_deg")
        require(azimuth < 360, f"{label}: azimuth must be below 360")
        if row.get("snr_db") is not None:
            number(row["snr_db"], -100, 100, f"{label}.snr_db")
        tags = row["tags"]
        require(type(tags) is list and len(tags) <= len(TAGS)
                and all(type(tag) is str and tag in TAGS for tag in tags),
                f"{label}: invalid acoustic tags")
        require(len(tags) == len(set(tags)), f"{label}: duplicate acoustic tags")
        require(("rear" in tags) == (135 <= azimuth <= 225), f"{label}: rear tag/azimuth mismatch")
        capture = closed_object(row["capture"], {"sample_rate_hz", "channels", "sample_format"},
                                f"{label}.capture")
        integer(capture["sample_rate_hz"], 8000, 192000, f"{label}.capture.sample_rate_hz")
        integer(capture["channels"], 1, 16, f"{label}.capture.channels")
        require(capture["sample_format"] == "pcm_s16le", f"{label}: capture must be PCM16")
        parts = audio_parts(row["input_path"], f"{label}.input_path")
        require(parts not in paths, f"{label}: duplicate audio path")
        paths.add(parts)
        actual_duration = inspect_audio(root_fd, parts, row, label)
        events = row["expected"]
        require(type(events) is list and len(events) <= 10000,
                f"{label}: expected must be a bounded array")
        previous_end = 0.0
        for event_index, item in enumerate(events):
            event_label = f"{label}.expected[{event_index}]"
            event = closed_object(item, {"keyword_id", "start_s", "end_s"}, event_label)
            keyword = integer(event["keyword_id"], 1, 2, f"{event_label}.keyword_id")
            start = number(event["start_s"], 0, actual_duration, f"{event_label}.start_s")
            end = number(event["end_s"], 0, actual_duration, f"{event_label}.end_s")
            require(previous_end <= start < end, f"{event_label}: events must be ordered and nonoverlapping")
            previous_end = end
            counts[keyword] += 1
        if events:
            speakers.add(row["speaker_id"])
        else:
            negative_seconds += actual_duration
    require(all(counts.values()), "corpus: both keyword IDs 1 and 2 require expected events")
    require(negative_seconds > 0, "corpus: negative recording exposure required")
    # No input paths, IDs, tags, per-file hashes, or free text are copied out.
    return {
        "schema_version": 1,
        "validation_kind": "restricted-development-structure",
        "corpus_role": "development-feedback",
        "recordings": len(rows),
        "positive_speakers": len(speakers),
        "expected_by_keyword": {str(key): value for key, value in counts.items()},
        "negative_audio_hours": negative_seconds / 3600,
        "structural_checks_passed": True,
        "audio_bytes_hashes_and_payload_verified": True,
        "consent_verified": False,
        "labels_verified": False,
        "anonymization_verified": False,
        "training_authority": False,
        "qualification_authority": False,
        "shipping_authority": False,
        "publication_authority": False,
    }


def publish_summary(path: pathlib.Path, payload: bytes) -> None:
    """Publish atomically without replacing any existing path; private permissions."""
    parent_fd = open_directory(path.parent)
    temporary = ".kws-development-summary-" + secrets.token_hex(16)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            # link() fails if destination is any existing file, symlink or directory.
            os.link(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    follow_symlinks=False)
        finally:
            os.unlink(temporary, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "error: invalid arguments (use --help)\n")


def main() -> int:
    parser = PrivateArgumentParser(
        prog="validate_real_human_development_corpus.py",
        description="Validate restricted development structure; grants no use/publication authority.")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path,
                        help="new aggregate JSON file outside audio root; parent must exist")
    args = parser.parse_args()
    try:
        manifest_path = absolute_path(args.manifest)
        audio_root = absolute_path(args.audio_root)
        summary_path = absolute_path(args.summary)
        require(summary_path != manifest_path and audio_root not in summary_path.parents
                and summary_path != audio_root, "summary: must be separate from inputs and audio root")
        manifest = read_manifest(manifest_path)
        root_fd = open_directory(audio_root)
        try:
            summary = validate(manifest, root_fd)
        finally:
            os.close(root_fd)
        payload = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        publish_summary(summary_path, payload)
    except Invalid as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, OverflowError, RecursionError):
        # OS/JSON exception messages may contain restricted paths or values.
        print("error: invalid input or filesystem operation failed", file=sys.stderr)
        return 2
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
