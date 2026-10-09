#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from statistical_bounds import poisson_rate_upper  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def integer(value, label: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def sha256_identity(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_shard(row: dict, path: pathlib.Path) -> dict:
    # Validate each shard before summing: cancellation and int() coercion can
    # otherwise hide invalid counts/exposure behind a plausible aggregate.
    version = integer(row.get("schema_version"), f"{path}: schema_version")
    if version not in (1, 2):
        raise ValueError(f"{path}: unsupported FAR summary schema")
    if version >= 2 and row.get("full_negative_manifest_coverage") is not True:
        raise ValueError(f"{path}: FAR shard did not cover its full hard-negative manifest")
    validated = {
        key: sha256_identity(row.get(key), f"{path}: {key}")
        for key in ("runner_sha256", "model_sha256", "keyword_pack_sha256")
    }
    negative_manifest = row.get("negative_manifest_sha256")
    if negative_manifest is not None:
        negative_manifest = sha256_identity(
            negative_manifest, f"{path}: negative_manifest_sha256"
        )
    validated["negative_manifest_sha256"] = negative_manifest
    validated["seed"] = integer(row.get("seed"), f"{path}: seed", nonnegative=False)
    validated["false_accepts"] = integer(row.get("false_accepts"), f"{path}: false_accepts")
    audio_hours = finite(row.get("audio_hours"), f"{path}: audio_hours")
    if audio_hours <= 0.0:
        raise ValueError(f"{path}: audio_hours must be positive")
    validated["audio_hours"] = audio_hours
    # Preserve version 1's optional hard-negative field defaults. Version 2
    # producers always emit these fields, even when exposure is zero.
    default = 0 if version == 1 else None
    validated["hard_negative_injections"] = integer(
        row.get("hard_negative_injections", default), f"{path}: hard_negative_injections"
    )
    for key in ("hard_negative_rate_per_minute", "hard_negative_audio_seconds"):
        value = finite(row.get(key, default), f"{path}: {key}")
        if value < 0.0:
            raise ValueError(f"{path}: {key} must be non-negative")
        validated[key] = value
    if validated["hard_negative_rate_per_minute"] > 60.0:
        raise ValueError(f"{path}: hard_negative_rate_per_minute must be <= 60")
    return validated


def write_result(path: pathlib.Path, text: str) -> None:
    # Replace only after serialization and writing succeed. Invalid invocations
    # preserve existing files; their nonzero exit and stderr identify stale output.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        try:
            stream.write(text + "\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-far-per-hour", required=True, type=float)
    parser.add_argument("--max-upper-bound-per-hour", required=True, type=float)
    parser.add_argument("--min-hard-negative-injections", type=int, default=0)
    parser.add_argument("--min-hard-negative-audio-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if len(args.summary) < 2:
        raise ValueError("FAR aggregation requires at least two independent shards")
    args.max_far_per_hour = finite(args.max_far_per_hour, "maximum FAR/hour")
    args.max_upper_bound_per_hour = finite(
        args.max_upper_bound_per_hour, "maximum FAR upper bound/hour"
    )
    args.confidence = finite(args.confidence, "confidence")
    if not 0.5 < args.confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    if args.max_far_per_hour < 0.0 or args.max_upper_bound_per_hour < 0.0:
        raise ValueError("FAR limits must be non-negative")
    min_hn_audio = finite(
        args.min_hard_negative_audio_seconds,
        "min hard-negative audio seconds",
    )
    if args.min_hard_negative_injections < 0 or min_hn_audio < 0.0:
        raise ValueError("hard-negative exposure limits must be non-negative")

    rows = []
    identity = None
    seeds: set[int] = set()
    for path in args.summary:
        row = validate_shard(load(path), path)
        current = (
            row["runner_sha256"],
            row["model_sha256"],
            row["keyword_pack_sha256"],
            row["negative_manifest_sha256"],
            row["hard_negative_rate_per_minute"],
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(
                "cannot aggregate FAR exposure across different runner/model/pack/hard-negative inputs"
            )
        seed = row["seed"]
        if seed in seeds:
            raise ValueError("FAR shard seeds must be unique")
        seeds.add(seed)
        rows.append((path, row))

    audio_hours = finite(sum(row["audio_hours"] for _, row in rows), "aggregate audio_hours")
    false_accepts = sum(row["false_accepts"] for _, row in rows)
    hard_negative_injections = sum(row["hard_negative_injections"] for _, row in rows)
    hard_negative_audio_seconds = finite(
        sum(row["hard_negative_audio_seconds"] for _, row in rows),
        "aggregate hard_negative_audio_seconds",
    )
    far = finite(false_accepts / audio_hours, "aggregate FAR/hour")
    upper = finite(
        poisson_rate_upper(false_accepts, audio_hours, args.confidence),
        "aggregate FAR upper bound/hour",
    )
    assert identity is not None
    violations = []
    if far > args.max_far_per_hour:
        violations.append("aggregate FAR/hour above maximum")
    if upper > args.max_upper_bound_per_hour:
        violations.append("aggregate FAR statistical upper bound above maximum")
    if hard_negative_injections < args.min_hard_negative_injections:
        violations.append("aggregate hard-negative injection count below minimum")
    if hard_negative_audio_seconds + 1.0e-12 < min_hn_audio:
        violations.append("aggregate hard-negative audio exposure below minimum")
    result = {
        "schema_version": 1,
        "evidence_class": "synthetic-streaming-far-aggregate",
        "qualified": not violations,
        "shards": len(rows),
        "seeds": sorted(seeds),
        "audio_hours": audio_hours,
        "false_accepts": false_accepts,
        "far_per_hour": far,
        "confidence_level": args.confidence,
        "far_upper_bound_per_hour": upper,
        "max_far_per_hour": args.max_far_per_hour,
        "max_upper_bound_per_hour": args.max_upper_bound_per_hour,
        "runner_sha256": identity[0],
        "model_sha256": identity[1],
        "keyword_pack_sha256": identity[2],
        "negative_manifest_sha256": identity[3],
        "hard_negative_rate_per_minute": identity[4],
        "hard_negative_injections": hard_negative_injections,
        "hard_negative_audio_seconds": hard_negative_audio_seconds,
        "min_hard_negative_injections": args.min_hard_negative_injections,
        "min_hard_negative_audio_seconds": min_hn_audio,
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path, _ in rows
        ],
        "violations": violations,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    write_result(args.output, serialized)
    print(serialized)
    return 0 if not violations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, OverflowError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "no aggregate written; any existing output is stale for this invocation",
            file=sys.stderr,
        )
        raise SystemExit(2)
