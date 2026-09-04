#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


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
        row = load(path)
        schema_version = int(row.get("schema_version", 0))
        if schema_version not in (1, 2):
            raise ValueError(f"{path}: unsupported FAR summary schema")
        if schema_version >= 2 and not bool(row.get("full_negative_manifest_coverage", False)):
            raise ValueError(f"{path}: FAR shard did not cover its full hard-negative manifest")
        negative_manifest = row.get("negative_manifest_sha256")
        if negative_manifest is not None and not isinstance(negative_manifest, str):
            raise ValueError(f"{path}: negative_manifest_sha256 must be a string or null")
        hard_negative_rate = finite(
            row.get("hard_negative_rate_per_minute", 0.0),
            "hard_negative_rate_per_minute",
        )
        if hard_negative_rate < 0.0:
            raise ValueError(f"{path}: hard-negative injection rate must be non-negative")
        current = (
            str(row.get("runner_sha256")),
            str(row.get("model_sha256")),
            str(row.get("keyword_pack_sha256")),
            negative_manifest or "",
            hard_negative_rate,
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(
                "cannot aggregate FAR exposure across different runner/model/pack/hard-negative inputs"
            )
        seed = int(row["seed"])
        if seed in seeds:
            raise ValueError("FAR shard seeds must be unique")
        seeds.add(seed)
        rows.append((path, row))

    audio_hours = sum(finite(row["audio_hours"], "audio_hours") for _, row in rows)
    false_accepts = sum(int(row["false_accepts"]) for _, row in rows)
    hard_negative_injections = sum(int(row.get("hard_negative_injections", 0)) for _, row in rows)
    hard_negative_audio_seconds = sum(
        finite(row.get("hard_negative_audio_seconds", 0.0), "hard_negative_audio_seconds")
        for _, row in rows
    )
    if audio_hours <= 0.0 or false_accepts < 0:
        raise ValueError("invalid aggregate FAR count/exposure")
    if hard_negative_injections < 0 or hard_negative_audio_seconds < 0.0:
        raise ValueError("invalid aggregate hard-negative exposure")
    far = false_accepts / audio_hours
    upper = poisson_rate_upper(false_accepts, audio_hours, args.confidence)
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
        "negative_manifest_sha256": identity[3] or None,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
