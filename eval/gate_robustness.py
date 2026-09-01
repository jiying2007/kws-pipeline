#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def required_keys(config: dict) -> list[tuple[str, str]]:
    gates = config.get("robustness_gates")
    if not isinstance(gates, dict):
        raise ValueError("robustness_gates config is required")
    result: list[tuple[str, str]] = []
    for value in gates.get("required_distance_bins", []):
        result.append(("distance_bin", str(value)))
    for value in gates.get("required_azimuth_deg", []):
        result.append(("azimuth_deg", str(int(value))))
    for value in gates.get("required_snr_bands", []):
        result.append(("snr", str(value)))
    if not result:
        raise ValueError("robustness_gates must require at least one slice")
    return result


def evaluate(summary: dict, config: dict) -> dict:
    gates = config.get("robustness_gates")
    if not isinstance(gates, dict):
        raise ValueError("robustness_gates config is required")
    domain_report = summary.get("qualification_domains")
    if not isinstance(domain_report, dict):
        raise ValueError("training summary is missing qualification_domains")
    domains = domain_report.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("qualification_domains is missing domains")

    max_frr = finite(gates.get("max_frr", 0.0), "robustness_gates.max_frr")
    max_far = finite(
        gates.get("max_far_per_hour", 0.0),
        "robustness_gates.max_far_per_hour",
    )
    min_expected = int(gates.get("min_expected_wakes", 1))
    min_negative_hours = finite(
        gates.get("min_negative_audio_hours", 0.0),
        "robustness_gates.min_negative_audio_hours",
    )
    if not 0.0 <= max_frr <= 1.0 or max_far < 0.0:
        raise ValueError("robustness FRR/FAR gates are invalid")
    if min_expected <= 0 or min_negative_hours < 0.0:
        raise ValueError("robustness support gates are invalid")

    failures: list[dict] = []
    slices: dict[str, dict] = {}
    for prefix, value in required_keys(config):
        key = f"{prefix}:{value}"
        metrics = domains.get(key)
        if not isinstance(metrics, dict):
            failures.append({"slice": key, "reason": "missing-slice"})
            continue
        expected = int(metrics.get("expected", 0))
        negative_hours = finite(
            metrics.get("negative_audio_hours", 0.0), f"{key}.negative_audio_hours"
        )
        frr = finite(metrics.get("frr", 1.0), f"{key}.frr")
        far = finite(metrics.get("far_per_hour", math.inf), f"{key}.far_per_hour")
        wake_rate = finite(metrics.get("wake_rate", 1.0 - frr), f"{key}.wake_rate")
        item = {
            "expected": expected,
            "negative_audio_hours": negative_hours,
            "wake_rate": wake_rate,
            "frr": frr,
            "far_per_hour": far,
        }
        slices[key] = item
        if expected < min_expected:
            failures.append(
                {
                    "slice": key,
                    "reason": "insufficient-positive-support",
                    "actual": expected,
                    "required": min_expected,
                }
            )
        if negative_hours + 1.0e-12 < min_negative_hours:
            failures.append(
                {
                    "slice": key,
                    "reason": "insufficient-negative-support",
                    "actual": negative_hours,
                    "required": min_negative_hours,
                }
            )
        if frr > max_frr + 1.0e-12:
            failures.append(
                {
                    "slice": key,
                    "reason": "frr",
                    "actual": frr,
                    "limit": max_frr,
                }
            )
        if far > max_far + 1.0e-12:
            failures.append(
                {
                    "slice": key,
                    "reason": "far-per-hour",
                    "actual": far,
                    "limit": max_far,
                }
            )

    return {
        "schema_version": 1,
        "evidence_class": "synthetic-domain-robustness-matrix",
        "qualified": not failures,
        "gates": {
            "max_frr": max_frr,
            "max_far_per_hour": max_far,
            "min_expected_wakes": min_expected,
            "min_negative_audio_hours": min_negative_hours,
        },
        "slices": slices,
        "failures": failures,
        "limitations": [
            "Slice FAR is a synthetic coverage gate, not a production statistical FAR claim.",
            "Physical rooms, real Mandarin speakers, shipping microphones and shipping AFE remain external qualification gates.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise ValueError("summary/config must be JSON objects")
    result = evaluate(summary, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
