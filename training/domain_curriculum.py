#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

DIMENSION_PREFIXES = {
    "distance": "distance:",
    "azimuth": "azimuth:",
    "rt60": "rt60:",
    "noise": "noise:",
    "playback": "playback:",
    "composite": "composite:",
}


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def metric_hardness(item: dict, label: str) -> float:
    frr = finite(item.get("frr", 0.0), f"{label}.frr")
    far = finite(item.get("far_per_hour", 0.0), f"{label}.far")
    latency = finite(item.get("p95_post_end_latency_ms", 0.0), f"{label}.latency")
    # FAR remains visible for negative-only domains while FRR dominates positive
    # wake-domain adaptation. Latency is deliberately a small tie-breaker.
    return max(
        0.0,
        frr * 20.0 + min(far, 20.0) * 0.05 + min(latency, 1500.0) / 3000.0,
    )


def previous_dimension(previous: dict | None, name: str) -> dict[str, float]:
    if not isinstance(previous, dict):
        return {}
    dimensions = previous.get("dimension_weights", previous)
    if not isinstance(dimensions, dict):
        return {}
    value = dimensions.get(name, {})
    if not isinstance(value, dict):
        return {}
    return {str(key): finite(weight, f"previous.{name}.{key}") for key, weight in value.items()}


def update_curriculum(
    domain_metrics: dict,
    *,
    previous: dict | None = None,
    strength: float = 2.0,
    max_weight: float = 6.0,
) -> dict:
    if strength < 0.0 or max_weight < 1.0:
        raise ValueError("curriculum strength/max_weight is invalid")
    domains = domain_metrics.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domain metrics is missing domains")

    dimension_hardness: dict[str, dict[str, float]] = {}
    dimension_weights: dict[str, dict[str, float]] = {}
    ranked_domains: list[tuple[float, str]] = []

    for dimension, prefix in DIMENSION_PREFIXES.items():
        hardness: dict[str, float] = {}
        for key, item in domains.items():
            if not key.startswith(prefix) or not isinstance(item, dict):
                continue
            value_name = key[len(prefix) :]
            score = metric_hardness(item, key)
            hardness[value_name] = score
            ranked_domains.append((score, key))
        if not hardness:
            continue

        maximum = max(hardness.values())
        base = previous_dimension(previous, dimension)
        weights: dict[str, float] = {}
        for value_name, score in hardness.items():
            relative = score / maximum if maximum > 1.0e-12 else 0.0
            target = 1.0 + strength * relative
            old = base.get(value_name, 1.0)
            # EMA prevents a single weak round from completely reshaping the next
            # render distribution while still accumulating persistent hard domains.
            weights[value_name] = min(
                max_weight,
                max(1.0, 0.55 * old + 0.45 * target),
            )
        dimension_hardness[dimension] = hardness
        dimension_weights[dimension] = weights

    ranked_domains.sort(key=lambda item: (-item[0], item[1]))
    return {
        "schema_version": 2,
        "dimension_weights": dimension_weights,
        "dimension_hardness": dimension_hardness,
        "worst_domains": [
            {"domain": key, "hardness": score}
            for score, key in ranked_domains[:12]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--previous", type=pathlib.Path)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=6.0)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    previous = None
    if args.previous:
        value = json.loads(args.previous.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("previous curriculum must be a JSON object")
        previous = value
    result = update_curriculum(
        metrics,
        previous=previous,
        strength=finite(args.strength, "strength"),
        max_weight=finite(args.max_weight, "max_weight"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
