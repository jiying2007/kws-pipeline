#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

DISTANCE_KEYS = ("near", "mid", "far")


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def update_curriculum(
    domain_metrics: dict,
    *,
    previous: dict[str, float] | None = None,
    strength: float = 2.0,
    max_weight: float = 6.0,
) -> dict:
    if strength < 0.0 or max_weight < 1.0:
        raise ValueError("curriculum strength/max_weight is invalid")
    domains = domain_metrics.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("domain metrics is missing domains")
    base = {name: float((previous or {}).get(name, 1.0)) for name in DISTANCE_KEYS}
    hardness: dict[str, float] = {}
    for name in DISTANCE_KEYS:
        item = domains.get(f"distance:{name}")
        if not isinstance(item, dict):
            hardness[name] = 0.0
            continue
        frr = finite(item.get("frr", 0.0), f"distance:{name}.frr")
        far = finite(item.get("far_per_hour", 0.0), f"distance:{name}.far")
        latency = finite(item.get("p95_post_end_latency_ms", 0.0), f"distance:{name}.latency")
        hardness[name] = max(0.0, frr * 20.0 + min(far, 10.0) * 0.05 + min(latency, 1500.0) / 3000.0)
    maximum = max(hardness.values(), default=0.0)
    weights: dict[str, float] = {}
    for name in DISTANCE_KEYS:
        relative = hardness[name] / maximum if maximum > 1.0e-12 else 0.0
        target = 1.0 + strength * relative
        # EMA avoids oscillating between near/far if one round moves the weakest domain.
        weights[name] = min(max_weight, max(1.0, 0.55 * base[name] + 0.45 * target))
    return {
        "schema_version": 1,
        "distance_band_weights": weights,
        "distance_band_hardness": hardness,
        "worst_distance_band": max(DISTANCE_KEYS, key=lambda name: hardness[name]),
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
        previous = value.get("distance_band_weights") if isinstance(value, dict) else None
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
