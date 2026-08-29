#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from domain_curriculum import update_curriculum  # noqa: E402


def main() -> int:
    metrics = {
        "domains": {
            "distance:near": {"frr": 0.01, "far_per_hour": 0.0, "p95_post_end_latency_ms": 50.0},
            "distance:mid": {"frr": 0.04, "far_per_hour": 0.0, "p95_post_end_latency_ms": 120.0},
            "distance:far": {"frr": 0.20, "far_per_hour": 0.5, "p95_post_end_latency_ms": 400.0},
        }
    }
    result = update_curriculum(metrics, strength=3.0, max_weight=6.0)
    weights = result["distance_band_weights"]
    assert result["worst_distance_band"] == "far"
    assert weights["far"] > weights["mid"] > weights["near"] >= 1.0
    next_result = update_curriculum(metrics, previous=weights, strength=3.0, max_weight=6.0)
    assert next_result["distance_band_weights"]["far"] >= weights["far"]
    print("test_domain_curriculum: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
