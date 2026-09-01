#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from domain_curriculum import update_curriculum  # noqa: E402


def metric(frr: float, far: float = 0.0, latency: float = 100.0) -> dict:
    return {
        "frr": frr,
        "far_per_hour": far,
        "p95_post_end_latency_ms": latency,
    }


def main() -> int:
    metrics = {
        "domains": {
            "distance:near": metric(0.01),
            "distance:mid": metric(0.04, latency=120.0),
            "distance:far": metric(0.20, far=0.5, latency=400.0),
            "azimuth:front": metric(0.01),
            "azimuth:side": metric(0.05),
            "azimuth:rear": metric(0.18),
            "snr:critical": metric(0.22, far=0.2, latency=420.0),
            "snr:low": metric(0.10),
            "snr:mid": metric(0.04),
            "snr:high": metric(0.01),
            "rt60:dry": metric(0.01),
            "rt60:medium": metric(0.04),
            "rt60:reverb": metric(0.16),
            "noise:fan": metric(0.02),
            "noise:motor": metric(0.12),
            "playback:no-playback": metric(0.02),
            "playback:playback": metric(0.15),
            "composite:distance=far|az=rear|rt60=reverb|noise=motor|playback": metric(
                0.25, far=0.8, latency=550.0
            ),
        }
    }
    result = update_curriculum(metrics, strength=3.0, max_weight=6.0)
    assert result["schema_version"] == 2
    weights = result["dimension_weights"]
    assert weights["distance"]["far"] > weights["distance"]["mid"] > weights["distance"]["near"] >= 1.0
    assert weights["azimuth"]["rear"] > weights["azimuth"]["front"]
    assert weights["snr"]["critical"] > weights["snr"]["low"] > weights["snr"]["high"]
    assert weights["rt60"]["reverb"] > weights["rt60"]["dry"]
    assert weights["noise"]["motor"] > weights["noise"]["fan"]
    assert weights["playback"]["playback"] > weights["playback"]["no-playback"]
    assert result["worst_domains"][0]["domain"].startswith("composite:")

    next_result = update_curriculum(
        metrics, previous=result, strength=3.0, max_weight=6.0
    )
    assert (
        next_result["dimension_weights"]["distance"]["far"]
        >= weights["distance"]["far"]
    )
    assert (
        next_result["dimension_weights"]["snr"]["critical"]
        >= weights["snr"]["critical"]
    )
    print("test_domain_curriculum: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
