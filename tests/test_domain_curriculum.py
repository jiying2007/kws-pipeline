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
            "distance:far": metric(0.05, far=0.1, latency=180.0),
            "azimuth:front": metric(0.01),
            "azimuth:side": metric(0.05),
            "azimuth:rear": metric(0.06),
            "snr:critical": metric(0.07),
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
            "distance_azimuth:distance_bin=5m|azimuth=rear": metric(0.28),
            "distance_snr:distance_bin=5m|snr=critical": metric(0.30),
            "azimuth_snr:azimuth=rear|snr=critical": metric(0.27),
            "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical": metric(
                0.35, far=0.8, latency=550.0
            ),
            "composite:distance=far|az=rear|rt60=reverb|noise=motor|playback": metric(
                0.25, far=0.8, latency=550.0
            ),
        },
        "keyword_domains": {
            "1": {
                "domains": {
                    "distance:far": metric(0.04),
                    "azimuth:rear": metric(0.05),
                    "snr:critical": metric(0.06),
                    "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical": metric(0.32),
                    "distance_azimuth_snr:distance_bin=3m|azimuth=rear|snr=critical": metric(0.08),
                }
            },
            "2": {
                "domains": {
                    "distance:far": metric(0.05),
                    "azimuth:rear": metric(0.06),
                    "snr:critical": metric(0.07),
                    "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical": metric(0.38),
                    "distance_azimuth_snr:distance_bin=5m|azimuth=side|snr=critical": metric(0.10),
                }
            },
        },
    }
    result = update_curriculum(metrics, strength=3.0, max_weight=6.0)
    assert result["schema_version"] == 2
    assert result["hardness_policy"] == "recognition-error-gated-latency-v1"
    assert result["interaction_projection"] == "max-to-distance-azimuth-snr-marginals-v1"
    weights = result["dimension_weights"]
    # The hard 5m/rear/critical triple must project into the marginals consumed
    # by the ordinary train scene sampler, not live only in the report.
    assert weights["distance"]["far"] > weights["distance"]["mid"] > weights["distance"]["near"] >= 1.0
    assert weights["azimuth"]["rear"] > weights["azimuth"]["front"]
    assert weights["snr"]["critical"] > weights["snr"]["high"]
    assert weights["rt60"]["reverb"] > weights["rt60"]["dry"]
    assert weights["noise"]["motor"] > weights["noise"]["fan"]
    assert weights["playback"]["playback"] > weights["playback"]["no-playback"]
    assert (
        weights["distance_azimuth_snr"]["distance_bin=5m|azimuth=rear|snr=critical"]
        > 1.0
    )
    assert result["worst_domains"][0]["domain"].startswith(
        "distance_azimuth_snr:"
    )

    kw = result["keyword_dimension_weights"]
    assert kw["1"]["distance"]["far"] > 1.0
    assert kw["1"]["azimuth"]["rear"] > 1.0
    assert kw["1"]["snr"]["critical"] > 1.0
    assert kw["2"]["distance"]["far"] > 1.0
    assert kw["2"]["azimuth"]["rear"] > kw["2"]["azimuth"]["side"]
    assert kw["2"]["snr"]["critical"] > 1.0
    assert (
        kw["2"]["distance_azimuth_snr"][
            "distance_bin=5m|azimuth=rear|snr=critical"
        ]
        > kw["2"]["distance_azimuth_snr"][
            "distance_bin=5m|azimuth=side|snr=critical"
        ]
    )
    assert result["keyword_worst_domains"]["1"][0]["domain"] == (
        "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical"
    )
    assert result["keyword_worst_domains"]["2"][0]["domain"] == (
        "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical"
    )

    next_result = update_curriculum(
        metrics, previous=result, strength=3.0, max_weight=6.0
    )
    assert (
        next_result["dimension_weights"]["distance"]["far"]
        >= weights["distance"]["far"]
    )
    assert (
        next_result["keyword_dimension_weights"]["2"]["snr"]["critical"]
        >= kw["2"]["snr"]["critical"]
    )

    # Regression for model-training #132: a strict zero-error calibration pass
    # must not turn ordinary 20..700 ms latency variation into full-strength
    # acoustic curriculum feedback. With no FR/FAR evidence, all fresh weights
    # stay at 1.0, zero-hardness interaction views are not advertised as an
    # adaptive focus, and any previous stress weights relax toward baseline.
    zero_error = {
        "domains": {
            "distance:near": metric(0.0, latency=20.0),
            "distance:far": metric(0.0, latency=700.0),
            "azimuth:rear": metric(0.0, latency=650.0),
            "snr:critical": metric(0.0, latency=600.0),
            "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical": metric(
                0.0, latency=700.0
            ),
        },
        "keyword_domains": {
            "1": {
                "domains": {
                    "distance:far": metric(0.0, latency=700.0),
                    "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical": metric(
                        0.0, latency=700.0
                    ),
                }
            }
        },
    }
    stable = update_curriculum(zero_error, strength=3.0, max_weight=6.0)
    assert stable["worst_domains"] == []
    assert stable["keyword_worst_domains"]["1"] == []
    for dimension in stable["dimension_weights"].values():
        assert all(abs(float(value) - 1.0) <= 1.0e-12 for value in dimension.values())
    for dimension in stable["keyword_dimension_weights"]["1"].values():
        assert all(abs(float(value) - 1.0) <= 1.0e-12 for value in dimension.values())

    relaxed = update_curriculum(
        zero_error, previous=result, strength=3.0, max_weight=6.0
    )
    assert 1.0 <= relaxed["dimension_weights"]["distance"]["far"] < weights["distance"]["far"]
    assert (
        1.0
        <= relaxed["keyword_dimension_weights"]["1"]["distance"]["far"]
        < kw["1"]["distance"]["far"]
    )

    no_playback_harder = {
        "domains": {
            "playback:no-playback": metric(0.20, latency=400.0),
            "playback:playback": metric(0.00, latency=20.0),
        }
    }
    guarded = update_curriculum(
        no_playback_harder, strength=3.0, max_weight=6.0
    )["dimension_weights"]["playback"]
    assert guarded["no-playback"] > 1.0
    assert guarded["playback"] == guarded["no-playback"]

    print("test_domain_curriculum: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
