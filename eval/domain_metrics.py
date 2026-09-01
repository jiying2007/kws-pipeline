#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

from score_events import (
    load_jsonl,
    match_keyword_events,
    score,
    validate_detections,
    validate_recordings,
)


DISTANCE_BIN_CONTRACT = {
    "0.5m": "distance_m <= 0.75",
    "1m": "0.75 < distance_m <= 1.50",
    "2m": "1.50 < distance_m <= 2.50",
    "3m": "2.50 < distance_m <= 4.00",
    "5m": "distance_m > 4.00",
}
SNR_BAND_CONTRACT = {
    "critical": "snr_db <= 6",
    "low": "6 < snr_db <= 12",
    "mid": "12 < snr_db <= 20",
    "high": "snr_db > 20",
}
PAIRWISE_SLICE_CONTRACT = {
    "distance_azimuth": "distance_bin:<bin> x azimuth:<front|side|rear>",
    "distance_snr": "distance_bin:<bin> x snr:<critical|low|mid|high>",
    "azimuth_snr": "azimuth:<front|side|rear> x snr:<critical|low|mid|high>",
}


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def distance_bin(distance_m: float) -> str:
    if distance_m <= 0.75:
        return "0.5m"
    if distance_m <= 1.50:
        return "1m"
    if distance_m <= 2.50:
        return "2m"
    if distance_m <= 4.00:
        return "3m"
    return "5m"


def azimuth_point(azimuth_deg: float) -> str:
    normalized = ((azimuth_deg + 180.0) % 360.0) - 180.0
    quantized = int(round(normalized / 30.0) * 30)
    quantized = max(-180, min(180, quantized))
    if quantized == -180:
        quantized = 180
    return str(quantized)


def azimuth_band(azimuth_deg: float) -> str:
    if abs(azimuth_deg) <= 30.0:
        return "front"
    if abs(azimuth_deg) <= 90.0:
        return "side"
    return "rear"


def snr_band(snr_db: float) -> str:
    if snr_db <= 6.0:
        return "critical"
    if snr_db <= 12.0:
        return "low"
    if snr_db <= 20.0:
        return "mid"
    return "high"


def domain_keys(row: dict) -> list[str]:
    domain = row.get("domain")
    if not isinstance(domain, dict):
        return ["all"]
    distance = str(domain.get("distance_band", "unknown"))
    distance_m = finite(domain.get("distance_m", 0.0), "domain.distance_m")
    azimuth = finite(domain.get("azimuth_deg", 0.0), "domain.azimuth_deg")
    rt60 = finite(domain.get("rt60_s", 0.0), "domain.rt60_s")
    snr_db = finite(domain.get("snr_db", 0.0), "domain.snr_db")
    noise = str(domain.get("noise_profile", "unknown"))
    playback = "playback" if domain.get("playback_sir_db") is not None else "no-playback"
    distance_value = distance_bin(distance_m)
    az_band = azimuth_band(azimuth)
    snr_value = snr_band(snr_db)
    if rt60 < 0.30:
        rt_band = "dry"
    elif rt60 < 0.55:
        rt_band = "medium"
    else:
        rt_band = "reverb"
    composite = f"distance={distance}|az={az_band}|rt60={rt_band}|noise={noise}|{playback}"
    return [
        "all",
        f"distance:{distance}",
        f"distance_bin:{distance_value}",
        f"azimuth:{az_band}",
        f"azimuth_deg:{azimuth_point(azimuth)}",
        f"snr:{snr_value}",
        f"distance_azimuth:distance_bin={distance_value}|azimuth={az_band}",
        f"distance_snr:distance_bin={distance_value}|snr={snr_value}",
        f"azimuth_snr:azimuth={az_band}|snr={snr_value}",
        f"rt60:{rt_band}",
        f"noise:{noise}",
        f"playback:{playback}",
        f"composite:{composite}",
    ]


def exposure_stats(names: list[str], recordings: dict[str, dict]) -> dict[str, float | int]:
    positive_seconds = 0.0
    negative_seconds = 0.0
    positive_recordings = 0
    negative_recordings = 0
    for name in names:
        recording = recordings[name]
        if recording["expected"]:
            positive_recordings += 1
            positive_seconds += float(recording["duration_s"])
        else:
            negative_recordings += 1
            negative_seconds += float(recording["duration_s"])
    return {
        "positive_recordings": positive_recordings,
        "negative_recordings": negative_recordings,
        "positive_audio_hours": positive_seconds / 3600.0,
        "negative_audio_hours": negative_seconds / 3600.0,
    }


def subset_score(
    names: list[str],
    recordings: dict[str, dict],
    detections: dict[str, list[dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> dict:
    selected = {name: recordings[name] for name in names}
    selected_detections = {
        name: detections.get(name, []) for name in names if name in detections
    }
    summary, _, _ = score(
        selected,
        selected_detections,
        pre_tolerance_s,
        post_tolerance_s,
    )
    frr = float(summary["frr"])
    far_per_hour = float(summary["far_per_hour"])
    return {
        **summary,
        "wake_rate": max(0.0, min(1.0, 1.0 - frr)),
        "false_wake_rate_per_hour": far_per_hour,
        **exposure_stats(names, recordings),
    }


def confusion_matrix(
    recordings: dict[str, dict],
    detections: dict[str, list[dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> dict:
    """Build keyword confusion from one global monotonic assignment per recording."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exact = 0
    wrong = 0
    missing = 0
    for name, recording in recordings.items():
        events = list(enumerate(recording["expected"]))
        dets = list(enumerate(detections.get(name, [])))
        pairs = match_keyword_events(events, dets, pre_tolerance_s, post_tolerance_s)
        matched_events = {event_index for event_index, _ in pairs}
        for event_index, detection_index in pairs:
            event = recording["expected"][event_index]
            detection = detections.get(name, [])[detection_index]
            expected_id = int(event["keyword_id"])
            detected_id = int(detection["keyword_id"])
            matrix[str(expected_id)][str(detected_id)] += 1
            if detected_id == expected_id:
                exact += 1
            else:
                wrong += 1
        for event_index, event in enumerate(recording["expected"]):
            if event_index in matched_events:
                continue
            expected_id = int(event["keyword_id"])
            matrix[str(expected_id)]["<miss>"] += 1
            missing += 1
    return {
        "assignment": "global-monotonic-one-to-one-v1",
        "expected_events": exact + wrong + missing,
        "correct_keyword": exact,
        "wrong_keyword": wrong,
        "missed": missing,
        "matrix": {key: dict(value) for key, value in sorted(matrix.items())},
    }


def domain_is_eligible(
    summary: dict,
    *,
    min_expected: int,
    min_negative_hours: float,
) -> bool:
    positive_supported = int(summary["expected"]) >= min_expected and int(summary["expected"]) > 0
    negative_supported = (
        float(summary["negative_audio_hours"]) >= min_negative_hours
        and int(summary["negative_recordings"]) > 0
    )
    return positive_supported or negative_supported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    parser.add_argument("--min-domain-expected", type=int, default=1)
    parser.add_argument("--min-domain-negative-hours", type=float, default=0.0)
    args = parser.parse_args()
    pre = finite(args.pre_tolerance_ms, "pre tolerance") / 1000.0
    post = finite(args.post_tolerance_ms, "post tolerance") / 1000.0
    min_negative_hours = finite(args.min_domain_negative_hours, "min domain negative hours")
    if pre < 0.0 or post < 0.0 or args.min_domain_expected < 0 or min_negative_hours < 0.0:
        raise ValueError("domain metric tolerances/counts/exposure must be non-negative")

    raw_rows = load_jsonl(args.references)
    recordings = validate_recordings(raw_rows)
    detections = validate_detections(load_jsonl(args.detections), recordings)
    raw_by_name = {str(row["recording"]): row for row in raw_rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for name, row in raw_by_name.items():
        for key in domain_keys(row):
            groups[key].append(name)

    metrics: dict[str, dict] = {}
    eligible: dict[str, dict] = {}
    for key, names in sorted(groups.items()):
        summary = subset_score(names, recordings, detections, pre, post)
        if key == "all" or domain_is_eligible(
            summary,
            min_expected=args.min_domain_expected,
            min_negative_hours=min_negative_hours,
        ):
            metrics[key] = summary
        if key != "all" and domain_is_eligible(
            summary,
            min_expected=args.min_domain_expected,
            min_negative_hours=min_negative_hours,
        ):
            eligible[key] = summary

    worst_key = None
    worst_score = -1.0
    for key, value in eligible.items():
        score_value = (
            float(value["frr"]) * 1000.0
            + float(value["far_per_hour"])
            + float(value["p95_post_end_latency_ms"]) * 0.001
        )
        if score_value > worst_score:
            worst_score = score_value
            worst_key = key

    result = {
        "schema_version": 4,
        "support_policy": {
            "min_domain_expected_wakes": args.min_domain_expected,
            "min_domain_negative_hours": min_negative_hours,
            "eligibility": "positive-wake-support OR negative-recording-support",
        },
        "slice_contract": {
            "distance_bins": DISTANCE_BIN_CONTRACT,
            "azimuth_quantization_deg": 30,
            "snr_bands": SNR_BAND_CONTRACT,
            "pairwise": PAIRWISE_SLICE_CONTRACT,
        },
        "overall": metrics.get("all", {}),
        "domains": metrics,
        "worst_domain": worst_key,
        "worst_domain_score": max(0.0, worst_score),
        "keyword_confusion": confusion_matrix(recordings, detections, pre, post),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
