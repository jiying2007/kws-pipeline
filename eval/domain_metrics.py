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
    score,
    validate_detections,
    validate_recordings,
)


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def domain_keys(row: dict) -> list[str]:
    domain = row.get("domain")
    if not isinstance(domain, dict):
        return ["all"]
    distance = str(domain.get("distance_band", "unknown"))
    azimuth = float(domain.get("azimuth_deg", 0.0))
    rt60 = float(domain.get("rt60_s", 0.0))
    noise = str(domain.get("noise_profile", "unknown"))
    playback = "playback" if domain.get("playback_sir_db") is not None else "no-playback"
    if abs(azimuth) <= 30.0:
        az_band = "front"
    elif abs(azimuth) <= 90.0:
        az_band = "side"
    else:
        az_band = "rear"
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
        f"azimuth:{az_band}",
        f"rt60:{rt_band}",
        f"noise:{noise}",
        f"playback:{playback}",
        f"composite:{composite}",
    ]


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
    return summary


def confusion_matrix(
    raw_rows: list[dict],
    detections: dict[str, list[dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> dict:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exact = 0
    wrong = 0
    missing = 0
    for row in raw_rows:
        name = str(row["recording"])
        dets = detections.get(name, [])
        for event in row.get("expected", []):
            expected_id = int(event["keyword_id"])
            lower = float(event["start_s"]) - pre_tolerance_s
            upper = float(event["end_s"]) + post_tolerance_s
            candidates = [
                item for item in dets if lower <= float(item["time_s"]) <= upper
            ]
            if not candidates:
                matrix[str(expected_id)]["<miss>"] += 1
                missing += 1
                continue
            best = min(
                candidates,
                key=lambda item: (
                    abs(float(item["time_s"]) - float(event["end_s"])),
                    -float(item["confidence"]),
                ),
            )
            detected_id = int(best["keyword_id"])
            matrix[str(expected_id)][str(detected_id)] += 1
            if detected_id == expected_id:
                exact += 1
            else:
                wrong += 1
    return {
        "expected_events": exact + wrong + missing,
        "correct_keyword": exact,
        "wrong_keyword": wrong,
        "missed": missing,
        "matrix": {key: dict(value) for key, value in sorted(matrix.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    parser.add_argument("--min-domain-expected", type=int, default=1)
    args = parser.parse_args()
    pre = finite(args.pre_tolerance_ms, "pre tolerance") / 1000.0
    post = finite(args.post_tolerance_ms, "post tolerance") / 1000.0
    if pre < 0.0 or post < 0.0 or args.min_domain_expected < 0:
        raise ValueError("domain metric tolerances/counts must be non-negative")

    raw_rows = load_jsonl(args.references)
    recordings = validate_recordings(raw_rows)
    detections = validate_detections(load_jsonl(args.detections), recordings)
    raw_by_name = {str(row["recording"]): row for row in raw_rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for name, row in raw_by_name.items():
        for key in domain_keys(row):
            groups[key].append(name)

    metrics: dict[str, dict] = {}
    for key, names in sorted(groups.items()):
        summary = subset_score(names, recordings, detections, pre, post)
        if key != "all" and int(summary["expected"]) < args.min_domain_expected:
            continue
        metrics[key] = summary

    eligible = {
        key: value
        for key, value in metrics.items()
        if key != "all" and int(value["expected"]) >= args.min_domain_expected
    }
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
        "schema_version": 1,
        "overall": metrics.get("all", {}),
        "domains": metrics,
        "worst_domain": worst_key,
        "worst_domain_score": max(0.0, worst_score),
        "keyword_confusion": confusion_matrix(raw_rows, detections, pre, post),
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
