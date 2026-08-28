#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def validate_recordings(rows: list[dict]) -> dict[str, dict]:
    recordings: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("recording", ""))
        duration = float(row.get("duration_s", 0.0))
        expected = row.get("expected", [])
        if not name or name in recordings:
            raise ValueError("recording names must be non-empty and unique")
        if duration <= 0.0:
            raise ValueError(f"{name}: duration_s must be > 0")
        if not isinstance(expected, list):
            raise ValueError(f"{name}: expected must be a list")
        normalized: list[dict] = []
        for event in expected:
            keyword_id = int(event["keyword_id"])
            start = float(event["start_s"])
            end = float(event["end_s"])
            if start < 0.0 or end < start or end > duration:
                raise ValueError(f"{name}: invalid expected window {start}..{end}")
            normalized.append(
                {"keyword_id": keyword_id, "start_s": start, "end_s": end}
            )
        recordings[name] = {
            "recording": name,
            "duration_s": duration,
            "path": row.get("path"),
            "expected": sorted(normalized, key=lambda item: item["start_s"]),
        }
    return recordings


def validate_detections(rows: list[dict], recordings: dict[str, dict]) -> dict[str, list[dict]]:
    by_recording: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        name = str(row.get("recording", ""))
        if name not in recordings:
            raise ValueError(f"detection references unknown recording: {name}")
        time_s = float(row["time_s"])
        if time_s < 0.0 or time_s > recordings[name]["duration_s"]:
            raise ValueError(f"{name}: detection time_s out of range: {time_s}")
        by_recording[name].append(
            {
                "recording": name,
                "keyword_id": int(row["keyword_id"]),
                "time_s": time_s,
                "confidence": float(row.get("confidence", 0.0)),
            }
        )
    for detections in by_recording.values():
        detections.sort(key=lambda item: item["time_s"])
    return by_recording


def score(
    recordings: dict[str, dict],
    detections: dict[str, list[dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> tuple[dict, list[dict]]:
    expected_total = 0
    matched_total = 0
    false_rejects = 0
    false_accepts: list[dict] = []
    latency_ms: list[float] = []
    by_keyword: dict[int, dict[str, int]] = defaultdict(
        lambda: {"expected": 0, "matched": 0, "false_rejects": 0, "false_accepts": 0}
    )

    for name, recording in recordings.items():
        dets = detections.get(name, [])
        used: set[int] = set()
        for event in recording["expected"]:
            expected_total += 1
            keyword_id = event["keyword_id"]
            by_keyword[keyword_id]["expected"] += 1
            candidates = [
                (idx, det)
                for idx, det in enumerate(dets)
                if idx not in used
                and det["keyword_id"] == keyword_id
                and event["start_s"] - pre_tolerance_s
                <= det["time_s"]
                <= event["end_s"] + post_tolerance_s
            ]
            if not candidates:
                false_rejects += 1
                by_keyword[keyword_id]["false_rejects"] += 1
                continue
            idx, det = min(
                candidates,
                key=lambda pair: abs(pair[1]["time_s"] - event["end_s"]),
            )
            used.add(idx)
            matched_total += 1
            by_keyword[keyword_id]["matched"] += 1
            latency_ms.append(max(0.0, (det["time_s"] - event["end_s"]) * 1000.0))

        for idx, det in enumerate(dets):
            if idx in used:
                continue
            item = dict(det)
            item["duration_s"] = recording["duration_s"]
            if recording.get("path") is not None:
                item["path"] = recording["path"]
            false_accepts.append(item)
            by_keyword[det["keyword_id"]]["false_accepts"] += 1

    total_seconds = sum(item["duration_s"] for item in recordings.values())
    total_hours = total_seconds / 3600.0
    far_per_hour = len(false_accepts) / total_hours if total_hours > 0.0 else 0.0
    frr = false_rejects / expected_total if expected_total else 0.0
    per_keyword = {}
    for keyword_id, stats in sorted(by_keyword.items()):
        expected = stats["expected"]
        per_keyword[str(keyword_id)] = {
            **stats,
            "frr": stats["false_rejects"] / expected if expected else 0.0,
        }

    summary = {
        "recordings": len(recordings),
        "audio_hours": total_hours,
        "expected": expected_total,
        "matched": matched_total,
        "false_rejects": false_rejects,
        "false_accepts": len(false_accepts),
        "frr": frr,
        "far_per_hour": far_per_hour,
        "p50_post_end_latency_ms": percentile(latency_ms, 0.50),
        "p95_post_end_latency_ms": percentile(latency_ms, 0.95),
        "per_keyword": per_keyword,
    }
    return summary, false_accepts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--false-positives", type=pathlib.Path)
    parser.add_argument("--max-far-per-hour", type=float)
    parser.add_argument("--max-frr", type=float)
    parser.add_argument("--max-p95-latency-ms", type=float)
    args = parser.parse_args()

    if args.pre_tolerance_ms < 0.0 or args.post_tolerance_ms < 0.0:
        parser.error("tolerances must be >= 0")

    recordings = validate_recordings(load_jsonl(args.references))
    detections = validate_detections(load_jsonl(args.detections), recordings)
    summary, false_accepts = score(
        recordings,
        detections,
        args.pre_tolerance_ms / 1000.0,
        args.post_tolerance_ms / 1000.0,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")
    if args.false_positives:
        args.false_positives.parent.mkdir(parents=True, exist_ok=True)
        with args.false_positives.open("w", encoding="utf-8") as stream:
            for row in false_accepts:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed = False
    if args.max_far_per_hour is not None and summary["far_per_hour"] > args.max_far_per_hour:
        print("gate failed: FAR/hour", file=sys.stderr)
        failed = True
    if args.max_frr is not None and summary["frr"] > args.max_frr:
        print("gate failed: FRR", file=sys.stderr)
        failed = True
    if (
        args.max_p95_latency_ms is not None
        and summary["p95_post_end_latency_ms"] > args.max_p95_latency_ms
    ):
        print("gate failed: p95 latency", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
