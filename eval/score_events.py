#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

UINT32_MAX = 0xFFFFFFFF


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def finite_float(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def uint32_value(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < 0 or result > UINT32_MAX:
        raise ValueError(f"{label} must fit uint32")
    return result


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
        duration = finite_float(row.get("duration_s", 0.0), f"{name}: duration_s")
        expected = row.get("expected", [])
        if not name or name in recordings:
            raise ValueError("recording names must be non-empty and unique")
        if duration <= 0.0:
            raise ValueError(f"{name}: duration_s must be > 0")
        if not isinstance(expected, list):
            raise ValueError(f"{name}: expected must be a list")
        normalized: list[dict] = []
        for index, event in enumerate(expected):
            if not isinstance(event, dict):
                raise ValueError(f"{name}: expected[{index}] must be an object")
            keyword_id = uint32_value(
                event["keyword_id"], f"{name}: expected[{index}].keyword_id"
            )
            start = finite_float(
                event["start_s"], f"{name}: expected[{index}].start_s"
            )
            end = finite_float(event["end_s"], f"{name}: expected[{index}].end_s")
            if start < 0.0 or end < start or end > duration:
                raise ValueError(f"{name}: invalid expected window {start}..{end}")
            normalized.append(
                {"keyword_id": keyword_id, "start_s": start, "end_s": end}
            )
        recordings[name] = {
            "recording": name,
            "duration_s": duration,
            "path": row.get("path"),
            "expected": sorted(
                normalized, key=lambda item: (item["start_s"], item["end_s"])
            ),
        }
    if not recordings:
        raise ValueError("reference file contains no recordings")
    return recordings


def validate_detections(
    rows: list[dict], recordings: dict[str, dict]
) -> dict[str, list[dict]]:
    by_recording: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        name = str(row.get("recording", ""))
        if name not in recordings:
            raise ValueError(f"detection references unknown recording: {name}")
        time_s = finite_float(row["time_s"], f"detection[{index}].time_s")
        confidence = finite_float(
            row.get("confidence", 0.0), f"detection[{index}].confidence"
        )
        keyword_id = uint32_value(
            row["keyword_id"], f"detection[{index}].keyword_id"
        )
        if time_s < 0.0 or time_s > recordings[name]["duration_s"]:
            raise ValueError(f"{name}: detection time_s out of range: {time_s}")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"{name}: detection confidence must be in [0,1]")
        by_recording[name].append(
            {
                "recording": name,
                "keyword_id": keyword_id,
                "time_s": time_s,
                "confidence": confidence,
            }
        )
    for items in by_recording.values():
        items.sort(key=lambda item: item["time_s"])
    return by_recording


def better_state(
    candidate: tuple[int, float, int], current: tuple[int, float, int]
) -> bool:
    candidate_matches, candidate_cost, candidate_priority = candidate
    current_matches, current_cost, current_priority = current
    if candidate_matches != current_matches:
        return candidate_matches > current_matches
    if not math.isclose(candidate_cost, current_cost, rel_tol=0.0, abs_tol=1.0e-12):
        return candidate_cost < current_cost
    return candidate_priority > current_priority


def match_keyword_events(
    events: list[tuple[int, dict]],
    detections: list[tuple[int, dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> list[tuple[int, int]]:
    """Maximum-cardinality monotonic match, then minimum end-time distance."""
    rows = len(events)
    cols = len(detections)
    scores = [[(0, 0.0) for _ in range(cols + 1)] for _ in range(rows + 1)]
    choices = [["" for _ in range(cols + 1)] for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        choices[i][0] = "event"
    for j in range(1, cols + 1):
        choices[0][j] = "detection"

    for i in range(1, rows + 1):
        event = events[i - 1][1]
        for j in range(1, cols + 1):
            detection = detections[j - 1][1]
            skip_event = (*scores[i - 1][j], 0)
            skip_detection = (*scores[i][j - 1], 1)
            best = skip_event
            choice = "event"
            if better_state(skip_detection, best):
                best = skip_detection
                choice = "detection"

            lower = event["start_s"] - pre_tolerance_s
            upper = event["end_s"] + post_tolerance_s
            if lower <= detection["time_s"] <= upper:
                previous_matches, previous_cost = scores[i - 1][j - 1]
                matched = (
                    previous_matches + 1,
                    previous_cost + abs(detection["time_s"] - event["end_s"]),
                    2,
                )
                if better_state(matched, best):
                    best = matched
                    choice = "match"

            scores[i][j] = (best[0], best[1])
            choices[i][j] = choice

    pairs: list[tuple[int, int]] = []
    i = rows
    j = cols
    while i > 0 or j > 0:
        choice = choices[i][j]
        if choice == "match":
            pairs.append((events[i - 1][0], detections[j - 1][0]))
            i -= 1
            j -= 1
        elif choice == "event":
            i -= 1
        elif choice == "detection":
            j -= 1
        else:
            raise RuntimeError("internal event-matching backtrack failure")
    pairs.reverse()
    return pairs


def score(
    recordings: dict[str, dict],
    detections: dict[str, list[dict]],
    pre_tolerance_s: float,
    post_tolerance_s: float,
) -> tuple[dict, list[dict], list[dict]]:
    expected_total = 0
    matched_total = 0
    false_reject_count = 0
    false_rejects: list[dict] = []
    false_accepts: list[dict] = []
    latency_ms: list[float] = []
    by_keyword: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "expected": 0,
            "matched": 0,
            "false_rejects": 0,
            "false_accepts": 0,
        }
    )

    for name, recording in recordings.items():
        events = recording["expected"]
        dets = detections.get(name, [])
        events_by_keyword: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        dets_by_keyword: dict[int, list[tuple[int, dict]]] = defaultdict(list)

        for event_index, event in enumerate(events):
            expected_total += 1
            by_keyword[event["keyword_id"]]["expected"] += 1
            events_by_keyword[event["keyword_id"]].append((event_index, event))
        for detection_index, detection in enumerate(dets):
            dets_by_keyword[detection["keyword_id"]].append(
                (detection_index, detection)
            )

        matched_events: set[int] = set()
        used_detections: set[int] = set()
        for keyword_id, keyword_events in events_by_keyword.items():
            pairs = match_keyword_events(
                keyword_events,
                dets_by_keyword.get(keyword_id, []),
                pre_tolerance_s,
                post_tolerance_s,
            )
            for event_index, detection_index in pairs:
                event = events[event_index]
                detection = dets[detection_index]
                matched_events.add(event_index)
                used_detections.add(detection_index)
                matched_total += 1
                by_keyword[keyword_id]["matched"] += 1
                latency_ms.append(
                    max(0.0, (detection["time_s"] - event["end_s"]) * 1000.0)
                )

        for event_index, event in enumerate(events):
            if event_index not in matched_events:
                false_reject_count += 1
                by_keyword[event["keyword_id"]]["false_rejects"] += 1
                item = {
                    "recording": name,
                    "keyword_id": event["keyword_id"],
                    "start_s": event["start_s"],
                    "end_s": event["end_s"],
                    "duration_s": recording["duration_s"],
                }
                if recording.get("path") is not None:
                    item["path"] = recording["path"]
                false_rejects.append(item)

        for detection_index, detection in enumerate(dets):
            if detection_index in used_detections:
                continue
            item = dict(detection)
            item["duration_s"] = recording["duration_s"]
            if recording.get("path") is not None:
                item["path"] = recording["path"]
            false_accepts.append(item)
            by_keyword[detection["keyword_id"]]["false_accepts"] += 1

    total_seconds = sum(item["duration_s"] for item in recordings.values())
    total_hours = total_seconds / 3600.0
    far_per_hour = len(false_accepts) / total_hours
    frr = false_reject_count / expected_total if expected_total else 0.0
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
        "false_rejects": false_reject_count,
        "false_accepts": len(false_accepts),
        "frr": frr,
        "far_per_hour": far_per_hour,
        "p50_post_end_latency_ms": percentile(latency_ms, 0.50),
        "p95_post_end_latency_ms": percentile(latency_ms, 0.95),
        "per_keyword": per_keyword,
    }
    return summary, false_accepts, false_rejects


def validate_gate(value: float | None, label: str, upper: float | None = None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or (
        upper is not None and value > upper
    ):
        suffix = f"..{upper}" if upper is not None else " or greater"
        raise ValueError(f"{label} must be finite and in 0{suffix}")


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--false-positives", type=pathlib.Path)
    parser.add_argument("--false-rejects", type=pathlib.Path)
    parser.add_argument("--max-far-per-hour", type=float)
    parser.add_argument("--max-frr", type=float)
    parser.add_argument("--max-p95-latency-ms", type=float)
    args = parser.parse_args()

    validate_gate(args.pre_tolerance_ms, "pre tolerance")
    validate_gate(args.post_tolerance_ms, "post tolerance")
    validate_gate(args.max_far_per_hour, "max FAR/hour")
    validate_gate(args.max_frr, "max FRR", 1.0)
    validate_gate(args.max_p95_latency_ms, "max p95 latency")

    recordings = validate_recordings(load_jsonl(args.references))
    detections = validate_detections(load_jsonl(args.detections), recordings)
    summary, false_accepts, false_rejects = score(
        recordings,
        detections,
        args.pre_tolerance_ms / 1000.0,
        args.post_tolerance_ms / 1000.0,
    )
    summary["references_sha256"] = sha256_file(args.references)
    summary["detections_sha256"] = sha256_file(args.detections)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")
    if args.false_positives:
        write_jsonl(args.false_positives, false_accepts)
    if args.false_rejects:
        write_jsonl(args.false_rejects, false_rejects)

    failed = False
    if (
        args.max_far_per_hour is not None
        and summary["far_per_hour"] > args.max_far_per_hour
    ):
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
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
