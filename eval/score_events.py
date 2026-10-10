#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

if __package__:
    from .context_policy import assess_context
else:
    from context_policy import assess_context

UINT32_MAX = 0xFFFFFFFF
DEFAULT_PRE_TOLERANCE_MS = 150.0
DEFAULT_POST_TOLERANCE_MS = 500.0
AFE_TIMING_CONTRACT = "afe-output-sample-v1"
AFE_SAMPLE_RATE_HZ = 16000


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


def poisson_cdf(count: int, mean: float) -> float:
    if isinstance(count, bool) or count < 0:
        raise ValueError("Poisson count must be a non-negative integer")
    if not math.isfinite(mean) or mean < 0.0:
        raise ValueError("Poisson mean must be finite and non-negative")
    if mean == 0.0:
        return 1.0
    terms = [
        index * math.log(mean) - math.lgamma(index + 1.0)
        for index in range(count + 1)
    ]
    maximum = max(terms)
    log_sum = maximum + math.log(sum(math.exp(value - maximum) for value in terms))
    return math.exp(-mean + log_sum)


def poisson_rate_upper_bound_per_hour(
    count: int,
    exposure_hours: float,
    *,
    confidence: float = 0.95,
) -> float | None:
    if isinstance(count, bool) or count < 0:
        raise ValueError("Poisson count must be a non-negative integer")
    if not math.isfinite(exposure_hours) or exposure_hours < 0.0:
        raise ValueError("Poisson exposure must be finite and non-negative")
    if exposure_hours == 0.0:
        return None
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Poisson confidence must be in (0,1)")

    alpha = 1.0 - confidence
    low = 0.0
    high = max(1.0, float(count + 1))
    while poisson_cdf(count, high) > alpha:
        high *= 2.0
        if high > 1.0e9:
            raise RuntimeError("Poisson upper-bound search did not converge")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if poisson_cdf(count, middle) > alpha:
            low = middle
        else:
            high = middle
    return high / exposure_hours


def validate_recordings(rows: list[dict]) -> dict[str, dict]:
    recordings: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("recording", ""))
        duration = finite_float(row.get("duration_s", 0.0), f"{name}: duration_s")
        if "expected" not in row:
            raise ValueError(f"{name}: expected annotation is required (use [] for negatives)")
        expected = row["expected"]
        if not name or name in recordings:
            raise ValueError("recording names must be non-empty and unique")
        if duration <= 0.0:
            raise ValueError(f"{name}: duration_s must be > 0")
        if not isinstance(expected, list):
            raise ValueError(f"{name}: expected must be a list")
        timing_contract = row.get("timing_contract")
        latency_samples = None
        if timing_contract is not None:
            if timing_contract != AFE_TIMING_CONTRACT:
                raise ValueError(f"{name}: unsupported timing_contract: {timing_contract}")
            latency_samples = row.get("afe_latency_samples")
            if (isinstance(latency_samples, bool)
                    or not isinstance(latency_samples, int)
                    or latency_samples < 0):
                raise ValueError(
                    f"{name}: afe_latency_samples must be a non-negative integer"
                )
            # This contract fixes both raw and output coordinates at 16 kHz.
            if row.get("sample_rate_hz", AFE_SAMPLE_RATE_HZ) != AFE_SAMPLE_RATE_HZ:
                raise ValueError(f"{name}: {AFE_TIMING_CONTRACT} requires 16000 Hz")
            try:
                latency_s = latency_samples / AFE_SAMPLE_RATE_HZ
            except OverflowError as exc:
                raise ValueError(f"{name}: AFE latency must be finite") from exc
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
            match_not_before = event.get("match_not_before_s")
            normalized_event = {
                "keyword_id": keyword_id,
                "start_s": start,
                "end_s": end,
            }
            if timing_contract == AFE_TIMING_CONTRACT:
                raw_start = finite_float(
                    event["raw_start_s"], f"{name}: expected[{index}].raw_start_s"
                )
                raw_end = finite_float(
                    event["raw_end_s"], f"{name}: expected[{index}].raw_end_s"
                )
                if raw_start < 0.0 or raw_end < raw_start:
                    raise ValueError(f"{name}: invalid raw expected window")
                if not (
                    math.isclose(start, raw_start + latency_s,
                                 rel_tol=0.0, abs_tol=1.0e-9)
                    and math.isclose(end, raw_end + latency_s,
                                     rel_tol=0.0, abs_tol=1.0e-9)
                ):
                    raise ValueError(f"{name}: raw/output timing mapping mismatch")
                normalized_event.update(raw_start_s=raw_start, raw_end_s=raw_end)
            if match_not_before is not None:
                match_not_before = finite_float(
                    match_not_before,
                    f"{name}: expected[{index}].match_not_before_s",
                )
                if not start <= match_not_before <= end:
                    raise ValueError(
                        f"{name}: match_not_before_s must lie inside expected window"
                    )
                normalized_event["match_not_before_s"] = match_not_before
            normalized.append(normalized_event)
        recordings[name] = {
            "recording": name,
            "duration_s": duration,
            "path": row.get("path"),
            "expected": sorted(
                normalized, key=lambda item: (item["start_s"], item["end_s"])
            ),
        }
        if timing_contract == AFE_TIMING_CONTRACT:
            recordings[name].update(timing_contract=timing_contract,
                                    afe_latency_samples=latency_samples)
    if not recordings:
        raise ValueError("reference file contains no recordings")
    return recordings


def validate_clip_references(rows: list[dict]) -> dict[str, dict]:
    """Presence labels have no event windows; never fabricate an alignment."""
    normalized = []
    for row in rows:
        if row.get("annotation_status") != "complete":
            raise ValueError("clip presence requires complete annotations; partial/unknown is unscored")
        if "expected" in row:
            raise ValueError("clip presence requires expected_keywords, not event windows")
        keywords = row.get("expected_keywords")
        if not isinstance(keywords, list):
            raise ValueError("clip presence requires explicit expected_keywords ([] for negatives)")
        ids = [uint32_value(value, "expected keyword") for value in keywords]
        if len(ids) != len(set(ids)):
            raise ValueError("clip presence keyword IDs must be unique")
        normalized.append(dict(row, expected=[]))
    recordings = validate_recordings(normalized)
    for row in rows:
        recordings[row["recording"]]["expected_keywords"] = row["expected_keywords"]
    return recordings


def score_clip_presence(recordings: dict, detections: dict) -> dict:
    counts = dict(positive_clips=0, positive_clips_with_target=0, negative_clips=0,
                  negative_clips_with_events=0, negative_events=0,
                  missing_keywords=0, wrong_keyword_events=0, repeat_events=0)
    details = []
    for name, row in recordings.items():
        targets = {uint32_value(value, "expected keyword") for value in row["expected_keywords"]}
        events = detections.get(name, [])
        seen = defaultdict(int)
        for event in events:
            seen[event["keyword_id"]] += 1
        missing = sorted(targets - seen.keys())
        wrong = sum(n for key, n in seen.items() if key not in targets) if targets else 0
        repeats = sum(max(0, n - 1) for n in seen.values())
        counts["positive_clips" if targets else "negative_clips"] += 1
        counts["positive_clips_with_target"] += bool(targets & seen.keys())
        counts["negative_clips_with_events"] += bool(events) and not targets
        counts["negative_events"] += len(events) if not targets else 0
        counts["missing_keywords"] += len(missing)
        counts["wrong_keyword_events"] += wrong
        counts["repeat_events"] += repeats
        details.append(dict(recording=name, expected_keywords=sorted(targets),
                            missing_keywords=missing, wrong_keyword_events=wrong,
                            repeat_events=repeats, detections=events))
    return dict(evidence_class="saved-clip-presence-diagnostic-v1", **counts,
                event_annotations_available=False, qualification_allowed=False,
                timing_scope="saved detection coordinates only; acoustic word end unknown",
                recordings=details)


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
        if (recordings[name].get("timing_contract") == AFE_TIMING_CONTRACT
                and row.get("timing_contract") == "afe-timeline-v1"):
            output_end = row.get("output_end_sample")
            decision_ns = row.get("decision_capture_ns")
            raw_ns = row.get("raw_acoustic_end_ns")
            if (any(type(value) is not int for value in (output_end, decision_ns, raw_ns))
                    or row.get("capture_epoch") != 0
                    or decision_ns != output_end * 62500
                    or raw_ns != decision_ns - recordings[name]["afe_latency_samples"] * 62500
                    or not math.isclose(time_s, output_end / AFE_SAMPLE_RATE_HZ,
                                        rel_tol=0.0, abs_tol=0.00000051)):
                raise ValueError(f"{name}: detection timeline conflicts with constant-delay references")
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
            if "match_not_before_s" in event:
                lower = max(lower, event["match_not_before_s"])
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
    signed_end_offset_ms: list[float] = []
    raw_signed_end_offset_ms: list[float] = []
    timing_recordings = sum(
        item.get("timing_contract") == AFE_TIMING_CONTRACT
        for item in recordings.values()
    )
    complete_afe_timing = timing_recordings == len(recordings)
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
                signed_end_offset_ms.append(
                    (detection["time_s"] - event["end_s"]) * 1000.0
                )
                if complete_afe_timing:
                    raw_signed_end_offset_ms.append(
                        (detection["time_s"] - event["raw_end_s"]) * 1000.0
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

    negative_recordings = {
        name for name, item in recordings.items() if not item["expected"]
    }
    negative_seconds = sum(
        float(recordings[name]["duration_s"]) for name in negative_recordings
    )
    longest_negative_recording_seconds = max(
        (float(recordings[name]["duration_s"]) for name in negative_recordings),
        default=0.0,
    )
    negative_hours = negative_seconds / 3600.0
    negative_false_accepts = sum(
        1 for item in false_accepts if item["recording"] in negative_recordings
    )
    negative_far_per_hour = (
        negative_false_accepts / negative_hours if negative_hours > 0.0 else None
    )
    negative_far_upper_95 = poisson_rate_upper_bound_per_hour(
        negative_false_accepts,
        negative_hours,
        confidence=0.95,
    )
    frr = false_reject_count / expected_total if expected_total else 0.0
    per_keyword = {}
    for keyword_id, stats in sorted(by_keyword.items()):
        expected = stats["expected"]
        per_keyword[str(keyword_id)] = {
            **stats,
            "frr": stats["false_rejects"] / expected if expected else 0.0,
        }

    summary = {
        "event_match_pre_tolerance_ms": pre_tolerance_s * 1000.0,
        "event_match_post_tolerance_ms": post_tolerance_s * 1000.0,
        "recordings": len(recordings),
        "audio_hours": total_hours,
        "expected": expected_total,
        "matched": matched_total,
        "false_rejects": false_reject_count,
        "false_accepts": len(false_accepts),
        "frr": frr,
        "far_per_hour": far_per_hour,
        "negative_recording_audio_hours": negative_hours,
        "longest_negative_recording_seconds": longest_negative_recording_seconds,
        "negative_recording_false_accepts": negative_false_accepts,
        "negative_recording_far_per_hour": negative_far_per_hour,
        "negative_recording_far_upper_95_per_hour": negative_far_upper_95,
        "negative_recording_far_confidence": 0.95,
        "negative_recording_far_policy": "negative-only-recordings-poisson-upper-v1",
        "p50_post_end_latency_ms": percentile(latency_ms, 0.50),
        "p95_post_end_latency_ms": percentile(latency_ms, 0.95),
        "p50_signed_end_offset_ms": percentile(signed_end_offset_ms, 0.50),
        "p95_signed_end_offset_ms": percentile(signed_end_offset_ms, 0.95),
        "early_detection_count": sum(value < 0.0 for value in signed_end_offset_ms),
        "timing_coordinates": {
            "scope": "post-afe-output" if complete_afe_timing else "reference",
            "afe_contract_recordings": timing_recordings,
            "total_recordings": len(recordings),
            "raw_metrics_available": complete_afe_timing,
        },
        "per_keyword": per_keyword,
    }
    if complete_afe_timing:
        summary["timing_coordinates"].update(
            contract=AFE_TIMING_CONTRACT,
            sample_rate_hz=AFE_SAMPLE_RATE_HZ,
            matched_events=matched_total,
            raw_scope="raw-capture-end-to-decision; includes declared AFE delay",
        )
        for label, values in (("post_afe", signed_end_offset_ms),
                              ("raw_capture", raw_signed_end_offset_ms)):
            for quantile, probability in ((50, 0.50), (95, 0.95)):
                summary[f"p{quantile}_{label}_signed_end_offset_ms"] = (
                    percentile(values, probability) if values else None
                )
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
    parser.add_argument("--clip-presence", action="store_true",
                        help="report whole-clip counts without FRR/FAR/latency")
    parser.add_argument("--context-manifest", type=pathlib.Path,
                        help="eval-context-v1 declarations bound to reference/detection bytes")
    parser.add_argument("--require-matched-context", action="store_true",
                        help="fail unless positive and negative context declarations match")
    parser.add_argument("--pre-tolerance-ms", type=float, default=DEFAULT_PRE_TOLERANCE_MS)
    parser.add_argument("--post-tolerance-ms", type=float, default=DEFAULT_POST_TOLERANCE_MS)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--false-positives", type=pathlib.Path)
    parser.add_argument("--false-rejects", type=pathlib.Path)
    parser.add_argument("--max-far-per-hour", type=float)
    parser.add_argument("--max-frr", type=float)
    parser.add_argument("--max-p95-latency-ms", type=float)
    parser.add_argument("--min-expected-per-keyword", action="append", nargs=2,
                        metavar=("KEYWORD_ID", "COUNT"), default=[])
    parser.add_argument("--min-negative-hours", type=float)
    parser.add_argument("--min-continuous-negative-seconds", type=float)
    parser.add_argument("--max-negative-far-upper-95-per-hour", type=float)
    args = parser.parse_args()

    validate_gate(args.pre_tolerance_ms, "pre tolerance")
    validate_gate(args.post_tolerance_ms, "post tolerance")
    validate_gate(args.max_far_per_hour, "max FAR/hour")
    validate_gate(args.max_frr, "max FRR", 1.0)
    validate_gate(args.max_p95_latency_ms, "max p95 latency")
    validate_gate(args.min_negative_hours, "min negative hours")
    validate_gate(args.min_continuous_negative_seconds,
                  "min continuous negative seconds")
    validate_gate(args.max_negative_far_upper_95_per_hour,
                  "max negative FAR upper 95/hour")
    required_keywords: dict[str, int] = {}
    for raw_id, raw_count in args.min_expected_per_keyword:
        keyword_id = str(uint32_value(raw_id, "required keyword ID"))
        count = uint32_value(raw_count, "required keyword count")
        if count == 0 or keyword_id in required_keywords:
            raise ValueError("required keyword counts must be positive and IDs unique")
        required_keywords[keyword_id] = count

    rows = load_jsonl(args.references)
    recordings = (validate_clip_references(rows) if args.clip_presence
                  else validate_recordings(rows))
    detections = validate_detections(load_jsonl(args.detections), recordings)
    context = assess_context(args.context_manifest, args.references, args.detections,
                             rows, clip_presence=args.clip_presence)
    if args.require_matched_context and not context["matched_context"]:
        raise ValueError("matched context required: " + context["status"] + "; " +
                         ", ".join(context["unknown"] + context["differences"]))
    if args.clip_presence:
        metric_options = (args.max_far_per_hour, args.max_frr, args.max_p95_latency_ms,
                          args.min_negative_hours, args.min_continuous_negative_seconds,
                          args.max_negative_far_upper_95_per_hour)
        if (any(value is not None for value in metric_options) or required_keywords
                or args.false_positives or args.false_rejects):
            raise ValueError("clip presence cannot satisfy event metric gates or mining outputs")
        summary = score_clip_presence(recordings, detections)
        summary.update(context=context, references_sha256=sha256_file(args.references),
                       detections_sha256=sha256_file(args.detections))
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
        print(rendered)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(rendered + "\n", encoding="utf-8")
        return 0
    summary, false_accepts, false_rejects = score(
        recordings,
        detections,
        args.pre_tolerance_ms / 1000.0,
        args.post_tolerance_ms / 1000.0,
    )
    summary["references_sha256"] = sha256_file(args.references)
    summary["detections_sha256"] = sha256_file(args.detections)
    summary["context"] = context
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
    if args.max_frr is not None and summary["expected"] == 0:
        print("gate failed: FRR has no positive events", file=sys.stderr)
        failed = True
    if args.max_p95_latency_ms is not None and summary["matched"] == 0:
        print("gate failed: latency has no matched events", file=sys.stderr)
        failed = True
    for keyword_id, count in required_keywords.items():
        observed = summary["per_keyword"].get(keyword_id, {}).get("expected", 0)
        if observed < count:
            print(f"gate failed: keyword {keyword_id} expected coverage", file=sys.stderr)
            failed = True
    if (args.min_negative_hours is not None
            and summary["negative_recording_audio_hours"] < args.min_negative_hours):
        print("gate failed: negative recording hours", file=sys.stderr)
        failed = True
    if (args.min_continuous_negative_seconds is not None
            and summary["longest_negative_recording_seconds"]
            < args.min_continuous_negative_seconds):
        print("gate failed: continuous negative recording duration", file=sys.stderr)
        failed = True
    if args.max_negative_far_upper_95_per_hour is not None:
        upper = summary["negative_recording_far_upper_95_per_hour"]
        if upper is None or upper > args.max_negative_far_upper_95_per_hour:
            print("gate failed: negative FAR upper 95/hour", file=sys.stderr)
            failed = True
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
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
