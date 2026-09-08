#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from statistical_bounds import poisson_rate_upper, wilson_upper  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    return rows


def in_slice(row: dict, rule: dict) -> bool:
    if "tag" in rule:
        return str(rule["tag"]) in set(row.get("tags", []))
    value = float(row[str(rule["field"])])
    if "min" in rule and value < float(rule["min"]):
        return False
    if "min_exclusive" in rule and value <= float(rule["min_exclusive"]):
        return False
    if "max" in rule and value > float(rule["max"]):
        return False
    if "max_exclusive" in rule and value >= float(rule["max_exclusive"]):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--intake-summary", required=True, type=pathlib.Path)
    parser.add_argument("--afe-summary", required=True, type=pathlib.Path)
    parser.add_argument("--score-summary", required=True, type=pathlib.Path)
    parser.add_argument("--false-accepts", required=True, type=pathlib.Path)
    parser.add_argument("--false-rejects", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    intake = json.loads(args.intake_summary.read_text(encoding="utf-8"))
    afe = json.loads(args.afe_summary.read_text(encoding="utf-8"))
    score = json.loads(args.score_summary.read_text(encoding="utf-8"))
    false_accepts = load_jsonl(args.false_accepts)
    false_rejects = load_jsonl(args.false_rejects)

    deployment_tag = str(policy["deployment_tag"])
    if manifest.get("deployment_tag") != deployment_tag or intake.get("deployment_tag") != deployment_tag or afe.get("deployment_tag") != deployment_tag:
        raise ValueError("deployment identity differs across policy/corpus/AFE")
    if afe.get("afe", {}).get("backend") != "command" or afe.get("afe", {}).get("shipping_authority") is not True:
        raise ValueError("final AFE evidence is not shipping-authoritative command backend")
    if intake.get("raw_audio_hashes_verified") is not True or intake.get("pii_fields_rejected") is not True:
        raise ValueError("corpus intake evidence is incomplete")

    recordings = {str(row["recording"]): row for row in manifest["recordings"]}
    negative_recordings = {name for name, row in recordings.items() if not row.get("expected")}
    positive_recordings = set(recordings) - negative_recordings
    negative_hours = sum(float(recordings[name]["duration_s"]) for name in negative_recordings) / 3600.0
    if negative_hours <= 0.0:
        raise ValueError("qualification has no negative exposure")

    false_accepts_negative = [row for row in false_accepts if str(row.get("recording")) in negative_recordings]
    false_accepts_positive = [row for row in false_accepts if str(row.get("recording")) in positive_recordings]
    false_reject_counts = Counter((str(row["recording"]), int(row["keyword_id"])) for row in false_rejects)

    confidence = float(policy["confidence_level"])
    expected_total = int(score["expected"])
    fr_total = int(score["false_rejects"])
    frr = fr_total / expected_total if expected_total else 0.0
    frr_upper = wilson_upper(fr_total, expected_total, confidence)
    far = len(false_accepts_negative) / negative_hours
    far_upper = poisson_rate_upper(len(false_accepts_negative), negative_hours, confidence)

    expected_by_keyword = Counter()
    fr_by_keyword = Counter()
    for row in recordings.values():
        for event in row.get("expected", []):
            keyword = int(event["keyword_id"])
            expected_by_keyword[keyword] += 1
            fr_by_keyword[keyword] += false_reject_counts[(str(row["recording"]), keyword)]
            false_reject_counts[(str(row["recording"]), keyword)] = 0

    per_keyword = {}
    for keyword in (1, 2):
        expected = expected_by_keyword[keyword]
        rejects = fr_by_keyword[keyword]
        per_keyword[str(keyword)] = {
            "expected": expected,
            "false_rejects": rejects,
            "frr": rejects / expected if expected else 0.0,
            "frr_upper_bound": wilson_upper(rejects, expected, confidence) if expected else 1.0,
        }

    slice_results = {}
    for rule in policy.get("critical_positive_slices", []):
        expected = Counter()
        rejects = Counter()
        for row in recordings.values():
            if not row.get("expected") or not in_slice(row, rule):
                continue
            recording = str(row["recording"])
            for event in row["expected"]:
                keyword = int(event["keyword_id"])
                expected[keyword] += 1
                rejects[keyword] += sum(1 for fr in false_rejects if str(fr.get("recording")) == recording and int(fr.get("keyword_id", 0)) == keyword)
        item = {}
        for keyword in (1, 2):
            count = expected[keyword]
            missed = rejects[keyword]
            item[str(keyword)] = {
                "expected": count,
                "false_rejects": missed,
                "frr": missed / count if count else 1.0,
            }
        slice_results[str(rule["name"])] = item

    failures = []
    minimums = policy["minimums"]
    gates = policy["gates"]
    if int(intake["positive_speakers"]) < int(minimums["speakers"]):
        failures.append("minimum-speakers")
    if int(intake["expected_wakes"]) < int(minimums["expected_wakes_total"]):
        failures.append("minimum-expected-wakes")
    if negative_hours + 1e-12 < float(minimums["negative_audio_hours"]):
        failures.append("minimum-negative-hours")
    if frr > float(gates["max_frr"]):
        failures.append("frr")
    if frr_upper > float(gates["max_frr_upper_bound"]):
        failures.append("frr-upper-bound")
    if far > float(gates["max_far_per_hour"]):
        failures.append("far")
    if far_upper > float(gates["max_far_upper_bound_per_hour"]):
        failures.append("far-upper-bound")
    if float(score["p95_post_end_latency_ms"]) > float(gates["max_p95_post_end_latency_ms"]):
        failures.append("p95-latency")
    if false_accepts_positive:
        failures.append("unexpected-detections-in-positive-recordings")

    for keyword in (1, 2):
        if expected_by_keyword[keyword] < int(minimums["expected_wakes_per_keyword"]):
            failures.append(f"keyword-{keyword}-minimum")
    for rule in policy.get("critical_positive_slices", []):
        name = str(rule["name"])
        for keyword in (1, 2):
            stats = slice_results[name][str(keyword)]
            if stats["expected"] < int(rule["min_expected_per_keyword"]):
                failures.append(f"slice-{name}-keyword-{keyword}-minimum")
            if stats["frr"] > float(rule["max_frr"]):
                failures.append(f"slice-{name}-keyword-{keyword}-frr")

    result = {
        "schema_version": 1,
        "phase": "real-human-final-afe-acoustic-qualification",
        "qualified": not failures,
        "shipping_approved": False,
        "deployment_tag": deployment_tag,
        "qualification_id": manifest["qualification_id"],
        "corpus_sha256": intake["corpus_sha256"],
        "confidence_level": confidence,
        "recordings": len(recordings),
        "positive_speakers": intake["positive_speakers"],
        "expected_wakes": expected_total,
        "false_rejects": fr_total,
        "frr": frr,
        "frr_upper_bound": frr_upper,
        "negative_audio_hours": negative_hours,
        "false_accepts_negative": len(false_accepts_negative),
        "far_per_hour": far,
        "far_upper_bound_per_hour": far_upper,
        "unexpected_detections_positive": len(false_accepts_positive),
        "p50_post_end_latency_ms": score["p50_post_end_latency_ms"],
        "p95_post_end_latency_ms": score["p95_post_end_latency_ms"],
        "per_keyword": per_keyword,
        "critical_positive_slices": slice_results,
        "afe": afe["afe"],
        "evidence_sha256": {
            "manifest": sha256_file(args.manifest),
            "policy": sha256_file(args.policy),
            "intake_summary": sha256_file(args.intake_summary),
            "afe_summary": sha256_file(args.afe_summary),
            "score_summary": sha256_file(args.score_summary),
            "false_accepts": sha256_file(args.false_accepts),
            "false_rejects": sha256_file(args.false_rejects),
            "detections": sha256_file(args.detections),
        },
        "failures": sorted(set(failures)),
        "next_gate": "physical-target-board-performance-and-soak" if not failures else "real-human-qualification-failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
