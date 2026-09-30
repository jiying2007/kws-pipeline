#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def normalize_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    return str(pathlib.Path(value).resolve())


def compact_domain(row: dict) -> dict:
    keys = (
        "kind",
        "family_id",
        "keyword_id",
        "tokens",
        "target_ids",
        "path",
        "wav_sha256",
        "source_path",
        "source_wav_sha256",
        "scene_seed",
        "scene",
        "domain_id",
        "split",
    )
    return {key: row[key] for key in keys if key in row}


def build_domain_index(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, row in enumerate(rows):
        path = normalize_path(row.get("path"), f"domain[{index}].path")
        if path in result:
            raise ValueError(f"duplicate rendered domain path: {path}")
        digest = row.get("wav_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"domain[{index}].wav_sha256 is invalid")
        result[path] = row
    return result


def normalize_injections(rows: list[dict], domain_by_path: dict[str, dict]) -> list[dict]:
    result: list[dict] = []
    previous_start = -1.0
    for index, row in enumerate(rows):
        start = finite_number(row.get("start_second"), f"injection[{index}].start_second")
        seconds = finite_number(row.get("source_seconds"), f"injection[{index}].source_seconds")
        if start < 0.0 or seconds <= 0.0:
            raise ValueError(f"injection[{index}] has invalid timing")
        if start < previous_start:
            raise ValueError("injections must be ordered by start_second")
        previous_start = start
        source_path = normalize_path(row.get("source_path"), f"injection[{index}].source_path")
        domain = domain_by_path.get(source_path)
        if domain is None:
            raise ValueError(f"injection source missing from domain index: {source_path}")
        source_sha = row.get("source_sha256")
        if source_sha != domain.get("wav_sha256"):
            raise ValueError(
                f"injection/domain SHA mismatch for {source_path}: "
                f"{source_sha} != {domain.get('wav_sha256')}"
            )
        result.append(
            {
                "index": index,
                "start_second": start,
                "end_second": start + seconds,
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_seconds": seconds,
                "gain": finite_number(row.get("gain"), f"injection[{index}].gain"),
                "coverage_required": bool(row.get("coverage_required", False)),
                "domain": compact_domain(domain),
            }
        )
    for left, right in zip(result, result[1:]):
        if right["start_second"] < left["end_second"] - 1.0e-9:
            raise ValueError("hard-negative injection intervals overlap")
    return result


def annotate(
    detections: list[dict],
    injections: list[dict],
) -> list[dict]:
    output: list[dict] = []
    for detection_index, detection in enumerate(detections):
        time_s = finite_number(
            detection.get("time_s"), f"detection[{detection_index}].time_s"
        )
        if time_s < 0.0:
            raise ValueError(f"detection[{detection_index}].time_s must be >= 0")
        active = [
            row
            for row in injections
            if row["start_second"] <= time_s < row["end_second"]
        ]
        if len(active) > 1:
            raise ValueError(
                f"detection[{detection_index}] maps to overlapping active injections"
            )
        previous = [
            row for row in injections if row["end_second"] <= time_s
        ]
        following = [
            row for row in injections if row["start_second"] > time_s
        ]
        active_row = active[0] if active else None
        previous_row = previous[-1] if previous else None
        next_row = following[0] if following else None
        output.append(
            {
                "schema_version": 1,
                "evidence_class": "far-detection-source-attribution-v1",
                "detection_index": detection_index,
                "detection": detection,
                "active_injection": active_row,
                "previous_injection": previous_row,
                "seconds_since_previous_end": (
                    None
                    if previous_row is None
                    else time_s - previous_row["end_second"]
                ),
                "next_injection": next_row,
                "seconds_until_next_start": (
                    None
                    if next_row is None
                    else next_row["start_second"] - time_s
                ),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-index", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--injections", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    domains = load_jsonl(args.domain_index)
    detections = load_jsonl(args.detections)
    injection_rows = load_jsonl(args.injections)
    domain_by_path = build_domain_index(domains)
    injections = normalize_injections(injection_rows, domain_by_path)
    annotated = annotate(detections, injections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for row in annotated
        )
        + ("\n" if annotated else ""),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "evidence_class": "far-detection-source-attribution-summary-v1",
        "detections": len(annotated),
        "active_source_attributions": sum(
            row["active_injection"] is not None for row in annotated
        ),
        "unattributed_detections": sum(
            row["active_injection"] is None for row in annotated
        ),
        "domain_rows": len(domains),
        "injections": len(injections),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
