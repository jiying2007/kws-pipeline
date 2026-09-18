#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
from collections import defaultdict

EVIDENCE_CLASS = "kws-v2-score-separation-diagnostic-v1"


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
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
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


def stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def parse_keywords(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_no}: expected id/text/threshold TSV")
        keyword_id = int(fields[0])
        threshold = finite(float(fields[2]), f"{path}:{line_no}: threshold")
        if keyword_id in seen or keyword_id < 0 or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: invalid keyword row")
        seen.add(keyword_id)
        rows.append(
            {
                "id": keyword_id,
                "text": fields[1],
                "threshold": threshold,
            }
        )
    if not rows:
        raise ValueError("keyword table is empty")
    return rows


def validate_references(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, row in enumerate(rows):
        recording = str(row.get("recording", ""))
        audio_path = row.get("audio_path") or row.get("path")
        duration = finite(row.get("duration_s"), f"reference[{index}].duration_s")
        expected = row.get("expected", [])
        if not recording or recording in result:
            raise ValueError("reference recordings must be non-empty and unique")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError(f"{recording}: audio path is required")
        if duration <= 0.0 or not isinstance(expected, list):
            raise ValueError(f"{recording}: invalid reference geometry")
        normalized: list[dict] = []
        for event_index, event in enumerate(expected):
            if not isinstance(event, dict):
                raise ValueError(f"{recording}: expected[{event_index}] must be an object")
            keyword_id = int(event["keyword_id"])
            start = finite(event["start_s"], f"{recording}: expected start")
            end = finite(event["end_s"], f"{recording}: expected end")
            if start < 0.0 or end < start or end > duration:
                raise ValueError(f"{recording}: invalid expected window")
            normalized.append(
                {"keyword_id": keyword_id, "start_s": start, "end_s": end}
            )
        domain = row.get("domain")
        distance_band = None
        if isinstance(domain, dict) and domain.get("distance_band") is not None:
            distance_band = str(domain["distance_band"])
        result[recording] = {
            "recording": recording,
            "audio_path": audio_path.strip(),
            "duration_s": duration,
            "expected": normalized,
            "distance_band": distance_band,
        }
    if not result:
        raise ValueError("reference corpus is empty")
    return result


def run_probe(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    references: dict[str, dict],
    frames_path: pathlib.Path,
) -> list[dict]:
    lines: list[str] = []
    for recording, ref in references.items():
        completed = subprocess.run(
            [
                str(runner),
                str(model),
                str(pack),
                str(ref["audio_path"]),
                recording,
                "--score-probe",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        lines.extend(raw for raw in completed.stdout.splitlines() if raw.strip())
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return load_jsonl(frames_path)


def validate_frames(
    rows: list[dict],
    references: dict[str, dict],
    keyword_ids: set[int],
) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    last_time: dict[tuple[str, int], float] = {}
    for index, row in enumerate(rows):
        recording = str(row.get("recording", ""))
        keyword_id = int(row.get("keyword_id", -1))
        if recording not in references:
            raise ValueError(f"probe[{index}]: unknown recording")
        if keyword_id not in keyword_ids:
            raise ValueError(f"probe[{index}]: unknown keyword id {keyword_id}")
        time_s = finite(row.get("time_s"), f"probe[{index}].time_s")
        confidence = finite(row.get("confidence"), f"probe[{index}].confidence")
        retention_log = finite(
            row.get("retention_log"), f"probe[{index}].retention_log"
        )
        if time_s < 0.0 or time_s > references[recording]["duration_s"]:
            raise ValueError(f"probe[{index}]: time outside recording")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"probe[{index}]: confidence outside [0,1]")
        key = (recording, keyword_id)
        if key in last_time and time_s < last_time[key] - 1.0e-12:
            raise ValueError(f"probe[{index}]: non-monotonic time")
        last_time[key] = time_s
        grouped[key].append(
            {
                "time_s": time_s,
                "confidence": confidence,
                "retention_log": retention_log,
            }
        )
    return grouped


def peak(items: list[dict], lower: float | None = None, upper: float | None = None) -> float:
    values = [
        float(item["confidence"])
        for item in items
        if (lower is None or float(item["time_s"]) >= lower)
        and (upper is None or float(item["time_s"]) <= upper)
    ]
    return max(values) if values else 0.0


def outside_peak(items: list[dict], windows: list[tuple[float, float]]) -> float:
    values: list[float] = []
    for item in items:
        time_s = float(item["time_s"])
        if not any(low <= time_s <= high for low, high in windows):
            values.append(float(item["confidence"]))
    return max(values) if values else 0.0


def pairwise_separation(target: list[float], background: list[float]) -> float:
    if not target or not background:
        return 0.0
    wins = 0.0
    total = 0
    for positive in target:
        for negative in background:
            total += 1
            if positive > negative:
                wins += 1.0
            elif math.isclose(positive, negative, rel_tol=0.0, abs_tol=1.0e-12):
                wins += 0.5
    return wins / total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only GRU terminal-confidence separation diagnostic. "
            "It observes decoder state and never changes product thresholds or selection."
        )
    )
    parser.add_argument("--runner", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--keyword-pack", type=pathlib.Path)
    parser.add_argument("--keyword-table", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--frames", required=True, type=pathlib.Path)
    parser.add_argument("--reuse-frames", action="store_true")
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    keywords = parse_keywords(args.keyword_table)
    keyword_ids = {int(row["id"]) for row in keywords}
    references = validate_references(load_jsonl(args.references))
    for recording in references.values():
        for event in recording["expected"]:
            if int(event["keyword_id"]) not in keyword_ids:
                raise ValueError("reference expected keyword is absent from keyword table")

    if args.reuse_frames:
        if not args.frames.is_file():
            raise ValueError("--reuse-frames requires an existing --frames file")
        frame_rows = load_jsonl(args.frames)
        runner_sha = None
        model_sha = None
        pack_sha = None
    else:
        for path, label in (
            (args.runner, "runner"),
            (args.model, "model"),
            (args.keyword_pack, "keyword pack"),
        ):
            if path is None or not path.is_file():
                raise ValueError(f"{label} is required and must exist")
        frame_rows = run_probe(
            runner=args.runner,
            model=args.model,
            pack=args.keyword_pack,
            references=references,
            frames_path=args.frames,
        )
        runner_sha = sha256_file(args.runner)
        model_sha = sha256_file(args.model)
        pack_sha = sha256_file(args.keyword_pack)

    grouped = validate_frames(frame_rows, references, keyword_ids)
    pre = finite(args.pre_tolerance_ms, "pre tolerance") / 1000.0
    post = finite(args.post_tolerance_ms, "post tolerance") / 1000.0
    if pre < 0.0 or post < 0.0:
        raise ValueError("tolerances must be non-negative")

    target_scores: list[float] = []
    negative_scores: list[float] = []
    cross_keyword_scores: list[float] = []
    outside_target_scores: list[float] = []
    by_keyword: dict[int, dict[str, list[float]]] = {
        int(row["id"]): {
            "target": [],
            "negative": [],
            "cross_keyword": [],
            "outside_target": [],
        }
        for row in keywords
    }
    by_distance: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"target": [], "background": []}
    )

    for recording, ref in references.items():
        expected_by_keyword: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for event in ref["expected"]:
            expected_by_keyword[int(event["keyword_id"])].append(
                (
                    max(0.0, float(event["start_s"]) - pre),
                    min(ref["duration_s"], float(event["end_s"]) + post),
                )
            )
        is_negative = not ref["expected"]
        distance = ref["distance_band"] or "unspecified"
        for keyword in keywords:
            keyword_id = int(keyword["id"])
            items = grouped.get((recording, keyword_id), [])
            if is_negative:
                value = peak(items)
                negative_scores.append(value)
                by_keyword[keyword_id]["negative"].append(value)
                by_distance[distance]["background"].append(value)
                continue

            windows = expected_by_keyword.get(keyword_id, [])
            if windows:
                for low, high in windows:
                    value = peak(items, low, high)
                    target_scores.append(value)
                    by_keyword[keyword_id]["target"].append(value)
                    by_distance[distance]["target"].append(value)
                value = outside_peak(items, windows)
                outside_target_scores.append(value)
                by_keyword[keyword_id]["outside_target"].append(value)
                by_distance[distance]["background"].append(value)
            else:
                value = peak(items)
                cross_keyword_scores.append(value)
                by_keyword[keyword_id]["cross_keyword"].append(value)
                by_distance[distance]["background"].append(value)

    background_scores = negative_scores + cross_keyword_scores + outside_target_scores
    per_keyword: dict[str, dict] = {}
    keyword_lookup = {int(row["id"]): row for row in keywords}
    for keyword_id, groups in by_keyword.items():
        background = (
            groups["negative"] + groups["cross_keyword"] + groups["outside_target"]
        )
        threshold = float(keyword_lookup[keyword_id]["threshold"])
        targets = groups["target"]
        per_keyword[str(keyword_id)] = {
            "text": keyword_lookup[keyword_id]["text"],
            "formal_threshold": threshold,
            "target": stats(targets),
            "negative": stats(groups["negative"]),
            "cross_keyword": stats(groups["cross_keyword"]),
            "outside_target": stats(groups["outside_target"]),
            "background": stats(background),
            "target_at_or_above_formal_threshold": (
                sum(value >= threshold for value in targets) / len(targets)
                if targets
                else 0.0
            ),
            "background_at_or_above_formal_threshold": (
                sum(value >= threshold for value in background) / len(background)
                if background
                else 0.0
            ),
            "pairwise_target_over_background": pairwise_separation(
                targets, background
            ),
            "p05_target_minus_p95_background": (
                stats(targets)["p05"] - stats(background)["p95"]
            ),
        }

    distance_summary = {
        distance: {
            "target": stats(groups["target"]),
            "background": stats(groups["background"]),
            "pairwise_target_over_background": pairwise_separation(
                groups["target"], groups["background"]
            ),
            "p05_target_minus_p95_background": (
                stats(groups["target"])["p05"] - stats(groups["background"])["p95"]
            ),
        }
        for distance, groups in sorted(by_distance.items())
    }

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "development-only",
        "diagnostic_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "observation": "decoder-terminal-confidence-before-keyword-threshold",
        "normal_runtime_behavior_changed": False,
        "runner_sha256": runner_sha,
        "model_sha256": model_sha,
        "keyword_pack_sha256": pack_sha,
        "keyword_table_sha256": sha256_file(args.keyword_table),
        "references_sha256": sha256_file(args.references),
        "frames_sha256": sha256_file(args.frames),
        "recordings": len(references),
        "probe_frames": len(frame_rows),
        "pre_tolerance_ms": float(args.pre_tolerance_ms),
        "post_tolerance_ms": float(args.post_tolerance_ms),
        "overall": {
            "target": stats(target_scores),
            "negative": stats(negative_scores),
            "cross_keyword": stats(cross_keyword_scores),
            "outside_target": stats(outside_target_scores),
            "background": stats(background_scores),
            "pairwise_target_over_background": pairwise_separation(
                target_scores, background_scores
            ),
            "p05_target_minus_p95_background": (
                stats(target_scores)["p05"] - stats(background_scores)["p95"]
            ),
        },
        "by_keyword": per_keyword,
        "by_distance": distance_summary,
        "limitations": [
            "This is development-only descriptive evidence and cannot select or change a threshold.",
            "Terminal confidence is observed before keyword threshold/prefix emission policy.",
            "Synthetic speech/domain evidence is not real-human or target-device qualification.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "score separation diagnostic complete: "
        f"recordings={len(references)} frames={len(frame_rows)} "
        f"target_p50={result['overall']['target']['p50']:.6f} "
        f"background_p95={result['overall']['background']['p95']:.6f} "
        f"margin={result['overall']['p05_target_minus_p95_background']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
