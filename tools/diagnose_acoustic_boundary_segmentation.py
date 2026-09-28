#!/usr/bin/env python3
"""Test train-only acoustic boundary segmentation feasibility across gap lengths.

This study follows the scalar separability failure. It remains development-only
and feedback-allowed, uses only the governed train split, and never changes the
shipping decoder/model. A fixed nearest-centroid linear classifier is evaluated
with leave-one-voice-out and leave-one-gap-out folds so a one-frame VAD/alignment
artifact at one gap duration cannot masquerade as a stable boundary signal.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from diagnose_boundary_separability import (  # noqa: E402
    FEATURES,
    build as build_gap_separability,
    write_json,
)

POLICY = "acoustic-boundary-segmentation-feasibility-v1"
EVIDENCE_CLASS = "acoustic-boundary-segmentation-development-v1"
DEFAULT_GAPS_MS = (160, 240, 320, 400, 480)

POSTERIOR_ONLY_FEATURES = tuple(
    feature for feature in FEATURES if feature != "gap_speech_active_ratio"
)
WITH_SPEECH_GATE_FEATURES = FEATURES
FEATURE_SETS = {
    "posterior_only": POSTERIOR_ONLY_FEATURES,
    "posterior_plus_speech_gate": WITH_SPEECH_GATE_FEATURES,
}


def sample_rows(gap_result: dict, gap_ms: int) -> list[dict]:
    rows: list[dict] = []
    for record in gap_result["records"]:
        common = {
            "voice_id": str(record["voice_id"]),
            "keyword_id": int(record["keyword_id"]),
            "gap_ms": int(gap_ms),
        }
        rows.append(
            {
                **common,
                "label": True,
                "features": dict(record["positive"]["features"]),
            }
        )
        rows.append(
            {
                **common,
                "label": False,
                "features": dict(record["negative"]["features"]),
            }
        )
    return rows


def finite_feature(row: dict, feature: str) -> float:
    value = float(row["features"][feature])
    if not math.isfinite(value):
        raise ValueError(f"non-finite feature: {feature}")
    return value


def fit_centroid(samples: list[dict], features: tuple[str, ...]) -> dict:
    if not samples or {bool(row["label"]) for row in samples} != {False, True}:
        raise ValueError("centroid fit requires both classes")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in features:
        values = [finite_feature(row, feature) for row in samples]
        means[feature] = statistics.fmean(values)
        scale = statistics.pstdev(values)
        scales[feature] = scale if scale > 1.0e-12 else 1.0

    def z(row: dict) -> list[float]:
        return [
            (finite_feature(row, feature) - means[feature]) / scales[feature]
            for feature in features
        ]

    positive = [z(row) for row in samples if bool(row["label"])]
    negative = [z(row) for row in samples if not bool(row["label"])]
    pos_centroid = [
        statistics.fmean(values) for values in zip(*positive)
    ]
    neg_centroid = [
        statistics.fmean(values) for values in zip(*negative)
    ]
    weight = [p - n for p, n in zip(pos_centroid, neg_centroid)]
    if sum(value * value for value in weight) <= 1.0e-18:
        raise ValueError("class centroids are indistinguishable")
    bias = -0.5 * (
        sum(value * value for value in pos_centroid)
        - sum(value * value for value in neg_centroid)
    )
    return {
        "features": list(features),
        "means": means,
        "scales": scales,
        "weight": weight,
        "bias": bias,
    }


def predict(rule: dict, row: dict) -> bool:
    score = float(rule["bias"])
    for feature, weight in zip(rule["features"], rule["weight"]):
        score += float(weight) * (
            finite_feature(row, feature) - float(rule["means"][feature])
        ) / float(rule["scales"][feature])
    return score >= 0.0


def accuracy(rule: dict, rows: list[dict]) -> tuple[int, int]:
    correct = sum(predict(rule, row) is bool(row["label"]) for row in rows)
    return correct, len(rows)


def cross_validate(
    samples: list[dict],
    *,
    features: tuple[str, ...],
    group_key: str,
) -> dict:
    groups = sorted({row[group_key] for row in samples})
    folds: list[dict] = []
    total_correct = 0
    total = 0
    for group in groups:
        train = [row for row in samples if row[group_key] != group]
        held = [row for row in samples if row[group_key] == group]
        rule = fit_centroid(train, features)
        correct, count = accuracy(rule, held)
        total_correct += correct
        total += count
        folds.append(
            {
                "held_out": group,
                "correct": correct,
                "total": count,
                "accuracy": correct / count,
            }
        )
    return {
        "group_key": group_key,
        "folds": folds,
        "correct": total_correct,
        "total": total,
        "accuracy": total_correct / total,
        "min_fold_accuracy": min(fold["accuracy"] for fold in folds),
    }


def evaluate_feature_set(samples: list[dict], features: tuple[str, ...]) -> dict:
    rule = fit_centroid(samples, features)
    train_correct, train_total = accuracy(rule, samples)
    voice = cross_validate(samples, features=features, group_key="voice_id")
    gap = cross_validate(samples, features=features, group_key="gap_ms")
    return {
        "features": list(features),
        "classifier": "zscore-nearest-centroid-linear-v1",
        "train": {
            "correct": train_correct,
            "total": train_total,
            "accuracy": train_correct / train_total,
        },
        "leave_one_voice_out": voice,
        "leave_one_gap_out": gap,
        "stable": (
            train_correct == train_total
            and voice["correct"] == voice["total"]
            and gap["correct"] == gap["total"]
        ),
    }


def build(args: argparse.Namespace) -> dict:
    gaps = tuple(sorted(set(int(value) for value in args.gap_ms)))
    if len(gaps) < 3 or any(value <= 0 or value % 20 for value in gaps):
        raise ValueError("gap set must contain >=3 unique positive 20-ms multiples")

    work = args.output_dir.resolve()
    cache = args.posterior_cache.resolve()
    all_samples: list[dict] = []
    gap_reports: dict[str, dict] = {}
    for gap_ms in gaps:
        gap_root = work / f"gap-{gap_ms}ms"
        result = build_gap_separability(
            SimpleNamespace(
                dataset_index=args.dataset_index,
                model=args.model,
                posterior_dump=args.posterior_dump,
                posterior_cache=cache,
                output_dir=gap_root,
                gap_ms=gap_ms,
                lead_ms=args.lead_ms,
                tail_ms=args.tail_ms,
            )
        )
        if (
            result.get("source_split") != "train"
            or result.get("selection_feedback_allowed") is not True
            or result.get("protected_evidence_used") is not False
        ):
            raise ValueError("gap diagnostic authority mismatch")
        rows = sample_rows(result, gap_ms)
        all_samples.extend(rows)
        gap_reports[str(gap_ms)] = {
            "pairs": int(result["pairs"]),
            "voices": int(result["voices"]),
            "stable_scalar_features": list(result["stable_scalar_features"]),
            "simple_scalar_separability_observed": bool(
                result["simple_scalar_separability_observed"]
            ),
        }

    feature_sets = {
        name: evaluate_feature_set(all_samples, features)
        for name, features in FEATURE_SETS.items()
    }
    stable_sets = [
        name for name, report in feature_sets.items() if report["stable"] is True
    ]
    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": True,
        "protected_evidence_used": False,
        "release_authority": False,
        "source_split": "train",
        "gap_ms": list(gaps),
        "samples": len(all_samples),
        "voices": len({row["voice_id"] for row in all_samples}),
        "keywords": len({row["keyword_id"] for row in all_samples}),
        "gap_reports": gap_reports,
        "feature_sets": feature_sets,
        "stable_feature_sets": stable_sets,
        "segmentation_signal_observed": bool(stable_sets),
    }
    write_json(work / "segmentation-feasibility.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate train-only acoustic boundary segmentation feasibility."
    )
    parser.add_argument("--dataset-index", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gap-ms", type=int, nargs="+", default=list(DEFAULT_GAPS_MS))
    parser.add_argument("--lead-ms", type=int, default=1000)
    parser.add_argument("--tail-ms", type=int, default=1000)
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                "samples": result["samples"],
                "voices": result["voices"],
                "gap_ms": result["gap_ms"],
                "stable_feature_sets": result["stable_feature_sets"],
                "segmentation_signal_observed": result["segmentation_signal_observed"],
                "selection_feedback_allowed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
