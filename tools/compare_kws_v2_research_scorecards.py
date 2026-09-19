#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib

SCORECARD_CLASS = "kws-v2-research-baseline-scorecard-v1"
LOSS_KEYS = (
    "ordered_token_loss_weight",
    "keyword_sequence_margin_loss_weight",
    "prefix_completion_loss_weight",
    "recurrent_release_loss_weight",
)
IDENTITY_KEYS = (
    "family",
    "frontend",
    "feature_dim",
    "hidden_dim",
    "config_sha256",
    "training_seed",
    "training_epochs",
)


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def nested(row: dict, *keys: str) -> object | None:
    value: object = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def metric(row: dict, *keys: str) -> float | None:
    value = nested(row, *keys)
    if value is None:
        return None
    return finite(value, ".".join(keys))


def delta(before: float | None, after: float | None) -> dict | None:
    if before is None or after is None:
        return None
    return {
        "baseline": before,
        "candidate": after,
        "delta": after - before,
    }


def verify_scorecard(path: pathlib.Path, row: dict) -> None:
    if row.get("evidence_class") != SCORECARD_CLASS:
        raise ValueError(f"{path}: scorecard identity mismatch")
    if row.get("evidence_scope") != "research-only":
        raise ValueError(f"{path}: expected research-only scorecard")
    if row.get("protected_evidence_used") is not False:
        raise ValueError(f"{path}: protected evidence was consumed")
    if row.get("promotion_allowed") is not False:
        raise ValueError(f"{path}: research scorecard unexpectedly permits promotion")


def identity(row: dict) -> dict:
    sidecar = row.get("ordinary_speech_negative_sidecar")
    balance = row.get("class_balance")
    if not isinstance(sidecar, dict) or sidecar.get("enabled") is not True:
        raise ValueError("scorecard must use the frozen ordinary-speech sidecar")
    if not isinstance(balance, dict):
        raise ValueError("scorecard is missing class-balance evidence")
    result = {key: row.get(key) for key in IDENTITY_KEYS}
    result["sidecar_receipt_sha256"] = sidecar.get("receipt_sha256")
    result["sidecar_total_recordings"] = sidecar.get("total_recordings")
    result["class_balance"] = balance
    return result


def loss_vector(row: dict) -> dict[str, float]:
    loss = row.get("loss_weights")
    if not isinstance(loss, dict):
        raise ValueError("scorecard is missing loss_weights")
    return {key: finite(loss.get(key), f"loss_weights.{key}") for key in LOSS_KEYS}


def distribution_deltas(before: dict, after: dict, split: str) -> dict:
    prefix = ("float_ctc_confidence", "splits", split)
    return {
        "wake_p10": delta(
            metric(before, *prefix, "positive_true_keyword_confidence", "p10"),
            metric(after, *prefix, "positive_true_keyword_confidence", "p10"),
        ),
        "nonwake_p99": delta(
            metric(before, *prefix, "nonwake_max_keyword_confidence", "p99"),
            metric(after, *prefix, "nonwake_max_keyword_confidence", "p99"),
        ),
        "tokenized_nonwake_p99": delta(
            metric(before, *prefix, "tokenized_nonwake_max_keyword_confidence", "p99"),
            metric(after, *prefix, "tokenized_nonwake_max_keyword_confidence", "p99"),
        ),
        "empty_target_nonwake_p99": delta(
            metric(before, *prefix, "empty_target_nonwake_max_keyword_confidence", "p99"),
            metric(after, *prefix, "empty_target_nonwake_max_keyword_confidence", "p99"),
        ),
        "separation_gap": delta(
            metric(before, *prefix, "separation_gap_p10_wake_minus_p99_nonwake"),
            metric(after, *prefix, "separation_gap_p10_wake_minus_p99_nonwake"),
        ),
        "wake_blank_top1_fraction": delta(
            metric(before, *prefix, "wake_top1_frame_diagnostics", "blank_top1_fraction"),
            metric(after, *prefix, "wake_top1_frame_diagnostics", "blank_top1_fraction"),
        ),
        "wake_keyword_root_top1_fraction": delta(
            metric(before, *prefix, "wake_top1_frame_diagnostics", "keyword_root_top1_fraction"),
            metric(after, *prefix, "wake_top1_frame_diagnostics", "keyword_root_top1_fraction"),
        ),
    }


def runtime_deltas(before: dict, after: dict) -> dict:
    left = before.get("soft_operating_points_by_far_budget")
    right = after.get("soft_operating_points_by_far_budget")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("scorecards are missing soft operating points")
    result: dict[str, dict] = {}
    for budget in sorted(set(left) & set(right), key=float):
        a = left[budget]
        b = right[budget]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        result[budget] = {
            "threshold": delta(
                finite(a["threshold"], f"{budget}.baseline.threshold"),
                finite(b["threshold"], f"{budget}.candidate.threshold"),
            ),
            "calibration_frr": delta(
                finite(a["calibration"]["frr"], f"{budget}.baseline.calibration.frr"),
                finite(b["calibration"]["frr"], f"{budget}.candidate.calibration.frr"),
            ),
            "calibration_far_per_hour": delta(
                finite(a["calibration"]["far_per_hour"], f"{budget}.baseline.calibration.far"),
                finite(b["calibration"]["far_per_hour"], f"{budget}.candidate.calibration.far"),
            ),
            "test_frr": delta(
                finite(a["test"]["frr"], f"{budget}.baseline.test.frr"),
                finite(b["test"]["frr"], f"{budget}.candidate.test.frr"),
            ),
            "test_far_per_hour": delta(
                finite(a["test"]["far_per_hour"], f"{budget}.baseline.test.far"),
                finite(b["test"]["far_per_hour"], f"{budget}.candidate.test.far"),
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two KWS Training Reset scorecards without granting promotion authority."
    )
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--expect-single-variable", action="store_true")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    baseline = load_object(args.baseline.resolve())
    candidate = load_object(args.candidate.resolve())
    verify_scorecard(args.baseline, baseline)
    verify_scorecard(args.candidate, candidate)

    baseline_identity = identity(baseline)
    candidate_identity = identity(candidate)
    if baseline_identity != candidate_identity:
        raise ValueError("research scorecard identity drift prevents causal comparison")

    baseline_loss = loss_vector(baseline)
    candidate_loss = loss_vector(candidate)
    changed = [
        key for key in LOSS_KEYS if baseline_loss[key] != candidate_loss[key]
    ]
    if args.expect_single_variable and len(changed) != 1:
        raise ValueError(f"expected exactly one loss-variable delta, observed {changed}")

    classifier_keys = (
        ("test", "positive_recall"),
        ("test", "negative_false_positive_clip_rate"),
        ("test", "score_distribution", "separation_gap_p10_positive_minus_p99_negative"),
    )
    classifier_control = {}
    for keys in classifier_keys:
        name = ".".join(keys)
        classifier_control[name] = delta(
            metric(baseline, "classifier_baseline", *keys),
            metric(candidate, "classifier_baseline", *keys),
        )

    report = {
        "schema_version": 1,
        "evidence_class": "kws-v2-research-scorecard-comparison-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "baseline_profile": baseline.get("loss_profile"),
        "candidate_profile": candidate.get("loss_profile"),
        "identity": baseline_identity,
        "baseline_loss_weights": baseline_loss,
        "candidate_loss_weights": candidate_loss,
        "changed_loss_variables": changed,
        "float_ctc": {
            split: distribution_deltas(baseline, candidate, split)
            for split in ("calibration", "test")
        },
        "runtime_by_far_budget": runtime_deltas(baseline, candidate),
        "continuous_negative_far_per_hour": delta(
            metric(baseline, "research_negative_exposure", "far_per_hour"),
            metric(candidate, "research_negative_exposure", "far_per_hour"),
        ),
        "classifier_control": classifier_control,
    }
    for split, values in report["float_ctc"].items():
        gap = values["separation_gap"]
        if gap is not None:
            values["separation_gap_moved_toward_separation"] = gap["delta"] > 0.0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"kws-research-compare: {report['baseline_profile']} -> "
        f"{report['candidate_profile']} changed={changed}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
