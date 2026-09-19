#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

EVIDENCE_CLASS = "kws-v2-research-confirm-report-v1"
SCORECARD_CLASS = "kws-v2-research-baseline-scorecard-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a predeclared KWS Training Reset model-seed confirmation cohort."
    )
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--scorecard", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    policy = load_object(args.policy.resolve())
    confirm = policy.get("confirm")
    if not isinstance(confirm, dict):
        raise ValueError("research policy has no confirm section")
    expected_seeds = [int(value) for value in confirm.get("training_seeds", [])]
    if len(expected_seeds) < int(confirm.get("minimum_seed_count", 0)):
        raise ValueError("predeclared confirm seed plan is incomplete")
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("predeclared confirm seeds must be unique")
    if confirm.get("same_acoustic_render_required") is not True:
        raise ValueError("confirm must require the same acoustic render")
    if confirm.get("same_corpus_identity_required") is not True:
        raise ValueError("confirm must require the same corpus identity")
    if confirm.get("same_loss_profile_required") is not True:
        raise ValueError("confirm must require the same loss profile")
    if confirm.get("protected_evidence_used") is not False:
        raise ValueError("confirm cannot consume protected evidence")
    if confirm.get("promotion_allowed") is not False:
        raise ValueError("confirm cannot promote a candidate")

    rows: list[dict] = []
    by_seed: dict[int, dict] = {}
    for path in args.scorecard:
        value = load_object(path.resolve())
        if value.get("evidence_class") != SCORECARD_CLASS:
            raise ValueError(f"{path}: research scorecard identity mismatch")
        if value.get("evidence_scope") != "research-only":
            raise ValueError(f"{path}: scorecard is not research-only")
        if value.get("protected_evidence_used") is not False:
            raise ValueError(f"{path}: protected evidence was used")
        if value.get("promotion_allowed") is not False:
            raise ValueError(f"{path}: scorecard unexpectedly permits promotion")
        seed = value.get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path}: scorecard lacks explicit training_seed")
        if seed in by_seed:
            raise ValueError(f"duplicate confirm training seed: {seed}")
        by_seed[seed] = value
        rows.append(value)

    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError(
            f"confirm cohort does not match predeclared seeds: "
            f"expected={expected_seeds} observed={sorted(by_seed)}"
        )

    identity_fields = (
        "family",
        "frontend",
        "feature_dim",
        "hidden_dim",
        "loss_profile",
        "config_sha256",
    )
    identity = {key: rows[0].get(key) for key in identity_fields}
    sidecar = rows[0].get("ordinary_speech_negative_sidecar")
    balance = rows[0].get("class_balance")
    if not isinstance(sidecar, dict) or sidecar.get("enabled") is not True:
        raise ValueError("confirm requires the ordinary-speech negative sidecar")
    if not isinstance(balance, dict):
        raise ValueError("confirm requires explicit class-balance evidence")

    for row in rows[1:]:
        for key, expected in identity.items():
            if row.get(key) != expected:
                raise ValueError(f"confirm identity drifted at {key}")
        if row.get("ordinary_speech_negative_sidecar") != sidecar:
            raise ValueError("confirm sidecar identity/counts drifted across seeds")
        if row.get("class_balance") != balance:
            raise ValueError("confirm class-balance policy/counts drifted across seeds")
        if row.get("single_acoustic_render") is not True:
            raise ValueError("confirm scorecard does not declare fixed acoustic render")

    budget_keys = sorted(
        rows[0]["soft_operating_points_by_far_budget"],
        key=lambda value: float(value),
    )
    operating: dict[str, dict] = {}
    for budget in budget_keys:
        points = [row["soft_operating_points_by_far_budget"].get(budget) for row in rows]
        if any(point is None for point in points):
            operating[budget] = {"complete": False}
            continue
        calibration_frr = [
            finite(point["calibration"]["frr"], f"{budget}.calibration.frr")
            for point in points
        ]
        calibration_far = [
            finite(point["calibration"]["far_per_hour"], f"{budget}.calibration.far")
            for point in points
        ]
        test_frr = [
            finite(point["test"]["frr"], f"{budget}.test.frr")
            for point in points
        ]
        test_far = [
            finite(point["test"]["far_per_hour"], f"{budget}.test.far")
            for point in points
        ]
        thresholds = [finite(point["threshold"], f"{budget}.threshold") for point in points]
        operating[budget] = {
            "complete": True,
            "threshold": stats(thresholds),
            "calibration_frr": stats(calibration_frr),
            "calibration_far_per_hour": stats(calibration_far),
            "test_frr": stats(test_frr),
            "test_far_per_hour": stats(test_far),
        }

    classifier = {
        "test_positive_recall": stats(
            [
                finite(row["classifier_baseline"]["test"]["positive_recall"], "classifier recall")
                for row in rows
            ]
        ),
        "test_negative_false_positive_clip_rate": stats(
            [
                finite(
                    row["classifier_baseline"]["test"]["negative_false_positive_clip_rate"],
                    "classifier negative FP",
                )
                for row in rows
            ]
        ),
        "test_separation_gap": stats(
            [
                finite(
                    row["classifier_baseline"]["test"]["score_distribution"][
                        "separation_gap_p10_positive_minus_p99_negative"
                    ],
                    "classifier separation gap",
                )
                for row in rows
            ]
        ),
    }

    negative_far = stats(
        [
            finite(row["research_negative_exposure"]["far_per_hour"], "negative FAR")
            for row in rows
        ]
    )

    report = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-confirm-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "promotion_allowed": False,
        "predeclared_training_seeds": expected_seeds,
        "completed_seed_count": len(rows),
        "same_acoustic_render_required": True,
        "same_corpus_identity_required": True,
        "same_loss_profile_required": True,
        "identity": identity,
        "ordinary_speech_negative_sidecar": sidecar,
        "class_balance": balance,
        "operating_points": operating,
        "classifier": classifier,
        "research_negative_far_per_hour": negative_far,
        "per_seed": [
            {
                "training_seed": int(row["training_seed"]),
                "model_sha256": str(row["model_sha256"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "pareto_thresholds": row["pareto_thresholds"],
            }
            for row in sorted(rows, key=lambda item: int(item["training_seed"]))
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"kws-research-confirm: seeds={len(rows)} "
        f"family={identity['family']} frontend={identity['frontend']} "
        f"profile={identity['loss_profile']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
