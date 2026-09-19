#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select(result: dict, config: dict) -> dict:
    calibration = {
        float(row["threshold"]): row
        for row in result["calibration"]["operating_curve"]
    }
    test = {float(row["threshold"]): row for row in result["test"]["operating_curve"]}
    selected = {}
    for spec in config["operating_points"]["constraints"]:
        limit = float(spec["max_negative_fp_rate"])
        eligible = [
            row
            for row in calibration.values()
            if float(row["negative_false_positive_rate"]) <= limit
        ]
        if not eligible:
            selected[str(spec["name"])] = None
            continue
        row = max(
            eligible,
            key=lambda item: (
                float(item["wake_exact_recall"]),
                -float(item["negative_false_positive_rate"]),
                -float(item["threshold"]),
            ),
        )
        threshold = float(row["threshold"])
        selected[str(spec["name"])] = {
            "threshold": threshold,
            "test": {
                "wake_recall": float(test[threshold]["wake_exact_recall"]),
                "negative_fp_rate": float(
                    test[threshold]["negative_false_positive_rate"]
                ),
            },
        }
    return selected


def delta(target: dict | None, reference: dict | None) -> dict | None:
    if target is None or reference is None:
        return None
    return {
        "wake_recall": (
            float(target["test"]["wake_recall"])
            - float(reference["test"]["wake_recall"])
        ),
        "negative_fp_rate": (
            float(target["test"]["negative_fp_rate"])
            - float(reference["test"]["negative_fp_rate"])
        ),
    }


def trial(config: dict, seed: int, root: pathlib.Path) -> dict:
    expected_seeds = [int(value) for value in config["fixed"]["model_seeds"]]
    if seed not in expected_seeds:
        raise ValueError("unregistered model seed")
    rows = {}
    environments = []
    for name, spec in config["candidates"].items():
        result = load_json(root / name / "classifier.json")
        if int(result["model_seed"]) != seed:
            raise ValueError(f"{name}: model seed drift")
        if int(result["sampler_seed"]) != int(config["fixed"]["sampler_seed"]):
            raise ValueError(f"{name}: sampler seed drift")
        if str(result["init_mode"]) != str(config["fixed"]["init_mode"]):
            raise ValueError(f"{name}: init mode drift")
        for key in ("frontend", "lr_schedule"):
            if str(result[key]) != str(spec[key]):
                raise ValueError(f"{name}: {key} drift")
        if int(result["hidden_dim"]) != int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden drift")
        if int(result["warmup_epochs"]) != int(spec["warmup_epochs"]):
            raise ValueError(f"{name}: warmup drift")
        if abs(float(result["min_lr_ratio"]) - float(spec["min_lr_ratio"])) > 1e-15:
            raise ValueError(f"{name}: min LR ratio drift")
        environments.append(result["training_environment"])
        history = result["history"]
        rows[name] = {
            "frontend": result["frontend"],
            "hidden_dim": int(result["hidden_dim"]),
            "lr_schedule": result["lr_schedule"],
            "initial_model_state_sha256": result["initial_model_state_sha256"],
            "final_model_state_sha256": result["model_state_sha256"],
            "final_train_loss": float(history[-1]["loss"]),
            "min_train_loss": min(float(item["loss"]) for item in history),
            "first_epoch_lr": float(history[0]["learning_rate"]),
            "last_epoch_lr": float(history[-1]["learning_rate"]),
            "operating_points": select(result, config),
        }
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("paired schedule candidates did not share training environment")
    primary = str(config["decision_rules"]["primary_operating_point"])
    reference = rows["b0-logmel64-fixed"]["operating_points"].get(primary)
    fixed = rows["fc1-pcen128-fixed"]["operating_points"].get(primary)
    scheduled = rows["fc1-pcen128-warmup-cosine"]["operating_points"].get(primary)
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-optimizer-schedule-stability-trial-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "model_seed": seed,
        "sampler_seed": int(config["fixed"]["sampler_seed"]),
        "paired_same_runner": True,
        "candidates": rows,
        "fixed_target_vs_reference_primary": delta(fixed, reference),
        "scheduled_target_vs_reference_primary": delta(scheduled, reference),
    }


def summarize(values: list[dict | None]) -> dict:
    usable = [row for row in values if row is not None]
    recall = [float(row["wake_recall"]) for row in usable]
    false_positive = [float(row["negative_fp_rate"]) for row in usable]
    return {
        "usable": len(usable),
        "wake_recall_delta_mean": statistics.fmean(recall) if recall else None,
        "wake_recall_delta_stddev": (
            statistics.pstdev(recall)
            if len(recall) > 1
            else 0.0 if recall else None
        ),
        "negative_fp_delta_mean": (
            statistics.fmean(false_positive) if false_positive else None
        ),
        "negative_fp_delta_stddev": (
            statistics.pstdev(false_positive)
            if len(false_positive) > 1
            else 0.0 if false_positive else None
        ),
        "positive_wake_recall_delta_seeds": sum(value > 0.0 for value in recall),
    }


def aggregate(config: dict, root: pathlib.Path) -> dict:
    rows = {}
    for path in root.rglob("trial-evidence.json"):
        row = load_json(path)
        seed = int(row["model_seed"])
        if seed in rows:
            raise ValueError(f"duplicate seed {seed}")
        rows[seed] = row
    expected = [int(value) for value in config["fixed"]["model_seeds"]]
    if set(rows) != set(expected):
        raise ValueError(f"seed mismatch: {sorted(rows)} vs {expected}")
    ordered = [rows[seed] for seed in expected]
    fixed = summarize(
        [row["fixed_target_vs_reference_primary"] for row in ordered]
    )
    scheduled = summarize(
        [row["scheduled_target_vs_reference_primary"] for row in ordered]
    )
    source = config["source_init_stability_evidence"]
    rules = config["decision_rules"]
    tolerance = float(rules["source_baseline_tolerance"])
    baseline_reproduced = (
        abs(
            float(fixed["wake_recall_delta_mean"])
            - float(source["default_primary_recall_delta_mean"])
        )
        <= tolerance
        and abs(
            float(fixed["wake_recall_delta_stddev"])
            - float(source["default_primary_recall_delta_stddev"])
        )
        <= tolerance
        and abs(
            float(fixed["negative_fp_delta_mean"])
            - float(source["default_primary_fp_delta_mean"])
        )
        <= tolerance
    )
    ratio = (
        None
        if fixed["wake_recall_delta_stddev"] in (None, 0.0)
        else float(scheduled["wake_recall_delta_stddev"])
        / float(fixed["wake_recall_delta_stddev"])
    )
    stability_pass = (
        baseline_reproduced
        and ratio is not None
        and ratio <= float(rules["max_stddev_ratio_vs_fixed"])
        and float(scheduled["wake_recall_delta_stddev"])
        <= float(rules["max_primary_wake_recall_delta_stddev"])
        and float(scheduled["wake_recall_delta_mean"])
        >= float(rules["min_primary_wake_recall_delta_mean"])
        and float(scheduled["negative_fp_delta_mean"])
        <= float(rules["max_primary_negative_fp_delta_mean"])
        and int(scheduled["positive_wake_recall_delta_seeds"])
        >= int(rules["min_positive_primary_delta_seeds"])
    )
    if not baseline_reproduced:
        next_experiment = str(rules["baseline_drift_next"])
    elif stability_pass:
        next_experiment = str(rules["pass_next"])
    else:
        next_experiment = str(rules["fail_next"])
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-optimizer-schedule-stability-summary-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "source_init_stability_evidence": source,
        "baseline_reproduced": baseline_reproduced,
        "fixed_schedule": fixed,
        "warmup_cosine_schedule": scheduled,
        "scheduled_to_fixed_stddev_ratio": ratio,
        "stability_pass": stability_pass,
        "seed_results": {str(seed): rows[seed] for seed in expected},
        "next_experiment": next_experiment,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("trial")
    one.add_argument("--config", required=True, type=pathlib.Path)
    one.add_argument("--model-seed", required=True, type=int)
    one.add_argument("--root", required=True, type=pathlib.Path)
    one.add_argument("--output", required=True, type=pathlib.Path)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--root", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = load_json(args.config)
    result = (
        trial(config, int(args.model_seed), args.root.resolve())
        if args.mode == "trial"
        else aggregate(config, args.root.resolve())
    )
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
