#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_operating_points(result: dict, config: dict) -> dict:
    cal = {float(row["threshold"]): row for row in result["calibration"]["operating_curve"]}
    test = {float(row["threshold"]): row for row in result["test"]["operating_curve"]}
    output: dict[str, dict | None] = {}
    for spec in config["operating_points"]["constraints"]:
        limit = float(spec["max_negative_fp_rate"])
        eligible = [
            row for row in cal.values()
            if float(row["negative_false_positive_rate"]) <= limit
        ]
        if not eligible:
            output[str(spec["name"])] = None
            continue
        selected = max(
            eligible,
            key=lambda row: (
                float(row["wake_exact_recall"]),
                -float(row["negative_false_positive_rate"]),
                -float(row["threshold"]),
            ),
        )
        threshold = float(selected["threshold"])
        output[str(spec["name"])] = {
            "threshold": threshold,
            "test": {
                "wake_recall": float(test[threshold]["wake_exact_recall"]),
                "negative_fp_rate": float(test[threshold]["negative_false_positive_rate"]),
            },
        }
    return output


def delta(target: dict | None, reference: dict | None) -> dict | None:
    if target is None or reference is None:
        return None
    return {
        "wake_recall": float(target["test"]["wake_recall"]) - float(reference["test"]["wake_recall"]),
        "negative_fp_rate": float(target["test"]["negative_fp_rate"]) - float(reference["test"]["negative_fp_rate"]),
    }


def build_trial(config: dict, axis: str, variable_seed: int, root: pathlib.Path) -> dict:
    axis_cfg = config["axes"][axis]
    if variable_seed not in [int(value) for value in axis_cfg["values"]]:
        raise ValueError("trial seed is not pre-registered")
    if axis == "model-init":
        expected_model_seed = variable_seed
        expected_sampler_seed = int(axis_cfg["fixed_sampler_seed"])
    elif axis == "sampler-path":
        expected_model_seed = int(axis_cfg["fixed_model_seed"])
        expected_sampler_seed = variable_seed
    else:
        raise ValueError(f"unknown axis: {axis}")

    rows: dict[str, dict] = {}
    environments: list[dict] = []
    for name, spec in config["candidates"].items():
        result = load_json(root / name / "classifier.json")
        if int(result.get("model_seed", -1)) != expected_model_seed:
            raise ValueError(f"{name}: model seed drift")
        if int(result.get("sampler_seed", -1)) != expected_sampler_seed:
            raise ValueError(f"{name}: sampler seed drift")
        if str(result.get("seed_policy")) != "independent-model-sampler-v1":
            raise ValueError(f"{name}: seed policy drift")
        if str(result["frontend"]) != str(spec["frontend"]):
            raise ValueError(f"{name}: frontend drift")
        if int(result["hidden_dim"]) != int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden dim drift")
        environments.append(result["training_environment"])
        history = result["history"]
        rows[name] = {
            "frontend": result["frontend"],
            "hidden_dim": int(result["hidden_dim"]),
            "model_seed": int(result["model_seed"]),
            "sampler_seed": int(result["sampler_seed"]),
            "initial_model_state_sha256": result["initial_model_state_sha256"],
            "final_model_state_sha256": result["model_state_sha256"],
            "final_train_loss": float(history[-1]["loss"]),
            "min_train_loss": min(float(row["loss"]) for row in history),
            "operating_points": select_operating_points(result, config),
            "test_argmax": {
                "wake_recall": float(result["test"]["positive_recall"]),
                "negative_fp_rate": float(result["test"]["negative_false_positive_clip_rate"]),
            },
        }
    if any(value != environments[0] for value in environments[1:]):
        raise ValueError("paired candidates did not share one training environment")
    rules = config["decision_rules"]
    primary = str(rules["primary_operating_point"])
    reference = rows["b0-logmel64"]
    target = rows["fc1-pcen128"]
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-random-source-trial-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "axis": axis,
        "variable_seed": variable_seed,
        "model_seed": expected_model_seed,
        "sampler_seed": expected_sampler_seed,
        "paired_same_runner": True,
        "training_environment_consistent": True,
        "candidates": rows,
        "target_vs_reference_primary": delta(
            target["operating_points"].get(primary),
            reference["operating_points"].get(primary),
        ),
    }


def axis_summary(rows: list[dict], axis: str, candidate: str) -> dict:
    selected = [row for row in rows if row["axis"] == axis]
    deltas = [row["target_vs_reference_primary"] for row in selected]
    usable_recall = [float(row["wake_recall"]) for row in deltas if row is not None]
    usable_fp = [float(row["negative_fp_rate"]) for row in deltas if row is not None]
    losses = [float(row["candidates"][candidate]["final_train_loss"]) for row in selected]
    initial_hashes = [
        str(row["candidates"][candidate]["initial_model_state_sha256"])
        for row in selected
    ]
    return {
        "trials": len(selected),
        "usable_primary_points": len(usable_recall),
        "primary_wake_recall_delta_mean": statistics.fmean(usable_recall) if usable_recall else None,
        "primary_wake_recall_delta_stddev": statistics.pstdev(usable_recall) if len(usable_recall) > 1 else 0.0 if usable_recall else None,
        "primary_negative_fp_delta_mean": statistics.fmean(usable_fp) if usable_fp else None,
        "primary_negative_fp_delta_stddev": statistics.pstdev(usable_fp) if len(usable_fp) > 1 else 0.0 if usable_fp else None,
        "target_final_train_loss_mean": statistics.fmean(losses),
        "target_final_train_loss_stddev": statistics.pstdev(losses),
        "target_initial_state_unique": len(set(initial_hashes)),
        "rows": {
            str(row["variable_seed"]): {
                "primary_delta": row["target_vs_reference_primary"],
                "reference": row["candidates"]["b0-logmel64"],
                "target": row["candidates"]["fc1-pcen128"],
            }
            for row in selected
        },
    }


def aggregate(config: dict, root: pathlib.Path) -> dict:
    trials: list[dict] = []
    for path in root.rglob("trial-evidence.json"):
        trials.append(load_json(path))
    expected = {
        (axis, int(seed))
        for axis, row in config["axes"].items()
        for seed in row["values"]
    }
    actual = {(str(row["axis"]), int(row["variable_seed"])) for row in trials}
    if actual != expected:
        raise ValueError(f"trial evidence mismatch: expected {sorted(expected)}, got {sorted(actual)}")

    init = axis_summary(trials, "model-init", "fc1-pcen128")
    sampler = axis_summary(trials, "sampler-path", "fc1-pcen128")
    if int(sampler["target_initial_state_unique"]) != 1:
        raise ValueError("sampler-path axis changed target initialization")
    if int(init["target_initial_state_unique"]) != len(config["axes"]["model-init"]["values"]):
        raise ValueError("model-init axis did not produce unique target initializations")

    rules = config["decision_rules"]
    init_std = init["primary_wake_recall_delta_stddev"]
    sampler_std = sampler["primary_wake_recall_delta_stddev"]
    floor = float(rules["material_wake_recall_stddev"])
    ratio = float(rules["dominance_ratio"])
    if init_std is None or sampler_std is None:
        classification = "mixed-or-threshold-instability"
        next_experiment = str(rules["mixed_next"])
    elif max(init_std, sampler_std) < floor:
        classification = "low-random-source-variance"
        next_experiment = str(rules["low_variance_next"])
    elif init_std >= ratio * max(sampler_std, 1e-12):
        classification = "model-initialization-dominant"
        next_experiment = str(rules["initialization_dominant_next"])
    elif sampler_std >= ratio * max(init_std, 1e-12):
        classification = "sampler-path-dominant"
        next_experiment = str(rules["sampler_dominant_next"])
    else:
        classification = "mixed-random-source-variance"
        next_experiment = str(rules["mixed_next"])

    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-seed-variance-summary-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "source_replication_evidence": config["source_replication_evidence"],
        "model_init_axis": init,
        "sampler_path_axis": sampler,
        "variance_classification": classification,
        "next_experiment": next_experiment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose KWS classifier model-init and sampler variance.")
    sub = parser.add_subparsers(dest="mode", required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--config", required=True, type=pathlib.Path)
    trial.add_argument("--axis", required=True, choices=("model-init", "sampler-path"))
    trial.add_argument("--variable-seed", required=True, type=int)
    trial.add_argument("--root", required=True, type=pathlib.Path)
    trial.add_argument("--output", required=True, type=pathlib.Path)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--config", required=True, type=pathlib.Path)
    agg.add_argument("--root", required=True, type=pathlib.Path)
    agg.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = load_json(args.config)
    result = (
        build_trial(config, args.axis, int(args.variable_seed), args.root.resolve())
        if args.mode == "trial"
        else aggregate(config, args.root.resolve())
    )
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
