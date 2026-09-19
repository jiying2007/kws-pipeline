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
            row for row in calibration.values()
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
                "negative_fp_rate": float(test[threshold]["negative_false_positive_rate"]),
            },
        }
    return selected


def delta(target: dict | None, reference: dict | None) -> dict | None:
    if target is None or reference is None:
        return None
    return {
        "wake_recall": float(target["test"]["wake_recall"]) - float(reference["test"]["wake_recall"]),
        "negative_fp_rate": float(target["test"]["negative_fp_rate"]) - float(reference["test"]["negative_fp_rate"]),
    }


def trial(config: dict, seed: int, root: pathlib.Path) -> dict:
    if seed not in [int(value) for value in config["fixed"]["model_seeds"]]:
        raise ValueError("unregistered model seed")
    rows = {}
    environments = []
    for name, spec in config["candidates"].items():
        result = load_json(root / name / "classifier.json")
        if int(result["model_seed"]) != seed:
            raise ValueError(f"{name}: model seed drift")
        if int(result["sampler_seed"]) != int(config["fixed"]["sampler_seed"]):
            raise ValueError(f"{name}: sampler seed drift")
        if str(result["frontend"]) != str(spec["frontend"]):
            raise ValueError(f"{name}: frontend drift")
        if int(result["hidden_dim"]) != int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden drift")
        if str(result["balance_mode"]) != str(spec["balance_mode"]):
            raise ValueError(f"{name}: balance mode drift")
        if str(result["lr_schedule"]) != str(config["fixed"]["lr_schedule"]):
            raise ValueError(f"{name}: LR schedule drift")
        if str(result["init_mode"]) != str(config["fixed"]["init_mode"]):
            raise ValueError(f"{name}: init mode drift")
        if spec["balance_mode"] == "exact-balanced-epoch-v2":
            per_class = result["sampler_exact_class_count_per_epoch"]
            if not isinstance(per_class, int) or per_class <= 0:
                raise ValueError(f"{name}: exact sampler count missing")
            if int(result["sampler_samples_per_epoch"]) != per_class * int(result["classes"]):
                raise ValueError(f"{name}: exact sampler is not class balanced")
        environments.append(result["research_cpu_contract"])
        rows[name] = {
            "frontend": result["frontend"],
            "hidden_dim": int(result["hidden_dim"]),
            "balance_mode": result["balance_mode"],
            "initial_model_state_sha256": result["initial_model_state_sha256"],
            "final_model_state_sha256": result["model_state_sha256"],
            "sampler_samples_per_epoch": int(result["sampler_samples_per_epoch"]),
            "sampler_exact_class_count_per_epoch": result["sampler_exact_class_count_per_epoch"],
            "final_train_loss": float(result["history"][-1]["loss"]),
            "operating_points": select(result, config),
        }
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("paired sampler candidates did not share CPU contract")
    primary = str(config["decision_rules"]["primary_operating_point"])
    weighted_ref = rows["b0-logmel64-weighted"]["operating_points"].get(primary)
    weighted_target = rows["fc1-pcen128-weighted"]["operating_points"].get(primary)
    exact_ref = rows["b0-logmel64-exact"]["operating_points"].get(primary)
    exact_target = rows["fc1-pcen128-exact"]["operating_points"].get(primary)
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-exact-balanced-sampler-trial-v2",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "model_seed": seed,
        "sampler_seed": int(config["fixed"]["sampler_seed"]),
        "paired_same_runner": True,
        "candidates": rows,
        "weighted_target_vs_reference_primary": delta(weighted_target, weighted_ref),
        "exact_target_vs_reference_primary": delta(exact_target, exact_ref),
    }


def summarize(values: list[dict | None]) -> dict:
    usable = [row for row in values if row is not None]
    recall = [float(row["wake_recall"]) for row in usable]
    fp = [float(row["negative_fp_rate"]) for row in usable]
    return {
        "usable": len(usable),
        "wake_recall_delta_mean": statistics.fmean(recall) if recall else None,
        "wake_recall_delta_stddev": statistics.pstdev(recall) if len(recall) > 1 else 0.0 if recall else None,
        "negative_fp_delta_mean": statistics.fmean(fp) if fp else None,
        "negative_fp_delta_stddev": statistics.pstdev(fp) if len(fp) > 1 else 0.0 if fp else None,
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
    weighted = summarize([row["weighted_target_vs_reference_primary"] for row in ordered])
    exact = summarize([row["exact_target_vs_reference_primary"] for row in ordered])

    sentinels = config["source_reproducibility_evidence"]["final_state_sentinels"]
    sentinel_checks = {}
    for candidate, by_seed in sentinels.items():
        for raw_seed, expected_sha in by_seed.items():
            seed = int(raw_seed)
            actual = rows[seed]["candidates"][candidate]["final_model_state_sha256"]
            sentinel_checks[f"{candidate}:{seed}"] = {
                "expected": str(expected_sha),
                "actual": str(actual),
                "match": str(actual) == str(expected_sha),
            }
    baseline_reproduced = bool(sentinel_checks) and all(
        check["match"] for check in sentinel_checks.values()
    )
    ratio = (
        None if weighted["wake_recall_delta_stddev"] in (None, 0.0)
        else float(exact["wake_recall_delta_stddev"]) / float(weighted["wake_recall_delta_stddev"])
    )
    rules = config["decision_rules"]
    sampler_pass = (
        baseline_reproduced
        and ratio is not None
        and ratio <= float(rules["max_stddev_ratio_vs_weighted"])
        and float(exact["wake_recall_delta_stddev"]) <= float(rules["max_primary_wake_recall_delta_stddev"])
        and float(exact["wake_recall_delta_mean"]) >= float(rules["min_primary_wake_recall_delta_mean"])
        and float(exact["negative_fp_delta_mean"]) <= float(rules["max_primary_negative_fp_delta_mean"])
        and int(exact["positive_wake_recall_delta_seeds"]) >= int(rules["min_positive_primary_delta_seeds"])
    )
    if not baseline_reproduced:
        next_experiment = str(rules["baseline_drift_next"])
    elif sampler_pass:
        next_experiment = str(rules["pass_next"])
    else:
        next_experiment = str(rules["fail_next"])
    result = {
        "schema_version": 2,
        "evidence_class": "kws-v2-classifier-exact-balanced-sampler-summary-v2",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "baseline_reproduced": baseline_reproduced,
        "baseline_sentinel_checks": sentinel_checks,
        "weighted_sampler": weighted,
        "exact_balanced_sampler": exact,
        "exact_to_weighted_stddev_ratio": ratio,
        "sampler_pass": sampler_pass,
        "seed_results": {str(seed): rows[seed] for seed in expected},
        "next_experiment": next_experiment,
    }
    return result


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
    if args.mode == "trial":
        result = trial(config, int(args.model_seed), args.root.resolve())
    else:
        result = aggregate(config, args.root.resolve())
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
