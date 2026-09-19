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
    cal_curve = {
        float(item["threshold"]): item
        for item in result["calibration"]["operating_curve"]
    }
    test_curve = {
        float(item["threshold"]): item
        for item in result["test"]["operating_curve"]
    }
    selected: dict[str, dict | None] = {}
    for spec in config["operating_points"]["constraints"]:
        name = str(spec["name"])
        limit = float(spec["max_negative_fp_rate"])
        eligible = [
            item
            for item in cal_curve.values()
            if float(item["negative_false_positive_rate"]) <= limit
        ]
        if not eligible:
            selected[name] = None
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
        test = test_curve[threshold]
        selected[name] = {
            "threshold": threshold,
            "calibration": {
                "wake_recall": float(row["wake_exact_recall"]),
                "negative_fp_rate": float(row["negative_false_positive_rate"]),
            },
            "test": {
                "wake_recall": float(test["wake_exact_recall"]),
                "negative_fp_rate": float(test["negative_false_positive_rate"]),
            },
        }
    return selected


def delta(left: dict | None, right: dict | None) -> dict | None:
    if left is None or right is None:
        return None
    return {
        "wake_recall": float(left["test"]["wake_recall"]) - float(right["test"]["wake_recall"]),
        "negative_fp_rate": float(left["test"]["negative_fp_rate"]) - float(right["test"]["negative_fp_rate"]),
    }


def directional_pass(value: dict | None, rules: dict, prefix: str) -> bool:
    if value is None:
        return False
    return (
        float(value["wake_recall"]) >= float(rules[f"{prefix}_min_wake_recall_delta"])
        and float(value["negative_fp_rate"]) <= float(rules[f"{prefix}_max_negative_fp_rate_delta"])
    )


def absolute_strong(row: dict, rules: dict) -> bool:
    primary = row["operating_points"].get(rules["primary_operating_point"])
    secondary = row["operating_points"].get(rules["secondary_operating_point"])
    if primary is None or secondary is None:
        return False
    return (
        float(primary["test"]["wake_recall"]) >= float(rules["strong_primary_min_wake_recall"])
        and float(primary["test"]["negative_fp_rate"]) <= float(rules["strong_primary_max_negative_fp_rate"])
        and float(secondary["test"]["wake_recall"]) >= float(rules["strong_secondary_min_wake_recall"])
        and float(secondary["test"]["negative_fp_rate"]) <= float(rules["strong_secondary_max_negative_fp_rate"])
    )


def seed_evidence(config: dict, seed: int, root: pathlib.Path) -> dict:
    candidates = config["candidates"]
    rows: dict[str, dict] = {}
    environments: list[dict] = []
    for name, spec in candidates.items():
        result = load_json(root / name / "classifier.json")
        if int(result["seed"]) != seed:
            raise ValueError(f"{name}: seed drift")
        if str(result.get("recurrent_impl")) != "tiny-streaming-gru-cell-v1":
            raise ValueError(f"{name}: recurrent implementation drift")
        if str(result["frontend"]) != str(spec["frontend"]):
            raise ValueError(f"{name}: frontend drift")
        if int(result["hidden_dim"]) != int(spec["hidden_dim"]):
            raise ValueError(f"{name}: hidden-dim drift")
        if int(result["feature_dim"]) != int(config["fixed"]["feature_dim"]):
            raise ValueError(f"{name}: feature-dim drift")
        if int(result["epochs"]) != int(config["fixed"]["epochs"]):
            raise ValueError(f"{name}: epoch drift")
        if str(result["balance_mode"]) != str(config["fixed"]["balance_mode"]):
            raise ValueError(f"{name}: balance-mode drift")
        environment = result.get("training_environment")
        if not isinstance(environment, dict):
            raise ValueError(f"{name}: missing training environment")
        environments.append(environment)
        rows[name] = {
            "frontend": str(result["frontend"]),
            "hidden_dim": int(result["hidden_dim"]),
            "trainable_parameters": int(result["trainable_parameters"]),
            "model_state_sha256": str(result["model_state_sha256"]),
            "operating_points": select_operating_points(result, config),
            "test_argmax": {
                "wake_recall": float(result["test"]["positive_recall"]),
                "negative_fp_rate": float(result["test"]["negative_false_positive_clip_rate"]),
                "gap": float(
                    result["test"]["score_distribution"][
                        "separation_gap_p10_positive_minus_p99_negative"
                    ]
                ),
            },
        }
    environment_consistent = all(value == environments[0] for value in environments[1:])
    if not environment_consistent:
        raise ValueError(f"seed {seed}: paired candidates did not share one training environment")

    rules = config["replication_rules"]
    reference = str(rules["reference_candidate"])
    target = str(rules["target_candidate"])
    control = str(rules["mechanism_control_candidate"])
    primary_name = str(rules["primary_operating_point"])
    secondary_name = str(rules["secondary_operating_point"])

    primary_delta = delta(
        rows[target]["operating_points"].get(primary_name),
        rows[reference]["operating_points"].get(primary_name),
    )
    secondary_delta = delta(
        rows[target]["operating_points"].get(secondary_name),
        rows[reference]["operating_points"].get(secondary_name),
    )
    control_primary_delta = delta(
        rows[target]["operating_points"].get(primary_name),
        rows[control]["operating_points"].get(primary_name),
    )
    control_secondary_delta = delta(
        rows[target]["operating_points"].get(secondary_name),
        rows[control]["operating_points"].get(secondary_name),
    )

    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-encoder-frontend-paired-seed-evidence-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "seed": seed,
        "paired_same_runner": True,
        "training_environment_consistent": True,
        "reference_candidate": reference,
        "target_candidate": target,
        "mechanism_control_candidate": control,
        "candidates": rows,
        "target_vs_reference": {
            "primary": primary_delta,
            "secondary": secondary_delta,
            "primary_directional_pass": directional_pass(primary_delta, rules, "primary"),
            "secondary_directional_pass": directional_pass(secondary_delta, rules, "secondary"),
            "absolute_strong": absolute_strong(rows[target], rules),
        },
        "target_vs_mechanism_control": {
            "primary": control_primary_delta,
            "secondary": control_secondary_delta,
        },
    }


def median_field(values: list[dict | None], field: str) -> float | None:
    usable = [float(value[field]) for value in values if value is not None]
    return statistics.median(usable) if usable else None


def aggregate_evidence(config: dict, root: pathlib.Path) -> dict:
    expected_seeds = [int(value) for value in config["seeds"]]
    rows: dict[int, dict] = {}
    for path in root.rglob("seed-evidence.json"):
        row = load_json(path)
        seed = int(row["seed"])
        if seed in rows:
            raise ValueError(f"duplicate seed evidence: {seed}")
        rows[seed] = row
    if set(rows) != set(expected_seeds):
        raise ValueError(
            f"seed evidence mismatch: expected {expected_seeds}, got {sorted(rows)}"
        )

    ordered = [rows[seed] for seed in expected_seeds]
    rules = config["replication_rules"]
    primary = [row["target_vs_reference"]["primary"] for row in ordered]
    secondary = [row["target_vs_reference"]["secondary"] for row in ordered]
    control_primary = [row["target_vs_mechanism_control"]["primary"] for row in ordered]
    control_secondary = [row["target_vs_mechanism_control"]["secondary"] for row in ordered]
    primary_passes = sum(bool(row["target_vs_reference"]["primary_directional_pass"]) for row in ordered)
    secondary_passes = sum(bool(row["target_vs_reference"]["secondary_directional_pass"]) for row in ordered)
    strong_passes = sum(bool(row["target_vs_reference"]["absolute_strong"]) for row in ordered)

    median_primary_recall = median_field(primary, "wake_recall")
    median_primary_fp = median_field(primary, "negative_fp_rate")
    median_secondary_recall = median_field(secondary, "wake_recall")
    median_secondary_fp = median_field(secondary, "negative_fp_rate")

    replication_pass = (
        primary_passes >= int(rules["min_primary_directional_passes"])
        and secondary_passes >= int(rules["min_secondary_directional_passes"])
        and strong_passes >= int(rules["min_absolute_strong_passes"])
        and median_primary_recall is not None
        and median_primary_recall >= float(rules["median_primary_min_wake_recall_delta"])
        and median_primary_fp is not None
        and median_primary_fp <= float(rules["median_primary_max_negative_fp_rate_delta"])
    )
    next_experiment = (
        str(rules["pass_next"])
        if replication_pass
        else str(rules["fail_next"])
    )

    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-encoder-frontend-paired-seed-summary-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "source_factorial_evidence": config["source_factorial_evidence"],
        "seeds": expected_seeds,
        "reference_candidate": rules["reference_candidate"],
        "target_candidate": rules["target_candidate"],
        "mechanism_control_candidate": rules["mechanism_control_candidate"],
        "paired_same_runner_per_seed": True,
        "seed_results": {str(row["seed"]): row for row in ordered},
        "replication": {
            "primary_directional_passes": primary_passes,
            "secondary_directional_passes": secondary_passes,
            "absolute_strong_passes": strong_passes,
            "median_primary_delta": {
                "wake_recall": median_primary_recall,
                "negative_fp_rate": median_primary_fp,
            },
            "median_secondary_delta": {
                "wake_recall": median_secondary_recall,
                "negative_fp_rate": median_secondary_fp,
            },
            "target_vs_mechanism_control_median_primary_delta": {
                "wake_recall": median_field(control_primary, "wake_recall"),
                "negative_fp_rate": median_field(control_primary, "negative_fp_rate"),
            },
            "target_vs_mechanism_control_median_secondary_delta": {
                "wake_recall": median_field(control_secondary, "wake_recall"),
                "negative_fp_rate": median_field(control_secondary, "negative_fp_rate"),
            },
            "pass": replication_pass,
        },
        "next_experiment": next_experiment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pre-registered paired KWS architecture replication.")
    sub = parser.add_subparsers(dest="mode", required=True)

    seed_parser = sub.add_parser("seed")
    seed_parser.add_argument("--config", required=True, type=pathlib.Path)
    seed_parser.add_argument("--seed", required=True, type=int)
    seed_parser.add_argument("--root", required=True, type=pathlib.Path)
    seed_parser.add_argument("--output", required=True, type=pathlib.Path)

    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--root", required=True, type=pathlib.Path)
    aggregate_parser.add_argument("--output", required=True, type=pathlib.Path)

    args = parser.parse_args()
    config = load_json(args.config)
    if args.mode == "seed":
        if int(args.seed) not in [int(value) for value in config["seeds"]]:
            raise ValueError(f"seed {args.seed} is not pre-registered")
        result = seed_evidence(config, int(args.seed), args.root.resolve())
    else:
        result = aggregate_evidence(config, args.root.resolve())
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
