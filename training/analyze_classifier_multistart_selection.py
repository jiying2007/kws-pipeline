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


def operating_points(result: dict, config: dict) -> dict:
    calibration = {
        float(row["threshold"]): row
        for row in result["calibration"]["operating_curve"]
    }
    test = {
        float(row["threshold"]): row
        for row in result["test"]["operating_curve"]
    }
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
            "calibration": {
                "wake_recall": float(row["wake_exact_recall"]),
                "negative_fp_rate": float(row["negative_false_positive_rate"]),
            },
            "test": {
                "wake_recall": float(test[threshold]["wake_exact_recall"]),
                "negative_fp_rate": float(test[threshold]["negative_false_positive_rate"]),
            },
        }
    return selected


def calibration_rank(row: dict, config: dict) -> tuple:
    selection = config["selection"]
    primary_name = str(selection["primary_operating_point"])
    secondary_name = str(selection["secondary_operating_point"])
    primary = row["operating_points"].get(primary_name)
    secondary = row["operating_points"].get(secondary_name)

    def recall(point: dict | None) -> float:
        return -1.0 if point is None else float(point["calibration"]["wake_recall"])

    def false_positive(point: dict | None) -> float:
        return 1.0 if point is None else float(point["calibration"]["negative_fp_rate"])

    return (
        recall(primary),
        recall(secondary),
        -false_positive(primary),
        -false_positive(secondary),
        -int(row["model_seed"]),
    )


def select_start(rows: list[dict], config: dict) -> dict:
    if not rows:
        raise ValueError("no starts to select")
    # This ranking is calibration-only by construction. Test metrics are
    # retained in the selected record but are never referenced here.
    return max(rows, key=lambda row: calibration_rank(row, config))


def test_delta(target: dict | None, reference: dict | None) -> dict | None:
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


def cohort(config: dict, cohort_name: str, root: pathlib.Path) -> dict:
    seeds = [int(value) for value in config["cohorts"].get(cohort_name, [])]
    if len(seeds) != int(config["fixed"]["starts_per_architecture"]):
        raise ValueError("cohort seed count drift")
    if len(seeds) != len(set(seeds)):
        raise ValueError("cohort contains duplicate seed")

    by_architecture: dict[str, list[dict]] = {}
    cpu_contracts = []
    for architecture, spec in config["architectures"].items():
        starts = []
        for seed in seeds:
            result = load_json(root / architecture / str(seed) / "classifier.json")
            if int(result["model_seed"]) != seed:
                raise ValueError(f"{architecture}:{seed}: model seed drift")
            if int(result["sampler_seed"]) != int(config["fixed"]["sampler_seed"]):
                raise ValueError(f"{architecture}:{seed}: sampler seed drift")
            if str(result["frontend"]) != str(spec["frontend"]):
                raise ValueError(f"{architecture}:{seed}: frontend drift")
            if int(result["hidden_dim"]) != int(spec["hidden_dim"]):
                raise ValueError(f"{architecture}:{seed}: hidden drift")
            if str(result["balance_mode"]) != str(config["fixed"]["balance_mode"]):
                raise ValueError(f"{architecture}:{seed}: balance drift")
            if str(result["lr_schedule"]) != str(config["fixed"]["lr_schedule"]):
                raise ValueError(f"{architecture}:{seed}: schedule drift")
            if str(result["init_mode"]) != str(config["fixed"]["init_mode"]):
                raise ValueError(f"{architecture}:{seed}: init drift")
            cpu_contracts.append(result["research_cpu_contract"])
            starts.append(
                {
                    "model_seed": seed,
                    "initial_model_state_sha256": result["initial_model_state_sha256"],
                    "final_model_state_sha256": result["model_state_sha256"],
                    "final_train_loss": float(result["history"][-1]["loss"]),
                    "operating_points": operating_points(result, config),
                }
            )
        by_architecture[architecture] = starts

    if any(contract != cpu_contracts[0] for contract in cpu_contracts[1:]):
        raise ValueError("cohort starts did not share research CPU contract")

    selected = {
        architecture: select_start(starts, config)
        for architecture, starts in by_architecture.items()
    }
    primary_name = str(config["selection"]["primary_operating_point"])
    secondary_name = str(config["selection"]["secondary_operating_point"])
    reference = selected["b0-logmel64"]
    target = selected["fc1-pcen128"]
    primary_delta = test_delta(
        target["operating_points"].get(primary_name),
        reference["operating_points"].get(primary_name),
    )
    secondary_delta = test_delta(
        target["operating_points"].get(secondary_name),
        reference["operating_points"].get(secondary_name),
    )

    rules = config["decision_rules"]
    target_primary = target["operating_points"].get(primary_name)
    target_secondary = target["operating_points"].get(secondary_name)
    primary_directional = (
        primary_delta is not None
        and float(primary_delta["wake_recall"])
        >= float(rules["primary_min_test_wake_recall_delta"])
        and float(primary_delta["negative_fp_rate"])
        <= float(rules["max_test_negative_fp_delta_regression"])
    )
    secondary_directional = (
        secondary_delta is not None
        and float(secondary_delta["wake_recall"])
        >= float(rules["secondary_min_test_wake_recall_delta"])
        and float(secondary_delta["negative_fp_rate"])
        <= float(rules["max_test_negative_fp_delta_regression"])
    )
    absolute_strong = (
        target_primary is not None
        and target_secondary is not None
        and float(target_primary["test"]["wake_recall"])
        >= float(rules["absolute_primary_min_test_wake_recall"])
        and float(target_primary["test"]["negative_fp_rate"])
        <= float(rules["absolute_primary_max_test_negative_fp_rate"])
        and float(target_secondary["test"]["wake_recall"])
        >= float(rules["absolute_secondary_min_test_wake_recall"])
        and float(target_secondary["test"]["negative_fp_rate"])
        <= float(rules["absolute_secondary_max_test_negative_fp_rate"])
    )
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-multistart-cohort-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "cohort": cohort_name,
        "selection_authority": config["selection"]["authority"],
        "test_metrics_used_for_selection": False,
        "seeds": seeds,
        "starts": by_architecture,
        "selected": selected,
        "primary_test_delta": primary_delta,
        "secondary_test_delta": secondary_delta,
        "primary_directional_pass": primary_directional,
        "secondary_directional_pass": secondary_directional,
        "absolute_strong_pass": absolute_strong,
    }


def aggregate(config: dict, root: pathlib.Path) -> dict:
    rows = {}
    for path in root.rglob("cohort-evidence.json"):
        row = load_json(path)
        name = str(row["cohort"])
        if name in rows:
            raise ValueError(f"duplicate cohort: {name}")
        rows[name] = row
    expected = list(config["cohorts"])
    if set(rows) != set(expected):
        raise ValueError(f"cohort mismatch: {sorted(rows)} vs {expected}")
    ordered = [rows[name] for name in expected]
    rules = config["decision_rules"]
    primary_passes = sum(bool(row["primary_directional_pass"]) for row in ordered)
    secondary_passes = sum(bool(row["secondary_directional_pass"]) for row in ordered)
    strong_passes = sum(bool(row["absolute_strong_pass"]) for row in ordered)

    primary_name = str(config["selection"]["primary_operating_point"])
    target_primary_recall = []
    target_primary_fp = []
    selected_target_seeds = []
    selected_reference_seeds = []
    for row in ordered:
        selected_target = row["selected"]["fc1-pcen128"]
        selected_reference = row["selected"]["b0-logmel64"]
        selected_target_seeds.append(int(selected_target["model_seed"]))
        selected_reference_seeds.append(int(selected_reference["model_seed"]))
        point = selected_target["operating_points"].get(primary_name)
        if point is None:
            raise ValueError(f"{row['cohort']}: selected target lacks primary operating point")
        target_primary_recall.append(float(point["test"]["wake_recall"]))
        target_primary_fp.append(float(point["test"]["negative_fp_rate"]))

    recall_mean = statistics.fmean(target_primary_recall)
    recall_stddev = (
        statistics.pstdev(target_primary_recall)
        if len(target_primary_recall) > 1
        else 0.0
    )
    fp_mean = statistics.fmean(target_primary_fp)

    pass_value = (
        primary_passes >= int(rules["min_primary_directional_passes"])
        and secondary_passes >= int(rules["min_secondary_directional_passes"])
        and strong_passes >= int(rules["min_absolute_strong_passes"])
        and recall_stddev <= float(rules["max_selected_target_primary_recall_stddev"])
        and recall_mean >= float(rules["min_selected_target_primary_recall_mean"])
    )
    result = {
        "schema_version": 1,
        "evidence_class": "kws-v2-classifier-multistart-summary-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "selection_authority": config["selection"]["authority"],
        "test_metrics_used_for_selection": False,
        "primary_directional_passes": primary_passes,
        "secondary_directional_passes": secondary_passes,
        "absolute_strong_passes": strong_passes,
        "selected_target_primary_test_wake_recall_mean": recall_mean,
        "selected_target_primary_test_wake_recall_stddev": recall_stddev,
        "selected_target_primary_test_negative_fp_mean": fp_mean,
        "selected_target_model_seeds": selected_target_seeds,
        "selected_reference_model_seeds": selected_reference_seeds,
        "multistart_pass": pass_value,
        "cohorts": {name: rows[name] for name in expected},
        "next_experiment": str(
            rules["pass_next"] if pass_value else rules["fail_next"]
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("cohort")
    one.add_argument("--config", required=True, type=pathlib.Path)
    one.add_argument("--cohort", required=True)
    one.add_argument("--root", required=True, type=pathlib.Path)
    one.add_argument("--output", required=True, type=pathlib.Path)
    many = sub.add_parser("aggregate")
    many.add_argument("--config", required=True, type=pathlib.Path)
    many.add_argument("--root", required=True, type=pathlib.Path)
    many.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = load_json(args.config)
    if args.mode == "cohort":
        result = cohort(config, str(args.cohort), args.root.resolve())
    else:
        result = aggregate(config, args.root.resolve())
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
