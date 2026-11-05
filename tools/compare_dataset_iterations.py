#!/usr/bin/env python3
"""Compare research scorecards under an explicit ablation or system mode.

Ablation retains the single-declared-variable and identical-code checks. System
mode permits different models/code/native frontends only after rechecking shared
input bytes, corpus provenance and the common scoring protocol. Its result does
not attribute a difference to one particular implementation change.

Schema 5 uses exact one-sided Poisson bounds. Schemas 1--4 retain historical
interpretation and cannot be mixed with the new policy. No verdict qualifies a
model or establishes statistical significance.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

from dataset_iteration_policy import (
    APPROX_FAR_POLICY, CONFIDENCE, DATASET_IDENTITY_POLICY, FAR_POLICY, MAX_FA_EVENTS,
    far_upper_bound_95, historical_far_upper_bound_95,
)
from dataset_iteration_inputs import (
    adjacent_evidence_root, system_input_paths, validate_output_paths, verify_system_evidence,
)
from export_artifacts import write_export_artifacts

IDENTITY_FIELDS = ("protocol", "dataset_id", "model_id", "code_sha", "seed")
METRIC_FIELDS = ("frr", "far_per_hour", "p95_post_end_latency_ms")
DOMAIN_ONLY_DATASET_IDENTITY_POLICY = "effective-domain-keys-v1"
LEGACY_DATASET_IDENTITY_POLICY = "legacy-top-level-distance-v1"
LEGACY_FAR_POLICY = "legacy-mixed-recordings-iteration-approx95-v1"


def load(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"scorecard not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key in ("identity", "metrics", "exposure", "far_rate"):
        if key not in value:
            raise ValueError(f"{path}: scorecard is missing '{key}'")
    identity = value["identity"]
    if not isinstance(identity, dict):
        raise ValueError(f"{path}: identity must be an object")
    for key in IDENTITY_FIELDS:
        if key not in identity:
            raise ValueError(f"{path}: identity is missing '{key}'")
    if "variable" not in identity:
        raise ValueError(f"{path}: identity is missing 'variable'")
    return value


def number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} is not a number: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def metric(row: dict, field: str) -> float:
    if field not in row["metrics"]:
        raise ValueError(f"metrics is missing required '{field}'")
    result = number(row["metrics"][field], f"metrics.{field}")
    if result < 0 or (field == "frr" and result > 1):
        raise ValueError(f"metrics.{field} is outside its valid range")
    return result


def far_bound(row: dict, label: str) -> float:
    return number(row["far_rate"].get("upper_bound_95_per_hour"), f"{label}.far_rate.upper_bound_95_per_hour")


def dataset_identity_policy(row: dict, label: str) -> str:
    """Classify semantics independently of caller-selected protocol labels."""
    schema = row.get("schema_version")
    identity = row["identity"]
    if type(schema) is not int or schema not in (1, 2, 3, 4, 5):
        raise ValueError(f"{label}: unsupported scorecard schema/dataset identity policy combination")
    if schema in (1, 2):
        if "dataset_identity_policy" in identity:
            raise ValueError(f"{label}: legacy schema must not declare a dataset identity policy")
        return LEGACY_DATASET_IDENTITY_POLICY
    expected_policy = (DOMAIN_ONLY_DATASET_IDENTITY_POLICY if schema == 3 else
                       DATASET_IDENTITY_POLICY)
    if identity.get("dataset_identity_policy") != expected_policy:
        raise ValueError(f"{label}: schema {schema} requires the known dataset identity policy")
    return expected_policy


def far_policy(row: dict, label: str) -> str:
    """Schema identifies legacy cards even when their protocol label is custom."""
    schema = row.get("schema_version")
    identity = row["identity"]
    if type(schema) is int and schema == 1 and "far_policy" not in identity:
        return LEGACY_FAR_POLICY
    if type(schema) is not int or schema not in (2, 3, 4, 5):
        raise ValueError(f"{label}: unsupported scorecard schema/FAR policy combination")
    expected_policy = FAR_POLICY if schema == 5 else APPROX_FAR_POLICY
    if identity.get("far_policy") != expected_policy:
        raise ValueError(f"{label}: schema {schema} requires the known FAR policy")
    if not isinstance(row["far_rate"], dict) or not isinstance(row["exposure"], dict):
        raise ValueError(f"{label}: schema {schema} FAR rate and exposure must be objects")
    count = row["far_rate"].get("observed_false_accepts")
    if type(count) is not int or count < 0:
        raise ValueError(f"{label}: negative false accepts must be a non-negative integer")
    if schema == 5 and count > MAX_FA_EVENTS:
        raise ValueError(f"{label}: negative false accepts exceed {MAX_FA_EVENTS}")
    raw_hours = row["exposure"].get("exposure_hours")
    raw_point = row["far_rate"].get("point_estimate_per_hour")
    raw_bound = row["far_rate"].get("upper_bound_95_per_hour")
    if any(isinstance(value, bool) for value in (raw_hours, raw_point, raw_bound)):
        raise ValueError(f"{label}: FAR tuple must not contain booleans")
    hours = number(raw_hours, f"{label}.exposure_hours")
    point = number(raw_point, f"{label}.far_rate.point_estimate_per_hour")
    bound = far_bound(row, label)
    if hours <= 0.0 or point < 0.0 or bound < 0.0 or point != count / hours:
        raise ValueError(f"{label}: inconsistent negative FAR tuple")
    if schema == 5:
        if (type(row["far_rate"].get("confidence_level")) not in (int, float)
                or row["far_rate"]["confidence_level"] != CONFIDENCE):
            raise ValueError(f"{label}: exact FAR requires one-sided 95% confidence")
        expected_bound = far_upper_bound_95(count, hours)
    else:
        # Preserve old bytes/interpretation; never silently relabel old bounds.
        expected_bound = historical_far_upper_bound_95(count, hours)
    if not math.isfinite(expected_bound) or bound != expected_bound:
        raise ValueError(f"{label}: inconsistent negative FAR bound")
    return expected_policy


def framing(before: dict, after: dict, *, mode: str = "ablation",
            baseline_evidence_root: pathlib.Path | None = None,
            candidate_evidence_root: pathlib.Path | None = None) -> dict:
    """Check mode and evidence before permitting any metric comparison."""
    if mode not in ("ablation", "system"):
        raise ValueError("comparison mode must be ablation or system")
    if mode != "system" and any(root is not None for root in
                               (baseline_evidence_root, candidate_evidence_root)):
        raise ValueError("evidence relocation requires system mode")
    if dataset_identity_policy(before, "before") != dataset_identity_policy(after, "after"):
        raise ValueError("dataset identity policy mismatch: re-run both sides under the same policy")
    if far_policy(before, "before") != far_policy(after, "after"):
        raise ValueError("FAR policy mismatch: re-run the baseline under the same measurement policy")
    left, right = before["identity"], after["identity"]
    for label, card in (("before", before), ("after", after)):
        declared = card["identity"].get("comparison_mode", "ablation")
        if declared != mode or (card["schema_version"] == 5 and
                                "comparison_mode" not in card["identity"]):
            raise ValueError(f"{label}: comparison mode mismatch; explicitly produce cards for {mode}")
    if left["protocol"] != right["protocol"]:
        raise ValueError(
            f"protocol mismatch: {left['protocol']} vs {right['protocol']} -- "
            "results measured under different protocols are not comparable"
        )
    if mode == "system":
        evidence_before = verify_system_evidence(before, "before", evidence_root=baseline_evidence_root)
        evidence_after = verify_system_evidence(after, "after", evidence_root=candidate_evidence_root)
        if evidence_before != evidence_after:
            raise ValueError("system comparison requires identical verified inputs and evaluation protocol")
        moved = [field for field in ("model_id", "code_sha", "runner_sha256", "native_frontend")
                 if left.get(field) != right.get(field)]
        if not moved:
            raise ValueError("nothing moved: the system identities are identical")
        return {"variable": "system", "moved": moved, "mode": mode,
                "verified_evaluation": evidence_after}
    if right["variable"] not in ("dataset", "model"):
        raise ValueError("ablation variable must be dataset or model")
    moved = [field for field in ("dataset_id", "model_id") if left[field] != right[field]]
    variable = str(right["variable"])
    if not moved:
        raise ValueError(
            "nothing moved: dataset and model are both identical, so there is no "
            "experiment to compare"
        )
    if len(moved) > 1:
        raise ValueError(
            "both dataset and model changed: with two variables moving there is no "
            "interpretation of the delta. Run them one at a time."
        )
    expected = "dataset_id" if variable == "dataset" else "model_id"
    if moved[0] != expected:
        raise ValueError(
            f"declared variable is '{variable}' (so {expected} should differ) but "
            f"{moved[0]} is what actually changed"
        )
    if left["code_sha"] != right["code_sha"]:
        raise ValueError(
            "code_sha differs: a code change confounds the comparison. Re-run the "
            "baseline on the current code, or produce both cards in explicit system mode."
        )
    return {"variable": variable, "moved": moved[0], "mode": mode}


def compare(before: dict, after: dict, far_budget: float, frr_tolerance: float,
            latency_budget: float, *, mode: str = "ablation",
            baseline_evidence_root: pathlib.Path | None = None,
            candidate_evidence_root: pathlib.Path | None = None) -> dict:
    if any(isinstance(value, bool) or not math.isfinite(value) for value in
           (far_budget, frr_tolerance, latency_budget)) or (
            far_budget < 0 or not 0 <= frr_tolerance <= 1 or latency_budget <= 0):
        raise ValueError("budgets must be finite and non-negative, FRR tolerance <= 1, latency positive")
    frame = framing(before, after, mode=mode, baseline_evidence_root=baseline_evidence_root,
                    candidate_evidence_root=candidate_evidence_root)
    deltas = {
        field: metric(after, field) - metric(before, field) for field in METRIC_FIELDS
    }
    deltas["far_upper_bound_95"] = far_bound(after, "after") - far_bound(before, "before")

    before_frr = metric(before, "frr")
    after_frr = metric(after, "frr")
    before_bound = far_bound(before, "before")
    after_bound = far_bound(after, "after")
    after_latency = metric(after, "p95_post_end_latency_ms")

    # A regression is measured, not absolute: worse recall, a false-accept bound
    # pushed past the declared budget, or latency past its ceiling.
    regressions: list[str] = []
    if after_frr > before_frr + frr_tolerance:
        regressions.append(
            f"frr regressed: {before_frr:g} -> {after_frr:g} "
            f"(tolerance {frr_tolerance:g})"
        )
    if after_bound > far_budget:
        regressions.append(
            f"far upper bound {after_bound:g}/h exceeds budget {far_budget:g}/h"
        )
    if after_latency > latency_budget:
        regressions.append(
            f"p95 latency {after_latency:g} ms exceeds budget {latency_budget:g} ms"
        )

    improved = (
        not regressions
        and after_frr < before_frr - frr_tolerance
        and after_bound <= before_bound
    )
    return {
        "schema_version": 2 if after["schema_version"] == 5 else 1,
        "protocol": after["identity"]["protocol"],
        "comparison_mode": frame["mode"],
        "comparison_scope": ("system-quality-without-single-change-attribution" if mode == "system"
                             else "one-declared-variable-with-fixed-code"),
        "verified_evaluation": frame.get("verified_evaluation"),
        "input_evidence_scope": ("rechecked WAV/PCM, references, saved events and corpus provenance; "
                                 "runner context is declared, not execution-attested" if mode == "system" else None),
        "far_policy": far_policy(after, "after"),
        "historical_statistics": after["schema_version"] < 5,
        "variable": frame["variable"],
        "moved": frame["moved"],
        "baseline_run": before.get("run_id"),
        "candidate_run": after.get("run_id"),
        "exposure_hours": {
            "baseline": number(before["exposure"].get("exposure_hours"), "before.exposure_hours"),
            "candidate": number(after["exposure"].get("exposure_hours"), "after.exposure_hours"),
        },
        "deltas": deltas,
        "budgets": {
            "far_upper_bound_95_per_hour": far_budget,
            "frr_tolerance": frr_tolerance,
            "p95_latency_ms": latency_budget,
        },
        "regressions": regressions,
        "verdict": "regressed" if regressions else ("improved" if improved else "neutral"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two dataset-iteration scorecards as a paired A/B result."
    )
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--mode", choices=("ablation", "system"), default="ablation",
                        help="system requires schema-5 measured-input evidence; default retains strict ablation")
    parser.add_argument("--baseline-evidence-root", type=pathlib.Path,
                        help="restored baseline origin_root tree; scorecard bytes remain unchanged")
    parser.add_argument("--candidate-evidence-root", type=pathlib.Path,
                        help="restored candidate origin_root tree; scorecard bytes remain unchanged")
    parser.add_argument("--far-budget", type=float, default=1.0,
                        help="ceiling on the 95%% upper bound of false accepts per hour")
    parser.add_argument("--frr-tolerance", type=float, default=0.0,
                        help="FRR increase that is still called neutral (default 0)")
    parser.add_argument("--latency-budget", type=float, default=800.0,
                        help="ceiling on p95 post-end latency in ms")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.far_budget < 0.0 or args.frr_tolerance < 0.0 or args.latency_budget <= 0.0:
        raise ValueError("budgets must be non-negative and latency budget positive")

    before, after = load(args.baseline), load(args.candidate)
    if args.mode == "system":
        args.baseline_evidence_root = (args.baseline_evidence_root or
                                      adjacent_evidence_root(args.baseline, before))
        args.candidate_evidence_root = (args.candidate_evidence_root or
                                       adjacent_evidence_root(args.candidate, after))
    report = compare(before, after, args.far_budget, args.frr_tolerance,
                     args.latency_budget, mode=args.mode,
                     baseline_evidence_root=args.baseline_evidence_root,
                     candidate_evidence_root=args.candidate_evidence_root)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        inputs = [args.baseline, args.candidate]
        if args.mode == "system":
            inputs.extend(system_input_paths(before, args.baseline_evidence_root))
            inputs.extend(system_input_paths(after, args.candidate_evidence_root))
        validate_output_paths({"comparison": args.output}, inputs)
        write_export_artifacts({args.output: (text + "\n").encode("utf-8")})
    print(text)
    if report["verdict"] == "regressed":
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
