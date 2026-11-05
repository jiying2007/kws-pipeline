#!/usr/bin/env python3
"""Compare research scorecards under an explicit ablation or system mode.

Ablation retains the single-declared-variable and identical-code checks. System
mode permits different models/code/native frontends only after rechecking shared
input bytes, corpus provenance and the common scoring protocol. Its result does
not attribute a difference to one particular implementation change.

Schema 5 uses exact one-sided Poisson FAR bounds and versioned, approximate
event-level FRR intervals. Schemas 1--4 retain historical interpretation and
cannot be mixed with the new policy. No verdict qualifies a model or proves
generalization to independent human speakers.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

from dataset_iteration_policy import (
    APPROX_FAR_POLICY, CONFIDENCE, DATASET_IDENTITY_POLICY, FAR_POLICY, MAX_FA_EVENTS,
    COMPARISON_POLICY, COMPARISON_SCHEMA_VERSION, DEFAULT_FRR_MARGIN,
    FRR_INTERVAL_CONFIDENCE, FRR_ARM_INTERVAL_CONFIDENCE, FRR_BOUND_CONFIDENCE,
    far_upper_bound_95, historical_far_upper_bound_95, positive_counts, frr_comparison,
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


def keyword_counts(card: dict, label: str) -> dict | None:
    """Read the existing scorer's per-word table, without inventing missing data."""
    domains = card.get("domains", {})
    if type(domains) is not dict:
        raise ValueError(label + ": domains must be an object")
    overall = domains.get("overall", {})
    if type(overall) is not dict:
        raise ValueError(label + ": domain overall metrics must be an object")
    if "per_keyword" not in overall:
        return None
    table = overall["per_keyword"]
    if type(table) is not dict:
        raise ValueError(label + ": per_keyword must be an object")
    total = positive_counts(card["metrics"], label)
    if positive_counts(overall, label + ".domains.overall") != total:
        raise ValueError(label + ": domain overall counts differ from metrics")
    result = {}
    for key, row in table.items():
        if (type(key) is not str or not key.isascii() or not key.isdecimal()
                or len(key) > 10 or not 0 < int(key) <= 0xffffffff or str(int(key)) != key):
            raise ValueError(label + ": invalid per-keyword ID")
        result[key] = positive_counts(row, label + ".per_keyword." + key)
        false_accepts = row.get("false_accepts")
        if type(false_accepts) is not int or false_accepts < 0:
            raise ValueError(label + ": per-keyword false accepts must be non-negative integers")
        result[key]["false_accepts"] = false_accepts
    for name in ("expected", "matched", "false_rejects"):
        if sum(row[name] for row in result.values()) != total[name]:
            raise ValueError(label + ": per-keyword counts do not sum to aggregate " + name)
    false_accepts = number(card["metrics"].get("false_accepts"), label + ".false_accepts")
    if (sum(row["false_accepts"] for row in result.values()) != false_accepts
            or overall.get("false_accepts") != false_accepts):
        raise ValueError(label + ": per-keyword false accepts do not sum to aggregate")
    return result


def current_assessment(before: dict, after: dict, far_budget: float,
                       margin: float, latency_budget: float) -> dict:
    aggregate = frr_comparison(before["metrics"], after["metrics"], margin)
    words_before, words_after = keyword_counts(before, "before"), keyword_counts(after, "after")
    same_dataset = before["identity"]["dataset_id"] == after["identity"]["dataset_id"]
    if same_dataset and aggregate["baseline"]["expected"] != aggregate["candidate"]["expected"]:
        raise ValueError("same evaluation dataset has inconsistent positive denominators")
    per_keyword = {}
    empty = {"expected": 0, "matched": 0, "false_rejects": 0, "frr": 0.0, "false_accepts": 0}
    if words_before is not None and words_after is not None:
        for key in sorted(words_before.keys() | words_after.keys(), key=int):
            left, right = words_before.get(key, empty), words_after.get(key, empty)
            if same_dataset and left["expected"] != right["expected"]:
                raise ValueError("same evaluation dataset has inconsistent per-keyword denominators")
            per_keyword[key] = {
                **frr_comparison(left, right, margin),
                "false_accepts": {"baseline": left["false_accepts"], "candidate": right["false_accepts"]},
                "scope": "per-word diagnostic only; no automatic aggregate veto",
            }

    def far_status(card, label):
        point = number(card["far_rate"]["point_estimate_per_hour"], label + ".far_point")
        upper = far_bound(card, label)
        status = ("met" if upper <= far_budget else
                  "observed_rate_above_budget" if point > far_budget else "not_demonstrated")
        return {"status": status, "observed_false_accepts": card["far_rate"]["observed_false_accepts"],
                "exposure_hours": number(card["exposure"]["exposure_hours"], label + ".hours"),
                "point_estimate_per_hour": point, "upper_bound_95_per_hour": upper}

    far_before, far_after = far_status(before, "before"), far_status(after, "after")
    common_far = far_before["status"] == far_after["status"] == "met"
    latency_before = metric(before, "p95_post_end_latency_ms")
    latency_after = metric(after, "p95_post_end_latency_ms")
    latency_met = latency_after <= latency_budget
    zero_event_hours = -math.log(1 - CONFIDENCE) / far_budget if far_budget else None
    if zero_event_hours is not None and not math.isfinite(zero_event_hours):
        zero_event_hours = None
    reasons = []
    if not common_far:
        reasons.append("common FAR budget not demonstrated on both sides; this is not proof of regression")
    if not latency_met:
        reasons.append("candidate latency point exceeds budget; no latency uncertainty estimate is available")
    if aggregate["verdict"] == "inconclusive":
        reasons.append("positive support or FRR interval cannot resolve the predeclared material margin")
    verdict = aggregate["verdict"] if common_far and latency_met else "inconclusive"
    return {
        "comparison_policy": COMPARISON_POLICY,
        "frr_direction": aggregate["direction"],
        "frr_assessment": aggregate,
        "per_keyword": per_keyword,
        "per_keyword_status": ("available" if words_before is not None and words_after is not None
                               else "not_available_on_both_scorecards"),
        "statistical_policy": {
            "method": "difference of Bonferroni-adjusted marginal Wilson intervals",
            "confidence_scope": "approximate nominal coverage conditional on independent Bernoulli events with a common miss probability within each arm",
            "nominal_difference_confidence": FRR_INTERVAL_CONFIDENCE,
            "marginal_two_sided_confidence": FRR_ARM_INTERVAL_CONFIDENCE,
            "one_sided_bound_confidence": FRR_BOUND_CONFIDENCE,
            "material_frr_margin": margin,
            "margin_selection_timing_authenticated": False,
            "paired_discordances_used": False,
            "source_cluster_adjustment": False,
            "per_keyword_familywise_adjustment": False,
        },
        "budget_status": {
            "far": {"baseline": far_before, "candidate": far_after,
                    "common_budget_demonstrated": common_far,
                    "zero_event_hours_required": zero_event_hours},
            "latency": {"baseline_p95_ms": latency_before, "candidate_p95_ms": latency_after,
                        "candidate_status": "met" if latency_met else "observed_above_budget",
                        "uncertainty_available": False,
                        "coordinate": "post-reference-end sample time; not callback wall time"},
        },
        "evidence_scope": {
            "statistical_unit": "annotated wake event in the measured research corpus",
            "independent_events_verified": False,
            "independent_human_evidence": False,
            "natural_negative_population_authenticated": False,
            "qualification_allowed": False,
        },
        "limitations": [
            "Wilson intervals have approximate coverage; the Bonferroni construction is nominal, not an exact guarantee.",
            "Shared speakers, repeated sources and synthetic material are not independent human observations; event clustering can invalidate these intervals.",
            "The comparison uses marginal counts, not matched-event discordances, and does not adjust for repeated looks or candidate selection.",
            "Per-word intervals are diagnostics, not simultaneous evidence across all words or a requirement that every slice improve.",
            "FAR applies only to the recorded negative population; synthetic or challenge-clip duration does not establish natural continuous-audio FAR.",
        ],
        "inconclusive_reasons": reasons,
        "regressions": (["FRR difference interval is entirely above the declared material margin"]
                        if verdict == "regressed" else []),
        "research_continuation_allowed": True,
        "verdict": verdict,
    }


def compare(before: dict, after: dict, far_budget: float, frr_tolerance: float | None,
            latency_budget: float, *, mode: str = "ablation",
            baseline_evidence_root: pathlib.Path | None = None,
            candidate_evidence_root: pathlib.Path | None = None) -> dict:
    if frr_tolerance is None:
        frr_tolerance = DEFAULT_FRR_MARGIN if after.get("schema_version") == 5 else 0.0
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

    # Preserve the historical point-estimate/budget interpretation for schemas
    # 1--4. Schema 5 receives the separately versioned assessment below.
    regressions: list[str] = []
    improved = False
    if after["schema_version"] < 5:
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
        improved = (not regressions and after_frr < before_frr - frr_tolerance
                    and after_bound <= before_bound)
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION if after["schema_version"] == 5 else 1,
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
    if after["schema_version"] == 5:
        report.update(current_assessment(before, after, far_budget, frr_tolerance, latency_budget))
    return report


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
    parser.add_argument("--frr-margin", "--frr-tolerance", dest="frr_tolerance", type=float, default=None,
                        help="predeclared material FRR margin (schema 5 default 0.02; historical schemas default 0)")
    parser.add_argument("--latency-budget", type=float, default=800.0,
                        help="ceiling on p95 post-end latency in ms")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if (args.far_budget < 0.0 or (args.frr_tolerance is not None and args.frr_tolerance < 0.0)
            or args.latency_budget <= 0.0):
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
