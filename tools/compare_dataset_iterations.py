#!/usr/bin/env python3
"""Compare two dataset-iteration scorecards as a paired A/B result.

The governed gate in this repository asks for zero false rejects *and* zero
false accepts at the same time. That is not a measurable target, and it is why
the frozen qualification has failed every time it ran. This comparator is what
replaces it for the iteration lane: instead of an absolute verdict, it answers
"did this change make things better, worse, or is there nothing to say yet?"

Three rules make the answer trustworthy:

1. **Same protocol, or refuse.** Numbers produced under different protocols are
   not the same measurement.
2. **Exactly one variable may move.** If both the dataset and the model changed,
   there is no interpretation of the delta -- it is a different experiment. The
   run declares which variable is under test, and the comparator enforces that
   only that one differs.
3. **Compare false accepts on the upper bound, not the point estimate.**
   Zero false accepts over 30 seconds and zero over 30 hours are wildly
different evidence, yet both read as ``far_per_hour = 0``. The bound carries the
   exposure; the point estimate does not.

The verdict is deliberately about *regression*, not about shipping: this lane
measures, it does not qualify.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

IDENTITY_FIELDS = ("protocol", "dataset_id", "model_id", "code_sha", "seed")
METRIC_FIELDS = ("frr", "far_per_hour", "p95_post_end_latency_ms")


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
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not a number: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def metric(row: dict, field: str) -> float:
    return number(row["metrics"].get(field, 0.0), f"metrics.{field}")


def far_bound(row: dict, label: str) -> float:
    return number(row["far_rate"].get("upper_bound_95_per_hour"), f"{label}.far_rate.upper_bound_95_per_hour")


def framing(before: dict, after: dict) -> dict:
    """Check that the two scorecards form a valid paired comparison."""
    left, right = before["identity"], after["identity"]
    if left["protocol"] != right["protocol"]:
        raise ValueError(
            f"protocol mismatch: {left['protocol']} vs {right['protocol']} -- "
            "results measured under different protocols are not comparable"
        )
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
            "baseline on the current code, or declare the code change as the variable."
        )
    return {"variable": variable, "moved": moved[0]}


def compare(before: dict, after: dict, far_budget: float, frr_tolerance: float,
            latency_budget: float) -> dict:
    frame = framing(before, after)
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
        "schema_version": 1,
        "protocol": after["identity"]["protocol"],
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
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

    report = compare(load(args.baseline), load(args.candidate),
                     args.far_budget, args.frr_tolerance, args.latency_budget)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
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
