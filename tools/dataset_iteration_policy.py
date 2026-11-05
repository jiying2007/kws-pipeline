"""Versioned statistical policies for dataset-iteration scorecards.

Schemas 2--4 retain their historical approximation for replay only. New cards
and comparisons use the same exact, one-sided Poisson helper as qualification.
FRR comparison intervals are separately versioned approximate research evidence.
"""
from __future__ import annotations

import math

from statistical_bounds import poisson_rate_upper, wilson_upper

SCHEMA_VERSION = 5
PROTOCOL = "dataset-iteration-v5"
FAR_POLICY = "negative-only-recordings-exact-poisson-one-sided95-v1"
APPROX_FAR_POLICY = "negative-only-recordings-iteration-approx95-v1"
DATASET_IDENTITY_POLICY = "effective-domain-keys-verified-afe-v2"
CONFIDENCE = 0.95
MAX_FA_EVENTS = 10000
MAX_POSITIVE_EVENTS = 1000000000
COMPARISON_SCHEMA_VERSION = 3
COMPARISON_POLICY = "research-operating-point-wilson-bonferroni-nominal95-v1"
DEFAULT_FRR_MARGIN = 0.02
FRR_INTERVAL_CONFIDENCE = 0.95
# Two two-sided marginal intervals, each at 97.5%. Their rectangle yields a
# nominal 95% difference interval without assuming the two systems independent.
# Wilson coverage is approximate; this is not an exact joint confidence claim.
FRR_ARM_INTERVAL_CONFIDENCE = 0.975
FRR_BOUND_CONFIDENCE = 0.9875


def positive_counts(metrics: dict, label: str) -> dict:
    """Reconstruct the FRR denominator before drawing an interval from a card."""
    if type(metrics) is not dict:
        raise ValueError(label + ": positive event metrics must be an object")
    names = ("expected", "matched", "false_rejects")
    counts = {name: metrics.get(name) for name in names}
    if any(type(value) is not int or not 0 <= value <= MAX_POSITIVE_EVENTS
           for value in counts.values()):
        raise ValueError(label + ": positive event counts must be bounded non-negative integers")
    if counts["matched"] + counts["false_rejects"] != counts["expected"]:
        raise ValueError(label + ": matched and false rejects must sum to expected events")
    frr = metrics.get("frr")
    expected = counts["false_rejects"] / counts["expected"] if counts["expected"] else 0.0
    if (type(frr) not in (int, float) or not 0 <= frr <= 1
            or not math.isfinite(frr) or frr != expected):
        raise ValueError(label + ": FRR differs from false rejects / expected events")
    return {**counts, "frr": expected}


def frr_comparison(before: dict, after: dict, margin: float) -> dict:
    """Conditional event-level research interval, not independent-human evidence.

    Each arm assumes independent Bernoulli wake events with a common miss
    probability. No independence between arms is assumed, and paired discordance
    information is not inferred from marginal counts. Repeated speakers/sources
    need a separate cluster analysis.
    """
    if type(margin) not in (int, float) or not 0 <= margin <= 1 or not math.isfinite(margin):
        raise ValueError("FRR margin must be finite and in [0,1]")
    baseline = positive_counts(before, "baseline")
    candidate = positive_counts(after, "candidate")

    def interval(counts):
        n, misses = counts["expected"], counts["false_rejects"]
        if not n:
            return None
        return [1.0 - wilson_upper(n - misses, n, FRR_BOUND_CONFIDENCE),
                wilson_upper(misses, n, FRR_BOUND_CONFIDENCE)]

    left, right = interval(baseline), interval(candidate)
    delta = candidate["frr"] - baseline["frr"]
    difference = None if left is None or right is None else [right[0] - left[1], right[1] - left[0]]
    verdict = "inconclusive"
    if difference is not None:
        if difference[1] < -margin:
            verdict = "improved"
        elif difference[0] > margin:
            verdict = "regressed"
        elif difference[0] >= -margin and difference[1] <= margin:
            verdict = "neutral"
    direction = ("not_estimated" if difference is None else
                 "lower" if delta < 0 else "higher" if delta > 0 else "unchanged")
    return {
        "baseline": {**baseline, "interval": left},
        "candidate": {**candidate, "interval": right},
        "delta": delta if difference is not None else None,
        "difference_interval": difference,
        "direction": direction,
        "verdict": verdict,
        "material_margin": margin,
    }


def far_upper_bound_95(false_accepts: int, hours: float) -> float:
    """Exact upper bound; bound work on untrusted scorecard event counts."""
    if type(false_accepts) is not int or not 0 <= false_accepts <= MAX_FA_EVENTS:
        raise ValueError(f"negative false accepts must be an integer in [0,{MAX_FA_EVENTS}]")
    if isinstance(hours, bool):
        raise ValueError("negative exposure must be finite and positive")
    try:
        bound = poisson_rate_upper(false_accepts, hours, CONFIDENCE)
    except (TypeError, OverflowError) as exc:
        raise ValueError("negative exposure must be finite and positive") from exc
    if not math.isfinite(bound):
        raise ValueError("negative FAR upper bound must be finite")
    return bound


def historical_far_upper_bound_95(false_accepts: int, hours: float) -> float:
    """Replay schemas 2--4 unchanged, including their 0-to-1 discontinuity."""
    return (2.9957 / hours if false_accepts == 0 else
            (false_accepts + 1.959964 * math.sqrt(false_accepts)) / hours)
