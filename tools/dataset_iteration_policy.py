"""Versioned statistical policies for dataset-iteration scorecards.

Schemas 2--4 retain their historical approximation for replay only. New cards
and comparisons use the same exact, one-sided Poisson helper as qualification.
"""
from __future__ import annotations

import math

from statistical_bounds import poisson_rate_upper

SCHEMA_VERSION = 5
PROTOCOL = "dataset-iteration-v5"
FAR_POLICY = "negative-only-recordings-exact-poisson-one-sided95-v1"
APPROX_FAR_POLICY = "negative-only-recordings-iteration-approx95-v1"
DATASET_IDENTITY_POLICY = "effective-domain-keys-verified-afe-v2"
CONFIDENCE = 0.95
MAX_FA_EVENTS = 10000


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
