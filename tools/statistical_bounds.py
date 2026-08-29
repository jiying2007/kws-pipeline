from __future__ import annotations

import math
from statistics import NormalDist


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not math.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError("confidence must be finite and in (0.5,1)")
    return value


def wilson_upper(successes: int, trials: int, confidence: float) -> float:
    """One-sided Wilson upper confidence bound for a binomial rate."""
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("binomial counts must be integers")
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    confidence = _validate_confidence(confidence)
    z = NormalDist().inv_cdf(confidence)
    z2 = z * z
    p = successes / trials
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return min(1.0, max(0.0, center + margin))


def _poisson_cdf(count: int, mean: float) -> float:
    if count < 0 or mean < 0.0 or not math.isfinite(mean):
        raise ValueError("invalid Poisson arguments")
    if mean == 0.0:
        return 1.0
    # Qualification FAR counts are intentionally kept low. Evaluate the CDF via
    # a log-domain recurrence so even large exposure windows do not underflow.
    logs = []
    log_term = -mean
    logs.append(log_term)
    for k in range(1, count + 1):
        log_term += math.log(mean) - math.log(k)
        logs.append(log_term)
    peak = max(logs)
    if peak < -745.0:
        return 0.0
    total = sum(math.exp(value - peak) for value in logs)
    return min(1.0, math.exp(peak) * total)


def poisson_rate_upper(events: int, exposure_hours: float, confidence: float) -> float:
    """Exact one-sided Poisson upper bound divided by exposure hours."""
    if isinstance(events, bool):
        raise ValueError("Poisson event count must be an integer")
    events = int(events)
    exposure_hours = float(exposure_hours)
    confidence = _validate_confidence(confidence)
    if events < 0 or not math.isfinite(exposure_hours) or exposure_hours <= 0.0:
        raise ValueError("invalid Poisson count/exposure")
    alpha = 1.0 - confidence
    if events == 0:
        mean_upper = -math.log(alpha)
        return mean_upper / exposure_hours

    # Solve P(X <= observed | mean_upper) = alpha. The CDF decreases monotonically
    # with mean, so bounded bisection is deterministic and dependency-free.
    low = float(events)
    high = max(1.0, float(events + 1))
    while _poisson_cdf(events, high) > alpha:
        high *= 2.0
        if high > 1.0e9:
            raise ValueError("failed to bracket Poisson upper bound")
    for _ in range(96):
        mid = 0.5 * (low + high)
        if _poisson_cdf(events, mid) > alpha:
            low = mid
        else:
            high = mid
    return high / exposure_hours


def qualification_bounds(
    *,
    false_rejects: int,
    expected_wakes: int,
    false_accepts: int,
    audio_hours: float,
    confidence: float,
) -> dict[str, float]:
    confidence = _validate_confidence(confidence)
    return {
        "confidence_level": confidence,
        "frr_upper_bound": wilson_upper(false_rejects, expected_wakes, confidence),
        "far_upper_bound_per_hour": poisson_rate_upper(
            false_accepts, audio_hours, confidence
        ),
    }
