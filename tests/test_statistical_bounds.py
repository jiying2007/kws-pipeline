#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from statistical_bounds import poisson_rate_upper, qualification_bounds, wilson_upper  # noqa: E402


def main() -> int:
    # Zero observed false accepts is not zero true FAR. At 95% confidence the
    # one-sided Poisson upper count is -ln(0.05) ~= 2.9957 events.
    zero_far = poisson_rate_upper(0, 24.0, 0.95)
    assert math.isclose(zero_far, -math.log(0.05) / 24.0, rel_tol=1e-12)
    assert zero_far > 0.12

    frr = wilson_upper(1, 10, 0.95)
    assert 0.2 < frr < 0.5
    assert wilson_upper(0, 1000, 0.95) > 0.0

    bounds = qualification_bounds(
        false_rejects=1,
        expected_wakes=10,
        false_accepts=1,
        audio_hours=24.0,
        confidence=0.95,
    )
    assert bounds["confidence_level"] == 0.95
    assert bounds["frr_upper_bound"] > 0.1
    assert bounds["far_upper_bound_per_hour"] > 1.0 / 24.0

    try:
        poisson_rate_upper(0, 0.0, 0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("zero exposure must fail")

    print("test_statistical_bounds: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
