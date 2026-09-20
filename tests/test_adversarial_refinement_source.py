#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from adversarial_refinement import (  # noqa: E402
    REFINEMENT_SOURCE_POLICY,
    select_refinement_source,
)


def row(
    round_index: int,
    *,
    cal_frr: float,
    test_frr: float,
    cal_far_frr: float,
    test_far_frr: float,
    cal_far: float,
    test_far: float,
    score: float,
    strict: bool = False,
) -> dict:
    return {
        "round": round_index,
        "frontend": "logmel",
        "score": score,
        "checkpoint": f"/tmp/round-{round_index}.pt",
        "calibration": {"frr": cal_frr, "far_per_hour": cal_far},
        "test": {"frr": test_frr, "far_per_hour": test_far},
        "calibration_domains": {
            "domains": {"distance:far": {"frr": cal_far_frr}}
        },
        "test_domains": {
            "domains": {"distance:far": {"frr": test_far_frr}}
        },
        "calibration_gate": strict,
        "test_gate": strict,
    }


def main() -> int:
    # Exact rounded operating-point shape from governed run 35447750283.
    # Round 2 wins the old zero-gate objective mainly by rejecting almost all
    # wakes. Refinement should instead start from the development checkpoint
    # that preserves the best worst-case recall, with FAR used only after
    # base/far-field recall.
    records = [
        row(
            0,
            cal_frr=0.84375,
            test_frr=0.84375,
            cal_far_frr=0.9230769231,
            test_far_frr=0.6153846154,
            cal_far=928.5468,
            test_far=1066.4509,
            score=235595.94,
        ),
        row(
            1,
            cal_frr=0.71875,
            test_frr=0.90625,
            cal_far_frr=0.6153846154,
            test_far_frr=0.8461538462,
            cal_far=790.6438,
            test_far=750.4654,
            score=188598.69,
        ),
        row(
            2,
            cal_frr=0.96875,
            test_frr=0.984375,
            cal_far_frr=0.9230769231,
            test_far_frr=1.0,
            cal_far=64.3547,
            test_far=98.7455,
            score=59537.30,
        ),
        row(
            3,
            cal_frr=0.875,
            test_frr=0.90625,
            cal_far_frr=0.8461538462,
            test_far_frr=0.8461538462,
            cal_far=524.0314,
            test_far=533.2254,
            score=144543.13,
        ),
    ]
    manifest = {
        "development_qualified": False,
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [],
            "qualification_used_for_selection": False,
            "selected_round": None,
            "selected_frontend": None,
            "selected_score": None,
            "objective_fallback_used": True,
            "objective_best_round": 2,
            "objective_best_frontend": "logmel",
            "objective_best_score": 59537.30,
        },
        "records": records,
    }
    selected, policy = select_refinement_source(manifest)
    assert selected["round"] == 1
    assert policy == REFINEMENT_SOURCE_POLICY

    strict = row(
        4,
        cal_frr=0.0,
        test_frr=0.0,
        cal_far_frr=0.0,
        test_far_frr=0.0,
        cal_far=0.0,
        test_far=0.0,
        score=0.0,
        strict=True,
    )
    strict_manifest = {
        "development_qualified": True,
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [4],
            "qualification_used_for_selection": False,
            "selected_round": 4,
            "selected_frontend": "logmel",
            "selected_score": 0.0,
            "objective_fallback_used": False,
        },
        "records": records + [strict],
    }
    selected, policy = select_refinement_source(strict_manifest)
    assert selected["round"] == 4
    assert policy == "strict-development-candidate"

    leaked = dict(manifest)
    leaked["candidate_selection"] = dict(manifest["candidate_selection"])
    leaked["candidate_selection"]["qualification_used_for_selection"] = True
    try:
        select_refinement_source(leaked)
    except ValueError as exc:
        assert "must not use qualification" in str(exc)
    else:
        raise AssertionError("qualification-backed refinement source was accepted")

    print("adversarial refinement source selection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
