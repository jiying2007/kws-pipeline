#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
EVAL = ROOT / "eval"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(EVAL))

from gate_robustness import evaluate as evaluate_robustness  # noqa: E402
from iterate_domain import base_gate, domain_gate, gate_values  # noqa: E402

POLICY = "gru-development-full-robustness-gate-v1"


def evaluate_development_split(base: dict, domains: dict, config: dict) -> dict:
    gates = gate_values(config.get("domain_gates", {}))
    base_domain_qualified = bool(base_gate(base, gates) and domain_gate(domains, gates))
    robustness = evaluate_robustness({"qualification_domains": domains}, config)
    if robustness.get("blocked") is not False:
        raise ValueError("development robustness gate unexpectedly blocked")
    robustness_qualified = robustness.get("qualified")
    # bool("false") is True, so a coercion here would certify a blocked
    # robustness evaluation. kws_landing_status reads this field with `is
    # True`, so the contract already assumes a real boolean.
    if not isinstance(robustness_qualified, bool):
        raise ValueError("robustness qualification verdict must be a boolean")
    return {
        "schema_version": 1,
        "policy": POLICY,
        "development_only": True,
        "base_domain_qualified": base_domain_qualified,
        "robustness_qualified": robustness_qualified,
        "qualified": bool(base_domain_qualified and robustness_qualified),
        "robustness": robustness,
    }


def gate_bool(record: dict, key: str) -> bool:
    """Read a gate that is already a boolean.

    bool("false") is True, so coercing here counts a round that did not pass
    and inflates the streak that qualifies a development candidate.
    """
    value = record.get(key)
    if value is None:
        # A round with no recorded gate did not pass. That is a fact, not
        # an error, and it must not be coerced into a pass either.
        return False
    if not isinstance(value, bool):
        raise ValueError("development record " + key + " must be a boolean")
    return value


def terminal_strict_streak(records: object) -> int:
    if not isinstance(records, list) or not records:
        return 0
    streak = 0
    for expected_round, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("development record must be an object")
        round_value = record.get("round")
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("development record round must be an integer")
        if round_value != expected_round:
            raise ValueError("development records must have contiguous rounds starting at zero")
        if gate_bool(record, "calibration_gate") and gate_bool(record, "test_gate"):
            streak += 1
        else:
            streak = 0
    return streak
