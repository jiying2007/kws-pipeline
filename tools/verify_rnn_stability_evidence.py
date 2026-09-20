#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

FREEZE_POLICY = "rnn-frozen-candidate-v1"
SOURCE_POLICY = "rnn-development-curriculum-loop-v1"
STABILITY_POLICY = "rnn-development-stability-evidence-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def terminal_strict_streak(records: object) -> int:
    if not isinstance(records, list) or not records:
        raise ValueError("RNN stability evidence requires non-empty round_gates")
    streak = 0
    for expected_round, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("RNN stability round gate must be an object")
        round_value = row.get("round")
        # True == 1 and 1.0 == 1, so a bare != lets a boolean or a float stand
        # in for a round number. The GRU twin rejects both explicitly.
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("RNN stability round must be an integer")
        if round_value != expected_round:
            raise ValueError("RNN stability rounds must be contiguous from zero")
        cal = row.get("calibration_gate"); test = row.get("test_gate")
        if not isinstance(cal, bool) or not isinstance(test, bool):
            raise ValueError("RNN stability gate values must be booleans")
        streak = streak + 1 if cal and test else 0
    return streak


def verify(root: pathlib.Path) -> dict:
    root = root.resolve()
    freeze = load_object(root / "freeze-manifest.json")
    policy = load_object(root / "source-development-policy.json")
    selection = load_object(root / "selection-evidence.json")
    stability_path = root / "stability-evidence.json"
    stability = load_object(stability_path)
    if freeze.get("policy") != FREEZE_POLICY or freeze.get("model_family") != "rnn":
        raise ValueError("RNN freeze identity mismatch")
    if policy.get("policy") != SOURCE_POLICY or policy.get("model_family") != "rnn":
        raise ValueError("RNN source policy identity mismatch")
    if stability.get("policy") != STABILITY_POLICY or stability.get("source_policy") != SOURCE_POLICY:
        raise ValueError("RNN stability evidence identity mismatch")
    if stability.get("model_family") != "rnn" or stability.get("development_only") is not True:
        raise ValueError("RNN stability evidence scope mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if stability.get(field) is not False:
            raise ValueError(f"RNN stability evidence requires {field}=false")
    digest = str(freeze.get("stability_evidence_sha256", ""))
    if len(digest) != 64 or sha256_file(stability_path) != digest:
        raise ValueError("RNN stability evidence digest mismatch")
    required = positive_int(policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds")
    observed = terminal_strict_streak(stability.get("round_gates"))
    if observed < required:
        raise ValueError(f"RNN stability streak insufficient: observed={observed} required={required}")
    for label, container in (("stability", stability), ("freeze", freeze), ("selection", selection)):
        if positive_int(container.get("stable_strict_pass_rounds_required"), f"{label}.required") != required:
            raise ValueError(f"{label} RNN required streak mismatch")
        raw = container.get("stable_strict_pass_rounds_observed")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw != observed:
            raise ValueError(f"{label} RNN observed streak mismatch")
    return {
        "verified": True,
        "model_family": "rnn",
        "stable_strict_pass_rounds_required": required,
        "stable_strict_pass_rounds_observed": observed,
        "stability_evidence_sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", required=True, type=pathlib.Path)
    args = parser.parse_args(); print(json.dumps(verify(args.candidate), sort_keys=True)); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr); raise SystemExit(2)
