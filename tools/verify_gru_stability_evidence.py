#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

FREEZE_POLICY = "gru-frozen-candidate-v1"
SOURCE_POLICY = "gru-development-curriculum-loop-v1"
STABILITY_POLICY = "gru-development-stability-evidence-v1"


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


def terminal_strict_streak(records: object) -> int:
    if not isinstance(records, list) or not records:
        raise ValueError("stability evidence requires non-empty round_gates")
    streak = 0
    for expected_round, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("stability round gate must be an object")
        round_value = row.get("round")
        if isinstance(round_value, bool) or not isinstance(round_value, int):
            raise ValueError("stability round must be an integer")
        if round_value != expected_round:
            raise ValueError("stability rounds must be contiguous starting at zero")
        calibration_gate = row.get("calibration_gate")
        test_gate = row.get("test_gate")
        if not isinstance(calibration_gate, bool) or not isinstance(test_gate, bool):
            raise ValueError("stability gate values must be booleans")
        if calibration_gate and test_gate:
            streak += 1
        else:
            streak = 0
    return streak


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def verify(root: pathlib.Path) -> dict:
    root = root.resolve()
    freeze = load_object(root / "freeze-manifest.json")
    policy = load_object(root / "source-development-policy.json")
    selection = load_object(root / "selection-evidence.json")
    stability_path = root / "stability-evidence.json"
    stability = load_object(stability_path)

    if freeze.get("policy") != FREEZE_POLICY:
        raise ValueError("freeze policy identity mismatch")
    if policy.get("policy") != SOURCE_POLICY:
        raise ValueError("source development policy identity mismatch")
    if stability.get("policy") != STABILITY_POLICY:
        raise ValueError("stability evidence policy identity mismatch")
    if stability.get("source_policy") != SOURCE_POLICY:
        raise ValueError("stability evidence source policy mismatch")
    if stability.get("development_only") is not True:
        raise ValueError("stability evidence must be development-only")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if stability.get(field) is not False:
            raise ValueError(f"stability evidence requires {field}=false")

    digest = str(freeze.get("stability_evidence_sha256", ""))
    if len(digest) != 64 or sha256_file(stability_path) != digest:
        raise ValueError("stability evidence digest mismatch")

    required = positive_int(policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds")
    observed = terminal_strict_streak(stability.get("round_gates"))
    if observed < required:
        raise ValueError(
            "stability evidence does not meet required terminal strict-pass streak: "
            f"observed={observed} required={required}"
        )

    for container_name, container in (
        ("stability evidence", stability),
        ("freeze manifest", freeze),
        ("selection evidence", selection),
    ):
        if positive_int(
            container.get("stable_strict_pass_rounds_required"),
            f"{container_name} stable_strict_pass_rounds_required",
        ) != required:
            raise ValueError(f"{container_name} required stable streak mismatch")
        raw_observed = container.get("stable_strict_pass_rounds_observed")
        if isinstance(raw_observed, bool) or not isinstance(raw_observed, int) or raw_observed < 0:
            raise ValueError(f"{container_name} observed stable streak must be a non-negative integer")
        if raw_observed != observed:
            raise ValueError(f"{container_name} observed stable streak mismatch")

    return {
        "verified": True,
        "stable_strict_pass_rounds_required": required,
        "stable_strict_pass_rounds_observed": observed,
        "round_count": len(stability["round_gates"]),
        "stability_evidence_sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.candidate), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
