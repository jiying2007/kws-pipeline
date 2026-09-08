#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    policy = load(args.policy)
    if policy.get("schema_version") != 1:
        raise ValueError("target policy schema_version must be 1")
    summaries = [load(path) for path in args.summary]
    if not summaries:
        raise ValueError("target cohort is empty")
    for index, item in enumerate(summaries):
        if item.get("schema_version") != 1 or item.get("phase") != "physical-target-dut-qualification":
            raise ValueError(f"summary[{index}] is not a target-DUT summary")
        if item.get("qualified") is not True or item.get("shipping_approved") is not False:
            raise ValueError(f"summary[{index}] is not a qualified pre-shipping DUT")

    cohort = policy["cohort"]
    duts = [str(item["dut_id"]) for item in summaries]
    if len(set(duts)) != len(duts):
        raise ValueError("target cohort contains duplicate DUT IDs")
    failures: list[str] = []
    if len(duts) < int(cohort["min_unique_duts"]):
        failures.append("minimum-unique-duts")

    identity_keys = ["deployment_tag", "deployment_target", "human_qualification_tag", "human_corpus_sha256", "final_afe_identity_sha256", "sku", "board_revision"]
    identities = {key: {str(item[key]) for item in summaries} for key in identity_keys}
    required_same = {
        "deployment_tag": bool(cohort["require_same_deployment"]),
        "deployment_target": bool(cohort["require_same_deployment"]),
        "human_qualification_tag": bool(cohort["require_same_phase_a"]),
        "human_corpus_sha256": bool(cohort["require_same_phase_a"]),
        "final_afe_identity_sha256": bool(cohort["require_same_final_afe_identity"]),
        "sku": bool(cohort["require_same_sku"]),
        "board_revision": bool(cohort["require_same_board_revision"]),
    }
    for key, required in required_same.items():
        if required and len(identities[key]) != 1:
            failures.append(f"identity-{key}")

    long_soak = float(cohort["min_long_soak_hours"])
    long_count = sum(1 for item in summaries if float(item["metrics"]["soak_hours"]) >= long_soak)
    if long_count < int(cohort["min_long_soak_duts"]):
        failures.append("minimum-long-soak-duts")

    metrics = {
        "max_p99_process_us": max(float(item["metrics"]["p99_process_us"]) for item in summaries),
        "max_rtf": max(float(item["metrics"]["rtf"]) for item in summaries),
        "min_p99_headroom": min(float(item["metrics"]["p99_headroom"]) for item in summaries),
        "min_soak_hours": min(float(item["metrics"]["soak_hours"]) for item in summaries),
        "max_soak_hours": max(float(item["metrics"]["soak_hours"]) for item in summaries),
        "max_cpu_percent": max(float(item["metrics"]["cpu_percent"]) for item in summaries),
        "max_rss_kib": max(float(item["metrics"]["rss_kib"]) for item in summaries),
        "max_stack_high_water_bytes": max(float(item["metrics"]["stack_high_water_bytes"]) for item in summaries),
        "max_temp_c": max(float(item["metrics"]["max_temp_c"]) for item in summaries),
        "max_average_power_mw": max(float(item["metrics"]["average_power_mw"]) for item in summaries),
    }

    first = summaries[0]
    result = {
        "schema_version": 1,
        "phase": "physical-target-cohort-qualification",
        "qualified": not failures,
        "shipping_approved": False,
        "deployment_tag": first["deployment_tag"],
        "deployment_target": first["deployment_target"],
        "human_qualification_tag": first["human_qualification_tag"],
        "human_corpus_sha256": first["human_corpus_sha256"],
        "final_afe_identity_sha256": first["final_afe_identity_sha256"],
        "sku": first["sku"],
        "board_revision": first["board_revision"],
        "dut_count": len(summaries),
        "dut_ids": sorted(duts),
        "long_soak_duts": long_count,
        "metrics": metrics,
        "dut_summaries": [
            {
                "dut_id": item["dut_id"],
                "sha256": sha256_file(path),
                "target_evidence_sha256": item["evidence_sha256"]["target_evidence"],
                "soak_hours": item["metrics"]["soak_hours"],
            }
            for path, item in zip(args.summary, summaries)
        ],
        "policy_sha256": sha256_file(args.policy),
        "failures": sorted(set(failures)),
        "next_gate": "shipping-approval-promotion" if not failures else "physical-target-cohort-failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
