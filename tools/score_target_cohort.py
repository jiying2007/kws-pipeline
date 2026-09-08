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


def current_policy_failures(item: dict, policy: dict) -> list[str]:
    dut = str(item["dut_id"])
    metrics = item.get("metrics")
    continuity = item.get("continuity")
    budget = item.get("resource_budget")
    if not isinstance(metrics, dict) or not isinstance(continuity, dict) or not isinstance(budget, dict):
        raise ValueError(f"DUT {dut} summary is missing metrics/continuity/resource budget")
    limits = budget.get("limits")
    if not isinstance(limits, dict):
        raise ValueError(f"DUT {dut} resource budget limits are missing")
    hard = policy["per_dut_hard_gates"]
    failures: list[str] = []
    if float(metrics["soak_hours"]) < float(hard["min_soak_hours"]):
        failures.append(f"dut-{dut}-soak-hours")
    if hard.get("require_p99_below_block_deadline") is True and not (
        float(metrics["p99_process_us"]) < float(metrics["block_deadline_us"])
    ):
        failures.append(f"dut-{dut}-p99-block-deadline")
    if hard.get("require_rtf_below_realtime") is True and not float(metrics["rtf"]) < 1.0:
        failures.append(f"dut-{dut}-rtf-realtime")
    if hard.get("require_p99_headroom_above_one") is True and not (
        float(metrics["p99_headroom"]) > 1.0
    ):
        failures.append(f"dut-{dut}-p99-headroom")
    continuity_limits = {
        "xrun_count": "max_xrun_count",
        "discontinuity_count": "max_discontinuity_count",
        "lost_samples": "max_lost_samples",
        "backpressure_count": "max_backpressure_count",
    }
    for measured, limit in continuity_limits.items():
        if int(continuity[measured]) > int(hard[limit]):
            failures.append(f"dut-{dut}-{measured}")
    resources = (
        ("cpu_percent", "max_cpu_percent"),
        ("rss_kib", "max_rss_kib"),
        ("stack_high_water_bytes", "max_stack_high_water_bytes"),
        ("max_temp_c", "max_temp_c"),
        ("average_power_mw", "max_average_power_mw"),
    )
    for measured, limit in resources:
        if float(metrics[measured]) > float(limits[limit]):
            failures.append(f"dut-{dut}-{measured}-budget")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    policy = load(args.policy)
    if policy.get("schema_version") != 2:
        raise ValueError("target policy schema_version must be 2")
    summaries = [load(path) for path in args.summary]
    if not summaries:
        raise ValueError("target cohort is empty")
    for index, item in enumerate(summaries):
        if item.get("schema_version") != 1 or item.get("phase") != "physical-target-dut-qualification":
            raise ValueError(f"summary[{index}] is not a target-DUT summary")
        if item.get("qualified") is not True or item.get("shipping_approved") is not False:
            raise ValueError(f"summary[{index}] is not a qualified pre-shipping DUT")
        budget = item.get("resource_budget")
        if not isinstance(budget, dict) or not budget.get("budget_id") or not budget.get("sha256"):
            raise ValueError(f"summary[{index}] is missing approved resource budget identity")

    cohort = policy["cohort"]
    duts = [str(item["dut_id"]) for item in summaries]
    if len(set(duts)) != len(duts):
        raise ValueError("target cohort contains duplicate DUT IDs")
    evidence_hashes = [str(item["evidence_sha256"]["target_evidence"]) for item in summaries]
    if len(set(evidence_hashes)) != len(evidence_hashes):
        raise ValueError("target cohort reuses one physical evidence object across multiple DUTs")

    failures: list[str] = []
    for item in summaries:
        failures.extend(current_policy_failures(item, policy))
    if len(duts) < int(cohort["min_unique_duts"]):
        failures.append("minimum-unique-duts")

    identity_values = {
        "deployment_tag": [str(item["deployment_tag"]) for item in summaries],
        "deployment_target": [str(item["deployment_target"]) for item in summaries],
        "human_qualification_tag": [str(item["human_qualification_tag"]) for item in summaries],
        "human_corpus_sha256": [str(item["human_corpus_sha256"]) for item in summaries],
        "final_afe_identity_sha256": [str(item["final_afe_identity_sha256"]) for item in summaries],
        "sku": [str(item["sku"]) for item in summaries],
        "board_revision": [str(item["board_revision"]) for item in summaries],
        "resource_budget_id": [str(item["resource_budget"]["budget_id"]) for item in summaries],
        "resource_budget_sha256": [str(item["resource_budget"]["sha256"]) for item in summaries],
        "measurement_contract_id": [str(item["resource_budget"]["measurement_contract_id"]) for item in summaries],
    }
    identities = {key: set(values) for key, values in identity_values.items()}
    required_same = {
        "deployment_tag": bool(cohort["require_same_deployment"]),
        "deployment_target": bool(cohort["require_same_deployment"]),
        "human_qualification_tag": bool(cohort["require_same_phase_a"]),
        "human_corpus_sha256": bool(cohort["require_same_phase_a"]),
        "final_afe_identity_sha256": bool(cohort["require_same_final_afe_identity"]),
        "sku": bool(cohort["require_same_sku"]),
        "board_revision": bool(cohort["require_same_board_revision"]),
        "resource_budget_id": bool(cohort["require_same_resource_budget"]),
        "resource_budget_sha256": bool(cohort["require_same_resource_budget"]),
        "measurement_contract_id": bool(cohort["require_same_resource_budget"]),
    }
    for key, required in required_same.items():
        if required and len(identities[key]) != 1:
            failures.append(f"identity-{key}")

    long_soak = float(cohort["min_long_soak_hours"])
    long_count = sum(
        1 for item in summaries if float(item["metrics"]["soak_hours"]) >= long_soak
    )
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
        "resource_budget_id": first["resource_budget"]["budget_id"],
        "resource_budget_sha256": first["resource_budget"]["sha256"],
        "measurement_contract_id": first["resource_budget"]["measurement_contract_id"],
        "dut_count": len(summaries),
        "dut_ids": sorted(duts),
        "long_soak_duts": long_count,
        "metrics": metrics,
        "dut_summaries": [
            {
                "dut_id": item["dut_id"],
                "sha256": sha256_file(path),
                "target_evidence_sha256": item["evidence_sha256"]["target_evidence"],
                "resource_budget_sha256": item["resource_budget"]["sha256"],
                "soak_hours": item["metrics"]["soak_hours"],
            }
            for path, item in zip(args.summary, summaries)
        ],
        "policy_sha256": sha256_file(args.policy),
        "failures": sorted(set(failures)),
        "next_gate": "shipping-approval-promotion" if not failures else "physical-target-cohort-failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
