#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_kws_v2_stage_a_search import (  # noqa: E402
    EXPECTED_STAGE_A,
    SHARD_CLASS,
    STAGE_CLASS,
    load_object,
    write_json,
)

EXPECTED_CANDIDATE_COUNT = 4


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def require_sha(value: object, label: str) -> str:
    result = require_text(value, label).lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be lowercase sha256")
    return result


def load_shard(path: pathlib.Path) -> dict:
    value = load_object(path.resolve())
    if int(value.get("schema_version", 0)) != 1 or value.get("evidence_class") != SHARD_CLASS:
        raise ValueError(f"{path}: Stage A shard identity mismatch")
    if value.get("evidence_scope") != "development-only" or value.get("stage") != "A":
        raise ValueError(f"{path}: shard must be development-only Stage A")
    if value.get("status") != "complete":
        raise ValueError(f"{path}: shard is not complete")
    if value.get("generalization_tier") != "search":
        raise ValueError(f"{path}: shard generalization tier must be search")
    if value.get("full_training_round_budget") is not True:
        raise ValueError(f"{path}: shard did not preserve the full training budget")
    for key in (
        "fresh_used",
        "shadow_used",
        "formal_qualification_used",
        "protected_evidence_used",
    ):
        if value.get(key) is not False:
            raise ValueError(f"{path}: protected flag {key} must be false")
    require_sha(value.get("product_head"), f"{path}: product_head")
    require_sha(value.get("matrix_sha256"), f"{path}: matrix_sha256")
    require_sha(
        value.get("external_base_bundle_sha256"),
        f"{path}: external_base_bundle_sha256",
    )
    require_sha(
        value.get("generalization_policy_sha256"),
        f"{path}: generalization_policy_sha256",
    )
    require_text(value.get("generalization_cohort_id"), f"{path}: cohort")
    candidate = value.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: candidate result is missing")
    if candidate.get("status") != "complete":
        raise ValueError(f"{path}: candidate result is not complete")
    return value


def scorecard(candidate: dict) -> dict:
    metrics = candidate.get("generalization_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{candidate.get('candidate_id')}: generalization_metrics missing")
    pooled = metrics.get("pooled")
    worst = metrics.get("worst_seed")
    if not isinstance(pooled, dict) or not isinstance(worst, dict):
        raise ValueError(f"{candidate.get('candidate_id')}: pooled/worst metrics missing")
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "model_family": str(candidate["model_family"]),
        "frontend": str(candidate["frontend"]),
        "hidden_dim": int(candidate["hidden_dim"]),
        "resource_contract_candidate": str(candidate["resource_contract_candidate"]),
        "frozen_model_sha256": require_sha(
            candidate.get("frozen_model_sha256"),
            f"{candidate.get('candidate_id')}: frozen_model_sha256",
        ),
        "generalization_plan_sha256": require_sha(
            candidate.get("generalization_plan_sha256"),
            f"{candidate.get('candidate_id')}: generalization_plan_sha256",
        ),
        "independent_seed_count": int(candidate["independent_seed_count"]),
        "pooled_frr": float(pooled["frr"]),
        "pooled_frr_upper_bound": float(pooled["frr_upper_bound"]),
        "pooled_far_per_hour": float(pooled["far_per_hour"]),
        "pooled_far_upper_bound_per_hour": float(pooled["far_upper_bound_per_hour"]),
        "worst_seed_frr": float(worst["frr"]),
        "worst_seed_far_per_hour": float(worst["far_per_hour"]),
        "coverage_passed": bool(metrics["coverage_passed"]),
        "eligible_for_threshold_calibration": bool(
            metrics["eligible_for_threshold_calibration"]
        ),
    }


def aggregate(paths: list[pathlib.Path]) -> dict:
    if len(paths) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"Stage A aggregation requires exactly {EXPECTED_CANDIDATE_COUNT} shards"
        )
    shards = [load_shard(path) for path in paths]
    common_fields = (
        "product_head",
        "matrix_sha256",
        "generalization_cohort_id",
        "external_base_bundle_sha256",
        "generalization_policy_sha256",
    )
    for field in common_fields:
        values = {str(shard[field]) for shard in shards}
        if len(values) != 1:
            raise ValueError(f"Stage A shards disagree on {field}")

    candidates = [dict(shard["candidate"]) for shard in shards]
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if len(set(candidate_ids)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Stage A shard candidate ids must be unique")
    identities = {
        (
            str(row.get("model_family", "")),
            str(row.get("frontend", "")),
            int(row.get("hidden_dim", 0)),
        )
        for row in candidates
    }
    if identities != EXPECTED_STAGE_A:
        raise ValueError(f"Stage A shard candidate identity drifted: {sorted(identities)}")

    seed_plans = [row.get("generalization_seed_plan") for row in candidates]
    if not all(isinstance(plan, list) and plan for plan in seed_plans):
        raise ValueError("Stage A shards must contain non-empty generalization seed plans")
    if any(plan != seed_plans[0] for plan in seed_plans[1:]):
        raise ValueError("Stage A shards did not use the identical generalization seed cohort")
    seed_count = len(seed_plans[0])
    if seed_count < 8:
        raise ValueError("Stage A shared generalization cohort must contain at least eight seeds")
    if any(int(row.get("independent_seed_count", -1)) != seed_count for row in candidates):
        raise ValueError("Stage A shard independent_seed_count mismatch")
    if any(row.get("generalization_plan_complete") is not True for row in candidates):
        raise ValueError("Stage A shard generalization plan is incomplete")

    plan_shas = {
        require_sha(
            row.get("generalization_plan_sha256"),
            f"{row.get('candidate_id')}: generalization_plan_sha256",
        )
        for row in candidates
    }
    if len(plan_shas) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Stage A candidate-specific generalization plan hashes must be unique")

    candidates.sort(key=lambda row: str(row["candidate_id"]))
    table = [scorecard(row) for row in candidates]
    first = shards[0]
    return {
        "schema_version": 1,
        "evidence_class": STAGE_CLASS,
        "evidence_scope": "development-only",
        "stage": "A",
        "status": "complete",
        "execution_mode": "sharded-v1",
        "product_head": str(first["product_head"]),
        "matrix_sha256": str(first["matrix_sha256"]),
        "generalization_cohort_id": str(first["generalization_cohort_id"]),
        "generalization_tier": "search",
        "external_base_bundle_sha256": str(first["external_base_bundle_sha256"]),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "candidates": candidates,
        "scorecard": table,
        "full_training_round_budget": True,
        "generalization_policy_sha256": str(first["generalization_policy_sha256"]),
        "shared_seed_plan": seed_plans[0],
        "all_candidates_complete": True,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "protected_evidence_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate four independently executed KWS v2 Stage A candidate shards."
    )
    parser.add_argument("--shard", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = aggregate(args.shard)
    write_json(args.output.resolve(), result)
    print(
        "kws-v2-stage-a-aggregate: "
        f"candidates={result['candidate_count']} "
        f"cohort={result['generalization_cohort_id']} "
        f"seeds={len(result['shared_seed_plan'])} "
        f"bundle={result['external_base_bundle_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
