#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from statistical_bounds import poisson_rate_upper, wilson_upper  # noqa: E402

POLICY = "development-generalization-v1"
SEED_EVIDENCE_CLASS = "development-generalization-seed-v1"
REPORT_EVIDENCE_CLASS = "development-generalization-report-v1"


def _load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_hours(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def load_policy(path: pathlib.Path) -> dict:
    value = _load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY:
        raise ValueError("development generalization policy identity mismatch")
    confidence = float(value.get("confidence", 0.0))
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5,1)")
    tiers = value.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != {"search", "freeze"}:
        raise ValueError("tiers must contain exactly search/freeze")
    for name, tier in tiers.items():
        if not isinstance(tier, dict) or int(tier.get("min_independent_seeds", 0)) <= 0:
            raise ValueError(f"tier {name} min_independent_seeds is invalid")
    if value.get("statistical_gate_mode") != "report-only-until-baseline-calibrated":
        raise ValueError("v1 requires report-only statistical gate mode")
    return value


def validate_seed(path: pathlib.Path, value: dict, expected_tier: str) -> dict:
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if value.get("evidence_class") != SEED_EVIDENCE_CLASS:
        raise ValueError(f"{path}: seed evidence class mismatch")
    if value.get("evidence_scope") != "development-only":
        raise ValueError(f"{path}: evidence_scope must be development-only")
    if value.get("tier") != expected_tier:
        raise ValueError(f"{path}: tier mismatch")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if value.get(key) is not False:
            raise ValueError(f"{path}: protected evidence flag {key} must be false")
    model_family = str(value.get("model_family", ""))
    candidate_id = str(value.get("candidate_id", ""))
    source_identity = str(value.get("source_identity", ""))
    if model_family not in {"rnn", "gru"}:
        raise ValueError(f"{path}: model_family must be rnn/gru")
    if not candidate_id or not source_identity:
        raise ValueError(f"{path}: candidate/source identity must be non-empty")
    seed = _non_negative_int(value.get("seed"), f"{path}.seed")
    expected = _non_negative_int(value.get("expected_wakes"), f"{path}.expected_wakes")
    false_rejects = _non_negative_int(value.get("false_rejects"), f"{path}.false_rejects")
    false_accepts = _non_negative_int(value.get("false_accepts"), f"{path}.false_accepts")
    if expected <= 0 or false_rejects > expected:
        raise ValueError(f"{path}: invalid wake counts")
    negative_hours = _positive_hours(value.get("negative_audio_hours"), f"{path}.negative_audio_hours")
    if value.get("coverage_passed") not in {True, False}:
        raise ValueError(f"{path}: coverage_passed must be boolean")
    keywords = value.get("keywords", {})
    if not isinstance(keywords, dict):
        raise ValueError(f"{path}: keywords must be an object")
    normalized_keywords: dict[str, dict] = {}
    for keyword_id, metrics in keywords.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: keyword metrics must be objects")
        k_expected = _non_negative_int(metrics.get("expected_wakes"), f"{path}.keywords.{keyword_id}.expected_wakes")
        k_fr = _non_negative_int(metrics.get("false_rejects"), f"{path}.keywords.{keyword_id}.false_rejects")
        if k_expected <= 0 or k_fr > k_expected:
            raise ValueError(f"{path}: invalid keyword wake counts")
        normalized_keywords[str(keyword_id)] = {
            "expected_wakes": k_expected,
            "false_rejects": k_fr,
        }
    return {
        "path": path.as_posix(),
        "model_family": model_family,
        "candidate_id": candidate_id,
        "source_identity": source_identity,
        "seed": seed,
        "expected_wakes": expected,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "negative_audio_hours": negative_hours,
        "coverage_passed": bool(value["coverage_passed"]),
        "keywords": normalized_keywords,
    }


def build_report(policy: dict, tier: str, seed_paths: list[pathlib.Path]) -> dict:
    if tier not in policy["tiers"]:
        raise ValueError(f"unknown tier: {tier}")
    seeds = [validate_seed(path, _load_object(path), tier) for path in seed_paths]
    min_seeds = int(policy["tiers"][tier]["min_independent_seeds"])
    if len(seeds) < min_seeds:
        raise ValueError(f"tier {tier} requires at least {min_seeds} seed summaries")
    families = {item["model_family"] for item in seeds}
    candidates = {item["candidate_id"] for item in seeds}
    seed_ids = [item["seed"] for item in seeds]
    sources = [item["source_identity"] for item in seeds]
    if len(families) != 1 or len(candidates) != 1:
        raise ValueError("all seed summaries must refer to one model family/candidate")
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("generalization seed ids must be unique")
    if len(set(sources)) != len(sources):
        raise ValueError("generalization source identities must be unique")

    expected = sum(item["expected_wakes"] for item in seeds)
    false_rejects = sum(item["false_rejects"] for item in seeds)
    false_accepts = sum(item["false_accepts"] for item in seeds)
    negative_hours = sum(item["negative_audio_hours"] for item in seeds)
    confidence = float(policy["confidence"])
    per_seed = []
    for item in seeds:
        per_seed.append(
            {
                "seed": item["seed"],
                "source_identity": item["source_identity"],
                "frr": item["false_rejects"] / item["expected_wakes"],
                "far_per_hour": item["false_accepts"] / item["negative_audio_hours"],
                "coverage_passed": item["coverage_passed"],
            }
        )

    keyword_ids = sorted({key for item in seeds for key in item["keywords"]})
    keyword_report: dict[str, dict] = {}
    for keyword_id in keyword_ids:
        if any(keyword_id not in item["keywords"] for item in seeds):
            raise ValueError(f"keyword {keyword_id} missing from one or more seed summaries")
        k_expected = sum(item["keywords"][keyword_id]["expected_wakes"] for item in seeds)
        k_fr = sum(item["keywords"][keyword_id]["false_rejects"] for item in seeds)
        keyword_report[keyword_id] = {
            "expected_wakes": k_expected,
            "false_rejects": k_fr,
            "frr": k_fr / k_expected,
            "frr_upper_bound": wilson_upper(k_fr, k_expected, confidence),
        }

    return {
        "schema_version": 1,
        "policy": POLICY,
        "evidence_class": REPORT_EVIDENCE_CLASS,
        "evidence_scope": "development-only",
        "tier": tier,
        "model_family": next(iter(families)),
        "candidate_id": next(iter(candidates)),
        "independent_seed_count": len(seeds),
        "minimum_independent_seed_count": min_seeds,
        "seed_ids": sorted(seed_ids),
        "confidence_level": confidence,
        "pooled": {
            "expected_wakes": expected,
            "false_rejects": false_rejects,
            "frr": false_rejects / expected,
            "frr_upper_bound": wilson_upper(false_rejects, expected, confidence),
            "negative_audio_hours": negative_hours,
            "false_accepts": false_accepts,
            "far_per_hour": false_accepts / negative_hours,
            "far_upper_bound_per_hour": poisson_rate_upper(false_accepts, negative_hours, confidence),
        },
        "worst_seed": {
            "frr": max(row["frr"] for row in per_seed),
            "far_per_hour": max(row["far_per_hour"] for row in per_seed),
        },
        "coverage_passed": all(row["coverage_passed"] for row in per_seed),
        "per_seed": sorted(per_seed, key=lambda row: int(row["seed"])),
        "keywords": keyword_report,
        "statistical_gate_mode": policy["statistical_gate_mode"],
        "eligible_for_threshold_calibration": True,
        "protected_evidence_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate reusable development-only generalization seeds.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--tier", required=True, choices=("search", "freeze"))
    parser.add_argument("--seed-summary", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    report = build_report(policy, args.tier, [path.resolve() for path in args.seed_summary])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"development-generalization: tier={args.tier} family={report['model_family']} "
        f"candidate={report['candidate_id']} seeds={report['independent_seed_count']} "
        f"frr_upper={report['pooled']['frr_upper_bound']:.6f} "
        f"far_upper/h={report['pooled']['far_upper_bound_per_hour']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
