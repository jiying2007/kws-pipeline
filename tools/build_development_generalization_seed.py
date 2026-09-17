#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

SEED_CLASS = "development-generalization-seed-v1"
PLAN_CLASS = "development-generalization-plan-v1"
PLAN_POLICY = "development-generalization-v1"
ROBUSTNESS_CLASS = "synthetic-domain-robustness-matrix"
COVERAGE_FAILURE_REASONS = {
    "missing-slice",
    "insufficient-positive-support",
    "insufficient-negative-recordings",
    "insufficient-negative-hours",
}


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def plan_sha256(value: dict) -> str:
    if int(value.get("schema_version", 0)) != 1 or value.get("evidence_class") != PLAN_CLASS:
        raise ValueError("generalization plan identity mismatch")
    if value.get("policy") != PLAN_POLICY or value.get("evidence_scope") != "development-only":
        raise ValueError("generalization plan policy/scope mismatch")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if value.get(key) is not False:
            raise ValueError(f"generalization plan protected flag {key} must be false")
    if value.get("selection_before_results") is not True:
        raise ValueError("generalization plan must be selected before results")
    expected = value.get("plan_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("generalization plan_sha256 is invalid")
    body = dict(value)
    body.pop("plan_sha256", None)
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("generalization plan_sha256 mismatch")
    return actual


def bind_plan(
    *,
    plan: dict,
    ordinal: int,
    tier: str,
    model_family: str,
    candidate_id: str,
    source_identity: str,
    seed: int,
) -> dict:
    digest = plan_sha256(plan)
    if plan.get("tier") != tier or plan.get("model_family") != model_family or plan.get("candidate_id") != candidate_id:
        raise ValueError("generalization plan tier/family/candidate mismatch")
    entries = plan.get("seed_plan")
    if not isinstance(entries, list) or int(plan.get("independent_seed_count", -1)) != len(entries):
        raise ValueError("generalization plan seed cohort is invalid")
    ordinal = int(ordinal)
    if ordinal < 0 or ordinal >= len(entries):
        raise ValueError("generalization plan ordinal is out of range")
    entry = entries[ordinal]
    if not isinstance(entry, dict) or int(entry.get("ordinal", -1)) != ordinal:
        raise ValueError("generalization plan ordinal identity mismatch")
    if int(entry.get("seed", -1)) != seed or str(entry.get("source_identity", "")) != source_identity:
        raise ValueError("generalization seed/source does not match predeclared plan")
    return {
        "plan_sha256": digest,
        "plan_ordinal": ordinal,
        "plan_model_sha256": str(plan.get("model_sha256", "")),
        "plan_product_head": str(plan.get("product_head", "")),
    }


def non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def positive_float(value: object, label: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def coverage_from_robustness(value: dict) -> tuple[bool, list[dict]]:
    if int(value.get("schema_version", 0)) != 4 or value.get("evidence_class") != ROBUSTNESS_CLASS:
        raise ValueError("robustness evidence identity mismatch")
    if value.get("blocked") is True:
        raise ValueError("robustness evidence is blocked")
    failures = value.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("robustness failures must be a list")
    coverage_failures: list[dict] = []
    for item in failures:
        if not isinstance(item, dict):
            raise ValueError("robustness failure entry must be an object")
        reason = str(item.get("reason", ""))
        if reason in COVERAGE_FAILURE_REASONS:
            coverage_failures.append(dict(item))
    return (not coverage_failures), coverage_failures


def build_seed(
    *,
    domain_metrics: dict,
    robustness: dict,
    tier: str,
    model_family: str,
    candidate_id: str,
    source_identity: str,
    seed: int,
    plan: dict | None = None,
    plan_ordinal: int | None = None,
) -> dict:
    if int(domain_metrics.get("schema_version", 0)) != 6:
        raise ValueError("domain metrics schema_version must be 6")
    if tier not in {"search", "freeze"}:
        raise ValueError("tier must be search/freeze")
    if model_family not in {"rnn", "gru"}:
        raise ValueError("model_family must be rnn/gru")
    if not candidate_id or not source_identity:
        raise ValueError("candidate/source identity must be non-empty")
    seed = non_negative_int(seed, "seed")
    if (plan is None) != (plan_ordinal is None):
        raise ValueError("plan and plan_ordinal must be supplied together")
    plan_binding = {}
    if plan is not None and plan_ordinal is not None:
        plan_binding = bind_plan(
            plan=plan, ordinal=plan_ordinal, tier=tier, model_family=model_family,
            candidate_id=candidate_id, source_identity=source_identity, seed=seed,
        )

    overall = domain_metrics.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("domain metrics overall is missing")
    expected = non_negative_int(overall.get("expected"), "overall.expected")
    false_rejects = non_negative_int(overall.get("false_rejects"), "overall.false_rejects")
    false_accepts = non_negative_int(overall.get("false_accepts"), "overall.false_accepts")
    negative_hours = positive_float(overall.get("negative_audio_hours"), "overall.negative_audio_hours")
    if expected <= 0 or false_rejects > expected:
        raise ValueError("invalid overall wake counts")

    keyword_domains = domain_metrics.get("keyword_domains")
    if not isinstance(keyword_domains, dict) or not keyword_domains:
        raise ValueError("keyword_domains must be non-empty")
    keywords: dict[str, dict] = {}
    for keyword_id, item in sorted(keyword_domains.items()):
        if not isinstance(item, dict) or not isinstance(item.get("overall"), dict):
            raise ValueError(f"keyword {keyword_id} overall metrics are missing")
        metrics = item["overall"]
        k_expected = non_negative_int(metrics.get("expected"), f"keyword {keyword_id}.expected")
        k_fr = non_negative_int(metrics.get("false_rejects"), f"keyword {keyword_id}.false_rejects")
        if k_expected <= 0 or k_fr > k_expected:
            raise ValueError(f"keyword {keyword_id} wake counts are invalid")
        keywords[str(keyword_id)] = {
            "expected_wakes": k_expected,
            "false_rejects": k_fr,
        }

    coverage_passed, coverage_failures = coverage_from_robustness(robustness)
    performance_failures = [
        dict(item)
        for item in robustness.get("failures", [])
        if isinstance(item, dict) and str(item.get("reason", "")) not in COVERAGE_FAILURE_REASONS
    ]

    return {
        "schema_version": 1,
        "evidence_class": SEED_CLASS,
        "evidence_scope": "development-only",
        "tier": tier,
        "model_family": model_family,
        "candidate_id": candidate_id,
        "source_identity": source_identity,
        "seed": seed,
        **plan_binding,
        "expected_wakes": expected,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "negative_audio_hours": negative_hours,
        "coverage_passed": coverage_passed,
        "coverage_failures": coverage_failures,
        "performance_failures_observed": performance_failures,
        "keywords": keywords,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "gate_split": {
            "coverage": "slice-presence-and-support-only-v1",
            "performance": "pooled-multi-seed-statistics-owned-by-development-generalization-v1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one reusable development-generalization seed summary.")
    parser.add_argument("--domain-metrics", required=True, type=pathlib.Path)
    parser.add_argument("--robustness", required=True, type=pathlib.Path)
    parser.add_argument("--tier", required=True, choices=("search", "freeze"))
    parser.add_argument("--model-family", required=True, choices=("rnn", "gru"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--plan-ordinal", type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    plan = load_object(args.plan.resolve()) if args.plan is not None else None
    result = build_seed(
        domain_metrics=load_object(args.domain_metrics),
        robustness=load_object(args.robustness),
        tier=args.tier,
        model_family=args.model_family,
        candidate_id=args.candidate_id,
        source_identity=args.source_identity,
        seed=args.seed,
        plan=plan,
        plan_ordinal=args.plan_ordinal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"development-generalization seed: tier={args.tier} family={args.model_family} "
        f"candidate={args.candidate_id} seed={args.seed} coverage={str(result['coverage_passed']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
