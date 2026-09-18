#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

POLICY = "development-generalization-v1"
PLAN_CLASS = "development-generalization-plan-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
HEAD_RE = re.compile(r"[0-9a-f]{40}")


def load_policy(path: pathlib.Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("generalization policy must be schema_version 1")
    if value.get("policy") != POLICY or value.get("evidence_scope") != "development-only":
        raise ValueError("generalization policy identity mismatch")
    tiers = value.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != {"search", "freeze"}:
        raise ValueError("policy tiers must contain search/freeze")
    for name, tier in tiers.items():
        if not isinstance(tier, dict):
            raise ValueError(f"tier {name} must be an object")
        minimum = int(tier.get("min_independent_seeds", 0))
        recommended = int(tier.get("recommended_independent_seeds", 0))
        maximum = int(tier.get("max_independent_seeds", 0))
        if minimum <= 0 or recommended < minimum or maximum < recommended or maximum > 32:
            raise ValueError(f"tier {name} seed-count contract is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def require_sha(value: str, label: str, pattern: re.Pattern[str]) -> str:
    value = value.strip().lower()
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{label} has invalid digest format")
    return value


def derive_seed(material: str) -> tuple[int, str]:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    # Keep acoustic seeds in signed 31-bit positive range for broad runtime compatibility.
    seed = 1 + (int(digest[:16], 16) % 2_147_483_646)
    return seed, digest


def build_plan(
    *,
    policy: dict,
    policy_sha: str,
    tier: str,
    candidate_id: str,
    model_family: str,
    model_sha: str,
    product_head: str,
    count: int | None,
    forbidden: set[int],
    cohort_id: str | None = None,
) -> dict:
    if tier not in policy["tiers"]:
        raise ValueError(f"unsupported tier: {tier}")
    if model_family not in {"rnn", "gru"}:
        raise ValueError("model_family must be rnn/gru")
    if not candidate_id.strip():
        raise ValueError("candidate_id must be non-empty")
    cohort = None
    if cohort_id is not None:
        cohort = cohort_id.strip()
        if not cohort or len(cohort) > 128:
            raise ValueError("cohort_id must be non-empty and <=128 characters")
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for ch in cohort):
            raise ValueError("cohort_id contains unsupported characters")
    tier_policy = policy["tiers"][tier]
    minimum = int(tier_policy["min_independent_seeds"])
    recommended = int(tier_policy["recommended_independent_seeds"])
    maximum = int(tier_policy["max_independent_seeds"])
    count = recommended if count is None else int(count)
    if count < minimum or count > maximum:
        raise ValueError(f"tier {tier} seed count must be in [{minimum},{maximum}]")

    entries: list[dict] = []
    seen: set[int] = set()
    for ordinal in range(count):
        if cohort is None:
            material = ":".join(
                [POLICY, policy_sha, tier, model_family, candidate_id, model_sha, product_head, str(ordinal)]
            )
        else:
            material = ":".join(
                [POLICY, policy_sha, tier, "shared-cohort-v1", cohort, product_head, str(ordinal)]
            )
        seed, digest = derive_seed(material)
        if seed in forbidden:
            raise ValueError(f"derived seed collides with forbidden namespace: {seed}")
        if seed in seen:
            raise ValueError(f"derived seed collision within cohort: {seed}")
        seen.add(seed)
        entries.append(
            {
                "ordinal": ordinal,
                "seed": seed,
                "source_identity": f"dev-generalization:{tier}:{digest[:24]}",
                "derivation_sha256": digest,
            }
        )

    body = {
        "schema_version": 1,
        "policy": POLICY,
        "policy_sha256": policy_sha,
        "evidence_class": PLAN_CLASS,
        "evidence_scope": "development-only",
        "tier": tier,
        "model_family": model_family,
        "candidate_id": candidate_id,
        "model_sha256": model_sha,
        "product_head": product_head,
        "independent_seed_count": len(entries),
        "minimum_independent_seed_count": minimum,
        "recommended_independent_seed_count": recommended,
        "maximum_independent_seed_count": maximum,
        "seed_plan": entries,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "selection_before_results": True,
    }
    if cohort is not None:
        body["cohort_id"] = cohort
        body["seed_derivation_policy"] = "shared-cohort-v1"
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description="Predeclare deterministic development-generalization seed cohorts.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--tier", required=True, choices=("search", "freeze"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-family", required=True, choices=("rnn", "gru"))
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--product-head", required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--cohort-id")
    parser.add_argument("--forbidden-seed", action="append", default=[], type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    policy, policy_sha = load_policy(args.policy.resolve())
    model_sha = require_sha(args.model_sha256, "model_sha256", SHA256_RE)
    product_head = require_sha(args.product_head, "product_head", HEAD_RE)
    forbidden = {int(value) for value in args.forbidden_seed}
    if any(seed < 0 for seed in forbidden):
        raise ValueError("forbidden seeds must be non-negative")
    plan = build_plan(
        policy=policy,
        policy_sha=policy_sha,
        tier=args.tier,
        candidate_id=args.candidate_id.strip(),
        model_family=args.model_family,
        model_sha=model_sha,
        product_head=product_head,
        count=args.count,
        forbidden=forbidden,
        cohort_id=args.cohort_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"development-generalization-plan: tier={args.tier} family={args.model_family} "
        f"candidate={args.candidate_id} cohort={plan.get('cohort_id', 'candidate-bound')} "
        f"seeds={plan['independent_seed_count']} plan={plan['plan_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
