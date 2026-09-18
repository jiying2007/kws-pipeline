#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from development_generalization_plan import (  # noqa: E402
    HEAD_RE,
    SHA256_RE,
    build_plan,
    load_policy,
    require_sha,
)

BINDING_CLASS = "kws-v2-runnable-candidate-binding-v1"
EVIDENCE_SCOPE = "development-only"
ALLOWED_COHORT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")


def load_binding(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runnable binding must be a JSON object")
    if int(value.get("schema_version", 0)) != 1 or value.get("evidence_class") != BINDING_CLASS:
        raise ValueError("runnable binding identity mismatch")
    if value.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("runnable binding must be development-only")
    if value.get("requires_predeclared_generalization_plan") is not True:
        raise ValueError("runnable binding must require a predeclared generalization plan")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if value.get(key) is not False:
            raise ValueError(f"runnable binding protected flag {key} must be false")

    candidate_id = str(value.get("candidate_id", "")).strip()
    model_family = str(value.get("model_family", "")).strip()
    tier = str(value.get("generalization_tier", "")).strip()
    cohort_id = str(value.get("generalization_cohort_id", "")).strip()
    if not candidate_id:
        raise ValueError("runnable binding candidate_id must be non-empty")
    if model_family not in {"rnn", "gru"}:
        raise ValueError("runnable binding model_family must be rnn/gru")
    if tier not in {"search", "freeze"}:
        raise ValueError("runnable binding generalization_tier must be search/freeze")
    if not cohort_id or len(cohort_id) > 128 or any(ch not in ALLOWED_COHORT_CHARS for ch in cohort_id):
        raise ValueError("runnable binding generalization_cohort_id is invalid")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one development-generalization plan from a KWS v2 runnable candidate binding."
    )
    parser.add_argument("--binding", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--product-head", required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--forbidden-seed", action="append", default=[], type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    binding = load_binding(args.binding.resolve())
    policy, policy_sha = load_policy(args.policy.resolve())
    model_sha = require_sha(args.model_sha256, "model_sha256", SHA256_RE)
    product_head = require_sha(args.product_head, "product_head", HEAD_RE)
    forbidden = {int(value) for value in args.forbidden_seed}
    if any(seed < 0 for seed in forbidden):
        raise ValueError("forbidden seeds must be non-negative")

    plan = build_plan(
        policy=policy,
        policy_sha=policy_sha,
        tier=str(binding["generalization_tier"]),
        candidate_id=str(binding["candidate_id"]),
        model_family=str(binding["model_family"]),
        model_sha=model_sha,
        product_head=product_head,
        count=args.count,
        forbidden=forbidden,
        cohort_id=str(binding["generalization_cohort_id"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"kws-v2 generalization plan: candidate={plan['candidate_id']} "
        f"cohort={plan['cohort_id']} seeds={plan['independent_seed_count']} plan={plan['plan_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
