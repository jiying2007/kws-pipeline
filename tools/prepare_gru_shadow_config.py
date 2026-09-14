#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from verify_gru_frozen_candidate import verify  # noqa: E402


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def prepare(candidate: pathlib.Path, output: pathlib.Path) -> dict:
    candidate = candidate.resolve()
    output = output.resolve()
    freeze = verify(candidate)
    config = load_object(candidate / "source-config.json")
    policy = load_object(candidate / "source-development-policy.json")
    candidate_policy = policy.get("candidate_freeze")
    if not isinstance(candidate_policy, dict):
        raise ValueError("source development policy has no candidate_freeze")
    arena_name = str(candidate_policy.get("shadow_arena", ""))
    registry = load_object(ROOT / "experiments" / "model_family" / "shadow_arena_registry.json")
    arenas = {str(row["name"]): row for row in registry.get("arenas", []) if isinstance(row, dict)}
    arena = arenas.get(arena_name)
    if not isinstance(arena, dict):
        raise ValueError(f"shadow arena is not registered: {arena_name}")
    if arena.get("status") != "reserved-untouched":
        raise ValueError("shadow arena is no longer reserved-untouched")
    seeds = [int(value) for value in arena.get("seeds", [])]
    if not 8 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds):
        raise ValueError("shadow arena seed contract is invalid")
    formal = int(config.get("qualification_holdout_seed", -1))
    retired = {int(value) for value in config.get("retired_qualification_holdout_seeds", [])}
    if formal in set(seeds) or retired & set(seeds):
        raise ValueError("shadow arena overlaps formal qualification namespace")

    effective = copy.deepcopy(config)
    raw = effective.get("shadow_qualification")
    if not isinstance(raw, dict):
        raw = {}
        effective["shadow_qualification"] = raw
    raw["enabled"] = True
    raw["seeds"] = seeds
    raw["expected_wakes_per_seed"] = 256
    raw.setdefault("min_surrogate_separation", 0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "policy": "frozen-gru-shadow-arena-binding-v1",
        "candidate_policy": str(freeze["policy"]),
        "model_sha256": str(freeze["model_sha256"]),
        "arena": arena_name,
        "arena_status": str(arena["status"]),
        "seeds": seeds,
        "formal_qualification_used": False,
        "validation_feedback_allowed": False,
        "threshold_feedback_allowed": False,
        "training_rule_feedback_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = prepare(args.candidate, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
