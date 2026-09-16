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
from verify_rnn_frozen_candidate import verify  # noqa: E402


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def prepare(candidate: pathlib.Path, output: pathlib.Path) -> dict:
    candidate = candidate.resolve(); output = output.resolve()
    freeze = verify(candidate)
    config = load_object(candidate / "source-config.json")
    policy = load_object(candidate / "source-development-policy.json")
    candidate_policy = policy.get("candidate_freeze")
    if not isinstance(candidate_policy, dict): raise ValueError("RNN source development policy has no candidate_freeze")
    arena_name = str(candidate_policy.get("shadow_arena", ""))
    registry_path = ROOT / str(candidate_policy.get("shadow_registry", "experiments/model_family/shadow_arena_registry.json"))
    registry = load_object(registry_path)
    rows = [row for row in registry.get("arenas", []) if isinstance(row, dict)]
    arenas = {str(row["name"]): row for row in rows}; arena = arenas.get(arena_name)
    if not isinstance(arena, dict) or arena.get("model_family") != "rnn": raise ValueError(f"RNN shadow arena is not registered: {arena_name}")
    if arena.get("status") != "reserved-untouched": raise ValueError("RNN shadow arena is no longer reserved-untouched")
    candidate_model_sha256 = str(freeze["model_sha256"])
    for row in rows:
        if str(row.get("name", "")) == arena_name or row.get("model_family") != "rnn" or row.get("status") == "reserved-untouched": continue
        if str(row.get("candidate_model_sha256", "")) == candidate_model_sha256: raise ValueError("RNN model already consumed by an earlier shadow arena")
    seeds = [int(value) for value in arena.get("seeds", [])]
    if not 8 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds): raise ValueError("RNN shadow arena seed contract is invalid")
    gru_formal = int(config.get("qualification_holdout_seed", -1))
    rnn_formal = int(candidate_policy.get("formal_qualification_seed", -1))
    retired = {int(value) for value in config.get("retired_qualification_holdout_seeds", [])}
    protected_formal = retired | {gru_formal, rnn_formal}
    if rnn_formal <= 0 or rnn_formal == gru_formal: raise ValueError("RNN formal seed is not independently reserved")
    if protected_formal & set(seeds): raise ValueError("RNN shadow arena overlaps formal qualification namespace")
    effective = copy.deepcopy(config)
    raw = effective.get("shadow_qualification")
    if not isinstance(raw, dict): raw = {}; effective["shadow_qualification"] = raw
    raw["enabled"] = True; raw["seeds"] = seeds; raw["expected_wakes_per_seed"] = 256
    raw.setdefault("min_surrogate_separation", 0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"schema_version": 1, "policy": "frozen-rnn-shadow-arena-binding-v1", "model_family": "rnn", "candidate_policy": str(freeze["policy"]), "model_sha256": candidate_model_sha256, "arena": arena_name, "arena_status": str(arena["status"]), "seeds": seeds, "rnn_formal_seed_reserved": rnn_formal, "gru_formal_seed_untouched": gru_formal, "formal_qualification_used": False, "validation_feedback_allowed": False, "threshold_feedback_allowed": False, "training_rule_feedback_allowed": False}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", required=True, type=pathlib.Path); parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(); print(json.dumps(prepare(args.candidate, args.output), ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
