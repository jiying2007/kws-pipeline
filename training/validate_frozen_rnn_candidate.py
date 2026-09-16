#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

from iterate_domain import base_gate, domain_gate, evaluate, gate_values, sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from verify_rnn_frozen_candidate import verify  # noqa: E402

POLICY = "frozen-rnn-fresh-validation-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def evaluation_hashes(index_path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    for line_no, raw in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{index_path}:{line_no}: expected object")
        if str(row.get("split")) not in {"calibration", "test", "qualification"}:
            continue
        digest = str(row.get("wav_sha256") or "")
        if len(digest) != 64:
            raise ValueError(f"{index_path}:{line_no}: missing WAV SHA")
        result.add(digest)
    return result


def validate(candidate: pathlib.Path, runner: pathlib.Path, output: pathlib.Path) -> dict:
    candidate = candidate.resolve(); runner = runner.resolve(); output = output.resolve()
    freeze = verify(candidate)
    if not runner.is_file():
        raise ValueError("RNN runtime runner is missing")
    config = load_object(candidate / "source-config.json")
    policy = load_object(candidate / "source-development-policy.json")
    freeze_policy = policy.get("candidate_freeze")
    if not isinstance(freeze_policy, dict):
        raise ValueError("RNN source policy missing candidate_freeze")
    namespace = int(freeze_policy.get("fresh_validation_seed_namespace", -1))
    if namespace <= 0:
        raise ValueError("RNN fresh validation seed namespace is not frozen")
    seed = int(config.get("seed", 1337)) + namespace

    registry_path = ROOT / str(freeze_policy.get("fresh_validation_registry", "experiments/model_family/rnn_fresh_validation_registry.json"))
    fresh_registry = load_object(registry_path)
    rows = [row for row in fresh_registry.get("namespaces", []) if isinstance(row, dict)]
    matches = [row for row in rows if int(row.get("namespace", -1)) == namespace]
    if len(matches) != 1:
        raise ValueError("RNN fresh namespace is not uniquely registered")
    fresh_entry = matches[0]
    if fresh_entry.get("model_family") != "rnn" or fresh_entry.get("status") != "reserved-untouched":
        raise ValueError("RNN fresh namespace is not reserved-untouched")
    candidate_model_sha256 = str(freeze["model_sha256"])
    for row in rows:
        if int(row.get("namespace", -1)) == namespace or row.get("model_family") != "rnn" or row.get("status") == "reserved-untouched":
            continue
        if str(row.get("candidate_model_sha256", "")) == candidate_model_sha256:
            raise ValueError("RNN model already consumed by an earlier fresh namespace")

    shadow_registry_path = ROOT / str(freeze_policy.get("shadow_registry", "experiments/model_family/shadow_arena_registry.json"))
    shadow_registry = load_object(shadow_registry_path)
    arena_name = str(freeze_policy.get("shadow_arena", ""))
    arenas = {str(row["name"]): row for row in shadow_registry.get("arenas", []) if isinstance(row, dict)}
    arena = arenas.get(arena_name)
    if not isinstance(arena, dict) or arena.get("model_family") != "rnn" or arena.get("status") != "reserved-untouched":
        raise ValueError("RNN shadow arena is not reserved-untouched")
    protected = {int(config.get("qualification_holdout_seed", -1))}
    protected.update(int(value) for value in config.get("retired_qualification_holdout_seeds", []))
    protected.update(int(value) for value in arena.get("seeds", []))
    if seed in protected:
        raise ValueError("RNN fresh seed overlaps protected qualification namespace")

    effective = copy.deepcopy(config)
    effective["seed"] = seed
    effective.pop("qualification_holdout_seed", None)
    effective.pop("retired_qualification_holdout_seeds", None)
    effective_path = output / "effective-config.json"
    write_object(effective_path, effective)
    dataset = output / "dataset"
    summary = render_domain_dataset(effective_path, dataset, curriculum_weights=None)

    corpus = load_object(candidate / "development-wav-sha256.json")
    development_hashes = {str(value) for value in corpus["wav_sha256"]}
    fresh_hashes = evaluation_hashes(dataset / "domain-index.jsonl")
    if development_hashes & fresh_hashes:
        raise ValueError("RNN fresh validation overlaps development/training WAV evidence")

    model = candidate / "model.kwm"; pack = candidate / "keywords.kwk"
    gates = gate_values(config["domain_gates"])
    split_rows: dict[str, dict] = {}; all_qualified = True
    for split in ("calibration", "test", "qualification"):
        base, domains = evaluate(
            runner=runner, model=model, pack=pack,
            references=dataset / f"{split}.references.jsonl",
            output=output / f"eval-{split}",
        )
        qualified = base_gate(base, gates) and domain_gate(domains, gates)
        all_qualified = all_qualified and qualified
        split_rows[split] = {
            "qualified": qualified, "metrics": base, "domains": domains,
            "references_sha256": sha256_file(dataset / f"{split}.references.jsonl"),
        }

    result = {
        "schema_version": 1,
        "policy": POLICY,
        "model_family": "rnn",
        "qualified": all_qualified,
        "frozen_candidate_policy": str(freeze["policy"]),
        "model_sha256": candidate_model_sha256,
        "pack_sha256": str(freeze["pack_sha256"]),
        "fresh_validation_registry_entry": str(fresh_entry.get("name", "")),
        "fresh_validation_registry_status": str(fresh_entry["status"]),
        "fresh_validation_seed": seed,
        "fresh_validation_seed_namespace": namespace,
        "fresh_validation_wav_count": len(fresh_hashes),
        "fresh_validation_wav_sha256": sorted(fresh_hashes),
        "development_overlap_count": 0,
        "thresholds_recalibrated": False,
        "training_performed": False,
        "validation_feedback_allowed": False,
        "threshold_feedback_allowed": False,
        "training_rule_feedback_allowed": False,
        "formal_qualification_used": False,
        "shadow_used": False,
        "domain_index_sha256": str(summary["domain_index_sha256"]),
        "splits": split_rows,
    }
    write_object(output / "fresh-validation-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(); result = validate(args.candidate, args.runner, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)); return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
