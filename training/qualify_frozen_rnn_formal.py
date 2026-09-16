#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING)); sys.path.insert(0, str(TOOLS))

from iterate_domain import base_gate, domain_gate, evaluate, gate_values, sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from render_qualification_holdout import _qualification_render_config, _reference_stats, _render_retired_seed, normalize_retired_qualification_seeds  # noqa: E402
from verify_rnn_evaluation_independence import split_hashes  # noqa: E402
from verify_rnn_frozen_candidate import verify  # noqa: E402

POLICY = "frozen-rnn-formal-qualification-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require_preformal(fresh_summary: pathlib.Path, shadow_summary: pathlib.Path) -> tuple[dict, dict]:
    fresh = load_object(fresh_summary); shadow = load_object(shadow_summary)
    if fresh.get("policy") != "frozen-rnn-fresh-validation-v1" or fresh.get("model_family") != "rnn" or fresh.get("qualified") is not True:
        raise ValueError("RNN fresh validation is not qualified")
    if fresh.get("training_performed") is not False or fresh.get("thresholds_recalibrated") is not False:
        raise ValueError("RNN fresh validation mutated training/threshold contract")
    if shadow.get("evidence_class") != "development-only-shadow-qualification" or shadow.get("qualified") is not True:
        raise ValueError("RNN reserved shadow qualification is not qualified")
    if shadow.get("formal_qualification_seed_consumed") is not False:
        raise ValueError("RNN shadow stage unexpectedly consumed formal qualification")
    return fresh, shadow


def shadow_hashes(root: pathlib.Path, seeds: list[int]) -> set[str]:
    result: set[str] = set()
    for seed in seeds:
        values = split_hashes(root / f"seed-{seed}" / "dataset" / "domain-index.jsonl", "qualification")
        if result & values: raise ValueError(f"RNN shadow seed {seed} overlaps earlier shadow seed")
        result.update(values)
    return result


def fresh_wav_hashes(fresh: dict) -> set[str]:
    values = fresh.get("fresh_validation_wav_sha256")
    if not isinstance(values, list) or not values: raise ValueError("RNN fresh WAV evidence missing")
    result = {str(value) for value in values}
    if len(result) != len(values) or int(fresh.get("fresh_validation_wav_count", -1)) != len(result): raise ValueError("RNN fresh WAV evidence invalid")
    return result


def qualify(candidate: pathlib.Path, runner: pathlib.Path, fresh_summary: pathlib.Path, shadow_root: pathlib.Path, output: pathlib.Path) -> dict:
    candidate = candidate.resolve(); runner = runner.resolve(); fresh_summary = fresh_summary.resolve(); shadow_root = shadow_root.resolve(); output = output.resolve()
    freeze = verify(candidate)
    if not runner.is_file(): raise ValueError("RNN runtime runner is missing")
    fresh, shadow = require_preformal(fresh_summary, shadow_root / "summary.json")
    if str(fresh.get("model_sha256")) != str(freeze["model_sha256"]): raise ValueError("RNN fresh model identity mismatch")
    fresh_hashes = fresh_wav_hashes(fresh)

    config = load_object(candidate / "source-config.json")
    source_policy = load_object(candidate / "source-development-policy.json")
    candidate_stage = source_policy.get("candidate_freeze")
    if not isinstance(candidate_stage, dict): raise ValueError("RNN candidate freeze policy missing")
    training_seed = int(config.get("seed", 1337))
    formal_seed = int(candidate_stage.get("formal_qualification_seed", -1))
    gru_active_formal = int(config.get("qualification_holdout_seed", -1))
    retired = normalize_retired_qualification_seeds(config.get("retired_qualification_holdout_seeds", []), training_seed=training_seed, qualification_seed=formal_seed)
    if formal_seed < 0 or formal_seed == training_seed: raise ValueError("RNN formal qualification seed invalid")
    if formal_seed == gru_active_formal or formal_seed in set(retired): raise ValueError("RNN formal seed overlaps GRU active/retired formal namespace")
    if int(freeze.get("candidate_stage", {}).get("formal_qualification_seed", -1)) != formal_seed: raise ValueError("RNN frozen formal seed binding mismatch")

    shadow_seeds = [int(value) for value in shadow.get("seeds", [])]
    if formal_seed in set(shadow_seeds) or set(retired) & set(shadow_seeds): raise ValueError("RNN formal/shadow namespaces overlap")
    if formal_seed == int(fresh.get("fresh_validation_seed", -1)): raise ValueError("RNN formal/Fresh namespaces overlap")

    corpus = load_object(candidate / "development-wav-sha256.json"); development_hashes = {str(value) for value in corpus["wav_sha256"]}
    if not development_hashes: raise ValueError("RNN development WAV evidence empty")
    seen_shadow = shadow_hashes(shadow_root, shadow_seeds)
    if development_hashes & fresh_hashes or development_hashes & seen_shadow or fresh_hashes & seen_shadow:
        raise ValueError("RNN development/fresh/shadow WAV evidence overlaps")

    shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True, exist_ok=True)
    effective = output / "effective-config.json"; write_object(effective, _qualification_render_config(config, formal_seed))
    dataset = output / "dataset"; rendered = render_domain_dataset(effective, dataset, curriculum_weights=None)
    active_index = dataset / "domain-index.jsonl"; active_refs = dataset / "qualification.references.jsonl"
    active_hashes = split_hashes(active_index, "qualification")
    if active_hashes & development_hashes or active_hashes & fresh_hashes or active_hashes & seen_shadow:
        raise ValueError("RNN active formal cohort overlaps earlier evidence")

    recordings, expected_wakes = _reference_stats(active_refs)
    retired_evidence: list[dict] = []; seen_retired: set[str] = set(); scratch = output / "retired-scratch"
    for retired_seed in retired:
        row = _render_retired_seed(config, scratch, retired_seed)
        index = scratch / f"seed-{retired_seed}" / "dataset" / "domain-index.jsonl"
        hashes = split_hashes(index, "qualification")
        if hashes != {str(value) for value in row["wav_hashes"]}: raise ValueError(f"RNN retired seed {retired_seed} renderer identity mismatch")
        for label, boundary in (("active", active_hashes), ("development", development_hashes), ("fresh", fresh_hashes), ("shadow", seen_shadow), ("retired", seen_retired)):
            if hashes & boundary: raise ValueError(f"RNN retired seed {retired_seed} overlaps {label} evidence")
        if int(row["recordings"]) != recordings or int(row["expected_wakes"]) != expected_wakes: raise ValueError(f"RNN retired seed {retired_seed} support differs from active cohort")
        seen_retired.update(hashes)
        retired_evidence.append({"seed": retired_seed, "references_sha256": str(row["references_sha256"]), "domain_index_sha256": str(row["domain_index_sha256"]), "wav_sha256_count": len(hashes)})
    shutil.rmtree(scratch, ignore_errors=True)

    base, domains = evaluate(runner=runner, model=candidate / "model.kwm", pack=candidate / "keywords.kwk", references=active_refs, output=output / "eval")
    gates = gate_values(config["domain_gates"]); qualified = base_gate(base, gates) and domain_gate(domains, gates)
    result = {
        "schema_version": 1, "policy": POLICY, "model_family": "rnn", "qualified": qualified,
        "model_sha256": str(freeze["model_sha256"]), "pack_sha256": str(freeze["pack_sha256"]),
        "fresh_validation_summary_sha256": sha256_file(fresh_summary), "shadow_summary_sha256": sha256_file(shadow_root / "summary.json"),
        "formal_seed": formal_seed, "gru_active_formal_seed_untouched": gru_active_formal, "formal_seed_consumed": True, "retired_formal_seeds": retired, "shadow_seeds": shadow_seeds,
        "recordings": recordings, "expected_wakes": expected_wakes, "development_overlap_count": 0, "fresh_overlap_count": 0, "shadow_overlap_count": 0, "retired_overlap_count": 0,
        "fresh_validation_wav_count": len(fresh_hashes), "shadow_wav_count": len(seen_shadow), "formal_wav_count": len(active_hashes), "retired_wav_count": len(seen_retired),
        "all_pairwise_sha_disjoint": True, "all_formal_cohorts_internal_sha_unique": True,
        "references_sha256": sha256_file(active_refs), "domain_index_sha256": str(rendered["domain_index_sha256"]),
        "metrics": base, "domains": domains, "retired_evidence": retired_evidence,
        "training_performed": False, "thresholds_recalibrated": False,
        "validation_feedback_allowed": False, "threshold_feedback_allowed": False, "training_rule_feedback_allowed": False,
    }
    write_object(output / "formal-qualification-summary.json", result); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", required=True, type=pathlib.Path); parser.add_argument("--runner", required=True, type=pathlib.Path); parser.add_argument("--fresh-summary", required=True, type=pathlib.Path); parser.add_argument("--shadow-root", required=True, type=pathlib.Path); parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(); result = qualify(args.candidate, args.runner, args.fresh_summary, args.shadow_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)); return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
