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
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

from iterate_domain import base_gate, domain_gate, evaluate, gate_values, sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from render_qualification_holdout import (  # noqa: E402
    _index_hashes,
    _qualification_render_config,
    _reference_stats,
    _render_retired_seed,
    normalize_retired_qualification_seeds,
)
from verify_gru_frozen_candidate import verify  # noqa: E402

POLICY = "frozen-gru-formal-qualification-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_preformal(fresh_summary: pathlib.Path, shadow_summary: pathlib.Path) -> tuple[dict, dict]:
    fresh = load_object(fresh_summary)
    shadow = load_object(shadow_summary)
    if fresh.get("policy") != "frozen-gru-fresh-validation-v1" or fresh.get("qualified") is not True:
        raise ValueError("fresh frozen-candidate validation is not qualified")
    if fresh.get("training_performed") is not False or fresh.get("thresholds_recalibrated") is not False:
        raise ValueError("fresh validation mutated candidate training/threshold contract")
    if shadow.get("evidence_class") != "development-only-shadow-qualification" or shadow.get("qualified") is not True:
        raise ValueError("reserved shadow qualification is not qualified")
    if shadow.get("formal_qualification_seed_consumed") is not False:
        raise ValueError("shadow stage unexpectedly consumed formal qualification")
    return fresh, shadow


def shadow_hashes(shadow_root: pathlib.Path, seeds: list[int]) -> set[str]:
    result: set[str] = set()
    for seed in seeds:
        index = shadow_root / f"seed-{seed}" / "dataset" / "domain-index.jsonl"
        values = _index_hashes(index, {"qualification"})
        if not values:
            raise ValueError(f"shadow seed {seed} has no qualification WAV identity evidence")
        result.update(values)
    return result


def fresh_wav_hashes(fresh: dict) -> set[str]:
    values = fresh.get("fresh_validation_wav_sha256")
    if not isinstance(values, list) or not values:
        raise ValueError("fresh validation WAV identity evidence is missing")
    result = {str(value) for value in values}
    if len(result) != len(values):
        raise ValueError("fresh validation WAV identity evidence contains duplicates")
    if any(len(value) != 64 for value in result):
        raise ValueError("fresh validation WAV identity evidence contains an invalid SHA")
    if int(fresh.get("fresh_validation_wav_count", -1)) != len(result):
        raise ValueError("fresh validation WAV identity count does not match the retained evidence")
    return result


def qualify(
    candidate: pathlib.Path,
    runner: pathlib.Path,
    fresh_summary: pathlib.Path,
    shadow_root: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    candidate = candidate.resolve()
    runner = runner.resolve()
    fresh_summary = fresh_summary.resolve()
    shadow_root = shadow_root.resolve()
    output = output.resolve()
    freeze = verify(candidate)
    if not runner.is_file():
        raise ValueError("GRU runtime runner is missing")
    fresh, shadow = require_preformal(fresh_summary, shadow_root / "summary.json")
    if str(fresh.get("model_sha256")) != str(freeze["model_sha256"]):
        raise ValueError("fresh validation model identity does not match frozen candidate")
    fresh_hashes = fresh_wav_hashes(fresh)

    config = load_object(candidate / "source-config.json")
    training_seed = int(config.get("seed", 1337))
    formal_seed = int(config.get("qualification_holdout_seed", -1))
    if formal_seed < 0 or formal_seed == training_seed:
        raise ValueError("active formal qualification seed is invalid")
    retired = normalize_retired_qualification_seeds(
        config.get("retired_qualification_holdout_seeds", []),
        training_seed=training_seed,
        qualification_seed=formal_seed,
    )
    shadow_seeds = [int(value) for value in shadow.get("seeds", [])]
    if formal_seed in set(shadow_seeds) or set(retired) & set(shadow_seeds):
        raise ValueError("formal/shadow qualification seed namespaces overlap")

    corpus = load_object(candidate / "development-wav-sha256.json")
    development_hashes = {str(value) for value in corpus["wav_sha256"]}
    if not development_hashes:
        raise ValueError("frozen development WAV identity set is empty")
    seen_shadow_hashes = shadow_hashes(shadow_root, shadow_seeds)
    if development_hashes & fresh_hashes:
        raise ValueError("fresh validation overlaps frozen development/training WAV identities")
    if development_hashes & seen_shadow_hashes:
        raise ValueError("shadow qualification overlaps frozen development/training WAV identities")
    if fresh_hashes & seen_shadow_hashes:
        raise ValueError("reserved shadow qualification overlaps fresh validation WAV identities")

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    effective = output / "effective-config.json"
    write_object(effective, _qualification_render_config(config, formal_seed))
    dataset = output / "dataset"
    rendered = render_domain_dataset(effective, dataset, curriculum_weights=None)
    active_index = dataset / "domain-index.jsonl"
    active_references = dataset / "qualification.references.jsonl"
    active_hashes = _index_hashes(active_index, {"qualification"})
    if not active_hashes:
        raise ValueError("active formal cohort has no qualification WAVs")
    if active_hashes & development_hashes:
        raise ValueError("formal qualification overlaps frozen development/training WAV identities")
    if active_hashes & fresh_hashes:
        raise ValueError("formal qualification overlaps fresh validation WAV identities")
    if active_hashes & seen_shadow_hashes:
        raise ValueError("formal qualification overlaps reserved shadow WAV identities")

    recordings, expected_wakes = _reference_stats(active_references)
    retired_evidence: list[dict] = []
    seen_retired_hashes: set[str] = set()
    scratch = output / "retired-scratch"
    for retired_seed in retired:
        row = _render_retired_seed(config, scratch, retired_seed)
        hashes = {str(value) for value in row["wav_hashes"]}
        boundaries = (
            ("active formal cohort", active_hashes),
            ("frozen development/training corpus", development_hashes),
            ("fresh validation cohort", fresh_hashes),
            ("reserved shadow cohort", seen_shadow_hashes),
            ("another retired formal cohort", seen_retired_hashes),
        )
        for label, boundary_hashes in boundaries:
            overlap = hashes & boundary_hashes
            if overlap:
                raise ValueError(
                    f"retired formal seed {retired_seed} overlaps {len(overlap)} WAV SHA(s) with {label}"
                )
        if int(row["recordings"]) != recordings or int(row["expected_wakes"]) != expected_wakes:
            raise ValueError(f"retired formal seed {retired_seed} support differs from active cohort")
        seen_retired_hashes.update(hashes)
        retired_evidence.append(
            {
                "seed": retired_seed,
                "references_sha256": str(row["references_sha256"]),
                "domain_index_sha256": str(row["domain_index_sha256"]),
            }
        )
    shutil.rmtree(scratch, ignore_errors=True)

    model = candidate / "model.kwm"
    pack = candidate / "keywords.kwk"
    gates = gate_values(config["domain_gates"])
    base, domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=active_references,
        output=output / "eval",
    )
    qualified = base_gate(base, gates) and domain_gate(domains, gates)
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "qualified": qualified,
        "model_sha256": str(freeze["model_sha256"]),
        "pack_sha256": str(freeze["pack_sha256"]),
        "fresh_validation_summary_sha256": sha256_file(fresh_summary),
        "shadow_summary_sha256": sha256_file(shadow_root / "summary.json"),
        "formal_seed": formal_seed,
        "formal_seed_consumed": True,
        "retired_formal_seeds": retired,
        "shadow_seeds": shadow_seeds,
        "recordings": recordings,
        "expected_wakes": expected_wakes,
        "development_overlap_count": 0,
        "fresh_overlap_count": 0,
        "shadow_overlap_count": 0,
        "retired_overlap_count": 0,
        "fresh_validation_wav_count": len(fresh_hashes),
        "shadow_wav_count": len(seen_shadow_hashes),
        "formal_wav_count": len(active_hashes),
        "retired_wav_count": len(seen_retired_hashes),
        "all_pairwise_sha_disjoint": True,
        "references_sha256": sha256_file(active_references),
        "domain_index_sha256": str(rendered["domain_index_sha256"]),
        "metrics": base,
        "domains": domains,
        "retired_evidence": retired_evidence,
        "training_performed": False,
        "thresholds_recalibrated": False,
        "validation_feedback_allowed": False,
        "threshold_feedback_allowed": False,
        "training_rule_feedback_allowed": False,
    }
    write_object(output / "formal-qualification-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--fresh-summary", required=True, type=pathlib.Path)
    parser.add_argument("--shadow-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = qualify(
        args.candidate,
        args.runner,
        args.fresh_summary,
        args.shadow_root,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
