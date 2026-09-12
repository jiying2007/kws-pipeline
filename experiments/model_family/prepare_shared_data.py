#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "tools"))

from adversarial_lexicon import mine_adversarial_lexicon  # noqa: E402
from development_failure_replay import render_development_failure_replay  # noqa: E402
from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from render_qualification_holdout import require_strict_development_candidate  # noqa: E402
from synthetic_audio import load_config  # noqa: E402
from iterate_domain import sha256_file  # noqa: E402

POLICY = "model-family-shared-development-data-v1"
TRAIN_ONLY_SEED_OFFSETS = (510_017, 620_033)


def selected_record(manifest: dict) -> dict:
    selection = manifest["candidate_selection"]
    round_id = int(selection["selected_round"])
    frontend = str(selection["selected_frontend"])
    rows = [
        row
        for row in manifest["records"]
        if isinstance(row, dict)
        and int(row.get("round", -1)) == round_id
        and str(row.get("frontend")) == frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("cannot resolve strict source checkpoint")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def write_train_only_config(cfg: dict, seed: int, path: pathlib.Path) -> pathlib.Path:
    value = copy.deepcopy(cfg)
    value["seed"] = seed
    value.pop("qualification_holdout_seed", None)
    value.pop("retired_qualification_holdout_seeds", None)
    # Only train.tsv is consumed from these namespaces. Minimize unrelated split
    # generation while preserving train split bytes and acoustic policy.
    for split in ("calibration", "test", "qualification"):
        value["dataset"][split] = {
            "positive_families_per_keyword": 1,
            "confusable_families_per_keyword": 1,
            "random_negative_families": 1,
            "background_seconds_per_profile": 0.1,
            "variants_per_family": 1,
        }
        value["domains"]["scenes_per_example"][split] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--rnn-work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-shadow-summary", type=pathlib.Path)
    parser.add_argument("--baseline-shadow-failure-summary", type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    rnn_work = args.rnn_work_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    shadow_seeds = {int(v) for v in cfg["shadow_qualification"]["seeds"]}
    formal_seeds = {int(cfg["qualification_holdout_seed"])} | {
        int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])
    }
    base_seed = int(cfg.get("seed", 1337))
    train_only_seeds = [base_seed + offset for offset in TRAIN_ONLY_SEED_OFFSETS]
    if set(train_only_seeds) & shadow_seeds:
        raise ValueError("training-only seed overlaps shadow namespace")
    if set(train_only_seeds) & formal_seeds:
        raise ValueError("training-only seed overlaps formal namespace")

    manifest = require_strict_development_candidate(rnn_work)
    source = selected_record(manifest)
    source_checkpoint = pathlib.Path(str(source["checkpoint"])).resolve()
    if not source_checkpoint.is_file():
        raise ValueError(f"source checkpoint missing: {source_checkpoint}")
    source_round = int(source["round"])
    frontend = str(source["frontend"])
    probe_round = max(int(row["round"]) for row in manifest["records"]) + 1
    curriculum = manifest.get("final_curriculum")
    if not isinstance(curriculum, dict):
        curriculum = None

    canonical = output / "canonical"
    dataset = canonical / "dataset"
    render_domain_dataset(config_path, dataset, curriculum_weights=curriculum)
    static = render_hard_negative_replay(
        config_path,
        canonical / "static-replay",
        round_index=probe_round,
        curriculum_weights=curriculum,
    )
    adversarial = mine_adversarial_lexicon(
        config_path,
        source_checkpoint,
        canonical / "adversarial-lexicon",
        round_index=probe_round,
        frontend=frontend,
    )
    failure = render_development_failure_replay(
        config_path,
        list(manifest.get("records", [])),
        rnn_work,
        canonical / "development-failure-replay",
    )
    if bool(adversarial.get("formal_qualification_used", True)):
        raise ValueError("shared adversarial mining touched formal qualification")
    if bool(failure.get("formal_qualification_used", True)):
        raise ValueError("shared failure replay touched formal qualification")

    extra_manifests: list[dict] = []
    for index, seed in enumerate(train_only_seeds):
        seed_root = output / "train-only" / f"seed-{seed}"
        seed_config = write_train_only_config(cfg, seed, seed_root / "effective-config.json")
        seed_dataset = seed_root / "dataset"
        render_domain_dataset(seed_config, seed_dataset, curriculum_weights=curriculum)
        train_manifest = seed_dataset / "train.tsv"
        extra_manifests.append(
            {
                "seed": seed,
                "manifest": str(train_manifest),
                "manifest_sha256": sha256_file(train_manifest),
                "effective_config_sha256": sha256_file(seed_config),
                "formal_qualification_used": False,
                "shadow_qualification_used": False,
            }
        )

    source_copy = output / "source" / "model.pt"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, source_copy)

    baseline = {}
    if args.baseline_shadow_summary and args.baseline_shadow_summary.is_file():
        target = output / "baseline" / "shadow-summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.baseline_shadow_summary, target)
        baseline["shadow_summary"] = str(target)
        baseline["shadow_summary_sha256"] = sha256_file(target)
    if args.baseline_shadow_failure_summary and args.baseline_shadow_failure_summary.is_file():
        target = output / "baseline" / "shadow-failure-summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.baseline_shadow_failure_summary, target)
        baseline["shadow_failure_summary"] = str(target)
        baseline["shadow_failure_summary_sha256"] = sha256_file(target)

    result = {
        "schema_version": 1,
        "policy": POLICY,
        "formal_qualification_used": False,
        "source_round": source_round,
        "source_frontend": frontend,
        "source_checkpoint": str(source_copy),
        "source_checkpoint_sha256": sha256_file(source_copy),
        "source_manifest_sha256": sha256_file(rnn_work / "domain-loop-manifest.json"),
        "probe_round": probe_round,
        "canonical_train_manifest": str(dataset / "train.tsv"),
        "canonical_calibration_references": str(dataset / "calibration.references.jsonl"),
        "canonical_test_references": str(dataset / "test.references.jsonl"),
        "canonical_qualification_references": str(dataset / "qualification.references.jsonl"),
        "static_manifest": str(static["manifest"]),
        "static_manifest_sha256": str(static["manifest_sha256"]),
        "static_replay_examples": int(static.get("examples", 0)),
        "adversarial_manifest": str(adversarial["manifest"]),
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "adversarial_replay_examples": int(adversarial.get("replay_examples", 0)),
        "failure_manifest": str(failure["manifest"]),
        "failure_manifest_sha256": str(failure["manifest_sha256"]),
        "failure_replay_examples": int(failure.get("examples", 0)),
        "train_only_seed_manifests": extra_manifests,
        "shadow_seeds": sorted(shadow_seeds),
        "formal_seed": int(cfg["qualification_holdout_seed"]),
        "baseline": baseline,
    }
    out = output / "shared-data.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
