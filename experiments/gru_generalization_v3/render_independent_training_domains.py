#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "tools"))

from iterate_domain import sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402

POLICY = "gru-v3-independent-training-domain-expansion-v1"
EVIDENCE_CLASS = "training-only-gru-v3-independent-domain-expansion"
SEED_OFFSETS = (730_049, 840_067)


def write_train_only_config(cfg: dict, seed: int, path: pathlib.Path) -> pathlib.Path:
    value = copy.deepcopy(cfg)
    value["seed"] = seed
    value.pop("qualification_holdout_seed", None)
    value.pop("retired_qualification_holdout_seeds", None)
    value.pop("robustness_gates", None)
    value["shadow_qualification"] = {"enabled": False, "seeds": []}
    # Only train.tsv is consumed. Keep tiny renderer-compatible evaluation splits
    # so this uses the same production scene renderer without importing any
    # canonical calibration/test/qualification references.
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_registry(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("shadow arena registry must be an object")
    if value.get("policy") != "development-shadow-arena-one-shot-registry-v1":
        raise ValueError("shadow arena registry policy drifted")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    registry = load_registry(ROOT / "experiments" / "model_family" / "shadow_arena_registry.json")

    base_seed = int(cfg.get("seed", 1337))
    seeds = [base_seed + offset for offset in SEED_OFFSETS]
    formal = {int(cfg["qualification_holdout_seed"])} | {
        int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])
    }
    shadow = {
        int(seed)
        for arena in registry.get("arenas", [])
        if isinstance(arena, dict)
        for seed in arena.get("seeds", [])
    }
    configured_shadow = {int(v) for v in cfg.get("shadow_qualification", {}).get("seeds", [])}
    forbidden = formal | shadow | configured_shadow
    if len(set(seeds)) != len(seeds):
        raise ValueError("GRU V3 independent training seeds must be unique")
    if set(seeds) & forbidden:
        raise ValueError("GRU V3 independent training seed overlaps evaluation namespace")

    rows: list[dict] = []
    for ordinal, seed in enumerate(seeds):
        root = output / f"seed-{seed}"
        effective = write_train_only_config(cfg, seed, root / "effective-config.json")
        rendered = json.loads(effective.read_text(encoding="utf-8"))
        if "qualification_holdout_seed" in rendered or "retired_qualification_holdout_seeds" in rendered:
            raise ValueError("GRU V3 train-only config retained formal seed namespace")
        if rendered.get("shadow_qualification", {}).get("enabled") is not False:
            raise ValueError("GRU V3 train-only config retained shadow evaluation")
        if "robustness_gates" in rendered:
            raise ValueError("GRU V3 train-only config retained evaluation robustness gates")

        dataset = root / "dataset"
        render_domain_dataset(effective, dataset)
        manifest = dataset / "train.tsv"
        if not manifest.is_file() or manifest.stat().st_size == 0:
            raise ValueError("GRU V3 independent renderer did not produce train.tsv")
        rows.append(
            {
                "ordinal": ordinal,
                "seed": seed,
                "policy": "training-only-domain-render-v1",
                "manifest": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "effective_config_sha256": sha256_file(effective),
                "formal_qualification_used": False,
                "shadow_qualification_used": False,
                "evaluation_splits_consumed": False,
            }
        )

    evidence = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "formal_qualification_used": False,
        "shadow_used": False,
        "evaluation_splits_consumed": False,
        "source_calibration_failure_used": False,
        "source_v2_candidate_metrics_used": False,
        "source_v2_failure_diagnostics_used": False,
        "seed_offsets": list(SEED_OFFSETS),
        "seeds": seeds,
        "manifests": rows,
        "config_sha256": sha256_file(config_path),
        "shadow_registry_sha256": sha256_file(ROOT / "experiments" / "model_family" / "shadow_arena_registry.json"),
    }
    evidence_path = output / "gru-v3-independent-training-domains.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
