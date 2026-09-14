#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

from iterate_domain import (  # noqa: E402
    base_gate,
    calibrate,
    domain_gate,
    evaluate,
    gate_values,
    repo_path,
    run,
    sha256_file,
)
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402

POLICY = "bounded-gru-generalization-v3"
SOURCE_MODEL_SHA256 = "8dc7d85505147fb6c9b402b2b07f6f0b047335207ca954800ac5e311c46a3c7d"
POSITIVE_EXAMPLE_WEIGHT = 2.45
REPAIR_EPOCHS = 6
REPAIR_LR_SCALE = 0.25
TRAINING_SEED_OFFSET = 9_700_019
BASE_REPLAY_EXPOSURE_REPEAT = 3
CALIBRATION_REPAIR_EXPOSURE_REPEAT = 1
VALIDATION_SEED_NAMESPACE = 181_000_019
RESERVED_SHADOW_ARENA = "gru-independent-shadow-v2"


def strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def manifest_repeat(paths: list[pathlib.Path], count: int) -> list[pathlib.Path]:
    result: list[pathlib.Path] = []
    for path in paths:
        result.extend([path] * count)
    return result


def validate_shared(path: pathlib.Path, cfg: dict) -> dict:
    shared = load_json(path)
    if shared.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared model-family data policy mismatch")
    if shared.get("formal_qualification_used") is not False:
        raise ValueError("shared data touched formal qualification")
    if int(shared.get("formal_seed", -1)) != int(cfg["qualification_holdout_seed"]):
        raise ValueError("shared/formal seed identity drifted")
    extras = shared.get("train_only_seed_manifests", [])
    if not isinstance(extras, list) or len(extras) != 2:
        raise ValueError("GRU V3 requires exactly two predeclared training-only acoustic manifests")
    for row in extras:
        if row.get("formal_qualification_used") is not False:
            raise ValueError("training-only manifest touched formal qualification")
        if row.get("shadow_qualification_used") is not False:
            raise ValueError("training-only manifest touched shadow qualification")
        if row.get("evaluation_splits_consumed") is not False:
            raise ValueError("training-only manifest consumed evaluation split")
        manifest = pathlib.Path(str(row["manifest"]))
        if not manifest.is_file() or sha256_file(manifest) != str(row["manifest_sha256"]):
            raise ValueError("training-only manifest SHA binding failed")
    return shared


def local_bound_manifest(evidence_path: pathlib.Path, expected_class: str, expected_policy: str) -> tuple[dict, pathlib.Path]:
    evidence = load_json(evidence_path)
    if evidence.get("evidence_class") != expected_class:
        raise ValueError(f"unexpected replay evidence class: {evidence.get('evidence_class')}")
    if evidence.get("policy") != expected_policy:
        raise ValueError(f"unexpected replay evidence policy: {evidence.get('policy')}")
    if evidence.get("formal_qualification_used") is not False:
        raise ValueError("replay evidence touched formal qualification")
    name = pathlib.Path(str(evidence.get("manifest", ""))).name
    manifest = evidence_path.parent / name
    if not manifest.is_file() or sha256_file(manifest) != str(evidence.get("manifest_sha256")):
        raise ValueError("downloaded replay manifest binding failed")
    return evidence, manifest


def write_validation_config(cfg: dict, validation_seed: int, path: pathlib.Path) -> pathlib.Path:
    value = copy.deepcopy(cfg)
    value["seed"] = validation_seed
    value.pop("qualification_holdout_seed", None)
    value.pop("retired_qualification_holdout_seeds", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--base-replay-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-repair-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--source-candidate-root", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    shared_path = args.shared_data.resolve()
    shared = validate_shared(shared_path, cfg)

    registry = load_json(ROOT / "experiments" / "model_family" / "shadow_arena_registry.json")
    arenas = {str(v["name"]): v for v in registry["arenas"]}
    reserved = arenas.get(RESERVED_SHADOW_ARENA)
    if not isinstance(reserved, dict) or reserved.get("status") != "reserved-untouched":
        raise ValueError("GRU V3 shadow arena is not reserved-untouched")
    reserved_seeds = [int(v) for v in reserved["seeds"]]
    if reserved_seeds != list(range(951101, 951109)):
        raise ValueError("GRU V3 reserved shadow seed contract drifted")

    formal_seeds = {
        int(cfg["qualification_holdout_seed"]),
        *[int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])],
    }
    validation_seed = int(cfg.get("seed", 1337)) + VALIDATION_SEED_NAMESPACE
    if validation_seed in formal_seeds or validation_seed in set(reserved_seeds):
        raise ValueError("fresh development validation seed overlaps protected seed namespace")

    source_root = args.source_candidate_root.resolve()
    source_summary_path = source_root / "candidate-summary.json"
    source_manifest_path = source_root / "domain-loop-manifest.json"
    source_checkpoint = source_root / "best" / "model.pt"
    source_model = source_root / "best" / "model.kwm"
    source_pack = source_root / "best" / "keywords.kwk"
    for path in (source_summary_path, source_manifest_path, source_checkpoint, source_model, source_pack):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"source V2 artifact member missing: {path}")
    source_summary = load_json(source_summary_path)
    if source_summary.get("policy") != "shadow-blind-gru-generalization-v2":
        raise ValueError("source V2 policy mismatch")
    if sha256_file(source_model) != SOURCE_MODEL_SHA256 or str(source_summary.get("model_sha256")) != SOURCE_MODEL_SHA256:
        raise ValueError("source V2 model identity drifted")
    if bool(source_summary.get("calibration_gate")) is not False:
        raise ValueError("source V2 calibration unexpectedly passed")
    if bool(source_summary.get("test_gate")) is not True or bool(source_summary.get("development_qualification_gate")) is not True:
        raise ValueError("source V2 failure is not calibration-only")
    if source_summary.get("formal_qualification_used") is not False or source_summary.get("shadow_used") is not False:
        raise ValueError("source V2 candidate touched protected evidence")
    if int(source_summary.get("training_manifest_count", -1)) != 12 or int(source_summary.get("unique_training_manifest_count", -1)) != 6:
        raise ValueError("source V2 training exposure contract drifted")

    base_replay_path = args.base_replay_evidence.resolve()
    base_replay, base_replay_manifest = local_bound_manifest(
        base_replay_path,
        "training-only-gru-generalization-resynthesis",
        "gru-generalization-resynthesis-v1",
    )
    if base_replay.get("shadow_wav_bytes_copied") is not False:
        raise ValueError("source V2 replay copied shadow WAV bytes")
    if int(base_replay.get("source_event_count", -1)) != 4 or int(base_replay.get("examples", -1)) != 32:
        raise ValueError("source V2 replay cardinality drifted")

    repair_path = args.calibration_repair_evidence.resolve()
    repair, repair_manifest = local_bound_manifest(
        repair_path,
        "training-only-gru-v3-calibration-repair-resynthesis",
        "gru-v3-calibration-repair-resynthesis-v1",
    )
    if repair.get("shadow_used") is not False or repair.get("development_source_wav_bytes_copied") is not False:
        raise ValueError("GRU V3 repair consumed protected source material")
    if repair.get("source_split") != "calibration" or int(repair.get("examples", -1)) != 8:
        raise ValueError("GRU V3 repair contract drifted")
    if str(repair.get("source_candidate_model_sha256")) != SOURCE_MODEL_SHA256:
        raise ValueError("GRU V3 repair source model drifted")

    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("experimental GRU runner missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    canonical = pathlib.Path(str(shared["canonical_train_manifest"]))
    static = pathlib.Path(str(shared["static_manifest"]))
    adversarial = pathlib.Path(str(shared["adversarial_manifest"]))
    extras = [pathlib.Path(str(v["manifest"])) for v in shared["train_only_seed_manifests"]]
    replay_manifests = [static, adversarial]
    failure = pathlib.Path(str(shared["failure_manifest"]))
    if int(shared.get("failure_replay_examples", 0)) > 0:
        replay_manifests.append(failure)
    manifests = [
        canonical,
        *extras,
        *manifest_repeat(replay_manifests, BASE_REPLAY_EXPOSURE_REPEAT),
        *([base_replay_manifest] * BASE_REPLAY_EXPOSURE_REPEAT),
        *([repair_manifest] * CALIBRATION_REPAIR_EXPOSURE_REPEAT),
    ]
    for path in manifests:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing GRU V3 manifest: {path}")

    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    candidate = output / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate / "model.pt"
    command = [sys.executable, str(TRAINING / "train_gru_ctc.py")]
    for manifest in manifests:
        command.extend(["--manifest", str(manifest)])
    command.extend(
        [
            "--tokens", str(tokens),
            "--keywords", str(keywords),
            "--frontend", str(shared["source_frontend"]),
            "--feature-dim", str(int(model_cfg.get("feature_dim", 32))),
            "--hidden-dim", "64",
            "--epochs", str(REPAIR_EPOCHS),
            "--batch-size", str(int(train_cfg.get("batch_size", 16))),
            "--lr", str(float(train_cfg.get("lr", 0.001)) * REPAIR_LR_SCALE),
            "--seed", str(int(cfg.get("seed", 1337)) + TRAINING_SEED_OFFSET),
            "--positive-example-weight", str(POSITIVE_EXAMPLE_WEIGHT),
            "--warm-start", str(source_checkpoint),
            "--output", str(checkpoint),
        ]
    )
    run(command)
    model = candidate / "model.kwg"
    run(
        [
            sys.executable,
            str(TRAINING / "export_gru_model.py"),
            "--checkpoint", str(checkpoint),
            "--tokens", str(tokens),
            "--output", str(model),
        ]
    )
    provenance = pathlib.Path(str(model) + ".provenance.json")

    gates = gate_values(cfg["domain_gates"])
    canonical_cal_base, canonical_cal_domains = evaluate(
        runner=runner,
        model=model,
        pack=source_pack,
        references=pathlib.Path(str(shared["canonical_calibration_references"])),
        output=candidate / "canonical-regression-calibration",
    )
    canonical_test_base, canonical_test_domains = evaluate(
        runner=runner,
        model=model,
        pack=source_pack,
        references=pathlib.Path(str(shared["canonical_test_references"])),
        output=candidate / "canonical-regression-test",
    )
    canonical_cal_gate = strict(canonical_cal_base, canonical_cal_domains, gates)
    canonical_test_gate = strict(canonical_test_base, canonical_test_domains, gates)

    validation_config = write_validation_config(
        cfg,
        validation_seed,
        output / "fresh-development-validation-config.json",
    )
    validation_dataset = output / "fresh-development-validation"
    render_domain_dataset(validation_config, validation_dataset, curriculum_weights=None)
    thresholds = [float(v) for v in cfg["calibration"]["thresholds"]]
    coordinate_rounds = int(cfg["calibration"]["coordinate_rounds"])
    calibrated, pack, val_cal_base, val_cal_domains = calibrate(
        runner=runner,
        model=model,
        tokens=tokens,
        source_keywords=keywords,
        references=validation_dataset / "calibration.references.jsonl",
        output=candidate / "fresh-validation-calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
    )
    val_test_base, val_test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=validation_dataset / "test.references.jsonl",
        output=candidate / "fresh-validation-test",
    )
    val_qual_base, val_qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=validation_dataset / "qualification.references.jsonl",
        output=candidate / "fresh-validation-qualification",
    )
    val_cal_gate = strict(val_cal_base, val_cal_domains, gates)
    val_test_gate = strict(val_test_base, val_test_domains, gates)
    val_qual_gate = strict(val_qual_base, val_qual_domains, gates)
    qualified = bool(
        canonical_cal_gate
        and canonical_test_gate
        and val_cal_gate
        and val_test_gate
        and val_qual_gate
    )

    manifest = {
        "schema_version": 1,
        "evidence_class": "development-only-bounded-gru-generalization-v3",
        "policy": POLICY,
        "development_qualified": bool(canonical_cal_gate and canonical_test_gate and val_cal_gate and val_test_gate),
        "qualification_qualified": bool(val_qual_gate),
        "qualified": qualified,
        "formal_qualification_used": False,
        "shadow_used": False,
        "source_candidate": {
            "policy": str(source_summary["policy"]),
            "model_sha256": SOURCE_MODEL_SHA256,
            "summary_sha256": sha256_file(source_summary_path),
            "manifest_sha256": sha256_file(source_manifest_path),
            "calibration_failure_used_as_mining_metadata": True,
            "calibration_wav_bytes_copied": False,
        },
        "repair": {
            "policy": str(repair["policy"]),
            "examples": int(repair["examples"]),
            "manifest_sha256": str(repair["manifest_sha256"]),
            "epochs": REPAIR_EPOCHS,
            "lr_scale": REPAIR_LR_SCALE,
            "positive_example_weight": POSITIVE_EXAMPLE_WEIGHT,
            "warm_start": True,
            "exposure_repeat": CALIBRATION_REPAIR_EXPOSURE_REPEAT,
        },
        "canonical_regression": {
            "used_for_training": True,
            "used_as_unbiased_validation": False,
            "calibration": canonical_cal_base,
            "calibration_domains": canonical_cal_domains,
            "calibration_gate": canonical_cal_gate,
            "test": canonical_test_base,
            "test_domains": canonical_test_domains,
            "test_gate": canonical_test_gate,
        },
        "fresh_validation": {
            "seed": validation_seed,
            "config_sha256": sha256_file(validation_config),
            "used_for_training": False,
            "calibration": val_cal_base,
            "calibration_domains": val_cal_domains,
            "calibration_gate": val_cal_gate,
            "test": val_test_base,
            "test_domains": val_test_domains,
            "test_gate": val_test_gate,
            "qualification": val_qual_base,
            "qualification_domains": val_qual_domains,
            "qualification_gate": val_qual_gate,
        },
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "selection_basis": "single-predeclared-bounded-repair-no-shadow-selection",
            "eligible_rounds": [0] if qualified else [],
            "selected_round": 0,
            "selected_frontend": str(shared["source_frontend"]),
            "qualification_used_for_selection": False,
            "shadow_used_for_selection": False,
            "fresh_development_validation_used_for_gate": True,
        },
    }
    manifest_path = output / "domain-loop-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    best = output / "best"
    best.mkdir(parents=True, exist_ok=True)
    for src, name in (
        (model, "model.kwm"),
        (checkpoint, "model.pt"),
        (provenance, "model-provenance.json"),
        (pack, "keywords.kwk"),
        (calibrated, "keywords.tsv"),
    ):
        shutil.copy2(src, best / name)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parameter_count = sum(int(t.numel()) for t in checkpoint_payload["state_dict"].values())
    summary = {
        "schema_version": 1,
        "policy": POLICY,
        "experimental": True,
        "formal_qualification_used": False,
        "shadow_used": False,
        "shadow_used_for_selection": False,
        "reserved_untouched_shadow_arena": RESERVED_SHADOW_ARENA,
        "reserved_untouched_shadow_seeds": reserved_seeds,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "model_sha256": sha256_file(model),
        "pack_sha256": sha256_file(pack),
        "model_bytes": model.stat().st_size,
        "parameter_count": parameter_count,
        "estimated_macs_per_frame": 18752,
        "parameters": {
            "positive_example_weight": POSITIVE_EXAMPLE_WEIGHT,
            "repair_epochs": REPAIR_EPOCHS,
            "repair_lr_scale": REPAIR_LR_SCALE,
            "training_seed_offset": TRAINING_SEED_OFFSET,
            "base_replay_exposure_repeat": BASE_REPLAY_EXPOSURE_REPEAT,
            "calibration_repair_exposure_repeat": CALIBRATION_REPAIR_EXPOSURE_REPEAT,
            "full_train_seed_count": 3,
        },
        "training_manifest_count": len(manifests),
        "unique_training_manifest_count": len({str(v.resolve()) for v in manifests}),
        "calibration_repair_examples": int(repair["examples"]),
        "canonical_regression_calibration": canonical_cal_base,
        "canonical_regression_test": canonical_test_base,
        "canonical_regression_calibration_gate": canonical_cal_gate,
        "canonical_regression_test_gate": canonical_test_gate,
        "fresh_validation_seed": validation_seed,
        "fresh_validation_used_for_training": False,
        "fresh_validation_calibration": val_cal_base,
        "fresh_validation_test": val_test_base,
        "fresh_validation_qualification": val_qual_base,
        "fresh_validation_calibration_gate": val_cal_gate,
        "fresh_validation_test_gate": val_test_gate,
        "fresh_validation_qualification_gate": val_qual_gate,
        "qualified": qualified,
        "shared_data_policy": str(shared["policy"]),
        "base_replay_evidence_sha256": sha256_file(base_replay_path),
        "calibration_repair_evidence_sha256": sha256_file(repair_path),
    }
    (output / "candidate-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
