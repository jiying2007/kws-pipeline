#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    objective,
    repo_path,
    run,
    sha256_file,
)
from synthetic_audio import load_config  # noqa: E402

POLICY = "shadow-blind-gru-generalization-v3-acoustic-diversity"
RESERVED_SHADOW_ARENA = "gru-independent-shadow-v2"
POSITIVE_EXAMPLE_WEIGHT = 2.45
EPOCHS = 12
LR_SCALE = 1.0
TRAINING_SEED_OFFSET = 9_000_019
EXPOSURE_REPEAT = 3
EXPECTED_SHARED_EXTRA_MANIFESTS = 2
EXPECTED_V3_EXTRA_MANIFESTS = 2


def strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_shared(path: pathlib.Path) -> dict:
    value = load_json(path)
    if value.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared model-family data policy mismatch")
    if value.get("formal_qualification_used") is not False:
        raise ValueError("shared data touched formal qualification")
    return value


def validate_replay(path: pathlib.Path) -> dict:
    evidence = load_json(path)
    if evidence.get("evidence_class") != "training-only-gru-generalization-resynthesis":
        raise ValueError("unexpected GRU replay evidence class")
    if evidence.get("policy") != "gru-generalization-resynthesis-v1":
        raise ValueError("unexpected GRU replay policy")
    if evidence.get("formal_qualification_used") is not False:
        raise ValueError("GRU replay touched formal qualification")
    if evidence.get("shadow_wav_bytes_copied") is not False:
        raise ValueError("GRU replay must not copy shadow WAV bytes")
    manifest = pathlib.Path(str(evidence["manifest"]))
    if not manifest.is_file() or sha256_file(manifest) != str(evidence["manifest_sha256"]):
        raise ValueError("GRU replay manifest binding failed")
    if int(evidence.get("examples", 0)) <= 0:
        raise ValueError("GRU replay is empty")
    return evidence


def validate_v3_expansion(path: pathlib.Path) -> tuple[dict, list[pathlib.Path]]:
    evidence = load_json(path)
    if evidence.get("evidence_class") != "training-only-gru-v3-independent-domain-expansion":
        raise ValueError("unexpected GRU V3 training-domain evidence class")
    if evidence.get("policy") != "gru-v3-independent-training-domain-expansion-v1":
        raise ValueError("unexpected GRU V3 training-domain policy")
    for key in (
        "formal_qualification_used",
        "shadow_used",
        "evaluation_splits_consumed",
        "source_calibration_failure_used",
        "source_v2_candidate_metrics_used",
        "source_v2_failure_diagnostics_used",
    ):
        if evidence.get(key) is not False:
            raise ValueError(f"GRU V3 training-domain evidence violated isolation: {key}")
    rows = evidence.get("manifests")
    if not isinstance(rows, list) or len(rows) != EXPECTED_V3_EXTRA_MANIFESTS:
        raise ValueError("GRU V3 requires exactly two independent train-only manifests")
    manifests: list[pathlib.Path] = []
    seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid GRU V3 training-domain manifest row")
        if row.get("formal_qualification_used") is not False:
            raise ValueError("GRU V3 train-only manifest touched formal qualification")
        if row.get("shadow_qualification_used") is not False:
            raise ValueError("GRU V3 train-only manifest touched shadow qualification")
        if row.get("evaluation_splits_consumed") is not False:
            raise ValueError("GRU V3 train-only manifest consumed an evaluation split")
        seed = int(row["seed"])
        if seed in seeds:
            raise ValueError("duplicate GRU V3 train-only acoustic seed")
        seeds.add(seed)
        manifest = pathlib.Path(str(row["manifest"]))
        if not manifest.is_file() or sha256_file(manifest) != str(row["manifest_sha256"]):
            raise ValueError("GRU V3 train-only manifest SHA binding failed")
        manifests.append(manifest)
    return evidence, manifests


def manifest_repeat(paths: list[pathlib.Path], count: int) -> list[pathlib.Path]:
    result: list[pathlib.Path] = []
    for path in paths:
        result.extend([path] * count)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--replay-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--v3-training-evidence", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    shared = load_shared(args.shared_data.resolve())
    replay = validate_replay(args.replay_evidence.resolve())
    v3_evidence, v3_extra = validate_v3_expansion(args.v3_training_evidence.resolve())
    if int(shared["formal_seed"]) != int(cfg["qualification_holdout_seed"]):
        raise ValueError("shared/formal seed identity drifted")

    registry = load_json(ROOT / "experiments" / "model_family" / "shadow_arena_registry.json")
    arenas = {str(v["name"]): v for v in registry["arenas"]}
    reserved = arenas.get(RESERVED_SHADOW_ARENA)
    if not isinstance(reserved, dict) or reserved.get("status") != "reserved-untouched":
        raise ValueError("GRU next shadow arena is not reserved-untouched")
    reserved_seeds = [int(v) for v in reserved["seeds"]]
    if reserved_seeds != list(range(951101, 951109)):
        raise ValueError("GRU reserved shadow seed contract drifted")

    forbidden = {int(cfg["qualification_holdout_seed"])} | {
        int(v) for v in cfg.get("retired_qualification_holdout_seeds", [])
    } | {
        int(seed)
        for arena in registry.get("arenas", [])
        if isinstance(arena, dict)
        for seed in arena.get("seeds", [])
    }
    if set(int(v) for v in v3_evidence["seeds"]) & forbidden:
        raise ValueError("GRU V3 acoustic expansion overlaps evaluation namespace")

    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("experimental GRU runner missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    canonical = pathlib.Path(shared["canonical_train_manifest"])
    static = pathlib.Path(shared["static_manifest"])
    adversarial = pathlib.Path(shared["adversarial_manifest"])
    shared_extras = [pathlib.Path(v["manifest"]) for v in shared.get("train_only_seed_manifests", [])]
    if len(shared_extras) != EXPECTED_SHARED_EXTRA_MANIFESTS:
        raise ValueError("GRU V3 requires exactly two shared predeclared train-only acoustic manifests")
    for row, path in zip(shared["train_only_seed_manifests"], shared_extras):
        if row.get("formal_qualification_used") is not False:
            raise ValueError("shared training-only manifest touched formal qualification")
        if row.get("shadow_qualification_used") is not False:
            raise ValueError("shared training-only manifest touched shadow qualification")
        if row.get("evaluation_splits_consumed") is not False:
            raise ValueError("shared training-only manifest consumed evaluation split")
        if sha256_file(path) != str(row["manifest_sha256"]):
            raise ValueError("shared training-only manifest SHA binding failed")

    base_manifests = [canonical, *shared_extras, *v3_extra]
    replay_manifests = [static, adversarial]
    failure = pathlib.Path(shared["failure_manifest"])
    if int(shared.get("failure_replay_examples", 0)) > 0:
        replay_manifests.append(failure)
    targeted = pathlib.Path(str(replay["manifest"]))
    manifests = [*base_manifests, *manifest_repeat(replay_manifests, EXPOSURE_REPEAT), *([targeted] * EXPOSURE_REPEAT)]
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
            "--epochs", str(EPOCHS),
            "--batch-size", str(int(train_cfg.get("batch_size", 16))),
            "--lr", str(float(train_cfg.get("lr", 0.001)) * LR_SCALE),
            "--seed", str(int(cfg.get("seed", 1337)) + TRAINING_SEED_OFFSET),
            "--positive-example-weight", str(POSITIVE_EXAMPLE_WEIGHT),
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

    thresholds = [float(v) for v in cfg["calibration"]["thresholds"]]
    coordinate_rounds = int(cfg["calibration"]["coordinate_rounds"])
    gates = gate_values(cfg["domain_gates"])
    calibrated, pack, cal_base, cal_domains = calibrate(
        runner=runner,
        model=model,
        tokens=tokens,
        source_keywords=keywords,
        references=pathlib.Path(shared["canonical_calibration_references"]),
        output=candidate / "calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
    )
    test_base, test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=pathlib.Path(shared["canonical_test_references"]),
        output=candidate / "test",
    )
    qual_base, qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=pathlib.Path(shared["canonical_qualification_references"]),
        output=candidate / "qualification",
    )
    cal_gate = strict(cal_base, cal_domains, gates)
    test_gate = strict(test_base, test_domains, gates)
    qual_gate = strict(qual_base, qual_domains, gates)
    score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)

    record = {
        "round": 0,
        "frontend": str(shared["source_frontend"]),
        "candidate": 0,
        "score": score,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "checkpoint": str(checkpoint),
        "provenance": str(provenance),
        "provenance_sha256": sha256_file(provenance),
        "keywords": str(calibrated),
        "pack": str(pack),
        "calibration": cal_base,
        "calibration_domains": cal_domains,
        "test": test_base,
        "test_domains": test_domains,
        "calibration_gate": cal_gate,
        "test_gate": test_gate,
        "formal_qualification_used": False,
    }
    manifest = {
        "schema_version": 1,
        "evidence_class": "development-only-shadow-blind-gru-generalization-v3",
        "policy": POLICY,
        "development_qualified": bool(cal_gate and test_gate),
        "qualification_qualified": bool(qual_gate),
        "qualified": bool(cal_gate and test_gate and qual_gate),
        "records": [record],
        "candidate_selection": {
            "policy": "single-predeclared-candidate-no-shadow-selection",
            "eligible_rounds": [0] if cal_gate and test_gate else [],
            "selected_round": 0,
            "selected_frontend": str(shared["source_frontend"]),
            "selected_score": score,
            "qualification_used_for_selection": False,
            "shadow_used_for_selection": False,
        },
        "qualification": qual_base,
        "qualification_domains": qual_domains,
        "formal_qualification_used": False,
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
        "calibration_feedback_used_for_training": False,
        "v2_candidate_metrics_used_for_training": False,
        "v2_failure_diagnostics_used_for_training": False,
        "reserved_untouched_shadow_arena": RESERVED_SHADOW_ARENA,
        "reserved_untouched_shadow_seeds": reserved_seeds,
        "parameters": {
            "positive_example_weight": POSITIVE_EXAMPLE_WEIGHT,
            "epochs": EPOCHS,
            "lr_scale": LR_SCALE,
            "training_seed_offset": TRAINING_SEED_OFFSET,
            "full_train_seed_count": len(base_manifests),
            "shared_train_only_seed_count": len(shared_extras),
            "v3_independent_train_only_seed_count": len(v3_extra),
            "replay_exposure_repeat": EXPOSURE_REPEAT,
        },
        "calibration": cal_base,
        "test": test_base,
        "development_qualification": qual_base,
        "calibration_gate": cal_gate,
        "test_gate": test_gate,
        "development_qualification_gate": qual_gate,
        "model_sha256": sha256_file(model),
        "pack_sha256": sha256_file(pack),
        "model_bytes": model.stat().st_size,
        "parameter_count": parameter_count,
        "estimated_macs_per_frame": 18752,
        "training_manifest_count": len(manifests),
        "unique_training_manifest_count": len({str(v.resolve()) for v in manifests}),
        "replay_evidence_sha256": sha256_file(args.replay_evidence.resolve()),
        "v3_training_evidence_sha256": sha256_file(args.v3_training_evidence.resolve()),
        "shared_data_policy": str(shared["policy"]),
    }
    (output / "candidate-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if cal_gate and test_gate and qual_gate else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
