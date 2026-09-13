#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
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

POLICY = "development-only-gru-tuning-v1"


def strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def load_shared(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared model-family data policy mismatch")
    if bool(value.get("formal_qualification_used", True)):
        raise ValueError("shared data touched formal qualification")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--positive-example-weight", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--lr-scale", required=True, type=float)
    args = parser.parse_args()

    cfg = load_config(args.config.resolve())
    shared = load_shared(args.shared_data.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("experimental GRU runner missing")
    if int(shared["formal_seed"]) != int(cfg["qualification_holdout_seed"]):
        raise ValueError("shared/formal seed identity drifted")
    if args.positive_example_weight <= 0.0 or args.epochs <= 0 or args.lr_scale <= 0.0:
        raise ValueError("GRU tuning parameters must be positive")

    manifests = [
        pathlib.Path(shared["canonical_train_manifest"]),
        pathlib.Path(shared["static_manifest"]),
        pathlib.Path(shared["adversarial_manifest"]),
    ]
    if int(shared.get("failure_replay_examples", 0)) > 0:
        manifests.append(pathlib.Path(shared["failure_manifest"]))
    for path in manifests:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing GRU tuning manifest: {path}")

    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    candidate_dir = output / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / "model.pt"
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
            "--epochs", str(args.epochs),
            "--batch-size", str(int(train_cfg.get("batch_size", 16))),
            "--lr", str(float(train_cfg.get("lr", 0.001)) * args.lr_scale),
            "--seed", str(int(cfg.get("seed", 1337)) + 9_000_019),
            "--positive-example-weight", str(args.positive_example_weight),
            "--output", str(checkpoint),
        ]
    )
    run(command)
    model = candidate_dir / "model.kwg"
    run([
        sys.executable,
        str(TRAINING / "export_gru_model.py"),
        "--checkpoint", str(checkpoint),
        "--tokens", str(tokens),
        "--output", str(model),
    ])
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
        output=candidate_dir / "calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
    )
    test_base, test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=pathlib.Path(shared["canonical_test_references"]),
        output=candidate_dir / "test",
    )
    qual_base, qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=pathlib.Path(shared["canonical_qualification_references"]),
        output=candidate_dir / "qualification",
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
        "evidence_class": "development-only-gru-tuning",
        "development_qualified": bool(cal_gate and test_gate),
        "qualification_qualified": bool(qual_gate),
        "qualified": bool(cal_gate and test_gate and qual_gate),
        "records": [record],
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [0] if cal_gate and test_gate else [],
            "selected_round": 0,
            "selected_frontend": str(shared["source_frontend"]),
            "selected_score": score,
            "qualification_used_for_selection": False,
        },
        "qualification": qual_base,
        "qualification_domains": qual_domains,
        "formal_qualification_used": False,
    }
    (output / "domain-loop-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
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

    shadow = None
    shadow_rc = None
    if cal_gate and test_gate and qual_gate:
        completed = subprocess.run(
            [
                sys.executable,
                str(TRAINING / "shadow_qualification.py"),
                "--config", str(args.config.resolve()),
                "--runner", str(runner),
                "--work-dir", str(output),
                "--output", str(output / "shadow-qualification"),
            ],
            check=False,
        )
        shadow_rc = int(completed.returncode)
        shadow_path = output / "shadow-qualification" / "summary.json"
        if shadow_path.is_file():
            shadow = json.loads(shadow_path.read_text(encoding="utf-8"))

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parameter_count = sum(int(t.numel()) for t in checkpoint_payload["state_dict"].values())
    failures = [] if shadow is None else [row for row in shadow.get("results", []) if not row.get("qualified")]
    summary = {
        "schema_version": 1,
        "policy": POLICY,
        "candidate": args.candidate,
        "experimental": True,
        "formal_qualification_used": False,
        "parameters": {
            "positive_example_weight": args.positive_example_weight,
            "epochs": args.epochs,
            "lr_scale": args.lr_scale,
        },
        "calibration": cal_base,
        "test": test_base,
        "development_qualification": qual_base,
        "calibration_gate": cal_gate,
        "test_gate": test_gate,
        "development_qualification_gate": qual_gate,
        "shadow_returncode": shadow_rc,
        "shadow": shadow,
        "shadow_failure_count": len(failures),
        "model_bytes": model.stat().st_size,
        "parameter_count": parameter_count,
        "estimated_macs_per_frame": 18752,
        "training_manifest_count": len(manifests),
        "source_checkpoint_sha256": str(shared["source_checkpoint_sha256"]),
        "shared_data_policy": str(shared["policy"]),
    }
    (output / "tuning-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
