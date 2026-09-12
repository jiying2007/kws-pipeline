#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"

from adversarial_lexicon import mine_adversarial_lexicon
from development_failure_replay import render_development_failure_replay
from hard_negative_replay import render_hard_negative_replay
from iterate_domain import (
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
from render_domains import render_domain_dataset
from render_qualification_holdout import require_strict_development_candidate
from synthetic_audio import load_config

POLICY = "development-only-tiny-gru-ab-v1"


def selected_record(manifest: dict) -> dict:
    selection = manifest["candidate_selection"]
    round_id = int(selection["selected_round"])
    frontend = str(selection["selected_frontend"])
    rows = [
        row
        for row in manifest["records"]
        if isinstance(row, dict)
        and int(row["round"]) == round_id
        and str(row["frontend"]) == frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("cannot resolve strict RNN source checkpoint for GRU A/B")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only Tiny-GRU vs RNN generalization probe.")
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--rnn-work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    rnn_work = args.rnn_work_dir.resolve()
    runner = args.runner.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    if not runner.is_file():
        raise ValueError("experimental GRU runner does not exist")
    manifest = require_strict_development_candidate(rnn_work)
    source = selected_record(manifest)
    source_checkpoint = pathlib.Path(str(source["checkpoint"])).resolve()
    if not source_checkpoint.is_file():
        raise ValueError(f"RNN source checkpoint is missing: {source_checkpoint}")
    source_round = int(source["round"])
    frontend = str(source["frontend"])
    probe_round = max(int(row["round"]) for row in manifest["records"]) + 1
    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    curriculum = manifest.get("final_curriculum")
    if not isinstance(curriculum, dict):
        curriculum = None

    dataset = output / "dataset"
    render_domain_dataset(config_path, dataset, curriculum_weights=curriculum)
    static = render_hard_negative_replay(
        config_path,
        output / "static-replay",
        round_index=probe_round,
        curriculum_weights=curriculum,
    )
    adversarial = mine_adversarial_lexicon(
        config_path,
        source_checkpoint,
        output / "adversarial-lexicon",
        round_index=probe_round,
        frontend=frontend,
    )
    failure = render_development_failure_replay(
        config_path,
        list(manifest.get("records", [])),
        rnn_work,
        output / "development-failure-replay",
    )
    if bool(adversarial.get("formal_qualification_used", True)):
        raise ValueError("GRU A/B adversarial mining touched formal qualification")
    if bool(failure.get("formal_qualification_used", True)):
        raise ValueError("GRU A/B failure replay touched formal qualification")

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    candidate = output / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate / "model.pt"
    model = candidate / "model.kwg"
    command = [
        sys.executable,
        str(TRAINING / "train_gru_ctc.py"),
        "--manifest",
        str(dataset / "train.tsv"),
        "--manifest",
        str(static["manifest"]),
        "--manifest",
        str(adversarial["manifest"]),
    ]
    if int(failure.get("examples", 0)) > 0:
        command.extend(["--manifest", str(failure["manifest"])])
    command.extend(
        [
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--frontend",
            frontend,
            "--feature-dim",
            str(int(model_cfg.get("feature_dim", 32))),
            "--hidden-dim",
            str(int(model_cfg.get("hidden_dim", 64))),
            "--epochs",
            str(int(train_cfg.get("epochs", 36))),
            "--batch-size",
            str(int(train_cfg.get("batch_size", 16))),
            "--lr",
            str(float(train_cfg.get("lr", 0.001))),
            "--seed",
            str(int(cfg.get("seed", 1337)) + 9_000_019),
            "--output",
            str(checkpoint),
        ]
    )
    run(command)
    run(
        [
            sys.executable,
            str(TRAINING / "export_gru_model.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(model),
        ]
    )
    provenance = pathlib.Path(str(model) + ".provenance.json")

    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    gates = gate_values(cfg.get("domain_gates", {}))
    calibrated, pack, cal_base, cal_domains = calibrate(
        runner=runner,
        model=model,
        tokens=tokens,
        source_keywords=keywords,
        references=dataset / "calibration.references.jsonl",
        output=candidate / "calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
    )
    test_base, test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=dataset / "test.references.jsonl",
        output=candidate / "test",
    )
    cal_gate = strict(cal_base, cal_domains, gates)
    test_gate = strict(test_base, test_domains, gates)
    score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)

    qualification_dataset = output / "qualification-dataset"
    render_domain_dataset(config_path, qualification_dataset, curriculum_weights=None)
    qual_base, qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=qualification_dataset / "qualification.references.jsonl",
        output=candidate / "qualification",
    )
    qual_gate = strict(qual_base, qual_domains, gates)

    record = {
        "round": 0,
        "stage": POLICY,
        "frontend": frontend,
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
        "architecture": "tiny-gru-v1",
        "formal_qualification_used": False,
    }
    ab_manifest = {
        "schema_version": 1,
        "evidence_class": "development-only-model-v2-ab",
        "development_qualified": bool(cal_gate and test_gate),
        "qualification_qualified": bool(qual_gate),
        "qualified": bool(cal_gate and test_gate and qual_gate),
        "records": [record],
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [0] if cal_gate and test_gate else [],
            "selected_round": 0,
            "selected_frontend": frontend,
            "selected_score": score,
            "qualification_used_for_selection": False,
        },
        "qualification": qual_base,
        "qualification_domains": qual_domains,
        "source_rnn_manifest_sha256": sha256_file(rnn_work / "domain-loop-manifest.json"),
        "source_rnn_round": source_round,
        "source_rnn_checkpoint_sha256": sha256_file(source_checkpoint),
        "formal_qualification_used": False,
    }
    (output / "domain-loop-manifest.json").write_text(
        json.dumps(ab_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
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

    shadow_result = None
    shadow_returncode = None
    if cal_gate and test_gate and qual_gate:
        completed = subprocess.run(
            [
                sys.executable,
                str(TRAINING / "shadow_qualification.py"),
                "--config",
                str(config_path),
                "--runner",
                str(runner),
                "--work-dir",
                str(output),
                "--output",
                str(output / "shadow-qualification"),
            ],
            check=False,
        )
        shadow_returncode = int(completed.returncode)
        shadow_path = output / "shadow-qualification" / "summary.json"
        if shadow_path.is_file():
            shadow_result = json.loads(shadow_path.read_text(encoding="utf-8"))

    summary = {
        "schema_version": 1,
        "policy": POLICY,
        "experimental": True,
        "architecture": "tiny-gru-v1",
        "formal_qualification_used": False,
        "source_rnn_round": source_round,
        "source_rnn_checkpoint_sha256": sha256_file(source_checkpoint),
        "calibration": cal_base,
        "test": test_base,
        "development_qualification": qual_base,
        "calibration_gate": cal_gate,
        "test_gate": test_gate,
        "development_qualification_gate": qual_gate,
        "shadow_returncode": shadow_returncode,
        "shadow": shadow_result,
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "failure_replay_examples": int(failure.get("examples", 0)),
        "static_replay_examples": int(static.get("examples", 0)),
    }
    (output / "model-v2-ab-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if shadow_result is not None and bool(shadow_result.get("qualified")) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
