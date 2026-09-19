#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"

EVIDENCE_CLASS = "kws-v2-phonetic-representation-scorecard-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def run(command: list[str], log: pathlib.Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; see {log}"
        )


def reset_dir(path: pathlib.Path) -> pathlib.Path:
    root = path.resolve()
    if root in {ROOT.resolve(), ROOT.parent.resolve(), pathlib.Path.home().resolve()}:
        raise ValueError(f"unsafe work-dir: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def finite_metric(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {key}")
    return float(value)


def runtime_points(curve: dict, budgets: list[float]) -> dict:
    rows = curve.get("operating_curve")
    if not isinstance(rows, list) or not rows:
        raise ValueError("threshold curve is empty")
    result: dict[str, dict | None] = {}
    for budget in budgets:
        eligible = [
            row
            for row in rows
            if float(row["calibration"]["far_per_hour"]) <= budget
        ]
        if not eligible:
            result[str(budget)] = None
            continue
        selected = min(
            eligible,
            key=lambda row: (
                float(row["calibration"]["frr"]),
                float(row["calibration"]["far_per_hour"]),
                float(row["threshold"]),
            ),
        )
        result[str(budget)] = {
            "threshold": float(selected["threshold"]),
            "calibration": selected["calibration"],
            "test": selected["test"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the research-only phonetic representation-v2 GRU."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    dataset = args.dataset.resolve()
    runner = args.runner.resolve()
    if not config_path.is_file() or not dataset.is_dir() or not runner.is_file():
        raise ValueError("config, dataset and runner must exist")

    config = load_object(config_path)
    if config.get("policy") != "kws-v2-phonetic-representation-experiment-v1":
        raise ValueError("phonetic representation experiment identity mismatch")
    if config.get("evidence_scope") != "research-only":
        raise ValueError("phonetic representation experiment must remain research-only")
    authority = config.get("authority")
    if not isinstance(authority, dict) or authority.get("promotion_allowed") is not False:
        raise ValueError("phonetic representation experiment cannot grant promotion authority")

    receipt_path = dataset / "representation-receipt.json"
    receipt = load_object(receipt_path)
    if (
        receipt.get("evidence_class")
        != "kws-v2-phonetic-representation-dataset-v1"
        or receipt.get("evidence_scope") != "research-only"
        or receipt.get("qualification_split_consumed") is not False
        or receipt.get("target_vocab_size")
        != int(config["representation"]["expected_vocab_size"])
    ):
        raise ValueError("phonetic representation dataset receipt mismatch")
    expected_counts = {
        "train": (768, 128, 640),
        "calibration": (384, 64, 320),
        "test": (384, 64, 320),
    }
    for split, expected in expected_counts.items():
        row = receipt["splits"][split]
        actual = (
            int(row["recordings"]),
            int(row["wake_recordings"]),
            int(row["nonwake_recordings"]),
        )
        if actual != expected:
            raise ValueError(
                f"phonetic representation split counts drifted: {split} {actual} != {expected}"
            )

    representation = config["representation"]
    tokens = (ROOT / str(representation["target_tokens"])).resolve()
    keywords = (ROOT / str(representation["target_keywords"])).resolve()
    if not tokens.is_file() or not keywords.is_file():
        raise ValueError("phonetic representation tokens/keywords missing")
    if sha256_file(tokens) != str(receipt["target_tokens_sha256"]):
        raise ValueError("phonetic token bytes do not match representation dataset")
    if sha256_file(keywords) != str(receipt["target_keywords_sha256"]):
        raise ValueError("phonetic keyword bytes do not match representation dataset")

    work = reset_dir(args.work_dir)
    logs = work / "logs"
    train = config["train"]
    model = config["model"]
    checkpoint = work / "model.pt"

    run(
        [
            sys.executable,
            str(TRAINING / "train_gru_ctc.py"),
            "--manifest",
            str(dataset / "train.tsv"),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--frontend",
            str(model["frontend"]),
            "--feature-dim",
            str(int(model["feature_dim"])),
            "--hidden-dim",
            str(int(model["hidden_dim"])),
            "--epochs",
            str(int(train["epochs"])),
            "--batch-size",
            str(int(train["batch_size"])),
            "--lr",
            str(float(train["learning_rate"])),
            "--seed",
            str(int(train["seed"])),
            "--positive-example-weight",
            str(float(train["target_bearing_weight"])),
            "--wake-example-weight",
            str(float(train["wake_example_weight"])),
            "--ordered-token-loss-weight",
            str(float(train["ordered_token_loss_weight"])),
            "--keyword-sequence-margin-loss-weight",
            str(float(train["keyword_sequence_margin_loss_weight"])),
            "--prefix-completion-loss-weight",
            str(float(train["prefix_completion_loss_weight"])),
            "--recurrent-release-loss-weight",
            str(float(train["recurrent_release_loss_weight"])),
            "--output",
            str(checkpoint),
        ],
        logs / "train.log",
    )

    thresholds = [float(value) for value in config["evaluation"]["thresholds"]]
    float_path = work / "float-ctc-confidence.json"
    run(
        [
            sys.executable,
            str(TRAINING / "diagnose_float_ctc_confidence.py"),
            "--family",
            "gru",
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--split-manifest",
            f"calibration={dataset / 'calibration.tsv'}",
            "--split-manifest",
            f"test={dataset / 'test.tsv'}",
            "--thresholds",
            *[str(value) for value in thresholds],
            "--batch-size",
            str(int(train["batch_size"])),
            "--output",
            str(float_path),
        ],
        logs / "float-ctc-confidence.log",
    )

    exported = work / "model.kwg"
    run(
        [
            sys.executable,
            str(TRAINING / "export_gru_model.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(exported),
        ],
        logs / "export.log",
    )

    threshold_config = work / "threshold-diagnostic-config.json"
    threshold_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_class": "kws-v2-research-threshold-gates-v1",
                "domain_gates": {
                    "max_frr": 0.0,
                    "max_far_per_hour": 0.0,
                    "max_p95_latency_ms": 800.0,
                    "max_far_frr": 0.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    curve_path = work / "threshold-operating-curve.json"
    run(
        [
            sys.executable,
            str(TOOLS / "diagnose_kws_threshold_operating_curve.py"),
            "--runner",
            str(runner),
            "--model",
            str(exported),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--config",
            str(threshold_config),
            "--calibration-references",
            str(dataset / "calibration.references.jsonl"),
            "--test-references",
            str(dataset / "test.references.jsonl"),
            "--thresholds",
            *[str(value) for value in thresholds],
            "--diagnostic-round-selection-policy",
            "phonetic-representation-v2-fixed-checkpoint-v1",
            "--work-dir",
            str(work / "threshold-sweep"),
            "--output",
            str(curve_path),
        ],
        logs / "threshold.log",
    )

    float_diag = load_object(float_path)
    curve = load_object(curve_path)
    test = float_diag["splits"]["test"]
    calibration = float_diag["splits"]["calibration"]
    test_gap = finite_metric(test, "separation_gap_p10_wake_minus_p99_nonwake")
    calibration_gap = finite_metric(
        calibration, "separation_gap_p10_wake_minus_p99_nonwake"
    )
    baseline_gap = float(config["baseline"]["test_float_gap"])
    gap_delta = test_gap - baseline_gap
    rules = config["decision_rules"]
    promising = (
        gap_delta
        >= float(rules["representation_promising_if_test_gap_improves_by_at_least"])
        and test_gap > float(rules["representation_promising_if_test_gap_above"])
    )
    runtime = runtime_points(curve, [30.0, 60.0, 120.0])

    scorecard = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "qualification_split_consumed": False,
        "promotion_allowed": False,
        "shipping_metric": False,
        "config_sha256": sha256_file(config_path),
        "representation_dataset_receipt_sha256": sha256_file(receipt_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_sha256": sha256_file(exported),
        "threshold_diagnostic_config_sha256": sha256_file(threshold_config),
        "vocab_size": int(receipt["target_vocab_size"]),
        "model": config["model"],
        "train": config["train"],
        "baseline": config["baseline"],
        "float_ctc": {
            "calibration_gap": calibration_gap,
            "test_gap": test_gap,
            "test_gap_delta_vs_four_token_m2": gap_delta,
            "test_wake_p10": test["positive_true_keyword_confidence"]["p10"],
            "test_nonwake_p99": test["nonwake_max_keyword_confidence"]["p99"],
            "test_tokenized_nonwake_p99": test[
                "tokenized_nonwake_max_keyword_confidence"
            ]["p99"],
            "test_operating_curve": test["operating_curve"],
        },
        "runtime_by_far_budget": runtime,
        "representation_promising": promising,
        "next_architecture_action": (
            str(rules["promising_next"])
            if promising
            else str(rules["weak_next"])
        ),
    }
    score_path = work / "phonetic-representation-scorecard.json"
    score_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"phonetic-representation-v2: gap={test_gap:.6f} "
        f"delta={gap_delta:.6f} promising={promising} "
        f"next={scorecard['next_architecture_action']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
