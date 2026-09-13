#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
HISTORICAL_MODEL_SHA256 = "07bc58753ff5480ffa2361b286bff314ae321453fcea674c90b10e71c7132ccd"
POLICY = "rnn-trainer-differential-reproduction-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def manifest_paths(shared: dict) -> list[pathlib.Path]:
    paths = [
        pathlib.Path(shared["canonical_train_manifest"]),
        pathlib.Path(shared["static_manifest"]),
        pathlib.Path(shared["adversarial_manifest"]),
    ]
    if int(shared.get("failure_replay_examples", 0)) > 0:
        paths.append(pathlib.Path(shared["failure_manifest"]))
    paths.extend(
        pathlib.Path(item["manifest"])
        for item in shared.get("train_only_seed_manifests", [])
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing shared training manifest: {path}")
    return paths


def trainer_command(
    *,
    trainer: pathlib.Path,
    manifests: list[pathlib.Path],
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    warm_start: pathlib.Path,
    output: pathlib.Path,
    feature_dim: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    tuned: bool,
) -> list[str]:
    command = [sys.executable, str(trainer)]
    if not tuned:
        command.extend(["--variant", "rnn-multiseed-terminal"])
    for manifest in manifests:
        command.extend(["--manifest", str(manifest)])
    command.extend(
        [
            "--tokens", str(tokens),
            "--keywords", str(keywords),
            "--frontend", "logmel",
            "--feature-dim", str(feature_dim),
            "--hidden-dim", str(hidden_dim),
            "--epochs", "12",
            "--batch-size", str(batch_size),
            "--lr", str(learning_rate),
            "--seed", str(seed),
            "--warm-start", str(warm_start),
        ]
    )
    if tuned:
        command.extend(
            [
                "--terminal-margin-log", "0.15",
                "--terminal-loss-weight", "0.10",
                "--negative-example-weight", "1.00",
            ]
        )
    command.extend(["--output", str(output)])
    return command


def state_dict_comparison(left_path: pathlib.Path, right_path: pathlib.Path) -> dict:
    left = torch.load(left_path, map_location="cpu", weights_only=True)["state_dict"]
    right = torch.load(right_path, map_location="cpu", weights_only=True)["state_dict"]
    if set(left) != set(right):
        return {
            "equal": False,
            "keys_equal": False,
            "left_only": sorted(set(left) - set(right)),
            "right_only": sorted(set(right) - set(left)),
        }
    rows = []
    all_equal = True
    for name in sorted(left):
        a = left[name].detach().cpu()
        b = right[name].detach().cpu()
        if tuple(a.shape) != tuple(b.shape):
            rows.append({"name": name, "shape_equal": False})
            all_equal = False
            continue
        equal = bool(torch.equal(a, b))
        max_abs = float((a - b).abs().max().item()) if a.numel() else 0.0
        rows.append(
            {
                "name": name,
                "shape_equal": True,
                "equal": equal,
                "max_abs_diff": max_abs,
            }
        )
        all_equal = all_equal and equal
    return {"equal": all_equal, "keys_equal": True, "parameters": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    cfg = load_json(args.config.resolve())
    shared = load_json(args.shared_data.resolve())
    if shared.get("policy") != "model-family-shared-development-data-v1":
        raise ValueError("shared model-family data policy mismatch")
    if bool(shared.get("formal_qualification_used", True)):
        raise ValueError("differential reproduction must not use formal qualification")
    if int(shared["formal_seed"]) != int(cfg["qualification_holdout_seed"]):
        raise ValueError("shared/formal seed identity drifted")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = manifest_paths(shared)
    if len(manifests) != 5 or len(shared.get("train_only_seed_manifests", [])) != 2:
        raise ValueError("historical RNN reproduction requires the exact five-manifest contract")

    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    warm_start = pathlib.Path(shared["source_checkpoint"])
    if not warm_start.is_file():
        raise ValueError("shared warm-start checkpoint is missing")
    expected_source_sha = str(shared["source_checkpoint_sha256"])
    actual_source_sha = sha256_file(warm_start)
    if actual_source_sha != expected_source_sha:
        raise ValueError("shared warm-start checkpoint SHA drifted")

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    feature_dim = int(model_cfg.get("feature_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    batch_size = int(train_cfg.get("batch_size", 16))
    learning_rate = float(train_cfg.get("lr", 0.001)) * 0.5
    seed = int(cfg.get("seed", 1337)) + 5_000_003

    historical_checkpoint = output / "historical-variant.pt"
    tuned_checkpoint = output / "tuned-equivalent.pt"
    run(
        trainer_command(
            trainer=ROOT / "experiments" / "model_family" / "train_rnn_variant.py",
            manifests=manifests,
            tokens=tokens,
            keywords=keywords,
            warm_start=warm_start,
            output=historical_checkpoint,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            tuned=False,
        )
    )
    run(
        trainer_command(
            trainer=ROOT / "experiments" / "model_family" / "train_rnn_tuned.py",
            manifests=manifests,
            tokens=tokens,
            keywords=keywords,
            warm_start=warm_start,
            output=tuned_checkpoint,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            tuned=True,
        )
    )

    historical_model = output / "historical-variant.kwm"
    tuned_model = output / "tuned-equivalent.kwm"
    for checkpoint, model in (
        (historical_checkpoint, historical_model),
        (tuned_checkpoint, tuned_model),
    ):
        run(
            [
                sys.executable,
                str(ROOT / "training" / "export_model.py"),
                "--checkpoint", str(checkpoint),
                "--tokens", str(tokens),
                "--output", str(model),
            ]
        )

    historical_model_sha = sha256_file(historical_model)
    tuned_model_sha = sha256_file(tuned_model)
    historical_reproduced = historical_model_sha == HISTORICAL_MODEL_SHA256
    state = state_dict_comparison(historical_checkpoint, tuned_checkpoint)
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "formal_qualification_used": False,
        "training_seed": seed,
        "training_seed_offset": 5_000_003,
        "epochs": 12,
        "terminal_margin_log": 0.15,
        "terminal_loss_weight": 0.10,
        "negative_example_weight": 1.00,
        "manifest_count": len(manifests),
        "manifest_sha256": [sha256_file(path) for path in manifests],
        "source_checkpoint_sha256": actual_source_sha,
        "historical_expected_model_sha256": HISTORICAL_MODEL_SHA256,
        "historical_model_sha256": historical_model_sha,
        "historical_reproduced": historical_reproduced,
        "tuned_model_sha256": tuned_model_sha,
        "model_byte_identical": historical_model_sha == tuned_model_sha,
        "state_dict": state,
    }
    report = output / "differential-reproduction.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if not historical_reproduced:
        raise ValueError("historical variant trainer failed byte-identical model reproduction")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
