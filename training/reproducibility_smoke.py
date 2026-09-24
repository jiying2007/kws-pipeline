#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import wave

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pcm16(path: pathlib.Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(payload)


def square_segment(period: int, length: int, amplitude: int) -> list[int]:
    half = max(1, period // 2)
    return [
        amplitude if ((index // half) % 2) == 0 else -amplitude
        for index in range(length)
    ]


def render_tokens(tokens: list[int], variant: int) -> list[int]:
    periods = {1: 64, 2: 52, 3: 44, 4: 36}
    lead = [0] * (1200 + variant * 17)
    tail = [0] * (1600 + variant * 13)
    body: list[int] = []
    for position, token in enumerate(tokens):
        amplitude = 9000 + token * 700 + variant * 31
        body.extend(square_segment(periods[token], 2200 + position * 40, amplitude))
        body.extend([0] * (480 + ((position + variant) % 3) * 40))
    samples = lead + body + tail
    if len(samples) < 16000:
        samples.extend([0] * (16000 - len(samples)))
    return samples[:20000]


def fixture(work: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    tokens = work / "tokens.txt"
    keywords = work / "keywords.tsv"
    manifest = work / "train.tsv"
    tokens.write_text(
        "<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n",
        encoding="utf-8",
    )
    keywords.write_text(
        "1\twake-one\t0.55\tni3 hao3 xiao3 wo1\n"
        "2\twake-two\t0.55\txiao3 wo1 xiao3 wo1\n",
        encoding="utf-8",
    )
    rows: list[tuple[pathlib.Path, list[int]]] = []
    sequences = [
        [1, 2, 3, 4],
        [3, 4, 3, 4],
        [2, 1, 3, 4],
        [3, 4, 4, 3],
        [1, 2, 3],
        [3, 4, 3],
    ]
    for variant in range(18):
        sequence = sequences[variant % len(sequences)]
        path = work / "wav" / f"fixture-{variant:02d}.wav"
        write_pcm16(path, render_tokens(sequence, variant % 5))
        rows.append((path, sequence))
    manifest.write_text(
        "".join(
            f"{path.relative_to(work).as_posix()}\t{' '.join(str(value) for value in target)}\n"
            for path, target in rows
        ),
        encoding="utf-8",
    )
    return tokens, keywords, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("reproducibility smoke requires PYTHONHASHSEED=0")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    tokens, keywords, manifest = fixture(work)
    checkpoint = work / "model.pt"
    model = work / "model.kwm"

    train_command = [
        sys.executable,
        str(TRAINING / "feature_cached_trainer.py"),
        "--trainer",
        "rnn",
        "--feature-cache-max-items",
        "128",
        "--",
        "--manifest",
        str(manifest),
        "--tokens",
        str(tokens),
        "--keywords",
        str(keywords),
        "--frontend",
        "logmel",
        "--feature-dim",
        "16",
        "--hidden-dim",
        "16",
        "--epochs",
        "3",
        "--batch-size",
        "6",
        "--lr",
        "0.001",
        "--seed",
        "424242",
        "--ordered-token-scope",
        "all-nonempty-targets-v1",
        "--sequence-margin-negative-policy",
        "runtime-executable-v1",
        "--output",
        str(checkpoint),
    ]
    subprocess.check_call(train_command, cwd=ROOT)
    subprocess.check_call(
        [
            sys.executable,
            str(TRAINING / "export_model.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(model),
        ],
        cwd=ROOT,
    )

    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    environment = checkpoint_payload.get("training_environment", {})
    if not isinstance(environment, dict):
        raise ValueError("checkpoint training_environment is missing")
    result = {
        "schema_version": 1,
        "evidence_class": "cross-runner-training-reproducibility-smoke-v1",
        "model_sha256": sha256_file(model),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fixture_manifest_sha256": sha256_file(manifest),
        "torch_num_threads": environment.get("torch_num_threads"),
        "torch_num_interop_threads": environment.get("torch_num_interop_threads"),
        "torch_version": environment.get("torch_version"),
        "cpu_runtime": environment.get("cpu_runtime"),
        "torch_runtime": environment.get("torch_runtime"),
        "training_code_sha256": environment.get("training_code_sha256"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "sitecustomize_loaded": (
            environment.get("torch_runtime", {})
            .get("thread_env", {})
            .get("KWS_SITECUSTOMIZE_LOADED")
        ),
    }
    if result["torch_num_threads"] != 1 or result["torch_num_interop_threads"] != 1:
        raise ValueError("reproducibility smoke did not use single-thread torch topology")
    thread_env = result["torch_runtime"].get("thread_env", {})
    expected_env = {
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "MKL_CBWR": "COMPATIBLE",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ATEN_CPU_CAPABILITY": "default",
        "PYTHONHASHSEED": "0",
    }
    actual_required = {key: thread_env.get(key) for key in expected_env}
    if actual_required != expected_env:
        raise ValueError(
            f"unexpected deterministic CPU environment: {actual_required}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "training-reproducibility-smoke "
        + json.dumps(
            {
                "model_sha256": result["model_sha256"],
                "fixture_manifest_sha256": result["fixture_manifest_sha256"],
                "torch_num_threads": result["torch_num_threads"],
                "torch_num_interop_threads": result["torch_num_interop_threads"],
                "cpu_model": result["cpu_runtime"].get("model"),
                "torch_config_sha256": result["torch_runtime"].get("config_sha256"),
                "torch_parallel_info_sha256": result["torch_runtime"].get("parallel_info_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
