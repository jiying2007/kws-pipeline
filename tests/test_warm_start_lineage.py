#!/usr/bin/env python3
"""Verify that a formal warm start binds its exact source and rejects research weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import torch
from export_model import training_metadata
from train_ctc import validate_warm_start


def write_tone(path: pathlib.Path, hz: float) -> None:
    samples = [int(8000 * math.sin(2 * math.pi * hz * i / 16000)) for i in range(16000)]
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def train(manifest: pathlib.Path, output: pathlib.Path, warm: pathlib.Path | None = None) -> None:
    command = [
        sys.executable, str(ROOT / "training/train_ctc.py"),
        "--manifest", str(manifest),
        "--tokens", str(ROOT / "keywords/tokens.example.txt"),
        "--keywords", str(ROOT / "keywords/zh_cn_example.tsv"),
        "--output", str(output),
        "--feature-dim", "32", "--hidden-dim", "64",
        "--epochs", "1", "--batch-size", "3", "--seed", "1337",
    ]
    if warm is not None:
        command.extend(["--warm-start", str(warm)])
    subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True, timeout=60)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="kws-warm-lineage-") as tmp:
        root = pathlib.Path(tmp)
        manifest = root / "train.tsv"
        rows = []
        for index, (hz, target) in enumerate(((300, "1 2 3 4"), (450, "3 4 3 4"), (600, "1 2 3"))):
            path = root / f"row-{index}.wav"
            write_tone(path, hz)
            rows.append(f"{path}\t{target}\n")
        manifest.write_text("".join(rows), encoding="utf-8")
        source, result = root / "source.pt", root / "result.pt"
        train(manifest, source)
        train(manifest, result, source)
        first = torch.load(source, map_location="cpu", weights_only=True)
        second = torch.load(result, map_location="cpu", weights_only=True)
        assert "warm_start_binding" not in first
        binding = second["warm_start_binding"]
        assert binding["source_checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert binding["source_float_state_sha256"] == first["float_state_identity"]["sha256"]
        assert binding["optimizer_state_restored"] is False
        assert training_metadata(second)["warm_start_binding"] == binding

        before = hashlib.sha256(source.read_bytes()).hexdigest()
        try:
            train(manifest, source, source)
        except subprocess.CalledProcessError as exc:
            assert "must not overwrite --warm-start" in exc.stderr
        else:
            raise AssertionError("warm start overwrote its source checkpoint")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before

        dev = copy.deepcopy(first)
        dev["development_recipe"] = {"development_only": True}
        dev_path = root / "development.pt"
        torch.save(dev, dev_path)
        args = argparse.Namespace(warm_start=dev_path, feature_dim=32, hidden_dim=64, frontend="logmel")
        try:
            validate_warm_start(dev, args, 5, first["vocab_fingerprint"], "0" * 64)
        except ValueError as exc:
            assert "development-only" in str(exc)
        else:
            raise AssertionError("development checkpoint reached formal warm start")

        drifted = copy.deepcopy(first)
        drifted["float_state_identity"]["sha256"] = "0" * 64
        try:
            validate_warm_start(drifted, args, 5, first["vocab_fingerprint"], "0" * 64)
        except ValueError as exc:
            assert "float state identity mismatch" in str(exc)
        else:
            raise AssertionError("changed source state identity was accepted")

        malformed = copy.deepcopy(second)
        malformed["warm_start_binding"]["source_checkpoint_sha256"] = "bad"
        try:
            training_metadata(malformed)
        except ValueError as exc:
            assert "warm_start_binding.source_checkpoint_sha256" in str(exc)
        else:
            raise AssertionError("exporter accepted malformed warm-start binding")
    print("test_warm_start_lineage: ok")


if __name__ == "__main__":
    main()
