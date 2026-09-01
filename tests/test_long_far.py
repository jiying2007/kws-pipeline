#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

from qualification_fixture import write_model, write_pack, write_tokens

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_RATE_HZ = 16000


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_negative_wav(path: pathlib.Path) -> None:
    samples = [0] * SAMPLE_RATE_HZ
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def run_far(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    output: pathlib.Path,
    seed: int,
    negative_manifest: pathlib.Path | None = None,
) -> dict:
    command = [
        sys.executable,
        str(ROOT / "eval" / "long_far_stream.py"),
        "--runner",
        str(runner.resolve()),
        "--model",
        str(model),
        "--keywords",
        str(pack),
        "--seconds",
        "3",
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        "--max-far-per-hour",
        "0",
    ]
    if negative_manifest is not None:
        command.extend(
            [
                "--negative-manifest",
                str(negative_manifest),
                "--hard-negative-rate-per-minute",
                "60",
            ]
        )
    subprocess.check_call(command)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def aggregate_far(
    *,
    outputs: list[pathlib.Path],
    target: pathlib.Path,
    min_injections: int,
    min_audio_seconds: float,
) -> tuple[int, dict]:
    command = [
        sys.executable,
        str(ROOT / "eval" / "aggregate_far.py"),
    ]
    for output in outputs:
        command.extend(["--summary", str(output / "summary.json")])
    command.extend(
        [
            "--output",
            str(target),
            "--confidence",
            "0.95",
            "--max-far-per-hour",
            "0",
            "--max-upper-bound-per-hour",
            "2000",
            "--min-hard-negative-injections",
            str(min_injections),
            "--min-hard-negative-audio-seconds",
            str(min_audio_seconds),
        ]
    )
    completed = subprocess.run(command, check=False)
    return completed.returncode, json.loads(target.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        _, fingerprint = write_tokens(tokens)
        model = root / "model.kwm"
        pack = root / "keywords.kwk"
        write_model(model, fingerprint)
        write_pack(pack, fingerprint)

        output = root / "far"
        summary = run_far(
            runner=args.runner,
            model=model,
            pack=pack,
            output=output,
            seed=404,
        )
        assert summary["qualified"] is True
        assert summary["seconds"] == 3
        assert summary["false_accepts"] == 0
        assert summary["far_per_hour"] == 0.0
        assert summary["hard_negative_injections"] == 0
        assert summary["negative_manifest_sha256"] is None
        assert sum(summary["profile_seconds"].values()) == 3
        assert (output / "detections.jsonl").read_text(encoding="utf-8") == ""

        negative_wav = root / "hard-negative.wav"
        write_negative_wav(negative_wav)
        negative_manifest = root / "hard-negatives.tsv"
        negative_manifest.write_text(f"{negative_wav.resolve()}\t1 2\n", encoding="utf-8")

        stressed_outputs = [root / "far-stress-a", root / "far-stress-b"]
        stressed = [
            run_far(
                runner=args.runner,
                model=model,
                pack=pack,
                output=stressed_outputs[0],
                seed=405,
                negative_manifest=negative_manifest,
            ),
            run_far(
                runner=args.runner,
                model=model,
                pack=pack,
                output=stressed_outputs[1],
                seed=406,
                negative_manifest=negative_manifest,
            ),
        ]
        for item, path in zip(stressed, stressed_outputs):
            assert item["qualified"] is True
            assert item["false_accepts"] == 0
            assert item["hard_negative_rate_per_minute"] == 60.0
            assert item["hard_negative_injections"] == 3
            assert item["hard_negative_audio_seconds"] == 3.0
            assert item["negative_manifest_sha256"] == sha256_file(negative_manifest)
            injections = (path / "hard-negative-injections.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            assert len(injections) == 3

        aggregate = root / "far-aggregate.json"
        code, aggregate_summary = aggregate_far(
            outputs=stressed_outputs,
            target=aggregate,
            min_injections=6,
            min_audio_seconds=6.0,
        )
        assert code == 0
        assert aggregate_summary["schema_version"] == 1
        assert aggregate_summary["qualified"] is True
        assert aggregate_summary["false_accepts"] == 0
        assert aggregate_summary["negative_manifest_sha256"] == sha256_file(
            negative_manifest
        )
        assert aggregate_summary["hard_negative_rate_per_minute"] == 60.0
        assert aggregate_summary["hard_negative_injections"] == 6
        assert aggregate_summary["hard_negative_audio_seconds"] == 6.0
        assert aggregate_summary["min_hard_negative_injections"] == 6
        assert aggregate_summary["min_hard_negative_audio_seconds"] == 6.0

        insufficient = root / "far-insufficient.json"
        code, insufficient_summary = aggregate_far(
            outputs=stressed_outputs,
            target=insufficient,
            min_injections=7,
            min_audio_seconds=7.0,
        )
        assert code == 1
        assert insufficient_summary["qualified"] is False
        assert {
            "aggregate hard-negative injection count below minimum",
            "aggregate hard-negative audio exposure below minimum",
        }.issubset(set(insufficient_summary["violations"]))

    print("test_long_far: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
