#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile
import wave

TARGET_SAMPLE_RATE = 16000


def require_file(path: pathlib.Path, label: str) -> pathlib.Path:
    result = path.resolve()
    if not result.is_file():
        raise ValueError(f"{label} is missing: {result}")
    return result


def inspect_pcm16_mono(path: pathlib.Path, expected_rate: int, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} did not create WAV: {path}")
    with wave.open(str(path), "rb") as reader:
        channels = int(reader.getnchannels())
        width = int(reader.getsampwidth())
        rate = int(reader.getframerate())
        frames = int(reader.getnframes())
        compression = reader.getcomptype()
    if channels != 1 or width != 2 or rate != expected_rate or compression != "NONE" or frames <= 0:
        raise ValueError(
            f"{label} WAV must be mono PCM16 {expected_rate}-Hz; "
            f"got channels={channels} width={width} rate={rate} compression={compression} frames={frames}"
        )


def run_checked(command: list[str], label: str) -> None:
    completed = subprocess.run(command, shell=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one VITS TTS request at its native rate and normalize it to mono PCM16 16 kHz."
    )
    parser.add_argument("--backend-executable", required=True, type=pathlib.Path)
    parser.add_argument("--resampler-executable", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--lexicon", required=True, type=pathlib.Path)
    parser.add_argument("--speaker-id", required=True, type=int)
    parser.add_argument("--length-scale", required=True, type=float)
    parser.add_argument("--source-sample-rate", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("text")
    args = parser.parse_args()

    if args.speaker_id < 0:
        raise ValueError("speaker-id must be non-negative")
    if not 0.5 <= args.length_scale <= 2.0:
        raise ValueError("length-scale must be in [0.5,2.0]")
    if args.source_sample_rate <= 0 or args.source_sample_rate >= TARGET_SAMPLE_RATE:
        raise ValueError("source-sample-rate must be positive and below 16000 for this adapter")

    backend = require_file(args.backend_executable, "TTS backend executable")
    resampler = require_file(args.resampler_executable, "resampler executable")
    model = require_file(args.model, "VITS model")
    tokens = require_file(args.tokens, "VITS tokens")
    lexicon = require_file(args.lexicon, "VITS lexicon")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="speech-like-native-", suffix=".wav", dir=output.parent, delete=False
    ) as stream:
        native = pathlib.Path(stream.name)
    native.unlink(missing_ok=True)

    try:
        run_checked(
            [
                str(backend),
                f"--vits-model={model}",
                f"--vits-tokens={tokens}",
                f"--vits-lexicon={lexicon}",
                f"--sid={args.speaker_id}",
                f"--vits-length-scale={args.length_scale}",
                f"--output-filename={native}",
                args.text,
            ],
            "VITS backend",
        )
        inspect_pcm16_mono(native, args.source_sample_rate, "VITS backend")

        run_checked(
            [
                str(resampler),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(native),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            "resampler",
        )
        inspect_pcm16_mono(output, TARGET_SAMPLE_RATE, "normalized provider")
    finally:
        native.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
