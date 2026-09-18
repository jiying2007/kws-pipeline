#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pathlib
import subprocess
import struct
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


def _sinc(value: float) -> float:
    if abs(value) < 1e-12:
        return 1.0
    x = math.pi * value
    return math.sin(x) / x


def resample_8k_to_16k_lanczos(source: pathlib.Path, output: pathlib.Path, radius: int = 8) -> None:
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 8000
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError("builtin resampler requires mono PCM16 8000-Hz WAV")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError("builtin resampler source WAV is empty")
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)

    taps: list[tuple[int, float]] = []
    for offset in range(-radius + 1, radius + 1):
        distance = 0.5 - offset
        if abs(distance) >= radius:
            continue
        weight = _sinc(distance) * _sinc(distance / radius)
        taps.append((offset, weight))
    norm = sum(weight for _, weight in taps)
    if abs(norm) < 1e-12:
        raise ValueError("builtin resampler coefficient normalization failed")
    taps = [(offset, weight / norm) for offset, weight in taps]

    rendered: list[int] = []
    last = len(samples) - 1
    for index, value in enumerate(samples):
        rendered.append(int(value))
        if index == last:
            rendered.append(int(value))
            continue
        acc = 0.0
        for offset, weight in taps:
            source_index = min(max(index + offset, 0), last)
            acc += samples[source_index] * weight
        interpolated = max(-32768, min(32767, int(round(acc))))
        rendered.append(interpolated)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TARGET_SAMPLE_RATE)
        writer.writeframes(
            b"".join(struct.pack("<h", sample) for sample in rendered)
        )


def run_checked(command: list[str], label: str, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, shell=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one VITS TTS request at its native rate and normalize it to mono PCM16 16 kHz."
    )
    parser.add_argument("--backend-executable", required=True, type=pathlib.Path)
    parser.add_argument("--backend-lib-dir", type=pathlib.Path)
    parser.add_argument("--resampler-executable", type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--lexicon", required=True, type=pathlib.Path)
    parser.add_argument("--phone-fst", type=pathlib.Path)
    parser.add_argument("--date-fst", type=pathlib.Path)
    parser.add_argument("--number-fst", type=pathlib.Path)
    parser.add_argument("--rule-far", type=pathlib.Path)
    parser.add_argument("--speaker-id", required=True, type=int)
    parser.add_argument("--length-scale", required=True, type=float)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-scale-w", type=float, default=0.0)
    parser.add_argument("--source-sample-rate", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("text")
    args = parser.parse_args()

    if args.speaker_id < 0:
        raise ValueError("speaker-id must be non-negative")
    if not 0.5 <= args.length_scale <= 2.0:
        raise ValueError("length-scale must be in [0.5,2.0]")
    if not 0.0 <= args.noise_scale <= 2.0:
        raise ValueError("noise-scale must be in [0,2]")
    if not 0.0 <= args.noise_scale_w <= 2.0:
        raise ValueError("noise-scale-w must be in [0,2]")
    if args.source_sample_rate <= 0 or args.source_sample_rate >= TARGET_SAMPLE_RATE:
        raise ValueError("source-sample-rate must be positive and below 16000 for this adapter")

    backend = require_file(args.backend_executable, "TTS backend executable")
    backend_lib_dir = None
    if args.backend_lib_dir is not None:
        backend_lib_dir = args.backend_lib_dir.resolve()
        if not backend_lib_dir.is_dir():
            raise ValueError(f"backend lib dir is missing: {backend_lib_dir}")
    else:
        inferred_lib_dir = (backend.parent.parent / "lib").resolve()
        if inferred_lib_dir.is_dir():
            backend_lib_dir = inferred_lib_dir
    resampler = (
        require_file(args.resampler_executable, "resampler executable")
        if args.resampler_executable is not None
        else None
    )
    model = require_file(args.model, "VITS model")
    tokens = require_file(args.tokens, "VITS tokens")
    lexicon = require_file(args.lexicon, "VITS lexicon")
    rule_values = (args.phone_fst, args.date_fst, args.number_fst, args.rule_far)
    if any(value is not None for value in rule_values) and any(value is None for value in rule_values):
        raise ValueError("phone/date/number FSTs and rule.far must be supplied together")
    phone_fst = date_fst = number_fst = rule_far = None
    if all(value is not None for value in rule_values):
        phone_fst = require_file(args.phone_fst, "phone FST")
        date_fst = require_file(args.date_fst, "date FST")
        number_fst = require_file(args.number_fst, "number FST")
        rule_far = require_file(args.rule_far, "rule FAR")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="speech-like-native-", suffix=".wav", dir=output.parent, delete=False
    ) as stream:
        native = pathlib.Path(stream.name)
    native.unlink(missing_ok=True)

    try:
        backend_command = [
            str(backend),
            f"--vits-model={model}",
            f"--vits-tokens={tokens}",
            f"--vits-lexicon={lexicon}",
        ]
        if phone_fst is not None:
            backend_command.extend(
                [
                    f"--tts-rule-fsts={phone_fst},{date_fst},{number_fst}",
                    f"--tts-rule-fars={rule_far}",
                ]
            )
        backend_command.extend(
            [
                f"--sid={args.speaker_id}",
                f"--vits-length-scale={args.length_scale}",
                f"--vits-noise-scale={args.noise_scale}",
                f"--vits-noise-scale-w={args.noise_scale_w}",
                f"--output-filename={native}",
                args.text,
            ]
        )
        backend_env = None
        if backend_lib_dir is not None:
            backend_env = dict(os.environ)
            current = backend_env.get("LD_LIBRARY_PATH", "")
            backend_env["LD_LIBRARY_PATH"] = (
                str(backend_lib_dir)
                if not current
                else str(backend_lib_dir) + os.pathsep + current
            )
        run_checked(backend_command, "VITS backend", env=backend_env)
        inspect_pcm16_mono(native, args.source_sample_rate, "VITS backend")

        if resampler is not None:
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
        else:
            if args.source_sample_rate != 8000:
                raise ValueError(
                    "builtin Lanczos resampler supports only 8000->16000 Hz"
                )
            resample_8k_to_16k_lanczos(native, output)
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
