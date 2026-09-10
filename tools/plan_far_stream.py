#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import wave

SAMPLE_RATE_HZ = 16000


def _read_clip_seconds(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"hard-negative WAV must be mono 16-kHz PCM16: {path}")
        frames = int(reader.getnframes())
    if frames <= 0:
        raise ValueError(f"hard-negative WAV is empty: {path}")
    return frames / SAMPLE_RATE_HZ


def load_manifest_clip_seconds(path: pathlib.Path) -> list[float]:
    manifest = path.resolve()
    if not manifest.is_file():
        raise ValueError(f"hard-negative manifest does not exist: {manifest}")
    durations: list[float] = []
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        source = pathlib.Path(raw.split("\t", 1)[0])
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        else:
            source = source.resolve()
        if not source.is_file():
            raise ValueError(f"{manifest}:{line_no}: hard-negative WAV is missing: {source}")
        durations.append(_read_clip_seconds(source))
    if not durations:
        raise ValueError("hard-negative manifest contains no WAV clips")
    return durations


def plan_stream_capacity(
    clip_seconds: list[float],
    *,
    injections_per_clip: int = 1,
    baseline_seconds: int = 900,
    minimum_payload_gap_seconds: float = 2.0,
    minimum_payload_rate_per_minute: float = 8.0,
) -> dict:
    if not clip_seconds:
        raise ValueError("at least one hard-negative clip is required")
    if injections_per_clip <= 0:
        raise ValueError("injections_per_clip must be > 0")
    if baseline_seconds <= 0:
        raise ValueError("baseline_seconds must be > 0")
    if minimum_payload_gap_seconds < 0.0 or not math.isfinite(minimum_payload_gap_seconds):
        raise ValueError("minimum_payload_gap_seconds must be finite and >= 0")
    if minimum_payload_rate_per_minute <= 0.0 or not math.isfinite(
        minimum_payload_rate_per_minute
    ):
        raise ValueError("minimum_payload_rate_per_minute must be finite and > 0")
    if any(seconds <= 0.0 or not math.isfinite(seconds) for seconds in clip_seconds):
        raise ValueError("clip durations must be finite and > 0")

    coverage_injections = len(clip_seconds) * injections_per_clip
    max_clip_seconds = max(clip_seconds)
    # long_far_stream schedules coverage starts on integer-second boundaries and
    # protects each active clip using ceil(duration). Plan against the same span
    # so a future manifest-size increase cannot make a formerly-valid fixed
    # duration impossible before the runtime is even started.
    max_clip_span_seconds = max(1, int(math.ceil(max_clip_seconds)))
    if coverage_injections == 1:
        minimum_coverage_seconds = max_clip_span_seconds
    else:
        stride = max_clip_span_seconds + minimum_payload_gap_seconds
        minimum_coverage_seconds = int(
            math.ceil(
                (coverage_injections - 1) * stride + max_clip_span_seconds
            )
        )
    planned_seconds = max(baseline_seconds, minimum_coverage_seconds)
    planned_payload_rate_per_minute = coverage_injections * 60.0 / planned_seconds
    if planned_payload_rate_per_minute + 1.0e-12 < minimum_payload_rate_per_minute:
        raise ValueError(
            "continuous FAR coverage cannot satisfy both minimum payload gap and minimum payload rate"
        )

    return {
        "schema_version": 1,
        "policy": "coverage-capacity-v1",
        "negative_manifest_clips": len(clip_seconds),
        "injections_per_clip": injections_per_clip,
        "coverage_injections": coverage_injections,
        "max_clip_seconds": max_clip_seconds,
        "max_clip_span_seconds": max_clip_span_seconds,
        "baseline_seconds": baseline_seconds,
        "minimum_payload_gap_seconds": minimum_payload_gap_seconds,
        "minimum_payload_rate_per_minute": minimum_payload_rate_per_minute,
        "minimum_coverage_seconds": minimum_coverage_seconds,
        "planned_seconds": planned_seconds,
        "planned_payload_rate_per_minute": planned_payload_rate_per_minute,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--injections-per-clip", type=int, default=1)
    parser.add_argument("--baseline-seconds", type=int, default=900)
    parser.add_argument("--minimum-payload-gap-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-payload-rate-per-minute", type=float, default=8.0)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    durations = load_manifest_clip_seconds(args.negative_manifest)
    plan = plan_stream_capacity(
        durations,
        injections_per_clip=args.injections_per_clip,
        baseline_seconds=args.baseline_seconds,
        minimum_payload_gap_seconds=args.minimum_payload_gap_seconds,
        minimum_payload_rate_per_minute=args.minimum_payload_rate_per_minute,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
