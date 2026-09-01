#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import queue
import random
import struct
import subprocess
import sys
import threading
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from frontend_spec import SAMPLE_RATE_HZ  # noqa: E402
from synthetic_audio import clamp16, noise_profile  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(b"".join(struct.pack("<h", value) for value in samples))


def read_wav(path: pathlib.Path) -> list[int]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"hard-negative WAV must be mono 16-kHz PCM16: {path}")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError(f"hard-negative WAV is empty: {path}")
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def load_negative_manifest(path: pathlib.Path) -> list[dict]:
    manifest = path.resolve()
    if not manifest.is_file():
        raise ValueError(f"hard-negative manifest does not exist: {manifest}")
    clips: list[dict] = []
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        source = pathlib.Path(cols[0])
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        else:
            source = source.resolve()
        if not source.is_file():
            raise ValueError(f"{manifest}:{line_no}: hard-negative WAV is missing: {source}")
        samples = read_wav(source)
        clips.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "samples": samples,
                "seconds": len(samples) / SAMPLE_RATE_HZ,
            }
        )
    if not clips:
        raise ValueError("hard-negative manifest contains no WAV clips")
    return clips


def reader_thread(stream, output: queue.Queue) -> None:
    try:
        for raw in stream:
            if raw.strip():
                output.put(("detection", raw.strip()))
    except BaseException as exc:  # pragma: no cover - defensive thread boundary
        output.put(("error", repr(exc)))
    finally:
        output.put(("eof", ""))


def generate_second(rng: random.Random, index: int) -> tuple[list[int], dict]:
    profiles = ("white", "fan", "motor", "media")
    profile = profiles[(index // 17 + rng.randrange(len(profiles))) % len(profiles)]
    noise = noise_profile(profile, SAMPLE_RATE_HZ, rng)
    amplitude = rng.uniform(600.0, 8500.0)
    # Slowly moving gain plus optional second colored layer models TV/motor mixtures.
    wobble_hz = rng.uniform(0.07, 0.45)
    result = [
        clamp16(value * amplitude * (0.78 + 0.22 * math.sin(2.0 * math.pi * wobble_hz * n / SAMPLE_RATE_HZ)))
        for n, value in enumerate(noise)
    ]
    mixed = rng.random() < 0.35
    if mixed:
        other = noise_profile("media" if profile != "media" else "motor", SAMPLE_RATE_HZ, rng)
        layer = rng.uniform(350.0, 2600.0)
        result = [clamp16(value + layer * extra) for value, extra in zip(result, other)]
    return result, {"profile": profile, "amplitude": amplitude, "mixed": mixed}


def drain_events(events: queue.Queue, pending: list[dict], detections: list[dict]) -> None:
    while True:
        try:
            kind, payload = events.get_nowait()
        except queue.Empty:
            return
        if kind == "error":
            raise RuntimeError(f"stream reader failed: {payload}")
        if kind != "detection":
            continue
        row = json.loads(payload)
        if not isinstance(row, dict):
            raise ValueError("raw stream runner emitted non-object JSON")
        detections.append(row)
        pending.append(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--max-far-per-hour", type=float, default=0.0)
    parser.add_argument("--capture-context-seconds", type=float, default=2.0)
    parser.add_argument("--negative-manifest", type=pathlib.Path)
    parser.add_argument("--hard-negative-rate-per-minute", type=float, default=0.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.max_far_per_hour < 0.0 or not math.isfinite(args.max_far_per_hour):
        parser.error("--max-far-per-hour must be finite and >= 0")
    if args.capture_context_seconds < 0.0 or args.capture_context_seconds > 10.0:
        parser.error("--capture-context-seconds must be in [0,10]")
    if (
        args.hard_negative_rate_per_minute < 0.0
        or args.hard_negative_rate_per_minute > 60.0
        or not math.isfinite(args.hard_negative_rate_per_minute)
    ):
        parser.error("--hard-negative-rate-per-minute must be finite and in [0,60]")
    if args.hard_negative_rate_per_minute > 0.0 and args.negative_manifest is None:
        parser.error("--negative-manifest is required when hard-negative injection is enabled")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    negative_manifest = args.negative_manifest.resolve() if args.negative_manifest else None
    negative_clips = load_negative_manifest(negative_manifest) if negative_manifest else []
    process = subprocess.Popen(
        [
            str(args.runner.resolve()),
            str(args.model.resolve()),
            str(args.keywords.resolve()),
            f"far-stream-seed-{args.seed}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    events: queue.Queue = queue.Queue()
    text_stdout = (line.decode("utf-8") for line in iter(process.stdout.readline, b""))
    thread = threading.Thread(target=reader_thread, args=(text_stdout, events), daemon=True)
    thread.start()

    context = int(round(args.capture_context_seconds * SAMPLE_RATE_HZ))
    history_limit = max(SAMPLE_RATE_HZ, context * 2 + 2 * SAMPLE_RATE_HZ)
    history: list[int] = []
    history_start = 0
    generated = 0
    detections: list[dict] = []
    pending: list[dict] = []
    captures: list[dict] = []
    profile_seconds: dict[str, int] = {}
    injections: list[dict] = []
    active_clip: dict | None = None
    active_offset = 0
    active_gain = 1.0
    injected_samples = 0
    injection_probability = args.hard_negative_rate_per_minute / 60.0

    try:
        for second in range(args.seconds):
            samples, scene = generate_second(rng, second)
            profile_seconds[scene["profile"]] = profile_seconds.get(scene["profile"], 0) + 1
            if active_clip is None and negative_clips and rng.random() < injection_probability:
                active_clip = rng.choice(negative_clips)
                active_offset = 0
                active_gain = rng.uniform(0.65, 1.0)
                injections.append(
                    {
                        "start_second": second,
                        "source_path": active_clip["path"],
                        "source_sha256": active_clip["sha256"],
                        "source_seconds": active_clip["seconds"],
                        "gain": active_gain,
                    }
                )
            if active_clip is not None:
                source_samples = active_clip["samples"]
                mix_count = min(len(samples), len(source_samples) - active_offset)
                samples = [
                    clamp16(value + source_samples[active_offset + index] * active_gain)
                    if index < mix_count
                    else value
                    for index, value in enumerate(samples)
                ]
                injected_samples += mix_count
                active_offset += mix_count
                if active_offset >= len(source_samples):
                    active_clip = None
                    active_offset = 0
                    active_gain = 1.0

            process.stdin.write(struct.pack("<" + "h" * len(samples), *samples))
            process.stdin.flush()
            history.extend(samples)
            generated += len(samples)
            if len(history) > history_limit:
                trim = len(history) - history_limit
                del history[:trim]
                history_start += trim
            drain_events(events, pending, detections)

            ready: list[dict] = []
            for row in pending:
                center = int(round(float(row["time_s"]) * SAMPLE_RATE_HZ))
                if generated >= center + context:
                    start = max(0, center - context)
                    end = center + context
                    if start >= history_start and end <= history_start + len(history):
                        clip = history[start - history_start : end - history_start]
                        path = output / "captures" / f"fa-{len(captures):04d}-kw{int(row['keyword_id'])}.wav"
                        write_wav(path, clip)
                        captures.append(
                            {
                                **row,
                                "path": str(path),
                                "sha256": sha256_file(path),
                                "start_sample": start,
                                "end_sample": end,
                            }
                        )
                    ready.append(row)
            for row in ready:
                pending.remove(row)
    finally:
        process.stdin.close()
    return_code = process.wait(timeout=30)
    thread.join(timeout=5)
    drain_events(events, pending, detections)
    if return_code != 0:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"raw stream runner failed ({return_code}): {stderr}")

    audio_hours = args.seconds / 3600.0
    far = len(detections) / audio_hours
    detections_path = output / "detections.jsonl"
    detections_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in detections)
        + ("\n" if detections else ""),
        encoding="utf-8",
    )
    captures_path = output / "captures.jsonl"
    captures_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in captures)
        + ("\n" if captures else ""),
        encoding="utf-8",
    )
    injections_path = output / "hard-negative-injections.jsonl"
    injections_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in injections)
        + ("\n" if injections else ""),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "evidence_class": "synthetic-streaming-far",
        "seconds": args.seconds,
        "audio_hours": audio_hours,
        "seed": args.seed,
        "false_accepts": len(detections),
        "far_per_hour": far,
        "max_far_per_hour": args.max_far_per_hour,
        "qualified": far <= args.max_far_per_hour,
        "profile_seconds": profile_seconds,
        "hard_negative_rate_per_minute": args.hard_negative_rate_per_minute,
        "hard_negative_injections": len(injections),
        "hard_negative_audio_seconds": injected_samples / SAMPLE_RATE_HZ,
        "negative_manifest_sha256": sha256_file(negative_manifest) if negative_manifest else None,
        "runner_sha256": sha256_file(args.runner),
        "model_sha256": sha256_file(args.model),
        "keyword_pack_sha256": sha256_file(args.keywords),
        "detections_sha256": sha256_file(detections_path),
        "captures_sha256": sha256_file(captures_path),
        "hard_negative_injections_sha256": sha256_file(injections_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
