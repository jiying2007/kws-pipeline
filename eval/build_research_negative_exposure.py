#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import struct
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from frontend_spec import SAMPLE_RATE_HZ  # noqa: E402
from synthetic_audio import clamp16, noise_profile  # noqa: E402

PROFILES = ("white", "fan", "motor", "media")
EVIDENCE_CLASS = "kws-v2-research-negative-exposure-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav(path: pathlib.Path) -> list[int]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"negative clip must be mono 16-kHz PCM16: {path}")
        frames = reader.readframes(reader.getnframes())
    if not frames:
        raise ValueError(f"negative clip is empty: {path}")
    return list(struct.unpack("<" + "h" * (len(frames) // 2), frames))


def load_manifest(path: pathlib.Path) -> list[dict]:
    root = path.parent.resolve()
    rows: list[dict] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        first = raw.split("\t", 1)[0].strip()
        source = pathlib.Path(first)
        source = source.resolve() if source.is_absolute() else (root / source).resolve()
        if not source.is_file():
            raise ValueError(f"{path}:{line_no}: missing WAV {source}")
        digest = sha256_file(source)
        if digest in seen:
            continue
        seen.add(digest)
        samples = read_wav(source)
        rows.append(
            {
                "path": str(source),
                "sha256": digest,
                "samples": samples,
                "seconds": len(samples) / SAMPLE_RATE_HZ,
            }
        )
    if not rows:
        raise ValueError("negative manifest contains no unique WAV clips")
    return rows


def background_second(rng: random.Random, second: int) -> tuple[list[int], str]:
    profile = PROFILES[(second // 19 + rng.randrange(len(PROFILES))) % len(PROFILES)]
    base = noise_profile(profile, SAMPLE_RATE_HZ, rng)
    amplitude = rng.uniform(450.0, 6000.0)
    wobble = rng.uniform(0.05, 0.35)
    samples = [
        clamp16(
            value
            * amplitude
            * (0.82 + 0.18 * math.sin(2.0 * math.pi * wobble * n / SAMPLE_RATE_HZ))
        )
        for n, value in enumerate(base)
    ]
    if rng.random() < 0.30:
        other = noise_profile("media" if profile != "media" else "motor", SAMPLE_RATE_HZ, rng)
        layer = rng.uniform(250.0, 1800.0)
        samples = [clamp16(a + layer * b) for a, b in zip(samples, other)]
    return samples, profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one continuous research-only negative WAV with deterministic speech-like negative injections."
    )
    parser.add_argument("--negative-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-injections-per-clip", type=int, default=1)
    parser.add_argument("--injection-gain", type=float, default=0.90)
    parser.add_argument("--output-wav", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be > 0")
    if args.min_injections_per_clip < 1:
        parser.error("--min-injections-per-clip must be >= 1")
    if not math.isfinite(args.injection_gain) or not 0.0 < args.injection_gain <= 1.0:
        parser.error("--injection-gain must be finite and in (0,1]")

    manifest = args.negative_manifest.resolve()
    if not manifest.is_file():
        raise ValueError(f"negative manifest missing: {manifest}")
    clips = load_manifest(manifest)
    longest = max(1, max(int(math.ceil(row["seconds"])) for row in clips))
    schedule: list[int] = []
    schedule_rng = random.Random(args.seed ^ 0x51A7C0DE)
    for _ in range(args.min_injections_per_clip):
        order = list(range(len(clips)))
        schedule_rng.shuffle(order)
        schedule.extend(order)
    usable = args.seconds - longest
    if usable < 0:
        raise ValueError("negative exposure is shorter than its longest injected clip")
    if len(schedule) == 1:
        targets = [0]
    else:
        stride = usable / float(len(schedule) - 1)
        if stride < float(longest):
            raise ValueError(
                "negative exposure is too short for non-overlapping minimum clip coverage"
            )
        targets = [int(round(i * stride)) for i in range(len(schedule))]

    output = args.output_wav.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    active: dict | None = None
    active_offset = 0
    cursor = 0
    injections: list[dict] = []
    profile_seconds: dict[str, int] = {}

    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        for second in range(args.seconds):
            samples, profile = background_second(rng, second)
            profile_seconds[profile] = profile_seconds.get(profile, 0) + 1
            if active is None and cursor < len(schedule) and second >= targets[cursor]:
                active = clips[schedule[cursor]]
                active_offset = 0
                injections.append(
                    {
                        "start_second": second,
                        "source_path": active["path"],
                        "source_sha256": active["sha256"],
                        "source_seconds": active["seconds"],
                        "gain": args.injection_gain,
                    }
                )
                cursor += 1

            if active is not None:
                source = active["samples"]
                count = min(SAMPLE_RATE_HZ, len(source) - active_offset)
                for index in range(count):
                    samples[index] = clamp16(
                        samples[index] + source[active_offset + index] * args.injection_gain
                    )
                active_offset += count
                if active_offset >= len(source):
                    active = None
                    active_offset = 0

            writer.writeframes(
                struct.pack("<" + "h" * len(samples), *samples)
            )

    if cursor != len(schedule):
        raise RuntimeError("not all required negative clips were injected")

    recording = "research-negative-exposure"
    references = args.references.resolve()
    references.parent.mkdir(parents=True, exist_ok=True)
    references.write_text(
        json.dumps(
            {
                "recording": recording,
                "path": str(output),
                "duration_s": float(args.seconds),
                "expected": [],
                "source_id": f"research-negative-exposure-seed-{args.seed}",
                "domain": {"kind": "research-negative-continuous"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "protected_evidence_used": False,
        "shipping_far_claim_allowed": False,
        "seed": args.seed,
        "seconds": args.seconds,
        "audio_hours": args.seconds / 3600.0,
        "negative_manifest_sha256": sha256_file(manifest),
        "unique_negative_clips": len(clips),
        "min_injections_per_clip": args.min_injections_per_clip,
        "injections": len(injections),
        "profile_seconds": profile_seconds,
        "wav": str(output),
        "wav_sha256": sha256_file(output),
        "references": str(references),
        "references_sha256": sha256_file(references),
        "injection_sources": injections,
    }
    target = args.receipt.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"research-negative-exposure: seconds={args.seconds} clips={len(clips)} "
        f"injections={len(injections)} wav={receipt['wav_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
