#!/usr/bin/env python3
"""Cross-architecture parity for the int8 inference path.

The Cortex-A32 build is the only one that takes the NEON branch of the int8 dot
product in src/kws.c; the hosted build takes the portable scalar loop. Both are
supposed to produce the same logits and therefore the same detection stream.

Two properties keep this test from being vacuous:

* the model fixture carries non-zero int8 weights, laid out exactly like
  make_weighted_test_model() in tests/test_kws.c. The zero-weight fixture used
  elsewhere computes 0 * x in every lane, which is blind to lane order, sign
  extension and accumulation width;
* the assertion is on the values the runner prints (per-detection confidence),
  not just on whether something fired. A permuted or truncated accumulation
  changes the confidence even when the winning token does not change.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from qualification_fixture import write_tokens  # noqa: E402

SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FEATURE_DIM = 32
HIDDEN_DIM = 4
VOCAB_SIZE = 5
WEIGHT_SCALE = 0.01
KEYWORD_ID = 42
KEYWORD_THRESHOLD = 0.10
AUDIO_SECONDS = 4
MODEL_HEADER_BYTES = 72
RECORDING_ID = "parity"


def write_weighted_model(path: pathlib.Path, fingerprint: int, sign: int) -> int:
    """Write the non-zero-weight model used by both the C fixture and this test.

    `sign` selects the phase of the alternating in-projection row. Flipping it
    flips the sign of the top logit, so exactly one of the two phases can clear
    the keyword threshold for a given clip.
    """
    wx = MODEL_HEADER_BYTES
    wh = wx + FEATURE_DIM * HIDDEN_DIM
    bh = wh + HIDDEN_DIM * HIDDEN_DIM
    wo = bh + HIDDEN_DIM * 4
    bo = wo + VOCAB_SIZE * HIDDEN_DIM
    total = bo + VOCAB_SIZE * 4

    blob = bytearray(total)
    blob[:MODEL_HEADER_BYTES] = struct.pack(
        "<4sHHHHHHIIIfffQIIIIII",
        b"KWSP",
        2,
        MODEL_HEADER_BYTES,
        FEATURE_DIM,
        HIDDEN_DIM,
        VOCAB_SIZE,
        0,
        SAMPLE_RATE_HZ,
        FRAME_LENGTH_SAMPLES,
        FRAME_HOP_SAMPLES,
        WEIGHT_SCALE,
        WEIGHT_SCALE,
        WEIGHT_SCALE,
        fingerprint,
        wx,
        wh,
        bh,
        wo,
        bo,
        total,
    )
    positive, negative = (127, 0x81) if sign == 0 else (0x81, 127)
    for index in range(FEATURE_DIM):
        blob[wx + index] = positive if index % 2 == 0 else negative
    for index in range(HIDDEN_DIM):
        blob[wh + index * HIDDEN_DIM + index] = 127
    for index in range(HIDDEN_DIM):
        blob[wo + HIDDEN_DIM + index] = 127
    path.write_bytes(bytes(blob))
    return total


def write_keyword_pack(path: pathlib.Path, fingerprint: int) -> int:
    tokens = [1] + [0] * 15
    record = struct.pack(
        "<IfHBBBBH16H",
        KEYWORD_ID,
        KEYWORD_THRESHOLD,
        1,
        0,
        0,
        0,
        0,
        0,
        *tokens,
    )
    total = 24 + len(record)
    header = struct.pack("<4sHHHHIQ", b"KWKP", 3, 24, 1, VOCAB_SIZE, total, fingerprint)
    path.write_bytes(header + record)
    return total


def write_audio(path: pathlib.Path, seconds: int) -> None:
    frames = bytearray()
    for index in range(SAMPLE_RATE_HZ * seconds):
        sample = 12000 if ((index // 20) & 1) else -12000
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(frames)


def run_runner(
    runner: pathlib.Path,
    emulator: pathlib.Path | None,
    sysroot: pathlib.Path | None,
    model: pathlib.Path,
    pack: pathlib.Path,
    wav: pathlib.Path,
) -> list[dict]:
    argv: list[str] = []
    if emulator is not None:
        argv.append(str(emulator))
        if sysroot is not None:
            argv += ["-L", str(sysroot)]
    argv += [str(runner), str(model), str(pack), str(wav), RECORDING_ID]
    completed = subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE)
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-runner", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-runner", required=True, type=pathlib.Path)
    parser.add_argument(
        "--emulator",
        type=pathlib.Path,
        help="user-mode emulator used to execute the candidate runner",
    )
    parser.add_argument(
        "--emulator-sysroot",
        type=pathlib.Path,
        help="sysroot passed to the emulator as -L; required with --emulator",
    )
    parser.add_argument("--confidence-rel-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    if args.emulator is not None and args.emulator_sysroot is None:
        parser.error("--emulator-sysroot is required together with --emulator")

    results: list[tuple[int, list[dict], list[dict]]] = []
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        _, fingerprint = write_tokens(root / "tokens.txt")
        pack = root / "keywords.kwk"
        wav = root / "audio.wav"
        write_keyword_pack(pack, fingerprint)
        write_audio(wav, AUDIO_SECONDS)

        for sign in (0, 1):
            model = root / f"model-{sign}.kwm"
            write_weighted_model(model, fingerprint, sign)
            results.append(
                (
                    sign,
                    run_runner(args.reference_runner, None, None, model, pack, wav),
                    run_runner(
                        args.candidate_runner,
                        args.emulator,
                        args.emulator_sysroot,
                        model,
                        pack,
                        wav,
                    ),
                )
            )

    # Without this the whole comparison passes whenever both sides stay silent.
    firing = [sign for sign, reference, _ in results if reference]
    assert firing, (
        "no weight phase produced a detection: the fixture no longer drives the "
        "decoder, so parity would be vacuous"
    )

    worst = 0.0
    detections = 0
    for sign, reference, candidate in results:
        assert len(candidate) == len(reference), (sign, reference, candidate)
        for got, want in zip(candidate, reference):
            assert got["keyword_id"] == want["keyword_id"] == KEYWORD_ID, (got, want)
            assert abs(got["time_s"] - want["time_s"]) <= 1.0e-6, (got, want)
            scale = max(abs(want["confidence"]), 1.0e-9)
            worst = max(worst, abs(got["confidence"] - want["confidence"]) / scale)
            detections += 1

    assert worst <= args.confidence_rel_tolerance, (worst, results)
    print(
        "test_arm_parity: ok "
        f"phases={len(results)} firing_phases={firing} "
        f"detections={detections} "
        f"max_relative_confidence_delta={worst:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
