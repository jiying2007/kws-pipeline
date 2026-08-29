#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from frontend_spec import FRONTEND_LOGMEL, FRONTEND_PCEN_LITE, features_pcm16  # noqa: E402

SAMPLE_RATE_HZ = 16000
FRAME_LEN = 400
HOP = 320
FEATURE_DIM = 32


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def fixtures() -> dict[str, list[int]]:
    wideband: list[int] = []
    multitone: list[int] = []
    impulse = [0] * 1040
    impulse[31] = 28000
    impulse[399] = -23000
    impulse[721] = 17000

    for index in range(1040):
        tone = 9000 if ((index // 13) & 1) else -7000
        ramp = (index % 37) * 41 - 740
        wideband.append(max(-32768, min(32767, tone + ramp)))
        value = (
            7000.0 * math.sin(2.0 * math.pi * 440.0 * index / SAMPLE_RATE_HZ)
            + 4500.0 * math.sin(2.0 * math.pi * 1234.0 * index / SAMPLE_RATE_HZ)
            + 2500.0 * math.sin(2.0 * math.pi * 3111.0 * index / SAMPLE_RATE_HZ)
        )
        multitone.append(int(round(value)))
    return {"wideband": wideband, "multitone": multitone, "impulse": impulse}


def frame_dbfs(samples: list[int]) -> float:
    normalized = [sample / 32768.0 for sample in samples]
    mean_square = sum(value * value for value in normalized) / len(normalized)
    return 10.0 * math.log10(mean_square + 1.0e-12)


def verify_fixture(
    runner: pathlib.Path,
    root: pathlib.Path,
    name: str,
    samples: list[int],
    frontend: str,
) -> float:
    wav = root / f"{name}-{frontend}.wav"
    write_wav(wav, samples)
    expected = features_pcm16(
        samples,
        feature_dim=FEATURE_DIM,
        frame_len=FRAME_LEN,
        hop=HOP,
        frontend=frontend,
    )
    completed = subprocess.run(
        [str(runner), str(wav), str(FEATURE_DIM), frontend],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    actual = [json.loads(line) for line in completed.stdout.splitlines() if line]
    assert len(actual) == len(expected) == 3
    max_abs_error = 0.0
    for frame_index, (row, reference) in enumerate(zip(actual, expected)):
        assert row["frame"] == frame_index
        assert len(row["features"]) == FEATURE_DIM
        for got, want in zip(row["features"], reference):
            max_abs_error = max(max_abs_error, abs(float(got) - want))
        start = frame_index * HOP
        want_dbfs = frame_dbfs(samples[start : start + FRAME_LEN])
        assert abs(float(row["dbfs"]) - want_dbfs) <= 2.0e-4
    return max_abs_error


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/kws_feature_dump", file=sys.stderr)
        return 2
    runner = pathlib.Path(sys.argv[1])
    maxima: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for frontend in (FRONTEND_LOGMEL, FRONTEND_PCEN_LITE):
            maximum = 0.0
            for name, samples in fixtures().items():
                maximum = max(
                    maximum,
                    verify_fixture(runner, root, name, samples, frontend),
                )
            maxima[frontend] = maximum

    assert maxima[FRONTEND_LOGMEL] <= 1.0e-4, maxima
    # PCEN contains powf/EMA state and therefore has slightly more float32 vs
    # Python-double drift. This still catches practical formula/state mismatch.
    assert maxima[FRONTEND_PCEN_LITE] <= 5.0e-4, maxima
    print(
        "test_frontend_parity: ok "
        f"logmel={maxima[FRONTEND_LOGMEL]:.6g} "
        f"pcen_lite={maxima[FRONTEND_PCEN_LITE]:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
