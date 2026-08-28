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
from frontend_spec import features_pcm16  # noqa: E402


def write_fixture(path: pathlib.Path) -> list[int]:
    samples: list[int] = []
    for index in range(1040):
        tone = 9000 if ((index // 13) & 1) else -7000
        ramp = (index % 37) * 41 - 740
        sample = max(-32768, min(32767, tone + ramp))
        samples.append(sample)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return samples


def frame_dbfs(samples: list[int]) -> float:
    normalized = [sample / 32768.0 for sample in samples]
    mean_square = sum(value * value for value in normalized) / len(normalized)
    return 10.0 * math.log10(mean_square + 1.0e-12)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/kws_feature_dump", file=sys.stderr)
        return 2
    runner = pathlib.Path(sys.argv[1])

    with tempfile.TemporaryDirectory() as td:
        wav = pathlib.Path(td) / "frontend.wav"
        samples = write_fixture(wav)
        expected = features_pcm16(samples, feature_dim=32, frame_len=400, hop=320)
        completed = subprocess.run(
            [str(runner), str(wav), "32"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        actual = [json.loads(line) for line in completed.stdout.splitlines() if line]

        assert len(actual) == len(expected) == 3
        max_abs_error = 0.0
        for frame_index, (row, reference) in enumerate(zip(actual, expected)):
            assert row["frame"] == frame_index
            assert len(row["features"]) == 32
            for got, want in zip(row["features"], reference):
                max_abs_error = max(max_abs_error, abs(float(got) - want))
            start = frame_index * 320
            want_dbfs = frame_dbfs(samples[start : start + 400])
            assert abs(float(row["dbfs"]) - want_dbfs) <= 2.0e-4

        # C uses float32 FFT arithmetic while the dependency-free spec uses
        # Python double precision. This bound is tight enough to catch changes
        # in windowing, FFT geometry, mel bins, energy scale or normalization.
        assert max_abs_error <= 2.0e-3, max_abs_error

    print(f"test_frontend_parity: ok max_abs_error={max_abs_error:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
