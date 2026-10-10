#!/usr/bin/env python3
from __future__ import annotations

import ast
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
CHUNK_PATTERNS = (None, "160", "320", "1,17,320,159,7,80")


def require(condition: bool, message: object = "frontend contract mismatch") -> None:
    if not condition:
        raise AssertionError(message)


def verify_training_short_admission(root: pathlib.Path) -> None:
    # Execute only the real manifest parser/constructor, with no Torch/model
    # import, optimizer, frontend inference or training entry point.
    sys.path.insert(0, str(ROOT / "tools"))
    from corpus_identity import canonical_audio_path, corpus_digest, inspect_pcm16_wav
    tree = ast.parse((ROOT / "training" / "train_ctc.py").read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"Manifest", "manifest_rows", "parse_token_ids"}]
    namespace = {
        "Dataset": object, "pathlib": pathlib, "json": json,
        "FRAME_LENGTH_SAMPLES": FRAME_LEN, "IDENTITY_FIELDS": (),
        "inspect_pcm16_wav": inspect_pcm16_wav, "corpus_digest": corpus_digest,
        "canonical_audio_path": canonical_audio_path,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 "train_ctc.py", "exec"), namespace)
    for count in (0, 1, 399, 400, 719, 720):
        wav = root / f"admission-{count}.wav"
        write_wav(wav, [0] * count)
        for tokens in ([], [1]):
            manifest = root / "admission.jsonl"
            manifest.write_text(json.dumps({"audio": wav.name, "tokens": tokens}) + "\n")
            try:
                dataset = namespace["Manifest"]([manifest], FEATURE_DIM, 5, FRONTEND_LOGMEL)
            except ValueError as exc:
                require(count < FRAME_LEN, str(exc))
                expected = "empty PCM payload" if count == 0 else "at least 400 samples"
                require(expected in str(exc), str(exc))
            else:
                require(count >= FRAME_LEN, f"accepted {count}-sample target {tokens}")
                require(len(dataset) == 1)


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
    expected_steps = 0 if len(samples) < FRAME_LEN else 1 + (len(samples) - FRAME_LEN) // HOP
    require(len(expected) == expected_steps, (name, len(expected), expected_steps))
    max_abs_error = 0.0
    baseline = None
    for chunks in CHUNK_PATTERNS:
        command = [str(runner), str(wav), str(FEATURE_DIM), frontend]
        if chunks is not None:
            command.append(chunks)
        completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
        actual = [json.loads(line) for line in completed.stdout.splitlines() if line]
        require(len(actual) == len(expected), (name, chunks, len(actual), len(expected)))
        if baseline is None:
            baseline = actual
        else:
            require(actual == baseline, (name, frontend, chunks, "chunk partition changed frames"))
        for frame_index, (row, reference) in enumerate(zip(actual, expected)):
            require(row["frame"] == frame_index)
            require(len(row["features"]) == FEATURE_DIM)
            for got, want in zip(row["features"], reference):
                max_abs_error = max(max_abs_error, abs(float(got) - want))
            start = frame_index * HOP
            want_dbfs = frame_dbfs(samples[start : start + FRAME_LEN])
            require(abs(float(row["dbfs"]) - want_dbfs) <= 2.0e-4)
    return max_abs_error


def verify_chunk_arguments(runner: pathlib.Path, root: pathlib.Path) -> None:
    wav = root / "chunks.wav"
    write_wav(wav, [0] * FRAME_LEN)
    for chunks in ("", "0", "321", "-1", "+1", "1,", "1,,2", ",1", "1x", ",".join(["1"] * 17)):
        completed = subprocess.run(
            [str(runner), str(wav), str(FEATURE_DIM), FRONTEND_LOGMEL, chunks],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        require(completed.returncode == 2, (chunks, completed.returncode))
        require("chunk sizes must be" in completed.stderr)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/kws_feature_dump", file=sys.stderr)
        return 2
    runner = pathlib.Path(sys.argv[1])
    maxima: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        verify_training_short_admission(root)
        verify_chunk_arguments(runner, root)
        edge_source = fixtures()["wideband"]
        for frontend in (FRONTEND_LOGMEL, FRONTEND_PCEN_LITE):
            maximum = 0.0
            for name, samples in fixtures().items():
                maximum = max(
                    maximum,
                    verify_fixture(runner, root, name, samples, frontend),
                )
            require(features_pcm16([], frontend=frontend) == [])
            for count, steps in ((1, 0), (399, 0), (400, 1), (719, 1), (720, 2)):
                samples = edge_source[:count]
                require(len(features_pcm16(samples, frontend=frontend)) == steps)
                maximum = max(maximum, verify_fixture(
                    runner, root, f"edge-{count}", samples, frontend))
            maxima[frontend] = maximum

    require(maxima[FRONTEND_LOGMEL] <= 1.0e-4, maxima)
    # PCEN contains powf/EMA state and therefore has slightly more float32 vs
    # Python-double drift. This still catches practical formula/state mismatch.
    require(maxima[FRONTEND_PCEN_LITE] <= 5.0e-4, maxima)
    print(
        "test_frontend_parity: ok "
        f"logmel={maxima[FRONTEND_LOGMEL]:.6g} "
        f"pcen_lite={maxima[FRONTEND_PCEN_LITE]:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
