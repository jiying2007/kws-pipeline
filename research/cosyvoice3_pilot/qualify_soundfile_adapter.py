"""Runner-only numeric qualification of the actual frontend-bound adapter.

Use only AFTER source/runtime integrity admission, in a fresh process:
    python qualify_soundfile_adapter.py --source-root SOURCE --scratch-dir NEW_DIR

The CLI prints one JSON receipt on success and exits nonzero on any failure.
It creates NEW_DIR exclusively, leaving its small deterministic PCM16 files
for audit. It never downloads anything, constructs a model/ORT session, loads
weights, or performs synthesis. Importing this module alone is stdlib-only.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import wave
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soundfile_adapter import bind_frontend, load_wav


def _fixture_definitions():
    """Known integer PCM avoids circular expected values from a decoder."""
    anchors = (-32768, 32767, -1, 0, 1, 16384, -16384, 8191)
    mono = anchors + tuple((i * 7919 + 12345) % 65536 - 32768 for i in range(249))
    # First eight means are known exactly in float32, including half-LSBs:
    # -1/65536, -1/65536, 0, 1/65536, 0, 0, -1/65536, 0.
    right_anchors = (32767, -32768, 1, 1, -1, -16384, 16383, -8191)
    right = right_anchors + tuple((i * 104729 + 97) % 65536 - 32768 for i in range(249))
    stereo = tuple(zip(mono, right))
    return (
        ("mono_16000.wav", 16000, tuple((value,) for value in mono)),
        ("stereo_16000.wav", 16000, stereo),
        ("mono_24000.wav", 24000, tuple((value,) for value in mono)),
        ("mono_8000.wav", 8000, tuple((value,) for value in mono)),
    )


def _write_pcm16(path, sample_rate, rows):
    """Write with stdlib wave/struct, independently of SoundFile/TorchAudio."""
    channels = len(rows[0])
    if channels not in (1, 2) or any(len(row) != channels for row in rows):
        raise ValueError("invalid fixture channel layout")
    values = [value for row in rows for value in row]
    if any(type(value) is not int or not -32768 <= value <= 32767 for value in values):
        raise ValueError("fixture values must be signed PCM16 integers")
    payload = struct.pack("<{}h".format(len(values)), *values)
    with path.open("xb") as destination:
        with wave.open(destination, "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(payload)


def _expected_tensor(torch, rows):
    # Integer addition is exact here; all means are exactly representable as
    # float32 binary fractions. This never reads a generated file or decoder.
    values = [sum(row) / (len(row) * 32768.0) for row in rows]
    return torch.tensor([values], dtype=torch.float32, device="cpu")


def _check_tensor(torch, actual, expected, frames, *, exact):
    if actual.dtype != torch.float32:
        raise AssertionError("loader output is not float32")
    if actual.device.type != "cpu" or tuple(actual.shape) != (1, frames):
        raise AssertionError("loader output is not CPU [1, expected_frames]")
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError("loader output contains nonfinite samples")
    if tuple(expected.shape) != (1, frames) or not bool(torch.isfinite(expected).all().item()):
        raise AssertionError("independent expected output has invalid shape/values")
    atol, rtol = (0.0, 0.0) if exact else (1e-7, 1e-6)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "device": str(actual.device),
        "finite": True,
        "max_absolute_error": float((actual - expected).abs().max().item()),
        "atol": atol,
        "rtol": rtol,
    }


def qualify_adapter(source_root, scratch_dir):
    """Exercise real dependencies and the bound frontend loader; fail closed.

    scratch_dir must not exist. Caller owns admission, resource limits,
    receipts persistence, and cleanup. No successful receipt is returned until
    every numeric and import-order check passes. This function is not idempotent:
    discard the process after failure; do not retry the binding in-process.
    """
    scratch = Path(scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=False)

    import numpy  # noqa: F401: verify actual dependency import, not a stub path
    import soundfile  # noqa: F401
    import torch
    import torchaudio

    frontend = bind_frontend(source_root)
    if frontend.load_wav is not load_wav:
        raise AssertionError("numeric tests would not exercise the bound adapter")
    bound_loader = frontend.load_wav
    fixtures = {}
    fixture_receipts = []
    for filename, rate, rows in _fixture_definitions():
        path = scratch / filename
        _write_pcm16(path, rate, rows)
        fixtures[filename] = (path, rate, rows)
        fixture_receipts.append({
            "filename": filename,
            "sample_rate_hz": rate,
            "channels": len(rows[0]),
            "frames": len(rows),
            "encoding": "PCM16_LE",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })

    cases = []
    # Equality cases include 8 -> 8 kHz, proving the minimum-rate check belongs
    # only to the resampling branch. Custom minimum covers that invariant too.
    exact_cases = (
        ("mono16_exact", "mono_16000.wav", 16000, 16000),
        ("stereo16_channel_mean_exact", "stereo_16000.wav", 16000, 16000),
        ("mono24_exact", "mono_24000.wav", 24000, 16000),
        ("mono8_same_rate_allowed", "mono_8000.wav", 8000, 16000),
        ("mono16_same_rate_ignores_custom_min", "mono_16000.wav", 16000, 24000),
    )
    for name, filename, target, minimum in exact_cases:
        path, rate, rows = fixtures[filename]
        expected = _expected_tensor(torch, rows)
        if minimum == 16000:
            actual = bound_loader(str(path), target)
        else:
            actual = bound_loader(str(path), target, min_sr=minimum)
        comparison = _check_tensor(torch, actual, expected, len(rows), exact=True)
        if name == "stereo16_channel_mean_exact":
            anchors = torch.tensor(
                [[-1 / 65536, -1 / 65536, 0, 1 / 65536, 0, 0, -1 / 65536, 0]],
                dtype=torch.float32,
            )
            torch.testing.assert_close(actual[:, :8], anchors, atol=0.0, rtol=0.0)
        cases.append({"name": name, "status": "PASS", "source_rate_hz": rate,
                      "target_rate_hz": target, "min_sr": minimum, **comparison})

    # Expected tensors start from fixture integer PCM, then use a NEW default
    # stock Resample transform. They do not reuse the adapter output or decoder.
    resample_cases = (
        ("mono16_to24", "mono_16000.wav", 24000, 16000),
        ("stereo16_to24", "stereo_16000.wav", 24000, 16000),
        ("mono24_to16", "mono_24000.wav", 16000, 16000),
        ("mono8_to24_custom_min_allowed", "mono_8000.wav", 24000, 8000),
    )
    for name, filename, target, minimum in resample_cases:
        path, rate, rows = fixtures[filename]
        independent = _expected_tensor(torch, rows)
        expected = torchaudio.transforms.Resample(orig_freq=rate, new_freq=target)(independent)
        if minimum == 16000:
            actual = bound_loader(str(path), target)
        else:
            actual = bound_loader(str(path), target, min_sr=minimum)
        frames = (len(rows) * target + rate - 1) // rate
        comparison = _check_tensor(torch, actual, expected, frames, exact=False)
        cases.append({"name": name, "status": "PASS", "source_rate_hz": rate,
                      "target_rate_hz": target, "min_sr": minimum, **comparison})

    rejection_cases = (
        ("mono8_to24_default_min_rejected", "mono_8000.wav", 24000, 16000),
        ("mono16_to24_custom_min_rejected", "mono_16000.wav", 24000, 24000),
    )
    for name, filename, target, minimum in rejection_cases:
        path, rate, _rows = fixtures[filename]
        message = "wav sample rate {} must be greater than {}".format(rate, target)
        try:
            if minimum == 16000:
                bound_loader(str(path), target)
            else:
                bound_loader(str(path), target, min_sr=minimum)
        except AssertionError as error:
            if str(error) != message:
                raise AssertionError("unexpected minimum-rate rejection") from error
        else:
            raise AssertionError("adapter accepted a forbidden resampling rate")
        cases.append({"name": name, "status": "PASS", "source_rate_hz": rate,
                      "target_rate_hz": target, "min_sr": minimum,
                      "expected_exception": "AssertionError", "message": message})

    if frontend.load_wav is not load_wav or importlib.import_module(
        "cosyvoice.utils.file_utils"
    ).load_wav is not load_wav:
        raise AssertionError("loader binding changed during qualification")

    return {
        "schema": "cosyvoice3.soundfile-adapter.numeric-qualification.v1",
        "status": "PASS",
        "qualification_boundary": "real dependency imports and numeric PCM fixtures only",
        "actual_frontend_module": frontend.__name__,
        "actual_frontend_file": str(Path(frontend.__file__).resolve()),
        "frontend_load_wav_is_adapter": frontend.load_wav is load_wav,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "soundfile", "torch", "torchaudio", "onnxruntime", "openai-whisper", "inflect")
        },
        "fixture_directory": str(scratch),
        "fixtures": fixture_receipts,
        "cases": cases,
        "passed_case_count": len(cases),
        "model_construction_requested": False,
        "ort_session_construction_requested": False,
        "speech_synthesis_requested": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scratch-dir", required=True)
    args = parser.parse_args(argv)
    receipt = qualify_adapter(args.source_root, args.scratch_dir)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
