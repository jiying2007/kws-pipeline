from __future__ import annotations

import hashlib
import math
import pathlib
import struct
import wave

SAMPLE_RATE_HZ = 16000
MAX_RIR_SAMPLES = 32000


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rir(path: pathlib.Path) -> list[float]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: measured RIR must be mono 16-kHz PCM16 WAV")
        frames = reader.getnframes()
        if frames <= 0 or frames > MAX_RIR_SAMPLES:
            raise ValueError(
                f"{path}: measured RIR length must be 1..{MAX_RIR_SAMPLES} samples"
            )
        raw = reader.readframes(frames)
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    rir = [float(value) / 32768.0 for value in values]
    if max(abs(value) for value in rir) < 1.0e-7:
        raise ValueError(f"{path}: measured RIR is effectively silent")
    return rir


def onset_sample(rir: list[float]) -> int:
    peak = max(abs(value) for value in rir)
    threshold = max(1.0e-7, peak * 1.0e-3)
    for index, value in enumerate(rir):
        if abs(value) >= threshold:
            return index
    raise ValueError("measured RIR has no detectable onset")


def _fft(values: list[complex], *, inverse: bool = False) -> None:
    count = len(values)
    if count <= 0 or count & (count - 1):
        raise ValueError("FFT length must be a positive power of two")
    j = 0
    for i in range(1, count):
        bit = count >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]
    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= count:
        angle = sign * 2.0 * math.pi / length
        wlen = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for base in range(0, count, length):
            w = 1.0 + 0.0j
            for offset in range(half):
                u = values[base + offset]
                v = values[base + offset + half] * w
                values[base + offset] = u + v
                values[base + offset + half] = u - v
                w *= wlen
        length <<= 1
    if inverse:
        inv = 1.0 / count
        for index in range(count):
            values[index] *= inv


def fft_convolve(signal: list[int] | list[float], rir: list[float]) -> list[float]:
    if not signal or not rir:
        raise ValueError("convolution inputs must be non-empty")
    output_length = len(signal) + len(rir) - 1
    fft_length = 1
    while fft_length < output_length:
        fft_length <<= 1
    left = [0.0j] * fft_length
    right = [0.0j] * fft_length
    for index, value in enumerate(signal):
        left[index] = complex(float(value), 0.0)
    for index, value in enumerate(rir):
        right[index] = complex(float(value), 0.0)
    _fft(left)
    _fft(right)
    for index in range(fft_length):
        left[index] *= right[index]
    _fft(left, inverse=True)
    return [left[index].real for index in range(output_length)]


def render_measured_rir(
    samples: list[int],
    mic1_path: pathlib.Path,
    mic2_path: pathlib.Path,
) -> tuple[list[float], list[float], dict]:
    mic1_path = mic1_path.resolve()
    mic2_path = mic2_path.resolve()
    if not mic1_path.is_file() or not mic2_path.is_file():
        raise ValueError("measured RIR files must exist")
    left_rir = read_rir(mic1_path)
    right_rir = read_rir(mic2_path)
    left = fft_convolve(samples, left_rir)
    right = fft_convolve(samples, right_rir)
    left_onset = onset_sample(left_rir)
    right_onset = onset_sample(right_rir)
    return left, right, {
        "rir_backend": "measured-dual-mic-v1",
        "mic1_rir_sha256": sha256_file(mic1_path),
        "mic2_rir_sha256": sha256_file(mic2_path),
        "mic1_rir_samples": len(left_rir),
        "mic2_rir_samples": len(right_rir),
        "mic1_onset_samples": left_onset,
        "mic2_onset_samples": right_onset,
        "direct_delay_samples": min(left_onset, right_onset),
        "itd_samples": float(right_onset - left_onset),
    }
