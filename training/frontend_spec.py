from __future__ import annotations

import math

SAMPLE_RATE_HZ = 16000
FFT_SIZE = 512
MEL_LOW_HZ = 80.0
MEL_HIGH_HZ = 7600.0
ENERGY_SCALE = 32.0
FEATURE_SCALE = 0.25


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_bins(feature_dim: int) -> list[int]:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    lo = hz_to_mel(MEL_LOW_HZ)
    hi = hz_to_mel(MEL_HIGH_HZ)
    result: list[int] = []
    for index in range(feature_dim + 2):
        hz = mel_to_hz(lo + (index / (feature_dim + 1)) * (hi - lo))
        result.append(min(FFT_SIZE // 2, int(math.floor(513.0 * hz / SAMPLE_RATE_HZ))))
    return result


def hann_window(frame_len: int) -> list[float]:
    if frame_len < 2:
        raise ValueError("frame_len must be >= 2")
    return [
        0.5 - 0.5 * math.cos((2.0 * math.pi * index) / (frame_len - 1))
        for index in range(frame_len)
    ]


def fft512_real(values: list[float]) -> list[complex]:
    if len(values) > FFT_SIZE:
        raise ValueError("input exceeds 512-point FFT")
    data = [complex(value, 0.0) for value in values]
    data.extend([0j] * (FFT_SIZE - len(data)))

    j = 0
    for i in range(1, FFT_SIZE):
        bit = FFT_SIZE // 2
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            data[i], data[j] = data[j], data[i]

    length = 2
    while length <= FFT_SIZE:
        angle = -2.0 * math.pi / length
        wlen = complex(math.cos(angle), math.sin(angle))
        for start in range(0, FFT_SIZE, length):
            w = 1.0 + 0.0j
            half = length // 2
            for offset in range(half):
                left = data[start + offset]
                right = data[start + offset + half] * w
                data[start + offset] = left + right
                data[start + offset + half] = left - right
                w *= wlen
        length *= 2
    return data[: FFT_SIZE // 2 + 1]


def frame_features_pcm16(samples: list[int], feature_dim: int = 32) -> list[float]:
    frame_len = len(samples)
    window = hann_window(frame_len)
    normalized = [sample / 32768.0 for sample in samples]
    spectrum = fft512_real(
        [normalized[index] * window[index] for index in range(frame_len)]
    )
    power = [value.real * value.real + value.imag * value.imag for value in spectrum]
    bins = mel_bins(feature_dim)
    features: list[float] = []
    for mel_index in range(feature_dim):
        left, center, right = bins[mel_index : mel_index + 3]
        center = max(center, left + 1)
        right = min(FFT_SIZE // 2 + 1, max(right, center + 1))
        energy = 0.0
        for bin_index in range(left, min(center, FFT_SIZE // 2 + 1)):
            weight = (bin_index - left) / (center - left)
            energy += weight * power[bin_index]
        for bin_index in range(center, right):
            weight = (right - bin_index) / (right - center)
            energy += weight * power[bin_index]
        features.append(math.log1p(ENERGY_SCALE * energy))
    mean = sum(features) / feature_dim
    return [(value - mean) * FEATURE_SCALE for value in features]


def features_pcm16(
    samples: list[int],
    feature_dim: int = 32,
    frame_len: int = 400,
    hop: int = 320,
) -> list[list[float]]:
    if frame_len < 2 or frame_len > FFT_SIZE or hop <= 0 or hop > frame_len:
        raise ValueError("invalid frontend geometry")
    if len(samples) < frame_len:
        samples = samples + [0] * (frame_len - len(samples))
    result: list[list[float]] = []
    offset = 0
    while offset + frame_len <= len(samples):
        result.append(frame_features_pcm16(samples[offset : offset + frame_len], feature_dim))
        offset += hop
    return result
