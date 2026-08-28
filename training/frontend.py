from __future__ import annotations

import torch

from frontend_spec import ENERGY_SCALE, FEATURE_SCALE, FFT_SIZE, mel_bins


def features(
    wave: torch.Tensor,
    feature_dim: int = 32,
    frame_len: int = 400,
    hop: int = 320,
) -> torch.Tensor:
    if wave.ndim != 1:
        raise ValueError("wave must be mono")
    if wave.numel() < frame_len:
        wave = torch.nn.functional.pad(wave, (0, frame_len - wave.numel()))

    frames = wave.unfold(0, frame_len, hop)
    window = torch.hann_window(
        frame_len,
        periodic=False,
        dtype=wave.dtype,
        device=wave.device,
    )
    spectrum = torch.fft.rfft(frames * window, n=FFT_SIZE)
    power = spectrum.real.square() + spectrum.imag.square()
    bins = mel_bins(feature_dim)
    feature_columns: list[torch.Tensor] = []

    for mel_index in range(feature_dim):
        left, center, right = bins[mel_index : mel_index + 3]
        center = max(center, left + 1)
        right = min(FFT_SIZE // 2 + 1, max(right, center + 1))
        energy = wave.new_zeros((frames.shape[0],))
        if left < center:
            weights = (
                torch.arange(
                    left,
                    center,
                    device=wave.device,
                    dtype=wave.dtype,
                )
                - left
            ) / (center - left)
            energy = energy + (power[:, left:center] * weights).sum(dim=1)
        if center < right:
            weights = (
                right
                - torch.arange(
                    center,
                    right,
                    device=wave.device,
                    dtype=wave.dtype,
                )
            ) / (right - center)
            energy = energy + (power[:, center:right] * weights).sum(dim=1)
        feature_columns.append(torch.log1p(ENERGY_SCALE * energy))

    result = torch.stack(feature_columns, dim=1)
    return (result - result.mean(dim=1, keepdim=True)) * FEATURE_SCALE
