from __future__ import annotations

import torch

from frontend_spec import (
    ENERGY_SCALE,
    FEATURE_SCALE,
    FFT_SIZE,
    FRONTEND_LOGMEL,
    FRONTEND_PCEN_LITE,
    PCEN_ALPHA,
    PCEN_DELTA,
    PCEN_EPSILON,
    PCEN_ROOT,
    PCEN_SMOOTHING,
    frontend_id,
    mel_bins,
)


def features(
    wave: torch.Tensor,
    feature_dim: int = 32,
    frame_len: int = 400,
    hop: int = 320,
    frontend: str = FRONTEND_LOGMEL,
) -> torch.Tensor:
    frontend_id(frontend)
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
    energy_columns: list[torch.Tensor] = []

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
        energy_columns.append(energy)

    energies = torch.stack(energy_columns, dim=1)
    if frontend == FRONTEND_LOGMEL:
        result = torch.log1p(ENERGY_SCALE * energies)
    elif frontend == FRONTEND_PCEN_LITE:
        smooth_rows: list[torch.Tensor] = []
        smooth = energies[0]
        smooth_rows.append(smooth)
        for frame_index in range(1, energies.shape[0]):
            smooth = (
                (1.0 - PCEN_SMOOTHING) * smooth
                + PCEN_SMOOTHING * energies[frame_index]
            )
            smooth_rows.append(smooth)
        smoother = torch.stack(smooth_rows, dim=0)
        normalized = energies / (PCEN_EPSILON + smoother).pow(PCEN_ALPHA)
        result = (normalized + PCEN_DELTA).pow(PCEN_ROOT) - PCEN_DELTA**PCEN_ROOT
    else:
        raise ValueError(f"unsupported frontend: {frontend}")
    return (result - result.mean(dim=1, keepdim=True)) * FEATURE_SCALE
