from __future__ import annotations

import hashlib
from collections import OrderedDict

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

# train_ctc repeatedly visits the same immutable WAV corpus for many epochs.
# Feature extraction is deterministic and has no trainable parameters, so a
# content-addressed cache removes repeated FFT/mel work without changing any
# sample, model arithmetic, optimizer step, or qualification gate. Keep the
# cache process-local and bounded; each domain round already runs train_ctc in
# a fresh process, so memory is released between rounds.
_FEATURE_CACHE_MAX_ENTRIES = 4096
_FEATURE_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _cache_key(
    wave: torch.Tensor,
    *,
    feature_dim: int,
    frame_len: int,
    hop: int,
    frontend: str,
) -> tuple | None:
    # The production training path is deterministic CPU inference over waveform
    # tensors with no gradient requirement. Do not cache accelerator tensors or
    # differentiable callers, preserving generic frontend semantics.
    if wave.device.type != "cpu" or wave.requires_grad:
        return None
    contiguous = wave.detach().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(raw).digest()
    return (
        digest,
        int(wave.numel()),
        str(wave.dtype),
        int(feature_dim),
        int(frame_len),
        int(hop),
        str(frontend),
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

    key = _cache_key(
        wave,
        feature_dim=feature_dim,
        frame_len=frame_len,
        hop=hop,
        frontend=frontend,
    )
    if key is not None:
        cached = _FEATURE_CACHE.get(key)
        if cached is not None:
            _FEATURE_CACHE.move_to_end(key)
            return cached

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

    output = (result - result.mean(dim=1, keepdim=True)) * FEATURE_SCALE
    if key is not None:
        _FEATURE_CACHE[key] = output
        _FEATURE_CACHE.move_to_end(key)
        while len(_FEATURE_CACHE) > _FEATURE_CACHE_MAX_ENTRIES:
            _FEATURE_CACHE.popitem(last=False)
    return output
