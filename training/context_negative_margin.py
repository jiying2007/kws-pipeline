"""Opt-in negative-only acoustic path margin for grounded development training.

A chronological max-path SURROGATE, not an exact C decoder or transcript score.
The positive target's own wake path is never penalized. Only activity frames
participate; ordinary transcribed CTC and non-speech blank supervision remain.
"""
from __future__ import annotations

import math
import torch

POLICY = 'activity-negative-wake-path-margin-v1'
KEYWORD_SEQUENCES = ((1, 2, 3, 4), (3, 4, 3, 4))
CEILING = 0.50


def negative_wake_margin(log_probs: torch.Tensor, targets: list[torch.Tensor],
                         lengths: list[int], spans: list[tuple[int, int]]) -> torch.Tensor:
    """Mean per-row strongest false-wake hinge in log-confidence units.

    All controls remain fixed: the threshold is a training margin, NOT a new
    C keyword threshold. Mask before cummax, never zero-pad real probabilities.
    """
    if (log_probs.ndim != 3 or log_probs.shape[2] != 5
            or len(targets) != log_probs.shape[1]
            or len(lengths) != len(targets) or len(spans) != len(targets)):
        raise ValueError('negative margin requires [T,B,5] and aligned batch metadata')
    if not targets or not torch.isfinite(log_probs).all():
        raise ValueError('negative margin requires finite nonempty inputs')
    steps, batch, _ = log_probs.shape
    active = torch.zeros((steps, batch), dtype=torch.bool, device=log_probs.device)
    labels = []
    for j, (y, n, span) in enumerate(zip(targets, lengths, spans)):
        if (y.ndim != 1 or y.dtype != torch.long or not y.numel()
                or type(n) is not int or not isinstance(span, (tuple, list)) or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= n <= steps):
            raise ValueError('invalid negative-margin target or activity span')
        label = tuple(y.detach().cpu().tolist())
        if any(v <= 0 or v >= 5 for v in label):
            raise ValueError('negative-margin targets must be nonblank vocabulary tokens')
        labels.append(label)
        active[span[0]:span[1], j] = True
    # A finite floor avoids undefined max(-inf, -inf) gradients on impossible
    # paths. Every valid log-probability is greater than this sentinel in normal
    # training; inputs outside this bound are rejected, never silently clipped.
    floor = -1.0e6
    if (log_probs < floor / 8).any() or (log_probs > 1.0e-5).any():
        raise ValueError('negative-margin input is outside log-probability bounds')
    scores = []
    for word in KEYWORD_SEQUENCES:
        path = log_probs[:, :, word[0]].masked_fill(~active, floor)
        for token in word[1:]:
            best = path.cummax(dim=0).values
            shifted = torch.cat([best.new_full((1, batch), floor), best[:-1]], dim=0)
            path = (shifted + log_probs[:, :, token]).masked_fill(~active, floor)
        score = path.amax(dim=0) / len(word)
        negative = torch.tensor([label != word for label in labels], device=log_probs.device)
        scores.append(torch.relu(score - math.log(CEILING)) * negative)
    return torch.stack(scores).amax(dim=0).mean()
