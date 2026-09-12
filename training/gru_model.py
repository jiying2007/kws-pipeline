from __future__ import annotations

import torch
from torch import nn


ARCHITECTURE = "tiny-gru-v1"


class TinyStreamingGRU(nn.Module):
    """One-state streaming GRU acoustic model for development-only A/B trials."""

    def __init__(self, feature_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.gru = nn.GRUCell(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def step(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.gru(x, h)
        return self.out_proj(h), h

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        del lengths
        h = x.new_zeros((x.shape[0], self.gru.hidden_size))
        ys = []
        for t in range(x.shape[1]):
            y, h = self.step(x[:, t, :], h)
            ys.append(y)
        return torch.stack(ys, dim=0)
