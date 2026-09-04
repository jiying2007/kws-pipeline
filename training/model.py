from __future__ import annotations

import torch
from torch import nn

# GitHub-hosted CPU runners have exposed different default inter-op thread
# counts across otherwise identical training runs (for example 4 vs 2).  The
# training loop already requests deterministic algorithms and a fixed RNG seed,
# so leave no scheduler topology implicit: pin both pools before model creation
# or any training work.  Keep this in the training-only model module so every
# caller that can instantiate TinyStreamingRNN receives the same runtime
# topology, including warm-start and reproducibility probes.
TRAINING_TORCH_NUM_THREADS = 2
TRAINING_TORCH_NUM_INTEROP_THREADS = 1

torch.set_num_threads(TRAINING_TORCH_NUM_THREADS)
torch.set_num_interop_threads(TRAINING_TORCH_NUM_INTEROP_THREADS)


class TinyStreamingRNN(nn.Module):
    """One-state streaming acoustic model mirrored by src/kws.c."""

    def __init__(self, feature_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.rec_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def step(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.in_proj(x) + self.rec_proj(h))
        return self.out_proj(h), h

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        del lengths
        h = x.new_zeros((x.shape[0], self.rec_proj.in_features))
        ys = []
        for t in range(x.shape[1]):
            y, h = self.step(x[:, t, :], h)
            ys.append(y)
        return torch.stack(ys, dim=0)
