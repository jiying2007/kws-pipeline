from __future__ import annotations

import os

# Keep the deterministic CPU contract visible in a training code file that is
# hashed into exported model provenance.  sitecustomize.py applies the same
# values before torch import; repeating them here makes the contract fail-safe
# for direct module callers that bypass normal interpreter startup discovery.
_DETERMINISTIC_CPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "default",
}
for _name, _value in _DETERMINISTIC_CPU_ENV.items():
    os.environ[_name] = _value

import torch
from torch import nn

# Fixed RNG seeds and torch deterministic algorithms still leave CPU math
# topology/dispatch implicit unless both pools and MKLDNN are pinned.  Use one
# thread for both pools and disable MKLDNN so model/FFT/linear execution does
# not depend on hosted-runner thread scheduling or oneDNN kernel selection.
TRAINING_TORCH_NUM_THREADS = 1
TRAINING_TORCH_NUM_INTEROP_THREADS = 1

torch.set_num_threads(TRAINING_TORCH_NUM_THREADS)
torch.set_num_interop_threads(TRAINING_TORCH_NUM_INTEROP_THREADS)
torch.backends.mkldnn.enabled = False


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
