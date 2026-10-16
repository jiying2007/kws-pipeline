"""Direct background/two-keyword classification, never CTC token decoding.

The existing research C1 factory supplies only its input projection and causal
residual blocks. Its five-token projection is discarded, and is neither a
parameter of this model nor called by it. Frozen C1 source is left unchanged.
Torch is loaded only when the caller explicitly builds or evaluates the model.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.candidate_encoders.contract import causal_ds_tcn_spec
from research.candidate_encoders.torch_models import build_encoder

CLASS_NAMES = ("background", "你好小窝", "小窝小窝")


@dataclass(frozen=True)
class DirectKWSSpec:
    """Explicit topology, using the existing bounded 32-feature backbone spec."""

    hidden_dim: int
    kernels: tuple[int, ...]
    dilations: tuple[int, ...]

    def __post_init__(self):
        self.backbone_spec()  # Reuse topology/allocation checks, not CTC labels.

    def backbone_spec(self):
        return causal_ds_tcn_spec(
            hidden_dim=self.hidden_dim, kernels=self.kernels, dilations=self.dilations)

    @property
    def context_frames(self):
        return 1 + sum(self.backbone_spec().history_frames)

    @classmethod
    def from_dict(cls, value):
        keys = {"schema", "architecture", "feature_dim", "classes", "hidden_dim",
                "kernels", "dilations", "objective"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("direct model spec missing or extra fields")
        if type(value["kernels"]) is not list or type(value["dilations"]) is not list:
            raise ValueError("serialized layer dimensions must be lists")
        spec = cls(value["hidden_dim"], tuple(value["kernels"]), tuple(value["dilations"]))
        if value != spec.to_dict():
            raise ValueError("direct model class, architecture or objective identity mismatch")
        return spec

    def to_dict(self):
        return {
            "schema": "research-direct-kws-spec-v1",
            "architecture": "causal_ds_tcn_direct",
            "feature_dim": 32, "classes": list(CLASS_NAMES),
            "hidden_dim": self.hidden_dim,
            "kernels": list(self.kernels), "dilations": list(self.dilations),
            "objective": "terminal-window-cross-entropy",
        }


def build_classifier(spec: DirectKWSSpec):
    """Build a float32 CPU model, consuming the caller's RNG state.

    This intentionally reuses C1's existing block implementation without changing
    its frozen source, spec or CTC interface. The temporary C1 output projection
    consumes initialization RNG but is discarded. Bind both source files and the
    seed for reproducibility. No model, seed, device or backend is selected here.
    """
    if type(spec) is not DirectKWSSpec:
        raise TypeError("spec must be a DirectKWSSpec")
    import torch
    from torch import nn

    class DirectClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            source = build_encoder(spec.backbone_spec())
            self.in_proj = source.in_proj
            self.blocks = source.blocks
            self.classifier = nn.Linear(
                spec.hidden_dim, len(CLASS_NAMES), device="cpu", dtype=torch.float32)
            # Do not retain source or its out_proj: no unused CTC parameters.

        def init_state(self, batch_size: int, *, device, dtype):
            shapes = spec.backbone_spec().state_shapes(batch_size)
            if device is None or dtype is None:
                raise ValueError("device and dtype must be explicit")
            device = torch.device(device)
            self._check_parameters(device, dtype)
            return tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)

        def reset_state(self, batch_size: int, *, device, dtype):
            return self.init_state(batch_size, device=device, dtype=dtype)

        def _check_parameters(self, device, dtype):
            if dtype != torch.float32 or any(
                p.dtype != dtype or p.device != device for p in self.parameters()
            ):
                raise ValueError("model/input/state must share float32 dtype and device")

        def forward_chunk(self, x, state):
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[2] != 32:
                raise ValueError("features must have shape [batch,time,32]")
            shapes = spec.backbone_spec().state_shapes(x.shape[0])
            self._check_parameters(x.device, x.dtype)
            if type(state) is not tuple or len(state) != len(shapes):
                raise ValueError("state must contain every layer's history")
            for value, shape in zip(state, shapes):
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != x.dtype or value.device != x.device):
                    raise ValueError("state shape, dtype or device mismatch")
            if not torch.isfinite(x).all() or any(not torch.isfinite(s).all() for s in state):
                raise ValueError("features and state must be finite")
            if x.shape[1] == 0:
                return x.new_empty((0, x.shape[0], len(CLASS_NAMES))), state
            hidden = torch.relu(self.in_proj(x)).transpose(1, 2)
            next_state = []
            for block, history in zip(self.blocks, state):
                hidden, tail = block.forward_chunk(hidden, history)
                next_state.append(tail)
            logits = self.classifier(hidden.transpose(1, 2)).transpose(0, 1)
            return logits, tuple(next_state)

        def step(self, x, state):
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                raise ValueError("step expects [batch,32]")
            logits, state = self.forward_chunk(x.unsqueeze(1), state)
            return logits[0], state

        def forward(self, x, lengths=None):
            if lengths is not None:
                raise ValueError("use window_cross_entropy lengths for padded window batches")
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                raise ValueError("features must have shape [batch,time,32]")
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
            return self.forward_chunk(x, state)[0]

    return DirectClassifier()


def window_cross_entropy(logits, targets, lengths=None):
    """CE on each labeled window's final valid frame, not CTC or frame alignment.

    Targets are 0=background, 1=你好小窝, 2=小窝小窝. Every label must describe
    the causal window at that endpoint. A whole-clip presence label alone does
    not establish an endpoint/window label. No keyword timing is inferred here.
    Padded tails are excluded using lengths; no masking or synthetic blank is
    applied to the remaining frames. Caller owns the optimizer and run budget.
    """
    import torch
    from torch.nn import functional as F

    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
            or logits.shape[2] != len(CLASS_NAMES) or min(logits.shape[:2]) < 1
            or logits.dtype != torch.float32):
        raise ValueError("logits must be nonempty float32 [time,batch,3]")
    frames, batch, _ = logits.shape
    if (not isinstance(targets, torch.Tensor) or targets.shape != (batch,)
            or targets.dtype != torch.long or targets.device != logits.device
            or bool(((targets < 0) | (targets >= len(CLASS_NAMES))).any())):
        raise ValueError("targets must be long [batch] in 0..2 on the logits device")
    if lengths is None:
        lengths = torch.full((batch,), frames, dtype=torch.long, device=logits.device)
    if (not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,)
            or lengths.dtype != torch.long or lengths.device != logits.device
            or bool(((lengths < 1) | (lengths > frames)).any())):
        raise ValueError("lengths must be long [batch] in 1..time on the logits device")
    selected = logits[lengths - 1, torch.arange(batch, device=logits.device)]
    if not torch.isfinite(selected).all():
        raise ValueError("selected endpoint logits must be finite")
    return F.cross_entropy(selected, targets)
