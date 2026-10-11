"""Unexecuted research-only streaming source, outside the product model path.

Importing this module only defines functions and imports the stdlib-only spec.
PyTorch is imported only by an explicit ``build_encoder`` call. This source has
no launcher, checkpoint loader, weight download, training loop or export path.

C0 delegates to a caller-owned TinyStreamingRNN-compatible instance. Structural
checks do not authenticate its implementation, checkpoint or provenance. C1 is
an original local causal depthwise/pointwise residual variant, not a copied
upstream implementation or a claim to reproduce the published WeKWS recipe.

Both interfaces consume [batch, time, feature] and emit [time, batch, vocabulary]
logits. Streaming state is always an explicit caller-owned tuple. The caller
must keep each batch row bound to the same stream and discard the entire tuple
at a session change, discontinuity or batch-row reorder. There is no internal
session cache. ``forward`` starts a new sequence; ``forward_chunk`` never resets
silently. An empty chunk inserts no frames and preserves the supplied state.

This source has not established numerical parity, training correctness, target
cost, allocation bounds for activations/autograd, or product qualification.
State tensor *storage* is bounded by the spec; autograd history is deliberately
not detached and can grow while the caller retains a computation graph.
"""

from __future__ import annotations

from .contract import EncoderSpec


def _retain_history(layer_inputs, history_frames: int):
    """Own a compact oldest-to-newest tail, including the kernel=1 empty case.

    A view alone would keep an arbitrarily large chunk's backing storage alive.
    ``-0:`` is the whole tensor, so zero history must use the separate ``:0``
    branch. Cloning gives the returned tensor compact independent storage while
    preserving gradients; autograd may retain additional tensors separately.
    """
    if history_frames == 0:
        return layer_inputs[:, :, :0].clone().contiguous()
    return layer_inputs[:, :, -history_frames:].clone().contiguous()


def build_encoder(spec: EncoderSpec, *, rnn_model=None):
    """Explicitly construct a source-only candidate when separately authorized.

    Every architecture dimension comes from a validated ``EncoderSpec``. C0
    requires an already-created reference model; this module does not import
    ``training.model`` or modify its global environment/backend settings. C1
    constructs float32 CPU parameters, consuming the caller's RNG state. There
    is no implicit seed, pretrained initialization or dtype conversion. A caller
    must explicitly move a model before supplying another device's tensors.
    Model construction/execution is not part of metadata-only CI.
    """
    if type(spec) is not EncoderSpec:
        raise TypeError("spec must be an EncoderSpec")
    if spec.architecture == "rnn64_control" and rnn_model is None:
        raise ValueError("C0 requires an explicit caller-owned reference model")
    if spec.architecture == "causal_ds_tcn_ctc" and rnn_model is not None:
        raise ValueError("C1 does not accept a reference RNN model")

    import torch
    from torch import nn

    class StreamingEncoder(nn.Module):
        """Shared explicit state interface; never keeps a session's tensors."""

        def __init__(self):
            super().__init__()
            self.spec = spec

        def _check_parameters(self, *, device, dtype):
            if dtype != torch.float32:
                raise ValueError("this research source requires float32 tensors")
            for parameter in self.parameters():
                if parameter.dtype != dtype or parameter.device != device:
                    raise ValueError("model, input and state must share dtype/device")

        def _check_tensor(self, value, shape, *, device):
            if not isinstance(value, torch.Tensor):
                raise TypeError("inputs and state entries must be tensors")
            if tuple(value.shape) != shape:
                raise ValueError("unexpected input/output/state shape")
            if value.dtype != torch.float32 or value.device != device:
                raise ValueError("input/output/state dtype or device mismatch")

        def _check_input(self, x, *, sequence):
            if not isinstance(x, torch.Tensor):
                raise TypeError("input must be a tensor")
            expected_rank = 3 if sequence else 2
            if x.ndim != expected_rank or x.shape[-1] != spec.feature_dim:
                raise ValueError("input must have the configured feature dimension")
            spec.state_shapes(x.shape[0])
            self._check_parameters(device=x.device, dtype=x.dtype)

        def _check_state(self, x, state):
            expected_shapes = spec.state_shapes(x.shape[0])
            if type(state) is not tuple or len(state) != len(expected_shapes):
                raise ValueError("state must be the complete explicit layer tuple")
            for value, shape in zip(state, expected_shapes):
                self._check_tensor(value, shape, device=x.device)

        def init_state(self, batch_size: int, *, device, dtype):
            """Create all-zero state with explicit batch, device and dtype."""
            shapes = spec.state_shapes(batch_size)
            if device is None or dtype is None:
                raise ValueError("device and dtype must be explicit")
            selected_device = torch.device(device)
            self._check_parameters(device=selected_device, dtype=dtype)
            return tuple(torch.zeros(shape, device=selected_device, dtype=dtype)
                         for shape in shapes)

        def reset_state(self, batch_size: int, *, device, dtype):
            """Reset every layer; call after any session or timeline gap."""
            return self.init_state(batch_size, device=device, dtype=dtype)

        def step(self, x, state):
            """Consume exactly one feature frame, without lookahead or flush."""
            self._check_input(x, sequence=False)
            logits, next_state = self.forward_chunk(x.unsqueeze(1), state)
            return logits[0], next_state

        def forward(self, x, lengths=None):
            """Full-sequence reset path; ragged/padded lengths are unsupported."""
            if lengths is not None:
                raise ValueError("split unpadded streams; lengths are unsupported")
            self._check_input(x, sequence=True)
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
            logits, _ = self.forward_chunk(x, state)
            return logits

    class RNNReferenceAdapter(StreamingEncoder):
        """Delegate C0 math to the supplied reference; do not duplicate it."""

        def __init__(self, reference):
            super().__init__()
            if not isinstance(reference, nn.Module) or not callable(
                getattr(reference, "step", None)
            ):
                raise TypeError("C0 reference must be an nn.Module with step")
            expected_layers = (
                ("in_proj", spec.feature_dim, spec.hidden_dim, True),
                ("rec_proj", spec.hidden_dim, spec.hidden_dim, False),
                ("out_proj", spec.hidden_dim, spec.vocab_size, True),
            )
            expected_parameters = {}
            for name, inputs, outputs, has_bias in expected_layers:
                layer = getattr(reference, name, None)
                if not isinstance(layer, nn.Linear):
                    raise ValueError("C0 must have the reference Linear structure")
                if layer.in_features != inputs or layer.out_features != outputs:
                    raise ValueError("C0 reference dimensions differ from the spec")
                if (layer.bias is not None) != has_bias:
                    raise ValueError("C0 reference bias structure differs")
                expected_parameters[name + ".weight"] = (outputs, inputs)
                if has_bias:
                    expected_parameters[name + ".bias"] = (outputs,)
            actual_parameters = {
                name: tuple(parameter.shape)
                for name, parameter in reference.named_parameters()
            }
            if actual_parameters != expected_parameters:
                raise ValueError("C0 reference parameter structure differs")
            self.rnn_model = reference

        def step(self, x, state):
            self._check_input(x, sequence=False)
            self._check_state(x, state)
            logits, hidden = self.rnn_model.step(x, state[0])
            self._check_tensor(logits, (x.shape[0], spec.vocab_size), device=x.device)
            self._check_tensor(hidden, (x.shape[0], spec.hidden_dim), device=x.device)
            return logits, (hidden,)

        def forward_chunk(self, x, state):
            self._check_input(x, sequence=True)
            self._check_state(x, state)
            if x.shape[1] == 0:
                return x.new_empty((0, x.shape[0], spec.vocab_size)), state
            outputs = []
            next_state = state
            for frame in range(x.shape[1]):
                logits, next_state = self.step(x[:, frame, :], next_state)
                outputs.append(logits)
            return torch.stack(outputs, dim=0), next_state

    class CausalResidualBlock(nn.Module):
        """Zero-lookahead DW→ReLU→PW→ReLU→residual, stride exactly one."""

        def __init__(self, kernel_size: int, dilation: int):
            super().__init__()
            self.depthwise = nn.Conv1d(
                spec.hidden_dim, spec.hidden_dim, kernel_size,
                stride=1, padding=0, dilation=dilation,
                groups=spec.hidden_dim, bias=True, device="cpu", dtype=torch.float32,
            )
            self.pointwise = nn.Conv1d(
                spec.hidden_dim, spec.hidden_dim, 1,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                device="cpu", dtype=torch.float32,
            )
            self.history_frames = (kernel_size - 1) * dilation

        def forward_chunk(self, layer_inputs, history):
            # Only this layer's prior inputs precede the current chunk. Padding
            # happens once in init_state; chunk boundaries add no new zeros.
            joined = torch.cat((history, layer_inputs), dim=2)
            temporal = torch.relu(self.depthwise(joined))
            projected = torch.relu(self.pointwise(temporal))
            outputs = layer_inputs + projected
            next_history = _retain_history(joined, self.history_frames)
            return outputs, next_history

    class CausalDepthwiseEncoder(StreamingEncoder):
        """Reduced local source variant; no normalization, dropout or subsample."""

        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(
                spec.feature_dim, spec.hidden_dim, bias=True,
                device="cpu", dtype=torch.float32,
            )
            self.blocks = nn.ModuleList(
                CausalResidualBlock(kernel, dilation)
                for kernel, dilation in zip(spec.kernels, spec.dilations)
            )
            self.out_proj = nn.Linear(
                spec.hidden_dim, spec.vocab_size, bias=True,
                device="cpu", dtype=torch.float32,
            )

        def forward_chunk(self, x, state):
            self._check_input(x, sequence=True)
            self._check_state(x, state)
            if x.shape[1] == 0:
                return x.new_empty((0, x.shape[0], spec.vocab_size)), state
            layer_inputs = torch.relu(self.in_proj(x)).transpose(1, 2)
            next_state = []
            for block, history in zip(self.blocks, state):
                layer_inputs, next_history = block.forward_chunk(layer_inputs, history)
                next_state.append(next_history)
            logits = self.out_proj(layer_inputs.transpose(1, 2)).transpose(0, 1)
            return logits, tuple(next_state)

    if spec.architecture == "rnn64_control":
        return RNNReferenceAdapter(rnn_model)
    return CausalDepthwiseEncoder()
