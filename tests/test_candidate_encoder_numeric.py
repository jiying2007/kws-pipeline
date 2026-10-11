#!/usr/bin/env python3
"""Bounded CPU numeric contracts on random features and random initial weights.

No audio, optimizer/update, checkpoint, export, network access or model download.
This checks only the isolated float32 source API, not training quality, the C
runtime, integer parity, acoustic effectiveness or SSC305 qualification. Missing
Torch is an error, never a silently skipped numeric pass. Run in the existing
pinned GitHub CPU environment; this suite does not install dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

# Pin dispatch before the first Torch import, including when invoked by path.
CPU_ENV = {
    "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "ATEN_CPU_CAPABILITY": "default",
}
for name, value in CPU_ENV.items():
    os.environ[name] = value
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The real reference establishes single-thread intra/inter-op and disables
# MKLDNN. Import it once, before construction; do not copy its equations.
from training.model import TinyStreamingRNN
import torch
from torch.nn import functional as F
from research.candidate_encoders.contract import (
    C0_SOURCE_BLOB, C0_SOURCE_PATH, causal_ds_tcn_spec, rnn64_spec,
)
from research.candidate_encoders.torch_models import build_encoder, _retain_history

SEED = 20261011
BATCH = 2
FRAMES = 13
CHUNKS = (2, 0, 1, 4, 0, 6)
# A single toy topology exercises multiple layers, dilation and kernel=1.
# These are fixed test dimensions, not a candidate recipe or a topology sweep.
SPECS = (rnn64_spec(), causal_ds_tcn_spec(
    hidden_dim=8, kernels=(3, 1, 2), dilations=(1, 3, 2)))
ATOL = 2e-6
RTOL = 2e-5
GRAD_ATOL = 2e-5
GRAD_RTOL = 1e-4
OBSERVATIONS = []
torch.use_deterministic_algorithms(True)


def make_model(spec):
    torch.manual_seed(SEED)
    reference = (TinyStreamingRNN(spec.feature_dim, spec.hidden_dim, spec.vocab_size)
                 if spec.architecture == "rnn64_control" else None)
    return build_encoder(spec, rnn_model=reference)


def features():
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return torch.randn(BATCH, FRAMES, 32, generator=generator)


def initial_state(model, *, nonzero=False):
    state = model.init_state(BATCH, device="cpu", dtype=torch.float32)
    if nonzero:
        generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
        return tuple(torch.randn(value.shape, generator=generator) for value in state)
    return state


def execute(model, x, state, mode):
    if mode == "whole":
        return model.forward_chunk(x, state)
    if mode == "frames":
        outputs = []
        for index in range(x.shape[1]):
            output, state = model.step(x[:, index, :], state)
            outputs.append(output)
        return torch.stack(outputs), state
    if mode != "chunks" or x.shape[1] != sum(CHUNKS):
        raise ValueError("unknown mode or mismatched fixed fixture length")
    outputs = []
    position = 0
    for size in CHUNKS:
        output, state = model.forward_chunk(x[:, position:position + size, :], state)
        outputs.append(output)
        position += size
    return torch.cat(outputs), state


def independent_c1(model, x, state):
    """Literal temporal-tap oracle, independently of Conv1d/streaming methods."""
    values = F.linear(x, model.in_proj.weight, model.in_proj.bias).relu().transpose(1, 2)
    next_state = []
    for index, block in enumerate(model.blocks):
        joined = torch.cat((state[index], values), dim=2)
        depthwise = block.depthwise
        kernel, dilation = model.spec.kernels[index], model.spec.dilations[index]
        temporal = depthwise.bias[None, :, None].expand_as(values)
        # PyTorch convolution is cross-correlation: oldest tap has index zero.
        for tap in range(kernel):
            start = tap * dilation
            temporal = temporal + joined[:, :, start:start + x.shape[1]] * depthwise.weight[:, 0, tap][None, :, None]
        pointwise = F.linear(temporal.relu().transpose(1, 2),
                             block.pointwise.weight[:, :, 0], block.pointwise.bias)
        values = values + pointwise.relu().transpose(1, 2)
        history = model.spec.history_frames[index]
        next_state.append(joined[:, :, joined.shape[2] - history:].clone())
    output = F.linear(values.transpose(1, 2), model.out_proj.weight,
                      model.out_proj.bias).transpose(0, 1)
    return output, tuple(next_state)


def gradients(model, x, state, mode):
    leaf = x.detach().clone().requires_grad_(True)
    states = tuple(value.detach().clone().requires_grad_(True) for value in state)
    output, final_state = (independent_c1(model, leaf, states) if mode == "oracle"
                           else execute(model, leaf, states, mode))
    # Include all output times and final state. autograd.grad compares automatic
    # derivatives across execution layouts without training or updating weights.
    loss = output.square().mean() + sum(value.square().sum() * 0.01 for value in final_state)
    parameters = tuple(model.parameters())
    grads = torch.autograd.grad(loss, (leaf,) + states + parameters)
    return output.detach(), tuple(value.detach() for value in final_state), grads


class CandidateEncoderNumericTests(unittest.TestCase):
    def close(self, left, right, *, gradient=False):
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(bool(torch.isfinite(left).all()))
        self.assertTrue(bool(torch.isfinite(right).all()))
        torch.testing.assert_close(left, right,
            atol=GRAD_ATOL if gradient else ATOL,
            rtol=GRAD_RTOL if gradient else RTOL)
        if left.numel():
            OBSERVATIONS.append({"test": self.id().rsplit(".", 1)[-1],
                                 "gradient": gradient,
                                 "max_absolute_error": (left - right).abs().max().item()})

    def state_close(self, left, right):
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            self.close(actual, expected)

    def test_cpu_policy_reference_identity_and_parameter_inventory(self):
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        self.assertFalse(torch.backends.mkldnn.enabled)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        raw = (ROOT / C0_SOURCE_PATH).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        self.assertEqual(blob, C0_SOURCE_BLOB)
        for spec in SPECS:
            model = make_model(spec)
            self.assertEqual(sum(p.numel() for p in model.parameters()), spec.parameter_elements)
            self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                                for p in model.parameters()))

    def test_c0_exact_reference_forward_and_nonzero_state(self):
        model = make_model(SPECS[0])
        x = features()
        self.close(model(x), model.rnn_model(x))
        state = initial_state(model, nonzero=True)
        expected_outputs = []
        hidden = state[0]
        for frame in range(FRAMES):
            logits, hidden = model.rnn_model.step(x[:, frame], hidden)
            expected_outputs.append(logits)
        actual, final_state = model.forward_chunk(x, state)
        self.close(actual, torch.stack(expected_outputs))
        self.close(final_state[0], hidden)
        # Compare gradients with the actual independently invoked reference too.
        actual_input = x.clone().requires_grad_(True)
        expected_input = x.clone().requires_grad_(True)
        params = tuple(model.parameters())
        actual_grads = torch.autograd.grad(model(actual_input).square().mean(), (actual_input,) + params)
        expected_grads = torch.autograd.grad(model.rnn_model(expected_input).square().mean(), (expected_input,) + params)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.close(actual_grad, expected_grad, gradient=True)

    def test_whole_frame_and_uneven_chunk_equivalence(self):
        for spec in SPECS:
            model = make_model(spec)
            for nonzero in (False, True):
                with self.subTest(architecture=spec.architecture, nonzero=nonzero):
                    state = initial_state(model, nonzero=nonzero)
                    expected, expected_state = execute(model, features(), state, "whole")
                    self.assertEqual(tuple(expected.shape), (FRAMES, BATCH, spec.vocab_size))
                    self.assertEqual(tuple(tuple(s.shape) for s in expected_state), spec.state_shapes(BATCH))
                    for mode in ("frames", "chunks"):
                        actual, actual_state = execute(model, features(), state, mode)
                        self.close(actual, expected)
                        self.state_close(actual_state, expected_state)
                    if not nonzero:
                        self.close(model(features()), expected)

    def test_c1_independent_temporal_tap_oracle(self):
        model = make_model(SPECS[1])
        for nonzero in (False, True):
            state = initial_state(model, nonzero=nonzero)
            expected, expected_state = independent_c1(model, features(), state)
            actual, actual_state = model.forward_chunk(features(), state)
            self.close(actual, expected)
            self.state_close(actual_state, expected_state)

    def test_empty_chunks_and_kernel_one_preserve_contract(self):
        for spec in SPECS:
            model = make_model(spec)
            state = initial_state(model, nonzero=True)
            output, same_state = model.forward_chunk(features()[:, :0], state)
            self.assertIs(same_state, state)
            self.assertEqual(tuple(output.shape), (0, BATCH, spec.vocab_size))
            self.assertEqual(tuple(model(features()[:, :0]).shape), (0, BATCH, spec.vocab_size))
            _, final_state = model.forward_chunk(features(), state)
            for value, shape in zip(final_state, spec.state_shapes(BATCH)):
                self.assertEqual(tuple(value.shape), shape)
                if value.numel() == 0:
                    self.assertEqual(value.untyped_storage().nbytes(), 0)
        self.assertEqual(SPECS[1].history_frames, (2, 0, 2))

    def test_reset_and_independent_streams_do_not_leak_state(self):
        for spec in SPECS:
            model = make_model(spec)
            x = features()
            state = initial_state(model, nonzero=True)
            saved = tuple(value.clone() for value in state)
            model.forward_chunk(x, state)
            self.state_close(state, saved)
            reset = model.reset_state(BATCH, device="cpu", dtype=torch.float32)
            fresh = initial_state(model)
            self.state_close(reset, fresh)
            self.assertTrue(all(not bool(value.any()) for value in reset))
            combined, _ = model.forward_chunk(x, reset)
            self.close(combined, model(x))
            # Interleave an unrelated stream, then replay the same explicit state.
            baseline, expected_state = model.forward_chunk(x, state)
            model(x.flip(1))
            replay, replay_state = model.forward_chunk(x, state)
            self.close(replay, baseline)
            self.state_close(replay_state, expected_state)
            for row in range(BATCH):
                separate = model(x[row:row + 1])
                self.close(combined[:, row:row + 1], separate)
            for left, right in zip(reset, fresh):
                if left.numel():
                    self.assertNotEqual(left.untyped_storage().data_ptr(), right.untyped_storage().data_ptr())

    def test_causality_in_outputs_and_input_gradients(self):
        cutoff = 7
        for spec in SPECS:
            model = make_model(spec)
            x = features()
            changed = x.clone()
            changed[:, cutoff:] = changed[:, cutoff:].flip(1) * -3.0 + 2.0
            original_output = model(x)
            changed_output = model(changed)
            self.close(original_output[:cutoff], changed_output[:cutoff])
            self.close(original_output[:cutoff], model(x[:, :cutoff]))
            self.assertGreater((original_output[cutoff:] - changed_output[cutoff:]).abs().max().item(), 1e-6)
            leaf = x.clone().requires_grad_(True)
            derivative, = torch.autograd.grad(model(leaf)[:cutoff].square().sum(), (leaf,))
            self.assertEqual(torch.count_nonzero(derivative[:, cutoff:]).item(), 0)
            self.assertGreater(derivative[:, :cutoff].abs().sum().item(), 0.0)

    def test_finite_equal_gradients_without_parameter_updates(self):
        for spec in SPECS:
            model = make_model(spec)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            state = initial_state(model, nonzero=True)
            expected, expected_state, expected_grads = gradients(model, features(), state, "whole")
            modes = ("frames", "chunks", "oracle") if spec.architecture == "causal_ds_tcn_ctc" else ("frames", "chunks")
            for mode in modes:
                with self.subTest(architecture=spec.architecture, mode=mode):
                    actual, actual_state, actual_grads = gradients(model, features(), state, mode)
                    self.close(actual, expected)
                    self.state_close(actual_state, expected_state)
                    self.assertEqual(len(actual_grads), len(expected_grads))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        self.close(actual_grad, expected_grad, gradient=True)
            self.assertGreater(expected_grads[0].abs().sum().item(), 0.0)
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.equal(parameter, before[name]))
                self.assertIsNone(parameter.grad)

    def test_cross_chunk_autograd_history_is_not_detached(self):
        for spec in SPECS:
            model = make_model(spec)
            leaf = features().clone().requires_grad_(True)
            _, state = model.forward_chunk(leaf[:, :3], initial_state(model))
            continuation, _ = model.forward_chunk(leaf[:, 3:5], state)
            derivative, = torch.autograd.grad(continuation.square().sum(), (leaf,))
            self.assertGreater(derivative[:, :3].abs().sum().item(), 0.0)
            self.assertEqual(torch.count_nonzero(derivative[:, 5:]).item(), 0)

    def test_returned_state_is_compact_nonaliased_and_input_is_unchanged(self):
        for spec in SPECS:
            model = make_model(spec)
            x = features()
            saved_x = x.clone()
            state = initial_state(model, nonzero=True)
            saved_state = tuple(value.clone() for value in state)
            output, next_state = model.forward_chunk(x, state)
            protected = (x, output) + state + tuple(model.parameters())
            for index, value in enumerate(next_state):
                self.assertTrue(value.is_contiguous())
                self.assertEqual(value.storage_offset(), 0)
                self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
                if value.numel():
                    for other in protected + next_state[:index]:
                        if other.numel():
                            self.assertNotEqual(value.untyped_storage().data_ptr(), other.untyped_storage().data_ptr())
            saved_output = output.clone()
            with torch.no_grad():
                for value in next_state:
                    value.add_(100.0)
            self.assertTrue(torch.equal(x, saved_x))
            self.assertTrue(torch.equal(output, saved_output))
            for actual, expected in zip(state, saved_state):
                self.assertTrue(torch.equal(actual, expected))

    def test_real_tail_storage_and_gradient_include_zero_history(self):
        for history in (0, 2):
            source = torch.arange(2 * 8 * 257, dtype=torch.float32).reshape(2, 8, 257).requires_grad_(True)
            tail = _retain_history(source, history)
            self.assertEqual(tuple(tail.shape), (2, 8, history))
            self.assertTrue(tail.is_contiguous())
            self.assertEqual(tail.untyped_storage().nbytes(), tail.numel() * tail.element_size())
            derivative, = torch.autograd.grad(tail.sum(), (source,))
            expected = torch.zeros_like(source)
            if history:
                expected[:, :, -history:] = 1.0
                self.assertNotEqual(tail.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
            self.assertTrue(torch.equal(derivative, expected))

    def test_invalid_lengths_shapes_and_dtypes_fail_closed(self):
        for spec in SPECS:
            model = make_model(spec)
            x = features()
            state = initial_state(model)
            for invalid in (list(state), (), state + (state[0],),
                            tuple(value.double() for value in state),
                            (state[0][:1],) + state[1:]):
                with self.assertRaises(ValueError):
                    model.forward_chunk(x, invalid)
            for invalid in (x.double(), x[:, :, :31], x[0]):
                with self.assertRaises(ValueError):
                    model.forward_chunk(invalid, state)
            with self.assertRaises(ValueError):
                model.step(x[:, :1], state)
            with self.assertRaises(ValueError):
                model(x, lengths=torch.tensor([FRAMES, FRAMES]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="write a public numeric-only JSON receipt")
    args = parser.parse_args()
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CandidateEncoderNumericTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema": "candidate-encoder-numeric-contract-v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "elapsed_seconds": time.monotonic() - start,
        "seed": SEED, "feature_seed": SEED + 1, "state_seed": SEED + 2,
        "batch": BATCH, "frames": FRAMES, "chunks": list(CHUNKS),
        "specs": [spec.to_dict() for spec in SPECS],
        "torch_version": torch.__version__, "python_version": sys.version.split()[0],
        "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "tolerance": {"atol": ATOL, "rtol": RTOL, "gradient_atol": GRAD_ATOL, "gradient_rtol": GRAD_RTOL},
        "observations": OBSERVATIONS,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "research/candidate_encoders/contract.py", "research/candidate_encoders/torch_models.py",
            "training/model.py", "tests/test_candidate_encoder_numeric.py")},
        "scope": {"audio_used": False, "pretrained_weights_used": False,
                  "optimizer_updates": 0, "training_admitted": False,
                  "product_qualification": False, "ssc305_evidence": False},
        "github": {key: os.environ.get(key) for key in (
            "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB")},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
