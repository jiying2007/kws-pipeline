#!/usr/bin/env python3
"""AST and fake-shape checks only: never import Torch or build/run a model.

These fixtures inspect source and execute only the extracted storage-tail helper
on objects containing dimension tuples. They cannot establish numeric equality,
Torch compatibility, gradients, accuracy, target resource use or qualification.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "candidate_encoders" / "torch_models.py"


def definition(tree, name):
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one definition of {name}")
    return matches[0]


def method(tree, class_name, method_name):
    cls = definition(tree, class_name)
    return definition(cls, method_name)


def calls(tree, expression):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == expression]


def keywords(call):
    return {item.arg: ast.unparse(item.value) for item in call.keywords}


class ShapeOnly:
    """No data values, tensor backend, model, or numeric operations exist here."""

    def __init__(self, shape, *, storage=None, operations=()):
        self.shape = shape
        self.storage = object() if storage is None else storage
        self.operations = operations

    def __getitem__(self, slices):
        if len(slices) != len(self.shape) or not all(
            isinstance(index, slice) for index in slices
        ):
            raise AssertionError("fixture supports only shape-preserving slices")
        shape = tuple(len(range(*index.indices(size)))
                      for index, size in zip(slices, self.shape))
        return ShapeOnly(shape, storage=self.storage,
                         operations=self.operations + ("slice",))

    def clone(self):
        return ShapeOnly(self.shape, operations=self.operations + ("clone",))

    def contiguous(self):
        return ShapeOnly(self.shape, storage=self.storage,
                         operations=self.operations + ("contiguous",))


class CandidateEncoderSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOURCE))

    def test_source_compiles_without_importing_or_executing_it(self):
        compile(self.tree, str(SOURCE), "exec")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("research.candidate_encoders.torch_models", sys.modules)

    def test_module_scope_defines_only_lazy_functions_and_contract_import(self):
        self.assertEqual([type(node).__name__ for node in self.tree.body],
                         ["Expr", "ImportFrom", "ImportFrom", "FunctionDef", "FunctionDef"])
        self.assertIsInstance(self.tree.body[0].value, ast.Constant)
        self.assertEqual(self.tree.body[1].module, "__future__")
        self.assertEqual(self.tree.body[2].module, "contract")
        self.assertEqual(self.tree.body[2].level, 1)
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                self.assertEqual(node.decorator_list, [])
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None:
                        self.assertIsInstance(default, ast.Constant)
        factory = definition(self.tree, "build_encoder")
        self.assertEqual([node.name for node in factory.body if isinstance(node, ast.ClassDef)],
                         ["StreamingEncoder", "RNNReferenceAdapter", "CausalResidualBlock",
                          "CausalDepthwiseEncoder"])

    def test_only_explicit_factory_imports_torch_no_product_imports_or_io(self):
        imports = [node for node in ast.walk(self.tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual([ast.unparse(node) for node in imports], [
            "from __future__ import annotations", "from .contract import EncoderSpec",
            "import torch", "from torch import nn",
        ])
        factory = definition(self.tree, "build_encoder")
        self.assertTrue(all(node in list(ast.walk(factory)) for node in imports[2:]))
        forbidden = ("torch.load", "torch.save", "torch.hub", "torch.manual_seed",
                     "torch.set_num_threads", "torch.set_num_interop_threads", "open",
                     "eval", "exec", "compile", "__import__")
        for call in [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]:
            self.assertNotIn(ast.unparse(call.func), forbidden)

    def test_reference_is_required_before_any_torch_import(self):
        factory = definition(self.tree, "build_encoder")
        torch_index = next(i for i, node in enumerate(factory.body)
                           if isinstance(node, ast.Import))
        before_import = ast.unparse(ast.Module(factory.body[:torch_index], type_ignores=[]))
        self.assertIn("type(spec) is not EncoderSpec", before_import)
        self.assertIn("spec.architecture == 'rnn64_control' and rnn_model is None", before_import)
        self.assertIn("spec.architecture == 'causal_ds_tcn_ctc' and rnn_model is not None",
                      before_import)
        self.assertEqual(len(factory.args.kwonlyargs), 1)
        self.assertEqual(factory.args.kwonlyargs[0].arg, "rnn_model")

    def test_c0_delegates_without_copying_reference_equations(self):
        step = method(self.tree, "RNNReferenceAdapter", "step")
        delegate = calls(step, "self.rnn_model.step")
        self.assertEqual(len(delegate), 1)
        self.assertEqual([ast.unparse(arg) for arg in delegate[0].args], ["x", "state[0]"])
        self.assertFalse(calls(self.tree, "torch.tanh"))
        constructor = ast.unparse(method(self.tree, "RNNReferenceAdapter", "__init__"))
        self.assertIn("actual_parameters != expected_parameters", constructor)
        self.assertIn("('in_proj', spec.feature_dim, spec.hidden_dim, True)", constructor)
        self.assertIn("('rec_proj', spec.hidden_dim, spec.hidden_dim, False)", constructor)
        self.assertIn("('out_proj', spec.hidden_dim, spec.vocab_size, True)", constructor)
        self.assertEqual(len(calls(step, "self._check_tensor")), 2)

    def test_c1_kernel_geometry_is_causal_depthwise_then_pointwise(self):
        constructor = method(self.tree, "CausalResidualBlock", "__init__")
        convolutions = calls(constructor, "nn.Conv1d")
        self.assertEqual(len(convolutions), 2)
        depthwise, pointwise = convolutions
        self.assertEqual([ast.unparse(arg) for arg in depthwise.args],
                         ["spec.hidden_dim", "spec.hidden_dim", "kernel_size"])
        self.assertEqual([ast.unparse(arg) for arg in pointwise.args],
                         ["spec.hidden_dim", "spec.hidden_dim", "1"])
        self.assertEqual(keywords(depthwise), dict(stride="1", padding="0", dilation="dilation",
                         groups="spec.hidden_dim", bias="True", device="'cpu'",
                         dtype="torch.float32"))
        self.assertEqual(keywords(pointwise), dict(stride="1", padding="0", dilation="1",
                         groups="1", bias="True", device="'cpu'", dtype="torch.float32"))
        self.assertIn("self.history_frames = (kernel_size - 1) * dilation", ast.unparse(constructor))

    def test_c1_state_retains_layer_input_and_not_layer_output(self):
        block = method(self.tree, "CausalResidualBlock", "forward_chunk")
        assignments = {ast.unparse(node.targets[0]): ast.unparse(node.value)
                       for node in block.body if isinstance(node, ast.Assign)}
        self.assertEqual(assignments, {
            "joined": "torch.cat((history, layer_inputs), dim=2)",
            "temporal": "torch.relu(self.depthwise(joined))",
            "projected": "torch.relu(self.pointwise(temporal))",
            "outputs": "layer_inputs + projected",
            "next_history": "_retain_history(joined, self.history_frames)",
        })
        self.assertFalse(calls(self.tree, "torch.nn.functional.pad"))

    def test_c1_projections_are_explicit_float32_cpu_with_bias(self):
        constructor = method(self.tree, "CausalDepthwiseEncoder", "__init__")
        projections = calls(constructor, "nn.Linear")
        self.assertEqual(len(projections), 2)
        for projection in projections:
            self.assertEqual(keywords(projection),
                             dict(bias="True", device="'cpu'", dtype="torch.float32"))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ("BatchNorm1d", "Dropout", "LSTM", "GRU", "detach"))

    def test_no_session_state_is_stored_on_modules(self):
        attributes = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and ast.unparse(target.value) == "self":
                        attributes.add(target.attr)
        self.assertEqual(attributes, {"spec", "rnn_model", "depthwise", "pointwise",
                                      "history_frames", "in_proj", "blocks", "out_proj"})
        self.assertFalse(calls(self.tree, "self.register_buffer"))

    def test_reset_allocates_all_layers_with_explicit_dtype_device(self):
        initialize = method(self.tree, "StreamingEncoder", "init_state")
        self.assertEqual([arg.arg for arg in initialize.args.kwonlyargs], ["device", "dtype"])
        self.assertEqual(initialize.args.kw_defaults, [None, None])
        self.assertEqual(len(calls(initialize, "spec.state_shapes")), 1)
        allocation = calls(initialize, "torch.zeros")
        self.assertEqual(len(allocation), 1)
        self.assertEqual(keywords(allocation[0]), dict(device="selected_device", dtype="dtype"))
        reset = method(self.tree, "StreamingEncoder", "reset_state")
        self.assertEqual(len(calls(reset, "self.init_state")), 1)

    def test_full_forward_is_a_reset_and_rejects_lengths(self):
        forward = method(self.tree, "StreamingEncoder", "forward")
        self.assertIsInstance(forward.body[1], ast.If)
        self.assertEqual(ast.unparse(forward.body[1].test), "lengths is not None")
        self.assertIsInstance(forward.body[1].body[0], ast.Raise)
        self.assertEqual(len(calls(forward, "self.init_state")), 1)
        self.assertEqual(len(calls(forward, "self.forward_chunk")), 1)
        self.assertFalse(calls(forward, "torch.zeros"))

    def test_empty_chunks_preserve_state_without_running_layers(self):
        for cls in ("RNNReferenceAdapter", "CausalDepthwiseEncoder"):
            chunk = method(self.tree, cls, "forward_chunk")
            empties = [node for node in chunk.body if isinstance(node, ast.If)]
            self.assertEqual(len(empties), 1)
            empty = empties[0]
            self.assertEqual(ast.unparse(empty.test), "x.shape[1] == 0")
            self.assertEqual(ast.unparse(empty.body[0]),
                             "return (x.new_empty((0, x.shape[0], spec.vocab_size)), state)")
            self.assertEqual(len(calls(chunk, "self.init_state")), 0)

    def test_layout_and_step_chunk_connection_are_explicit(self):
        step = method(self.tree, "StreamingEncoder", "step")
        self.assertEqual(ast.unparse(calls(step, "self.forward_chunk")[0]),
                         "self.forward_chunk(x.unsqueeze(1), state)")
        self.assertEqual(ast.unparse(step.body[-1]), "return (logits[0], next_state)")
        c0_chunk = method(self.tree, "RNNReferenceAdapter", "forward_chunk")
        self.assertEqual(keywords(calls(c0_chunk, "torch.stack")[0]), dict(dim="0"))
        c1_chunk = ast.unparse(method(self.tree, "CausalDepthwiseEncoder", "forward_chunk"))
        self.assertIn("self.out_proj(layer_inputs.transpose(1, 2)).transpose(0, 1)", c1_chunk)

    def test_complete_state_shapes_and_parameter_dtype_device_are_checked(self):
        state = ast.unparse(method(self.tree, "StreamingEncoder", "_check_state"))
        self.assertIn("type(state) is not tuple or len(state) != len(expected_shapes)", state)
        self.assertIn("expected_shapes = spec.state_shapes(x.shape[0])", state)
        parameter_check = ast.unparse(method(self.tree, "StreamingEncoder", "_check_parameters"))
        self.assertIn("dtype != torch.float32", parameter_check)
        self.assertIn("parameter.dtype != dtype or parameter.device != device", parameter_check)

    def test_fake_shapes_have_bounded_independent_cache_storage(self):
        # Execute only this audited helper definition, never the source module,
        # factory, model classes, layer methods, Torch or numerical tensors.
        helper = definition(self.tree, "_retain_history")
        self.assertEqual([ast.unparse(node.func) for node in ast.walk(helper)
                          if isinstance(node, ast.Call)], [
                              "layer_inputs[:, :, -history_frames:].clone().contiguous",
                              "layer_inputs[:, :, :0].clone().contiguous",
                              "layer_inputs[:, :, -history_frames:].clone",
                              "layer_inputs[:, :, :0].clone",
                          ])
        namespace = {"__builtins__": {"int": int}}
        isolated = ast.Module(body=[helper], type_ignores=[])
        exec(compile(isolated, str(SOURCE) + "::_retain_history", "exec"), namespace)
        retain = namespace["_retain_history"]
        for history in (0, 1, 2, 17, 2048):
            for chunk_frames in (1, 2, 31, 10000):
                with self.subTest(history=history, chunk_frames=chunk_frames):
                    joined = ShapeOnly((2, 7, history + chunk_frames))
                    cache = retain(joined, history)
                    self.assertEqual(cache.shape, (2, 7, history))
                    self.assertIsNot(cache.storage, joined.storage)
                    self.assertEqual(cache.operations, ("slice", "clone", "contiguous"))
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
