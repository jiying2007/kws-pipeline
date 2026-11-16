"""Independent, invented-tensor checks of the static C frontend subset.

Uses only Python's standard library and a C compiler. No actual model, model
runtime, network, training, or numerical-parity claim is involved. Expectations
are hand calculated, independent of the exporter and C indexing implementation.

Semantics checked against the official schemas applicable to opset 13:
https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11
https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12
https://onnx.ai/onnx/operators/onnx__ReduceMax.html#reducemax-13
https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-13
https://onnx.ai/onnx/repo-docs/Broadcasting.html
"""
from __future__ import annotations

import ctypes as ct
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "independent_frontend_export", HERE / "export_fixed_frontend.py")
exporter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = exporter
_spec.loader.exec_module(exporter)


class Tensor(ct.Structure):
    _fields_ = [("rank", ct.c_uint32), ("dims", ct.c_uint32 * 6),
                ("count", ct.c_uint32), ("storage", ct.c_uint32),
                ("offset", ct.c_uint32)]


class Node(ct.Structure):
    _fields_ = [("op", ct.c_uint32), ("input_count", ct.c_uint32),
                ("inputs", ct.c_uint32 * 8), ("output", ct.c_uint32),
                ("p", ct.c_int32 * 16), ("alpha", ct.c_float)]


class Graph(ct.Structure):
    _fields_ = [("tensors", ct.POINTER(Tensor)), ("tensor_count", ct.c_uint32),
                ("nodes", ct.POINTER(Node)), ("node_count", ct.c_uint32),
                ("input", ct.c_uint32), ("output", ct.c_uint32),
                ("constants", ct.POINTER(ct.c_float)),
                ("constant_count", ct.c_uint32), ("workspace_floats", ct.c_uint32)]


def constant(shape, values, dtype="float32"):
    return {"shape": list(shape), "dtype": dtype, "data": list(values)}


def node(op, inputs, output="y", **attrs):
    return {"op": op, "inputs": list(inputs), "output": output, "attrs": attrs}


def graph(shape, nodes, constants=None, output="y"):
    return {"input": {"name": "x", "shape": list(shape), "dtype": "float32"},
            "initializers": constants or {}, "nodes": nodes, "output": output}


def fixtures():
    cases = {}

    def add(name, spec, values, outshape, expected):
        cases[name] = (spec, list(values), tuple(outshape), list(expected))

    # Cross-correlation, not a reversed convolution: taps at -1,1,3 and 1,3,5.
    add("conv1_dilated_asymmetric", graph([1, 1, 5], [
        node("Conv", ["x", "w", "b"], dilations=[2], strides=[2], pads=[1, 2])
    ], {"w": constant([1, 1, 3], [1, 10, 100]), "b": constant([1], [.5])}),
        [1, 2, 3, 4, 5], [1, 1, 2], [420.5, 42.5])

    # Two batches and two channels per group catch both group and batch offsets.
    add("conv1_groups_batches", graph([2, 4, 2], [
        node("Conv", ["x", "w", "b"], group=2)
    ], {"w": constant([4, 2, 1], [1, 10, -1, 2, 3, -1, 0, 4]),
        "b": constant([4], [1, -2, 3, -4])}),
        list(range(1, 9)) + list(range(11, 19)), [2, 4, 2],
        [32, 43, 3, 4, 11, 13, 24, 28, 142, 153, 13, 14, 31, 33, 64, 68])

    add("conv2_depthwise_dilated", graph([1, 2, 3, 3], [
        node("Conv", ["x", "w"], group=2, dilations=[2, 1],
             strides=[1, 2], pads=[1, 0, 0, 1])
    ], {"w": constant([2, 1, 2, 2], [1, 2, 3, 4, -1, 1, 0, 2])}),
        list(range(1, 10)) + list(range(101, 110)), [1, 2, 2, 2],
        [32, 18, 58, 30, 210, 0, 217, -103])

    for mode, expected in (("SAME_UPPER", [321, 43]),
                           ("SAME_LOWER", [210, 432]), ("VALID", [321])):
        add("conv1_" + mode.lower(), graph([1, 1, 4], [
            node("Conv", ["x", "w"], auto_pad=mode, strides=[2])
        ], {"w": constant([1, 1, 3], [1, 10, 100])}),
            [1, 2, 3, 4], [1, 1, len(expected)], expected)

    for mode, expected in (("SAME_UPPER", [24, 22]), ("SAME_LOWER", [14, 30])):
        add("conv2_" + mode.lower(), graph([1, 1, 2, 4], [
            node("Conv", ["x", "w"], auto_pad=mode, strides=[2, 2])
        ], {"w": constant([1, 1, 3, 3], [1] * 9)}),
            list(range(1, 9)), [1, 1, 1, 2], expected)

    # Padding must be excluded: zero padding would incorrectly win every window.
    add("pool2_negative_asymmetric", graph([1, 1, 2, 3], [
        node("MaxPool", ["x"], kernel_shape=[2, 2], strides=[1, 2], pads=[1, 0, 0, 1])
    ]), [-9, -8, -7, -6, -5, -4], [1, 1, 2, 2], [-8, -7, -5, -4])

    for ceil, expected in ((0, [-2]), (1, [-2, -4])):
        add("pool1_ceil_" + str(ceil), graph([1, 1, 4], [
            node("MaxPool", ["x"], kernel_shape=[3], strides=[2], ceil_mode=ceil)
        ]), [-5, -2, -9, -4], [1, 1, len(expected)], expected)

    # The nominal last ceil window starts at index 4, wholly in right padding.
    add("pool1_ceil_drop_right", graph([1, 1, 4], [
        node("MaxPool", ["x"], kernel_shape=[3], strides=[2], pads=[0, 2], ceil_mode=1)
    ]), [-1, -2, -3, -4], [1, 1, 2], [-1, -3])
    add("pool1_ceil_negative_numerator", graph([1, 1, 2], [
        node("MaxPool", ["x"], kernel_shape=[3], strides=[2], ceil_mode=1)
    ]), [-7, -3], [1, 1, 1], [-3])
    add("pool1_dilation", graph([1, 1, 5], [
        node("MaxPool", ["x"], kernel_shape=[2], dilations=[2], strides=[2])
    ]), [-8, -1, -6, -2, -4], [1, 1, 2], [-6, -4])

    for mode, expected in (("SAME_UPPER", [-1, -1]), ("SAME_LOWER", [-4, -1])):
        add("pool1_" + mode.lower(), graph([1, 1, 4], [
            node("MaxPool", ["x"], kernel_shape=[3], strides=[2], auto_pad=mode)
        ]), [-7, -4, -1, -9], [1, 1, 2], expected)

    add("broadcast_axes", graph([2, 1, 3], [node("Add", ["x", "b"])],
        {"b": constant([1, 4, 1], [10, 20, 30, 40])}),
        [1, 2, 3, 4, 5, 6], [2, 4, 3],
        [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
         14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46])
    add("broadcast_variadic_scalar", graph([2, 1], [node("Max", ["x", "b", "c"])],
        {"b": constant([3], [-2, 0, 6]), "c": constant([], [1])}),
        [-3, 4], [2, 3], [1, 1, 6, 4, 4, 6])
    add("broadcast_sub_order", graph([3], [node("Sub", ["b", "x"])],
        {"b": constant([2, 1], [10, 20])}),
        [1, 2, 3], [2, 3], [9, 8, 7, 19, 18, 17])
    add("broadcast_scalar_input", graph([], [node("Div", ["b", "x"])],
        {"b": constant([2, 2], [2, 4, 6, 8])}), [2], [2, 2], [1, 2, 3, 4])

    add("transpose_explicit", graph([2, 3, 2], [
        node("Transpose", ["x"], perm=[1, 2, 0])]), range(12), [3, 2, 2],
        [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    add("transpose_default", graph([2, 3, 2], [node("Transpose", ["x"])]),
        range(12), [2, 3, 2], [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    for keep, shape in ((0, [3]), (1, [1, 3, 1])):
        add("reduce_axes_" + str(keep), graph([2, 3, 2], [
            node("ReduceMax", ["x"], axes=[0, -1], keepdims=keep)]),
            range(-12, 0), shape, [-5, -3, -1])
    add("reduce_scalar_output", graph([2, 3, 2], [
        node("ReduceMax", ["x"], keepdims=0)]), range(-12, 0), [], [-1])
    add("reduce_empty_axes", graph([2, 3, 2], [
        node("ReduceMax", ["x"], axes=[], keepdims=1)]), range(-12, 0), [1, 1, 1], [-1])
    add("reduce_scalar_input", graph([], [node("ReduceMax", ["x"], keepdims=0)]),
        [-9], [], [-9])

    # a must survive two aliases and an intervening branch's temporary reuse.
    add("alias_branch_liveness", graph([2, 3], [
        node("Add", ["x", "ten"], "a"),
        node("Reshape", ["a", "shape32"], "r"),
        node("Unsqueeze", ["r", "axes"], "u"),
        node("Mul", ["a", "two"], "b"),
        node("Add", ["b", "ten"], "c"),
        node("Add", ["b", "c"], "d"),
        node("Reshape", ["u", "shape23"], "v"),
        node("Sub", ["d", "v"])
    ], {"ten": constant([], [10]), "two": constant([], [2]),
        "shape32": constant([2], [3, 2], "int64"),
        "shape23": constant([2], [2, 3], "int64"),
        "axes": constant([2], [0, -1], "int64")}),
        range(1, 7), [2, 3], [43, 46, 49, 52, 55, 58])
    add("alias_graph_output", graph([2, 2], [
        node("Add", ["x", "ten"], "a"),
        node("Reshape", ["a", "shape"], "y"),
        node("Mul", ["x", "two"], "b"),
        node("Add", ["b", "two"], "unused")
    ], {"ten": constant([], [10]), "two": constant([], [2]),
        "shape": constant([1], [-1], "int64")}),
        [1, 2, 3, 4], [4], [11, 12, 13, 14])
    return cases


class IndependentOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if not compiler:
            raise unittest.SkipTest("C compiler required for invented-tensor source checks")
        cls.directory = tempfile.TemporaryDirectory(prefix="independent-frontend-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.cases = fixtures()
        cls.lowered = {key: exporter.lower_graph(case[0]) for key, case in cls.cases.items()}
        path = Path(cls.directory.name)
        generated = path / "invented.c"
        generated.write_text("\n".join(exporter.emit_c(value, "ind_" + key)
                                       for key, value in cls.lowered.items()))
        library = path / "invented.so"
        subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
                        "-I", str(HERE), str(HERE / "fixed_frontend.c"),
                        str(generated), "-lm", "-o", str(library)],
                       check=True, capture_output=True, text=True)
        cls.library = ct.CDLL(str(library))
        cls.library.ef_graph_validate.argtypes = [ct.POINTER(Graph)]
        cls.library.ef_graph_validate.restype = ct.c_int
        cls.library.ef_graph_run.argtypes = [ct.POINTER(Graph), ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.POINTER(ct.c_float)),
            ct.c_void_p, ct.c_void_p]
        cls.library.ef_graph_run.restype = ct.c_int

    def check_case(self, name):
        _, values, expected_shape, expected_values = self.cases[name]
        runtime = Graph.in_dll(self.library, "ind_" + name)
        output_tensor = runtime.tensors[runtime.output]
        actual_shape = tuple(output_tensor.dims[:output_tensor.rank])
        self.assertEqual(actual_shape, expected_shape)
        self.assertEqual(output_tensor.count, len(expected_values))
        self.assertEqual(self.library.ef_graph_validate(ct.byref(runtime)), 0)
        inputs = (ct.c_float * len(values))(*values)
        # NaN poison catches unwritten values; boundary sentinels catch overruns.
        count = runtime.workspace_floats
        sentinel = 123456.0
        workspace = (ct.c_float * (count + 8))(*([sentinel] * 4 +
                      [float("nan")] * count + [sentinel] * 4))
        pointer = ct.cast(ct.byref(workspace, 4 * ct.sizeof(ct.c_float)), ct.POINTER(ct.c_float))
        output = ct.POINTER(ct.c_float)()
        status = self.library.ef_graph_run(ct.byref(runtime), inputs, pointer,
                                          count, ct.byref(output), None, None)
        self.assertEqual(status, 0, name)
        self.assertTrue(bool(output))
        self.assertEqual(list(output[:len(expected_values)]), expected_values, name)
        self.assertEqual(list(inputs), values, "input changed: " + name)
        self.assertEqual(list(workspace[:4]) + list(workspace[count + 4:]), [sentinel] * 8)

    def check_prefix(self, prefix):
        names = [name for name in self.cases if name.startswith(prefix)]
        self.assertTrue(names)
        for name in names:
            with self.subTest(case=name):
                self.check_case(name)

    def test_conv1d_group_dilation_padding(self):
        self.check_prefix("conv1_")

    def test_conv2d_depthwise_dilation_padding(self):
        self.check_prefix("conv2_")

    def test_maxpool_padding_ceil_dilation(self):
        self.check_prefix("pool")

    def test_multidirectional_and_scalar_broadcast(self):
        self.check_prefix("broadcast_")

    def test_transpose_coordinates(self):
        self.check_prefix("transpose_")

    def test_reduce_axes_and_scalar_shape(self):
        self.check_prefix("reduce_")

    def test_alias_lifetimes_across_branches_and_graph_output(self):
        self.check_prefix("alias_")


class IndependentRejections(unittest.TestCase):
    def test_rejects_shapes_without_silent_reinterpretation(self):
        cases = [
            ("BROADCAST_SHAPE", graph([2, 3], [node("Add", ["x", "b"])],
                {"b": constant([2], [1, 2])})),
            ("TRANSPOSE_PERM", graph([2, 3], [node("Transpose", ["x"], perm=[0, 0])])),
            ("REDUCE_DUPLICATE_AXES", graph([2, 3], [
                node("ReduceMax", ["x"], axes=[1, -1])])),
            ("CONV_WEIGHT_GROUP_SHAPE", graph([1, 3, 4], [
                node("Conv", ["x", "w"], group=2)],
                {"w": constant([2, 1, 1], [1, 2])})),
        ]
        for message, spec in cases:
            with self.subTest(reason=message), self.assertRaisesRegex(ValueError, message):
                exporter.lower_graph(spec)

    def test_rejects_unimplemented_or_empty_pool_windows(self):
        cases = [
            ("POOL_AUTO_PAD_WITH_CEIL_UNSUPPORTED", {"kernel_shape": [3],
                "auto_pad": "SAME_UPPER", "ceil_mode": 1}, [1, 1, 4]),
            ("POOL_INDICES_STORAGE_ORDER_UNSUPPORTED", {"kernel_shape": [1],
                "storage_order": 1}, [1, 1, 4]),
            ("EMPTY_POOL_WINDOW", {"kernel_shape": [1], "pads": [1, 0]}, [1, 1, 1]),
            # Geometrically overlapping but dilation lands entirely outside x.
            ("EMPTY_POOL_WINDOW", {"kernel_shape": [2], "dilations": [2],
                "pads": [1, 1]}, [1, 1, 1]),
        ]
        for message, attrs, shape in cases:
            with self.subTest(reason=message, attrs=attrs), self.assertRaisesRegex(ValueError, message):
                exporter.lower_graph(graph(shape, [node("MaxPool", ["x"], **attrs)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
