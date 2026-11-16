"""Stdlib + host C compiler; every graph, PCM value and weight is invented.

No ONNX/model/generated-weight file, ML runtime or real audio is opened.
The test-only C wrappers include the unchanged baseline or the candidate TU;
the wrappers expose sizeof/guard inspection, not a production API change.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BASELINE_SHA = 'c45821d94887d82a6b9f22375875210aacd9a80f6653a520f6761062bf2aee40'
HEADER_SHA = '55ffd7edffd06744f9146088c4431635cd57498195cb05019df4e0d476e6cef9'
CANDIDATE = HERE / 'fixed_frontend_mel.c'
CC = shutil.which('cc')
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
         '-fno-fast-math', '-ffp-contract=off']
NONE = 0xffffffff
CONV, POOL = 13, 14
PARAMETERS = (1, 1, 512, 1, 160, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)


def baseline_directory():
    # Production repository path first; the explicit scratch preparation path
    # is the only fallback. Neither path searches for model/artifact files.
    for directory in (HERE.parent / 'english_final_embedding_frontend',
                      HERE.parent / 'kws-pipeline/research/english_final_embedding_frontend'):
        if (directory / 'fixed_frontend.c').is_file():
            return directory
    raise FileNotFoundError('the frozen baseline C/header directory is required')


BASELINE = baseline_directory()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tensor(C.Structure):
    _fields_ = [('rank', C.c_uint32), ('dims', C.c_uint32 * 6),
                ('count', C.c_uint32), ('storage', C.c_uint32), ('offset', C.c_uint32)]


class Node(C.Structure):
    _fields_ = [('op', C.c_uint32), ('input_count', C.c_uint32),
                ('inputs', C.c_uint32 * 8), ('output', C.c_uint32),
                ('p', C.c_int32 * 16), ('alpha', C.c_float)]


class Graph(C.Structure):
    _fields_ = [('tensors', C.POINTER(Tensor)), ('tensor_count', C.c_uint32),
                ('nodes', C.POINTER(Node)), ('node_count', C.c_uint32),
                ('input', C.c_uint32), ('output', C.c_uint32),
                ('constants', C.POINTER(C.c_float)), ('constant_count', C.c_uint32),
                ('workspace_floats', C.c_uint32)]


def invented_values(count, seed, divisor=1):
    result = []
    for _ in range(count):
        seed = (1664525 * seed + 1013904223) & 0xffffffff
        result.append((((seed >> 9) & 65535) - 32768) / divisor)
    return result


def tensor(shape, storage, offset=0):
    return Tensor(len(shape), (C.c_uint32 * 6)(*shape), math.prod(shape), storage, offset)


def clone(value):
    return type(value).from_buffer_copy(bytes(value))


class Fixture:
    """One caller-owned invented Conv/Pool graph, with guarded output storage."""
    def __init__(self, *, shape=(1, 1, 16000), weight_shape=(257, 1, 512),
                 parameters=PARAMETERS, values=None, weights=None, bias=None, pool=False):
        shape, parameters = tuple(shape), tuple(parameters)
        if len(parameters) != 16:
            raise ValueError('16 normalized spatial parameters required')
        output_channels = shape[1] if pool else weight_shape[0]
        output_shape = [shape[0], output_channels]
        for axis in range(len(shape) - 2):
            size = shape[axis + 2] + parameters[8 + axis] + parameters[10 + axis]
            size -= parameters[6 + axis] * (parameters[2 + axis] - 1) + 1
            if parameters[12]:
                size += parameters[4 + axis] - 1
            output_shape.append(size // parameters[4 + axis] + 1)
        self.output_count = math.prod(output_shape)
        self.output_offset = 4
        self.values = (C.c_float * math.prod(shape))(*(values if values is not None else
            invented_values(math.prod(shape), 0x42424242)))
        items = [tensor(shape, 0)]
        constants = []
        inputs = [0]
        if not pool:
            constants = list(weights if weights is not None else
                             invented_values(math.prod(weight_shape), 0x1234abcd, 32768))
            if len(constants) != math.prod(weight_shape):
                raise ValueError('invented weight shape/data mismatch')
            items.append(tensor(weight_shape, 1))
            inputs.append(1)
            if bias is not None:
                if len(bias) != output_channels:
                    raise ValueError('invented bias shape/data mismatch')
                items.append(tensor((output_channels,), 1, len(constants)))
                inputs.append(2)
                constants.extend(bias)
        self.output_id = len(items)
        items.append(tensor(output_shape, 2, self.output_offset))
        self.tensors = (Tensor * len(items))(*items)
        self.nodes = (Node * 1)(Node(POOL if pool else CONV, len(inputs),
            (C.c_uint32 * 8)(*(inputs + [NONE] * (8 - len(inputs)))), self.output_id,
            (C.c_int32 * 16)(*parameters), 0.0))
        self.constants = (C.c_float * len(constants))(*constants)
        self.graph = Graph(self.tensors, len(items), self.nodes, 1, 0, self.output_id,
                           self.constants, len(constants), self.output_count + 8)
        self.bias = None if bias is None else C.cast(C.byref(self.constants,
            math.prod(weight_shape) * C.sizeof(C.c_float)), C.POINTER(C.c_float))

    def guard(self, library):
        w = None if self.nodes[0].op == POOL else C.byref(self.tensors[1])
        return library.fixture_guard(C.byref(self.tensors[0]), w,
            C.byref(self.tensors[self.output_id]), C.byref(self.nodes[0]), self.bias)

    def run(self, library, *, shortage=False, callback_result=0, trace=False):
        work = (C.c_float * self.graph.workspace_floats)(
            *([1234567.0] * self.graph.workspace_floats))
        result = C.POINTER(C.c_float)()
        snapshots = []
        callback_type = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32,
                                   C.POINTER(Tensor), C.POINTER(C.c_float))

        def callback(_context, index, output_tensor, values):
            count = output_tensor.contents.count
            snapshots.append((index, count, C.string_at(values, count * 4)))
            return callback_result

        callback_object = callback_type(callback) if trace else None
        status = library.ef_graph_run(C.byref(self.graph), self.values, work,
            self.graph.workspace_floats - int(shortage), C.byref(result),
            C.cast(callback_object, C.c_void_p) if trace else None, None)
        raw = C.string_at(C.addressof(work) + self.output_offset * 4, self.output_count * 4)
        guards = bytes(work)[:16] + bytes(work)[-16:]
        return status, raw, guards, snapshots, bool(result)


def compile_library(directory, source, candidate):
    wrapper = directory / ('candidate.c' if candidate else 'baseline.c')
    text = '#include ' + json.dumps(str(source)) + '\n'
    text += '''
size_t fixture_tensor_size(void) { return sizeof(ef_tensor); }
size_t fixture_node_size(void) { return sizeof(ef_node); }
size_t fixture_graph_size(void) { return sizeof(ef_graph); }
size_t fixture_frontend_size(void) { return sizeof(ef_frontend); }
'''
    if candidate:
        text += '''
int fixture_guard(const ef_tensor *x, const ef_tensor *w, const ef_tensor *y,
                  const ef_node *n, const float *bias) {
    return fixed_mel_conv1d_matches(x, w, y, n, bias);
}
'''
    wrapper.write_text(text)
    output = wrapper.with_suffix('.so')
    subprocess.run([CC, *FLAGS, '-shared', '-fPIC', '-I', str(BASELINE), str(wrapper),
                    '-lm', '-o', str(output)], check=True, capture_output=True, text=True, timeout=60)
    library = C.CDLL(str(output))
    library.ef_graph_run.argtypes = [C.POINTER(Graph), C.POINTER(C.c_float), C.POINTER(C.c_float),
        C.c_size_t, C.POINTER(C.POINTER(C.c_float)), C.c_void_p, C.c_void_p]
    library.ef_graph_run.restype = C.c_int
    for name in ['tensor', 'node', 'graph', 'frontend']:
        getattr(library, 'fixture_' + name + '_size').restype = C.c_size_t
    if candidate:
        library.fixture_guard.argtypes = [C.POINTER(Tensor), C.POINTER(Tensor), C.POINTER(Tensor),
                                          C.POINTER(Node), C.POINTER(C.c_float)]
        library.fixture_guard.restype = C.c_int
    return library


class SourceIdentityTests(unittest.TestCase):
    def test_only_marked_helper_and_dispatch_added_to_exact_baseline(self):
        self.assertEqual(digest(BASELINE / 'fixed_frontend.c'), BASELINE_SHA)
        self.assertEqual(digest(BASELINE / 'fixed_frontend.h'), HEADER_SHA)
        candidate = CANDIDATE.read_bytes()
        original = candidate
        blocks = [br'/\* BEGIN FIXED MEL CONV1D SPECIALIZATION\n.*?/\* END FIXED MEL CONV1D SPECIALIZATION \*/\n\n',
                  br'    /\* BEGIN FIXED MEL CONV1D DISPATCH \*/\n.*?    /\* END FIXED MEL CONV1D DISPATCH \*/\n']
        for pattern in blocks:
            original, count = re.subn(pattern, b'', original, flags=re.S)
            self.assertEqual(count, 1)
        self.assertEqual(original, (BASELINE / 'fixed_frontend.c').read_bytes())
        self.assertEqual(hashlib.sha256(original).hexdigest(), BASELINE_SHA)
        self.assertNotEqual(candidate, original)


class CompiledSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CC:
            raise unittest.SkipTest('C compiler unavailable; invented numerical checks not performed')
        if digest(BASELINE / 'fixed_frontend.c') != BASELINE_SHA or digest(BASELINE / 'fixed_frontend.h') != HEADER_SHA:
            raise ValueError('frozen baseline C/header identity mismatch')
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.original = compile_library(Path(cls.temporary.name), BASELINE / 'fixed_frontend.c', False)
        cls.candidate = compile_library(Path(cls.temporary.name), CANDIDATE, True)

    def compare(self, fixture, *, expected_guard, expected_status=0, **kwargs):
        self.assertEqual(fixture.guard(self.candidate), expected_guard)
        original, candidate = fixture.run(self.original, **kwargs), fixture.run(self.candidate, **kwargs)
        self.assertEqual(original[0], expected_status)
        self.assertEqual(candidate, original, 'status/output bits/guards/callback/result pointer changed')
        self.assertEqual(candidate[2], struct.pack('<8f', *([1234567.0] * 8)))
        return candidate

    def test_abi_and_state_size_unchanged(self):
        for name, kind in [('tensor', Tensor), ('node', Node), ('graph', Graph)]:
            self.assertEqual(getattr(self.original, 'fixture_' + name + '_size')(), C.sizeof(kind))
            self.assertEqual(getattr(self.candidate, 'fixture_' + name + '_size')(), C.sizeof(kind))
        self.assertEqual(self.candidate.fixture_frontend_size(), self.original.fixture_frontend_size())

    def test_exact_lcg_pcm_and_invented_weights_bitexact(self):
        self.compare(Fixture(), expected_guard=1)

    def test_order_sensitive_cancellation_still_returns_one(self):
        # Scalar (((2^24 + 1) - 2^24) + 1) loses the first 1. A reassociated
        # reduction or multiple accumulators need not produce this exact result.
        weights = [16777216.0, 1.0, -16777216.0, 1.0] * (257 * 128)
        result = self.compare(Fixture(values=[1.0] * 16000, weights=weights), expected_guard=1)
        self.assertEqual(result[1], struct.pack('<f', 1.0) * (257 * 97))

    def test_first_last_taps_and_unconsumed_tail(self):
        values = [0.0] * 16000
        for index, value in [(0, 3.0), (511, -7.0), (15360, 11.0), (15871, -13.0), (15999, 32000.0)]:
            values[index] = value
        weights = [0.0] * (257 * 512)
        for channel in range(257):
            weights[channel * 512] = 1.0
            weights[channel * 512 + 511] = 2.0
        result = self.compare(Fixture(values=values, weights=weights), expected_guard=1)
        floats = struct.unpack('<%df' % (257 * 97), result[1])
        self.assertEqual(floats[0], -11.0)
        self.assertEqual(floats[96], -15.0)
        values[-1] = -32000.0
        changed_tail = self.compare(Fixture(values=values, weights=weights), expected_guard=1)
        self.assertEqual(changed_tail[1], result[1])

    def test_guard_rejects_each_parameter_rank_and_dimension_deviation(self):
        fixture = Fixture()
        x, w, y, node = fixture.tensors[0], fixture.tensors[1], fixture.tensors[fixture.output_id], fixture.nodes[0]
        def guard(a, b, c, n):
            return self.candidate.fixture_guard(C.byref(a), C.byref(b), C.byref(c), C.byref(n), None)
        for index in range(16):
            with self.subTest(parameter=index):
                changed = clone(node); changed.p[index] += 1
                self.assertEqual(guard(x, w, y, changed), 0)
        for which, value in enumerate((x, w, y)):
            for field in ['rank', 0, 1, 2]:
                with self.subTest(tensor=which, dimension=field):
                    changed = [clone(x), clone(w), clone(y)]
                    if field == 'rank':
                        changed[which].rank = 4
                    else:
                        changed[which].dims[field] += 1
                    self.assertEqual(guard(*changed, node), 0)
        changed = clone(node); changed.op = POOL
        self.assertEqual(guard(x, w, y, changed), 0)
        changed = clone(node); changed.input_count = 3
        self.assertEqual(guard(x, w, y, changed), 0)
        bias = (C.c_float * 1)(0.0)
        self.assertEqual(self.candidate.fixture_guard(C.byref(x), C.byref(w), C.byref(y), C.byref(node), bias), 0)

    def test_bias_preserves_generic_fallback(self):
        self.compare(Fixture(bias=[(i - 128) / 16 for i in range(257)]), expected_guard=0)

    def test_rank_four_preserves_generic_fallback(self):
        p = list(PARAMETERS); p[0] = 2
        self.compare(Fixture(shape=(1, 1, 16000, 1), weight_shape=(257, 1, 512, 1), parameters=p), expected_guard=0)

    def test_nonzero_padding_with_same_output_shape_falls_back(self):
        p = list(PARAMETERS); p[8] = 1
        fixture = Fixture(parameters=p)
        self.assertEqual(fixture.output_count, 257 * 97)
        self.compare(fixture, expected_guard=0)

    def test_stride_dilation_and_kernel_fall_back(self):
        for change in ['stride', 'dilation', 'kernel']:
            with self.subTest(change=change):
                p = list(PARAMETERS); shape = (257, 1, 512)
                if change == 'stride':
                    p[4] = 161  # Still 97 output frames: guard must inspect stride.
                elif change == 'dilation':
                    p[6] = 2
                else:
                    p[2] = 511; shape = (257, 1, 511)
                self.compare(Fixture(weight_shape=shape, parameters=p), expected_guard=0)

    def test_channel_group_and_batch_fall_back(self):
        self.compare(Fixture(shape=(1, 2, 16000), weight_shape=(257, 2, 512)), expected_guard=0)
        p = list(PARAMETERS); p[1] = 2
        self.compare(Fixture(shape=(1, 2, 16000), weight_shape=(258, 1, 512), parameters=p), expected_guard=0)
        self.compare(Fixture(shape=(2, 1, 16000)), expected_guard=0)

    def test_optional_absent_third_input_falls_back(self):
        fixture = Fixture()
        fixture.nodes[0].input_count = 3
        fixture.nodes[0].inputs[2] = NONE
        self.compare(fixture, expected_guard=0)

    def test_maxpool_stays_on_generic_path(self):
        self.compare(Fixture(pool=True), expected_guard=0)

    def test_nonfinite_and_workspace_errors_preserved(self):
        for target in ['input', 'weight', 'workspace']:
            with self.subTest(target=target):
                fixture = Fixture()
                if target == 'input':
                    fixture.values[17] = math.nan
                elif target == 'weight':
                    fixture.constants[13] = math.inf
                self.compare(fixture, expected_guard=1, expected_status=-2 if target == 'workspace' else -3,
                             shortage=target == 'workspace')

    def test_node_trace_and_callback_rejection_preserved(self):
        fixture = Fixture()
        result = self.compare(fixture, expected_guard=1, trace=True)
        self.assertEqual(result[3], [(0, 257 * 97, result[1])])
        self.compare(fixture, expected_guard=1, expected_status=-5, trace=True, callback_result=1)


if __name__ == '__main__':
    unittest.main()
