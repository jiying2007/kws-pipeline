"""Invented tensor/PCM tests only. No pretrained files or ML runtime imports."""
from __future__ import annotations
import ctypes
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('source_frontend_export', HERE / 'export_fixed_frontend.py')
_exporter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _exporter
_spec.loader.exec_module(_exporter)
ASSETS, emit_c, f32 = _exporter.ASSETS, _exporter.emit_c, _exporter.f32
lower_graph, read_pinned_onnx, tensor_proto = _exporter.lower_graph, _exporter.read_pinned_onnx, _exporter.tensor_proto
CC = shutil.which('cc')
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fno-fast-math', '-ffp-contract=off']


def tensor(dims, data, dtype='float32'):
    return {'shape': list(dims), 'dtype': dtype, 'data': list(data)}


def fixture(dims, nodes, constants=None, output='y', output_shape=None):
    spec = {'input': {'name': 'x', 'shape': list(dims), 'dtype': 'float32'},
            'nodes': nodes, 'initializers': constants or {}, 'output': output}
    if output_shape is not None:
        spec['output_shape'] = list(output_shape)
    return spec


def node(op, inputs, output='y', **attrs):
    return {'op': op, 'inputs': inputs, 'output': output, 'attrs': attrs}


def invented_pair():
    """Shapes match the interface; the invented math is deliberately not STFT.

    Mel is 97 strided point samples scaled into32 channels. Embedding is max
    across76 times repeated three times. Tests can compute every value by hand.
    """
    mel = fixture((1, 16000), [node('Unsqueeze', ['x', 'axis'], 'a'),
        node('Conv', ['a', 'w'], 'b', strides=[165]),
        node('Transpose', ['b'], 'c', perm=[0, 2, 1]),
        node('Unsqueeze', ['c', 'axis'])],
        {'axis': tensor((1,), [1], 'int64'), 'w': tensor((32, 1, 1), [(c + 1) / 32 for c in range(32)])},
        output_shape=(1, 1, 97, 32))
    embedding = fixture((1, 76, 32, 1), [node('Reshape', ['x', 's'], 'a'),
        node('ReduceMax', ['a'], 'b', axes=[1], keepdims=0),
        node('MatMul', ['b', 'w'], 'c'), node('Reshape', ['c', 'o'])],
        {'s': tensor((3,), [1, 76, 32], 'int64'), 'o': tensor((4,), [1, 1, 1, 96], 'int64'),
         'w': tensor((32, 96), [1 if i == j % 32 else 0 for i in range(32) for j in range(96)])},
        output_shape=(1, 1, 1, 96))
    return lower_graph(mel), lower_graph(embedding)


def compile_c(output, sources, includes=(), extra=()):
    if not CC:
        raise unittest.SkipTest('C compiler unavailable; no numerical test performed')
    command = [CC, *FLAGS, '-I', str(HERE)]
    for p in includes:
        command.extend(['-I', str(p)])
    command += list(extra) + [str(p) for p in sources] + ['-lm', '-o', str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def graph_run(spec, values, *, workspace_short=False):
    graph = lower_graph(spec)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        code = emit_c(graph, 'test_graph')
        code += '''
#include <string.h>
int test_run(const float *input, float *out, int shortage) {
    float work[%d]; const float *result = 0;
    size_t count = %d;
    int status;
    if (shortage && count) --count;
    status = ef_graph_run(&test_graph, input, work, count, &result, 0, 0);
    if (!status) memcpy(out, result, %d * sizeof(float));
    return status;
}
''' % (max(1, graph['workspace_floats']), graph['workspace_floats'],
       graph['tensors'][graph['output']]['count'])
        (root / 'fixture.c').write_text(code)
        compile_c(root / 'fixture.so', [HERE / 'fixed_frontend.c', root / 'fixture.c'],
                  extra=['-shared', '-fPIC'])
        lib = ctypes.CDLL(str(root / 'fixture.so'))
        lib.test_run.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.test_run.restype = ctypes.c_int
        x = (ctypes.c_float * len(values))(*values)
        out = (ctypes.c_float * graph['tensors'][graph['output']]['count'])()
        status = lib.test_run(x, out, int(workspace_short))
        return status, list(out), graph


def read_trace(path):
    blob = path.read_bytes()
    if blob[:8] != b'EFFTR01\0':
        raise ValueError('TRACE_MAGIC')
    offset, rows = 8, []
    while offset < len(blob):
        if len(blob) - offset < 16:
            raise ValueError('TRACE_TRUNCATED_HEADER')
        stage, count, sample = struct.unpack_from('<IIQ', blob, offset)
        offset += 16
        if count != {1: 3104, 2: 2432, 3: 96, 4: 8}.get(stage) or offset + count * 4 > len(blob):
            raise ValueError('TRACE_RECORD')
        values = struct.unpack_from('<' + 'f' * count, blob, offset)
        offset += count * 4
        if not all(math.isfinite(v) for v in values):
            raise ValueError('TRACE_NONFINITE')
        rows.append((stage, sample, values))
    return rows


class TensorTests(unittest.TestCase):
    def assertValues(self, actual, expected, delta=1e-6):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=delta)

    def test_elementwise_unary_clip_and_max(self):
        spec = fixture((2, 3), [node('Add', ['x', 'row'], 'a'), node('Mul', ['a', 'two'], 'b'),
            node('Sub', ['b', 'one'], 'c'), node('Div', ['c', 'two'], 'd'),
            node('LeakyRelu', ['d'], 'e', alpha=.25), node('Clip', ['e', 'low', 'high'], 'f'),
            node('Max', ['f', 'floor'])],
            {'row': tensor((3,), [1, -2, 3]), 'two': tensor((), [2]), 'one': tensor((), [1]),
             'low': tensor((), [-1]), 'high': tensor((), [4]), 'floor': tensor((2, 1), [.5, 1.5])})
        status, values, _ = graph_run(spec, [-5, 2, 3, 4, 5, -7])
        self.assertEqual(status, 0)
        self.assertValues(values, [.5, .5, 4, 4, 2.5, 1.5])

    def test_pow_log_and_optional_clip_limits(self):
        spec = fixture((4,), [node('Pow', ['x', 'exp'], 'a'), node('Log', ['a'], 'b'),
                             node('Clip', ['b', '', 'high'])],
                       {'exp': tensor((), [2]), 'high': tensor((), [2])})
        status, values, _ = graph_run(spec, [.5, 1, 2, 4])
        self.assertEqual(status, 0)
        self.assertValues(values, [math.log(.25), 0, math.log(4), 2])
        # ONNX Clip when min > max produces max for every element.
        spec = fixture((3,), [node('Clip', ['x', 'a', 'b'])],
                       {'a': tensor((), [3]), 'b': tensor((), [1])})
        self.assertValues(graph_run(spec, [-5, 2, 9])[1], [1, 1, 1])

    def test_matmul_rank_one_and_batched_broadcast(self):
        cases = [((3,), (3,), [1, 2, 3], [4, 5, 6], (), [32]),
                 ((3,), (3, 2), [1, 2, 3], [1, 2, 3, 4, 5, 6], (2,), [22, 28]),
                 ((2, 3), (3,), [1, 2, 3, 4, 5, 6], [1, 2, 3], (2,), [14, 32]),
                 ((2, 1, 2), (1, 2, 2), [1, 2, 3, 4], [1, 2, 3, 4], (2, 1, 2), [7, 10, 15, 22])]
        for a, b, x, w, outshape, expected in cases:
            with self.subTest(a=a, b=b):
                spec = fixture(a, [node('MatMul', ['x', 'w'])], {'w': tensor(b, w)}, output_shape=outshape)
                status, values, _ = graph_run(spec, x)
                self.assertEqual(status, 0)
                self.assertValues(values, expected)

    def test_static_shape_views_constant_cast_and_transpose(self):
        spec = fixture((2, 3), [node('Constant', [], 's32', value=tensor((2,), [3, 2], 'int32')),
            node('Cast', ['s32'], 's', to=7), node('Reshape', ['x', 's'], 'a'),
            node('Unsqueeze', ['a', 'axes'], 'b'), node('Cast', ['b'], 'c', to=1),
            node('Transpose', ['c'], perm=[1, 0, 2])],
            {'axes': tensor((1,), [0], 'int64')}, output_shape=(3, 1, 2))
        status, values, graph = graph_run(spec, [1, 2, 3, 4, 5, 6])
        self.assertEqual(status, 0)
        self.assertValues(values, [1, 2, 3, 4, 5, 6])
        self.assertEqual([n['op'] for n in graph['nodes']], ['Transpose'])
        self.assertEqual(graph['constants'], [])

    def test_reshape_zero_infer_and_reduce_scalar(self):
        spec = fixture((2, 3, 2), [node('Reshape', ['x', 's'], 'a'),
                                  node('ReduceMax', ['a'], axes=[0, -1], keepdims=0)],
                       {'s': tensor((2,), [0, -1], 'int64')}, output_shape=())
        status, values, _ = graph_run(spec, list(range(-12, 0)))
        self.assertEqual(status, 0)
        self.assertValues(values, [-1])

    def test_alias_liveness_and_no_input_overwrite(self):
        spec = fixture((4,), [node('Add', ['x', 'one'], 'a'), node('Reshape', ['a', 's'], 'view'),
            node('Mul', ['a', 'two'], 'b'), node('Add', ['b', 'one'], 'c'),
            node('Reshape', ['c', 's'], 'later'), node('Add', ['view', 'later'])],
            {'one': tensor((), [1]), 'two': tensor((), [2]), 's': tensor((2,), [2, 2], 'int64')})
        status, values, graph = graph_run(spec, [1, 2, 3, 4])
        self.assertEqual(status, 0)
        self.assertValues(values, [7, 10, 13, 16])
        live = [v for v in graph['lifetimes'].values() if v['offset'] is not None]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if max(a['born'], b['born']) <= min(a['last'], b['last']):
                    self.assertNotEqual(a['offset'], b['offset'])
        self.assertLess(graph['workspace_floats'], sum(t['count'] for t in graph['tensors'] if t['storage'] == 'EF_WORK'))

    def test_nonfinite_and_workspace_reject(self):
        spec = fixture((2,), [node('Log', ['x'])])
        self.assertEqual(graph_run(spec, [1, 2], workspace_short=True)[0], -2)
        self.assertEqual(graph_run(spec, [math.nan, 2])[0], -3)
        self.assertEqual(graph_run(spec, [0, 2])[0], -3)

    def test_unknown_attributes_types_and_shapes_reject(self):
        cases = [fixture((1,), [node('Sin', ['x'])]),
                 fixture((1,), [node('Log', ['x'], surprise=1)]),
                 fixture((1,), [node('Cast', ['x'], to=7)]),
                 fixture((2, 2), [node('Reshape', ['x', 'x'])]),
                 fixture((2, 2), [node('Transpose', ['x'], perm=[0, 0])]),
                 fixture((1, 1, 4), [node('MaxPool', ['x'], kernel_shape=[2], auto_pad='SAME_UPPER', ceil_mode=1)]),
                 fixture((1, 1, 4), [node('MaxPool', ['x'], kernel_shape=[2], storage_order=1)]),
                 fixture((1, 1, 4), [node('Conv', ['x', 'w'], strides=[0])],
                         {'w': tensor((1, 1, 2), [1, 1])})]
        for spec in cases:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError): lower_graph(spec)

    def test_pinned_asset_checked_before_onnx_import(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'melspectrogram.onnx'
            path.write_bytes(b'this is an invented file, not a model')
            with self.assertRaisesRegex(ValueError, 'PINNED_ASSET_SHA'):
                read_pinned_onnx(path, ASSETS[path.name])

    def test_inline_tensor_bytes_parser(self):
        class Proto:
            external_data = []
            data_location = 0
            data_type = 1
            dims = [2]
            float_data = []
            raw_data = struct.pack('<ff', .25, -2)
            def HasField(self, field): return False
        self.assertEqual(tensor_proto(Proto()), tensor((2,), [.25, -2]))
        bad = Proto(); bad.external_data = ['data_file']
        with self.assertRaisesRegex(ValueError, 'EXTERNAL'): tensor_proto(bad)

    def test_runtime_has_no_ml_package_import(self):
        # Public stdlib lowering is importable without importing an inference runtime.
        self.assertNotIn('onnxruntime', __import__('sys').modules)
        self.assertNotIn('torch', __import__('sys').modules)
        source = (HERE / 'export_fixed_frontend.py').read_text()
        self.assertNotIn('import numpy', source)

    def test_once_export_guard_uses_pure_fixture_context_only(self):
        spec = importlib.util.spec_from_file_location('_source_static_export_guard', HERE / 'export_fixed_frontend_once.py')
        once = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(once)
        self.assertEqual(once.EXPECTED_ASSETS, {name: {'bytes': row[0], 'sha256': row[1]}
                                               for name, row in ASSETS.items()})
        # These invented dictionaries only exercise a pure validator. No
        # environment mutation, execute_stage/main, download or export occurs.
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REPOSITORY': once.REPOSITORY, 'GITHUB_REF': 'refs/heads/main',
               'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '123'}
        event = {'repository': {'full_name': once.REPOSITORY, 'private': False},
                 'ref': 'refs/heads/main', 'before': once.BEFORE, 'after': 'a' * 40}
        self.assertEqual(once.validate_context(env, event, 'a' * 40, '')['source_head'], 'a' * 40)
        for key, value in (('GITHUB_ACTIONS', 'false'), ('GITHUB_EVENT_NAME', 'pull_request'),
                           ('GITHUB_REF', 'refs/heads/other'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_REPOSITORY', 'fork/elsewhere'), ('GITHUB_SHA', once.BEFORE),
                           ('GITHUB_RUN_ID', '0')):
            with self.subTest(env_field=key):
                with self.assertRaises(ValueError):
                    once.validate_context(dict(env, **{key: value}), event, 'a' * 40, '')
        for patch in ({'before': 'b' * 40}, {'after': 'b' * 40},
                      {'repository': {'full_name': once.REPOSITORY, 'private': True}}):
            with self.subTest(event_patch=patch):
                with self.assertRaises(ValueError):
                    once.validate_context(env, dict(event, **patch), 'a' * 40, '')
        with self.assertRaises(ValueError): once.validate_context(env, event, 'b' * 40, '')
        with self.assertRaises(ValueError): once.validate_context(env, event, 'a' * 40, ' M changed')


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        mel, embedding = invented_pair()
        (cls.root / 'fixed_models.c').write_text(emit_c(mel, 'ef_fixed_mel') + emit_c(embedding, 'ef_fixed_embedding'))
        (cls.root / 'fixed_models.h').write_text('#include "fixed_frontend.h"\n'
            'extern const ef_graph ef_fixed_mel, ef_fixed_embedding;\n')
        compile_c(cls.root / 'replay', [HERE / 'fixed_frontend.c', HERE / 'frontend_replay.c',
                                      cls.root / 'fixed_models.c'], includes=[cls.root])
        (cls.root / 'state_test.c').write_text(r'''
#include "fixed_frontend.h"
#include "fixed_models.h"
#include <stdlib.h>
typedef struct { unsigned calls; uint64_t last; int fail; } observed;
static int callback(void *p, uint64_t sample, const float *value) {
    observed *o = p; (void)value; ++o->calls; o->last = sample; return o->fail;
}
#define CHECK(x) do { if (!(x)) { result = __LINE__; goto end; } } while (0)
int state_test(void) {
    ef_frontend *s = calloc(1, sizeof(*s));
    size_t bytes = ef_frontend_workspace_bytes(&ef_fixed_mel, &ef_fixed_embedding), used = 0;
    float *work = malloc(bytes);
    int16_t *pcm = calloc(16001, sizeof(*pcm));
    observed o = {0}; int result = 0;
    CHECK(s && work && pcm);
    CHECK(ef_frontend_init(s, &ef_fixed_mel, &ef_fixed_embedding, work, bytes / 4) == EF_OK);
    CHECK(ef_frontend_feed(s, pcm, 15999, 0, callback, &o, &used) == EF_OK && used == 15999 && o.calls == 0);
    CHECK(ef_frontend_feed(s, pcm, 1, 15999, callback, &o, &used) == EF_OK && o.calls == 1 && o.last == 16000);
    CHECK(ef_frontend_feed(s, 0, 0, 16000, 0, 0, &used) == EF_OK && used == 0 && o.calls == 1);
    CHECK(ef_frontend_feed(s, pcm, 1, 99, callback, &o, &used) == EF_CLOCK && used == 0);
    CHECK(ef_frontend_discontinuity(s, 15999) == EF_CLOCK);
    CHECK(ef_frontend_discontinuity(s, 20000) == EF_OK && s->observed == 20000);
    CHECK(ef_frontend_feed(s, pcm, 15999, 20000, callback, &o, &used) == EF_OK && o.calls == 1);
    CHECK(ef_frontend_feed(s, pcm, 1, 35999, callback, &o, &used) == EF_OK && o.calls == 2 && o.last == 36000);
    ef_frontend_reset(s);
    CHECK(s->observed == 0 && s->filled == 0 && s->next_end == 16000);
    o.fail = 1;
    CHECK(ef_frontend_feed(s, pcm, 16001, 0, callback, &o, &used) == EF_CALLBACK && used == 16000 && s->failed);
    CHECK(ef_frontend_feed(s, pcm, 1, 16000, callback, &o, &used) == EF_INVALID && used == 0);
    CHECK(ef_frontend_discontinuity(s, UINT64_MAX) == EF_CLOCK);
    CHECK(ef_frontend_discontinuity(s, 16000) == EF_OK && !s->failed);
    o.fail = 0;
    CHECK(ef_frontend_feed(s, pcm, 16000, 16000, callback, &o, &used) == EF_OK && o.last == 32000);
end:
    free(s); free(work); free(pcm); return result;
}
''')
        compile_c(cls.root / 'state.so', [HERE / 'fixed_frontend.c', cls.root / 'fixed_models.c',
                                         cls.root / 'state_test.c'], includes=[cls.root], extra=['-shared', '-fPIC'])
        cls.pcm = [(i % 15000) - 7500 for i in range(40037)]
        (cls.root / 'input.pcm').write_bytes(struct.pack('<' + 'h' * len(cls.pcm), *cls.pcm))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_replay(self, stem, chunks, input_name='input.pcm', binary='replay', head=None):
        trace = self.root / (stem + '.bin')
        args = [str(self.root / binary), str(self.root / input_name), str(trace), chunks]
        if head: args.append(str(head))
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        return json.loads(result.stdout), read_trace(trace), trace.read_bytes()

    def test_chunk_invariance_and_raw_normalized_embedding_taps(self):
        info, rows, blob = self.run_replay('uneven', '1,1279,257,4096')
        other, _, other_blob = self.run_replay('whole', '65536')
        self.assertEqual(blob, other_blob)
        self.assertEqual(info, other)
        ends = list(range(16000, len(self.pcm) + 1, 1280))
        self.assertEqual(info['samples'], len(self.pcm))
        self.assertEqual(info['embeddings'], len(ends))
        self.assertEqual([(stage, end) for stage, end, _ in rows], [(s, e) for e in ends for s in (1, 2, 3)])
        for i, end in enumerate(ends):
            raw = [f32(self.pcm[end - 16000 + 165 * t] * ((c + 1) / 32))
                   for t in range(97) for c in range(32)]
            normalized = [f32(f32(v / 10) + 2) for v in raw[21 * 32:]]
            expected = [max(normalized[t * 32 + c] for t in range(76)) for c in range(32)] * 3
            self.assertEqual(list(rows[3 * i][2]), raw)
            self.assertEqual(list(rows[3 * i + 1][2]), normalized)
            self.assertEqual(list(rows[3 * i + 2][2]), expected)

    def test_no_startup_or_eof_padding_and_prefix_causality(self):
        prefix = self.pcm[:17280]
        (self.root / 'prefix.pcm').write_bytes(struct.pack('<' + 'h' * len(prefix), *prefix))
        info, rows, _ = self.run_replay('prefix-out', '333', 'prefix.pcm')
        self.assertEqual(info['embeddings'], 2)
        full, fullrows, _ = self.run_replay('prefix-full', '513')
        self.assertEqual(rows, fullrows[:6])
        (self.root / 'short.pcm').write_bytes(b'\x01\x00' * 15999)
        short, shortrows, blob = self.run_replay('short-out', '17', 'short.pcm')
        self.assertEqual(short['embeddings'], 0)
        self.assertEqual(shortrows, [])
        self.assertEqual(blob, b'EFFTR01\0')

    def test_reset_discontinuity_clock_and_callback_failure(self):
        lib = ctypes.CDLL(str(self.root / 'state.so'))
        lib.state_test.restype = ctypes.c_int
        self.assertEqual(lib.state_test(), 0, 'nonzero result identifies C assertion line')

    def test_replay_rejects_odd_pcm_and_existing_output(self):
        (self.root / 'odd.pcm').write_bytes(b'\x00\x00\x7f')
        out = self.root / 'odd-output.bin'
        result = subprocess.run([str(self.root / 'replay'), str(self.root / 'odd.pcm'), str(out), '10'],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        kept = self.root / 'existing.bin'
        kept.write_bytes(b'preserve this')
        result = subprocess.run([str(self.root / 'replay'), str(self.root / 'input.pcm'), str(kept), '17'],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(kept.read_bytes(), b'preserve this')

    def test_actual_head_abi_with_invented_constant_weights(self):
        candidates = [HERE.parent / 'kws-pipeline' / 'research' / 'english_final_embedding',
                      HERE.parent / 'english_final_embedding']
        head_dir = next((p for p in candidates if (p / 'english_causal_head.c').is_file()), None)
        if head_dir is None:
            self.skipTest('head source absent: frontend tests remain source-only')
        compile_c(self.root / 'replay-head', [HERE / 'fixed_frontend.c', HERE / 'frontend_replay.c',
                  self.root / 'fixed_models.c', head_dir / 'english_causal_head.c'],
                  includes=[self.root, head_dir], extra=['-DEF_WITH_HEAD'])
        values = [0.] * 96 + [1.] * 96 + [0.] * (32 * 1536 + 32 + 3 * 32) + [0., 5., 0.]
        head = self.root / 'invented-head.bin'
        head.write_bytes(b'EHKWS01\0' + struct.pack('<' + 'f' * len(values), *values))
        info, rows, blob = self.run_replay('head-out', '127,1,2049', binary='replay-head', head=head)
        _, _, whole = self.run_replay('head-whole', '65536', binary='replay-head', head=head)
        self.assertEqual(blob, whole)
        heads = [(sample, values) for stage, sample, values in rows if stage == 4]
        self.assertTrue(info['head_enabled'])
        self.assertEqual([sample for sample, values in heads if values[0] == 1], list(range(35200, 40038, 1280)))
        self.assertEqual([(sample, int(values[1])) for sample, values in heads if values[1]], [(37760, 1)])
        self.assertEqual(info['events'], 1)


if __name__ == '__main__':
    unittest.main()
