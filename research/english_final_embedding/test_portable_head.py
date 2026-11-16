"""Invented arrays only. No model weights, model download, Torch or NumPy."""
from __future__ import annotations
import ctypes as C
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portable_head as portable

DIM, HISTORY, HIDDEN = 96, 16, 32
Row = C.c_float * DIM
class Model(C.Structure):
    _fields_ = [('mean96', Row), ('scale96', Row), ('w1', (C.c_float * 1536) * 32),
                ('b1', C.c_float * 32), ('w2', (C.c_float * 32) * 3), ('b2', C.c_float * 3)]
class State(C.Structure):
    _fields_ = [('history', Row * 16), ('next', C.c_uint), ('count', C.c_uint),
                ('candidate', C.c_uint), ('confirmations', C.c_uint),
                ('last_sample', C.c_uint64), ('blocked_until', C.c_uint64)]
class Output(C.Structure):
    _fields_ = [('ready', C.c_int), ('keyword', C.c_uint), ('sample', C.c_uint64),
                ('logits', C.c_float * 3), ('probabilities', C.c_float * 3)]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def invented_model():
    model = Model()
    for d in range(96):
        model.scale96[d] = 1.0
    return model


def binary_model(model):
    groups = [list(model.mean96), list(model.scale96)]
    groups.extend(list(row) for row in model.w1)
    groups.append(list(model.b1))
    groups.extend(list(row) for row in model.w2)
    groups.append(list(model.b2))
    values = [value for group in groups for value in group]
    return portable.MAGIC + struct.pack('<' + 'f' * len(values), *values)


class SyntheticPortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='english-head-synthetic-')
        cls.root = Path(cls.tmp.name)
        cls.exe = cls.root / 'replay'
        portable.compile_replay(cls.exe)
        lib = cls.root / 'head.so'
        subprocess.run(['cc', '-std=c99', '-O2', '-ffp-contract=off', '-fno-fast-math',
                        '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
                        str(portable.DIRECTORY / 'english_causal_head.c'), '-lm', '-o', str(lib)], check=True)
        cls.lib = C.CDLL(str(lib))
        cls.lib.ehead_load.argtypes = [C.POINTER(Model), C.c_void_p, C.c_size_t]
        cls.lib.ehead_reset.argtypes = [C.POINTER(State)]
        cls.lib.ehead_discontinuity.argtypes = [C.POINTER(State)]
        cls.lib.ehead_step.argtypes = [C.POINTER(Model), C.POINTER(State), C.POINTER(C.c_float),
                                      C.c_uint64, C.POINTER(Output)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def step(self, model, state, value, sample):
        row, result = Row(), Output()
        row[0] = value
        self.assertEqual(self.lib.ehead_step(C.byref(model), C.byref(state), row, sample, C.byref(result)), 0)
        return (result.sample, result.ready, result.keyword, tuple(result.logits), tuple(result.probabilities))

    def test_no_fake_history_and_exact_clock(self):
        model, state = invented_model(), State()
        model.b2[:] = [-2.0, 4.0, -2.0]
        results = [self.step(model, state, 0.0, 16000 + i * 1280) for i in range(35)]
        self.assertTrue(all(row[1] == 0 and row[2] == 0 for row in results[:15]))
        self.assertEqual(results[15][0], 35200)
        self.assertEqual([r[0] for r in results if r[2]], [37760, 56960])
        self.assertTrue(all(r[2] in (0, 1) for r in results))
        self.assertEqual(state.count, 16)
        self.assertEqual(C.sizeof(State), 6176)

    def test_sparse_temporal_math_and_normalization(self):
        model, state = invented_model(), State()
        model.mean96[0], model.scale96[0] = 0.25, 0.5
        model.w1[0][0] = 0.75
        model.w1[1][15 * 96] = 0.25
        model.b1[0], model.b1[1] = -0.5, 0.1
        model.w2[1][0], model.w2[1][1] = 1.25, 0.5
        model.w2[2][0], model.w2[2][1] = 0.25, 1.5
        model.b2[:] = [0.3, -0.2, 0.1]
        values = [f32(math.sin(i * 0.4) + 1.5) for i in range(49)]
        for i, value in enumerate(values):
            row = self.step(model, state, value, 16000 + i * 1280)
            if i < 15:
                continue
            old = f32(f32(values[i - 15] - 0.25) / 0.5)
            new = f32(f32(value - 0.25) / 0.5)
            a = max(0.0, f32(-0.5 + f32(0.75 * old)))
            b = max(0.0, f32(f32(0.1) + f32(0.25 * new)))
            expected = [f32(0.3), f32(f32(f32(-0.2) + f32(1.25 * a)) + f32(0.5 * b)),
                        f32(f32(f32(0.1) + f32(0.25 * a)) + f32(1.5 * b))]
            for actual, wanted in zip(row[3], expected):
                self.assertAlmostEqual(actual, wanted, delta=2e-6)
            self.assertAlmostEqual(sum(row[4]), 1.0, delta=2e-6)

    def test_segmented_feeds_future_prefix_and_reset(self):
        model = invented_model()
        model.w1[0][0] = model.w1[1][15 * 96] = 0.5
        model.w2[1][0], model.w2[2][1] = 2.0, 2.0
        values = [math.sin(i / 3) + 1 for i in range(50)]
        def run(groups, reset=False):
            state, output, index = State(), [], 0
            if reset:
                for i in range(24):
                    self.step(model, state, 7.0, 16000 + i * 1280)
                self.lib.ehead_reset(C.byref(state))
            for group in groups:
                for value in group:
                    output.append(self.step(model, state, value, 16000 + index * 1280))
                    index += 1
            return output
        full = run([values])
        self.assertEqual(full, run([values[:1], values[1:17], values[17:32], values[32:]]))
        self.assertEqual(full, run([values], reset=True))
        changed = run([values[:31] + [99.0] * 19])
        self.assertEqual(full[:31], changed[:31])

    def test_discontinuity_preserves_time_but_discards_acoustic_history(self):
        model, state = invented_model(), State()
        model.b2[:] = [-2.0, 4.0, -2.0]
        for i in range(18):
            self.step(model, state, 0.0, 16000 + i * 1280)
        clock = state.last_sample
        self.assertGreater(state.blocked_until, clock)
        self.lib.ehead_discontinuity(C.byref(state))
        self.assertEqual((state.last_sample, state.count, state.candidate, state.blocked_until), (clock, 0, 0, 0))
        outputs = [self.step(model, state, 0.0, clock + 16000 + i * 1280) for i in range(18)]
        self.assertTrue(all(row[1] == 0 for row in outputs[:15]))
        self.assertEqual(outputs[-1][2], 1)

    def test_invalid_inputs_do_not_advance_state(self):
        model, state, out = invented_model(), State(), Output()
        self.step(model, state, 0.0, 16000)
        before = bytes(state)
        for row, sample in [(Row(float('nan')), 17280), (Row(), 17000), (Row(), 16000)]:
            self.assertEqual(self.lib.ehead_step(C.byref(model), C.byref(state), row, sample, C.byref(out)), -1)
            self.assertEqual(bytes(state), before)

    def test_binary_geometry_keys_and_rejected_values(self):
        original, restored = invented_model(), Model()
        original.w1[31][1535] = -0.123
        original.w2[2][31] = 7.25
        raw = binary_model(original)
        self.assertEqual(len(raw), 197908)
        self.assertEqual(self.lib.ehead_load(C.byref(restored), raw, len(raw)), 0)
        self.assertEqual(bytes(original), bytes(restored))
        corrupted = bytearray(raw)
        corrupted[8:12] = struct.pack('<f', float('nan'))
        zero_scale = bytearray(raw)
        zero_scale[8 + 96 * 4: 12 + 96 * 4] = struct.pack('<f', 0.0)
        for data in (raw[:-1], raw + b'0', bytes(corrupted), bytes(zero_scale), b'BADKWS01' + raw[8:]):
            self.assertEqual(self.lib.ehead_load(C.byref(restored), data, len(data)), -1)

    def test_replay_executable_matches_state_api_and_rejects_trailing_bytes(self):
        model, state = invented_model(), State()
        model.b2[:] = [-2.0, -2.0, 4.0]
        rows = [Row() for _ in range(35)]
        samples = [16000 + i * 1280 for i in range(35)]
        body = portable.STREAM_MAGIC + struct.pack('<I', len(rows)) + b''.join(
            struct.pack('<QI96f', sample, 0, *row) for sample, row in zip(samples, rows))
        weights, stream = self.root / 'model.bin', self.root / 'stream.bin'
        weights.write_bytes(binary_model(model)); stream.write_bytes(body)
        run = subprocess.run([str(self.exe), str(weights), str(stream)], check=True, text=True, capture_output=True)
        actual = [line.split() for line in run.stdout.splitlines()]
        expected = [self.step(model, state, 0.0, sample) for sample in samples]
        self.assertEqual([(int(r[0]), int(r[1]), int(r[2])) for r in actual], [r[:3] for r in expected])
        stream.write_bytes(body + b'extra')
        self.assertNotEqual(subprocess.run([str(self.exe), str(weights), str(stream)], capture_output=True).returncode, 0)

    def test_source_checks_do_not_import_tensor_runtime(self):
        self.assertFalse({'numpy', 'torch', 'onnxruntime'} & set(sys.modules))


if __name__ == '__main__':
    unittest.main()
