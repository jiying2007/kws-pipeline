"""Stdlib admission regressions with fake tensors/CDLL; no model or library load."""
import ast
import contextlib
import io
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ctypes_api
import export_a20


class FakeTruth:
    def __init__(self, value): self.value = value
    def all(self): return self.value


class FakeTensor:
    """Only byte/shape metadata, never a numerical tensor or computation."""
    def __init__(self, shape):
        self.shape = shape
        self.dtype = 'fixture-float32'
        self.raw = b'\0' * (4 * math.prod(shape))
        self.finite = True
        self.positive = True
    def detach(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return self.raw
    def __gt__(self, other): return FakeTruth(self.positive)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'fake-checkpoint'
        self.checkpoint.write_bytes(b'invented fixture; no checkpoint deserialization')
        self.state = {name: FakeTensor(shape) for name, shape in export_a20.inventory()}
        self.cp = dict(arm='A', step=1200, symbols=export_a20.SYMBOLS[:], model=self.state)
        self.loader = mock.Mock(side_effect=lambda *a, **kw: self.cp)
        self.torch = types.SimpleNamespace(Tensor=FakeTensor, float32='fixture-float32',
            load=self.loader, isfinite=lambda tensor: FakeTruth(tensor.finite))
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, {'torch': self.torch}).start()

    def allow_fake_file_identity(self):
        mock.patch.object(export_a20, 'FILE_SHA', export_a20.sha(self.checkpoint)).start()

    def allow_fake_state_identity(self):
        digest = hashlib.sha256()
        for name, tensor in self.state.items():
            digest.update(name.encode()); digest.update(tensor.raw)
        mock.patch.object(export_a20, 'STATE_SHA', digest.hexdigest()).start()

    def test_unpinned_checkpoint_is_rejected_before_load_or_output(self):
        output = self.root / 'output'
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
            export_a20.export(self.checkpoint, output)
        self.loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_big_endian_host_is_rejected_before_load(self):
        with mock.patch.object(export_a20.sys, 'byteorder', 'big'):
            with self.assertRaisesRegex(ValueError, 'little-endian'):
                export_a20.read_state(self.checkpoint)
        self.loader.assert_not_called()

    def test_checkpoint_metadata_and_inventory_are_required(self):
        self.allow_fake_file_identity()
        for field, bad in (('arm', 'WRONG'), ('step', True), ('symbols', [])):
            original = self.cp[field]; self.cp[field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'metadata'):
                export_a20.read_state(self.checkpoint)
            self.cp[field] = original
        self.cp['model'] = dict(reversed(list(self.state.items())))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            export_a20.read_state(self.checkpoint)

    def test_tensor_shape_dtype_and_finiteness_are_required(self):
        self.allow_fake_file_identity()
        tensor = next(iter(self.state.values()))
        for field, bad in (('shape', [1]), ('dtype', 'wrong'), ('finite', False)):
            original = getattr(tensor, field); setattr(tensor, field, bad)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'tensor type/shape/dtype/finite'):
                export_a20.read_state(self.checkpoint)
            setattr(tensor, field, original)

    def test_tensor_byte_length_is_required(self):
        self.allow_fake_file_identity()
        next(iter(self.state.values())).raw = b''
        with self.assertRaisesRegex(ValueError, 'tensor byte count'):
            export_a20.read_state(self.checkpoint)

    def test_state_digest_is_required(self):
        self.allow_fake_file_identity()
        with self.assertRaisesRegex(ValueError, 'state SHA256'):
            export_a20.read_state(self.checkpoint)

    def test_positive_cmvn_is_required(self):
        self.allow_fake_file_identity(); self.allow_fake_state_identity()
        self.state['global_cmvn.istd'].positive = False
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            export_a20.read_state(self.checkpoint)

    def test_valid_byte_fixture_checks_all_loader_options(self):
        self.allow_fake_file_identity(); self.allow_fake_state_identity()
        _, payload, tensors = export_a20.read_state(self.checkpoint)
        self.assertEqual(len(payload), 1565280)
        self.assertEqual(len(tensors), 30)
        self.loader.assert_called_once_with(self.checkpoint, map_location='cpu', weights_only=True)

    def test_abi_mismatch_rejected_without_loading_native_code(self):
        fields = {
            'pcm_batch_bytes': ctypes_api.C.sizeof(ctypes_api.PCM_Batch),
            'stream_batch_bytes': ctypes_api.C.sizeof(ctypes_api.Stream_Batch),
            'decoder_result_bytes': ctypes_api.C.sizeof(ctypes_api.Decoder_Result),
            'stream_bytes': 64, 'stream_alignment': 8, 'pcm_bytes': 32,
            'pcm_alignment': 8, 'frontend_bytes': 16, 'frontend_alignment': 8}
        functions = dict(a20fft64s_state_bytes=64, a20fft64s_state_alignment=8,
            donor_fft64_pcm_state_bytes=32, donor_fft64_pcm_state_alignment=8,
            donor_fft64_state_bytes=16, donor_fft64_state_alignment=8)
        class Library:
            def __getattr__(self, name):
                function = mock.Mock(return_value=functions.get(name, 0))
                setattr(self, name, function)
                return function
        path = self.root / 'resources.json'
        with mock.patch.object(ctypes_api.C, 'CDLL', return_value=Library()):
            path.write_text(json.dumps(fields))
            self.assertEqual(ctypes_api.load('not-a-library', path)[1], fields)
            for name in fields:
                for bad in (0, -1, float(fields[name]), None):
                    path.write_text(json.dumps({**fields, name: bad}))
                    with self.subTest(name=name, bad=bad), self.assertRaisesRegex(ValueError, 'ABI/resource'):
                        ctypes_api.load('not-a-library', path)

    def test_invalid_arena_geometry_is_rejected_before_allocation(self):
        with mock.patch.object(ctypes_api.C, 'create_string_buffer') as allocate:
            for size, alignment in ((0, 8), (1, 0), (1, 3), (True, 8), (1, 8.0)):
                with self.subTest(size=size, alignment=alignment), self.assertRaises(ValueError):
                    ctypes_api.Aligned_Arena(size, alignment)
            allocate.assert_not_called()

    def test_source_manifest_mismatch_is_rejected_in_optimized_python(self):
        fixture = self.root / 'manifest-root'
        identity = fixture / 'baseline/export/a20_identity.h'
        identity.parent.mkdir(parents=True)
        identity.write_text('a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d')
        provenance = fixture / 'PROVENANCE.json'
        provenance.write_text(json.dumps(dict(preserved_native_sources=[],
            shipping_files_changed=False, native_math_or_decoder_changed=False)))
        rows = {}
        for path in (identity, provenance):
            raw = path.read_bytes()
            rows[str(path.relative_to(fixture))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (fixture / 'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files=rows)))
        source = (ROOT / 'tests/check_manifest.py').read_text()
        code = compile(source, 'manifest-fixture', 'exec')
        namespace = dict(__name__='manifest_fixture', __file__=str(fixture / 'tests/check_manifest.py'))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, dict(namespace))
        identity.write_text('wrong identity')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            exec(code, dict(namespace))

    def test_runtime_admission_checks_are_not_removable_asserts(self):
        for name in ('export_a20.py', 'ctypes_api.py', 'build.py',
                     'generate_twiddles.py', 'tests/check_manifest.py'):
            tree = ast.parse((ROOT / name).read_text())
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name)


if __name__ == '__main__':
    unittest.main()
