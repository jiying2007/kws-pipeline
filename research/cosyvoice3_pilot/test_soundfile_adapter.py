"""Stdlib-only checks. These are NOT real dependency/numeric qualification.

Run locally with:
    python -B -m unittest discover -s generation-prep -p test_soundfile_adapter.py -v

Fake modules verify interface and import ordering without importing NumPy,
Torch, TorchAudio, SoundFile, the retained upstream frontend, or any model.
"""

import ast
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soundfile_adapter as adapter
import qualify_soundfile_adapter as qualification


HERE = Path(__file__).resolve().parent


class FakeTensor:
    """Deliberately tiny interface fake, not a substitute DSP implementation."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.calls = []

    def transpose(self, dim0, dim1):
        self.calls.append(("transpose", dim0, dim1))
        if (dim0, dim1) != (0, 1):
            raise AssertionError("wrong transpose")
        result = FakeTensor(zip(*self.rows))
        result.calls = self.calls
        return result

    def mean(self, *, dim, keepdim):
        self.calls.append(("mean", dim, keepdim))
        if dim != 0 or keepdim is not True:
            raise AssertionError("wrong averaging dimensions")
        result = FakeTensor([[sum(values) / len(values) for values in zip(*self.rows)]])
        result.calls = self.calls
        return result


@contextmanager
def fake_audio_modules(rows, rate):
    calls = []
    returned = []
    soundfile = types.ModuleType("soundfile")
    torch = types.ModuleType("torch")
    torchaudio = types.ModuleType("torchaudio")

    def read(wav, **kwargs):
        calls.append(("read", wav, kwargs))
        return rows, rate

    def from_numpy(array):
        calls.append(("from_numpy", array))
        return FakeTensor(array)

    class Resample:
        def __init__(self, **kwargs):
            calls.append(("Resample", kwargs))

        def __call__(self, tensor):
            calls.append(("resample_call", tensor.rows))
            returned.append(tensor)
            return tensor

    soundfile.read = read
    torch.from_numpy = from_numpy
    torchaudio.transforms = types.SimpleNamespace(Resample=Resample)
    with mock.patch.dict(sys.modules, {"soundfile": soundfile, "torch": torch, "torchaudio": torchaudio}):
        yield calls, returned


@contextmanager
def clean_cosyvoice_modules():
    old_modules = {name: value for name, value in sys.modules.items()
                   if name == "cosyvoice" or name.startswith("cosyvoice.")}
    old_path = list(sys.path)
    for name in old_modules:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "cosyvoice" or name.startswith("cosyvoice."):
                del sys.modules[name]
        sys.modules.update(old_modules)
        sys.path[:] = old_path


def fake_source(root, frontend_text=None):
    for relative in ("cosyvoice", "cosyvoice/utils", "cosyvoice/cli"):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (root / "cosyvoice/utils/file_utils.py").write_text(
        "def load_wav(*args, **kwargs):\n    raise RuntimeError('stock loader was used')\n",
        encoding="utf-8",
    )
    (root / "cosyvoice/cli/frontend.py").write_text(
        frontend_text or (
            "from cosyvoice.utils.file_utils import load_wav\n"
            "if load_wav.__module__ != 'soundfile_adapter':\n"
            "    raise RuntimeError('frontend imported before patch')\n"
            "class CosyVoiceFrontEnd:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise RuntimeError('frontend constructor must not run')\n"
        ), encoding="utf-8",
    )


class AdapterInterfaceTests(unittest.TestCase):
    def test_stock_signature(self):
        signature = inspect.signature(adapter.load_wav)
        self.assertEqual(str(signature), "(wav, target_sr, min_sr=16000)")

    def test_mono_decode_float32_always_2d_and_channel_mean(self):
        with fake_audio_modules([[0.25], [-0.5], [0.0]], 16000) as (calls, _):
            output = adapter.load_wav("fixture.wav", 16000)
        self.assertEqual(calls[0], ("read", "fixture.wav", {"dtype": "float32", "always_2d": True}))
        self.assertEqual(output.rows, [[0.25, -0.5, 0.0]])
        self.assertEqual(output.calls, [("transpose", 0, 1), ("mean", 0, True)])
        self.assertFalse(any(call[0] == "Resample" for call in calls))

    def test_stereo_averages_channels_before_resampling(self):
        with fake_audio_modules([[0.5, -0.25], [-0.5, 0.75]], 16000) as (calls, returned):
            output = adapter.load_wav("stereo.wav", 24000)
        self.assertEqual(output.rows, [[0.125, 0.125]])
        self.assertEqual(returned[0].rows, [[0.125, 0.125]])
        self.assertEqual(calls[-2], ("Resample", {"orig_freq": 16000, "new_freq": 24000}))
        self.assertEqual(calls[-1], ("resample_call", [[0.125, 0.125]]))

    def test_reject_8k_resampling_with_stock_error(self):
        with fake_audio_modules([[0.0]], 8000) as (calls, _):
            with self.assertRaisesRegex(AssertionError, "wav sample rate 8000 must be greater than 24000"):
                adapter.load_wav("low.wav", 24000)
        self.assertFalse(any(call[0] == "Resample" for call in calls))

    def test_same_8k_rate_does_not_apply_minimum(self):
        with fake_audio_modules([[0.25]], 8000) as (calls, _):
            self.assertEqual(adapter.load_wav("low.wav", 8000).rows, [[0.25]])
        self.assertFalse(any(call[0] == "Resample" for call in calls))

    def test_custom_minimum_allows_resampling(self):
        with fake_audio_modules([[0.25]], 8000) as (calls, _):
            adapter.load_wav("low.wav", 24000, min_sr=8000)
        self.assertIn(("Resample", {"orig_freq": 8000, "new_freq": 24000}), calls)

    def test_custom_minimum_rejects_only_if_resampling(self):
        with fake_audio_modules([[0.25]], 16000):
            self.assertEqual(adapter.load_wav("a.wav", 16000, min_sr=24000).rows, [[0.25]])
            with self.assertRaises(AssertionError):
                adapter.load_wav("a.wav", 24000, min_sr=24000)

    def test_adapter_import_and_qualifier_import_are_stdlib_only(self):
        script = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import soundfile_adapter, qualify_soundfile_adapter\n"
            "for name in ('numpy', 'soundfile', 'torch', 'torchaudio', 'onnxruntime', 'whisper', 'cosyvoice'):\n"
            "    assert name not in sys.modules, name\n"
        )
        subprocess.run([sys.executable, "-B", "-S", "-c", script, str(HERE)], check=True)

    def test_adapter_preserves_upstream_apache_notice(self):
        text = (HERE / "soundfile_adapter.py").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2021 Mobvoi Inc.", text)
        self.assertIn("2025 Alibaba Inc", text)
        self.assertIn('Licensed under the Apache License, Version 2.0', text)
        self.assertIn("074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", text)

    def test_wrapper_delegates_without_altering_arguments(self):
        with mock.patch.object(qualification, "qualify_adapter", return_value={"status": "stub"}) as function:
            result = adapter.qualify_adapter("approved-source", "new-scratch")
        function.assert_called_once_with("approved-source", "new-scratch")
        self.assertEqual(result, {"status": "stub"})


class BindingTests(unittest.TestCase):
    def test_patch_precedes_import_and_actual_binding_matches(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            root = Path(temporary)
            fake_source(root)
            frontend = adapter.bind_frontend(root)
            self.assertIs(frontend.load_wav, adapter.load_wav)
            self.assertIs(sys.modules["cosyvoice.utils.file_utils"].load_wav, adapter.load_wav)
            self.assertEqual(Path(frontend.__file__).resolve(), root / "cosyvoice/cli/frontend.py")

    def test_second_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            root = Path(temporary)
            fake_source(root)
            adapter.bind_frontend(root)
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                adapter.bind_frontend(root)

    def test_preimported_frontend_is_rejected_before_source_lookup(self):
        with clean_cosyvoice_modules():
            sys.modules["cosyvoice.cli.frontend"] = types.ModuleType("cosyvoice.cli.frontend")
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                adapter.bind_frontend("/nonexistent")

    def test_partially_imported_frontend_is_rejected(self):
        with clean_cosyvoice_modules():
            sys.modules["cosyvoice.cli.frontend"] = None
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                adapter.bind_frontend("/nonexistent")

    def test_frontend_overwriting_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            root = Path(temporary)
            fake_source(root, "def load_wav(*args):\n    pass\n")
            with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                adapter.bind_frontend(root)

    def test_cached_different_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            root = Path(temporary)
            fake_source(root)
            wrong = types.ModuleType("cosyvoice")
            wrong.__file__ = "/a/different/checkout/cosyvoice/__init__.py"
            sys.modules["cosyvoice"] = wrong
            with self.assertRaisesRegex(RuntimeError, "outside approved root"):
                adapter.bind_frontend(root)

    def test_cached_matching_file_utils_can_be_patched(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            root = Path(temporary)
            fake_source(root)
            sys.path.insert(0, str(root))
            file_utils = importlib.import_module("cosyvoice.utils.file_utils")
            original = file_utils.load_wav
            frontend = adapter.bind_frontend(root)
            self.assertIsNot(frontend.load_wav, original)
            self.assertIs(frontend.load_wav, adapter.load_wav)

    def test_source_missing_frontend_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            with self.assertRaisesRegex(RuntimeError, "missing or escaped"):
                adapter.bind_frontend(temporary)

    def test_source_file_symlink_escaping_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, clean_cosyvoice_modules():
            parent = Path(temporary)
            root = parent / "source"
            fake_source(root)
            outside = parent / "outside.py"
            outside.write_text("pass\n", encoding="utf-8")
            frontend = root / "cosyvoice/cli/frontend.py"
            frontend.unlink()
            frontend.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "missing or escaped"):
                adapter.bind_frontend(root)

    def test_module_origin_validation_rejects_unexpected_file(self):
        module = types.ModuleType("example")
        module.__file__ = "/wrong/location.py"
        with self.assertRaisesRegex(RuntimeError, "unexpected source"):
            adapter._module_file(module, Path("/right/location.py"))


class FixtureAndStaticQualificationTests(unittest.TestCase):
    def test_deterministic_pcm16_headers_and_integer_payload(self):
        fixtures = qualification._fixture_definitions()
        self.assertEqual(len(fixtures), 4)
        self.assertEqual(fixtures, qualification._fixture_definitions())
        with tempfile.TemporaryDirectory() as temporary:
            for filename, rate, rows in fixtures:
                path = Path(temporary) / filename
                qualification._write_pcm16(path, rate, rows)
                with wave.open(str(path), "rb") as wav:
                    self.assertEqual(wav.getframerate(), rate)
                    self.assertEqual(wav.getsampwidth(), 2)
                    self.assertEqual(wav.getnchannels(), len(rows[0]))
                    self.assertEqual(wav.getnframes(), 257)
                    payload = wav.readframes(257)
                actual = struct.unpack("<{}h".format(257 * len(rows[0])), payload)
                self.assertEqual(actual, tuple(value for row in rows for value in row))
                self.assertEqual(path.stat().st_size, 44 + 514 * len(rows[0]))
                self.assertEqual(len(hashlib.sha256(path.read_bytes()).hexdigest()), 64)

    def test_known_stereo_means_include_half_lsb_values(self):
        fixtures = {filename: rows for filename, _rate, rows in qualification._fixture_definitions()}
        first = fixtures["stereo_16000.wav"][:8]
        self.assertEqual([sum(row) / 65536 for row in first],
                         [-1 / 65536, -1 / 65536, 0, 1 / 65536, 0, 0, -1 / 65536, 0])

    def test_fixture_writer_will_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.wav"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                qualification._write_pcm16(path, 16000, ((0,),))
            self.assertEqual(path.read_bytes(), b"keep")

    def test_fixture_writer_rejects_invalid_pcm_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.wav"
            with self.assertRaises(ValueError):
                qualification._write_pcm16(path, 16000, ((40000,),))
            self.assertFalse(path.exists())

    def test_existing_scratch_directory_rejected_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                qualification.qualify_adapter("/nonexistent", temporary)

    def test_qualification_calls_bound_frontend_loader(self):
        source = inspect.getsource(qualification.qualify_adapter)
        self.assertIn("bound_loader = frontend.load_wav", source)
        self.assertIn("actual = bound_loader(", source)
        self.assertIn("independent = _expected_tensor(torch, rows)", source)
        self.assertIn("orig_freq=rate, new_freq=target)(independent)", source)
        for case in ("mono16_to24", "stereo16_to24", "mono8_to24_default_min_rejected",
                     "mono8_same_rate_allowed", "stereo16_channel_mean_exact"):
            self.assertIn(case, source)

    def test_production_has_no_model_session_or_downloader_calls(self):
        prohibited = {"CosyVoiceFrontEnd", "CosyVoice3", "AutoModel", "InferenceSession",
                      "SessionOptions", "load_model", "load_state_dict", "snapshot_download",
                      "hf_hub_download", "urlopen", "urlretrieve", "system", "Popen"}
        for filename in ("soundfile_adapter.py", "qualify_soundfile_adapter.py"):
            tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    self.assertNotIn(name, prohibited, "prohibited call in " + filename)
            self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "pickle" for alias in node.names)
                                 for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
