#!/usr/bin/env python3
"""Mock constructor only; no Whisper imports, weights, model calls or audio."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = next((parent for parent in Path(__file__).resolve().parents
    if (parent / "tools/asr_native_contracts.py").is_file()), HERE)
BASE = Path(os.environ.get("KWS_NATIVE_BASE", str(DEFAULT_BASE)))
SOURCE = HERE / "source" if (HERE / "source").is_dir() else BASE
for root in (BASE / "tools", BASE / "tests"):
    sys.path.insert(0, str(root))
from asr_native_fixtures import native_fixture
from asr_native_contracts import WHISPER_ADAPTER, canonical_sha256
spec = importlib.util.spec_from_file_location("prospective_whisper_binding", SOURCE / "research/qwen12-native-admission-v1/whisper_binding.py")
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.writes = []
        self.fake = SimpleNamespace()
        def write_once(path, value):
            self.writes.append(Path(path).name)
            with Path(path).open("xb") as stream:
                stream.write(encoded(value))
        self.fake.write_once = self.original = write_once
        self.failure = False
        self.skip = False
        self.duplicate = False
        outer = self
        class FixedWhisper:
            def __init__(self, runtime, output):
                self.config, _ = native_fixture(WHISPER_ADAPTER)
                if outer.failure:
                    raise MemoryError("invented load failure")
                if not outer.skip:
                    outer.fake.write_once(output / "native-config.json", self.config)
                if outer.duplicate:
                    outer.fake.write_once(output / "native-config.json", self.config)
                outer.fake.write_once(output / "load-receipt.json", {
                    "source_lock_sha256": self.config["source_lock_sha256"],
                    "asset_lock_sha256": self.config["asset_lock_sha256"],
                    "config_sha256": canonical_sha256(self.config), "audio_calls": 0})
            def decode(self, *args):
                raise AssertionError("no mock audio calls permitted")
        self.fake.FixedWhisper = FixedWhisper
        self.source = encoded({"schema": "retained-asr-source-lock-v1", "executed_head_sha": "a" * 40,
            "files": [{"path": "invented.py", "bytes": 1, "sha256": "b" * 64}]})
        self.asset = encoded({"schema": "retained-asr-model-lock-v1", "model_family": "whisper",
            "assets": [{"path": "invented.pt", "bytes": 1, "sha256": "c" * 64}]})

    def call(self):
        return w.construct_whisper(self.fake, Path("unused"), self.output, self.source, self.asset)

    def test_binds_before_first_persistence_and_preserves_decode(self):
        result = self.call()
        self.assertEqual(self.writes, ["native-config.json", "load-receipt.json"])
        config = json.loads((self.output / "native-config.json").read_bytes())
        receipt = json.loads((self.output / "load-receipt.json").read_bytes())
        self.assertEqual(config, result.config)
        self.assertEqual(config["source_lock_sha256"], hashlib.sha256(self.source).hexdigest())
        self.assertEqual(config["asset_lock_sha256"], hashlib.sha256(self.asset).hexdigest())
        self.assertEqual(receipt["source_lock_sha256"], config["source_lock_sha256"])
        self.assertEqual(receipt["asset_lock_sha256"], config["asset_lock_sha256"])
        self.assertEqual(receipt["config_sha256"], canonical_sha256(config))
        self.assertEqual(receipt["audio_calls"], 0)
        self.assertIs(self.fake.write_once, self.original)
        self.assertIs(result.decode.__func__, self.fake.FixedWhisper.decode)

    def test_existing_config_never_overwritten(self):
        path = self.output / "native-config.json"; path.write_bytes(b"original evidence")
        with self.assertRaisesRegex(ValueError, "fresh native config"):
            self.call()
        self.assertEqual(path.read_bytes(), b"original evidence")
        self.assertEqual(self.writes, [])

    def test_rejects_old_lock_schema_and_wrong_family(self):
        for source, asset in ((b'{"schema":"old-source"}', self.asset),
                              (self.source, b'{"schema":"retained-asr-model-lock-v1","model_family":"qwen3-asr"}')):
            with self.subTest(source=source, asset=asset):
                with self.assertRaisesRegex(ValueError, "resolver-compatible"):
                    w.construct_whisper(self.fake, Path("unused"), self.output, source, asset)
        self.assertEqual(self.writes, [])

    def test_constructor_oom_restores_writer_without_capture(self):
        self.failure = True
        with self.assertRaises(MemoryError):
            self.call()
        self.assertIs(self.fake.write_once, self.original)
        self.assertEqual(self.writes, [])

    def test_missing_config_observation_rejected(self):
        self.skip = True
        with self.assertRaisesRegex(ValueError, "freeze was not observed"):
            self.call()
        self.assertIs(self.fake.write_once, self.original)

    def test_second_config_write_rejected_without_overwrite(self):
        self.duplicate = True
        with self.assertRaisesRegex(ValueError, "frozen once"):
            self.call()
        self.assertEqual(self.writes, ["native-config.json"])
        self.assertIs(self.fake.write_once, self.original)

    def test_import_does_not_load_model_packages(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "whisper", "transformers")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
