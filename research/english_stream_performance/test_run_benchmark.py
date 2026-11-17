"""Offline input-contract checks; no model or SDK is loaded."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("performance_driver", Path(__file__).with_name("run_benchmark.py"))
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pcm = self.root / "invented-fixture.pcm"
        payload = b"\x01\x00" * 51200
        self.pcm.write_bytes(payload)
        self.case = {"id": "fixture", "path": self.pcm.name,
                     "sha256": hashlib.sha256(payload).hexdigest(), "samples": 51200,
                     "sample_rate_hz": 16000, "encoding": "s16le_mono",
                     "speech_kind": "natural_speech", "continuity": "single_recording",
                     "source": {"scope": "invented unit fixture, not speech evidence"}}

    def load(self, cases):
        manifest = self.root / "inputs.json"
        manifest.write_text(json.dumps({"schema": "english-stream-performance-inputs-v1", "cases": cases}), encoding="utf-8")
        return DRIVER.load_inputs(manifest)

    def test_valid_bytes_preserve_source_and_resolve_path(self):
        manifest, inputs = self.load([self.case])
        self.assertEqual(inputs[0][1], self.pcm.resolve())
        self.assertEqual(inputs[0][0]["source"], self.case["source"])
        self.assertEqual(manifest["cases"][0]["sha256"], self.case["sha256"])

    def test_same_length_tamper_rejected(self):
        with self.pcm.open("r+b") as stream:
            stream.write(b"\x02\x00")
        with self.assertRaisesRegex(ValueError, "length/hash"):
            self.load([self.case])

    def test_truncated_pcm_rejected(self):
        self.pcm.write_bytes(self.pcm.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "length/hash"):
            self.load([self.case])

    def test_short_recording_has_no_steady_claim(self):
        case = copy.deepcopy(self.case)
        case["samples"] = 35200
        with self.assertRaisesRegex(ValueError, "3.2..300"):
            self.load([case])

    def test_duplicate_case_cannot_overwrite_evidence(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load([self.case, copy.deepcopy(self.case)])

    def test_artificial_signal_is_not_admitted_as_speech_probe(self):
        case = copy.deepcopy(self.case)
        case["speech_kind"] = "zero_verification_signal"
        with self.assertRaisesRegex(ValueError, "actual input type"):
            self.load([case])

    def test_timeout_bytes_and_missing_output_can_be_retained(self):
        DRIVER.save_process_logs(self.root, "timeout", b"partial\n", None)
        self.assertEqual((self.root / "timeout.stdout.json").read_text(), "partial\n")
        self.assertEqual((self.root / "timeout.stderr").read_text(), "")


if __name__ == "__main__":
    unittest.main()
