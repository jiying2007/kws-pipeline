"""Offline consistency checks; do not fetch or load the historical runtime."""
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_archived", ROOT / "scripts/audit_arm_runtime.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def report(name):
    return json.loads((ROOT / "report" / name).read_text())


class ArchivedEvidenceTests(unittest.TestCase):
    def test_lock_and_retained_readelf_results_agree(self):
        lock = report("runtime-arm32.lock.json")
        audit.validate_lock(lock)
        retained = report("static-audit-result.json")
        self.assertTrue(retained["static_audit_passed"])
        self.assertFalse(retained["target_execution_tested"])
        self.assertFalse(retained["adapter_cross_build_tested"])
        self.assertIsNone(retained["ssc305_compatible"])
        self.assertEqual(retained["runtime_selected_bytes"], 25718673)
        for locked, recorded, fixture in zip(lock["libraries"], retained["libraries"], ["sherpa-readelf.txt", "ort-readelf.txt"]):
            self.assertEqual(locked["sha256"], recorded["sha256"])
            self.assertEqual(locked["bytes"], recorded["bytes"])
            parsed = audit.parse_elf((ROOT / "evidence" / fixture).read_text())
            self.assertEqual(parsed, recorded["elf"])

    def test_expected_no_espeak_failure_is_not_promoted(self):
        gate = report("pinned-runtime-no-espeak-gate.json")
        self.assertFalse(gate["static_gates_passed"])
        self.assertFalse(gate["legal_review_completed"])
        self.assertIsNone(gate["ssc305_compatible"])
        self.assertIn("Defined eSpeak/phonemize symbols remain", gate["failures"])
        self.assertTrue(any("espeak_Initialize" in line for line in gate["defined_espeak_or_phonemize_symbols"]))

    def test_example_ceilings_remain_failures(self):
        gate = report("older-sdk-threshold-example.json")
        self.assertFalse(gate["static_gates_passed"])
        self.assertTrue(any("GLIBC_2.34 exceeds" in line for line in gate["failures"]))
        self.assertTrue(any("GLIBCXX_3.4.29 exceeds" in line for line in gate["failures"]))

    def test_recipe_is_unexecuted_and_tts_disabled(self):
        recipe = report("kws-only-build-recipe.json")
        self.assertTrue(recipe["status"].startswith("UNEXECUTED:"))
        self.assertIn("-DSHERPA_ONNX_ENABLE_TTS=OFF", recipe["configure_argv_template"])
        self.assertIn("-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF", recipe["configure_argv_template"])
        self.assertIn("-DSHERPA_ONNX_ENABLE_BINARY=OFF", recipe["configure_argv_template"])

    def test_no_binary_audio_model_or_toolchain_payload(self):
        forbidden = {".so", ".a", ".onnx", ".wav", ".pcm", ".xz", ".bz2", ".zip", ".whl"}
        for path in ROOT.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                self.assertNotIn(path.suffix, forbidden, str(path))
                self.assertNotEqual(path.read_bytes()[:4], b"\x7fELF", str(path))


if __name__ == "__main__":
    unittest.main()
