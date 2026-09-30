import importlib.util
import json
import pathlib
import tempfile
import unittest
import copy
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_arm_runtime.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.insert(0, str(ROOT / "scripts"))
import check_native_gates as gates


class AuditTests(unittest.TestCase):
    def test_parse_c_api_snapshot(self):
        info = audit.parse_elf((ROOT / "evidence/sherpa-readelf.txt").read_text())
        self.assertEqual(info["class"], "ELF32")
        self.assertEqual(info["vfp_args"], "VFP registers")
        self.assertIn("GLIBC_2.34", info["required_version_labels"])
        self.assertIn("CXXABI_1.3.13", info["required_version_labels"])
        self.assertNotIn("libasound.so.2", info["needed"])

    def test_missing_attributes_stay_unknown(self):
        self.assertIsNone(audit.parse_elf("")["machine"])

    def test_defined_versions_are_not_requirements(self):
        text = """Version definition section '.gnu.version_d' contains 1 entry:
  Name: GLIBC_9.99
Version needs section '.gnu.version_r' contains 1 entry:
  Name: GLIBC_2.34
Attribute Section: aeabi
  Name: GLIBC_8.88
"""
        self.assertEqual(audit.parse_elf(text)["required_version_labels"], ["GLIBC_2.34"])

    def test_malformed_locks_rejected_before_file_access(self):
        good = json.loads((ROOT / "report/runtime-arm32.lock.json").read_text())
        cases = []
        for entries in [[], good["libraries"][:1], [good["libraries"][0]] * 2]:
            bad = copy.deepcopy(good)
            bad["libraries"] = entries
            cases.append(bad)
        for name in ["/tmp/libsherpa-onnx-c-api.so", "../libsherpa-onnx-c-api.so", "other.so"]:
            bad = copy.deepcopy(good)
            bad["libraries"][0]["filename"] = name
            cases.append(bad)
        for key, value in [("bytes", True), ("bytes", "5101612"), ("sha256", "0" * 64), ("needed", "libc.so.6")]:
            bad = copy.deepcopy(good)
            bad["libraries"][0][key] = value
            cases.append(bad)
        for lock in cases:
            with self.subTest(lock=lock):
                with self.assertRaises(ValueError):
                    audit.verify(pathlib.Path("/must-not-read"), lock, readelf="must-not-run")

    def test_numeric_version_ceiling(self):
        self.assertEqual(gates.check_versions(["GLIBC_2.9"], {"GLIBC": "2.34"}), [])
        self.assertTrue(gates.check_versions(["GLIBC_2.34"], {"GLIBC": "2.31"}))
        self.assertTrue(gates.check_versions(["CXXABI_1.3.13"], {"CXXABI": "1.3.11"}))
        with self.assertRaises(ValueError):
            gates.version("glibc-2.34")

    def test_defined_symbol_evidence(self):
        lines = "00012 t espeak_Initialize\n00013 t piper_phonemize\n00014 T normal_function\n"
        self.assertEqual(len(gates.defined_espeak_symbols(lines)), 2)

    def test_changed_binary_rejected_before_readelf(self):
        lock = json.loads((ROOT / "report/runtime-arm32.lock.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / lock["libraries"][0]["filename"]).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "bytes/hash mismatch"):
                audit.verify(directory, lock, readelf="must-not-run")

    def test_symlink_rejected_before_readelf(self):
        lock = json.loads((ROOT / "report/runtime-arm32.lock.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            path = directory / lock["libraries"][0]["filename"]
            path.symlink_to(ROOT / "report/runtime-arm32.lock.json")
            with self.assertRaisesRegex(ValueError, "regular file"):
                audit.verify(directory, lock, readelf="must-not-run")


if __name__ == "__main__":
    unittest.main()
