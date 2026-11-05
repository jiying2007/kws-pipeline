#!/usr/bin/env python3
"""Offline runtime selection and isolation checks; no model imports or downloads."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_runtime as runtime


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = runtime.load_manifest()

    def test_minor_patch_and_implementation_are_checked(self):
        current = self.manifest["python"]["version"]
        self.assertEqual(runtime.check_interpreter(self.manifest, ("CPython", current))["version"], current)
        for wrong in (("CPython", "3.8.20"), ("CPython", "3.11.16"),
                      ("CPython", "3.13.1"), ("CPython", "3.12.11"), ("PyPy", current)):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "active research Python"):
                runtime.check_interpreter(self.manifest, wrong)

    def test_historical_locks_agree_with_new_role_without_being_rewritten(self):
        tts = runtime.verify_reference_locks(self.manifest, "tts")
        asr = runtime.verify_reference_locks(self.manifest, "asr")
        self.assertEqual(tts["transformers"], "4.57.3")
        self.assertEqual(asr["transformers"], "4.57.6")
        self.assertIn("more-itertools", asr)
        self.assertNotIn("qwen-tts", asr)
        self.assertIn(self.manifest["validation"]["model_smoke"], ("pending", "passed"))

    def test_source_lock_hash_and_package_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = copy.deepcopy(self.manifest)
            manifest["roles"]["asr"]["packages"] = {"numpy": "1.26.4"}
            refs = []
            for name, version in (("a.json", "1.26.4"), ("b.json", "2.3.2")):
                raw = json.dumps({"files": [{"name": "numpy", "version": version}]}).encode()
                (root / name).write_bytes(raw)
                refs.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
            manifest["roles"]["asr"]["reference_locks"] = refs
            with self.assertRaisesRegex(ValueError, "conflicting numpy"):
                runtime.verify_reference_locks(manifest, "asr", root)
            (root / "a.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "historical runtime lock changed"):
                runtime.verify_reference_locks(manifest, "asr", root)

    def test_roles_keep_conflicting_dependencies_separate(self):
        asr = self.manifest["roles"]["asr"]["packages"]
        tts = self.manifest["roles"]["tts"]["packages"]
        self.assertNotEqual(tts["torch"], asr["torch"])
        self.assertNotEqual(tts["transformers"], asr["transformers"])
        self.assertEqual(runtime.requirement_lines(self.manifest, "core", "pytorch_cpu"),
                         ["torch==2.13.0+cpu"])
        self.assertEqual(runtime.requirement_lines(self.manifest, "core", "pypi"),
                         ["numpy==2.3.2", "scipy==1.17.1"])

    def test_package_check_rejects_missing_or_wrong_wheel_without_importing_torch(self):
        packages = self.manifest["roles"]["core"]["packages"]
        with patch.object(importlib.metadata, "version", side_effect=packages.__getitem__):
            self.assertEqual(runtime.check_packages(self.manifest, "core"), packages)
        with patch.object(importlib.metadata, "version", return_value="0.0.0"):
            with self.assertRaisesRegex(ValueError, "observed 0.0.0"):
                runtime.check_packages(self.manifest, "core")
        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(ValueError, "not installed"):
                runtime.check_packages(self.manifest, "core")

    def test_training_assertions_are_derived_from_manifest(self):
        self.assertEqual((ROOT / "training/requirements.lock").read_text(), runtime.training_lock(self.manifest))

    def test_cli_uses_real_interpreter_and_propagates_child_exit_status(self):
        current = self.manifest["python"]["version"]
        observed = runtime.platform.python_version()
        process = subprocess.run([sys.executable, str(ROOT / "tools/research_runtime.py"), "check"],
                                 capture_output=True, text=True)
        if observed != current:
            self.assertEqual(process.returncode, 2)
            self.assertIn(observed, process.stderr)
            return
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["version"], current)
        self.assertEqual(json.loads(process.stdout)["model_smoke"], self.manifest["validation"]["model_smoke"])
        process = subprocess.run([sys.executable, str(ROOT / "tools/research_runtime.py"),
                                  "run", "--", "-c", "raise SystemExit(7)"],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 7, process.stderr)


if __name__ == "__main__":
    unittest.main()
