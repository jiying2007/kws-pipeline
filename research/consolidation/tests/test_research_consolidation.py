#!/usr/bin/env python3
"""Offline tamper/missing-file checks for the public source retention map."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("retention", ROOT / "tools/verify_research_consolidation.py")
RETENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETENTION)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = b"retained source\n"
        (self.root / "source.txt").write_bytes(raw)
        self.manifest = {"schema": "research-consolidation-retention-v1",
            "sources": [{"pr": 1, "commit": "a" * 40}], "files": [{
            "path": "source.txt", "mode": "100644", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
            "source_pr": 1, "source_commit": "a" * 40}]}

    def test_complete_integrated_inventory(self):
        manifest = json.loads((ROOT / "research/consolidation/core-2026-10-07.json").read_text())
        self.assertEqual(RETENTION.verify(ROOT, manifest), 102)

    def test_exact_source(self):
        self.assertEqual(RETENTION.verify(self.root, self.manifest), 1)

    def test_changed_source(self):
        (self.root / "source.txt").write_bytes(b"tampered source\n")
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)

    def test_missing_source(self):
        (self.root / "source.txt").unlink()
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)

    def test_duplicate_path(self):
        self.manifest["files"].append(copy.deepcopy(self.manifest["files"][0]))
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)

    def test_noncanonical_and_outside_paths(self):
        for path in ("../source.txt", "/source.txt", "./source.txt", "a//source.txt", "a\\source.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["path"] = path
                RETENTION.verify(self.root, manifest)

    def test_symlink_is_not_retention(self):
        (self.root / "link.txt").symlink_to(self.root / "source.txt")
        self.manifest["files"][0]["path"] = "link.txt"
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)

    @unittest.skipUnless(os.name == "posix", "executable bits use the POSIX file contract")
    def test_executable_mode_mismatch(self):
        path = self.root / "source.txt"
        path.chmod(0o755)
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)
        self.manifest["files"][0]["mode"] = "100755"
        self.assertEqual(RETENTION.verify(self.root, self.manifest), 1)
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            RETENTION.verify(self.root, self.manifest)

    def test_undeclared_source(self):
        for field, value in (("source_pr", 2), ("source_commit", "b" * 40)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0][field] = value
                RETENTION.verify(self.root, manifest)

    def test_malformed_identity(self):
        for field, value in (("bytes", True), ("mode", "120000"), ("sha256", "x" * 64), ("source_commit", "main")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0][field] = value
                RETENTION.verify(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
