#!/usr/bin/env python3
"""Offline tamper/missing-file checks for the public source retention map."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("retention", ROOT / "tools/verify_research_consolidation.py")
RETENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETENTION)


sys.path.insert(0, str(ROOT / "tools"))
import verify_research_sources as SOURCE_RETENTION
from run_research_source_checks import make_projection


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

    def test_original_ci_archive_and_active_ci_are_separately_pinned(self):
        core = json.loads((ROOT / "research/consolidation/core-2026-10-07.json").read_text())
        archive = next(row for row in core["files"] if row.get("source_path") == ".github/workflows/ci.yml")
        self.assertEqual(archive["path"], "research/consolidation/maintained-sources/pr-485/ci.yml")
        self.assertEqual(archive["source_pr"], 485)
        self.assertEqual(archive["source_commit"], "6f2461ff11cddf0a1264f5e2a04282c40b5dc4ce")
        self.assertEqual(archive["sha256"], "81fa55fe96fbc5b8008129a5442b95b219a4940dc4925b20eac13295d4f8fa0d")
        self.assertEqual(archive["git_blob_sha1"], "1a9d3eaa494922fe2481749741f94b7639ab277a")
        self.assertEqual(archive["bytes"], 23870)
        current = json.loads((ROOT / SOURCE_RETENTION.MANIFEST).read_text())
        active = next(row for row in current["baseline_active_workflows"] if row["path"] == archive["source_path"])
        self.assertNotEqual(active["sha256"], archive["sha256"])
        for row in (archive, active):
            path = self.root / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / row["path"]).read_bytes())
        (self.root / ".github/workflows/research-source-consolidation.yml").write_text("name: fixture\n")
        arm = self.root / SOURCE_RETENTION.ARM
        arm.parent.mkdir(parents=True)
        arm.write_text(json.dumps(SOURCE_RETENTION.DISABLED_ARM))
        core["files"] = [archive]
        source = dict(schema="research-source-retention-v1", repository="jiying2007/kws-pipeline",
                      base_commit=current["base_commit"], files=[archive],
                      provenance=[dict(archive, original_path=archive["source_path"],
                          source_url="https://github.com/jiying2007/kws-pipeline/blob/" + archive["source_commit"] + "/" + archive["source_path"])],
                      source_projections=[], baseline_active_workflows=[active], pointer_only=[],
                      allowed_new_active_workflows=[".github/workflows/research-source-consolidation.yml"])
        self.assertEqual(RETENTION.verify(self.root, core), 1)
        self.assertEqual(SOURCE_RETENTION.verify(self.root, source), 1)
        old = self.root / archive["path"]
        raw = old.read_bytes()
        old.write_bytes(raw + b"# tampered archive\n")
        with self.assertRaisesRegex(ValueError, "retained byte count"):
            RETENTION.verify(self.root, core)
        old.write_bytes(raw)
        live = self.root / active["path"]
        live.write_bytes(live.read_bytes() + b"# tampered active workflow\n")
        self.assertEqual(RETENTION.verify(self.root, core), 1)
        with self.assertRaisesRegex(ValueError, "retained byte count"):
            SOURCE_RETENTION.verify(self.root, source)

    def test_offline_projection_copies_archived_ci_as_inactive_data(self):
        manifest = json.loads((ROOT / SOURCE_RETENTION.MANIFEST).read_text())
        projected = self.root / "projected"
        projected.mkdir()
        make_projection(ROOT, projected, manifest)
        name = "research/consolidation/maintained-sources/pr-485/ci.yml"
        self.assertEqual((projected / name).read_bytes(), (ROOT / name).read_bytes())
        # The closed projection now copies the retained historical path. It must
        # not silently replace it with the maintained active CI source.
        self.assertFalse((projected / ".github/workflows/ci.yml").exists())

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
