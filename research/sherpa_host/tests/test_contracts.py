#!/usr/bin/env python3
"""Offline source/patch/wrapper contracts and materializer rejection tests."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from materialize import ROOT, checked_bytes, digest, exact_patch, load_json, materialize, relative, validate_manifest, verify_tree


class SourceContracts(unittest.TestCase):
    def test_frozen_sources(self):
        lock = load_json(ROOT / "sources.lock.json")
        paths = validate_manifest(lock)
        self.assertEqual(len(paths), 491)
        deps = load_json(ROOT / "dependencies.lock.json")
        self.assertEqual({r['component'] for r in lock['files']} - {'repository'}, set(deps['components']))
        for row in lock["files"]:
            if row["component"] == "repository":
                checked_bytes(ROOT, row["path"], row["sha256"], row["bytes"])
            if "patch" in row:
                checked_bytes(ROOT, row["patch"], row["patch_sha256"])
        header = deps["shared_header"]
        checked_bytes(ROOT.parents[1], header["path"], header["sha256"])
        licenses = load_json(ROOT / "licenses.json")
        rows = {r["path"]: r for r in lock["files"]}
        for row in licenses["retained_B_licenses_and_notices"]:
            self.assertEqual(rows[row["path"]]["sha256"], row["sha256"])

    def test_wrapper_original_spans(self):
        lock = load_json(ROOT / "wrapper-extraction.json")
        data = checked_bytes(ROOT, lock["path"], lock["sha256"])
        self.assertEqual(len(lock["segments"]), 19)
        self.assertEqual(sum(r["kind"] == "export" for r in lock["segments"]), 14)
        for row in lock["segments"]:
            start = row["new_byte_offset"]
            self.assertEqual(digest(data[start:start + row["bytes"]]), row["sha256"])
        exports = (ROOT / "cmake/kws-exports.lds").read_text()
        for row in lock["segments"]:
            if row["kind"] == "export":
                self.assertIn(row["name"] + ";", exports)

    def test_compile_contract(self):
        rows = load_json(ROOT / "compile-contracts.json")["units"]
        self.assertEqual(len(rows), 53)
        manifest = {r["path"]: r for r in load_json(ROOT / "sources.lock.json")["files"]}
        for row in rows:
            self.assertEqual(row["sha256"], manifest[row["source"]]["sha256"])
        self.assertEqual(sum("-ffast-math" in r["options"] for r in rows), 2)

    def test_patch(self):
        diff = b"--- a/file\n+++ b/file\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        self.assertEqual(exact_patch(b"a\nb\n", diff), b"a\nc\n")
        for bad in (b"a\nx\n", b"a\nb", b"x\na\nb\n"):
            with self.assertRaises(ValueError): exact_patch(bad, diff)
        for bad in (diff.replace(b"-1,2", b"-1,3"), diff + b"--- a/other\n", diff.replace(b"b/file", b"b/../file")):
            with self.assertRaises(ValueError): exact_patch(b"a\nb\n", bad)

    def test_paths(self):
        for value in ("/x", "../x", "a/../x", "./x", "a//x", "a\\x", "", "a\nx"):
            with self.assertRaises(ValueError): relative(value)

    def test_materializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); repo = base / "repo/research/sherpa_host"; repo.mkdir(parents=True)
            upstream = base / "upstream"; upstream.mkdir(); (upstream / "file").write_bytes(b"a\nb\n")
            row = {"path":"source/file", "input_path":"file", "component":"tiny", "bytes":4,
                   "sha256":digest(b"a\nb\n"), "input_sha256":digest(b"a\nb\n")}
            manifest = {"schema":1, "files":[row]}; inputs = {"tiny":str(upstream)}
            output = base / "output"
            result = materialize(inputs, output, manifest, repo)
            self.assertEqual(result["files"], 1); verify_tree(output, manifest)
            with self.assertRaises(ValueError): materialize(inputs, output, manifest, repo)
            (output / "extra").write_text("unexpected")
            with self.assertRaises(ValueError): verify_tree(output, manifest)
            with self.assertRaises(ValueError): materialize({}, base / "missing", manifest, repo)
            with self.assertRaises(ValueError): materialize(inputs, upstream / "nested", manifest, repo)
            bad = copy.deepcopy(manifest); bad["files"][0]["sha256"] = "0" * 64
            with self.assertRaises(ValueError): materialize(inputs, base / "changed", bad, repo)
            self.assertFalse((base / "changed").exists())
            (upstream / "file").unlink(); (upstream / "file").symlink_to(output / "source/file")
            with self.assertRaises(ValueError): materialize(inputs, base / "linked", manifest, repo)
            self.assertFalse((base / "linked").exists())
            for path in ("../escape", "/absolute"):
                bad = copy.deepcopy(manifest); bad["files"][0]["path"] = path
                with self.assertRaises(ValueError): validate_manifest(bad)
            bad = copy.deepcopy(manifest); bad["files"].append(row)
            with self.assertRaises(ValueError): validate_manifest(bad)


if __name__ == "__main__":
    unittest.main()
