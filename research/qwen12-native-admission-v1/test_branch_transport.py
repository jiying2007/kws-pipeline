"""Local synthetic-only branch transport adversarial tests; no network/ASR."""
import copy
import io
import json
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch_transport as b
import log_transport as t
from emit_synthetic_branch_fixture import build_synthetic_package, verify_synthetic

IDENTITY = {"repository": "jiying2007/kws-pipeline", "run_id": 123,
    "run_attempt": 1, "job_key": "qwen12", "executed_head_sha": "a" * 40}


class BranchTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "fixture.json").write_bytes(random.Random(13).randbytes(600000))
        cls.payload, cls.internal, cls.lines = t.build_package(cls.root, ["fixture.json"], IDENTITY)
        cls.files = b.build_file_set(cls.payload, IDENTITY, "native")
        cls.raw = cls.files[b.manifest_path("native")]
        cls.manifest = t.decode(cls.raw)
        cls.chunks = [(r["path"], cls.files[r["path"]]) for r in cls.manifest["chunks"]]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def parse(self, m=None, chunks=None, expected=None, **kwargs):
        return b.reassemble_file_set(self.raw if m is None else t.canonical(m),
            self.chunks if chunks is None else chunks, expected or IDENTITY, "native", **kwargs)

    def test_exact_zip_round_trip_and_log_parity(self):
        payload, internal, receipt = self.parse(evidence_commit="e" * 40)
        self.assertEqual(payload, self.payload)
        self.assertEqual(internal, self.internal)
        self.assertEqual(t.parse_job_log("\n".join(self.lines), IDENTITY)[0], payload)
        self.assertEqual(receipt["evidence_commit"], "e" * 40)
        self.assertIsNone(receipt["github_actions_artifact_id"])

    def test_missing_duplicate_extra_reordered_chunks(self):
        bad = [self.chunks[:-1], self.chunks + [self.chunks[-1]],
            [self.chunks[0], self.chunks[0], *self.chunks[2:]], self.chunks[::-1]]
        for chunks in bad:
            with self.subTest(chunks=len(chunks)), self.assertRaises(ValueError): self.parse(chunks=chunks)

    def test_wrong_content_size_hash_path_and_type(self):
        name, raw = self.chunks[0]
        for pair in [(name, raw[:-1]), (name, raw + b"x"), (name, b"x" * len(raw)),
                     (name + "/../other", raw), (name, bytearray(raw)), (name, raw.decode("latin1")),
                     (False, raw), {"path": name, "bytes": raw}]:
            with self.subTest(pairtype=type(pair)), self.assertRaises(ValueError):
                self.parse(chunks=[pair, *self.chunks[1:]])

    def test_wrong_run_head_job_attempt_repo(self):
        for key, val in [("run_id", 124), ("run_id", True), ("run_attempt", 2),
            ("executed_head_sha", "b" * 40), ("job_key", "transport-fixture"), ("repository", "else/repo")]:
            with self.subTest(key=key), self.assertRaises(ValueError): self.parse(expected=dict(IDENTITY, **{key: val}))

    def test_manifest_missing_extra_fields_and_duplicate_keys(self):
        for key in self.manifest:
            bad = copy.deepcopy(self.manifest); del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError): self.parse(bad)
        bad = copy.deepcopy(self.manifest); bad["surprise"] = True
        with self.assertRaises(ValueError): self.parse(bad)
        for raw in [b'{"schema":"x",' + self.raw[1:], self.raw + b"\n", b" " + self.raw,
                    b"[]", b"null", b"x" * (b.MAX_MANIFEST + 1)]:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                b.reassemble_file_set(raw, self.chunks, IDENTITY, "native")

    def test_manifest_strict_types_bounds_and_binding(self):
        alterations = {"archive_bytes": [True, 0, b.MAX_COMPRESSED + 1, "600000"],
            "uncompressed_bytes": [True, 0, b.MAX_UNCOMPRESSED + 1],
            "member_count": [True, 0, b.MAX_MEMBERS + 1], "chunk_count": [True, 0, 33],
            "chunk_bytes": [True, b.MAX_CHUNK - 1], "chunks": [{}, [], True],
            "archive_sha256": [True, "A" * 64, "0" * 64], "manifest_sha256": [None, "0" * 64],
            "purpose": ["synthetic", False], "transport": ["github-job-log-chunks-v1"]}
        for key, values in alterations.items():
            for value in values:
                bad = copy.deepcopy(self.manifest); bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.parse(bad)

    def test_chunk_descriptor_strict_types_order_names_and_hashes(self):
        for key, value in [("index", False), ("index", 1), ("bytes", True),
                ("bytes", b.MAX_CHUNK + 1), ("bytes", 0), ("path", "../../escape"),
                ("path", "research/qwen12-native-admission-v1/evidence/native-v1/chunks/0001.bin"),
                ("sha256", "0" * 64), ("sha256", False), ("unknown", "x")]:
            bad = copy.deepcopy(self.manifest); bad["chunks"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.parse(bad)
        bad = copy.deepcopy(self.manifest); bad["chunks"].reverse()
        with self.assertRaises(ValueError): self.parse(bad)

    def test_commit_type_validation(self):
        for value in [True, "main", "A" * 40, "a" * 39, "a" * 41]:
            with self.assertRaises(ValueError): self.parse(evidence_commit=value)

    def test_local_writer_reader_fixed_scope_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            path = b.write_package(root, self.payload, IDENTITY, "native")
            self.assertEqual(path, Path(root) / "qwen12-native-admission-v1-123/evidence-package")
            files, payload, internal, _ = b.read_package(root, IDENTITY, "native")
            self.assertEqual(files, self.files); self.assertEqual(payload, self.payload); self.assertEqual(internal, self.internal)
            with self.assertRaises(FileExistsError): b.write_package(root, self.payload, IDENTITY, "native")
            (path / "extra").write_bytes(b"unexpected")
            with self.assertRaises(ValueError): b.read_package(root, IDENTITY, "native")

    def test_local_symlink_hardlink_missing_and_oversize_fail_closed(self):
        for mode in ["symlink", "hardlink", "missing", "oversize", "extra"]:
            with tempfile.TemporaryDirectory() as root:
                path = b.write_package(root, self.payload, IDENTITY, "native")
                chunk = path / "chunks/0000.bin"
                if mode == "symlink":
                    chunk.unlink(); chunk.symlink_to(self.root / "fixture.json")
                elif mode == "hardlink": os.link(chunk, path.parent / "alias")
                elif mode == "missing": chunk.unlink()
                elif mode == "oversize": chunk.write_bytes(b"x" * (b.MAX_CHUNK + 1))
                else: (path / "chunks/9999.bin").write_bytes(b"x")
                with self.subTest(mode=mode), self.assertRaises((ValueError, OSError)): b.read_package(root, IDENTITY, "native")

    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            actual = Path(root) / "actual"; actual.mkdir()
            link = Path(root) / "link"; link.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ValueError): b.write_package(link, self.payload, IDENTITY, "native")

    def corrupt_zip(self, mutate):
        out = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(self.payload)) as old, zipfile.ZipFile(out, "w") as new:
            entries = [(info, old.read(info)) for info in old.infolist()]
            mutate(entries)
            for info, raw in entries: new.writestr(info, raw)
        return out.getvalue()

    def test_unsafe_zip_rejected_before_any_extraction(self):
        def traversal(entries): entries[1][0].filename = "../escaped.json"
        def symlink(entries): entries[1][0].external_attr = (stat.S_IFLNK | 0o600) << 16
        def mismatch(entries): entries[1] = (entries[1][0], b"changed")
        def missing(entries): entries.pop()
        def stored(entries): entries[1][0].compress_type = zipfile.ZIP_STORED
        def mixed_identity(entries):
            info, raw = entries[0]; value = t.decode(raw); value["identity"]["run_id"] = 999
            entries[0] = info, t.canonical(value)
        for mutation in [traversal, symlink, mismatch, missing, stored, mixed_identity]:
            with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError):
                b.build_file_set(self.corrupt_zip(mutation), IDENTITY, "native")
        self.assertFalse((self.root.parent / "escaped.json").exists())

    def test_exact_public_near_cap_fixture_32_chunks(self):
        identity = dict(IDENTITY, job_key="transport-fixture")
        payload, internal = build_synthetic_package(identity)
        files = b.build_file_set(payload, identity, "synthetic")
        raw = files[b.manifest_path("synthetic")]; manifest = b.parse_manifest(raw, identity, "synthetic")
        self.assertEqual(manifest["chunk_count"], 32)
        self.assertGreaterEqual(len(payload), b.MAX_COMPRESSED * 98 // 100)
        self.assertTrue(all(len(files[r["path"]]) <= b.MAX_CHUNK for r in manifest["chunks"]))
        actual, measured, _ = b.reassemble_file_set(raw, [(r["path"], files[r["path"]]) for r in manifest["chunks"]], identity, "synthetic")
        self.assertEqual(actual, payload); self.assertEqual(measured, internal)
        verify_synthetic(payload, identity)
        with self.assertRaises(ValueError): verify_synthetic(payload[:-1], identity)


if __name__ == "__main__": unittest.main(verbosity=2)
