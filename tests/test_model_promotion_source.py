#!/usr/bin/env python3
"""Offline source/ZIP admission regressions, including optimized interpreters."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_model_promotion_source.py"
sys.path.insert(0, str(ROOT / "tools"))
import verify_model_promotion_source as verifier

HEAD = "a" * 40
REPOSITORY_ID = 123
RUN_ID = 456
ARTIFACT_ID = 789
# CI runs this complete suite with normal Python, -O, and -OO. Preserve the
# invoking interpreter mode in every production-CLI subprocess.
MODES = (("-" + "O" * sys.flags.optimize,) if sys.flags.optimize else (),)


def zip_bytes(rows=None, *, method=zipfile.ZIP_DEFLATED, descriptor=False, force_zip64=False):
    class NonSeeking(io.BytesIO):
        def seekable(self):
            return False

        def seek(self, *args):
            raise io.UnsupportedOperation("fixture deliberately streams")

    output = NonSeeking() if descriptor else io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(output, "w", compression=method) as archive:
            for name, data in rows or [("best/model.kwm", b"invented opaque model bytes"),
                                      ("training-run-summary.json", b'{}\n')]:
                if force_zip64:
                    info = zipfile.ZipInfo(name) if isinstance(name, str) else name
                    info.compress_type = method
                    with archive.open(info, "w", force_zip64=True) as target:
                        target.write(data)
                else:
                    archive.writestr(name, data)
    return output.getvalue()


def documents(raw):
    repository = dict(id=REPOSITORY_ID, full_name=verifier.REPOSITORY)
    run = dict(id=RUN_ID, workflow_id=321, path=verifier.WORKFLOW_PATH,
               name="untrusted display name", status="completed", conclusion="success",
               head_sha=HEAD, head_branch="main", event="workflow_dispatch",
               repository=repository, head_repository=copy.deepcopy(repository))
    workflow = dict(id=321, path=verifier.WORKFLOW_PATH)
    artifact = dict(id=ARTIFACT_ID, name=verifier.ARTIFACT_NAME, expired=False,
                    size_in_bytes=len(raw), digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                    workflow_run=dict(id=RUN_ID, head_sha=HEAD, head_branch="main",
                                      repository_id=REPOSITORY_ID, head_repository_id=REPOSITORY_ID))
    return dict(run=run, workflow=workflow, artifacts=dict(total_count=1, artifacts=[artifact]))


def mutate_header(raw, *, local_offset=None, central_offset=None, value=0, fmt="I"):
    result = bytearray(raw)
    if local_offset is not None:
        struct.pack_into("<" + fmt, result, local_offset, value)
    if central_offset is not None:
        central = result.index(b"PK\x01\x02")
        struct.pack_into("<" + fmt, result, central + central_offset, value)
    return bytes(result)


def manually_sized_zip(data, claimed_size, *, method=zipfile.ZIP_DEFLATED, trailing=b""):
    name = b"best/model.kwm"
    compressor = zlib.compressobj(wbits=-15)
    compressed = (compressor.compress(data) + compressor.flush()
                  if method == zipfile.ZIP_DEFLATED else data) + trailing
    # The claimed-prefix CRC deliberately lets ZipExtFile return a valid prefix.
    crc = zlib.crc32(data[:claimed_size])
    local = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, method, 0, 0,
                        crc, len(compressed), claimed_size, len(name), 0) + name + compressed
    central = struct.pack("<4s6H3I5H2I", b"PK\x01\x02", 20, 20, 0, method, 0, 0,
                          crc, len(compressed), claimed_size, len(name), 0, 0, 0, 0, 0, 0) + name
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, len(central), len(local), 0)
    return local + central + end


def zip64_end(raw):
    end = raw.rfind(b"PK\x05\x06")
    _, _, _, _, count, length, start, _ = struct.unpack("<4s4H2IH", raw[end:])
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                         count, count, length, start)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end, 1)
    saturated = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 0xffff, 0xffff,
                            0xffffffff, 0xffffffff, 0)
    return raw[:end] + record + locator + saturated


def oversized_logical_zip():
    name = b"oversized"
    size = verifier.MAX_UNCOMPRESSED_BYTES + 1
    local_extra = struct.pack("<HHQQ", 1, 16, size, 0)
    local = struct.pack("<4s5H3I2H", b"PK\x03\x04", 45, 0, 0, 0, 0,
                        0, 0xffffffff, 0xffffffff, len(name), len(local_extra)) + name + local_extra
    central_extra = struct.pack("<HHQ", 1, 8, size)
    central = struct.pack("<4s6H3I5H2I", b"PK\x01\x02", 45, 45, 0, 0, 0, 0,
                          0, 0, 0xffffffff, len(name), len(central_extra), 0, 0, 0, 0, 0) + name + central_extra
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, len(central), len(local), 0)
    return local + central + end


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="promotion-source-")
        self.root = Path(self.temporary.name)
        self.archive = self.root / "model.zip"
        self.output = self.root / "artifact"

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, phase, raw, docs=None, mode=(), extra=(), raw_json=None):
        self.archive.write_bytes(raw)
        docs = documents(raw) if docs is None else docs
        for name in ("run", "workflow", "artifacts"):
            path = self.root / f"{name}.json"
            path.write_text(raw_json[name] if raw_json and name in raw_json else json.dumps(docs[name]),
                            encoding="utf-8")
        command = [sys.executable, "-S", "-B", *mode, str(TOOL), phase,
                   "--repository", verifier.REPOSITORY, "--repository-id", str(REPOSITORY_ID),
                   "--training-run-id", str(RUN_ID), "--expected-head-sha", HEAD,
                   "--run-json", str(self.root / "run.json"),
                   "--workflow-json", str(self.root / "workflow.json"),
                   "--artifacts-json", str(self.root / "artifacts.json"), *extra]
        if phase == "extract":
            command.extend(["--zip", str(self.archive), "--output", str(self.output)])
        return subprocess.run(command, text=True, capture_output=True, timeout=15)

    def accepted(self, raw, docs=None):
        for mode in MODES:
            with self.subTest(mode=mode):
                result = self.invoke("select", raw, docs, mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = documents(raw)["artifacts"]["artifacts"][0]
                self.assertEqual(result.stdout,
                                 f"artifact_id={ARTIFACT_ID}\nartifact_digest={expected['digest']}\n"
                                 f"artifact_size={len(raw)}\n")
                result = self.invoke("extract", raw, docs, mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertTrue(self.output.is_dir())
                shutil.rmtree(self.output)

    def rejected(self, raw, docs=None, *, phase="extract", extra=(), raw_json=None):
        for mode in MODES:
            with self.subTest(mode=mode):
                result = self.invoke(phase, raw, docs, mode, extra, raw_json)
                self.assertNotEqual(result.returncode, 0, "unsafe source unexpectedly accepted")
                self.assertEqual(result.stdout, "", "rejection must not produce GitHub output")
                self.assertFalse(self.output.exists(), "rejection must remove the whole output tree")

    def test_valid_stored_deflated_streamed_and_zip64(self):
        for method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            for descriptor in (False, True):
                for force_zip64 in (False, True):
                    with self.subTest(method=method, descriptor=descriptor, zip64=force_zip64):
                        self.accepted(zip_bytes(method=method, descriptor=descriptor, force_zip64=force_zip64))
        self.accepted(zip64_end(zip_bytes()))

    def test_valid_content_and_directories(self):
        raw = zip_bytes([("best/", b""), ("best/model.kwm", b"opaque"),
                         ("empty", b""), ("evidence/证据.json", b"{}")])
        for mode in MODES:
            result = self.invoke("extract", raw, mode=mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.output / "best/model.kwm").read_bytes(), b"opaque")
            self.assertEqual((self.output / "evidence/证据.json").read_bytes(), b"{}")
            self.assertEqual((self.output / "empty").read_bytes(), b"")
            shutil.rmtree(self.output)

    def test_paginated_and_push_sources(self):
        raw = zip_bytes()
        docs = documents(raw)
        docs["run"]["event"] = "push"
        artifact = docs["artifacts"]["artifacts"][0]
        expired = dict(artifact, id=790, expired=True)
        docs["artifacts"] = [dict(total_count=2, artifacts=[expired]),
                             dict(total_count=2, artifacts=[artifact])]
        self.accepted(raw, docs)

    def test_reject_run_and_workflow_identity_mutations(self):
        raw = zip_bytes()
        mutations = [
            ("id", RUN_ID + 1), ("id", str(RUN_ID)), ("id", True),
            ("workflow_id", 322), ("workflow_id", True), ("path", ".github/workflows/impostor.yml"),
            ("status", "in_progress"), ("conclusion", "failure"), ("head_sha", "b" * 40),
            ("head_branch", "feature"), ("event", "pull_request"), ("event", "pull_request_target"),
            ("event", "workflow_run"), ("event", "schedule"),
        ]
        for field, value in mutations:
            docs = documents(raw)
            docs["run"][field] = value
            with self.subTest(field=field, value=value):
                self.rejected(raw, docs, phase="select")
        for key in ("repository", "head_repository"):
            for field, value in (("id", REPOSITORY_ID + 1), ("id", True),
                                 ("full_name", "attacker/kws-pipeline")):
                docs = documents(raw)
                docs["run"][key][field] = value
                self.rejected(raw, docs, phase="select")
        for field, value in (("path", ".github/workflows/impostor.yml"), ("id", 322), ("id", True)):
            docs = documents(raw)
            docs["workflow"][field] = value
            self.rejected(raw, docs, phase="select")

    def test_reject_requested_identity_mutations(self):
        raw = zip_bytes()
        for extra in (("--repository", "attacker/kws-pipeline"), ("--repository-id", "124"),
                      ("--repository-id", "0"), ("--training-run-id", "0"),
                      ("--expected-head-sha", "A" * 40), ("--training-run-id", "457")):
            self.rejected(raw, phase="select", extra=extra)

    def test_reject_artifact_metadata_mutations(self):
        raw = zip_bytes()
        for field, value in (("name", "impostor"), ("expired", True), ("expired", 0),
                             ("expired", None), ("id", True), ("id", 0),
                             ("size_in_bytes", 0), ("size_in_bytes", True),
                             ("size_in_bytes", verifier.MAX_ARCHIVE_BYTES + 1),
                             ("digest", None), ("digest", ""), ("digest", "sha256:" + "A" * 64),
                             ("digest", "sha256:" + "a" * 64 + "\nartifact_id=666"),
                             ("workflow_run", None)):
            docs = documents(raw)
            docs["artifacts"]["artifacts"][0][field] = value
            with self.subTest(field=field, value=value):
                self.rejected(raw, docs, phase="select")
        for field, value in (("id", 457), ("repository_id", 124), ("head_repository_id", 124),
                             ("id", True), ("head_sha", "b" * 40), ("head_branch", "feature")):
            docs = documents(raw)
            docs["artifacts"]["artifacts"][0]["workflow_run"][field] = value
            self.rejected(raw, docs, phase="select")
        for key in ("digest", "expired", "workflow_run"):
            docs = documents(raw)
            del docs["artifacts"]["artifacts"][0][key]
            self.rejected(raw, docs, phase="select")

    def test_reject_artifact_pagination_mutations(self):
        raw = zip_bytes()
        artifact = documents(raw)["artifacts"]["artifacts"][0]
        bad_pages = [[], {}, dict(total_count=2, artifacts=[artifact]),
                     dict(total_count=True, artifacts=[artifact]),
                     dict(total_count=2, artifacts=[artifact, dict(artifact, id=790)]),
                     dict(total_count=2, artifacts=[artifact, artifact]),
                     [dict(total_count=2, artifacts=[artifact]), dict(total_count=1, artifacts=[])],
                     [dict(total_count=1, artifacts=[artifact]), dict(total_count=1, artifacts=[artifact])]]
        for pages in bad_pages:
            docs = documents(raw)
            docs["artifacts"] = pages
            self.rejected(raw, docs, phase="select")

    def test_reject_json_ambiguity(self):
        raw = zip_bytes()
        encoded = json.dumps(documents(raw)["run"])
        for malformed in ('{"id": 0, ' + encoded[1:], encoded[:-1] + ', "extra": NaN}', "[]"):
            self.rejected(raw, phase="select", raw_json={"run": malformed})

    def test_extraction_rechecks_metadata_and_carrier(self):
        raw = zip_bytes()
        docs = documents(raw)
        docs["run"]["head_repository"]["id"] += 1
        self.rejected(raw, docs)
        docs = documents(raw)
        docs["artifacts"]["artifacts"][0]["size_in_bytes"] += 1
        self.rejected(raw, docs)
        docs = documents(raw)
        docs["artifacts"]["artifacts"][0]["digest"] = "sha256:" + "0" * 64
        self.rejected(raw, docs)

    def test_reject_path_and_type_mutations(self):
        for name in ("../escape", "/absolute", "a/../escape", "a//file", "a/./file", "./file",
                     "a\\file", "C:/drive", "file:alternate", "a\nfile", "a\x7ffile", "a//",
                     "e\u0301", "/", "a/" * 65 + "file"):
            with self.subTest(name=name):
                self.rejected(zip_bytes([(name, b"")]))
        self.rejected(zip_bytes([("same", b"one"), ("same", b"two")]))
        self.rejected(zip_bytes([("same", b""), ("same/", b"")]))
        self.rejected(zip_bytes([("a", b"one"), ("a/b", b"two")]))
        self.rejected(zip_bytes([("directory/", b"content")]))
        for kind in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK):
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (kind | 0o777) << 16
            self.rejected(zip_bytes([(info, b"../../escape")]))
        info = zipfile.ZipInfo("device")
        info.external_attr = 0x400
        self.rejected(zip_bytes([(info, b"")]))

    def test_reject_original_nul_and_local_name_disagreement(self):
        raw = zip_bytes([("best/model.kwm", b"data")])
        # Build raw names directly; ZipInfo construction itself truncates NUL.
        self.rejected(raw.replace(b"best/model.kwm", b"best\x00model.kwm"))
        central = raw.index(b"PK\x01\x02")
        self.rejected(raw[:central].replace(b"best/model.kwm", b"best/other.kwm") + raw[central:])

    def test_reject_unsupported_methods_flags_and_extras(self):
        raw = zip_bytes([("file", b"data")])
        self.rejected(mutate_header(raw, local_offset=6, central_offset=8, value=1, fmt="H"))
        self.rejected(mutate_header(raw, local_offset=8, central_offset=10, value=99, fmt="H"))
        self.rejected(mutate_header(raw, local_offset=6, central_offset=8, value=0x20, fmt="H"))
        for kind in (0x756e, 0x000d, 0x7075, 0xffff):
            info = zipfile.ZipInfo("file")
            info.extra = struct.pack("<HH", kind, 0)
            self.rejected(zip_bytes([(info, b"data")]))

    def test_reject_raw_expansion_truncation_crc_and_deflate_trailing_data(self):
        for method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            forged = manually_sized_zip(b"prefix" + b"x" * 100_000, 6, method=method)
            # Prove this is a real ZipExtFile prefix-acceptance regression.
            with zipfile.ZipFile(io.BytesIO(forged)) as permissive_reader:
                self.assertEqual(permissive_reader.read("best/model.kwm"), b"prefix")
            self.rejected(forged)
            self.rejected(manually_sized_zip(b"short", 9, method=method))
        self.rejected(manually_sized_zip(b"exact", 5, trailing=b"trailing"))
        self.rejected(manually_sized_zip(b"exact", 5, trailing=zlib.compress(b"second stream")))
        raw = zip_bytes([("file", b"data")])
        self.rejected(mutate_header(raw, local_offset=14, central_offset=16, value=123))
        self.rejected(raw[:-2])
        self.rejected(raw + b"trailing")
        self.rejected(b"prefix" + raw)

    def test_reject_size_count_layout_and_descriptor_mutations(self):
        raw = zip_bytes([("file", b"data")])
        self.rejected(oversized_logical_zip())
        # Advertised central size is bounded before constructing ZipFile.
        malicious = bytearray(raw)
        end = malicious.rfind(b"PK\x05\x06")
        struct.pack_into("<I", malicious, end + 12, verifier.MAX_DIRECTORY_BYTES + 1)
        self.rejected(bytes(malicious))
        malicious = bytearray(zip64_end(raw))
        end64 = malicious.index(b"PK\x06\x06")
        struct.pack_into("<QQ", malicious, end64 + 24, verifier.MAX_MEMBERS + 1, verifier.MAX_MEMBERS + 1)
        self.rejected(bytes(malicious))
        self.rejected(mutate_header(raw, central_offset=42, value=1))
        streamed = bytearray(zip_bytes([("file", b"data")], descriptor=True))
        descriptor = streamed.index(b"PK\x07\x08")
        struct.pack_into("<I", streamed, descriptor + 8, 1)
        self.rejected(bytes(streamed))

    def test_failure_removes_previously_extracted_members(self):
        raw = zip_bytes([("first/good", b"good"), ("second/bad", b"bad")], method=zipfile.ZIP_STORED)
        raw = raw.replace(b"badPK\x01\x02", b"BADPK\x01\x02")
        self.rejected(raw)

    def test_external_error_text_is_sanitized(self):
        raw = zip_bytes([("file", b"data")])
        malformed = bytearray(raw)
        central = malformed.index(b"PK\x01\x02")
        struct.pack_into("<H", malformed, central + 6, 999)
        for mode in MODES:
            result = self.invoke("extract", bytes(malformed), mode=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertNotIn(str(self.root), result.stderr)
            self.assertNotIn("file version", result.stderr)
            self.assertFalse(self.output.exists())

    def test_existing_output_and_symlink_are_not_modified(self):
        raw = zip_bytes()
        for mode in MODES:
            self.output.mkdir()
            sentinel = self.output / "sentinel"
            sentinel.write_text("preserve")
            result = self.invoke("extract", raw, mode=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "preserve")
            shutil.rmtree(self.output)
            external = self.root / "external"
            external.mkdir(exist_ok=True)
            self.output.symlink_to(external, target_is_directory=True)
            result = self.invoke("extract", raw, mode=mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(self.output.is_symlink())
            self.assertEqual(list(external.iterdir()), [])
            self.output.unlink()


if __name__ == "__main__":
    unittest.main()
