#!/usr/bin/env python3
"""Synthetic-only adversarial contract for the standalone development validator."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_real_human_development_corpus as validator  # noqa: E402


class DevelopmentCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="kws-development-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.manifest = self.root / "private-manifest.json"
        self.counter = 0
        self.fixture = {
            "schema_version": 1,
            "dataset_id": "private-dataset-canary",
            "corpus_role": "development-feedback",
            "recordings": [],
        }
        for index in range(3):
            path = self.audio / f"private-audio-{index}.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(struct.pack("<h", index + 1) * 16000)
            row = {
                "recording": f"private-recording-{index}",
                "input_path": path.name,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_bytes": path.stat().st_size,
                "duration_s": 1.0,
                "speaker_id": f"private-speaker-{index}",
                "session_id": "private-session",
                "source_id": f"private-source-{index}",
                "room_id": "private-room",
                "device_id": "private-device",
                "distance_m": 1.0,
                "azimuth_deg": 0.0,
                "snr_db": 20.0,
                "tags": ["household"],
                "expected": ([{"keyword_id": index + 1, "start_s": 0.1, "end_s": 0.3}]
                             if index < 2 else []),
                "capture": {"sample_rate_hz": 16000, "channels": 1,
                            "sample_format": "pcm_s16le"},
                "consent_scope": "product-kws-development",
                "retention_class": "restricted-raw-audio",
            }
            self.fixture["recordings"].append(row)

    def write(self, manifest: object | None = None) -> None:
        self.manifest.write_text(json.dumps(self.fixture if manifest is None else manifest),
                                 encoding="utf-8")

    def cli(self, *, summary: pathlib.Path | None = None,
            audio: pathlib.Path | None = None, manifest: pathlib.Path | None = None,
            extra: tuple[str, ...] = ()) -> tuple[subprocess.CompletedProcess, pathlib.Path]:
        self.counter += 1
        output = summary or self.root / f"summary-{self.counter}.json"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/validate_real_human_development_corpus.py"),
             "--manifest", str(manifest or self.manifest), "--audio-root", str(audio or self.audio),
             "--summary", str(output), *extra],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return completed, output

    def rejected(self, manifest: object | None = None, *, raw: str | bytes | None = None) -> None:
        if raw is None:
            self.write(manifest)
        elif isinstance(raw, bytes):
            self.manifest.write_bytes(raw)
        else:
            self.manifest.write_text(raw, encoding="utf-8")
        result, output = self.cli()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".kws-development-summary-*")))
        self.assertNotIn("private-", result.stderr)
        self.assertNotIn(str(self.root), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_success_aggregate_and_no_authority(self) -> None:
        self.write()
        before = {path: path.read_bytes() for path in [self.manifest, *self.audio.iterdir()]}
        result, output = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, output.read_text())
        summary = json.loads(result.stdout)
        self.assertEqual(summary["recordings"], 3)
        self.assertEqual(summary["positive_speakers"], 2)
        self.assertEqual(summary["expected_by_keyword"], {"1": 1, "2": 1})
        self.assertEqual(summary["negative_audio_hours"], 1 / 3600)
        for field in ("consent_verified", "labels_verified", "anonymization_verified",
                      "training_authority", "qualification_authority", "shipping_authority",
                      "publication_authority"):
            self.assertIs(summary[field], False)
        self.assertIs(summary["structural_checks_passed"], True)
        self.assertIs(summary["audio_bytes_hashes_and_payload_verified"], True)
        self.assertNotIn("private-", result.stdout)
        self.assertNotIn(str(self.root), result.stdout)
        for row in self.fixture["recordings"]:
            self.assertNotIn(row["input_sha256"], result.stdout)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)
        again, _ = self.cli()
        self.assertEqual(again.stdout, result.stdout)

    def test_two_files_can_cover_both_keywords_and_negative(self) -> None:
        self.fixture["recordings"][0]["expected"].append(
            {"keyword_id": 2, "start_s": 0.3, "end_s": 1.0})
        del self.fixture["recordings"][1]
        self.write()
        result, _ = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_valid_nested_stereo_capture_and_metadata_boundaries(self) -> None:
        row = self.fixture["recordings"][0]
        old_path = self.audio / row["input_path"]
        nested = self.audio / "capture"
        nested.mkdir()
        path = nested / "stereo.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(struct.pack("<hh", 7, 8) * 8000)
        row["input_path"] = "capture/stereo.wav"
        row["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["input_bytes"] = path.stat().st_size
        row["capture"] = {"sample_rate_hz": 8000, "channels": 2, "sample_format": "pcm_s16le"}
        row["distance_m"] = 10
        row["snr_db"] = None
        row["azimuth_deg"] = 135
        row["tags"] = ["rear", "household"]
        old_path.unlink()
        del self.fixture["recordings"][1]["snr_db"]
        self.write()
        result, _ = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        row["azimuth_deg"] = 225
        self.write()
        result, _ = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_closed_objects_and_required_keys(self) -> None:
        for target in ((), ("recordings", 0), ("recordings", 0, "capture"),
                       ("recordings", 0, "expected", 0)):
            original = self.fixture
            for key in target:
                original = original[key]
            for field in [*original, "private-unknown-name"]:
                if field == "snr_db":
                    continue
                with self.subTest(target=target, field=field):
                    candidate = copy.deepcopy(self.fixture)
                    node = candidate
                    for key in target:
                        node = node[key]
                    if field in node:
                        del node[field]
                    else:
                        node[field] = {"private-content": "private-person@example.invalid"}
                    self.rejected(candidate)
        for field in ("notes", "metadata", "qualification_id", "deployment_tag", "name"):
            candidate = copy.deepcopy(self.fixture)
            candidate[field] = "private-pii"
            self.rejected(candidate)

    def test_duplicate_keys_and_malformed_json(self) -> None:
        raw = json.dumps(self.fixture)
        for needle, replacement in (
            ('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
            ('"duration_s": 1.0', '"duration_s": 1.0, "duration_s": 1.0'),
            ('"channels": 1', '"channels": 1, "channels": 1'),
            ('"keyword_id": 1', '"keyword_id": 1, "keyword_id": 1'),
        ):
            self.rejected(raw=raw.replace(needle, replacement, 1))
        for raw in ('{} trailing', '[]', 'null', '{"private-pii":', b"\xff", '[' * 1200,
                    '{"schema_version": NaN}', '{"schema_version": Infinity}',
                    '{"schema_version": -Infinity}', '{"schema_version": 1e999}'):
            self.rejected(raw=raw)

    def test_strict_numbers(self) -> None:
        fields = (("schema_version",), ("recordings", 0, "input_bytes"),
                  ("recordings", 0, "duration_s"), ("recordings", 0, "distance_m"),
                  ("recordings", 0, "azimuth_deg"), ("recordings", 0, "snr_db"),
                  ("recordings", 0, "capture", "sample_rate_hz"),
                  ("recordings", 0, "capture", "channels"),
                  ("recordings", 0, "expected", 0, "keyword_id"),
                  ("recordings", 0, "expected", 0, "start_s"),
                  ("recordings", 0, "expected", 0, "end_s"))
        for target in fields:
            for bad in (True, False, "1", [], {}, float("nan"), float("inf"), -1, 10**1000):
                if target[-1] == "snr_db" and bad == -1:
                    continue
                with self.subTest(target=target, bad_type=type(bad).__name__):
                    candidate = copy.deepcopy(self.fixture)
                    node = candidate
                    for key in target[:-1]:
                        node = node[key]
                    node[target[-1]] = bad
                    self.rejected(candidate)
        for target in (fields[0], fields[1], fields[6], fields[7], fields[8]):
            candidate = copy.deepcopy(self.fixture)
            node = candidate
            for key in target[:-1]:
                node = node[key]
            node[target[-1]] = float(node[target[-1]])
            self.rejected(candidate)

    def test_scope_identity_and_bounded_metadata(self) -> None:
        for field, values in {
            "recording": ("", "x" * 65, "private person", None, 1),
            "speaker_id": ("", "x" * 65, "private@example.invalid", [], 2),
            "session_id": (False,), "source_id": ({},), "room_id": ([],), "device_id": (None,),
            "consent_scope": ("qualification", True),
            "retention_class": ("public", None),
            "tags": (["private-tag"], ["household", "household"], [["household"]], ["rear"], "household"),
            "input_sha256": ("F" * 64, "f" * 63, True),
            "duration_s": (0, 3601), "distance_m": (11,), "azimuth_deg": (360,),
            "snr_db": (-101, 101),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    candidate = copy.deepcopy(self.fixture)
                    candidate["recordings"][0][field] = value
                    self.rejected(candidate)
        for field, value in (("dataset_id", "short"), ("dataset_id", 1),
                             ("corpus_role", "fresh-held-out-qualification"),
                             ("schema_version", 2), ("recordings", []),
                             ("recordings", {}), ("recordings", [None] * 10001)):
            candidate = copy.deepcopy(self.fixture)
            candidate[field] = value
            self.rejected(candidate)
        candidate = copy.deepcopy(self.fixture)
        candidate["recordings"][1]["recording"] = candidate["recordings"][0]["recording"]
        self.rejected(candidate)

    def test_event_coverage_timing_and_order(self) -> None:
        for value in ([{}], None, "private-event", [1],
                      [{"keyword_id": 0, "start_s": 0, "end_s": 1}],
                      [{"keyword_id": 3, "start_s": 0, "end_s": 1}],
                      [{"keyword_id": 1, "start_s": 0.3, "end_s": 0.3}],
                      [{"keyword_id": 1, "start_s": 0.1, "end_s": 1.000001}],
                      [{"keyword_id": 1, "start_s": 0.1, "end_s": 0.4},
                       {"keyword_id": 2, "start_s": 0.3, "end_s": 0.6}],
                      [{"keyword_id": 1, "start_s": 0.6, "end_s": 0.7},
                       {"keyword_id": 2, "start_s": 0.1, "end_s": 0.2}]):
            candidate = copy.deepcopy(self.fixture)
            candidate["recordings"][0]["expected"] = value
            self.rejected(candidate)
        candidate = copy.deepcopy(self.fixture)
        candidate["recordings"][0]["expected"] = []
        self.rejected(candidate)
        candidate = copy.deepcopy(self.fixture)
        candidate["recordings"][1]["expected"][0]["keyword_id"] = 1
        self.rejected(candidate)
        candidate = copy.deepcopy(self.fixture)
        candidate["recordings"][2]["expected"] = [{"keyword_id": 1, "start_s": 0, "end_s": 1}]
        self.rejected(candidate)

    def test_absolute_traversal_and_noncanonical_paths(self) -> None:
        for value in (str(self.audio / "private-audio-0.wav"), "../private-audio-0.wav",
                      "sub/../../private-audio-0.wav", "./private-audio-0.wav", "sub//a.wav",
                      "sub/../a.wav", "C:\\private\\a.wav", "a\\b.wav", "a.wav\x00", "",
                      "a/" * 8 + "a.wav", "x" * 65 + ".wav", "missing.wav", [], None):
            candidate = copy.deepcopy(self.fixture)
            candidate["recordings"][0]["input_path"] = value
            self.rejected(candidate)

    def test_symlinks_directories_hardlinks_and_fifo(self) -> None:
        original = self.audio / "private-audio-0.wav"
        outside = self.root / "private-outside.wav"
        outside.write_bytes(original.read_bytes())
        link = self.audio / "link.wav"
        link.symlink_to(outside)
        directory_link = self.audio / "linked"
        directory_link.symlink_to(self.root, target_is_directory=True)
        directory = self.audio / "directory.wav"
        directory.mkdir()
        fifo = self.audio / "fifo.wav"
        os.mkfifo(fifo)
        hardlink = self.audio / "hardlink.wav"
        os.link(outside, hardlink)
        for value in ("link.wav", "linked/private-outside.wav", "directory.wav", "fifo.wav", "hardlink.wav"):
            candidate = copy.deepcopy(self.fixture)
            candidate["recordings"][0]["input_path"] = value
            self.rejected(candidate)
        self.write()
        audio_link = self.root / "audio-link"
        audio_link.symlink_to(self.audio, target_is_directory=True)
        manifest_link = self.root / "manifest-link.json"
        manifest_link.symlink_to(self.manifest)
        for options in ({"audio": audio_link}, {"manifest": manifest_link}):
            result, output = self.cli(**options)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertNotIn(str(self.root), result.stderr)

    def test_byte_hash_capture_duration_and_duplicate_checks(self) -> None:
        for field, value in (("input_sha256", "0" * 64), ("input_bytes", 32043),
                             ("duration_s", 0.999999), ("duration_s", 1.000001),
                             ("capture", {"sample_rate_hz": 8000, "channels": 1, "sample_format": "pcm_s16le"}),
                             ("capture", {"sample_rate_hz": 16000, "channels": 2, "sample_format": "pcm_s16le"}),
                             ("capture", {"sample_rate_hz": 16000, "channels": 1, "sample_format": "float32"})):
            candidate = copy.deepcopy(self.fixture)
            candidate["recordings"][0][field] = value
            self.rejected(candidate)
        first, second = self.fixture["recordings"][:2]
        (self.audio / second["input_path"]).write_bytes((self.audio / first["input_path"]).read_bytes())
        second["input_sha256"] = first["input_sha256"]
        self.rejected(self.fixture)

    def test_full_wav_payload_validation_even_with_correct_manifest_hash(self) -> None:
        path = self.audio / self.fixture["recordings"][0]["input_path"]
        original = path.read_bytes()
        variants = [original[:-2], original + b"private-trailer", original[:20],
                    original[:44], original[:44] + original[44:-1]]
        for offset, fmt, value in ((0, "4s", b"RF64"), (8, "4s", b"NOTW"),
                                   (12, "4s", b"JUNK"), (16, "I", 18), (20, "H", 3),
                                   (22, "H", 0), (22, "H", 17), (24, "I", 0),
                                   (24, "I", 200000), (28, "I", 1), (32, "H", 1),
                                   (34, "H", 8), (36, "4s", b"LIST"), (40, "I", 31999),
                                   (4, "I", 999999)):
            data = bytearray(original)
            struct.pack_into("<" + fmt, data, offset, value)
            variants.append(bytes(data))
        # A RIFF whose sizes accurately describe an incomplete PCM sample is invalid too.
        odd = bytearray(original[:-1])
        struct.pack_into("<I", odd, 4, len(odd) - 8)
        struct.pack_into("<I", odd, 40, len(odd) - 44)
        variants.append(bytes(odd))
        # Zero frames, despite consistent RIFF/data sizes.
        empty = bytearray(original[:44])
        struct.pack_into("<I", empty, 4, 36)
        struct.pack_into("<I", empty, 40, 0)
        variants.append(bytes(empty))
        for index, data in enumerate(variants):
            with self.subTest(variant=index):
                path.write_bytes(data)
                candidate = copy.deepcopy(self.fixture)
                candidate["recordings"][0]["input_bytes"] = len(data)
                candidate["recordings"][0]["input_sha256"] = hashlib.sha256(data).hexdigest()
                self.rejected(candidate)

    def test_no_overwrite_or_output_inside_audio_tree(self) -> None:
        self.write()
        existing = self.root / "existing.json"
        existing.write_text("private-existing-summary")
        symlink = self.root / "output-link.json"
        symlink.symlink_to(self.manifest)
        output_directory = self.root / "output-directory"
        output_directory.mkdir()
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.audio, target_is_directory=True)
        for output in (self.manifest, self.audio / "private-audio-0.wav", self.audio / "new.json",
                       existing, symlink, output_directory, parent_link / "new.json",
                       self.root / "missing-parent" / "new.json"):
            protected = {p: p.read_bytes() for p in [self.manifest, existing, *self.audio.iterdir()]}
            result, _ = self.cli(summary=output)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn(str(self.root), result.stderr)
            for path, before in protected.items():
                self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(self.root.glob(".kws-development-summary-*")))
        self.assertFalse((self.audio / "new.json").exists())
        self.assertFalse((self.root / "missing-parent").exists())

    def test_atomic_publication_failure_cleans_temporary(self) -> None:
        output = self.root / "summary.json"
        for call in ("os.fsync", "os.link"):
            with self.subTest(call=call), mock.patch.object(
                    getattr(validator, call.split(".")[0]), call.split(".")[1], side_effect=OSError):
                with self.assertRaises(OSError):
                    validator.publish_summary(output, b'{"structural_checks_passed":true}\n')
            self.assertFalse(output.exists())
            self.assertFalse(list(self.root.glob(".kws-development-summary-*")))

    def test_post_publication_cleanup_failure_preserves_complete_output(self) -> None:
        self.write()
        output = self.root / "summary.json"
        before = {path: path.read_bytes() for path in [self.manifest, *self.audio.iterdir()]}
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["validator", "--manifest", str(self.manifest), "--audio-root", str(self.audio),
                "--summary", str(output)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                validator.os, "unlink", side_effect=OSError("private-cleanup-failure")), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = validator.main()
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "error: invalid input or filesystem operation failed\n")
        complete = output.read_bytes()
        summary = json.loads(complete)
        self.assertIs(summary["structural_checks_passed"], True)
        self.assertEqual(summary["expected_by_keyword"], {"1": 1, "2": 1})
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        remaining = list(self.root.glob(".kws-development-summary-*"))
        self.assertEqual(len(remaining), 1)
        self.assertTrue(remaining[0].samefile(output))
        self.assertEqual(remaining[0].read_bytes(), complete)
        # A retry must not overwrite the complete destination left by the failure.
        retry, _ = self.cli(summary=output)
        self.assertEqual(retry.returncode, 2)
        self.assertEqual(retry.stdout, "")
        self.assertEqual(output.read_bytes(), complete)
        self.assertEqual(list(self.root.glob(".kws-development-summary-*")), remaining)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_argument_errors_and_os_errors_do_not_echo_private_values(self) -> None:
        self.write()
        result, output = self.cli(extra=("--private-unrecognized-option", str(self.root)))
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertNotIn("private-", result.stderr)
        self.assertNotIn(str(self.root), result.stderr)
        result, output = self.cli(manifest=self.root / "private-missing.json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertFalse(output.exists())
        self.assertNotIn("private-", result.stderr)

    def test_manifest_size_and_changed_input(self) -> None:
        self.manifest.write_bytes(b" " * (validator.MAX_MANIFEST_BYTES + 1))
        result, output = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.write()
        with mock.patch.object(validator, "snapshot", side_effect=[(1,), (2,)]):
            with self.assertRaisesRegex(validator.Invalid, "changed during validation"):
                validator.read_manifest(self.manifest)
        fd = validator.open_directory(self.audio)
        self.addCleanup(os.close, fd)
        row = self.fixture["recordings"][0]
        with mock.patch.object(validator, "snapshot", side_effect=[(1,), (2,)]):
            with self.assertRaisesRegex(validator.Invalid, "changed during validation"):
                validator.inspect_audio(fd, (row["input_path"],), row, "recordings[0]")

    def test_schema_and_ci_contract(self) -> None:
        schema = json.loads((ROOT / "commercial/real-human-development-corpus.schema.json").read_text())
        row = schema["properties"]["recordings"]["items"]
        self.assertEqual(set(row["required"]), validator.ROW_KEYS)
        self.assertEqual(set(row["properties"]), validator.ROW_KEYS | {"snr_db"})
        self.assertEqual(set(row["properties"]["tags"]["items"]["enum"]), validator.TAGS)
        for obj in (schema, row, row["properties"]["capture"], row["properties"]["expected"]["items"]):
            self.assertIs(obj["additionalProperties"], False)
        pattern = row["properties"]["input_path"]["pattern"]
        for path in ("a.wav", "sub/a.wav", "x" * 60 + ".wav"):
            self.assertIsNotNone(re.fullmatch(pattern, path))
            validator.audio_parts(path, "input_path")
        for path in ("/a.wav", "../a.wav", "a//b.wav", "x" * 61 + ".wav"):
            self.assertIsNone(re.fullmatch(pattern, path))
            with self.assertRaises(validator.Invalid):
                validator.audio_parts(path, "input_path")
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("python3 tests/test_restricted_development_dataset.py", workflow)
        self.assertIn("Verify tracked worktree after Restricted development corpus contract", workflow)
        self.assertFalse((ROOT / ".github/workflows/restricted-development-dataset-iteration.yml").exists())
        source = (ROOT / "tools/validate_real_human_development_corpus.py").read_text()
        self.assertNotIn("from validate_real_human_corpus import", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
