#!/usr/bin/env python3
"""Offline artifact/claim regressions; no package install or model execution."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import active_asr_batch as batch
import setup_asr_runtime as setup


class SetupTests(unittest.TestCase):
    def test_primary_wheel_metadata_excludes_nested_vendored_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setuptools.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("setuptools-81.0.0.dist-info/METADATA", "Name: setuptools\nVersion: 81.0.0\n")
                archive.writestr("setuptools/_vendor/other-1.dist-info/METADATA", "Name: other\nVersion: 1\n")
            self.assertEqual(setup.wheel_identity(path, {"name": "setuptools", "version": "81.0.0"})["version"], "81.0.0")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                setup.wheel_identity(path, {"name": "setuptools", "version": "80.0.0"})

    def test_good_download_is_reused_and_wrong_existing_body_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"expected")
            row = {"sha256": hashlib.sha256(b"expected").hexdigest(), "size_bytes": 8, "url": "https://invalid.invalid/artifact"}
            with patch.object(setup.urllib.request, "urlopen") as network:
                self.assertTrue(setup.acquire(row, path, time.monotonic() + 10)["reused"])
                path.write_bytes(b"changed!")
                with self.assertRaisesRegex(ValueError, "existing artifact differs"):
                    setup.acquire(row, path, time.monotonic() + 10)
                self.assertEqual(path.read_bytes(), b"changed!")
                network.assert_not_called()

    def test_joint_lock_reuses_identical_artifacts(self):
        rows = setup.merged_packages(setup.research_runtime.load_manifest())
        self.assertEqual(len(rows), 125)
        self.assertEqual(rows["torch"]["version"], "2.12.1+cpu")
        self.assertEqual(rows["more-itertools"]["version"], "10.8.0")


class InputTests(unittest.TestCase):
    def archive(self, names):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in names:
                archive.writestr(name, b"x" * 44)
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_audio_count_and_all_member_paths_checked_before_extraction(self):
        with self.archive(["lane/a.wav", "generation.json"]) as archive:
            self.assertEqual([row.filename for row in batch.archive_audio_names(archive, 1)], ["lane/a.wav"])
            with self.assertRaisesRegex(ValueError, "count"):
                batch.archive_audio_names(archive, 2)
        with self.archive(["lane/a.wav", "../escape.json"]) as archive:
            with self.assertRaisesRegex(ValueError, "unsafe"):
                batch.archive_audio_names(archive, 1)

    def test_input_manifest_rejects_text_context_and_modified_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm = root / "clip.f32"
            pcm.write_bytes(b"\0" * 16)
            clip = {"id": "clip-000", "original_member": "a.wav", "original_wav": {}, "original_pcm": {},
                    "derived_pcm": {"path": "clip.f32", "frames": 4, "sample_rate": 16000, "channels": 1,
                                    "dtype": "<f4", "sha256": setup.sha_file(pcm)}}
            manifest = root / "inputs.json"
            setup.save(manifest, {"schema": "active-asr-inputs-v1", "clips": [clip]})
            batch.verify_inputs(manifest)
            clip["target_text"] = "a reference must not reach the recognizer"
            setup.save(manifest, {"schema": "active-asr-inputs-v1", "clips": [clip]})
            with self.assertRaisesRegex(ValueError, "audio identities only"):
                batch.verify_inputs(manifest)
            del clip["target_text"]
            setup.save(manifest, {"schema": "active-asr-inputs-v1", "clips": [clip]})
            pcm.write_bytes(b"\1" * 16)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                batch.verify_inputs(manifest)


class CallLedgerTests(unittest.TestCase):
    def complete_fixture(self, root, clip, identity, key):
        common = {"identity_sha256": key, "model": "whisper", "clip_id": clip["id"],
                  "attempt": "fixture-001", "input": clip, "model_identity": identity}
        claim = {"schema": "active-asr-call-claim-v1", **common}
        native_path = root / "native/whisper/clip-000.json"
        native = {"schema": "active-asr-native-output-v1", "identity_sha256": key,
                  "model": "whisper", "clip_id": clip["id"], "native": {"text": "fixture"}}
        batch.write_once(native_path, native)
        result = {"schema": "active-asr-result-v1", "status": "complete", **common,
                  "native_output_sha256": setup.sha_file(native_path)}
        return claim, result

    def test_completed_identity_is_reused_and_claim_only_is_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = {"id": "clip-000", "derived_pcm": {"sha256": "a" * 64}}
            identity = {"model": "fixture", "config": {"language": "zh"}}
            status, key = batch.existing_result(root, "whisper", clip, identity)
            self.assertEqual(status, "pending")
            claim = root / "claims/whisper/clip-000.json"
            claim_row, result = self.complete_fixture(root, clip, identity, key)
            batch.write_once(claim, claim_row)
            self.assertEqual(batch.existing_result(root, "whisper", clip, identity)[0], "consumed_without_complete_result")
            with self.assertRaises(FileExistsError):
                batch.write_once(claim, {"identity_sha256": key})
            batch.write_once(root / "results/whisper/clip-000.json", result)
            self.assertEqual(batch.existing_result(root, "whisper", clip, identity)[0], "complete")
            with self.assertRaisesRegex(ValueError, "different identity"):
                batch.existing_result(root, "whisper", clip, {**identity, "config": {"language": "en"}})

    def test_missing_changed_or_rebound_native_output_is_rejected_without_consuming_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip, identity = {"id": "clip-000"}, {"model": "fixture"}
            _, key = batch.existing_result(root, "whisper", clip, identity)
            claim, result = self.complete_fixture(root, clip, identity, key)
            batch.write_once(root / "claims/whisper/clip-000.json", claim)
            result_path = root / "results/whisper/clip-000.json"
            batch.write_once(result_path, result)
            native_path = root / "native/whisper/clip-000.json"
            original = native_path.read_bytes()
            native_path.unlink()
            with self.assertRaisesRegex(ValueError, "native output is missing"):
                batch.existing_result(root, "whisper", clip, identity)
            native_path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                batch.existing_result(root, "whisper", clip, identity)
            for field, value in (("schema", "wrong"), ("model", "qwen06"),
                                 ("clip_id", "clip-001"), ("identity_sha256", "f" * 64)):
                native = json.loads(original)
                native[field] = value
                setup.save(native_path, native)
                result["native_output_sha256"] = setup.sha_file(native_path)
                setup.save(result_path, result)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "native output metadata differs"):
                    batch.existing_result(root, "whisper", clip, identity)
            self.assertEqual(len(list((root / "claims").glob("*/*.json"))), 1)

    def test_orphan_success_cannot_hide_missing_call_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip, identity = {"id": "clip-000"}, {"model": "fixture"}
            _, key = batch.existing_result(root, "whisper", clip, identity)
            batch.write_once(root / "results/whisper/clip-000.json", {"identity_sha256": key, "status": "complete"})
            with self.assertRaisesRegex(ValueError, "matching durable call claim"):
                batch.existing_result(root, "whisper", clip, identity)


if __name__ == "__main__":
    unittest.main()
