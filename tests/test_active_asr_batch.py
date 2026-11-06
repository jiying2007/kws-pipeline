#!/usr/bin/env python3
"""Offline artifact/claim regressions; no package install or model execution."""
from pathlib import Path
import hashlib
import io
import json
from contextlib import contextmanager
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import active_asr_batch as batch
import setup_asr_runtime as setup
import research_d2_checkpoints as checkpoints


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


class CompletedPilotTests(unittest.TestCase):
    """Synthetic local evidence only; no provider access or model execution."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        setup.save(self.runtime / "setup.json", {"status": "complete", "installed_packages": {},
            "lock_sha256": "a" * 64, "active_python": "fictional-python"})

    def inputs(self, name, count):
        root = self.root / name
        clips = []
        for index in range(count):
            identifier = f"clip-{index:03d}"
            pcm, wav = root / (identifier + ".f32"), root / (identifier + ".wav")
            root.mkdir(exist_ok=True)
            pcm.write_bytes(index.to_bytes(4, "little") * 4)
            wav.write_bytes(b"fictional-audio-identity-only-" + bytes([index]))
            clips.append({"id": identifier, "original_member": name + "/" + wav.name,
                "original_wav": {"path": wav.name, "sha256": setup.sha_file(wav), "bytes": wav.stat().st_size},
                "original_pcm": {"sha256": setup.sha_file(pcm)},
                "derived_pcm": {"path": pcm.name, "frames": 4, "sample_rate": 16000,
                    "channels": 1, "dtype": "<f4", "sha256": setup.sha_file(pcm)}})
        path = root / "inputs.json"
        setup.save(path, {"schema": "active-asr-inputs-v1", "clips": clips})
        return path, clips

    def pilot(self):
        inputs_path, clips = self.inputs("pilot-inputs", 2)
        output = self.root / "pilot-queue/jobs/pilot/outputs/native"
        old = self.root / "frozen-source"
        names = ["tools/active_asr_batch.py", "tools/active_asr_worker.py", "tools/setup_asr_runtime.py",
            "tools/research_runtime.py", "tools/research_job_lock.py", "tools/summarize_active_asr_batch.py",
            "configs/runtime/research-runtime.json", "research/qwen6_asr/model-locks.json",
            "research/source-screen32-whisper-v1/source-lock.json"]
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "research/qwen6_asr/core").rglob("*.py")]
        for name in names:
            (old / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, old / name)
        worker_sha = setup.sha_file(old / "tools/active_asr_worker.py")
        job = {"identity": {"input_manifest_sha256": setup.sha_file(inputs_path), "models": list(batch.MODELS),
            "max_calls": 6, "heavy_model_concurrency": 1, "threads": 4}, "boundary": batch.BOUNDARY,
            "deadline_unix": 12345}
        batch.write_once(output / "job.json", job)
        for model in batch.MODELS:
            identity = batch.model_identity(model, self.runtime, inputs_path, 4)
            for clip in clips:
                key = batch.digest({"model_identity": identity, "input": clip})
                shared = {"identity_sha256": key, "model": model, "clip_id": clip["id"],
                    "attempt": "fictional-pilot", "input": clip, "model_identity": identity}
                batch.write_once(output / "claims" / model / (clip["id"] + ".json"),
                    {"schema": "active-asr-call-claim-v1", **shared})
                native_path = output / "native" / model / (clip["id"] + ".json")
                batch.write_once(native_path, {"schema": "active-asr-native-output-v1", "identity_sha256": key,
                    "model": model, "clip_id": clip["id"], "native": {"text": "fictional"}})
                batch.write_once(output / "results" / model / (clip["id"] + ".json"),
                    {"schema": "active-asr-result-v1", "status": "complete", **shared,
                     "native_output_sha256": setup.sha_file(native_path), "decoder_source_sha256": worker_sha})
        files = {}
        for prefix, folder in (("inputs", inputs_path.parent), ("", output)):
            for path in folder.rglob("*"):
                if path.is_file():
                    name = "/".join(filter(None, (prefix, path.relative_to(folder).as_posix())))
                    files[name] = {k: checkpoints.pin(path)[k] for k in ("bytes", "sha256")}
        inventory = output / "audit/file-hashes.json"
        batch.write_once(inventory, {"roots": {"inputs": str(inputs_path.parent), "results": str(output)},
            "file_count": len(files), "files_sha256": batch.digest(files), "files": files})
        batch.write_once(output / "audit/final-summary.json", {"schema": "active-asr-batch-final-audit-v1",
            "status": "complete", "errors": [], "input_manifest_sha256": setup.sha_file(inputs_path),
            "expected_matrix_cells": 6, "verified_matrix_cells": 6, "original_recognition_calls": 6,
            "duplicate_model_clip_pairs": 0, "finished_before_deadline": True, "persisted_deadline_unix": 12345,
            "inventory_sha256": setup.sha_file(inventory), "decoder_source_sha256": worker_sha,
            "auditor_source_sha256": setup.sha_file(old / "tools/summarize_active_asr_batch.py")})
        job_pin = checkpoints.pin(output / "job.json")
        result_path = output.parent.parent / "result.json"
        batch.write_once(result_path, {"schema": "kws-local-research-job-result-v1", "status": "completed",
            "exit_code": 0, "actual_duration_known": True, "automatic_retry": False,
            "outputs": {"native/job.json": {k: job_pin[k] for k in ("bytes", "sha256")}}})
        receipt_root = self.root / "fictional-audit"
        batch.write_once(receipt_root / "command.json", {"argv": ["fictional-python", "-E", "-S", "-B",
            str(old / "tools/summarize_active_asr_batch.py"), "--runtime", str(self.runtime),
            "--inputs", str(inputs_path), "--output", str(output)]})
        for name in ("stdout.txt", "stderr.txt"):
            (receipt_root / name).write_text("fictional test receipt; no decoder ran")
        receipt = receipt_root / "receipt.json"
        batch.write_once(receipt, {"returncode": 0, "timed_out": False, "launch_error": None,
            **{field: checkpoints.pin(receipt_root / name) for field, name in
               (("command", "command.json"), ("stdout", "stdout.txt"), ("stderr", "stderr.txt"))}})
        folder = self.root / "fictional-checkpoint"
        paths = [p for root in (old, inputs_path.parent, output.parent.parent, receipt_root)
                 for p in root.rglob("*") if p.is_file()]
        descriptor = checkpoints.create_capsule(folder, {"fixture": "synthetic; no actual provider or ASR"}, paths)
        saved = self.root / "fictional-provider-response.json"
        setup.save(saved, {"schema": checkpoints.SAVE, "status": "saved", "provider": "fixture-only",
            "object_id": "fictional-object", "version": 0, "provider_response": {"fixture": "fictional-object"},
            "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
        checkpoints.acknowledge(folder, saved)
        proof = self.root / "completed-pilot.json"
        batch.write_once(proof, {"schema": "active-asr-completed-pilot-v1", "inputs": checkpoints.pin(inputs_path),
            "output": str(output), "source_root": str(old), "queue_result": checkpoints.pin(result_path),
            "audit_receipt": checkpoints.pin(receipt), "checkpoint": str(folder),
            "checkpoint_descriptor": checkpoints.pin(folder / "capsule.json"),
            "checkpoint_ack": checkpoints.pin(folder / "ack.json")})
        return proof, inputs_path, output, old, folder

    def test_saved_complete_pilot_is_checked_without_modifying_or_importing_models(self):
        proof, inputs, output, old, folder = self.pilot()
        originals = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        value = batch.verify_completed_pilot(proof, setup.sha_file(proof), self.runtime, 4)
        self.assertEqual((value["verified_calls"], value["object_id"]), (6, "fictional-object"))
        self.assertEqual(set(value["model_identities_without_input"]), set(batch.MODELS))
        self.assertEqual(originals, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "qwen_asr", "funasr", "whisper")))

    def test_changed_declaration_or_missing_actual_ack_is_rejected(self):
        proof, inputs, output, old, folder = self.pilot()
        with self.assertRaisesRegex(ValueError, "DECLARATION_CHANGED"):
            batch.verify_completed_pilot(proof, "f" * 64, self.runtime, 4)
        (folder / "ack.json").unlink()
        with self.assertRaises(FileNotFoundError):
            batch.verify_completed_pilot(proof, setup.sha_file(proof), self.runtime, 4)

    def test_changed_saved_native_or_unknown_call_cannot_qualify(self):
        proof, inputs, output, old, folder = self.pilot()
        (output / "results/qwen06/clip-000.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            batch.verify_completed_pilot(proof, setup.sha_file(proof), self.runtime, 4)

    def test_changed_current_model_configuration_is_not_only_an_input_change(self):
        proof, inputs, output, old, folder = self.pilot()
        with patch.dict(batch.CONFIGS, qwen06={**batch.CONFIGS["qwen06"], "max_new_tokens": 257}):
            with self.assertRaisesRegex(ValueError, "different identity"):
                batch.verify_completed_pilot(proof, setup.sha_file(proof), self.runtime, 4)

    def test_changed_current_numerical_source_refuses_saved_pilot(self):
        proof, inputs, output, old, folder = self.pilot()
        changed = self.root / "changed-current-source"
        shutil.copytree(old, changed)
        with (changed / "tools/active_asr_worker.py").open("a") as stream:
            stream.write("\n# fictional source mutation\n")
        with patch.object(batch, "ROOT", changed), self.assertRaisesRegex(ValueError, "NUMERICAL_SOURCE_CHANGED"):
            batch.verify_completed_pilot(proof, setup.sha_file(proof), self.runtime, 4)

    def args(self, phase="full", pilot=True):
        inputs, clips = self.inputs("new-inputs", 28)
        args = SimpleNamespace(inputs=inputs, runtime=self.runtime, output=self.root / "new-output",
            attempt="full-001", phase=phase, models=list(batch.MODELS), threads=4, budget_minutes=9,
            completed_pilot=self.root / "stand-in-proof.json" if pilot else None,
            completed_pilot_sha256="b" * 64 if pilot else None)
        return args, clips

    @contextmanager
    def controls(self, args, inherited=True, supervised=True):
        clock, calls = [1000.0], []
        binding = {"proof_sha256": "b" * 64, "input_manifest_sha256": "c" * 64,
                   "original_pcm_sha256": ["d" * 64], "derived_pcm_sha256": ["e" * 64]}
        @contextmanager
        def lease(*unused, **unused_keywords):
            yield SimpleNamespace(inherited=inherited, supervised=supervised)
        def worker(command, environment, log, runtime, remaining):
            calls.append({"model": command[command.index("--model") + 1],
                "ids": command[command.index("--ids") + 1].split(","), "remaining": remaining,
                "deadline": batch.read_json(args.output / "job.json")["deadline_unix"]})
            clock[0] += 10
            return 0  # This control fixture never launches a worker or model.
        with patch.object(batch.research_runtime, "check_interpreter"), \
             patch.object(batch, "verify_completed_pilot", return_value=binding), \
             patch.object(batch, "heavy_model_slot", lease), patch.object(batch, "run_worker", worker), \
             patch.object(batch.time, "time", side_effect=lambda: clock[0]):
            yield calls, binding

    def test_one_full_matrix_uses_three_workers_one_deadline_and_no_smoke(self):
        args, clips = self.args()
        with self.controls(args) as (calls, binding):
            self.assertEqual(batch.run(args), 0)
        self.assertEqual([row["model"] for row in calls], list(batch.MODELS))
        self.assertEqual([row["remaining"] for row in calls], [540, 530, 520])
        self.assertEqual([row["deadline"] for row in calls], [1540] * 3)
        self.assertTrue(all(row["ids"] == [clip["id"] for clip in clips] for row in calls))
        job = batch.read_json(args.output / "job.json")
        self.assertEqual(job["identity"]["max_calls"], 84)
        self.assertEqual(job["identity"]["completed_pilot"], binding)
        self.assertEqual([p.name for p in (args.output / "attempts").iterdir()], ["full-001"])

    def test_consumed_output_cannot_get_another_attempt_or_reset_deadline(self):
        args, clips = self.args()
        with self.controls(args) as (calls, binding):
            batch.run(args)
            before = (args.output / "job.json").read_bytes()
            args.attempt = "full-002"
            with self.assertRaisesRegex(ValueError, "unused output"):
                batch.run(args)
            self.assertEqual(len(calls), 3)
            self.assertEqual((args.output / "job.json").read_bytes(), before)

    def test_completed_pilot_requires_supervised_inherited_shared_slot(self):
        args, clips = self.args()
        with self.controls(args, supervised=False) as (calls, binding):
            with self.assertRaisesRegex(ValueError, "supervised inherited"):
                batch.run(args)
            self.assertFalse(args.output.exists())
            self.assertEqual(calls, [])

    def test_pilot_pcm_cannot_be_replayed_with_new_ids_or_reordered_manifest(self):
        args, clips = self.args()
        manifest = batch.read_json(args.inputs)
        manifest["clips"] = list(reversed(manifest["clips"]))
        manifest["clips"][0]["id"] = "clip-100"
        setup.save(args.inputs, manifest)
        for kind in ("original_pcm", "derived_pcm"):
            with self.subTest(kind=kind), self.controls(args) as (calls, binding):
                binding[kind + "_sha256"] = [clips[-1][kind]["sha256"]]
                with self.assertRaisesRegex(ValueError, "PCM must not be recognized again"):
                    batch.run(args)
                self.assertFalse(args.output.exists())
                self.assertEqual(calls, [])

    def test_unchanged_default_full_requires_current_input_smoke(self):
        args, clips = self.args(pilot=False)
        with self.controls(args) as (calls, binding):
            with self.assertRaisesRegex(ValueError, "same bounded smoke samples"):
                batch.run(args)
            self.assertEqual(calls, [])

    def test_unchanged_default_smoke_selects_first_and_last(self):
        args, clips = self.args(phase="smoke", pilot=False)
        with self.controls(args) as (calls, binding):
            self.assertEqual(batch.run(args), 0)
            self.assertTrue(all(row["ids"] == [clips[0]["id"], clips[-1]["id"]] for row in calls))
            self.assertNotIn("completed_pilot", batch.read_json(args.output / "job.json")["identity"])

    def test_verified_pilot_cannot_expand_the_fixed_model_time_or_thread_contract(self):
        args, clips = self.args()
        with self.controls(args) as (calls, binding):
            for field, value in (("phase", "smoke"), ("models", ["whisper"]), ("threads", 3), ("budget_minutes", 10)):
                original = getattr(args, field)
                setattr(args, field, value)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "nine minutes"):
                    batch.run(args)
                setattr(args, field, original)
            self.assertFalse(args.output.exists())
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
