#!/usr/bin/env python3
"""Offline stdlib-only tests; no package installation/model imports or inference."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bounded_asr_worker", HERE / "asr_worker.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def wav(value, frames=8):
    pcm = struct.pack("<h", value) * frames
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16,
                       1, 1, 16000, 32000, 2, 16, b"data", len(pcm)) + pcm


def make_input(root, count):
    root.mkdir()
    (root / "audio").mkdir()
    rows = []
    for index in range(count):
        raw = wav(index + 1)
        sha = hashlib.sha256(raw).hexdigest()
        (root / "audio" / (sha + ".wav")).write_bytes(raw)
        rows.append({"audio_id": "", "audio_path": "audio/" + sha + ".wav", "wav_sha256": sha})
    rows.sort(key=lambda row: row["wav_sha256"])
    for index, row in enumerate(rows, 1):
        row["audio_id"] = f"clip-{index:06d}"
    job = {"schema": "blind-source-screen32-job-v1", "clips": rows}
    (root / "job.json").write_bytes(w.canonical(job))
    return job


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        w.verify_sources()
        cls.adapters, cls.execution = w.import_helpers()

    def test_only_stdlib_imports(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "qwen_asr", "funasr", "transformers")))
        tree = ast.parse((HERE / "asr_worker.py").read_text())
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == "contract" for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == "contract" for alias in node.names) for node in ast.walk(tree)))

    def test_scientific_loader_bodies_unchanged(self):
        before = ast.parse((w.RETAINED / "core/asr_stage/adapters.py").read_text())
        after = ast.parse((HERE / "asr_worker.py").read_text())
        for name in ("load_qwen_after_approval", "load_sense_after_approval"):
            first = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
            second = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(first, include_attributes=False), ast.dump(second, include_attributes=False))

    def test_bounded_counts_and_old_six_unchanged(self):
        from asr6_contract import validate_job as old_validate_job
        for count in (0, 1, 6, 16):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "input"
                job = make_input(root, count)
                _, _, raw = w.bind_inputs(root)
                self.assertEqual(len(w.validate_decoder(raw, w.digest(raw))), count)
                old_job = copy.deepcopy(job)
                old_job["schema"] = "blind-asr-job-v1"
                encoded = w.canonical(old_job)
                if count == 6:
                    self.assertEqual(len(old_validate_job(encoded, w.digest(encoded))["clips"]), 6)
                else:
                    with self.assertRaises(ValueError):
                        old_validate_job(encoded, w.digest(encoded))
        with tempfile.TemporaryDirectory() as tmp:
            job = make_input(Path(tmp) / "input", 17)
            with self.assertRaises(ValueError):
                w.validate_job(w.canonical(job))

    def test_rejects_text_ids_order_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = make_input(Path(tmp) / "input", 2)
            bad = []
            changed = copy.deepcopy(job); changed["intended_text"] = "forbidden"; bad.append(changed)
            changed = copy.deepcopy(job); changed["clips"][0]["hotwords"] = []; bad.append(changed)
            changed = copy.deepcopy(job); changed["clips"][0]["audio_id"] = "screen32-001"; bad.append(changed)
            changed = copy.deepcopy(job); changed["clips"].reverse(); bad.append(changed)
            changed = copy.deepcopy(job); changed["clips"][1] = copy.deepcopy(changed["clips"][0]); bad.append(changed)
            for changed in bad:
                with self.assertRaises(ValueError):
                    w.validate_job(w.canonical(changed))
            with self.assertRaises(ValueError):
                w.validate_job(b'{"schema":"x","schema":"y","clips":[]}')

    def test_rejects_extra_alias_and_pcm_mutation(self):
        for kind in ("extra", "symlink", "bad_pcm"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "input"
                job = make_input(root, 1)
                path = root / job["clips"][0]["audio_path"]
                if kind == "extra":
                    (root / "targets.json").write_text("{}")
                elif kind == "symlink":
                    other = Path(tmp) / "wave.wav"; path.rename(other); path.symlink_to(other)
                else:
                    raw = bytearray(path.read_bytes()); raw[22] = 2
                    sha = w.digest(raw); path.unlink()
                    (root / "audio" / (sha + ".wav")).write_bytes(raw)
                    job["clips"][0].update(wav_sha256=sha, audio_path="audio/" + sha + ".wav")
                    (root / "job.json").write_bytes(w.canonical(job))
                with self.assertRaises(ValueError):
                    w.bind_inputs(root)

    def test_decoder_hash_and_descriptor_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"; make_input(root, 1)
            _, _, raw = w.bind_inputs(root)
            with self.assertRaises(ValueError):
                w.validate_decoder(raw, "0" * 64)
            decoder = json.loads(raw); decoder["clips"][0]["descriptor"]["raw_pcm16"]["sample_rate_hz"] = 8000
            raw = w.canonical(decoder)
            with self.assertRaises(ValueError):
                w.validate_decoder(raw, w.digest(raw))

    def test_scope_precedes_helpers(self):
        with patch.object(w, "verify_scope", side_effect=RuntimeError("scope")), patch.object(w, "import_helpers") as helpers:
            with self.assertRaises(RuntimeError):
                w.run("qwen06", Path("unused"), Path("unused"), Path("unused"))
            helpers.assert_not_called()

    def test_drift_pin_rejected(self):
        with patch.object(w, "RETAINED_PINS", {"model-locks.json": "0" * 64}):
            with self.assertRaises(ValueError):
                w.verify_sources()


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters, cls.execution = w.import_helpers()
        import setup_adapter
        cls.setup = setup_adapter

    def exercise(self, count, failure_index=None, model="qwen06", completeness="complete", load_failure=False, terminal_mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_input(root / "input", count); (root / "output").mkdir()
            calls = []
            def infer(_runtime, bound, manifest_sha):
                calls.append(bound.opaque_id)
                if len(calls) == failure_index:
                    raise RuntimeError("fictional failure")
                return {"raw_text": "fictional transcript", "completeness": completeness, "quality_flags": [] if completeness == "complete" else ["decoding_warning"]}, {"fictional": True}
            loader_name = "load_qwen_after_approval" if model == "qwen06" else "load_sense_after_approval"
            infer_name = "infer_qwen_once_after_approval" if model == "qwen06" else "infer_sense_once_after_approval"
            with patch.object(w, "verify_scope", return_value={"test_only": True}) as scope, \
                 patch.object(w, "prepare_caches"), \
                 patch.object(self.execution, "require_offline_flags", return_value={"test_only": True}), \
                 patch.object(self.setup, "verify_runtime") as setup, \
                 patch.object(w, loader_name, return_value=SimpleNamespace(receipt={"test_only": True}), side_effect=RuntimeError("fictional load failure") if load_failure else None) as load, \
                 patch.object(self.adapters, infer_name, side_effect=infer):
                summary = w.run(model, root / "input", root / "output", root / "runtime")
                folder = root / "output" / model
                rows = json.loads((folder / "outcomes.json").read_bytes())
                if count:
                    current_progress = json.loads((root / "output/progress.json").read_bytes())
                    self.assertEqual(set(current_progress), {"stage", "index", "started_monotonic"})
                    self.assertEqual(current_progress["stage"], "load" if load_failure else "cell")
                    self.assertEqual(current_progress["index"], len(calls))
                    self.assertFalse(list((root / "output").glob(".progress-*.tmp")))
                raw_freeze = json.loads((folder / "raw-freeze.json").read_bytes())
                for name, sha in raw_freeze["files"].items():
                    self.assertEqual(w.digest((folder / name).read_bytes()), sha)
                for row in rows:
                    if row["execution_receipt_sha256"]:
                        receipt_raw = (folder / (row["opaque_id"] + ".receipt.json")).read_bytes()
                        self.assertEqual(w.digest(receipt_raw), row["execution_receipt_sha256"])
                        receipt = json.loads(receipt_raw)
                        self.assertEqual(w.digest((folder / (row["opaque_id"] + ".decoder.json")).read_bytes()), receipt["decoder_evidence_sha256"])
                if terminal_mutation:
                    terminal_mutation(folder)
                preserved = {path.name: path.read_bytes() for path in folder.iterdir()}
                terminal = w.terminal_outcomes(root / "input", root / "output", model)
                terminal_rows = json.loads((root / "output/terminal-outcomes.json").read_bytes())
                self.assertEqual(preserved, {path.name: path.read_bytes() for path in folder.iterdir()})
                self.assertFalse(terminal["snapshots_used"])
                self.assertFalse(terminal["human_gold"])
                self.assertEqual(terminal["requested_clips"], count)
                if not terminal_mutation:
                    self.assertEqual(terminal["success"], summary["success"])
                    self.assertEqual(terminal["attempted"], summary["attempted"])
                with self.assertRaises(ValueError):
                    w.terminal_outcomes(root / "input", root / "output", model)
                self.last_terminal = (terminal, terminal_rows)
                with self.assertRaises((FileExistsError, ValueError)):
                    w.run(model, root / "input", root / "output", root / "runtime")
                self.assertFalse(raw_freeze["labels_joined"])
                self.assertFalse(summary["human_gold"])
                self.assertFalse(summary["training_admitted"])
                self.assertEqual(summary["context"], "")
                self.assertEqual(summary["hotwords"], [])
                self.assertEqual(setup.call_count, 0 if count == 0 else 1)
                self.assertEqual(load.call_count, 0 if count == 0 else 1)
                self.assertGreaterEqual(scope.call_count, 2 + len(calls))
                return summary, rows, calls

    def test_zero_does_not_load_setup_or_models(self):
        summary, rows, calls = self.exercise(0)
        self.assertEqual(summary["status"], "empty_generated_subset")
        self.assertEqual((rows, calls), ([], []))
        self.assertEqual(summary["attempted"], 0)

    def test_sixteen_single_calls_two_families_separately(self):
        for model in ("qwen06", "sensevoice"):
            with self.subTest(model=model):
                summary, rows, calls = self.exercise(16, model=model)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["attempted"], 16)
                self.assertEqual(calls, [f"clip-{index:06d}" for index in range(1, 17)])
                self.assertTrue(all(row["status"] == "success" for row in rows))

    def test_load_failure_preserves_zero_attempts(self):
        summary, rows, calls = self.exercise(16, load_failure=True)
        self.assertEqual(summary["attempted"], 0)
        self.assertEqual(calls, [])
        self.assertTrue(all(row["status"] == "not_run" for row in rows))
        self.assertEqual(summary["status"], "stopped")

    def test_unknown_quality_preserves_raw_without_gold(self):
        summary, rows, calls = self.exercise(1, model="sensevoice", completeness="unknown")
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["completeness"], "unknown")
        self.assertEqual(rows[0]["raw_text"], "fictional transcript")
        self.assertEqual(rows[0]["quality_flags"], ["decoding_warning"])
        self.assertFalse(summary["human_gold"])

    def test_terminal_interrupted_receipt_ignores_success_snapshot(self):
        def interrupt(folder):
            (folder / "raw-freeze.json").unlink()
            (folder / "clip-000002.receipt.json").unlink()
        self.exercise(2, terminal_mutation=interrupt)
        summary, rows = self.last_terminal
        self.assertEqual([row["status"] for row in rows], ["success", "failed_no_retry"])
        self.assertIsNone(rows[1]["raw_text"])
        self.assertEqual(summary["attempted"], 2)

    def test_terminal_detects_frozen_receipt_or_evidence_tamper(self):
        for filename in ("clip-000001.receipt.json", "clip-000001.decoder.json", "clip-000001.started.json"):
            with self.subTest(filename=filename):
                def tamper(folder):
                    path = folder / filename
                    value = json.loads(path.read_bytes()); value["wav_sha256"] = "0" * 64
                    path.write_bytes(w.canonical(value))
                self.exercise(1, terminal_mutation=tamper)
                _, rows = self.last_terminal
                self.assertEqual(rows[0]["status"], "failed_no_retry")
                self.assertIsNone(rows[0]["raw_text"])

    def test_terminal_deleted_started_evidence_remains_consumed(self):
        def missing(folder):
            for suffix in ("started", "receipt", "decoder"):
                (folder / ("clip-000001." + suffix + ".json")).unlink()
        self.exercise(1, terminal_mutation=missing)
        _, rows = self.last_terminal
        self.assertEqual(rows[0]["status"], "failed_no_retry")

    def test_terminal_ignores_snapshot_only_transcript(self):
        def missing(folder):
            (folder / "raw-freeze.json").unlink()
            for suffix in ("started", "receipt", "decoder"):
                (folder / ("clip-000001." + suffix + ".json")).unlink()
        self.exercise(1, terminal_mutation=missing)
        _, rows = self.last_terminal
        self.assertEqual(rows[0]["status"], "not_run")
        self.assertIsNone(rows[0]["raw_text"])

    def test_failure_consumes_call_and_stops(self):
        summary, rows, calls = self.exercise(16, failure_index=3)
        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual([row["status"] for row in rows], ["success", "success", "error"] + ["not_run"] * 13)
        self.assertEqual(summary["status"], "stopped")


class TerminalOnlyTests(unittest.TestCase):
    def test_never_started_and_zero_need_no_helpers(self):
        for count in (0, 16):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); make_input(root / "input", count)
                with patch.object(w, "import_helpers", side_effect=AssertionError("No helpers")), \
                     patch.object(w, "verify_scope", side_effect=AssertionError("Host finalizer is not model execution")):
                    summary = w.terminal_outcomes(root / "input", root / "output", "sensevoice")
                self.assertEqual(summary["requested_clips"], count)
                self.assertEqual(summary["attempted"], 0)
                self.assertEqual(summary["not_run"], count)
                self.assertFalse(summary["human_gold"])
                self.assertFalse(summary["training_admitted"])

    def test_cache_creation_requires_scope(self):
        with patch.object(w, "verify_scope", side_effect=RuntimeError("scope")), patch.object(w, "Path") as paths:
            with self.assertRaises(RuntimeError):
                w.prepare_caches()
            paths.assert_not_called()

    def test_caches_use_known_scratch_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            def path(value):
                self.assertEqual(value, "/scratch")
                return Path(tmp)
            with patch.object(w, "verify_scope", return_value={"test_only": True}), patch.object(w, "Path", side_effect=path), patch.dict(w.os.environ, {}, clear=False):
                w.prepare_caches()
                self.assertEqual({p.name for p in Path(tmp).iterdir()}, {key.lower() for key in w.CACHE_KEYS} | {"home"})
                for key in w.CACHE_KEYS:
                    self.assertEqual(w.os.environ[key], str(Path(tmp) / key.lower()))


if __name__ == "__main__":
    unittest.main()
