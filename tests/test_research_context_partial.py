#!/usr/bin/env python3
"""Counterexamples for partial export; fixture parsers are not real audio proof."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("context_partial", ROOT / "tools/research_context_partial.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Partial(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.g, self.n = self.root / "generation", self.root / "native"
        self.cell = {"cell_id": "cell-1", "source_family_id": "family-1", "seed": 7, "prompt": "fixture intent"}
        self.ref = {"reference_id": "ref-1", "reference_text_status": "dataset-provided", "wav_sha256": "1" * 64, "pcm_sha256": "2" * 64}
        self.generation = {"max_new_tokens": 256}
        self.call = {"schema": "active-tts-generation-call-v1", "cell_id": "cell-1", "source_family_id": "family-1",
            "reference_id": "ref-1", "reference_wav_sha256": "1" * 64, "reference_pcm_sha256": "2" * 64,
            "reference_text_status": "dataset-provided", "reference_mode": "x-vector-only", "reference_text_passed": False,
            "seed": 7, "attempt": 1, "generation": self.generation, "intended_text": "fixture intent", "reference_prompt_seconds": 0.3}
        self.weak = SimpleNamespace(_d1_started=lambda *args: None, float_wav_pcm=lambda raw: b"\0" * 16,
                                   _d1_native_array=lambda *args: None, _d1_completed=lambda *args: None)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else m.canonical(value))

    def started(self):
        self.save(self.g / "cells/cell-1/started.json", {})
        self.save(self.n / "cell-1/call.json", self.call)

    def returned(self):
        self.started()
        f = self.n / "cell-1"
        raw, array = b"fixture wav bytes", b"fixture array bytes"
        self.save(f / "returned-native.wav", raw)
        self.save(f / "returned-array.npy", array)
        self.save(f / "native.json", {"schema": "active-tts-native-return-v1", "array": m.record(f / "returned-array.npy"),
            "wav": m.record(f / "returned-native.wav"), "all_finite": True, "technical_gate_applied": False, "sample_rate_hz": 24000})
        self.save(f / "returned.json", {"generation_seconds": 1.2, "total_seconds": 1.3,
                                       "native_stop_reason": "not_exposed_by_voice_clone_API"})
        self.save(self.g / "cells/cell-1/failed.json", {"schema": "kws-bulk-clone-failure-v1", "cell_id": "cell-1",
            "status": "FAILED_NO_RETRY", "error_type": "ValueError", "training_admitted": False})

    def classify(self):
        return m.classify_cell(self.g, self.n, self.cell, self.ref, "3" * 64, self.generation, self.weak)

    def timeout(self, **changes):
        binding = {"recipe_sha256": "r", "queue_sha256": "q", "job_definition_sha256": "d", "job_id": "generate-001"}
        claim = {**binding, "automatic_retry": False, "attempt": 1, "allowed_runner_nanoseconds": 600_000_000_000}
        result = {**binding, "schema": "kws-local-research-job-result-v1", "status": "timed_out", "exit_code": -9,
            "reason": "actual_runner_deadline", "automatic_retry": False, "actual_duration_known": True, "claim_sha256": "c",
            "allowed_runner_nanoseconds": 600_000_000_000, "actual_elapsed_ns": 600_000_000_123, "deadline_overrun_nanoseconds": 123}
        result.update(changes)
        return m.check_timeout(result, claim, recipe_sha="r", queue_sha="q", claim_sha="c", definition_sha="d")

    def test_never_claimed_has_no_audio(self):
        row, raw = self.classify()
        self.assertEqual(row["status"], "never-claimed")
        self.assertIsNone(raw)

    def test_consumed_unknown_has_no_audio_or_retry(self):
        self.started()
        row, raw = self.classify()
        self.assertEqual(row["status"], "consumed-unknown")
        self.assertFalse(row["may_retry"])
        self.assertIsNone(raw)

    def test_complete_failed_native_is_observable_but_not_promoted(self):
        self.returned()
        row, raw = self.classify()
        self.assertEqual(row["status"], "failed-native")
        self.assertEqual(raw, b"fixture wav bytes")

    def test_partial_unknown_with_donated_wav_is_rejected(self):
        self.started()
        self.save(self.n / "cell-1/returned-native.wav", b"donated")
        with self.assertRaisesRegex(ValueError, "FILE_ROSTER"):
            self.classify()

    def test_return_without_claim_is_rejected(self):
        self.save(self.n / "cell-1/returned-native.wav", b"unclaimed")
        with self.assertRaisesRegex(ValueError, "UNOWNED"):
            self.classify()

    def test_changed_native_array_rejected_before_parser(self):
        self.returned()
        self.save(self.n / "cell-1/returned-array.npy", b"changed")
        with self.assertRaisesRegex(ValueError, "BYTE_BINDING"):
            self.classify()

    def test_changed_native_call_intent_rejected(self):
        self.started()
        self.save(self.n / "cell-1/call.json", {**self.call, "intended_text": "replacement"})
        with self.assertRaisesRegex(ValueError, "CALL_CHANGED"):
            self.classify()

    def test_failed_native_training_promotion_rejected(self):
        self.returned()
        p = self.g / "cells/cell-1/failed.json"
        self.save(p, {**m.load(p), "training_admitted": True})
        with self.assertRaisesRegex(ValueError, "FAILURE_CHANGED"):
            self.classify()

    def test_authentic_timeout_shape(self):
        self.timeout()

    def test_completed_or_altered_timeout_cannot_enter(self):
        for change in ({"status": "completed"}, {"exit_code": 0}, {"recipe_sha256": "other"},
                       {"automatic_retry": True}, {"actual_duration_known": False},
                       {"deadline_overrun_nanoseconds": 0}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "AUTHENTIC_TIMEOUT"):
                self.timeout(**change)

    def test_symlink_is_rejected(self):
        target = self.root / "target"
        target.write_bytes(b"x")
        alias = self.root / "alias"
        alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "SYMLINK"):
            m.read(alias)

    def test_export_does_not_overwrite(self):
        with self.assertRaisesRegex(ValueError, "CREATE_ONLY"):
            m.export({}, self.root)

    def test_output_manifest_excludes_itself_and_replays(self):
        folder = self.root / "export"
        proof = {"counts": {"completed": 12, "failed-native": 1, "consumed-unknown": 1, "never-claimed": 2}}
        # Only envelope mechanics are mocked. This is not native evidence.
        with patch.object(m, "collect", return_value=(proof, b"fixture archive")):
            m.export({}, folder)
            self.assertNotIn("MANIFEST.json", m.load(folder / "MANIFEST.json"))
            result = m.verify(folder, m.sha(m.read(folder / "proof.json")))
        self.assertEqual(result["status"], "verified")

    def test_import_has_no_model_side_effect(self):
        self.assertFalse(any(name in sys.modules for name in m.ML_MODULES))


if __name__ == "__main__":
    unittest.main()
