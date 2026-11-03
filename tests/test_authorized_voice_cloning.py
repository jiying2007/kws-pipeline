#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import authorized_voice_clone_batch as clone  # noqa: E402
import voice_clone_effect_audit as effect  # noqa: E402
import kws_data_bulk_effect_boundary as bulk_boundary  # noqa: E402
import prioritize_kws_data_feedback as feedback  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def make_wav(path: Path, duration_s: int = 3) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x10" * (16000 * duration_s))


STUB = '''import pathlib, sys, wave
reference, txt, output, cache, seed = sys.argv[1:]
assert pathlib.Path(reference).is_file() and txt and seed.isdecimal()
cache = pathlib.Path(cache)
cache.mkdir(parents=True, exist_ok=True)
(cache / 'prompt.cache').write_text('reused speaker prompt')
with (cache / 'calls.txt').open('a') as f: f.write(txt + ':' + seed + '\\n')
with wave.open(output, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b'\\x11\\x22' * 8000)
'''


class AuthorizedCloningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference = self.root / "reference.wav"
        make_wav(self.reference)
        self.rights = self.root / "permission.json"
        self.license = {"schema_version": 1, "source_family_id": "family-A",
                        "voice_cloning_permitted": True,
                        "synthetic_kws_training_permitted": True,
                        "covered_speaker_ids": ["anonymous-1"],
                        "authorization_basis": "explicit source-level voice reuse permission"}
        dump(self.rights, self.license)
        self.stub = self.root / "stub.py"
        self.stub.write_text(STUB, encoding="utf-8")
        self.plan = self.root / "plan.json"
        self.output = self.root / "private-output"
        self.payload = {"schema_version": 1, "policy": clone.POLICY,
                        "campaign_id": "first-campaign", "max_attempts": 6,
                        "provider": {"kind": "offline-voice-clone",
                                     "executable": {"path": sys.executable, "sha256": sha(Path(sys.executable))},
                                     "assets": [{"role": "stub", "path": "stub.py", "sha256": sha(self.stub)}],
                                     "argv_template": ["{executable}", "{asset:stub}", "{reference}",
                                                       "{text}", "{output}", "{prompt_cache}", "{seed}"],
                                     "timeout_seconds": 30},
                        "speakers": [{"speaker_id": "anonymous-1", "source_family_id": "family-A",
                                      "reference": {"path": "reference.wav", "sha256": sha(self.reference)},
                                      "rights_evidence": {"path": "permission.json", "sha256": sha(self.rights)},
                                      "reference_text": "读给我听"}],
                        "phrases": [{"id": "k1", "text": "你好小窝", "intent": "wake"},
                                    {"id": "n1", "text": "你好小屋", "intent": "near_miss"},
                                    {"id": "n2", "text": "今天的天气", "intent": "ordinary_negative"}],
                        "seeds": [0, 1]}
        self.save_plan()

    def save_plan(self) -> None:
        dump(self.plan, self.payload)

    def loaded(self) -> dict:
        return clone.load_campaign(self.plan)

    def test_new_active_workflow_is_exactly_retained(self) -> None:
        manifest = json.loads((ROOT / "research/consolidation/source-retention-2026-10-07.json")
                              .read_text(encoding="utf-8"))
        workflow = ROOT / ".github/workflows/authorized-voice-clone-contracts.yml"
        path = str(workflow.relative_to(ROOT))
        self.assertIn(path, manifest["allowed_new_active_workflows"])
        records = [r for r in manifest["current_added_active_workflows"] if r["path"] == path]
        self.assertEqual(len(records), 1)
        raw = workflow.read_bytes()
        row = records[0]
        self.assertEqual(row["mode"], "100644")
        self.assertEqual(row["bytes"], len(raw))
        self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(row["git_blob_sha1"], hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest())

    def test_dry_run_does_not_invoke_or_write(self) -> None:
        self.assertEqual(clone.main(["--plan", str(self.plan), "--output-root", str(self.output)]), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.loaded()["recordings"], 6)

    def test_bounded_generation_and_idempotent_readback(self) -> None:
        campaign = self.loaded()
        report = clone.execute(campaign, self.output, workers=2)
        self.assertEqual(report["recordings"], 6)
        self.assertFalse(report["qualification_allowed"])
        self.assertFalse(report["training_admitted"])
        rows = [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual({x["requested_intent"] for x in rows}, set(clone.KINDS))
        self.assertEqual({x["speaker_id"] for x in rows}, {"anonymous-1"})
        self.assertEqual({x["reference_audio_sha256"] for x in rows}, {sha(self.reference)})
        calls = list((self.output / "private-prompt-cache").rglob("calls.txt"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].read_text().splitlines()), 6)
        self.assertEqual(clone.execute(campaign, self.output, workers=1), report)
        self.assertEqual(len(calls[0].read_text().splitlines()), 6)

    def test_rejects_changed_completed_audio(self) -> None:
        campaign = self.loaded()
        clone.execute(campaign, self.output, workers=1)
        audio = sorted((self.output / "audio").rglob("*.wav"))[0]
        audio.write_bytes(audio.read_bytes() + b"extra")
        with self.assertRaisesRegex(RuntimeError, "completed output WAV changed"):
            clone.execute(campaign, self.output, workers=1)

    def test_tampered_receipt_cannot_claim_training_or_qualification(self) -> None:
        campaign = self.loaded()
        clone.execute(campaign, self.output, workers=1)
        receipt = sorted((self.output / "attempts").rglob("*.receipt.json"))[0]
        item = json.loads(receipt.read_text(encoding="utf-8"))
        item["qualification_allowed"] = True
        dump(receipt, item)
        with self.assertRaisesRegex(RuntimeError, "receipt metadata/output mismatch"):
            clone.execute(campaign, self.output, workers=1)

    def test_concurrent_attempt_claim_has_exactly_one_winner(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        marker = self.root / "attempts" / "one.intent.json"

        def claim() -> str:
            try:
                clone.claim_invocation_once(marker, '{"campaign":"single"}\\n')
            except ValueError as exc:
                self.assertIn("already claimed", str(exc))
                return "blocked"
            return "winner"

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _index: claim(), range(16)))
        self.assertEqual(results.count("winner"), 1)
        self.assertEqual(results.count("blocked"), 15)
        self.assertEqual(marker.read_text(), '{"campaign":"single"}\\n')

    def test_incomplete_attempt_is_consumed_not_retried(self) -> None:
        campaign = self.loaded()
        marker = self.output / "attempts" / "anonymous-1" / "anonymous-1-k1-s0.intent.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"schema_version": 1, "campaign_sha256": campaign["sha256"],
                                      "item_id": "anonymous-1-k1-s0", "reference_audio_sha256": sha(self.reference),
                                      "rights_evidence_sha256": sha(self.rights), "requested_intent": "wake",
                                      "seed": 0}, sort_keys=True) + "\n")
        with self.assertRaisesRegex(RuntimeError, "unknown; no automatic retry"):
            clone.execute(campaign, self.output, workers=1)
        self.assertFalse(list((self.output / "audio").rglob("*.wav")))

    def test_rights_permission_and_scope_are_fail_closed(self) -> None:
        for patch in ({"voice_cloning_permitted": False},
                      {"synthetic_kws_training_permitted": False},
                      {"covered_speaker_ids": ["other"]},
                      {"source_family_id": "unrelated"},
                      {"authorization_basis": ""}):
            with self.subTest(patch=patch):
                rights = {**self.license, **patch}
                dump(self.rights, rights)
                self.payload["speakers"][0]["rights_evidence"]["sha256"] = sha(self.rights)
                self.save_plan()
                with self.assertRaises(ValueError):
                    self.loaded()
        self.payload["speakers"][0]["rights_evidence"]["sha256"] = "0" * 64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "file hash mismatch"):
            self.loaded()

    def test_rejects_duplicate_reference_as_independent_speaker(self) -> None:
        duplicate = dict(self.payload["speakers"][0])
        duplicate["speaker_id"] = "anonymous-2"
        self.payload["speakers"].append(duplicate)
        self.payload["max_attempts"] = 12
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "same reference reused"):
            self.loaded()

    def test_normalizes_punctuation_before_negative_keyword_check(self) -> None:
        self.payload["phrases"][1]["text"] = "你好，小窝呀！"
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "negative intent contains"):
            self.loaded()

    def test_duplicate_pcm_with_different_file_bytes_does_not_add_speaker(self) -> None:
        other = self.root / "rewrapped-reference.wav"
        other.write_bytes(self.reference.read_bytes() + b"extra-container-bytes")
        duplicate = dict(self.payload["speakers"][0])
        duplicate["speaker_id"] = "anonymous-2"
        duplicate["reference"] = {"path": str(other), "sha256": sha(other)}
        self.payload["speakers"].append(duplicate)
        self.payload["max_attempts"] = 12
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "same decoded PCM reused"):
            self.loaded()

    def test_blocks_wrong_negative_and_unbounded_budget(self) -> None:
        self.payload["phrases"][1]["text"] = "你好小窝呀"
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "negative intent contains"):
            self.loaded()
        self.payload["phrases"][1]["text"] = "你好小屋"
        self.payload["max_attempts"] = 0
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "explicit budget"):
            self.loaded()

    def test_rejects_unknown_command_placeholders_and_unpinned_assets(self) -> None:
        self.payload["provider"]["argv_template"].append("{untrusted_path}")
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "unrecognized placeholder"):
            self.loaded()
        self.payload["provider"]["argv_template"].pop()
        self.payload["provider"]["assets"][0]["sha256"] = "a" * 64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "file hash mismatch"):
            self.loaded()

    def test_protects_repository_from_private_audio(self) -> None:
        campaign = self.loaded()
        with self.assertRaisesRegex(ValueError, "cannot be inside"):
            clone.execute(campaign, ROOT / "private-voice-clones", workers=1)


    def test_effect_audit_requires_real_generated_audio_and_keeps_gates_closed(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        summary = effect.audit(self.output)
        self.assertEqual(summary["total_generated"], 6)
        self.assertEqual(summary["screened"], 0)
        self.assertEqual(summary["source_family_roster"], 1)
        self.assertEqual(summary["unique_output_pcm"], 1)  # fake provider repeats all clips
        self.assertIn("THREE_ASR_SCREENING_INCOMPLETE", summary["gaps"])
        self.assertIn("MISSING_PRODUCT_KEYWORD_GENERATION", summary["gaps"])
        self.assertFalse(summary["label_authentication_verified"])
        self.assertFalse(summary["training_admitted"])
        self.assertFalse(summary["qualification_allowed"])
        self.assertFalse(summary["shipping_approved"])

    def test_effect_audit_three_asr_text_matches_are_never_training_authority(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        rows = [json.loads(x) for x in (self.output / "manifest.jsonl").read_text().splitlines()]
        screen = self.root / "screen.jsonl"
        with screen.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({
                    "item_id": row["item_id"], "file_sha256": row["file_sha256"],
                    "observations": [
                        {"model_family": f"diagnostic-{i}", "raw_text": row["text_requested"]}
                        for i in range(3)]}, ensure_ascii=False) + "\n")
        summary = effect.audit(self.output, screen)
        self.assertEqual(summary["screened"], 6)
        self.assertEqual(summary["screen_counts"][effect.SCREEN_MATCH], 6)
        self.assertFalse(summary["label_authentication_verified"])
        self.assertFalse(summary["training_admitted"])
        self.assertEqual(summary["conclusion"], "SCREENING_ONLY_NOT_TRAINING_OR_QUALIFICATION")

    def test_effect_audit_negative_wake_confusion_is_review_not_negative_label(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        rows = [json.loads(x) for x in (self.output / "manifest.jsonl").read_text().splitlines()]
        target = next(r for r in rows if r["requested_intent"] == "near_miss")
        screen = self.root / "screen.jsonl"
        screen.write_text(json.dumps({
            "item_id": target["item_id"], "file_sha256": target["file_sha256"],
            "observations": [
                {"model_family": "f1", "raw_text": "你好小屋"},
                {"model_family": "f2", "raw_text": "你好小窝"},
                {"model_family": "f3", "raw_text": "你好小屋"}]}, ensure_ascii=False) + "\n")
        summary = effect.audit(self.output, screen)
        self.assertEqual(summary["screen_counts"]["NEGATIVE_WAKE_CONFUSION_REVIEW"], 1)
        self.assertIn("NEGATIVE_WAKE_CONFUSION_NEEDS_REVIEW", summary["gaps"])
        self.assertFalse(summary["training_admitted"])

    def test_effect_audit_detects_changed_pcm_and_screen_audio_mismatch(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        rows = [json.loads(x) for x in (self.output / "manifest.jsonl").read_text().splitlines()]
        screen = self.root / "screen.jsonl"
        row = rows[0]
        screen.write_text(json.dumps({"item_id": row["item_id"], "file_sha256": "a" * 64,
            "observations": [{"model_family": f"f{i}", "raw_text": row["text_requested"]}
                             for i in range(3)]}) + "\n")
        with self.assertRaisesRegex(ValueError, "bound to a different WAV"):
            effect.audit(self.output, screen)
        wav = self.output / row["audio"]
        wav.write_bytes(wav.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "generated PCM/file identity mismatch"):
            effect.audit(self.output)

    def test_effect_audit_disallows_cross_speaker_identical_pcm(self) -> None:
        reference2 = self.root / "reference2.wav"
        with wave.open(str(reference2), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\x22\x33" * (16000 * 3))
        license2 = self.root / "license2.json"
        dump(license2, {**self.license, "covered_speaker_ids": ["anonymous-2"]})
        self.payload["speakers"].append({
            "speaker_id": "anonymous-2", "source_family_id": "family-A",
            "reference": {"path": "reference2.wav", "sha256": sha(reference2)},
            "rights_evidence": {"path": "license2.json", "sha256": sha(license2)},
            "reference_text": "新的参考音频"})
        self.payload["max_attempts"] = 12
        self.save_plan()
        clone.execute(self.loaded(), self.output, workers=1)
        with self.assertRaisesRegex(ValueError, "identical decoded PCM duplicated across original"):
            effect.audit(self.output)

    def test_effect_audit_forged_authority_and_path_traversal_fail(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        manifest = self.output / "manifest.jsonl"
        summary_path = self.output / "summary.json"
        original = [json.loads(x) for x in manifest.read_text().splitlines()]
        summary = json.loads(summary_path.read_text())
        for field, value, error in (
            ("training_admitted", True, "unauthorized authority claim"),
            ("audio", "audio/../../outside.wav", "WAV path escapes")):
            rows = [dict(row) for row in original]
            rows[0][field] = value
            manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                        for row in rows), encoding="utf-8")
            summary["manifest_sha256"] = sha(manifest)
            dump(summary_path, summary)
            with self.assertRaisesRegex(ValueError, error):
                effect.audit(self.output)

    def test_effect_audit_rejects_duplicate_asr_families(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        row = json.loads((self.output / "manifest.jsonl").read_text().splitlines()[0])
        screen = self.root / "screen.jsonl"
        screen.write_text(json.dumps({"item_id": row["item_id"],
            "file_sha256": row["file_sha256"],
            "observations": [{"model_family": "same", "raw_text": row["text_requested"]}] * 3}) + "\n")
        with self.assertRaisesRegex(ValueError, "families must be named and distinct"):
            effect.audit(self.output, screen)


    def make_bulk_data_fixture(self) -> tuple[dict, dict, Path]:
        root = self.root / "bulk-generated"
        root.mkdir(exist_ok=True)
        ref = {"reference_id": "ref-one", "speaker_group_id": "speaker-one",
               "source_family_id": "family-one", "role": "train-candidate",
               "wav_sha256": "e" * 64}
        cells = []
        phrases = (("bulk-a111", "你好小窝", "positive-intent", 1),
                   ("bulk-b222", "你好小屋", "hard-negative-intent", None))
        for cell_id, text, kind, keyword in phrases:
            cells.append({"cell_id": cell_id, "reference_id": ref["reference_id"],
                          "speaker_group_id": ref["speaker_group_id"],
                          "source_family_id": ref["source_family_id"],
                          "role": ref["role"], "prompt": text, "intended_kind": kind,
                          "intended_keyword_id": keyword, "label_authority": "intended_only",
                          "synthetic": True, "training_admitted": False,
                          "qualification_allowed": False, "seed": 1000 + len(cells)})
        plan = {"schema": bulk_boundary.PLAN_SCHEMA, "model_id": bulk_boundary.MODEL,
                "references": [ref], "cells": cells,
                "budgets": {"max_total_calls": 2, "max_calls_per_invocation": 64,
                            "max_retries": 0, "max_paid_usd": 0},
                "source_rights_authenticated": False,
                "actual_spoken_labels_verified": False, "training_admitted": False,
                "qualification_allowed": False, "shipping_approved": False}
        dump(root / "plan-lock.json", {"schema": "kws-bulk-clone-execution-lock-v1",
            "plan_sha256": bulk_boundary.canonical_sha(plan),
            "model_id": bulk_boundary.MODEL, "training_admitted": False})
        statuses = ("MACHINE_POSITIVE_CANDIDATE", "NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT")
        rows = []
        for idx, (cell, status) in enumerate(zip(cells, statuses)):
            folder = root / "cells" / cell["cell_id"]
            folder.mkdir(parents=True, exist_ok=True)
            dump(folder / "started.json", {
                "schema": bulk_boundary.START_RECEIPT, "cell_id": cell["cell_id"],
                "reference_id": cell["reference_id"],
                "reference_wav_sha256": ref["wav_sha256"],
                "plan_sha256": bulk_boundary.canonical_sha(plan),
                "source_family_id": ref["source_family_id"],
                "seed": cell["seed"], "attempts": 1,
                "training_admitted": False})
            wav = folder / "audio-16k.wav"
            with wave.open(str(wav), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(bytes((idx+1, 24)) * 16000)
            from corpus_identity import inspect_pcm16_wav
            meta = inspect_pcm16_wav(wav)
            dump(folder / "completed.json", {
                "schema": bulk_boundary.TTS_RECEIPT, "cell_id": cell["cell_id"],
                "status": "GENERATED_PENDING_ASR",
                "source_family_id": cell["source_family_id"],
                "intended_kind": cell["intended_kind"],
                "audio": {"wav_sha256": meta["file_sha256"],
                          "pcm_sha256": meta["pcm_sha256"], "audio_frames": meta["frames"],
                          "wav_bytes": wav.stat().st_size},
                "training_admitted": False, "qualification_allowed": False,
                "shipping_approved": False})
            rows.append({"cell_id": cell["cell_id"],
                         "reference_id": cell["reference_id"],
                         "source_family_id": cell["source_family_id"],
                         "role": cell["role"],
                         "intended_kind": cell["intended_kind"],
                         "asr_completed": 3, "status": status,
                         "training_admitted": False})
        screen = {"schema": bulk_boundary.SCREEN_SCHEMA,
                  "plan_sha256": bulk_boundary.canonical_sha(plan),
                  "adapter_lock_sha256": "a" * 64, "rows": rows,
                  "counts": {status: 1 for status in statuses},
                  "human_gold": False, "rights_authenticated": False,
                  "training_admitted": False, "qualification_allowed": False,
                  "shipping_approved": False}
        return plan, screen, root

    def test_bulk_effect_bridge_measures_real_pcm_and_actual_data_gaps(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        report = bulk_boundary.measure(plan, screen, root)
        self.assertEqual(report["total_planned"], 2)
        self.assertEqual(report["total_generated_audio_verified"], 2)
        self.assertEqual(report["unique_pcm_verified"], 2)
        self.assertEqual(report["screen_by_keyword"]["1"]["MACHINE_POSITIVE_CANDIDATE"], 1)
        self.assertEqual(report["screen_by_challenge"][
            "near_miss:homophone_wu_requested"]["NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT"], 1)
        self.assertIn("HARD_NEGATIVE_WAKE_CONFUSION_NEEDS_DEVELOPMENT_REVIEW", report["gaps"])
        self.assertFalse(report["training_admitted"])
        self.assertFalse(report["asr_source_authenticated"])
        self.assertFalse(report["shipping_approved"])
        self.assertTrue(all(row["generated_audio"]["pcm_sha256"] for row in report["rows"]))
        plan_path, screen_path = self.root / "bulk-plan.json", self.root / "bulk-screen.json"
        dump(plan_path, plan)
        dump(screen_path, screen)
        output = self.root / "bulk-report-private.json"
        self.assertEqual(bulk_boundary.main(["--plan", str(plan_path),
            "--screen", str(screen_path), "--generated-root", str(root),
            "--output", str(output)]), 0)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            bulk_boundary.main(["--plan", str(plan_path), "--screen", str(screen_path),
                "--generated-root", str(root), "--output", str(output)])

    def test_bulk_effect_bridge_rejects_forged_machine_positive_and_counts(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        screen["rows"][0]["status"] = "MACHINE_NEGATIVE_CANDIDATE"
        screen["counts"] = {"MACHINE_NEGATIVE_CANDIDATE": 1,
                            "NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT": 1}
        with self.assertRaisesRegex(ValueError, "MACHINE_NEGATIVE_KIND"):
            bulk_boundary.measure(plan, screen, root)
        screen["rows"][0]["status"] = "MACHINE_POSITIVE_CANDIDATE"
        with self.assertRaisesRegex(ValueError, "SCREEN_COUNTS_OR_MEMBERSHIP_DRIFT"):
            bulk_boundary.measure(plan, screen, root)

    def test_bulk_effect_bridge_rejects_changed_source_wav_and_plan_drift(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        wav = root / "cells" / "bulk-a111" / "audio-16k.wav"
        wav.write_bytes(wav.read_bytes() + b"drift")
        with self.assertRaisesRegex(ValueError, "GENERATION_WAV_OR_PCM_CHANGED"):
            bulk_boundary.measure(plan, screen, root)
        plan, screen, root = self.make_bulk_data_fixture()
        plan["cells"][0]["prompt"] = "你好小屋"
        with self.assertRaisesRegex(ValueError, "BULK_CELL_INTENT_OR_LINEAGE"):
            bulk_boundary.measure(plan, screen, root)

    def test_bulk_effect_bridge_preserves_source_family_split(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        ref = dict(plan["references"][0], reference_id="ref-other",
                   role="development-only")
        plan["references"].append(ref)
        with self.assertRaisesRegex(ValueError, "SPEAKER_FAMILY_ROLE_LEAKAGE"):
            bulk_boundary.measure(plan, screen, root)

    def test_bulk_effect_bridge_rejects_hidden_generation_and_false_authority(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        screen["rows"][1] = {"cell_id": "bulk-b222", "status": "NO_GENERATION",
                             "training_admitted": False}
        screen["counts"] = {"MACHINE_POSITIVE_CANDIDATE": 1, "NO_GENERATION": 1}
        with self.assertRaisesRegex(ValueError, "UNREPORTED_GENERATED_WAV"):
            bulk_boundary.measure(plan, screen, root)
        plan, screen, root = self.make_bulk_data_fixture()
        screen["training_admitted"] = True
        with self.assertRaisesRegex(ValueError, "UNTRUSTED_BULK_SCREEN"):
            bulk_boundary.measure(plan, screen, root)


    def test_bulk_effect_bridge_never_writes_private_report_into_public_repo(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        plan_path, screen_path = self.root / "bulk-plan.json", self.root / "bulk-screen.json"
        dump(plan_path, plan)
        dump(screen_path, screen)
        forbidden = ROOT / "restricted-effect-report-should-never-exist.json"
        self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(ValueError, "PRIVATE_REPORT_MUST_NOT_BE_IN_PUBLIC_REPOSITORY"):
            bulk_boundary.main(["--plan", str(plan_path), "--screen", str(screen_path),
                "--generated-root", str(root), "--output", str(forbidden)])
        self.assertFalse(forbidden.exists())


    def test_bulk_effect_bridge_requires_exact_source_start_and_one_shot_intent(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        start = root / "cells" / "bulk-a111" / "started.json"
        receipt = json.loads(start.read_text(encoding="utf-8"))
        receipt["attempts"] = 2
        dump(start, receipt)
        with self.assertRaisesRegex(ValueError, "SOURCE_START_RECEIPT_DRIFT"):
            bulk_boundary.measure(plan, screen, root)
        receipt["attempts"] = 1
        receipt["reference_wav_sha256"] = "f" * 64
        dump(start, receipt)
        with self.assertRaisesRegex(ValueError, "SOURCE_START_RECEIPT_DRIFT"):
            bulk_boundary.measure(plan, screen, root)
        start.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            bulk_boundary.measure(plan, screen, root)


    def test_data_feedback_targets_actual_negative_confusion_and_missing_kw2(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        report = bulk_boundary.measure(plan, screen, root)
        result = feedback.derive(report)
        codes = [item["code"] for item in result["ranked_actions"]]
        self.assertIn("QUARANTINE_POSSIBLE_WAKE_IN_NEGATIVE", codes)
        self.assertIn("NO_MACHINE_SCREENED_KEYWORD_POSITIVE", codes)
        self.assertIn("VERIFY_POSITIVE_SOURCE_LABELS", codes)
        self.assertIn("EXTERNAL_PRODUCT_DATA_AND_AUTHORITY_MISSING", codes)
        self.assertEqual(result["source_families_declared"], 1)
        self.assertFalse(result["next_campaign_authorized"])
        self.assertFalse(result["training_admitted"])
        self.assertTrue(all(x["new_model_or_generator_calls_authorized"] == 0
                            for x in result["ranked_actions"]))

    def test_data_feedback_rejects_forged_counts_authorities_and_pcm(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        report = bulk_boundary.measure(plan, screen, root)
        for field, bad, error in (
            ("training_admitted", True, "UNEXPECTED_EFFECT_AUTHORITY"),
            ("screen_counts", {"MACHINE_POSITIVE_CANDIDATE": 2}, "EFFECT_COUNTS_DRIFT"),
            ("unique_pcm_verified", 1, "EFFECT_DUPLICATE_OR_AUDIO_COUNT_DRIFT")):
            mutated = json.loads(json.dumps(report))
            mutated[field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                feedback.derive(mutated)
        modified = json.loads(json.dumps(report))
        modified["screen_by_challenge"]["near_miss:homophone_wu_requested"] = {}
        with self.assertRaisesRegex(ValueError, "EFFECT_STRATUM_COUNTS_DRIFT"):
            feedback.derive(modified)
        # K2 may legitimately be unsampled, but its explicit empty bucket is
        # part of the upstream report contract and must not be omitted/forged.
        modified = json.loads(json.dumps(report))
        self.assertEqual(modified["screen_by_keyword"]["2"], {})
        modified["screen_by_keyword"].pop("2")
        with self.assertRaisesRegex(ValueError, "EFFECT_STRATUM_COUNTS_DRIFT"):
            feedback.derive(modified)

    def test_data_feedback_does_not_create_campaign_or_overwrite_report(self) -> None:
        plan, screen, root = self.make_bulk_data_fixture()
        report = bulk_boundary.measure(plan, screen, root)
        source = self.root / "private-effect.json"
        output = self.root / "private-priority.json"
        dump(source, report)
        self.assertEqual(feedback.main(["--effect", str(source), "--output", str(output)]), 0)
        self.assertTrue(output.is_file())
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        rows = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(rows["input_effect_sha256"], feedback.canonical_sha(report))
        self.assertFalse(rows["shipping_approved"])
        with self.assertRaises(FileExistsError):
            feedback.main(["--effect", str(source), "--output", str(output)])
        with self.assertRaisesRegex(ValueError, "PRIVATE_OUTPUT_ONLY"):
            feedback.main(["--effect", str(source), "--output",
                           str(ROOT / "do-not-write-this-priority.json")])


if __name__ == "__main__":
    unittest.main()
