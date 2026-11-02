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
import voice_clone_three_asr_batch as batch_asr  # noqa: E402


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


    def test_typed_challenges_bind_generation_and_screening_by_family(self) -> None:
        typed = ("exact_requested", "homophone_requested", "ordinary_conversation_requested")
        for phrase, family in zip(self.payload["phrases"], typed):
            phrase["challenge_family"] = family
        self.save_plan()
        campaign = self.loaded()
        self.assertEqual([p["challenge_family"] for p in campaign["phrases"]], list(typed))
        clone.execute(campaign, self.output, workers=1)
        report = effect.audit(self.output)
        for kind, family in zip(clone.KINDS, typed):
            self.assertEqual(report["screen_by_challenge_family"][
                kind + ":" + family][effect.SCREEN_NONE], 2)
        self.assertFalse(report["qualification_allowed"])
        self.assertFalse(report["training_admitted"])
        self.assertEqual(clone.execute(campaign, self.output, workers=1)["recordings"], 6)
        receipt = sorted((self.output / "attempts").rglob("*.receipt.json"))[0]
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data["challenge_family"] = "other_near_miss_requested"
        dump(receipt, data)
        with self.assertRaisesRegex(RuntimeError, "receipt metadata/output mismatch"):
            clone.execute(campaign, self.output, workers=1)

    def test_challenge_family_wrong_intent_unknown_and_wrong_type_are_rejected(self) -> None:
        for bad in ("homophone_requested", "not-a-taxonomy-family", [], False):
            with self.subTest(bad=bad):
                self.payload["phrases"][0]["challenge_family"] = bad
                self.save_plan()
                with self.assertRaisesRegex(ValueError, "not allowed for intent"):
                    self.loaded()
        self.payload["phrases"][0].pop("challenge_family")
        self.payload["phrases"][1]["challenge_family"] = "homophone_requested"
        self.save_plan()
        self.assertEqual(self.loaded()["phrases"][1]["challenge_family"], "homophone_requested")

    def test_auditor_rejects_forged_cross_intent_challenge_family(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        manifest = self.output / "manifest.jsonl"
        summary_file = self.output / "summary.json"
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        row = next(record for record in records if record["requested_intent"] == "wake")
        row["challenge_family"] = "homophone_requested"
        manifest.write_text("".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                                    for record in records), encoding="utf-8")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        summary["manifest_sha256"] = sha(manifest)
        dump(summary_file, summary)
        with self.assertRaisesRegex(ValueError, "challenge family conflicts"):
            effect.audit(self.output)


    def test_all_empty_asr_outputs_do_not_prove_silence_or_wake_correctness(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        row = json.loads((self.output / "manifest.jsonl").read_text().splitlines()[0])
        screen = self.root / "empty-screen.jsonl"
        screen.write_text(json.dumps({"item_id": row["item_id"],
            "file_sha256": row["file_sha256"],
            "observations": [{"model_family": f"distinct-{i}", "raw_text": ""}
                             for i in range(3)]}) + "\n")
        report = effect.audit(self.output, screen)
        self.assertEqual(report["screen_counts"]["ASR_ALL_EMPTY_UNVERIFIED"], 1)
        self.assertIn("EMPTY_ASR_SCREEN_IS_NOT_SILENCE_EVIDENCE", report["gaps"])
        self.assertFalse(report["label_authentication_verified"])
        self.assertFalse(report["training_admitted"])
        self.assertFalse(report["qualification_allowed"])


    def make_offline_asr_plan(self, *, drop_second: bool = False) -> Path:
        stub = self.root / "batch-asr-stub.py"
        stub.write_text(
            "import json, pathlib, sys\n"
            "source, target, model = map(pathlib.Path, sys.argv[1:4])\n"
            "rows = [json.loads(x) for x in source.read_text().splitlines()]\n"
            "assert rows and all(set(x) == {'item_id','file_sha256','pcm_sha256','audio'} for x in rows)\n"
            "if model.read_text().strip() == 'drop-last': rows = rows[:-1]\n"
            "target.write_text(''.join(json.dumps({'item_id': x['item_id'],\n"
            "  'file_sha256': x['file_sha256'], 'raw_text': '你好小窝',\n"
            "  'status': 'complete'}, ensure_ascii=False) + '\\n' for x in rows))\n",
            encoding="utf-8")
        providers = []
        for i in range(3):
            model = self.root / f"asr-model-{i}.bin"
            model.write_text("drop-last" if i == 1 and drop_second else f"model-{i}")
            providers.append({
                "model_family": f"independent-{i}", "model_id": f"model-{i}",
                "model_revision": "v1",
                "decode_config": dict(batch_asr.DEFAULT_CONFIG),
                "executable": {"path": sys.executable, "sha256": sha(Path(sys.executable))},
                "assets": [{"role": "model", "path": model.name, "sha256": sha(model)},
                           {"role": "adapter", "path": stub.name, "sha256": sha(stub)}],
                "argv_template": ["{executable}", "{asset:adapter}",
                                  "{input}", "{output}", "{asset:model}"],
                "timeout_seconds": 30,
            })
        plan = self.root / "three-asr-plan.json"
        dump(plan, {"schema_version": 1, "policy": batch_asr.POLICY,
                    "max_recordings": 6, "recognizers": providers})
        return plan

    def test_batch_asr_requires_opt_in_and_omits_requested_text_from_input(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        asr_plan = self.make_offline_asr_plan()
        destination = self.root / "private-asr-output"
        batch_asr.main(["--plan", str(asr_plan), "--campaign-root", str(self.output),
                        "--output-root", str(destination)])
        self.assertFalse(destination.exists())
        report = batch_asr.execute(batch_asr.parse_plan(asr_plan), self.output, destination)
        self.assertEqual(report["screened_recordings"], 6)
        self.assertEqual(len(report["recognizers"]), 3)
        self.assertFalse(report["asr_source_authenticated"])
        self.assertFalse(report["training_admitted"])
        self.assertFalse(report["qualification_allowed"])
        self.assertFalse(report["shipping_approved"])
        inputs = [json.loads(row) for row in
                  (destination / "audio-only-batch.jsonl").read_text().splitlines()]
        self.assertTrue(all(set(row) == {"item_id", "audio", "file_sha256", "pcm_sha256"}
                            for row in inputs))
        self.assertNotIn("你好小窝", (destination / "audio-only-batch.jsonl").read_text())
        self.assertNotIn("text_requested", (destination / "audio-only-batch.jsonl").read_text())
        self.assertEqual(len((destination / "screen.jsonl").read_text().splitlines()), 6)
        self.assertEqual(report, batch_asr.execute(batch_asr.parse_plan(asr_plan),
                                                   self.output, destination))

    def test_batch_asr_fails_closed_on_duplicate_models_prompt_and_unbounded_run(self) -> None:
        self.make_offline_asr_plan()
        plan_file = self.root / "three-asr-plan.json"
        original = json.loads(plan_file.read_text(encoding="utf-8"))
        def bad(patch, pattern):
            plan = json.loads(json.dumps(original))
            patch(plan)
            dump(plan_file, plan)
            with self.assertRaisesRegex(ValueError, pattern):
                batch_asr.parse_plan(plan_file)
        bad(lambda plan: plan["recognizers"][1].update(model_family="independent-0"),
            "repeated ASR family")
        bad(lambda plan: plan["recognizers"][1]["assets"].__setitem__(
            0, plan["recognizers"][0]["assets"][0]), "distinct pinned model")
        bad(lambda plan: plan["recognizers"][0]["decode_config"].update(prompt="你好小窝"),
            "without prompt/hotwords")
        bad(lambda plan: plan.update(max_recordings=100000), "explicit bound")
        dump(plan_file, original)

    def test_batch_asr_missing_output_consumes_attempt_without_replay(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        plan = self.make_offline_asr_plan(drop_second=True)
        destination = self.root / "incomplete-asr-output"
        with self.assertRaisesRegex(ValueError, "does not cover every WAV"):
            batch_asr.execute(batch_asr.parse_plan(plan), self.output, destination)
        with self.assertRaisesRegex(ValueError, "unknown and consumed"):
            batch_asr.execute(batch_asr.parse_plan(plan), self.output, destination)
        self.assertFalse((destination / "screen.jsonl").exists())
        self.assertFalse((destination / "summary.json").exists())

    def test_batch_asr_readback_detects_tamper_and_forged_authority(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        plan = self.make_offline_asr_plan()
        dest = self.root / "screen-results"
        batch_asr.execute(batch_asr.parse_plan(plan), self.output, dest)
        receipt = dest / "attempts" / "independent-0.receipt.json"
        item = json.loads(receipt.read_text())
        item["training_admitted"] = True
        dump(receipt, item)
        with self.assertRaisesRegex(ValueError, "readback receipt mismatch"):
            batch_asr.execute(batch_asr.parse_plan(plan), self.output, dest)

    def test_batch_asr_frozen_sources_cannot_change_after_screen(self) -> None:
        clone.execute(self.loaded(), self.output, workers=1)
        plan = self.make_offline_asr_plan()
        destination = self.root / "asr-drift-results"
        batch_asr.execute(batch_asr.parse_plan(plan), self.output, destination)
        wav = sorted((self.output / "audio").rglob("*.wav"))[0]
        wav.write_bytes(wav.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "generated PCM/file identity mismatch"):
            batch_asr.execute(batch_asr.parse_plan(plan), self.output, destination)


if __name__ == "__main__":
    unittest.main()
