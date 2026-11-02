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


if __name__ == "__main__":
    unittest.main()
