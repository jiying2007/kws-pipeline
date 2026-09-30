#!/usr/bin/env python3
"""Synthetic fixture-only tests: no real human review receipts or Qwen audio."""
import copy
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import speech_like_corpus_plan as plan
from corpus_identity import inspect_pcm16_wav


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.intents_path = self.root / "intents.jsonl"
        self.review_path = self.root / "reviews.jsonl"
        self.generated = self.root / "generated"
        self.rows, self.intents, self.reviews = [], [], []
        for index, (group, split) in enumerate((("train", "train"), ("generalization-search", "test"))):
            directory = self.generated / group
            directory.mkdir(parents=True)
            audio = directory / "fixture.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(struct.pack('<160h', *([index + 1] * 160)))
            row = {"schema_version": 1, "evidence_class": plan.MANIFEST_CLASS,
                   "synthetic": True, "provider_kind": "offline-tts", "voice_id": f"fixture-voice-{index}",
                   "source_id": f"fixture-source-{index}", "audio": "fixture.wav", "text": "测试",
                   "tokens": ["fixture"], **inspect_pcm16_wav(audio)}
            self.rows.append(row)
            self.intents.append({**row, "evidence_class": plan.INTENT_CLASS, "provider_group": group,
                                 "split": split, "kind": "positive", "keyword_id": 0})
            self.reviews.append({"schema_version": 1, "evidence_class": "speech-like-audio-review-v1",
                                 "source_id": row["source_id"], "file_sha256": row["file_sha256"],
                                 "reviewer_id": "synthetic-test-fixture-not-human", "intended_text": row["text"],
                                 "kind": "positive", "keyword_id": 0, "verdict": "accepted"})

    def write(self):
        def dump(path, rows):
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        dump(self.intents_path, self.intents)
        dump(self.review_path, self.reviews)
        for index, group in enumerate(("train", "generalization-search")):
            dump(self.generated / group / "manifest.jsonl", [self.rows[index]])

    def admit(self):
        self.write()
        return plan.admit_reviewed_recordings(self.intents_path, self.generated, self.review_path, expected_count=2)

    def test_matching_receipts_and_cli(self):
        result = self.admit()
        self.assertEqual(result["recordings"], 2)
        self.assertFalse(result["qualification_allowed"])
        self.assertFalse(result["generator_family_independence_verified"])
        files_before = sorted(str(p) for p in self.root.rglob('*'))
        command = [sys.executable, str(ROOT/'tools/speech_like_corpus_plan.py'), 'admit-reviewed',
                   '--intents', str(self.intents_path), '--generated-root', str(self.generated),
                   '--audio-review', str(self.review_path), '--expected-count', '2']
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['recordings'], 2)
        self.assertEqual(files_before, sorted(str(p) for p in self.root.rglob('*')))
        command[-1] = '20'
        failure = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(failure.returncode, 0)
        self.assertFalse(failure.stdout)

    def test_legacy_and_reviewed_materialization(self):
        self.write()
        intents, reviews, groups = [], [], {}
        for index, (group, split) in enumerate((("train", "train"), ("generalization-search", "calibration"),
                                               ("generalization-search", "test"), ("generalization-freeze", "qualification"))):
            directory = self.generated / group
            directory.mkdir(exist_ok=True)
            audio = directory / f"legacy-{index}.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(struct.pack('<160h', *([index + 1] * 160)))
            row = {**self.rows[0], "voice_id": f"legacy-voice-{index}", "source_id": f"legacy-source-{index}",
                   "audio": audio.name, **inspect_pcm16_wav(audio)}
            groups.setdefault(group, []).append(row)
            intents.append({**self.intents[0], **row, "evidence_class": plan.INTENT_CLASS,
                            "provider_group": group, "split": split})
            reviews.append({**self.reviews[0], "source_id": row["source_id"], "file_sha256": row["file_sha256"]})
        for group, rows in groups.items():
            (self.generated/group/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.intents_path.write_text(''.join(json.dumps(row)+'\n' for row in intents))
        self.review_path.write_text(''.join(json.dumps(row)+'\n' for row in reviews))
        old = plan.materialize_labels(self.intents_path, self.generated, self.root/'legacy')
        guarded = plan.materialize_labels(self.intents_path, self.generated, self.root/'guarded', audio_review=self.review_path)
        self.assertEqual(old['recordings'], 4)
        self.assertNotIn('reviewed_admission', old)
        self.assertFalse(guarded['reviewed_admission']['qualification_allowed'])
        for split in plan.SPLITS:
            for name in ('manifest.jsonl', 'labels.jsonl', 'labeled-manifest.jsonl'):
                self.assertEqual((self.root/'legacy'/split/name).read_bytes(),
                                 (self.root/'guarded'/split/name).read_bytes())

    def test_real_plan_build_cli(self):
        plan_path = ROOT / "configs/training/speech-like-corpus-plan-v1.json"
        config = json.loads(plan_path.read_text())
        inventory = self.root / 'voices.jsonl'
        voices = [{"schema_version": 1, "evidence_class": plan.VOICE_CLASS,
                   "slot": slot, "voice_id": f"fixture-{slot}", "parameters": {}}
                  for role in config['split_roles'].values() for slot in role['voice_slots']]
        inventory.write_text(''.join(json.dumps(row)+'\n' for row in voices))
        requests, intents, summary = plan.build_requests(plan_path, inventory)
        self.assertEqual((len(requests), len(intents), summary['voice_slots']), (384, 384, 24))
        self.assertEqual(summary['split_counts'], {'train': 128, 'calibration': 64, 'test': 64, 'qualification': 128})
        self.assertNotIn('reviewed_admission', summary)
        requests_path, intents_path, summary_path = [self.root / name for name in ('built-requests.jsonl', 'built-intents.jsonl', 'built-summary.json')]
        result = subprocess.run([sys.executable, str(ROOT/'tools/speech_like_corpus_plan.py'), 'build',
                                 '--plan', str(plan_path), '--voice-inventory', str(inventory),
                                 '--requests', str(requests_path), '--intents', str(intents_path),
                                 '--summary', str(summary_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(summary_path.read_text()), summary)
        self.assertEqual(len(requests_path.read_text().splitlines()), 384)
        self.assertEqual(len(intents_path.read_text().splitlines()), 384)

    def test_typed_receipt_labels_and_schema(self):
        for bad in (True, 1.0, None, '1'):
            with self.subTest(keyword_id=bad):
                self.intents[0]['keyword_id'] = 1
                self.reviews[0]['keyword_id'] = bad
                with self.assertRaises(ValueError): self.admit()
        self.reviews[0]['keyword_id'] = 1
        for bad in (True, 1.0, None, '1'):
            with self.subTest(schema_version=bad):
                self.reviews[0]['schema_version'] = bad
                with self.assertRaises(ValueError): self.admit()
        self.reviews[0]['schema_version'] = 1
        del self.reviews[0]['keyword_id']
        with self.assertRaisesRegex(ValueError, 'keyword_id is missing'): self.admit()
        self.intents[0].update(kind='negative', keyword_id=None)
        self.reviews[0]['kind'] = 'negative'
        with self.assertRaisesRegex(ValueError, 'keyword_id is missing'): self.admit()
        self.reviews[0]['keyword_id'] = None
        self.assertEqual(self.admit()['recordings'], 2)

    def test_bad_receipts(self):
        original = copy.deepcopy(self.reviews)
        for field, value in [('file_sha256', 'a'*64), ('intended_text', 'wrong'), ('kind', 'negative'),
                             ('keyword_id', 1), ('verdict', 'uncertain'), ('reviewer_id', ''),
                             ('evidence_class', 'wrong')]:
            with self.subTest(field=field):
                self.reviews = copy.deepcopy(original)
                self.reviews[0][field] = value
                with self.assertRaises(ValueError): self.admit()
        for changed in [original[:1], original+[original[0]], [original[0],original[0]]]:
            self.reviews = changed
            with self.assertRaises(ValueError): self.admit()

    def test_missing_and_extra_intent(self):
        self.intents.pop()
        with self.assertRaises(ValueError): self.admit()

    def test_wav_tamper(self):
        (self.generated/'train/fixture.wav').write_bytes(b'not wav')
        with self.assertRaises((ValueError, EOFError, wave.Error)): self.admit()

    def test_pcm_hash_mismatch(self):
        self.rows[0]['pcm_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'pcm_sha256 mismatch'): self.admit()

    def test_duplicate_pcm(self):
        src = self.generated/'train/fixture.wav'
        dst = self.generated/'generalization-search/fixture.wav'
        dst.write_bytes(src.read_bytes())
        self.rows[1].update(inspect_pcm16_wav(dst))
        self.reviews[1]['file_sha256'] = self.rows[1]['file_sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate decoded PCM'): self.admit()

    def test_voice_leak(self):
        self.rows[1]['voice_id'] = self.rows[0]['voice_id']
        self.intents[1]['voice_id'] = self.rows[0]['voice_id']
        with self.assertRaisesRegex(ValueError, 'voice_id crosses splits'): self.admit()

    def test_source_leak(self):
        self.rows[1]['source_id'] = self.rows[0]['source_id']
        self.intents[1]['source_id'] = self.rows[0]['source_id']
        self.reviews[1]['source_id'] = self.rows[0]['source_id']
        with self.assertRaisesRegex(ValueError, 'duplicate source_id'): self.admit()

    def test_bad_label(self):
        self.intents[0]['keyword_id'] = -1
        self.reviews[0]['keyword_id'] = -1
        with self.assertRaisesRegex(ValueError, 'non-negative'): self.admit()

    def test_materialization_rejects_before_writes(self):
        self.write()
        self.reviews[0]['verdict'] = 'rejected'
        self.write()
        out = self.root / 'must-not-exist'
        with self.assertRaises(ValueError):
            plan.materialize_labels(self.intents_path, self.generated, out, audio_review=self.review_path)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
