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
from speech_label_admission import REAL_MODE, FIXTURE_MODE, TRANSCRIPT_POLICY, canonical_sha256
from materialize_speech_like_base_index import materialize, corpus_sha
from external_base_dataset import load_external_base_bundle


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
                writer.writeframes(struct.pack('<160h', *([1000 + index] * 160)))
            row = {"schema_version": 1, "evidence_class": plan.MANIFEST_CLASS,
                   "synthetic": True, "provider_kind": "offline-tts", "voice_id": f"fixture-voice-{index}",
                   "source_id": f"fixture-source-{index}", "source_family_id": f"fixture-family-{index}",
                   "provider_name": "schema-fixture", "provider_version": "1", "license_id": "fixture-only",
                   "generation_config_sha256": "a"*64,
                   "audio": "fixture.wav", "text": "你好小窝",
                   "tokens": ["ni3", "hao3", "xiao3", "wo1"], **inspect_pcm16_wav(audio)}
            self.rows.append(row)
            self.intents.append({**row, "evidence_class": plan.INTENT_CLASS, "provider_group": group,
                                 "split": split, "kind": "positive", "keyword_id": 1})
            self.reviews.append({"schema_version": 2, "evidence_class": "speech-like-audio-review-v2",
                                 "source_id": row["source_id"], "file_sha256": row["file_sha256"], "pcm_sha256": row["pcm_sha256"],
                                 "source_family_id": row["source_family_id"],
                                 "reviewer_id": "schema-test-only-not-an-operational-receipt", "intended_text": row["text"],
                                 "actual_text": row["text"], "actual_tokens": row["tokens"], "transcript_policy": TRANSCRIPT_POLICY,
                                 "review_origin": "human", "allowed_purpose": "ctc-training-only", "speech_present": True,
                                 "review_revision": 1, "supersedes_review_sha256": None, "acoustic_complete": True,
                                 "kind": "positive", "keyword_id": 1, "verdict": "accepted"})

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

    def four_splits(self):
        self.write()
        intents, reviews, groups = [], [], {}
        for index, (group, split) in enumerate((("train", "train"), ("generalization-search", "calibration"),
                                               ("generalization-search", "test"), ("generalization-freeze", "qualification"))):
            directory = self.generated / group
            directory.mkdir(exist_ok=True)
            audio = directory / f"legacy-{index}.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(struct.pack('<160h', *([1000 + index] * 160)))
            row = {**self.rows[0], "voice_id": f"legacy-voice-{index}", "source_id": f"legacy-source-{index}",
                   "source_family_id": f"legacy-family-{index}", "audio": audio.name, **inspect_pcm16_wav(audio)}
            groups.setdefault(group, []).append(row)
            intents.append({**self.intents[0], **row, "evidence_class": plan.INTENT_CLASS,
                            "provider_group": group, "split": split})
            reviews.append({**self.reviews[0], "source_id": row["source_id"], "source_family_id": row["source_family_id"],
                            "file_sha256": row["file_sha256"], "pcm_sha256": row["pcm_sha256"]})
        for group, rows in groups.items():
            (self.generated/group/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.intents_path.write_text(''.join(json.dumps(row)+'\n' for row in intents))
        self.review_path.write_text(''.join(json.dumps(row)+'\n' for row in reviews))
        return intents, reviews, groups

    def test_fixture_and_reviewed_materialization(self):
        self.four_splits()
        with self.assertRaisesRegex(ValueError, "mandatory human"):
            plan.materialize_labels(self.intents_path, self.generated, self.root/'missing-review')
        self.assertFalse((self.root/'missing-review').exists())
        fixture = plan.materialize_labels(self.intents_path, self.generated, self.root/'fixture', admission_mode=FIXTURE_MODE)
        guarded = plan.materialize_labels(self.intents_path, self.generated, self.root/'guarded', audio_review=self.review_path)
        self.assertFalse(fixture['ctc_training_allowed'])
        self.assertTrue(guarded['ctc_training_allowed'])
        self.assertFalse(guarded['reviewed_admission']['qualification_allowed'])
        for split in plan.SPLITS:
            manifest = self.root/'guarded'/split/'labeled-manifest.jsonl'
            row = json.loads(manifest.read_text())
            self.assertEqual(row['admission']['mode'], REAL_MODE)
            self.assertEqual(row['actual_text'], row['text'])
            self.assertEqual(row['tokens'], row['admission']['review_history'][-1]['actual_tokens'])
            rows, summary = materialize(manifest=manifest, split=split,
                                        tokens_path=ROOT/'keywords/tokens.example.txt', keywords_path=ROOT/'keywords/zh_cn_example.tsv')
            self.assertTrue(summary['ctc_training_allowed'])
            self.assertEqual(rows[0]['family_id'], row['source_family_id'])
        fixture_manifest = self.root/'fixture'/'train'/'labeled-manifest.jsonl'
        with self.assertRaisesRegex(ValueError, "admission mode"):
            materialize(manifest=fixture_manifest, split='train', tokens_path=ROOT/'keywords/tokens.example.txt',
                        keywords_path=ROOT/'keywords/zh_cn_example.tsv')
        _, summary = materialize(manifest=fixture_manifest, split='train', tokens_path=ROOT/'keywords/tokens.example.txt',
                                 keywords_path=ROOT/'keywords/zh_cn_example.tsv', admission_mode=FIXTURE_MODE)
        self.assertFalse(summary['ctc_training_allowed'])

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
        self.reviews[0]['schema_version'] = 2
        del self.reviews[0]['keyword_id']
        with self.assertRaisesRegex(ValueError, 'keyword_id is missing'): self.admit()
        self.intents[0].update(kind='negative', keyword_id=None)
        self.reviews[0].update(kind='negative', actual_text='你好', actual_tokens=['ni3','hao3'])
        with self.assertRaisesRegex(ValueError, 'keyword_id is missing'): self.admit()
        self.reviews[0]['keyword_id'] = None
        self.assertEqual(self.admit()['recordings'], 2)

    def test_bad_receipts(self):
        original = copy.deepcopy(self.reviews)
        for field, value in [('file_sha256', 'a'*64), ('intended_text', 'wrong'), ('kind', 'negative'),
                             ('verdict', 'uncertain'), ('reviewer_id', ''),
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
        self.reviews[1]['pcm_sha256'] = self.rows[1]['pcm_sha256']
        with self.assertRaisesRegex(ValueError, 'duplicate decoded PCM'): self.admit()

    def test_voice_leak(self):
        self.rows[1]['voice_id'] = self.rows[0]['voice_id']
        self.intents[1]['voice_id'] = self.rows[0]['voice_id']
        with self.assertRaisesRegex(ValueError, 'voice_id crosses splits'): self.admit()

    def test_speaker_leak(self):
        for row, review in zip(self.rows, self.reviews):
            row['speaker_id'] = review['speaker_id'] = 'known-shared-speaker'
        with self.assertRaisesRegex(ValueError, 'speaker_id crosses splits'): self.admit()

    def test_source_leak(self):
        self.rows[1]['source_id'] = self.rows[0]['source_id']
        self.intents[1]['source_id'] = self.rows[0]['source_id']
        self.reviews[1]['source_id'] = self.rows[0]['source_id']
        with self.assertRaisesRegex(ValueError, 'duplicate source_id'): self.admit()

    def test_bad_label(self):
        self.intents[0]['keyword_id'] = -1
        self.reviews[0]['keyword_id'] = -1
        with self.assertRaisesRegex(ValueError, 'positive integer'): self.admit()

    def test_materialization_rejects_before_writes(self):
        self.write()
        self.reviews[0]['verdict'] = 'rejected'
        self.write()
        out = self.root / 'must-not-exist'
        with self.assertRaises(ValueError):
            plan.materialize_labels(self.intents_path, self.generated, out, audio_review=self.review_path)
        self.assertFalse(out.exists())

    def test_generation_defaults_block_before_side_effects(self):
        work = self.root/'never-generated'
        command = [sys.executable, str(ROOT/'tools/run_speech_like_corpus_generation.py'),
                   '--runtime-archive', str(self.root/'not-a-runtime'), '--license-evidence', str(self.root/'not-a-license'),
                   '--backend-executable', str(self.root/'not-a-backend'), '--work-dir', str(work)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --audio-review', result.stderr)
        self.assertFalse(work.exists())
        command = [sys.executable, str(ROOT/'tools/bootstrap_speech_like_stage_a.py'),
                   '--cache-dir', str(self.root/'never-cached'), '--work-dir', str(work)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --audio-review', result.stderr)
        self.assertFalse(work.exists())

    def test_actual_labels_replace_intent(self):
        self.four_splits()
        reviews = [json.loads(line) for line in self.review_path.read_text().splitlines()]
        reviews[0].update(actual_text='你好小', actual_tokens=['ni3','hao3','xiao3'], kind='confusable', keyword_id=None)
        self.review_path.write_text(''.join(json.dumps(row)+'\n' for row in reviews))
        plan.materialize_labels(self.intents_path, self.generated, self.root/'corrected', audio_review=self.review_path)
        row = json.loads((self.root/'corrected/train/labeled-manifest.jsonl').read_text())
        self.assertEqual(row['intended_text'], '你好小窝')
        self.assertEqual(row['actual_text'], '你好小')
        self.assertEqual(row['tokens'], ['ni3','hao3','xiao3'])
        self.assertEqual(row['kind'], 'confusable')
        self.assertIsNone(row['keyword_id'])

    def test_acoustic_text_and_purpose_fail_closed(self):
        original = copy.deepcopy(self.reviews)
        cases = [('acoustic_complete', False), ('actual_text', '你好'), ('actual_text', '未知'),
                 ('actual_tokens', []), ('review_revision', 0), ('review_origin', 'synthetic-fixture'),
                 ('allowed_purpose', 'diagnostic-only'), ('speech_present', False), ('source_family_id', '')]
        for field, value in cases:
            with self.subTest(field=field):
                self.reviews = copy.deepcopy(original)
                self.reviews[0][field] = value
                with self.assertRaises(ValueError): self.admit()

    def test_latest_revision_supersedes_prior_acceptance(self):
        first = copy.deepcopy(self.reviews[0])
        second = {**first, 'review_revision': 2, 'supersedes_review_sha256': canonical_sha256(first), 'verdict': 'rejected'}
        self.reviews.append(second)
        with self.assertRaisesRegex(ValueError, 'latest audio review'): self.admit()
        second['verdict'] = 'accepted'
        self.assertEqual(self.admit()['recordings'], 2)
        second['supersedes_review_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'revision chain'): self.admit()
        second['supersedes_review_sha256'] = canonical_sha256(first)
        self.reviews.remove(first)
        with self.assertRaisesRegex(ValueError, 'revision chain'): self.admit()

    def test_rejected_oov_revision_can_be_corrected(self):
        first = copy.deepcopy(self.reviews[0])
        first.update(verdict='rejected', actual_text='未知', actual_tokens=['unknown'])
        second = {**self.reviews[0], 'review_revision': 2, 'supersedes_review_sha256': canonical_sha256(first)}
        self.reviews = [first, self.reviews[1], second]
        self.assertEqual(self.admit()['recordings'], 2)
        second['keyword_id'] = 2
        with self.assertRaisesRegex(ValueError, 'canonical keyword_id'): self.admit()
        second.update(kind='negative', keyword_id=None)
        with self.assertRaisesRegex(ValueError, 'canonical wake path'): self.admit()

    def test_cross_field_source_family_leaks(self):
        from source_identity import SOURCE_ID_FIELDS
        original = copy.deepcopy((self.rows, self.reviews, self.intents))
        for left in SOURCE_ID_FIELDS:
            for right in SOURCE_ID_FIELDS:
                with self.subTest(left=left, right=right):
                    self.rows, self.reviews, self.intents = copy.deepcopy(original)
                    for index, field in ((0, left), (1, right)):
                        for collection in (self.rows, self.reviews, self.intents):
                            collection[index][field] = "cross-field-family"
                    with self.assertRaisesRegex(ValueError, "crosses splits|duplicate source_id"):
                        self.admit()

    def test_optional_family_alias_requires_matching_review(self):
        for field in ("source_family", "family_id", "derivation_family"):
            with self.subTest(field=field):
                self.rows[0][field] = "unreviewed-lineage"
                with self.assertRaisesRegex(ValueError, field + " differs from source"):
                    self.admit()
                self.rows[0].pop(field)

    def test_family_aliases_survive_materialization_render_lineage_and_training_admission(self):
        from source_identity import SOURCE_FAMILY_FIELDS
        from render_domains import source_lineage, domain_reference_row
        from training_admission import verify_training_manifests
        for index, (row, review) in enumerate(zip(self.rows, self.reviews)):
            for field in SOURCE_FAMILY_FIELDS:
                row[field] = review[field] = f"fixture-{index}-{field}"
        self.write()
        labeled = self.root / "aliases"
        plan.materialize_labels(self.intents_path, self.generated, labeled, audio_review=self.review_path)
        manifest = labeled / "train" / "labeled-manifest.jsonl"
        rows, _ = materialize(manifest=manifest, split="train",
                              tokens_path=ROOT / "keywords/tokens.example.txt",
                              keywords_path=ROOT / "keywords/zh_cn_example.tsv")
        row = rows[0]
        provenance = row["speech_like_provenance"]
        for field in SOURCE_FAMILY_FIELDS:
            self.assertEqual(provenance[field], self.rows[0][field])
        self.assertEqual(row["family_id"], provenance["family_id"])
        lineage = source_lineage(row)
        for field in SOURCE_FAMILY_FIELDS:
            self.assertEqual(lineage[field], provenance[field])
        training_row = {**row, **lineage}
        training_manifest = self.root / "alias-training.jsonl"
        training_manifest.write_text(json.dumps(training_row) + "\n")
        admitted = verify_training_manifests(
            [training_manifest], {"<blk>": 0, "ni3": 1, "hao3": 2, "xiao3": 3, "wo1": 4}
        )
        self.assertEqual(admitted["reviewed_rows"], 1)
        reference = domain_reference_row(
            {**training_row, "scene": {"direct_delay_samples": 0, "afe_latency_samples": 0},
             "domain_id": "invented"}, recording="invented", frames=row["frames"]
        )
        for field in SOURCE_FAMILY_FIELDS:
            self.assertEqual(reference[field], provenance[field])

    def test_external_bundle_rejects_cross_field_source_aliases(self):
        config = self.bundle()
        original = {}
        for split in ("train", "test"):
            spec = config["generator"]["external_base_dataset"][split]
            original[split] = json.loads(pathlib.Path(spec["index"]).read_text())
        for left, right in (("source_id", "family_id"),
                            ("source_family_id", "derivation_family_id"),
                            ("source_family", "derivation_family"),
                            ("family_id", "source_family_id")):
            with self.subTest(left=left, right=right):
                for split, field in (("train", left), ("test", right)):
                    def update(row, split=split, field=field):
                        row.clear()
                        row.update(copy.deepcopy(original[split]))
                        row["speech_like_provenance"][field] = "cross-field-family"
                        review = row["admission"]["review_history"][0]
                        review[field] = "cross-field-family"
                        row["admission"]["review_record_sha256"] = canonical_sha256(review)
                    self.rewrite_split(config, split, update)
                with self.assertRaisesRegex(ValueError, "cross-split .* overlap"):
                    load_external_base_bundle(self.root / "config.json", config)

    def test_source_family_leak(self):
        self.rows[1]['source_family_id'] = self.rows[0]['source_family_id']
        self.reviews[1]['source_family_id'] = self.rows[0]['source_family_id']
        with self.assertRaisesRegex(ValueError, 'source_family_id crosses splits'): self.admit()

    def bundle(self):
        self.four_splits()
        plan.materialize_labels(self.intents_path, self.generated, self.root/'labeled', audio_review=self.review_path)
        config = {'tokens': str(ROOT/'keywords/tokens.example.txt'), 'keywords': str(ROOT/'keywords/zh_cn_example.tsv'),
                  'generator': {'external_base_dataset': {}}}
        for split in plan.SPLITS:
            directory = self.root/'bundle'/split
            directory.mkdir(parents=True)
            rows, summary = materialize(manifest=self.root/'labeled'/split/'labeled-manifest.jsonl', split=split,
                                        tokens_path=ROOT/'keywords/tokens.example.txt', keywords_path=ROOT/'keywords/zh_cn_example.tsv',
                                        portable_audio_dir=directory/'audio', path_base=directory)
            index_path, summary_path = directory/'index.jsonl', directory/'summary.json'
            index_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            summary_path.write_text(json.dumps(summary))
            config['generator']['external_base_dataset'][split] = {
                'index': str(index_path), 'summary': str(summary_path),
                'index_sha256': plan.sha256_file(index_path), 'summary_sha256': plan.sha256_file(summary_path)}
        return config

    def rewrite_split(self, config, split, update):
        spec = config['generator']['external_base_dataset'][split]
        index_path, summary_path = pathlib.Path(spec['index']), pathlib.Path(spec['summary'])
        row = json.loads(index_path.read_text())
        update(row)
        index_path.write_text(json.dumps(row)+'\n')
        summary = json.loads(summary_path.read_text())
        summary['corpus_sha256'] = corpus_sha([row])
        summary_path.write_text(json.dumps(summary))
        spec.update(index_sha256=plan.sha256_file(index_path), summary_sha256=plan.sha256_file(summary_path))

    def test_external_bundle_mandatory_bound_admission(self):
        config = self.bundle()
        rows, summary = load_external_base_bundle(self.root/'config.json', config)
        self.assertTrue(summary['ctc_training_allowed'])
        self.assertEqual(len(rows), 4)
        self.rewrite_split(config, 'test', lambda row: row.pop('admission'))
        with self.assertRaisesRegex(ValueError, 'admission mode'):
            load_external_base_bundle(self.root/'config.json', config)
        config['generator']['external_base_admission_mode'] = 'historical-frozen-diagnostic-v1'
        with self.assertRaises(ValueError):
            load_external_base_bundle(self.root/'config.json', config)

    def test_arbitrary_bundle_cannot_claim_historical_exception(self):
        config = self.bundle()
        for split in plan.SPLITS:
            def strip_review(row):
                row.pop('admission')
                row.pop('actual_text')
            self.rewrite_split(config, split, strip_review)
        config['generator']['external_base_admission_mode'] = 'historical-frozen-diagnostic-v1'
        with self.assertRaisesRegex(ValueError, 'exact pinned frozen bundle'):
            load_external_base_bundle(self.root/'config.json', config)

    def test_other_known_lineage_leaks(self):
        for field in ('speaker_id', 'reference_audio_sha256', 'session_id', 'derivation_family_id'):
            with self.subTest(field=field):
                value = 'a'*64 if field.endswith('sha256') else 'known-shared-identity'
                for row, review in zip(self.rows, self.reviews):
                    row[field] = review[field] = value
                with self.assertRaisesRegex(ValueError, field+' crosses splits'): self.admit()
                for row, review in zip(self.rows, self.reviews):
                    row.pop(field); review.pop(field)

    def test_external_receipt_and_actual_target_tamper(self):
        config = self.bundle()
        def tamper(row):
            row['admission']['review_history'][0]['actual_text'] = '你好'
        self.rewrite_split(config, 'test', tamper)
        with self.assertRaisesRegex(ValueError, 'actual_text and actual_tokens disagree'):
            load_external_base_bundle(self.root/'config.json', config)

    def test_external_source_family_and_rewrapped_pcm_leaks(self):
        config = self.bundle()
        train_spec = config['generator']['external_base_dataset']['train']
        train = json.loads(pathlib.Path(train_spec['index']).read_text())
        def update_family(row):
            row['speech_like_provenance']['source_family_id'] = train['speech_like_provenance']['source_family_id']
            review = row['admission']['review_history'][0]
            review['source_family_id'] = train['speech_like_provenance']['source_family_id']
            row['admission']['review_record_sha256'] = canonical_sha256(review)
        self.rewrite_split(config, 'test', update_family)
        with self.assertRaisesRegex(ValueError, 'source_family_id overlap'):
            load_external_base_bundle(self.root/'config.json', config)
        def rewrap(row):
            audio = pathlib.Path(config['generator']['external_base_dataset']['test']['index']).parent/row['path']
            train_audio = pathlib.Path(train_spec['index']).parent/train['path']
            data = bytearray(train_audio.read_bytes())
            data.extend(b'JUNK' + struct.pack('<I', 2) + b'xx')
            data[4:8] = struct.pack('<I', len(data)-8)
            audio.write_bytes(data)
            inspected = inspect_pcm16_wav(audio)
            row['wav_sha256'] = inspected['file_sha256']
            row['speech_like_provenance']['pcm_sha256'] = inspected['pcm_sha256']
            row['speech_like_provenance']['source_family_id'] = 'separate-test-family'
            review = row['admission']['review_history'][0]
            review.update(file_sha256=inspected['file_sha256'], pcm_sha256=inspected['pcm_sha256'], source_family_id='separate-test-family')
            row['admission']['review_record_sha256'] = canonical_sha256(review)
        self.rewrite_split(config, 'test', rewrap)
        with self.assertRaisesRegex(ValueError, 'decoded PCM overlap'):
            load_external_base_bundle(self.root/'config.json', config)



class ThreeASRMetadataTests(unittest.TestCase):
    """Pure invented metadata; do not execute ASR or synthesize speech."""
    def test_unanimous_train_only_without_human_assertion(self):
        from asr_consensus_fixtures import consensus_fixture
        from speech_asr_consensus import admission_from_consensus, validate_consensus_admission
        record = consensus_fixture()
        admission = admission_from_consensus(record, 'f' * 64)
        self.assertEqual(admission['label_origin'], 'machine-consensus')
        self.assertNotIn('review_history', admission)
        self.assertFalse(admission['authentication_independently_verified'])
        self.assertFalse(admission['ctc_training_allowed'])
        self.assertEqual(admission['status'], 'PENDING_EXTERNAL_VERIFICATION')
        with self.assertRaisesRegex(ValueError, 'PENDING_EXTERNAL_VERIFICATION'):
            validate_consensus_admission(admission, split='train', for_training=True)
        forged = copy.deepcopy(admission); forged['ctc_training_allowed'] = True
        with self.assertRaises(ValueError): validate_consensus_admission(forged, split='train')
        validate_consensus_admission(admission, split='train', frames=160)
        for role in (None, 'test', 'calibration', 'qualification', 'source_screen_only'):
            with self.subTest(role=role), self.assertRaises(ValueError):
                validate_consensus_admission(admission, split=role)
        with self.assertRaises(ValueError):
            validate_consensus_admission(admission, split='train', frames=159)

    def test_missing_duplicate_failed_partial_biased_or_human_vote_rejected(self):
        from asr_consensus_fixtures import consensus_fixture, rebind_fixture
        from speech_asr_consensus import validate_consensus_record
        changes = [
            lambda r: r['observations'].pop(),
            lambda r: r['recognizer_policy']['recognizers'].__setitem__(2, copy.deepcopy(r['recognizer_policy']['recognizers'][0])),
            lambda r: r['recognizer_policy']['recognizers'][2].update(model_family='invented-family-0'),
            lambda r: r['recognizer_policy']['recognizers'][2].update(model_sha256='1' * 64),
            lambda r: r['observations'][2].update(status='error'),
            lambda r: r['observations'][2].update(origin='human'),
            lambda r: r['observations'][2].update(transcript_complete=False),
            lambda r: r['observations'][2].update(coverage_complete=False),
            lambda r: r['observations'][2].update(end_frame=159),
            lambda r: r['observations'][2].update(pcm_sha256='c' * 64),
            lambda r: r['observations'][2].update(source_id='other-source'),
            lambda r: r['recognizer_policy']['recognizers'][0]['decode_config'].update(hotwords=['小窝']),
            lambda r: r['authentication_verification'].update(verified=False),
        ]
        for index, change in enumerate(changes):
            record = consensus_fixture(); change(record); rebind_fixture(record)
            with self.subTest(index=index), self.assertRaises(ValueError): validate_consensus_record(record)

    def test_disagreement_oov_empty_and_wake_negative_rejected(self):
        from asr_consensus_fixtures import consensus_fixture, rebind_fixture
        from speech_asr_consensus import validate_consensus_record
        record = consensus_fixture(); record['observations'][2]['actual_text'] = '你好小屋'
        with self.assertRaisesRegex(ValueError, 'not unanimous'):
            validate_consensus_record(rebind_fixture(record))
        for text in ('你好小屋', '你好小挖', ''):
            record = consensus_fixture(); record['actual_text'] = text
            for row in record['observations']: row['actual_text'] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_consensus_record(rebind_fixture(record))
        record = consensus_fixture(); record.update(kind='negative', keyword_id=None)
        with self.assertRaisesRegex(ValueError, 'wake path'): validate_consensus_record(record)

    def test_human_observation_not_rewritten_or_used_as_third_vote(self):
        from asr_consensus_fixtures import consensus_fixture
        from speech_asr_consensus import validate_consensus_record
        record = consensus_fixture()
        human = dict(schema_version=2, evidence_class='speech-like-audio-review-v2',
            source_id=record['source_id'], file_sha256=record['file_sha256'], pcm_sha256=record['pcm_sha256'],
            source_family_id=record['source_family_id'], intended_text=record['intended_text'],
            reviewer_id='invented-human', review_origin='human', review_revision=1,
            supersedes_review_sha256=None, allowed_purpose='ctc-training-only',
            transcript_policy=TRANSCRIPT_POLICY, verdict='rejected', acoustic_complete=True,
            speech_present=True, kind='positive', keyword_id=1, actual_text='你好小屋', actual_tokens=[])
        record.update(human_review_history=[human], human_conflict=True, historical_human_conflict=True)
        before = copy.deepcopy(record)
        with self.assertRaisesRegex(ValueError, 'latest human observation'): validate_consensus_record(record)
        self.assertEqual(record, before)
        second = {**human, 'review_revision': 2, 'supersedes_review_sha256': canonical_sha256(human),
                  'verdict': 'accepted', 'actual_text': record['actual_text'], 'actual_tokens': record['actual_tokens']}
        record['human_review_history'].append(second); record['human_conflict'] = False
        validate_consensus_record(record)
        self.assertEqual(record['human_review_history'][0]['actual_text'], '你好小屋')


    def test_pending_machine_receipt_cannot_use_existing_human_admission(self):
        from asr_consensus_fixtures import consensus_fixture
        from speech_asr_consensus import admission_from_consensus
        from speech_label_admission import validate_admission, validate_review_record
        record = consensus_fixture(); admission = admission_from_consensus(record, 'f' * 64)
        args = dict(mode=REAL_MODE, source_id=record['source_id'], file_sha256=record['file_sha256'],
                    pcm_sha256=record['pcm_sha256'], text=record['actual_text'], tokens=record['actual_tokens'],
                    kind=record['kind'], keyword_id=record['keyword_id'], provenance=record, split='train', frames=160)
        with self.assertRaisesRegex(ValueError, 'PENDING_EXTERNAL_VERIFICATION'):
            validate_admission(admission, **args)
        for mutation in ({'ctc_training_allowed': True}, {'evidence_class': 'speech-like-audio-review-v2'},
                         {'label_origin': 'human'}, {'authentication_independently_verified': True}):
            altered = {**admission, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_admission(altered, **args)
        with self.assertRaisesRegex(ValueError, 'review v2'): validate_review_record(record)
        # A human-looking outer envelope cannot erase explicit machine lineage.
        human_envelope = {'mode': REAL_MODE, 'ctc_training_allowed': True, 'allowed_purpose': 'ctc-training-only'}
        for provenance in ({'label_origin': 'machine-consensus'}, {'label_source': 'machine-three-asr-consensus-v1'},
                           {'consensus_record': record}, {'status': 'PENDING_EXTERNAL_VERIFICATION'}):
            with self.subTest(provenance=provenance), self.assertRaisesRegex(ValueError, 'PENDING_EXTERNAL_VERIFICATION'):
                validate_admission(human_envelope, **{**args, 'provenance': provenance})

    def test_human_producer_cannot_overwrite_pending_source_receipt(self):
        from unittest.mock import patch
        from asr_consensus_fixtures import consensus_fixture
        from speech_asr_consensus import admission_from_consensus
        record = consensus_fixture()
        source = {'source_id': record['source_id'], 'file_sha256': record['file_sha256'],
                  'pcm_sha256': record['pcm_sha256'], 'source_family_id': record['source_family_id'],
                  'admission': admission_from_consensus(record, 'f' * 64)}
        human = dict(schema_version=2, evidence_class='speech-like-audio-review-v2',
            source_id=record['source_id'], file_sha256=record['file_sha256'], pcm_sha256=record['pcm_sha256'],
            source_family_id=record['source_family_id'], intended_text=record['intended_text'],
            reviewer_id='invented-human', review_origin='human', review_revision=1,
            supersedes_review_sha256=None, allowed_purpose='ctc-training-only', transcript_policy=TRANSCRIPT_POLICY,
            verdict='accepted', acoustic_complete=True, speech_present=True, kind='positive', keyword_id=1,
            actual_text=record['actual_text'], actual_tokens=record['actual_tokens'])
        before = copy.deepcopy(source)
        with patch.object(plan, 'load_jsonl', return_value=[human]), \
             patch.object(plan, 'sha256_file', return_value='f' * 64):
            with self.assertRaisesRegex(ValueError, 'PENDING_EXTERNAL_VERIFICATION'):
                plan._review_admissions(pathlib.Path('/invented/review'),
                                        {'key': {'text': record['intended_text']}}, {'key': source})
        self.assertEqual(source, before)

    def test_readonly_inspector_binds_source_role_origin_rights_and_frames(self):
        from unittest.mock import patch
        from asr_consensus_fixtures import consensus_fixture
        record = consensus_fixture()
        source = {**record['source_provenance'], 'source_id': record['source_id'],
                  'source_family_id': record['source_family_id'], 'file_sha256': record['file_sha256'],
                  'pcm_sha256': record['pcm_sha256'], 'audio': '/invented/unused.wav'}
        measured = {'file_sha256': record['file_sha256'], 'pcm_sha256': record['pcm_sha256'], 'frames': 160}
        intent = {'split': 'train', 'provider_group': 'train', 'text': record['intended_text']}
        with patch.object(plan, 'load_planned_recordings', return_value=({'key': intent}, {'key': source}, {})), \
             patch.object(plan, 'load_jsonl', return_value=[record]), \
             patch.object(plan, 'sha256_file', return_value='f' * 64), \
             patch.object(plan, 'inspect_pcm16_wav', return_value=measured):
            result = plan.inspect_consensus_recordings(pathlib.Path('/invented/intents'), pathlib.Path('/invented'),
                                                       pathlib.Path('/invented/receipts'), expected_count=1)
            self.assertEqual(result['label_eligible_recordings'], 1)
            self.assertEqual(result['training_admitted_recordings'], 0)
            self.assertFalse(result['ctc_training_allowed'])
            for field, changed in [('synthetic', False), ('source_role', 'source_screen_only'),
                                   ('rights_evidence_sha256', '8' * 64), ('source_type', 'recorded-speech')]:
                original = source[field]; source[field] = changed
                with self.subTest(field=field), self.assertRaises(ValueError):
                    plan.inspect_consensus_recordings(pathlib.Path('/invented/intents'), pathlib.Path('/invented'),
                                                      pathlib.Path('/invented/receipts'))
                source[field] = original
            for metadata in (source, intent):
                for field in ('role', 'allowed_role', 'split_role'):
                    metadata[field] = 'source_screen_only'
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        plan.inspect_consensus_recordings(pathlib.Path('/invented/intents'), pathlib.Path('/invented'),
                                                          pathlib.Path('/invented/receipts'))
                    del metadata[field]
            measured['frames'] = 159
            with self.assertRaises(ValueError):
                plan.inspect_consensus_recordings(pathlib.Path('/invented/intents'), pathlib.Path('/invented'),
                                                  pathlib.Path('/invented/receipts'))

    def test_diagnostic_reference_projection_preserves_machine_markers(self):
        from render_domains import domain_reference_row
        from speech_asr_consensus import require_independent_reference_labels
        for marker in ({'evidence_class': 'speech-three-asr-consensus-v1'},
                       {'label_origin': 'machine-consensus'}, {'label_source': 'machine-three-asr-consensus-v1'},
                       {'status': 'PENDING_EXTERNAL_VERIFICATION'}, {'consensus_record': {'fixture': True}},
                       {'source_provenance': {'label_origin': 'machine-consensus'}},
                       {'label_provenance': {'label_origin': 'machine-consensus'}}):
            row = dict(path='/invented/unused.wav', kind='negative', scene={}, domain_id='invented', **marker)
            reference = domain_reference_row(row, recording='invented', frames=160)
            for key, value in marker.items(): self.assertEqual(reference[key], value)
            with self.assertRaisesRegex(ValueError, 'independent reference'):
                require_independent_reference_labels([reference])

    def test_qualification_paths_reject_machine_gold_but_diagnostics_remain_available(self):
        import ast
        import hashlib
        from asr_consensus_fixtures import consensus_fixture
        from speech_asr_consensus import admission_from_consensus, require_independent_reference_labels
        from qualification_metrics import validate_event_metrics
        admission = admission_from_consensus(consensus_fixture(), 'f' * 64)
        for row in ({'admission': admission}, {'speech_like_provenance': {'label_origin': 'machine-consensus'}},
                    {'label_source': 'machine-three-asr-consensus-v1'},
                    {'admission': {'review_history': [{'label_origin': 'machine-consensus'}]}}):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'independent reference'):
                require_independent_reference_labels([row])
        raw = {'references': json.dumps({'admission': admission}) + '\n', 'detections': ''}
        hashes = {key + '_sha256': hashlib.sha256(value.encode()).hexdigest() for key, value in raw.items()}
        with self.assertRaisesRegex(ValueError, 'independent reference'):
            validate_event_metrics({}, raw, hashes)
        for filename, minimum in [('validate_real_human_corpus.py', 1), ('score_real_human_qualification.py', 2),
                                  ('run_final_afe_corpus.py', 1)]:
            tree = ast.parse((ROOT / 'tools' / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'require_independent_reference_labels']
            self.assertGreaterEqual(len(calls), minimum)
        # Generic score_events stays available for development diagnostics.
        self.assertNotIn('require_independent_reference_labels', (ROOT / 'eval/score_events.py').read_text())


if __name__ == '__main__':
    unittest.main()
