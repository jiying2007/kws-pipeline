#!/usr/bin/env python3
"""Tiny invented ZIP/WAV/JSON contracts only; no ASR, model, or numerical run."""
import contextlib
import copy
import io
import json
import pathlib
import sys
import unittest
import wave
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'training')]
import speech_like_corpus_plan as plan
from asr_resolver_fixtures import trusted_fixture
from corpus_identity import inspect_pcm16_wav
from speech_asr_consensus import LABEL_SOURCE
from speech_label_admission import REAL_MODE, FIXTURE_MODE, TRANSCRIPT_POLICY, fixture_admission


class TrustedProducerMetadataTests(unittest.TestCase):
    def setUp(self):
        manager = trusted_fixture()
        self.fixture = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        self.root, self.record = self.fixture.root, self.fixture.record
        self.generated = self.root / 'generated'
        self.intents = self.root / 'intents.jsonl'
        self.consensus = self.root / 'consensus.jsonl'
        self.reviews = self.root / 'reviews.jsonl'
        self.output = self.root / 'output'
        row = {'schema_version': 1, 'evidence_class': plan.MANIFEST_CLASS,
               **copy.deepcopy(self.record['source_provenance']),
               'provider_kind': 'offline-tts', 'provider_name': 'invented-fixture', 'provider_version': '1',
               'generation_config_sha256': 'a' * 64, 'voice_id': 'invented-machine-voice',
               'source_id': self.record['source_id'], 'source_family_id': self.record['source_family_id'],
               'audio': str(self.fixture.source), 'text': self.record['intended_text'],
               'tokens': copy.deepcopy(self.record['actual_tokens']), **inspect_pcm16_wav(self.fixture.source)}
        self.sources = [('train', 'train', row)]
        self.human_reviews = []
        self.records = [copy.deepcopy(self.record)]
        self.write()

    @staticmethod
    def dump(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

    def write(self):
        groups, intents = {}, []
        for group, split, row in self.sources:
            groups.setdefault(group, []).append(row)
            intents.append({**row, 'evidence_class': plan.INTENT_CLASS, 'provider_group': group,
                            'split': split, 'kind': 'positive', 'keyword_id': 1})
        for group, rows in groups.items():
            self.dump(self.generated / group / 'manifest.jsonl', rows)
        self.dump(self.intents, intents)
        self.dump(self.consensus, self.records)
        self.dump(self.reviews, self.human_reviews)

    def review_for(self, row):
        return {'schema_version': 2, 'evidence_class': 'speech-like-audio-review-v2',
                'source_id': row['source_id'], 'file_sha256': row['file_sha256'],
                'pcm_sha256': row['pcm_sha256'], 'source_family_id': row['source_family_id'],
                'reviewer_id': 'invented-reviewer-not-an-actual-listener',
                'intended_text': row['text'], 'actual_text': row['text'], 'actual_tokens': row['tokens'],
                'transcript_policy': TRANSCRIPT_POLICY, 'review_origin': 'human',
                'allowed_purpose': 'ctc-training-only', 'speech_present': True,
                'review_revision': 1, 'supersedes_review_sha256': None, 'acoustic_complete': True,
                'kind': 'positive', 'keyword_id': 1, 'verdict': 'accepted'}

    def add_human(self, *, group='train', split='train'):
        path = self.root / 'invented-human.wav'
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
            output.writeframes(b'\x01\x00' * 160)
        row = {**copy.deepcopy(self.sources[0][2]), 'audio': str(path), **inspect_pcm16_wav(path),
               'source_id': 'invented-human-source', 'source_family_id': 'invented-human-family',
               'voice_id': 'invented-human-voice'}
        for field in ('source_role', 'source_type', 'rights_evidence_sha256'):
            row.pop(field, None)
        self.sources.append((group, split, row))
        self.human_reviews.append(self.review_for(row))
        self.write()
        return row

    def materialize(self, **kwargs):
        return plan.materialize_labels(self.intents, self.generated, self.output,
                                       asr_consensus=self.consensus, evidence_dir=self.fixture.store, **kwargs)

    def admit(self, **kwargs):
        return plan.admit_reviewed_recordings(self.intents, self.generated,
                                             asr_consensus=self.consensus, evidence_dir=self.fixture.store, **kwargs)

    def test_real_resolver_machine_admission_and_portable_materialization(self):
        before = sorted(str(path) for path in self.root.rglob('*'))
        summary = self.admit(expected_count=1)
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
        self.assertEqual((summary['human_review_recordings'], summary['machine_consensus_recordings']), (0, 1))
        result = self.materialize()
        self.assertTrue(result['ctc_training_allowed'])
        self.assertEqual(result['recordings'], 1)
        self.assertEqual([result['splits'][split]['recordings'] for split in plan.SPLITS], [1, 0, 0, 0])
        labeled = json.loads((self.output / 'train/labeled-manifest.jsonl').read_text())
        raw = json.loads((self.output / 'train/manifest.jsonl').read_text())
        label = json.loads((self.output / 'train/labels.jsonl').read_text())
        self.assertNotIn('label_provenance_sha256', raw)
        self.assertEqual(labeled['label_source'], LABEL_SOURCE)
        self.assertEqual(label['label_source'], LABEL_SOURCE)
        self.assertEqual(labeled['label_origin'], 'machine-consensus')
        self.assertEqual(labeled['split'], 'train')
        self.assertEqual(labeled['text'], self.record['actual_text'])
        self.assertEqual(labeled['intended_text'], self.record['intended_text'])
        self.assertEqual(labeled['tokens'], self.record['actual_tokens'])
        self.assertNotIn('review_history', labeled['admission'])
        self.assertTrue(labeled['admission']['authentication_independently_verified'])
        self.assertEqual(labeled['admission']['source_verification']['record_sha256'],
                         plan.canonical_sha256(self.record))
        self.assertFalse(pathlib.Path(labeled['audio']).is_absolute())
        self.assertEqual((self.output / 'train' / labeled['audio']).read_bytes(), self.fixture.audio)
        self.assertEqual(result['splits']['train']['labeled_manifest_sha256'],
                         plan.sha256_file(self.output / 'train/labeled-manifest.jsonl'))

    def test_mixed_human_and_machine_coverage_keeps_distinct_label_bases(self):
        self.add_human()
        result = self.materialize(audio_review=self.reviews)
        self.assertEqual(result['recordings'], 2)
        summary = result['reviewed_admission']
        self.assertEqual((summary['human_review_recordings'], summary['machine_consensus_recordings']), (1, 1))
        rows = plan.load_jsonl(self.output / 'train/labeled-manifest.jsonl')
        self.assertEqual([row['label_source'] for row in rows], [LABEL_SOURCE, 'human-actual-transcript-v2'])
        self.assertIn('consensus_record', rows[0]['admission'])
        self.assertNotIn('review_history', rows[0]['admission'])
        self.assertIn('review_history', rows[1]['admission'])
        self.assertNotIn('consensus_record', rows[1]['admission'])

    def test_native_v2_private_uncertain_oov_history_survives_producer_and_base_validation(self):
        import asr_source_resolver as resolver
        from materialize_speech_like_base_index import materialize
        self.fixture.use_native_v2()
        private_note = 'invented private listening note: 屋 or 窝 remains uncertain'
        self.fixture.add_raw_observation('你好小屋', 'uncertain', original=private_note.encode())
        self.records = [copy.deepcopy(self.fixture.record)]
        self.write()
        with patch.object(resolver, '_SOURCE_ROOTS', (self.fixture.trust_root,)):
            result = self.materialize()
            manifest = self.output / 'train/labeled-manifest.jsonl'
            serialized = manifest.read_text()
            source = json.loads(serialized)
            consensus = source['admission']['consensus_record']
            self.assertEqual(consensus['schema_version'], 2)
            self.assertTrue(consensus['human_conflict'])
            self.assertTrue(source['admission']['ctc_training_allowed'])
            self.assertEqual(consensus['human_observation_history'], self.fixture.record['human_observation_history'])
            self.assertEqual(consensus['actual_text'], '你好小窝')
            for private in (private_note, '你好小屋', 'observed_text', 'reviewer_id'):
                self.assertNotIn(private, serialized)
            self.assertEqual(result['reviewed_admission']['machine_consensus_recordings'], 1)
            rows, summary = materialize(manifest=manifest, split='train',
                tokens_path=ROOT / 'keywords/tokens.example.txt',
                keywords_path=ROOT / 'keywords/zh_cn_example.tsv', evidence_dir=self.fixture.store)
            self.assertTrue(summary['ctc_training_allowed'])
            self.assertEqual(rows[0]['label_origin'], 'machine-consensus')
            self.assertEqual(rows[0]['admission'], source['admission'])
            self.assertEqual(rows[0]['actual_text'], '你好小窝')
            self.assertNotIn(private_note, json.dumps(rows, ensure_ascii=False))

    def test_missing_overlapping_and_duplicate_coverage_rejected_before_writes(self):
        self.add_human()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.materialize()
        self.human_reviews.append(self.review_for(self.sources[0][2])); self.write()
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            self.materialize(audio_review=self.reviews)
        self.human_reviews.pop(); self.records.append(copy.deepcopy(self.record)); self.write()
        with self.assertRaisesRegex(ValueError, 'duplicate machine'):
            self.materialize(audio_review=self.reviews)
        self.assertFalse(self.output.exists())

    def test_role_rights_identity_and_lineage_changes_fail_before_writes(self):
        original = copy.deepcopy(self.sources[0][2])
        for field, value in (('source_role', 'source_screen_only'), ('source_screen_only', True),
                             ('human_gold', True), ('review_origin', 'human'), ('synthetic', False),
                             ('source_type', 'recorded-speech'), ('license_id', 'unbound-license'),
                             ('rights_evidence_sha256', '0' * 64), ('pcm_sha256', '0' * 64),
                             ('source_id', 'different-source'), ('source_family_id', 'different-family'),
                             ('speaker_id', 'unbound-speaker'), ('label_origin', 'human'),
                             ('label_source', 'human-actual-transcript-v2'), ('actual_text', '小窝小窝'),
                             ('frames', 159), ('frames', True), ('schema_version', True)):
            self.sources[0] = ('train', 'train', {**copy.deepcopy(original), field: value}); self.write()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.materialize()
            self.assertFalse(self.output.exists())
        for field in self.record['source_provenance']:
            row = copy.deepcopy(original); del row[field]
            self.sources[0] = ('train', 'train', row); self.write()
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.materialize()
            self.assertFalse(self.output.exists())

    def test_machine_cannot_enter_heldout_splits_or_nontraining_provider_groups(self):
        row = self.sources[0][2]
        for group, split in (('train', 'calibration'), ('train', 'test'), ('train', 'qualification'),
                             ('generalization-search', 'train'), ('generalization-freeze', 'train')):
            self.sources[0] = (group, split, row); self.write()
            with self.subTest(group=group, split=split), self.assertRaisesRegex(ValueError, 'train-role'):
                self.materialize()
            self.assertFalse(self.output.exists())

    def test_fixed_source_actual_text_always_requires_explicit_supersession(self):
        source = self.sources[0][2]
        for text in (self.record['actual_text'], '你好小屋', None):
            source['actual_text'] = text
            self.write()
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'source-bound supersession'):
                self.materialize()
            self.assertFalse(self.output.exists())

    def test_machine_receipt_cannot_be_laundered_through_human_or_fixture_mode(self):
        row = self.sources[0][2]
        row['admission'] = self.fixture.admission()
        self.human_reviews = [self.review_for(row)]; self.write()
        with self.assertRaises(ValueError):
            plan.admit_reviewed_recordings(self.intents, self.generated, self.reviews)
        # Populate the other groups only to satisfy the original fixture-mode
        # manifest loader, retaining machine markers until its explicit guard.
        with self.assertRaises(ValueError):
            self.materialize(admission_mode=FIXTURE_MODE)
        with self.assertRaises(ValueError):
            self.materialize()
        row['admission'] = fixture_admission(); self.write()
        with self.assertRaisesRegex(ValueError, 'fixture/historical'):
            self.materialize()
        self.assertFalse(self.output.exists())

    def test_mixed_batch_checks_every_later_human_row_before_any_write(self):
        self.add_human()
        self.human_reviews[0]['verdict'] = 'rejected'; self.write()
        with self.assertRaises(ValueError):
            self.materialize(audio_review=self.reviews)
        self.assertFalse(self.output.exists())

    def test_cross_split_voice_and_lineage_still_block_mixed_batches(self):
        human = self.add_human(group='generalization-search', split='test')
        human['voice_id'] = self.sources[0][2]['voice_id']; self.write()
        with self.assertRaisesRegex(ValueError, 'voice_id crosses splits'):
            self.materialize(audio_review=self.reviews)
        human['voice_id'] = 'different-voice'
        human['source_family_id'] = self.sources[0][2]['source_family_id']
        self.human_reviews[0]['source_family_id'] = human['source_family_id']; self.write()
        with self.assertRaisesRegex(ValueError, 'source_family_id crosses splits'):
            self.materialize(audio_review=self.reviews)
        self.assertFalse(self.output.exists())

    def test_stale_evidence_is_rejected_before_materialization(self):
        self.fixture.archive.write_bytes(self.fixture.archive.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            self.materialize()
        self.assertFalse(self.output.exists())

    def test_batch_rehashes_before_writes_and_again_before_success(self):
        # Only file-copy/hash helpers are wrapped to change retained
        # bytes at a controlled point. Resolver/authority validators are real.
        original = self.fixture.archive.read_bytes()
        hash_value = plan.canonical_sha256
        changed = False
        def change_before_write(row):
            nonlocal changed
            result = hash_value(row)
            if isinstance(row, dict) and row.get('label_source') == LABEL_SOURCE and not changed:
                self.fixture.archive.write_bytes(original + b'changed-before-write')
                changed = True
            return result
        with patch.object(plan, 'canonical_sha256', side_effect=change_before_write):
            with self.assertRaisesRegex(ValueError, 'changed during verification session'):
                self.materialize()
        self.assertFalse(self.output.exists())
        self.fixture.archive.write_bytes(original)
        copyfile = plan._copy_audio_exclusive
        def change_during_write(source, target):
            result = copyfile(source, target)
            self.fixture.archive.write_bytes(original + b'changed-after-copy')
            return result
        with patch.object(plan, '_copy_audio_exclusive', side_effect=change_during_write):
            with self.assertRaisesRegex(ValueError, 'changed during verification session'):
                self.materialize()
        self.assertFalse(list(self.output.rglob('*.jsonl')))

    def test_source_target_and_exclusive_create_races_publish_no_success_metadata(self):
        self.add_human()
        original_source = self.fixture.source.read_bytes()
        copy_audio = plan._copy_audio_exclusive
        for race in ('source-changed', 'earlier-target-changed', 'target-created'):
            self.output = self.root / ('output-' + race)
            summary = self.output / 'summary.json'
            copied = []
            def racing_copy(source, target):
                if race == 'source-changed' and not copied:
                    source.write_bytes(b'changed-after-validation')
                if race == 'target-created':
                    target.write_bytes(b'created-after-preflight')
                result = copy_audio(source, target)
                if race == 'earlier-target-changed' and copied:
                    copied[0].write_bytes(b'earlier-copy-changed')
                copied.append(target)
                return result
            command = ['speech_like_corpus_plan.py', 'materialize', '--intents', str(self.intents),
                       '--generated-root', str(self.generated), '--asr-consensus', str(self.consensus),
                       '--asr-evidence-dir', str(self.fixture.store), '--audio-review', str(self.reviews),
                       '--output-root', str(self.output), '--summary', str(summary)]
            with self.subTest(race=race), patch.object(plan, '_copy_audio_exclusive', side_effect=racing_copy), \
                 patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(ValueError):
                plan.main()
            self.assertFalse(summary.exists())
            self.assertFalse(list(self.output.rglob('*.jsonl')))
            if race == 'target-created':
                targets = list(self.output.rglob('*.wav'))
                self.assertEqual(len(targets), 1)
                self.assertEqual(targets[0].read_bytes(), b'created-after-preflight')
            self.fixture.source.write_bytes(original_source)

    def test_final_session_exit_failure_never_publishes_prepared_admissions(self):
        original = self.fixture.archive.read_bytes()
        serialize = plan._jsonl_text
        changed = False
        def change_during_preparation(rows):
            nonlocal changed
            result = serialize(rows)
            if rows and rows[0].get('admission') and not changed:
                self.fixture.archive.write_bytes(original + b'changed-at-final-preparation')
                changed = True
            return result
        # Serialization is wrapped only to mutate evidence after the prewrite
        # checkpoint. The actual context exit must catch the changed bytes.
        with patch.object(plan, '_jsonl_text', side_effect=change_during_preparation), \
             self.assertRaisesRegex(ValueError, 'changed during verification session'):
            self.materialize()
        self.assertTrue(changed)
        self.assertFalse(list(self.output.rglob('*.jsonl')))

    def test_metadata_publication_failure_removes_newly_admitted_manifests(self):
        original_open = pathlib.Path.open
        def fail_second_metadata(path, mode='r', *args, **kwargs):
            if path == self.output / 'train/labels.jsonl' and mode == 'x':
                raise OSError('invented metadata publication failure')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(pathlib.Path, 'open', fail_second_metadata), \
             self.assertRaisesRegex(OSError, 'publication failure'):
            self.materialize()
        self.assertFalse(list(self.output.rglob('*.jsonl')))

    def test_metadata_collision_preserves_prior_and_concurrently_created_files(self):
        existing = self.output / 'train/labels.jsonl'
        existing.parent.mkdir(parents=True)
        existing.write_text('prior output')
        with self.assertRaisesRegex(ValueError, 'metadata target already exists'):
            self.materialize()
        self.assertEqual(existing.read_text(), 'prior output')
        self.assertFalse(list(self.output.rglob('*.wav')))
        existing.unlink()
        serialize = plan._jsonl_text
        created = False
        def create_during_preparation(rows):
            nonlocal created
            result = serialize(rows)
            if rows and rows[0].get('admission') and not created:
                existing.write_text('concurrently created output')
                created = True
            return result
        with patch.object(plan, '_jsonl_text', side_effect=create_during_preparation), \
             self.assertRaises(FileExistsError):
            self.materialize()
        self.assertTrue(created)
        self.assertEqual(existing.read_text(), 'concurrently created output')
        self.assertEqual(list(self.output.rglob('*.jsonl')), [existing])

    def test_cli_accepts_location_only_asr_options_and_inspector_stays_pending(self):
        command = ['speech_like_corpus_plan.py', 'admit-reviewed', '--intents', str(self.intents),
                   '--generated-root', str(self.generated), '--asr-consensus', str(self.consensus),
                   '--asr-evidence-dir', str(self.fixture.store), '--expected-count', '1']
        output = io.StringIO()
        with patch.object(sys, 'argv', command), contextlib.redirect_stdout(output):
            self.assertEqual(plan.main(), 0)
        self.assertEqual(json.loads(output.getvalue())['machine_consensus_recordings'], 1)
        before = sorted(str(path) for path in self.root.rglob('*'))
        inspected = plan.inspect_consensus_recordings(self.intents, self.generated, self.consensus, expected_count=1)
        self.assertEqual(inspected['status'], 'PENDING_EXTERNAL_VERIFICATION')
        self.assertEqual(inspected['training_admitted_recordings'], 0)
        self.assertFalse(inspected['ctc_training_allowed'])
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob('*')))
        with patch.object(sys, 'argv', command + ['--trust-root', str(self.root)]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            plan.main()


if __name__ == '__main__':
    unittest.main()
