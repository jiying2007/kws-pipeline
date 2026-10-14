import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from effect_dataset_split import plan, quality_tier


def reviewed(name):
    return {'record_id': name, 'path': name + '.wav', 'audio_status': 'verified_pcm16',
            'speaker_id': name, 'session_id': name, 'source_family_id': name,
            'human_review': {'status': 'complete', 'reviewer_id': 'reviewer',
                             'listened_at_utc': '2026-10-14T00:00:00Z', 'heard_text': '你好小窝',
                             'asr_or_target_seen_before_listening': False}}


class SplitTests(unittest.TestCase):
    def test_unknown_never_sealed(self):
        rows = [reviewed(str(i)) for i in range(100)]
        for row in rows:
            row['speaker_id'] = 'unknown'
        self.assertEqual(plan(rows)['counts'], {'quarantine': 100})

    def test_real_partitions_and_stable_order(self):
        rows = [reviewed(str(i)) for i in range(100)]
        result = plan(rows)
        self.assertEqual(set(result['counts']), {'train', 'development', 'sealed_acceptance'})
        self.assertEqual(result, plan(list(reversed(rows))))
        self.assertFalse(result['product_qualification_allowed'])

    def test_transitive_augmentation_and_development_lock(self):
        rows = [reviewed(str(i)) for i in range(3)]
        rows[0].update(pcm_sha256='a'*64, development_only=True)
        rows[1].update(source_pcm_sha256='a'*64, derivation_family_id='augmentation')
        rows[2]['derivation_family_id'] = 'augmentation'
        result = plan(rows)
        self.assertEqual(result['groups'], 1)
        self.assertEqual(result['counts'], {'development': 3})

    def test_machine_consensus_not_gold_or_acceptance(self):
        row = reviewed('machine'); row.pop('human_review')
        row['asr_evidence'] = [{'family': f, 'status': 'complete', 'raw_text': '你好，小屋。'}
                               for f in ('qwen06', 'sensevoice', 'whisper')]
        result = plan([row])
        self.assertEqual(result['rows'][0]['quality_tier'], 'machine_consensus')
        self.assertEqual(result['counts'], {'train': 1})
        self.assertFalse(result['rows'][0]['training_admitted'])
        row['asr_evidence'][1]['raw_text'] = '你好小窝'
        self.assertEqual(plan([row])['counts'], {'quarantine': 1})

    def test_incomplete_not_disagreement(self):
        row = {'asr_evidence': [{'family': 'a', 'status': 'complete', 'raw_text': '你好'},
                                {'family': 'b', 'status': 'failed', 'raw_text': None}]}
        self.assertEqual(quality_tier(row), 'unreviewed_or_incomplete')

    def test_target_zero_not_six_errors(self):
        rows = []
        for i in range(6):
            row = reviewed(str(i)); row.pop('human_review')
            row['asr_evidence'] = [{'family': f, 'status': 'complete', 'raw_text': '你好小屋'}
                                   for f in ('a', 'b', 'c')]
            rows.append(row)
        self.assertEqual(plan(rows)['quality_counts'], {'machine_consensus': 6})

    def test_human_boolean_alone_no_authority(self):
        row = reviewed('x'); row.pop('human_review'); row['human_gold'] = True
        self.assertEqual(plan([row])['counts'], {'quarantine': 1})

    def test_different_roots_do_not_alias_by_relative_path(self):
        rows = [reviewed(str(i)) for i in range(2)]
        for i, row in enumerate(rows):
            row.update(path='x.wav', audio_root_id=str(i))
        self.assertEqual(plan(rows)['groups'], 2)

    def test_missing_audio_blocks_acceptance(self):
        row = reviewed('x'); row['audio_status'] = 'missing'
        self.assertEqual(plan([row])['counts'], {'quarantine': 1})

    def test_existing_holdouts_never_recycle_to_training(self):
        rows = [reviewed('test'), reviewed('qualification'), reviewed('calibration')]
        for row in rows:
            row['split'] = row['record_id']
        result = plan(rows)
        self.assertNotIn('train', result['counts'])
        self.assertEqual(result['counts'], {'development': 1, 'sealed_acceptance': 2})

    def test_existing_training_holdout_overlap_rejected(self):
        a, b = reviewed('a'), reviewed('b')
        a.update(split='train', source_family_id='shared')
        b.update(split='qualification', source_family_id='shared')
        with self.assertRaises(ValueError):
            plan([a, b])

    def test_all_historical_aliases_preserved_not_randomized(self):
        for alias, expected in [('development', 'development'), ('dev', 'development'),
                                ('selection', 'development'), ('train', 'train'),
                                ('training', 'train'), ('calibration', 'development')]:
            rows = [reviewed(str(i)) for i in range(100)]
            for row in rows:
                row['split'] = alias
            self.assertEqual(plan(rows)['counts'], {expected: 100}, alias)

    def test_qualification_development_flag_conflict(self):
        a, b = reviewed('a'), reviewed('b')
        a.update(split='qualification', source_family_id='shared')
        b.update(development_only=True, source_family_id='shared')
        with self.assertRaises(ValueError):
            plan([a, b])
        a['development_only'] = True
        with self.assertRaises(ValueError):
            plan([a])

    def test_training_and_tuning_history_never_sealed(self):
        for flag, expected in [('previously_used_for_tuning', 'development'),
                               ('previously_used_for_training', 'train')]:
            rows = [reviewed(str(i)) for i in range(100)]
            for row in rows:
                row[flag] = True
            self.assertEqual(plan(rows)['counts'], {expected: 100})

    def test_history_conflicts_and_unknown_roles_fail_closed(self):
        for extra in [{'split': 'train', 'planned_split': 'sealed_acceptance'},
                      {'split': 'sealed_acceptance', 'previous_split': 'dev'},
                      {'split': 'unrecognized-role'}]:
            with self.assertRaises(ValueError):
                plan([{**reviewed('a'), **extra}])

    def test_history_locks_unpartitioned_derivatives_and_is_idempotent(self):
        rows = [reviewed('a'), reviewed('b')]
        rows[0].update(split='selection', pcm_sha256='a' * 64)
        rows[1]['source_pcm_sha256'] = 'a' * 64
        report = plan(rows)
        self.assertEqual(report['counts'], {'development': 2})
        self.assertEqual(report, plan(report['rows']))

    def test_identity_placeholders_never_enter_new_sealed_partition(self):
        for pattern in ('unknown-voice{}', 'TBD', '?', 'pending_{}', 'unset:{}',
                        'unspecified {}', 'N/A', 'na', 'missing-{}'):
            rows = [reviewed(str(i)) for i in range(100)]
            for i, row in enumerate(rows):
                row['speaker_id'] = pattern.format(i)
            self.assertEqual(plan(rows)['counts'], {'quarantine': 100}, pattern)

    def test_identity_filter_not_applied_to_heard_natural_language(self):
        row = reviewed('speaker')
        row['human_review']['heard_text'] = 'unknown-voice is what I heard'
        self.assertEqual(quality_tier(row), 'human_review_recorded')

    def test_alias_namespaces_connect_source_families(self):
        for source, target in [('source_id', 'source_family'),
                               ('family_id', 'derivation_family_id'),
                               ('derivation_family', 'source_family_id')]:
            a, b = reviewed('a'), reviewed('b')
            a.update(development_only=True)
            a[source], b[target] = 'shared', 'shared'
            self.assertEqual(plan([a, b])['counts'], {'development': 2})

    def test_audio_hash_and_pcm_domains_are_separate(self):
        a,b=reviewed('a'),reviewed('b')
        a.update(development_only=True, file_sha256='a'*64)
        b.update(reference_audio_sha256='a'*64)
        self.assertEqual(plan([a,b])['groups'],1)
        b.pop('reference_audio_sha256'); b['pcm_sha256']='a'*64
        self.assertEqual(plan([a,b])['groups'],2)
        b['pcm_sha256']='invalid'
        with self.assertRaises(ValueError):plan([a,b])

    def test_duplicate_family_and_record_rejected(self):
        with self.assertRaises(ValueError):
            plan([reviewed('x'), reviewed('x')])
        with self.assertRaises(ValueError):
            quality_tier({'asr_evidence': [{'family': 'a', 'status': 'complete', 'raw_text': 'x'}] * 3})


if __name__ == '__main__':
    unittest.main()
