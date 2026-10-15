import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from effect_dataset_split import plan, quality_tier, check_role_eligibility


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



class RoleEligibilityTests(unittest.TestCase):
    def candidate(self, name, role='calibration'):
        return {**reviewed(name), 'evaluation_role': role, 'history_complete': True,
                'exposure_history': [], 'source_kind': 'real', 'history_evidence_id': 'fixture-history',
                'pcm_sha256': __import__('hashlib').sha256(name.encode()).hexdigest()}

    def test_explicit_roles_are_separate_and_do_not_change_input(self):
        rows = [self.candidate(role, role) for role in ('calibration', 'selection', 'sealed_test')]
        original = copy.deepcopy(rows)
        report = check_role_eligibility(rows)
        self.assertEqual(rows, original)
        for row in report['rows']:
            self.assertEqual([r for r, ok in row['role_eligibility'].items() if ok], [row['record_id']])
            self.assertTrue(row['fresh_heldout'])
            self.assertFalse(row['human_gold'])
            self.assertFalse(row['training_admitted'])

    def test_machine_and_human_boolean_cannot_claim_gold(self):
        row = self.candidate('machine', 'sealed_test')
        row.pop('human_review')
        row.update(human_gold=True, asr_evidence=[{'family': f, 'status': 'complete', 'raw_text': '你好小窝'} for f in ('a','b','c')])
        self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()))
        row['human_review'] = reviewed('x')['human_review']
        row['label_origin'] = 'machine-consensus'
        self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()))

    def test_nested_machine_history_cannot_be_laundered(self):
        for extra in ({'machine_train_only_candidate': True},
                      {'three_model_training_candidate': True},
                      {'admission': {'label_origin': 'machine-consensus'}},
                      {'source_kind': 'synthetic'}, {'history_evidence_id': None}):
            row = {**self.candidate('x'), **extra}
            self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()), extra)

    def test_non_boolean_flags_never_allow_evaluation(self):
        fields = ('previously_used_for_training', 'previously_used_for_tuning',
                  'development_only', 'previously_exposed', 'test_results_seen',
                  'machine_train_only_candidate', 'three_model_training_candidate',
                  'history_complete', 'human_gold')
        for field in fields:
            for value in (0, 1, 'true', 'false', '', None, [], {}):
                row = self.candidate('x', 'sealed_test'); row[field] = value
                report = check_role_eligibility([row])['rows'][0]
                self.assertFalse(any(report['role_eligibility'].values()), (field, value))
                self.assertFalse(report['fresh_heldout'])
                nested = self.candidate('x', 'sealed_test')
                nested['admission'] = {field: value}
                self.assertFalse(any(check_role_eligibility([nested])['rows'][0]['role_eligibility'].values()), (field, value))

    def test_nested_exposure_cannot_be_cleared_by_outer_false(self):
        for field in ('previously_used_for_training', 'previously_used_for_tuning',
                      'development_only', 'previously_exposed', 'test_results_seen'):
            row = self.candidate('x', 'sealed_test')
            row[field] = False; row['admission'] = {field: True}
            self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()))

    def test_nested_machine_aliases_and_unknown_labels_fail_closed(self):
        for extra in ({'label_origin': 'machine_consensus'}, {'label_origin': 'invented'},
                      {'label_source': 'invented'}, {'tier': 'gold'},
                      {'quality_tier': 'invented'}, {'tier': None}):
            for nested in (False, True):
                row = self.candidate('x', 'sealed_test')
                row.update({'admission': extra} if nested else extra)
                self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()), extra)

    def test_historical_development_never_proves_fine_roles(self):
        for value in ('development', 'dev', 'validation'):
            row = self.candidate('x'); row['split'] = value
            report = check_role_eligibility([row])
            self.assertFalse(any(report['rows'][0]['role_eligibility'].values()))
            self.assertEqual(report['rows'][0]['input']['split'], value)
            self.assertEqual(report['historical_support'][0]['historical_roles'], ['development'])

    def test_exposed_support_cannot_become_fresh_heldout(self):
        for role in ('train', 'screening', 'target_informed_selection', 'selection', 'sealed_test'):
            row = self.candidate('x', 'sealed_test')
            row['exposure_history'] = [{'role': role, 'evidence': 'retained'}]
            result = check_role_eligibility([row])
            self.assertFalse(result['rows'][0]['role_eligibility']['sealed_test'])
            self.assertFalse(result['rows'][0]['fresh_heldout'])
            self.assertEqual(result['historical_support'][0]['exposure_history'], row['exposure_history'])

    def test_legacy_flags_block_fresh_evaluation(self):
        for flag in ('previously_used_for_training', 'previously_used_for_tuning', 'development_only', 'previously_exposed', 'test_results_seen'):
            row=self.candidate('x', 'sealed_test'); row[flag]=True
            self.assertFalse(check_role_eligibility([row])['rows'][0]['role_eligibility']['sealed_test'])

    def test_missing_unknown_or_incomplete_evidence_fails_closed(self):
        for extra in ({'speaker_id': 'unknown-voice'}, {'source_kind':'unknown'},
                      {'source_family_id':None}, {'history_complete':False},
                      {'exposure_history':None}, {'exposure_history':[{'role':'invented'}]},
                      {'exposure_history':['train']}, {'evaluation_role':'development'},
                      {'pcm_sha256':None}, {'audio_status':'missing'}):
            row={**self.candidate('x'), **extra}
            self.assertFalse(any(check_role_eligibility([row])['rows'][0]['role_eligibility'].values()), extra)

    def test_preset_links_across_engines_without_filling_speaker(self):
        a,b=self.candidate('a'),self.candidate('b','selection')
        a.update(engine='qwen06',preset_voice_id='same')
        b.update(engine='qwen17',preset_voice_id='same')
        self.assertEqual(plan([a,b])['groups'],1)
        result=check_role_eligibility([a,b])
        self.assertTrue(all(not any(r['role_eligibility'].values()) for r in result['rows']))
        a['speaker_id']=None
        self.assertEqual(plan([a])['rows'][0]['speaker_id'],None)
        self.assertFalse(any(check_role_eligibility([a])['rows'][0]['role_eligibility'].values()))

    def test_calibration_selection_source_overlap_and_transitive_history(self):
        a,b,c=self.candidate('a'),self.candidate('b','selection'),self.candidate('c')
        a['source_pcm_sha256']=b['pcm_sha256']
        b['derivation_family_id']=c['derivation_family_id']='shared'
        c['exposure_history']=[{'role':'train'}]
        report=check_role_eligibility([a,b,c])
        self.assertEqual(len({r['group_id'] for r in report['rows']}),1)
        self.assertTrue(all(not any(r['role_eligibility'].values()) for r in report['rows']))

    def test_exact_history_retained_but_not_fresh(self):
        row=self.candidate('x'); row['split']='calibration'
        row['exposure_history']=[{'role':'calibration','source':'original'}]
        result=check_role_eligibility([row])['rows'][0]
        self.assertTrue(result['role_eligibility']['calibration'])
        self.assertFalse(result['role_eligibility']['selection'])
        self.assertFalse(result['fresh_heldout'])

    def test_missing_assignment_never_fresh_or_eligible(self):
        row=self.candidate('x'); row.pop('evaluation_role')
        result=check_role_eligibility([row])['rows'][0]
        self.assertFalse(any(result['role_eligibility'].values()))
        self.assertFalse(result['fresh_heldout'])

    def test_unassigned_linked_member_is_not_given_an_evaluation_role(self):
        a,b = self.candidate('a'), self.candidate('b')
        b.pop('evaluation_role'); b['source_pcm_sha256'] = a['pcm_sha256']
        report = check_role_eligibility([a,b])
        self.assertTrue(all(not any(r['role_eligibility'].values()) for r in report['rows']))

    def test_unknown_history_and_conflicting_original_partitions_still_raise(self):
        for extra in ({'split':'unrecognized'}, {'split':'train','previous_split':'test'}):
            with self.assertRaises(ValueError):
                check_role_eligibility([{**self.candidate('x'), **extra}])

    def test_required_roles_validated_and_deterministic_order(self):
        for roles in ((), ('train',), ('selection','selection')):
            with self.assertRaises(ValueError):check_role_eligibility([],roles)
        rows=[self.candidate('a'),self.candidate('b','selection')]
        self.assertEqual(check_role_eligibility(rows),check_role_eligibility(reversed(rows)))


if __name__ == '__main__':
    unittest.main()
