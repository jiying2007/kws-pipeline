"""Offline migration, strict schema and real preparation-entrypoint regressions."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import campaign_capsule as c
import prepare_campaign as p

# Original revision-bound consumer imports are test-only; the new library never
# imports or silently rewires these modules.
sys.path.insert(0, str(ROOT / 'research/k1-punctuation6-asr18-v1'))
import campaign_contract as legacy
import prepare_prerequisite as prerequisite
import test_control_fixture as historical

FUTURE = 'k1-punctuation6-asr18-r6'
POLICY = 'k1-asr18-scientific-v1'


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.index, policies = p.load_catalog()
        self.policy = policies[POLICY]
        self.row = self.index['policies'][0]
        self.draft = c.new_draft(FUTURE, self.index, policy_id=POLICY)

    def controls(self, draft=None, policy=None):
        return c.disabled_controls(self.draft if draft is None else draft, self.index,
                                   self.policy if policy is None else policy, expected_campaign=FUTURE)

    def test_current_catalog_has_no_active_campaign(self):
        self.assertIsNone(self.index['active_campaign'])
        self.assertIn('k1-punctuation6-asr18-r4', self.index['sealed_campaigns'])
        self.assertIn('k1-punctuation6-asr18-r5', self.index['sealed_campaigns'])

    def test_original_scientific_digest_is_exactly_equal(self):
        freeze = legacy.read_json(ROOT / 'research/k1-punctuation6-asr18-v1/execution-freeze.json')
        with self.assertRaisesRegex(ValueError, 'verified scientific source byte identity'):
            prerequisite.scientific_identity(ROOT, freeze)
        with historical.historical_source_tree(ROOT) as source:
            self.assertEqual(c.scientific_identity(self.policy), prerequisite.scientific_identity(source, freeze))
        self.assertEqual(len(self.policy['scientific_projection']['source_files']), 47)
        self.assertEqual(c.canonical(self.policy['scientific_projection']['limits']), legacy.canonical(legacy.LIMITS))

    def test_scientific_body_verification_uses_real_bytes(self):
        with self.assertRaisesRegex(ValueError, 'source bytes drift'):
            c.verify_scientific_members(self.policy, lambda path: (ROOT / path).read_bytes())
        with historical.historical_source_tree(ROOT) as source:
            result = c.verify_scientific_members(self.policy, lambda path: (source / path).read_bytes())
            self.assertEqual(result, self.row['scientific_sha256'])
        with self.assertRaisesRegex(ValueError, 'source bytes drift'):
            c.verify_scientific_members(self.policy, lambda path: b'only an asserted digest')
        # This test-only projection must fail if an exact historical body is
        # missing, changed, or replaced with the maintained runtime bytes.
        read = legacy.read_bytes
        for name in sorted(historical.HISTORICAL_RETAINED):
            archive = ROOT / historical.HISTORICAL_ARCHIVE / name
            for label, replacement in (('missing', None), ('changed', b'changed archive'),
                                       ('current', (ROOT / name).read_bytes())):
                def changed_read(path, **kwargs):
                    if Path(path) == archive:
                        if replacement is None:
                            raise FileNotFoundError(path)
                        return replacement
                    return read(path, **kwargs)
                with self.subTest(member=name, replacement=label), \
                     patch.object(legacy, 'read_bytes', side_effect=changed_read), \
                     self.assertRaises((ValueError, OSError)):
                    with historical.historical_source_tree(ROOT):
                        self.fail('invalid archive supplied a historical source tree')

    def test_schema_does_not_duplicate_activation_in_science(self):
        before = c.scientific_identity(self.policy)
        controls = self.controls()
        controls['execution-release.json']['enabled'] = True
        controls['claim-admission-control.json']['campaign'] = 'invented'
        self.assertEqual(c.scientific_identity(self.policy), before)
        self.assertFalse(any('release.json' in path for path in self.policy['scientific_projection']['source_files']))

    def test_four_controls_are_deterministic_and_isolated(self):
        first, second = self.controls(), self.controls()
        self.assertEqual(c.canonical(first), c.canonical(second))
        first['publication-release.json']['effective_execution_envelope']['workflow_minutes'] = 999
        self.assertEqual(self.policy['effective_execution_envelope']['workflow_minutes'], 80)
        self.assertEqual(second['publication-release.json']['effective_execution_envelope']['workflow_minutes'], 80)

    def test_all_execution_review_and_retry_approvals_remain_false(self):
        controls = self.controls()
        keys = {'enabled', 'independent_source_review', 'root_execution_release', 'root_publication_release',
                'root_transfer_recovery_release', 'retry_authorized', 'new_native_calls_authorized',
                'historical_screening_reviewed'}
        for value in controls.values():
            for key in keys & value.keys():
                self.assertIs(value[key], False)
        self.assertTrue(all(controls['execution-release.json'][key] is None for key in c.INPUT_AUTHORITIES))
        self.assertIs(controls['transfer-recovery-release.json']['authority'], None)

    def test_generated_claim_pin_binds_actual_projected_bytes(self):
        controls = self.controls()
        self.assertEqual(controls['execution-release.json']['claim_admission_control_sha256'],
                         c.digest(c.canonical(controls['claim-admission-control.json'])))

    def test_generated_controls_pass_original_release_shape(self):
        controls = self.controls()
        legacy.validate_release_shape(controls['execution-release.json'])
        legacy.validate_claim_control(controls['claim-admission-control.json'])
        self.assertEqual(controls['execution-release.json']['max_new_asr_calls'], 18)
        self.assertEqual(controls['execution-release.json']['old_calls_allowed'], 0)

    def test_generated_draft_is_rejected_by_original_execution_gate(self):
        with self.assertRaisesRegex(ValueError, 'disabled|review absent'):
            legacy.require_execution_release(self.controls()['execution-release.json'], {})

    def test_missing_one_approval_remains_rejected_by_original_consumer(self):
        controls = self.controls(); release = controls['execution-release.json']
        for key in legacy.RELEASE_APPROVALS:
            release[key] = True
        for key in legacy.RELEASE_AUTHORITIES:
            release[key] = 'a' * 64
        for missing in legacy.RELEASE_APPROVALS:
            changed = copy.deepcopy(release); changed[missing] = False
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                legacy.validate_worker_admission(changed, {}, {}, {})

    def test_future_scope_cannot_bypass_compiled_legacy_scope(self):
        control = self.controls()['claim-admission-control.json']
        control.update({key: True for key in legacy.CLAIM_APPROVALS})
        with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
            legacy.require_current_scope(control)

    def test_original_wrong_budget_bool_and_retry_still_rejected(self):
        release = self.controls()['execution-release.json']
        for key, value in (('max_new_asr_calls', 19), ('max_new_asr_calls', True), ('old_calls_allowed', 1),
                           ('retries', 1), ('run_attempt', 2)):
            changed = copy.deepcopy(release); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                legacy.validate_release_shape(changed)

    def test_sealed_r4_r5_and_old_campaigns_cannot_reopen(self):
        for name in self.index['sealed_campaigns']:
            with self.subTest(campaign=name), self.assertRaises(ValueError):
                c.new_draft(name, self.index, policy_id=POLICY)

    def test_unknown_campaign_shape_or_policy_rejected(self):
        for name in ('unknown', 'k1-punctuation6-asr18-r0', 'k1-punctuation6-asr18-r06', FUTURE + '/../../x'):
            with self.subTest(campaign=name), self.assertRaises(ValueError):
                c.new_draft(name, self.index, policy_id=POLICY)
        with self.assertRaises(ValueError):
            c.new_draft(FUTURE, self.index, policy_id='unreviewed-policy')

    def test_foreign_identity_and_old_claim_injection_rejected(self):
        for field, value in (('claim_owner', {'schema': 'k1-asr18-once-source-claim-v1'}),
                             ('source_binding', {'executed_head_sha': 'a' * 40}),
                             ('execution_authority', 'approved'), ('status', 'enabled')):
            changed = copy.deepcopy(self.draft); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.validate_draft(changed, self.index, expected_campaign=FUTURE)
        with self.assertRaises(ValueError):
            c.validate_draft(self.draft, self.index, expected_campaign='k1-punctuation6-asr18-r7')

    def test_every_unknown_or_missing_draft_field_is_rejected(self):
        changed = copy.deepcopy(self.draft); changed['enabled'] = True
        with self.assertRaises(ValueError): c.validate_draft(changed, self.index, expected_campaign=FUTURE)
        for key in self.draft:
            changed = copy.deepcopy(self.draft); changed.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_draft(changed, self.index, expected_campaign=FUTURE)

    def test_nested_identity_unknown_fields_are_rejected(self):
        for name, value in (('repository', 'foreign/repo'), ('workflow', '.github/workflows/other.yml'),
                            ('job_key', 'different'), ('extra', 'not allowed')):
            changed = copy.deepcopy(self.draft); changed['identity'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.validate_draft(changed, self.index, expected_campaign=FUTURE)

    def test_no_actual_authority_can_be_inserted_into_draft(self):
        for key in c.INPUT_AUTHORITIES:
            changed = copy.deepcopy(self.draft); changed['input_authorities'][key] = 'a' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.controls(draft=changed)

    def test_policy_drift_cannot_be_hidden_by_draft_budget(self):
        changed = copy.deepcopy(self.policy)
        changed['scientific_projection']['limits']['new_starts'] = 19
        with self.assertRaisesRegex(ValueError, 'policy bytes'): self.controls(policy=changed)
        changed = copy.deepcopy(self.policy); changed['effective_execution_envelope']['workflow_minutes'] = 90
        with self.assertRaisesRegex(ValueError, 'policy bytes'): self.controls(policy=changed)
        changed = copy.deepcopy(self.draft); changed['policy_sha256'] = 'b' * 64
        with self.assertRaises(ValueError): self.controls(draft=changed)

    def test_policy_strict_type_and_extra_fields_under_reviewed_pin(self):
        for target, name, value in (('limits', 'new_starts', True), ('limits', 'extra_budget', 1),
                                   ('envelope', 'workflow_minutes', True)):
            policy = copy.deepcopy(self.policy)
            mapping = policy['scientific_projection']['limits'] if target == 'limits' else policy['effective_execution_envelope']
            mapping[name] = value
            raw = c.canonical(policy) + b'\n'; row = dict(self.row, **c.identity(raw))
            row['scientific_sha256'] = c.scientific_identity(policy)
            with self.subTest(name=name), self.assertRaises(ValueError): c.validate_policy(raw, row)

    def test_duplicate_json_nonfinite_and_nonbytes_rejected(self):
        for raw in (b'{"enabled":false,"enabled":true}', b'{"x":NaN}', b'{"x":Infinity}', '{}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): c.decode(raw)

    def test_index_cannot_select_active_or_drop_sealed_history(self):
        for key, value in (('active_campaign', FUTURE), ('sealed_campaigns', []),
                            ('sealed_campaigns', self.index['sealed_campaigns'][:-2] + self.index['sealed_campaigns'][-1:])):
            changed = copy.deepcopy(self.index); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_index(changed)

    def test_core_projection_has_no_network_process_or_legacy_import(self):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), \
             patch('subprocess.Popen', side_effect=AssertionError('process forbidden')):
            p.projected_bundle(self.draft, self.index, self.policy, campaign=FUTURE)
        source = (HERE / 'campaign_capsule.py').read_text()
        self.assertNotIn('import campaign_contract', source)
        self.assertNotIn('import prepare_prerequisite', source)


class PreparationEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'closed-draft'

    def create(self):
        return p.create_bundle(self.output, FUTURE, POLICY)

    def test_real_create_and_read_only_check_agree(self):
        first = self.create()
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        stats = {path.name: path.stat().st_mtime_ns for path in self.output.iterdir()}
        self.assertEqual(p.check_bundle(self.output, FUTURE), first)
        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, original)
        self.assertEqual({path.name: path.stat().st_mtime_ns for path in self.output.iterdir()}, stats)
        self.assertEqual(first['files'], 6)
        self.assertEqual(first['execution_authority'], 'absent')

    def test_cli_create_and_check_without_python_site_packages(self):
        command = [sys.executable, '-I', '-S', '-B', str(HERE / 'prepare_campaign.py')]
        created = subprocess.run(command + ['create', '--campaign', FUTURE, '--output', str(self.output)],
                                 check=True, capture_output=True)
        checked = subprocess.run(command + ['check', '--campaign', FUTURE, '--bundle', str(self.output)],
                                 check=True, capture_output=True)
        self.assertEqual(created.stdout, checked.stdout)
        self.assertEqual(created.stderr, b'')

    def test_existing_directory_never_overwritten(self):
        self.create(); original = (self.output / 'capsule.json').read_bytes()
        with self.assertRaises(FileExistsError): self.create()
        self.assertEqual((self.output / 'capsule.json').read_bytes(), original)

    def test_each_derived_member_mutation_rejected(self):
        self.create()
        for path in self.output.iterdir():
            raw = path.read_bytes(); path.write_bytes(raw + b' ')
            with self.subTest(member=path.name), self.assertRaises(ValueError): p.check_bundle(self.output, FUTURE)
            path.write_bytes(raw)

    def test_unknown_extra_member_rejected(self):
        self.create(); (self.output / 'execution-freeze.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'member roster'): p.check_bundle(self.output, FUTURE)

    def test_missing_member_rejected(self):
        self.create(); (self.output / 'publication-release.json').unlink()
        with self.assertRaisesRegex(ValueError, 'member roster'): p.check_bundle(self.output, FUTURE)

    def test_symlink_and_hardlink_are_not_pinned_bytes(self):
        self.create(); target = self.output / 'execution-release.json'; raw = target.read_bytes()
        outside = Path(self.temp.name) / 'external.json'; outside.write_bytes(raw)
        target.unlink(); target.symlink_to(outside)
        with self.assertRaises((ValueError, OSError)): p.check_bundle(self.output, FUTURE)
        target.unlink(); os.link(outside, target)
        with self.assertRaises((ValueError, OSError)): p.check_bundle(self.output, FUTURE)

    def test_output_parent_symlink_and_traversal_rejected(self):
        alias = Path(self.temp.name) / 'alias'; alias.symlink_to(self.temp.name, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)): p.create_bundle(alias / 'draft', FUTURE, POLICY)
        with self.assertRaises(ValueError): p.directory_fd(Path(self.temp.name) / '..')

    def test_read_only_check_rejects_wrong_expected_identity(self):
        self.create()
        with self.assertRaises(ValueError): p.check_bundle(self.output, 'k1-punctuation6-asr18-r7')


if __name__ == '__main__':
    unittest.main()
