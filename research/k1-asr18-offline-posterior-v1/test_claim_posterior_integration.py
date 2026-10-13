"""Source-bound owner-v2 normal/recovered custody and isolated offline CLI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import asr_posterior as a
import fixture_cases as f
from test_offline_custody import write_bundle
import test_offline_custody as custody


class ClaimPosteriorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.normal=f.normal_case();cls.recovered=f.recovery_case()
    def accept(self,case):return a.accept(**case.arguments())
    def test_normal_and_recovered_complete_owner_history2_custody(self):
        for case in (self.normal,self.recovered):
            with self.subTest(mode=case.expected['mode']):
                result=self.accept(case)
                self.assertEqual(result['producer_history']['total_count'],2)
                self.assertEqual(result['verified_complete_slots'],18)
                self.assertIsNone(result['consumed_attempts'])
                self.assertEqual(result['network_calls'],0)
                self.assertFalse(result['original_publisher_reclassified'])
    def test_independent_owner_raw_ref_commit_required_and_exact(self):
        for role in ('claim_ref','claim_commit'):
            for change in ('missing','raw_bytes','owner_target'):
                case=copy.deepcopy(self.normal)
                if change=='missing':case.records.pop(role)
                elif change=='raw_bytes':case.records[role]+=b' '
                else:
                    value=a.decode(case.records[role])
                    if role=='claim_ref':value['object']['sha']='0'*40
                    else:value['parents'][0]['sha']='0'*40
                    raw=a.canonical(value);case.records[role]=raw
                    case.expected['claim_owner']['ref_response_sha256' if role=='claim_ref' else 'commit_response_sha256']=a.sha(raw)
                with self.subTest(role=role,change=change),self.assertRaises(ValueError):self.accept(case)
    def test_root_expected_v1_cannot_downgrade_v2_source(self):
        case=copy.deepcopy(self.normal);case.expected['schema']='k1-asr18-offline-acceptance-v1';case.expected.pop('claim_owner')
        case.records['producer_history']['total_count']=1;case.records['producer_history']['workflow_runs']=case.records['producer_history']['workflow_runs'][:1]
        with self.assertRaises(ValueError):self.accept(case)
    def test_wrong_external_scope_workflow_run_attempt_and_raw_history_reject(self):
        for change in ('workflow','ref','run','attempt','job','history_raw'):
            case=copy.deepcopy(self.normal)
            if change=='workflow':case.expected['claim_owner']['workflow']='.github/workflows/other.yml'
            elif change=='ref':case.expected['claim_owner']['ref']='refs/heads/claims/other'
            elif change=='run':case.records['producer_run']['run_number']=2
            elif change=='attempt':case.records['producer_run']['run_attempt']=2
            elif change=='job':case.records['producer_jobs']['jobs'][0]['id']+=1
            else:case.records['_response_bodies']['producer_history']=b'{}'
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)
    def test_recovery_own_history2_stays_forbidden(self):
        case=copy.deepcopy(self.recovered);page=case.records['recovery_history'];page['total_count']=2;page['workflow_runs']*=2
        with self.assertRaises(ValueError):self.accept(case)
    def test_claim_does_not_upgrade_failed_native_or_missing_input(self):
        for name in ('first_native_failure18','all_notrun'):
            result=self.accept(f.normal_case(name))
            self.assertFalse(result['native_success_established']);self.assertEqual(result['verified_complete_slots'],0)
            self.assertIsNone(result['consumed_attempts'])
    def test_fresh_isolated_cli_normal_and_recovery_use_external_custody(self):
        for case in (self.normal,self.recovered):
            with tempfile.TemporaryDirectory() as folder:
                args=write_bundle(Path(folder),case)
                result=custody.CustodyTests().run_cli(args)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                value=a.decode(result.stdout)
                self.assertEqual(value['producer_history']['total_count'],2)
                self.assertTrue({'claim_ref','claim_commit'} <= {row['role'] for row in value['input_roles']})
                if case.expected['recovery'] is not None:
                    self.assertEqual(value['producer_terminal']['publisher_step_conclusion'],'failure')

if __name__=='__main__':unittest.main()
