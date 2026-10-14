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


REPOSITORY_HISTORY_LOCATOR = (
    'https://api.github.com/repos/jiying2007/kws-pipeline/actions/runs'
    '?branch=research%2Fk1-punctuation6-asr18-r4&per_page=100'
)


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


    def repository_case(self, case=None, *, include_foreign=True):
        case=copy.deepcopy(self.normal if case is None else case)
        page=case.records['producer_history']
        if include_foreign:
            other=copy.deepcopy(page['workflow_runs'][0])
            other.update(id=max(row['id'] for row in page['workflow_runs'])+100,
                workflow_id=other['workflow_id']+100,
                path='.github/workflows/other-public-audit.yml',
                head_sha='a'*40,event='workflow_dispatch',run_attempt=2)
            self.assertNotEqual(other['head_sha'],page['workflow_runs'][0]['head_sha'])
            page['workflow_runs'].insert(1,other)
            page['total_count']=len(page['workflow_runs'])
        case.records['_response_locators']={'producer_history':REPOSITORY_HISTORY_LOCATOR}
        self.sync_repository_body(case)
        return case

    def sync_repository_body(self, case):
        # Deliberate original, noncanonical bytes: never synthesize a workflow response.
        raw=json.dumps(case.records['producer_history'],indent=3).encode()+b'\n'
        case.records['_response_bodies']['producer_history']=raw
        return raw

    def repository_bundle(self, root, case, locator=REPOSITORY_HISTORY_LOCATOR):
        args=write_bundle(root,case)
        index=a.decode(args['custody_path'].read_bytes())
        row=next(row for row in index['entries'] if row['role']=='producer_history')
        row['locator']=locator
        args['custody_path'].write_bytes(a.canonical(index))
        args['custody_sha256']=a.sha(args['custody_path'].read_bytes())
        return args

    def assert_repository_custody(self, result, case):
        page=case.records['producer_history'];raw=case.records['_response_bodies']['producer_history']
        identity=a.t.decode(f.members(case.payload)['admission.json'])['owner']['identity']
        selected=[row for row in page['workflow_runs']
            if row['workflow_id']==identity['workflow_id'] and row['path']==identity['workflow']]
        excluded=[row for row in page['workflow_runs'] if row not in selected]
        projected={'total_count':len(selected),'workflow_runs':selected}
        expected={
            'schema':'k1-repository-branch-history-projection-v1',
            'resource_locator':REPOSITORY_HISTORY_LOCATOR,
            'raw_response_sha256':a.sha(raw),
            'original_total_count':page['total_count'],
            'original_rows':[a.c._observed_run_fields(row,identity) for row in page['workflow_runs']],
            'selected_run_ids':[row['id'] for row in selected],
            'excluded_run_ids':[row['id'] for row in excluded],
            'target':{'workflow_id':identity['workflow_id'],'workflow':identity['workflow'],
                'source_branch':identity['source_branch']},
            'projection_sha256':a.sha(a.c.canonical(projected)),
            'interpretation':'complete repository branch collection; explicit workflow-id-and-path projection; no workflow HTTP response claimed',
        }
        self.assertEqual(result['producer_history_custody'],expected)
        self.assertEqual(result['producer_history'],a.c.api_bound_history(identity,projected,
            'posterior',raw_sha256=a.sha(raw)))
        self.assertEqual(result['producer_history']['total_count'],2)
        self.assertEqual(result['verified_complete_slots'],18)
        self.assertEqual(result['network_calls'],0)
        self.assertEqual(result['model_calls'],0)
        self.assertIsNone(result['consumed_attempts'])
        self.assertFalse(result['new_attempts_authorized'])
        self.assertFalse(result['original_publisher_reclassified'])

    def test_repository_history_accepts_normal_and_recovered_target_only_page(self):
        for base in (self.normal,self.recovered):
            with self.subTest(mode=base.expected['mode']):
                case=self.repository_case(base,include_foreign=False)
                self.assert_repository_custody(self.accept(case),case)

    def test_repository_history_keeps_all_rows_raw_bytes_and_cross_workflow_ordinal(self):
        case=self.repository_case();before=copy.deepcopy(case.records)
        original=case.records['producer_history']['workflow_runs']
        self.assertEqual(original[0]['run_number'],original[1]['run_number'])
        self.assertNotEqual(original[0]['workflow_id'],original[1]['workflow_id'])
        with mock.patch.object(a.r,'_verify_run',wraps=a.r._verify_run) as verify:
            result=self.accept(case)
        self.assert_repository_custody(result,case)
        self.assertEqual(case.records,before)
        get,routes=verify.call_args.args[:2]
        route=routes['producer_history']
        self.assertEqual(route,REPOSITORY_HISTORY_LOCATOR.removeprefix('https://api.github.com'))
        self.assertEqual(get.response_bodies[route],before['_response_bodies']['producer_history'])
        self.assertEqual(a.decode(get.response_bodies[route]),before['producer_history'])

    def test_fresh_isolated_cli_repository_history_normal_and_recovery(self):
        for base in (self.normal,self.recovered):
            with self.subTest(mode=base.expected['mode']),tempfile.TemporaryDirectory() as folder:
                case=self.repository_case(base);args=self.repository_bundle(Path(folder),case)
                result=custody.CustodyTests().run_cli(args)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                value=a.decode(result.stdout);self.assert_repository_custody(value,case)
                history_role=next(row for row in value['input_roles'] if row['role']=='producer_history')
                self.assertEqual(history_role['locator'],REPOSITORY_HISTORY_LOCATOR)
                self.assertEqual(history_role['sha256'],a.sha(case.records['_response_bodies']['producer_history']))
                self.assertEqual((args['root']/'producer_history.bin').read_bytes(),
                    case.records['_response_bodies']['producer_history'])
                if case.expected['recovery'] is not None:
                    self.assertEqual(value['producer_terminal']['publisher_step_conclusion'],'failure')
                    self.assertEqual(value['recovery_terminal'],{'run':'success','job':'success','step':'success'})

    def test_fresh_isolated_cli_explicit_original_workflow_history_locator(self):
        for case in (self.normal,self.recovered):
            owner=case.expected['claim_owner']
            locator=('https://api.github.com/repos/jiying2007/kws-pipeline/actions/workflows/'
                +owner['workflow'].split('/')[-1]+'/runs?branch='+owner['source_branch']+'&per_page=100')
            self.assertIn(locator,a.locators(case.expected)['producer_history'])
            with self.subTest(mode=case.expected['mode']),tempfile.TemporaryDirectory() as folder:
                args=self.repository_bundle(Path(folder),case,locator)
                result=custody.CustodyTests().run_cli(args)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                value=a.decode(result.stdout)
                self.assertEqual(value['status'],'complete_native_verified')
                self.assertEqual(value['verified_complete_slots'],18)
                self.assertEqual(value['producer_history']['total_count'],2)
                self.assertNotIn('producer_history_custody',value)
                history_role=next(row for row in value['input_roles'] if row['role']=='producer_history')
                self.assertEqual(history_role['locator'],locator)
                self.assertEqual(history_role['sha256'],a.sha(case.records['_response_bodies']['producer_history']))
                if case.expected['recovery'] is not None:
                    self.assertEqual(value['producer_terminal']['publisher_step_conclusion'],'failure')

    def test_repository_history_rejects_one_sided_workflow_identity_matches(self):
        for change in ('matching_id_only','matching_path_only'):
            case=self.repository_case();rows=case.records['producer_history']['workflow_runs']
            foreign,target=rows[1],rows[0]
            if change=='matching_id_only':foreign.update(workflow_id=target['workflow_id'],run_number=3)
            else:foreign['path']=target['path']
            self.sync_repository_body(case)
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_truncated_or_oversized_page(self):
        for change in ('truncated','oversized','empty','boolean_count','string_count'):
            case=self.repository_case();page=case.records['producer_history']
            if change=='truncated':page['total_count']+=1
            elif change=='oversized':
                other=page['workflow_runs'][1];maximum=max(row['id'] for row in page['workflow_runs'])
                for index in range(98):
                    row=copy.deepcopy(other);row.update(id=maximum+index+1,run_number=index+2)
                    page['workflow_runs'].append(row)
                page['total_count']=len(page['workflow_runs'])
                self.assertEqual(page['total_count'],101)
            elif change=='empty':page.update(total_count=0,workflow_runs=[])
            elif change=='boolean_count':page['total_count']=True
            else:page['total_count']=str(page['total_count'])
            self.sync_repository_body(case)
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_duplicate_run_ids_or_within_workflow_ordinals(self):
        for change in ('duplicate_id_across_workflows','duplicate_target_ordinal','duplicate_foreign_ordinal'):
            case=self.repository_case();page=case.records['producer_history'];rows=page['workflow_runs']
            if change=='duplicate_id_across_workflows':rows[1]['id']=rows[0]['id']
            elif change=='duplicate_target_ordinal':rows[2]['run_number']=rows[0]['run_number']
            else:
                row=copy.deepcopy(rows[1]);row['id']+=1;rows.append(row);page['total_count']=len(rows)
            self.sync_repository_body(case)
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_foreign_branch_or_repository(self):
        for change in ('branch','repository','head_repository','private_repository','private_head_repository'):
            case=self.repository_case();row=case.records['producer_history']['workflow_runs'][1]
            if change=='branch':row['head_branch']='research/foreign'
            elif change.startswith('private_'):row[change[len('private_'):]]['private']=True
            else:row[change]['full_name']='other/public-repository'
            self.sync_repository_body(case)
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_malformed_foreign_row_fields(self):
        mutations=(('head_sha','not-a-git-sha'),('head_sha','A'*40),
            ('id',True),('id',0),('id','123'),('run_number',False),('run_number',0),
            ('run_attempt',False),('run_attempt',0),('workflow_id',True),('workflow_id',0),
            ('path','../other.yml'),('path','.github/workflows/other.txt'),
            ('event',''),('event','workflow_dispatch\n'),('status','invented'),('conclusion','invented'))
        for key,value in mutations:
            case=self.repository_case();case.records['producer_history']['workflow_runs'][1][key]=value
            self.sync_repository_body(case)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_wrong_target_source_event_or_attempt(self):
        for key,value in (('head_sha','b'*40),('event','workflow_dispatch'),('run_attempt',2)):
            case=self.repository_case()
            # Mutate the later target row, not the separately pinned owner run.
            case.records['producer_history']['workflow_runs'][2][key]=value
            self.sync_repository_body(case)
            with self.subTest(key=key),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_requires_owner_once_and_original_raw_body(self):
        for change in ('owner_absent','body_mismatch','body_missing','locator_missing'):
            case=self.repository_case();page=case.records['producer_history']
            if change=='owner_absent':
                page['workflow_runs'].pop(0);page['total_count']=len(page['workflow_runs']);self.sync_repository_body(case)
            elif change=='body_mismatch':case.records['_response_bodies']['producer_history']=b'{}'
            elif change=='body_missing':case.records['_response_bodies'].pop('producer_history')
            else:case.records.pop('_response_locators')
            with self.subTest(change=change),self.assertRaises(ValueError):self.accept(case)

    def test_repository_history_rejects_event_filtered_or_unpinned_locator(self):
        bad=(REPOSITORY_HISTORY_LOCATOR.replace('&per_page=100','&event=push&per_page=100'),
            REPOSITORY_HISTORY_LOCATOR.replace('&per_page=100','&status=completed&per_page=100'),
            REPOSITORY_HISTORY_LOCATOR+'&page=2',
            REPOSITORY_HISTORY_LOCATOR.replace('research%2Fk1-punctuation6-asr18-r4','research%2Fforeign'),
            REPOSITORY_HISTORY_LOCATOR.replace('/jiying2007/kws-pipeline/','/other/public-repository/'))
        for locator in bad:
            case=self.repository_case();case.records['_response_locators']['producer_history']=locator
            with self.subTest(locator=locator),self.assertRaises(ValueError):self.accept(case)
        with tempfile.TemporaryDirectory() as folder:
            case=self.repository_case();args=self.repository_bundle(Path(folder),case,bad[0])
            with self.assertRaises(ValueError):a.load_and_accept(**args)

if __name__=='__main__':unittest.main()
