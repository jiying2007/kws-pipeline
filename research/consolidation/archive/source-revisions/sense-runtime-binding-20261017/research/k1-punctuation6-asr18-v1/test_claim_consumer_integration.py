"""Full source-bound consumer paths using generated source and fake external APIs."""
import copy
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'k1-asr18-offline-posterior-v1'))
import fixture_cases as f
import campaign_contract as c
import evidence_transport as t
import evidence_publisher as p
import transfer_recovery as r
from test_asr18_transport import MockAPI, setup
from test_transfer_recovery import FakeAPI


def publication(case):
    v=case.expected['producer'];files=f.members(case.payload)
    execution=t.decode(files['source/'+f.CAMPAIGN+'execution-release.json'])
    owner=t.decode(files['admission.json'])['owner']['identity']
    env,event,_,_=setup()
    release={'schema':'k1-asr18-publisher-bound-release-v2','purpose':t.PURPOSE,
        'source_sha':v['identity']['executed_head_sha'],'source_tree':v['source_tree'],
        'source_closure_sha256':v['source_closure_sha256'],
        'prior_claim_history_sha256':execution['prior_public_claim_history_sha256'],
        'commit_timestamp':v['source_commit_timestamp'],'claim_admission_mode':'git-owner-v2',
        'execution_release_sha256':t.sha(files['source/'+f.CAMPAIGN+'execution-release.json']),
        'claim_admission_control_sha256':t.sha(files['source/'+f.CAMPAIGN+c.CLAIM_CONTROL_FILE]),
        **{key:owner[key] for key in ('source_branch','workflow','result_branch')}}
    env.update(GITHUB_SHA=release['source_sha'],GITHUB_WORKFLOW_SHA=release['source_sha'],
        GITHUB_REF='refs/heads/'+owner['source_branch'],GITHUB_WORKFLOW_REF=t.REPOSITORY+'/'+owner['workflow']+'@refs/heads/'+owner['source_branch'])
    event.update(after=release['source_sha'],ref=env['GITHUB_REF'])
    rows=[row for row in case.records['producer_source_tree']['tree'] if row['type']=='blob']
    class API(MockAPI):
        def __init__(self):
            super().__init__(env,event,release,rows)
            self.response_bodies={}
            self.run.update(c.bound_run_row(owner),conclusion=None,
                head_repository={'full_name':t.REPOSITORY,'private':False})
            later=copy.deepcopy(self.run);later.update(id=owner['run_id']+1,run_number=2,status='queued')
            self.history={'total_count':2,'workflow_runs':[copy.deepcopy(self.run),later]}
        def __call__(self,method,route,body=None,expected=200):
            result=super().__call__(method,route,body,expected)
            if route==self.paths['source']:result['committer']={'date':release['commit_timestamp']}
            if result is not None:self.response_bodies[route]=json.dumps(result,indent=2).encode()+b'\n'
            return result
    return env,event,release,API()


def recovered(case):
    a=f.recovery_authority(case);rec=case.expected['recovery']
    control=r.disabled_release();control.update({key:True for key in r.APPROVALS});control['authority']=a
    env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':t.REPOSITORY,'GITHUB_EVENT_NAME':'push',
        'GITHUB_REF':'refs/heads/'+r.SOURCE_BRANCH,'GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_ID':str(rec['identity']['run_id']),
        'GITHUB_SHA':rec['identity']['executed_head_sha'],'GITHUB_JOB':r.JOB_KEY,
        'GITHUB_WORKFLOW_REF':t.REPOSITORY+'/'+r.WORKFLOW+'@refs/heads/'+r.SOURCE_BRANCH,
        'GITHUB_WORKFLOW_SHA':rec['identity']['executed_head_sha'],'RUNNER_ENVIRONMENT':'github-hosted',
        'RUNNER_OS':'Linux','RUNNER_ARCH':'X64','ImageOS':'ubuntu24'}
    event={'repository':{'full_name':t.REPOSITORY,'private':False},'ref':env['GITHUB_REF'],
        'created':True,'deleted':False,'forced':False,'before':'0'*40,'after':env['GITHUB_SHA']}
    source={'source_sha':env['GITHUB_SHA'],'source_tree':rec['source_tree'],
        'source_closure_sha256':rec['source_closure_sha256'],'commit_timestamp':rec['source_commit_timestamp']}
    fix=types.SimpleNamespace(run=rec['identity'],source=source,authority=a,original=case.expected['producer']['identity'],
        recovery_rows=[x for x in case.records['recovery_source_tree']['tree'] if x['type']=='blob'],
        original_rows=[x for x in case.records['producer_source_tree']['tree'] if x['type']=='blob'],
        complete_rows=[x for x in case.records['result_tree']['tree'] if x['type']=='blob'],
        files={t.manifest_path():case.outer_raw,**dict(case.chunks)})
    class API(FakeAPI):
        def __init__(self):
            super().__init__(fix);self.response_bodies={}
            for prefix in ('producer','recovery'):
                for leaf in ('run','history','jobs'):
                    if prefix=='producer':self.rows[self.routes[prefix+'_'+leaf]]=copy.deepcopy(case.records[prefix+'_'+leaf])
                self.rows[self.routes[prefix+'_source']]=copy.deepcopy(case.records[prefix+'_source_commit'])
            owner=t.decode(f.members(case.payload)['admission.json'])['owner']['identity']
            self.rows[p.producer_history_route(owner)]=copy.deepcopy(case.records['producer_history'])
        def __call__(self,method,route,body=None,expected=200):
            result=super().__call__(method,route,body,expected)
            if result is not None:self.response_bodies[route]=json.dumps(result,indent=2).encode()+b'\n'
            return result
    return env,event,control,source,API()


def recovery_main_case():
    case=f.recovery_case();env,event,control,source,api=recovered(case)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'transfer-recovery-release.json').write_bytes(c.canonical(control))
        event_path=root/'event.json';event_path.write_bytes(c.canonical(event))
        env.update(GITHUB_EVENT_PATH=str(event_path),GITHUB_TOKEN='ghs_SYNTHETIC')
        real_recover=r.recover
        class SyntheticAPI:
            def __new__(cls,*args,**kwargs):return api
        def original_checkpoint(*args,**kwargs):return real_recover(*args,**kwargs,raw_factory=api.raw_factory)
        with mock.patch.object(r,'HERE',root),mock.patch.object(r,'bound_source',return_value=source), \
             mock.patch.object(r,'RecoveryAPI',new=SyntheticAPI),mock.patch.object(r,'recover',side_effect=original_checkpoint), \
             mock.patch.dict(os.environ,env,clear=True),mock.patch('http.client.HTTPSConnection',side_effect=AssertionError('no network')):
            result=r.main([])
        if len(api.calls)!=17 or sum(row[0]=='POST' for row in api.calls)!=1:raise AssertionError('fixed original recovery budget')
        return result



def materialized_scope_paths():
    """Fresh copied-code process, real validators, fake external/model observations."""
    import asr_posterior as a
    from test_offline_custody import write_bundle
    import test_offline_custody as custody
    scope=c.current_scope()
    if scope['campaign']==c.LEGACY_V1_SCOPE['campaign']:raise AssertionError('distinct compiled fixture required')
    normal=f.normal_case();recovery=f.recovery_case()
    env,event,release,api=publication(normal)
    published=p.publish(env,event,release,normal.payload,api,api.raw_factory)
    if published['producer_history']['total_count']!=2:raise AssertionError('full current publisher history')
    env,event,control,source,api=recovered(recovery)
    receipt=r.recover(env,event,control,source,api,api.raw_factory)
    if len(api.calls)!=17 or receipt['final_ref']!='refs/heads/'+scope['result_branch']:raise AssertionError('fixed recovery scope/budget')
    for case in (normal,recovery):
        result=a.accept(**case.arguments())
        if result['verified_complete_slots']!=18:raise AssertionError('native evidence required')
        with tempfile.TemporaryDirectory() as folder:
            run=custody.CustodyTests().run_cli(write_bundle(Path(folder),case))
            if run.returncode:raise AssertionError(run.stdout+run.stderr)
            value=t.decode(run.stdout)
            if value['verified_complete_slots']!=18:raise AssertionError('fresh CLI evidence required')
        wrong=copy.deepcopy(case)
        wrong.records['result_ref']['ref']='refs/heads/'+c.LEGACY_V1_SCOPE['result_branch']
        try:a.accept(**wrong.arguments())
        except ValueError:pass
        else:raise AssertionError('old r3 result ref cannot satisfy a new compiled campaign')
    print(json.dumps({'scope':scope,'normal_and_recovered':True,'fresh_posterior_cli':2,
        'publisher_requests':published['api_request_count'] if 'api_request_count' in published else 'bounded_by_original_client',
        'recovery_requests':len(api.calls),'old_r3_ref_rejected':True,'synthetic':True}))


def materialize_scope_copy(root):
    """Change compiled selectors in an isolated source copy, then remeasure bytes."""
    source=HERE.parents[1]
    shutil.copytree(source,root,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    campaign='k1-punctuation6-asr18-synthetic-next'
    workflow='.github/workflows/k1-asr18-synthetic-next.yml'
    package=root/'research/k1-punctuation6-asr18-v1'
    path=package/'campaign_contract.py';text=path.read_text()
    text=re.sub(r"(?m)^EXPERIMENT = .*?$",'EXPERIMENT = '+repr(campaign),text,count=1)
    text=re.sub(r"(?m)^CURRENT_WORKFLOW = .*?$",'CURRENT_WORKFLOW = '+repr(workflow),text,count=1)
    path.write_text(text)
    release=json.loads((package/'execution-release.json').read_bytes())
    release.update(experiment=campaign,source_ref='research/'+campaign,evidence_ref='evidence/'+campaign,workflow=workflow)
    (package/'execution-release.json').write_bytes(c.canonical(release))
    control=json.loads((package/'publication-release.json').read_bytes())
    control.update(source_branch='research/'+campaign,result_branch='evidence/'+campaign,workflow=workflow)
    (package/'publication-release.json').write_bytes(c.canonical(control))
    control=json.loads((package/'transfer-recovery-release.json').read_bytes())
    control['result_branch']='evidence/'+campaign
    (package/'transfer-recovery-release.json').write_bytes(c.canonical(control))
    raw=(package/'workflow.yml').read_bytes().replace(c.current_scope()['campaign'].encode(),campaign.encode())
    (package/'workflow.yml').write_bytes(raw);(root/workflow).write_bytes(raw)
    # Bootstrap in the copied source before importing the posterior's pinned validator.
    code="""import sys,json,hashlib,re
from pathlib import Path
root=Path(sys.argv[1]);package=root/'research/k1-punctuation6-asr18-v1'
sys.path.insert(0,str(package));import campaign_contract as c,prepare_freeze
(package/'execution-freeze.json').write_bytes(c.canonical(prepare_freeze.build(root)))
manifest=root/'research/k1-asr18-offline-posterior-v1/validator-dependencies.json'
value=json.loads(manifest.read_bytes())
for row in value['files']:
 raw=(root/row['path']).read_bytes();row.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\\0'+raw).hexdigest())
raw=c.canonical(value);manifest.write_bytes(raw)
path=manifest.with_name('asr_posterior.py');path.write_text(re.sub(r"MANIFEST_SHA = '[0-9a-f]{64}'","MANIFEST_SHA = '"+hashlib.sha256(raw).hexdigest()+"'",path.read_text()))
import test_claim_consumer_integration as test;test.materialized_scope_paths()
"""
    return code


class ClaimConsumerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal=f.normal_case();cls.recovery=f.recovery_case()
    def setUp(self):
        guard=mock.patch('http.client.HTTPSConnection',side_effect=AssertionError('no external network'))
        guard.start();self.addCleanup(guard.stop)
    def test_control_only_future_scope_with_current_prior_rejects_before_api(self):
        from claim_consumer_fixtures import bound_public_fixture
        scope=c.current_scope();campaign='k1-control-only-synthetic-next'
        future={**scope,'campaign':campaign,'source_branch':'research/'+campaign,
            'result_branch':'evidence/'+campaign,'claim_ref':'refs/heads/claims/'+campaign,
            'directory':'research/k1-punctuation6-asr18-v1/evidence/'+campaign,
            'workflow':'.github/workflows/k1-control-only-synthetic-next.yml'}
        fixture=bound_public_fixture(scope=future);owner=fixture['owner_fixture'];files=fixture['files']
        self.assertEqual(owner['release']['prior_public_claim_history_sha256'],
            c.digest(c.canonical(c.bound_prior_history_identity(owner['control']))))
        c.validate_bound_admission(owner['admission'],owner['identity'])
        with self.assertRaisesRegex(ValueError,'sole compiled current scope'):
            t.bind_archive_admission(files,fixture['identity'],owner['identity']['freeze_sha256'],456,fixture['source_tree'])
        payload,_,_=t._zip_files(files,fixture['identity'])
        with self.assertRaisesRegex(ValueError,'sole compiled current scope'):
            t.validate_archive(payload,fixture['identity'])
        env,event,release,api=publication(self.normal)
        release.update({key:future[key] for key in ('source_branch','result_branch','workflow')})
        with self.assertRaisesRegex(ValueError,'compiled source scope'):
            p.publish(env,event,release,payload,api,api.raw_factory)
        self.assertEqual(api.calls,[])
        proof={'identity':owner['identity'],'admission':owner['admission']}
        probe=mock.Mock(side_effect=AssertionError('API before scope gate'))
        with self.assertRaisesRegex(ValueError,'sole compiled current scope'):
            r._verify_run(probe,{},'producer',fixture['identity'],future['source_branch'],future['workflow'],'asr18',
                self.recovery.expected['producer']['terminal'],source_claim=proof)
        probe.assert_not_called()
        import asr_posterior as a
        expected=copy.deepcopy(self.normal.expected)
        expected['producer'].update(identity=fixture['identity'],source_tree=fixture['source_tree'],
            source_closure_sha256=owner['identity']['freeze_sha256'],source_commit_timestamp=fixture['source_commit_timestamp'])
        expected['archive'].update(bytes=len(payload),sha256=t.sha(payload))
        records=copy.deepcopy(self.normal.records)
        records.update(producer_source_commit={'sha':fixture['identity']['executed_head_sha'],
            'tree':{'sha':fixture['source_tree']},'committer':{'date':fixture['source_commit_timestamp']}},
            producer_source_tree=f.source_tree_response(fixture['source_rows']))
        with self.assertRaisesRegex(ValueError,'sole compiled current scope'):
            a.source_bound_producer(expected,records,payload)
        expected['claim_owner'].update(ref=future['claim_ref'],**{key:future[key]
            for key in ('source_branch','result_branch','workflow')})
        for branch in (future['result_branch'],scope['result_branch']):
            records['result_ref']['ref']='refs/heads/'+branch
            with self.subTest(result_branch=branch),self.assertRaisesRegex(ValueError,'compiled source scope'):
                a.accept(expected,records,self.normal.outer_raw,self.normal.chunks,payload)
        original_read=c.read_bytes
        def source_read(path,**kwargs):
            name='source/'+str(Path(path).relative_to(p.ROOT)) if Path(path).is_relative_to(p.ROOT) else None
            return files[name] if name in files else original_read(path,**kwargs)
        with mock.patch.object(c,'read_bytes',side_effect=source_read),mock.patch.object(p,'_source_git',side_effect=AssertionError('Git before scope gate')) as git:
            with self.assertRaisesRegex(ValueError,'sole compiled current scope'):p.bound_release({}, {})
            git.assert_not_called()

    def test_new_compiled_campaign_full_normal_recovery_and_fresh_cli(self):
        with tempfile.TemporaryDirectory(prefix='k1-compiled-scope-') as folder:
            root=Path(folder)/'source';code=materialize_scope_copy(root)
            options=['-I','-S','-B']+(['-'+'O'*sys.flags.optimize] if sys.flags.optimize else [])
            run=subprocess.run([sys.executable,*options,'-c',code,str(root)],capture_output=True,timeout=120,
                env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'})
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            value=t.decode(run.stdout)
            self.assertNotEqual(value['scope']['campaign'],c.current_scope()['campaign'])
            self.assertTrue(value['old_r3_ref_rejected'])
            self.assertEqual(value['fresh_posterior_cli'],2)

    def test_whole_archive_publisher_preserves_history2_and_request_cap(self):
        env,event,release,api=publication(self.normal)
        result=p.publish(env,event,release,self.normal.payload,api,api.raw_factory)
        self.assertEqual(result['producer_history']['total_count'],2)
        self.assertEqual(result['numeric_job_id'],456)
        self.assertLessEqual(len(api.calls),67)
        owner=self.normal.expected['claim_owner']
        self.assertFalse(any('/git/ref/'+owner['ref'][5:] in row[1] or '/git/commits/'+owner['commit'] in row[1] for row in api.calls))
    def test_real_publisher_main_binds_archive_before_run_history(self):
        env,event,release,api=publication(self.normal)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);event_path=root/'event.json';event_path.write_bytes(c.canonical(event))
            files=f.members(self.normal.payload)
            (root/'publication-release.json').write_bytes(files['source/'+f.CAMPAIGN+'publication-release.json'])
            env.update(RUNNER_TEMP=folder,GITHUB_EVENT_PATH=str(event_path),GITHUB_TOKEN='ghs_SYNTHETIC')
            t.write_package(root,self.normal.payload,self.normal.expected['producer']['identity'])
            # Source Git/API are explicit synthetic boundaries. The entrypoint,
            # gate, package read, source binding, publication and receipts run.
            with mock.patch.object(p,'HERE',root),mock.patch.object(p,'bound_release',return_value=release), \
                 mock.patch.object(p,'GitHubAPI',return_value=api),mock.patch.object(p,'RawReader',new=api.raw_factory), \
                 mock.patch.dict(os.environ,env,clear=True),mock.patch('builtins.print'):
                answer=p.main([])
            self.assertEqual(answer['producer_history']['total_count'],2)
            self.assertTrue((root/('k1-asr18-publication-'+str(self.normal.expected['producer']['identity']['run_id']))).is_dir())

    def test_whole_archive_recovery_reads_proof_before_producer_and_only_final_post(self):
        env,event,control,source,api=recovered(self.recovery)
        result=r.recover(env,event,control,source,api,api.raw_factory)
        self.assertEqual(len(api.calls),17)
        self.assertEqual(result['producer_history']['total_count'],2)
        self.assertEqual(result['original_terminal']['publisher_step_conclusion'],'failure')
        self.assertFalse(result['original_publisher_reclassified'])
        self.assertEqual(sum(row[0]=='POST' for row in api.calls),1)
        calls=[row[1] for row in api.calls]
        self.assertLess(calls.index(api.routes['checkpoint_tree']),calls.index(api.routes['producer_run']))
    def test_fresh_isolated_recovery_main_runs_original_checkpoint_path(self):
        code='import sys;sys.path.insert(0,'+repr(str(HERE))+');import test_claim_consumer_integration as test;test.recovery_main_case()'
        options=['-I','-S','-B']+(['-'+'O'*sys.flags.optimize] if sys.flags.optimize else [])
        run=subprocess.run([sys.executable,*options,'-c',code],capture_output=True,timeout=90,
            env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        result=t.decode(run.stdout)
        self.assertEqual(result['producer_history']['total_count'],2)
        self.assertEqual(result['original_terminal']['publisher_step_conclusion'],'failure')
        self.assertFalse(result['original_publisher_reclassified'])

    def test_recovery_own_history2_rejects_before_checkpoint_read_and_write(self):
        env,event,control,source,api=recovered(self.recovery)
        row=api.rows[api.routes['recovery_history']];row['total_count']=2;row['workflow_runs']*=2
        with self.assertRaises(ValueError):r.recover(env,event,control,source,api,api.raw_factory)
        self.assertFalse(api.raw_calls);self.assertFalse(any(row[0]=='POST' for row in api.calls))
    def test_legacy_predicate_keeps_exact_history1(self):
        env,event,release,rows=setup()
        api=MockAPI(env,event,release,rows)
        self.assertEqual(p.verify_run(api,p.context(env,event,release),release),456)
        api=MockAPI(env,event,release,rows)
        api.history={'total_count':2,'workflow_runs':[copy.deepcopy(api.run),copy.deepcopy(api.run)]}
        with self.assertRaises(ValueError):p.verify_run(api,p.context(env,event,release),release)

    def test_unclaimed_input_failure_still_publishes_only_admission_observation(self):
        case=f.normal_case('all_notrun')
        self.assertNotIn('authority/source-claim.json',f.members(case.payload))
        env,event,release,api=publication(case)
        answer=p.publish(env,event,release,case.payload,api,api.raw_factory)
        self.assertEqual(answer['producer_history']['total_count'],2)
        self.assertTrue(answer['semantic_inference_success_not_established_by_publisher'])

    def test_v2_owner_job_history_and_raw_body_drift_fail_before_write(self):
        for mode in ('job','ordinal','incomplete','raw_hash'):
            env,event,release,api=publication(self.normal)
            if mode=='job':api.job['id']+=1
            elif mode=='ordinal':api.run['run_number']=2
            elif mode=='incomplete':api.history['total_count']=3
            else:
                def observation(*args,**kwargs):raise ValueError('unavailable actual raw response')
            context=mock.patch.object(p,'history_observation',side_effect=observation) if mode=='raw_hash' else mock.patch.object(p,'MAX_API_REQUESTS',67)
            with self.subTest(mode=mode),context,self.assertRaises(ValueError):p.publish(env,event,release,self.normal.payload,api,api.raw_factory)
            self.assertFalse(any(row[0]=='POST' for row in api.calls))
    def test_actual_recovery_client_rebinds_only_source_verified_history_ticket(self):
        env,event,control,source,fake=recovered(self.recovery)
        api=r.RecoveryAPI('ghs_SYNTHETIC',fake.fixture.run,source,control)
        old=api.producer_history_route;before=set(api.allowed)
        with self.assertRaises(ValueError):api.bind_producer_history(b'bad',fake.fixture.original_rows)
        self.assertEqual(api.allowed,before)
        route=api.bind_producer_history(self.recovery.payload,fake.fixture.original_rows)
        self.assertEqual(api.allowed,before-{('GET',old)}|{('GET',route)})
        self.assertEqual(api.count,0)
        with self.assertRaises(ValueError):api.bind_producer_history(self.recovery.payload,fake.fixture.original_rows)
    def test_explicit_closed_cli_rejects_before_client_or_token(self):
        from test_asr18_entrypoints import disabled_source
        with disabled_source() as source:
            for module,client in ((p,'GitHubAPI'),(r,'RecoveryAPI')):
                with self.subTest(module=module.__name__),mock.patch.object(module,'HERE',source), \
                     mock.patch.object(module,client,side_effect=AssertionError('client forbidden')):
                    with self.assertRaises((p.PublicationDisabled,r.RecoveryDisabled)):module.main([])
    def test_v2_publication_control_cannot_select_legacy_four_step_workflow(self):
        raw=c.read_bytes(HERE/'workflow.yml')
        start=raw.index(b'      - name: Reserve fixed campaign owner without models\n')
        end=raw.index(b'      - name: Admit complete new input and run fixed ASR18 once\n')
        legacy=(raw[:start]+raw[end:]).replace(b'timeout-minutes: 4\n',b'timeout-minutes: 5\n',1)
        legacy=legacy.replace(b" && steps.campaign_claim.outcome == 'success' && steps.campaign_claim.outputs.ready == 'true'",b'')
        control=p.disabled_publication_release()
        with self.assertRaises(ValueError):p.validate_transfer_workflow(legacy,control)
        old={key:value for key,value in control.items() if key!='claim_admission_mode'}
        old['schema']='k1-asr18-publication-control-v1'
        old.update({key:c.LEGACY_V1_SCOPE[key] for key in ('source_branch','result_branch','workflow')})
        legacy=legacy.replace(c.current_scope()['campaign'].encode(),c.LEGACY_V1_SCOPE['campaign'].encode())
        p.validate_transfer_workflow(legacy,old)

    def test_exact_new_workflow_and_token_isolation(self):
        raw=c.read_bytes(HERE/'workflow.yml');p.validate_transfer_workflow(raw)
        for old,new in ((b'github.run_number == 1',b'github.run_number == 2'),
            (b'steps.campaign_claim.outputs.ready',b'steps.host.outputs.ready'),
            (b'timeout-minutes: 1\n',b'timeout-minutes: 2\n')):
            with self.subTest(old=old),self.assertRaises(ValueError):p.validate_transfer_workflow(raw.replace(old,new,1))

if __name__=='__main__':unittest.main()
