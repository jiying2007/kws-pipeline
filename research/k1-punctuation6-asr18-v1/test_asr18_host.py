"""Synthetic metadata, fake API/Git and mocked runtime only. Zero model/Docker calls."""
import ast
from contextlib import ExitStack
from functools import wraps
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import campaign_contract as c
import hosted_run as h
import runtime_session as r
import asr18_metadata as m
from test_asr18_contract import invented_projection
from test_asr18_binding import setup_binding as unbound_setup_binding
from input_binding import verify_screening_receipt
from test_control_fixture import DISABLED_APPROVALS, synthetic_enabled_source, synthetic_prerequisite


def synthetic_enabled_host_test(function):
    """Run real gates against a separately re-frozen test-only enabled copy."""
    @wraps(function)
    def wrapped(self):
        with synthetic_enabled_source(HERE) as source, \
             patch.object(h,'ROOT',source.parents[1]), patch.object(h,'HERE',source), \
             patch.dict(globals(),{'HERE':source}):
            return function(self)
    return wrapped


def release_fixture():
    release=c.disabled_release()
    for key in c.RELEASE_APPROVALS: release[key]=True
    for key in c.RELEASE_AUTHORITIES: release[key]='a'*64
    release['prior_public_claim_history_sha256']=c.digest(c.canonical(h.prior_history_identity()))
    return release


def context():
    env={'GITHUB_REPOSITORY':h.REPOSITORY,'GITHUB_REF':'refs/heads/'+h.SOURCE_REF,
        'GITHUB_EVENT_NAME':'push','GITHUB_RUN_ATTEMPT':'1','GITHUB_JOB':'asr18',
        'RUNNER_ENVIRONMENT':'github-hosted','RUNNER_OS':'Linux','RUNNER_ARCH':'X64','ImageOS':'ubuntu24',
        'GITHUB_SHA':'c'*40,'GITHUB_RUN_ID':'123','GITHUB_WORKFLOW_REF':h.REPOSITORY+'/'+h.WORKFLOW+'@refs/heads/'+h.SOURCE_REF,
        'GITHUB_WORKFLOW_SHA':'c'*40}
    event={'created':True,'deleted':False,'forced':False,'before':'0'*40,'after':'c'*40,
        'ref':'refs/heads/'+h.SOURCE_REF,'repository':{'full_name':h.REPOSITORY,'private':False}}
    return env,event,h.event_identity(env,event)


class FakeAPI:
    def __init__(self):
        self.calls=[]; self.response_body_bytes=0; _,_,run=context(); base='/repos/'+h.REPOSITORY
        current={'id':123,'run_attempt':1,'status':'in_progress','conclusion':None,'event':'push','head_branch':h.SOURCE_REF,'head_sha':'c'*40,
            'path':h.WORKFLOW,'repository':{'full_name':h.REPOSITORY,'private':False},
            'head_repository':{'full_name':h.REPOSITORY},'workflow_id':9}
        self.rows={base+'/actions/runs/123':current,base+'/actions/workflows/9':{'id':9,'path':h.WORKFLOW},
            base+'/actions/workflows/9/runs?branch=research%2Fk1-punctuation6-asr18-r2&per_page=100&page=1':{'total_count':1,'workflow_runs':[current]},
            base+'/actions/runs/123/attempts/1/jobs?per_page=100&page=1':{'total_count':1,'jobs':[{'id':456,'name':'asr18','run_id':123,'head_sha':'c'*40,'status':'in_progress','conclusion':None,
                'labels':['ubuntu-24.04'],'runner_name':'GitHub Actions 1','runner_group_name':'GitHub Actions'}]},
            base+'/git/ref/heads/'+h.SOURCE_REF:{'ref':'refs/heads/'+h.SOURCE_REF,'object':{'type':'commit','sha':'c'*40}},
            base+'/git/ref/heads/'+h.EVIDENCE_REF:None}
    def get(self,path,*,absent=False):
        self.calls.append(path)
        value=copy.deepcopy(self.rows[path])
        if absent: c.require(value is None,'not absent')
        self.response_body_bytes+=len(c.canonical(value))
        return value
    def accounting(self):
        return {'gets':len(self.calls),'response_body_bytes':self.response_body_bytes,'response_body_cap_bytes':c.MIB,
            'scope':'public_api_metadata_separate_from_model_audio_bodies','hard_wire_byte_cap':False}



def setup_binding():
    raw,kwargs=unbound_setup_binding()
    policy_sha=c.identity(HERE/'asr18_metadata.py')['sha256']
    value=json.loads(raw);value['screening_policy_sha256']=policy_sha
    raw=c.canonical(value)
    return raw,{**kwargs,'expected_policy_sha256':policy_sha,'expected_receipt_sha256':c.digest(raw)}


def public_fixture():
    _,_,run=context(); raw,kwargs=setup_binding(); verified=verify_screening_receipt(raw,**kwargs)
    blind,binding=verified['blind'],verified['worker_input_binding']
    inp={'schema':'k1-asr18-input-freeze-v1','execution_enabled':False,
        'actual_new_cells':verified['projection']['cells'],'actual_blind_job':blind['job'],
        'actual_opaque_order':blind['execution_plan'],'actual_screening_receipt':json.loads(raw),'actual_worker_binding':binding}
    release=release_fixture(); release.update(actual_input_freeze_sha256=c.digest(c.canonical(inp)),
        actual_tts_authority_sha256=binding['tts_authority_sha256'],actual_screening_receipt_sha256=binding['screening_receipt_sha256'],
        actual_worker_binding_sha256=c.digest(c.canonical(binding)),actual_opaque_order_sha256=c.digest(c.canonical(blind['execution_plan'])))
    import prepare_freeze
    import prepare_prerequisite
    names=set(prepare_prerequisite.SCIENTIFIC_MEMBERS)|prepare_prerequisite.EXCLUDED
    source={name:c.read_bytes(HERE.parents[1]/name) for name in names}
    preliminary=prepare_freeze.build(HERE.parents[1])
    source[h.NEW+'prepare-prerequisite.json']=synthetic_prerequisite(release,root=HERE.parents[1],freeze=preliminary)
    source[h.NEW+'execution-release.json']=c.canonical(release)
    freeze={'schema':'k1-asr18-source-freeze-v1',**{role:{name:{'bytes':len(body),'sha256':c.digest(body)} for name,body in source.items()}
        for role in ('host_code','setup_code','inference_code')}}
    freeze_raw=c.canonical(freeze); host=h.host_receipt(run,c.digest(freeze_raw),'d'*40,binding['screening_receipt_sha256'])
    host.update(status='stopped_no_retry',stage='package',cleanup_verified=True)
    terminal={'schema':'k1-asr18-terminal-outcomes-v1','identity':run,
        'projection_sha256':c.digest(c.canonical(verified['projection'])),'slots':c.initial_ledger(verified['projection']),'accounting_valid':True}
    values={'host-receipt.json':host,'run-preflight.json':h.api_preflight(FakeAPI(),run,release),
        'input-freeze.json':inp,'public-screening-receipt.json':json.loads(raw),'new-cell-projection.json':verified['projection'],
        'authority/execution-freeze.json':freeze,'terminal-outcomes.json':terminal}
    files={name:c.canonical(value) for name,value in values.items()}
    files.update({'source/'+name:body for name,body in source.items()});files['source/'+h.NEW+'execution-freeze.json']=freeze_raw
    return files,run


class HostTests(unittest.TestCase):
    def test_checked_in_controls_disabled_and_prior_unknown_not_reclaimed(self):
        for filename,keys in DISABLED_APPROVALS.items():
            value=c.read_json(HERE/filename)
            self.assertTrue(all(value[key] is False for key in keys))
        release=c.read_json(HERE/'execution-release.json');history=h.prior_history_identity()
        self.assertIs(release['historical_screening_reviewed'],True)
        with self.assertRaisesRegex(ValueError,'disabled'):h.release_first()
        self.assertEqual(release['experiment'],'k1-punctuation6-asr18-r2')
        self.assertEqual(release['source_ref'],'research/k1-punctuation6-asr18-r2')
        self.assertEqual(history['prior_runs'],[])
        self.assertEqual(history['prior_runs_scope'],'only_new_source_branch_and_fixed_workflow')
        prior=history['sealed_prior_campaigns'][0]
        self.assertEqual(prior['run_id'],38207026177);self.assertIsNone(prior['actual_consumed_attempts'])
        self.assertEqual(prior['reusable_slots'],0)
        self.assertEqual(release['prior_public_claim_history_sha256'],c.digest(c.canonical(history)))
        self.assertEqual(len(history['sealed_prior_campaigns']),2)
        prior_r1=history['sealed_prior_campaigns'][1]
        self.assertEqual(prior_r1['run_id'],38219171568);self.assertIsNone(prior_r1['actual_consumed_attempts'])
        self.assertEqual(prior_r1['reusable_slots'],0)
        import copy
        wrong=copy.deepcopy(history);wrong['sealed_prior_campaigns'][0]['actual_consumed_attempts']=0
        bad=dict(release,prior_public_claim_history_sha256=c.digest(c.canonical(wrong)))
        with self.assertRaisesRegex(ValueError,'prior public claim history'):
            h.api_preflight(FakeAPI(),context()[2],bad)
        env,event,_=context()
        for change in ({'GITHUB_RUN_ID':'38207026177'}, {'GITHUB_RUN_ID':'38219171568'}, {'GITHUB_REF':'refs/heads/research/k1-punctuation6-asr18-v1'},
                       {'GITHUB_RUN_ATTEMPT':'2'}):
            with self.subTest(change=change),self.assertRaises(ValueError):h.event_identity({**env,**change},event)

    def test_explicit_disabled_copy_precedes_every_external_or_mutating_operation(self):
        # Exercise the real disabled reader without changing production controls.
        before=c.read_bytes(HERE/'execution-release.json')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)
            (source/'execution-release.json').write_bytes(c.canonical(c.disabled_release()))
            with patch.object(h,'HERE',source), patch.object(h,'PublicAPI',side_effect=AssertionError('network')) as api, \
                 patch.object(h,'git_read',side_effect=AssertionError('git')) as git, \
                 patch.object(h,'execute',side_effect=AssertionError('runtime')) as execute, \
                 patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'disabled'):h.main([])
                api.assert_not_called();git.assert_not_called();execute.assert_not_called()
        self.assertEqual(c.read_bytes(HERE/'execution-release.json'),before)

    @synthetic_enabled_host_test
    def test_synthetic_enabled_candidate_release_and_authorities_remain_bound(self):
        # Real production gate, only temporary approval flags enabled.
        release=h.release_first()
        self.assertTrue(all(release[key] is True for key in c.RELEASE_APPROVALS))
        for key in c.RELEASE_AUTHORITIES:c.sha(release[key])
        source=c.read_json(HERE/'execution-freeze.json')
        inputs,authority=h.input_authorities(release,source)
        self.assertEqual(c.digest(c.canonical(inputs)),release['actual_input_freeze_sha256'])
        self.assertEqual(c.digest(c.canonical(authority)),release['actual_tts_authority_sha256'])
        self.assertEqual(len(inputs['actual_new_cells']),6)
        self.assertGreater(len(inputs['actual_blind_job']['clips']),0)

    @synthetic_enabled_host_test
    def test_synthetic_enabled_r2_controls_pass_real_git_api_and_input_gates(self):
        # Keep the temporary enabled controls and re-frozen public source bytes.
        # Only the host event and six GitHub responses are invented. A clean
        # temporary local commit supplies the actual source HEAD/tree/blob;
        # no source, release, API or input admission gate is replaced.
        production=c.read_bytes(HERE/'execution-release.json')
        source_raw=c.read_bytes(HERE/'execution-freeze.json')
        source=c.decode_json(source_raw)
        names=set().union(*(set(source[role]) for role in ('host_code','setup_code','inference_code')))
        names.update((h.NEW+'execution-freeze.json',h.WORKFLOW))
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp)/'checkout';root.mkdir()
            for name in sorted(names):
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(c.read_bytes(h.ROOT/name))
            # No ambient credentials, user config, hooks, remotes or signing.
            env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent',
                'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null',
                'GIT_AUTHOR_DATE':'2000-01-01T00:00:00+0000',
                'GIT_COMMITTER_DATE':'2000-01-01T00:00:00+0000'}
            def git(*args):
                return subprocess.run(['git','-c','core.hooksPath=/dev/null',
                    '-c','commit.gpgsign=false','-c','user.name=Offline Fixture',
                    '-c','user.email=offline-fixture@example.invalid',*args],
                    cwd=root,env=env,check=True,capture_output=True,timeout=30).stdout
            git('init','--quiet','--initial-branch='+h.SOURCE_REF)
            git('add','--all');git('commit','--quiet','-m','Offline exact candidate source fixture')
            self.assertEqual(git('remote'),b'')
            stack.enter_context(patch.object(h,'ROOT',root))
            stack.enter_context(patch.object(h,'HERE',root/h.NEW))
            traps=[]
            for module,name in ((h,'PublicAPI'),(h,'verify_download'),(h,'source_claim'),
                (h,'write_once'),(h,'execute'),(c,'durable_mkdir'),
                (r,'setup_runtimes'),(r,'supervise')):
                traps.append(stack.enter_context(patch.object(module,name,
                    side_effect=AssertionError('forbidden downstream operation: '+name))))
            head=h.git_read('rev-parse','HEAD').decode().strip()
            host,event,_=context()
            host.update(GITHUB_SHA=head,GITHUB_WORKFLOW_SHA=head)
            event['after']=head
            release=h.release_first()
            self.assertEqual(c.canonical(release),production)
            self.assertTrue(all(release[key] is True for key in c.RELEASE_APPROVALS))
            self.assertEqual(release['prior_public_claim_history_sha256'],
                '5658a010184522f001652089177f8ced1a8ffec919916379ccdb43f7bf766bcd')
            run=h.event_identity(host,event)
            api=FakeAPI();base='/repos/'+h.REPOSITORY
            api.rows[base+'/actions/runs/123']['head_sha']=head
            next(v for k,v in api.rows.items() if '/jobs?' in k)['jobs'][0]['head_sha']=head
            api.rows[base+'/git/ref/heads/'+h.SOURCE_REF]['object']['sha']=head
            preflight=h.api_preflight(api,run,release)
            frozen,freeze_sha,tree=h.source_authority(run,release)
            inputs,authority=h.input_authorities(release,frozen)
            self.assertEqual(frozen,source)
            self.assertEqual(freeze_sha,c.digest(source_raw))
            self.assertEqual(tree,h.git_read('rev-parse',head+'^{tree}').decode().strip())
            self.assertEqual(preflight['identity'],run)
            self.assertEqual(preflight['numeric_job_id'],456)
            self.assertEqual(preflight['prior_public_claim_history_sha256'],
                c.digest(c.canonical(h.prior_history_identity())))
            self.assertEqual(len(api.calls),6)
            self.assertTrue(preflight['evidence_ref_absent'])
            self.assertEqual(c.digest(c.canonical(inputs)),release['actual_input_freeze_sha256'])
            self.assertEqual(c.digest(c.canonical(authority)),release['actual_tts_authority_sha256'])
            self.assertEqual(len(inputs['actual_new_cells']),6)
            self.assertGreater(len(inputs['actual_blind_job']['clips']),0)
            self.assertEqual(h.git_read('status','--porcelain=v1','--untracked-files=all'),b'')
            for trap in traps:trap.assert_not_called()
        self.assertEqual(c.read_bytes(HERE/'execution-release.json'),production)
        self.assertEqual(c.read_bytes(HERE/'execution-freeze.json'),source_raw)

    @synthetic_enabled_host_test
    def test_synthetic_enabled_r2_release_rejects_noncanonical_lf_history_sha(self):
        # The public observer's bytes retain their trailing LF. Its file digest
        # must never stand in for the canonical JSON identity accepted by the
        # production API gate. Reconstruct only this public fixed identity;
        # do not rewrite the observer or substitute a synthetic released value.
        production=c.read_bytes(HERE/'execution-release.json')
        release=h.release_first()
        canonical=c.canonical(h.prior_history_identity())
        raw_observer=canonical+b'\n'
        self.assertEqual(c.decode_json(raw_observer,require_canonical=False),h.prior_history_identity())
        self.assertEqual(c.digest(canonical),'5658a010184522f001652089177f8ced1a8ffec919916379ccdb43f7bf766bcd')
        self.assertEqual(c.digest(raw_observer),'018f0dc3050f266e1410b844419bb0632d848562d8abdc71bac726b07608b2c7')
        self.assertEqual(release['prior_public_claim_history_sha256'],c.digest(canonical))
        wrong=copy.deepcopy(release)
        wrong['prior_public_claim_history_sha256']=c.digest(raw_observer)
        _,_,run=context();api=FakeAPI()
        with self.assertRaisesRegex(ValueError,'independently released prior public claim history'):
            h.api_preflight(api,run,wrong)
        self.assertEqual(len(api.calls),3)
        self.assertFalse(any('/jobs?' in path or '/git/ref/' in path for path in api.calls))
        self.assertEqual(c.read_bytes(HERE/'execution-release.json'),production)

    @synthetic_enabled_host_test
    def test_synthetic_enabled_real_entry_rejects_invalid_host_or_event_before_side_effects(self):
        # release_first/read_json/event_identity/main are all the real functions.
        # Traps observe only downstream external or mutating operations; no
        # gate is mocked into returning success or into inventing a rejection.
        production=c.read_bytes(HERE/'execution-release.json')
        self.assertTrue(h.release_first()['enabled'])
        cases=(('host',{'RUNNER_ENVIRONMENT':'self-hosted'},{}),
            ('repository',{'GITHUB_REPOSITORY':'unapproved/repository'},{}),
            ('event',{}, {'created':False}))
        for label,env_delta,event_delta in cases:
            with self.subTest(case=label),tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
                env,event,_=context();env.update(env_delta);event.update(event_delta)
                path=Path(temp)/'event.json';path.write_bytes(c.canonical(event))
                env['GITHUB_EVENT_PATH']=str(path)
                traps=[]
                for module,name in ((h,'PublicAPI'),(h,'git_read'),(h,'verify_download'),
                    (h,'source_claim'),(h,'write_once'),(c,'durable_mkdir'),
                    (r,'setup_runtimes'),(r,'supervise')):
                    traps.append(stack.enter_context(patch.object(module,name,
                        side_effect=AssertionError('forbidden downstream operation: '+name))))
                stack.enter_context(patch.dict(os.environ,env,clear=True))
                with self.assertRaisesRegex(ValueError,'fixed public standard|first public source creation'):
                    h.main([])
                for trap in traps:trap.assert_not_called()
        self.assertEqual(c.read_bytes(HERE/'execution-release.json'),production)

    def test_fixed_public_standard_host_first_created_identity(self):
        env,event,run=context(); self.assertEqual(run['run_id'],123)
        for key,val in (('RUNNER_ENVIRONMENT','self-hosted'),('ImageOS','ubuntu22'),('GITHUB_RUN_ATTEMPT','2'),
            ('GITHUB_JOB','other'),('GITHUB_REPOSITORY','another/repo'),('GITHUB_WORKFLOW_SHA','e'*40)):
            with self.subTest(key=key),self.assertRaises(ValueError): h.event_identity({**env,key:val},event)
        for key,val in (('created',False),('deleted',True),('forced',True),('before','a'*40),('after','e'*40)):
            with self.subTest(key=key),self.assertRaises(ValueError): h.event_identity(env,{**event,key:val})

    def test_api_actual_run_complete_unique_history_and_result_404(self):
        _,_,run=context(); api=FakeAPI(); pre=h.api_preflight(api,run,release_fixture())
        self.assertEqual(pre['numeric_job_id'],456); self.assertEqual(len(api.calls),6)
        self.assertTrue(pre['evidence_ref_absent'])

    def test_history_replay_incomplete_page_and_changed_job_block(self):
        _,_,run=context()
        for kind in ('replay','truncated','wrong_job','wrong_run','result_exists'):
            api=FakeAPI(); history=next(v for k,v in api.rows.items() if '/runs?branch=' in k)
            if kind=='replay': history.update(total_count=2,workflow_runs=history['workflow_runs']*2)
            if kind=='truncated': history['total_count']=101
            if kind=='wrong_job': next(v for k,v in api.rows.items() if '/jobs?' in k)['jobs'][0]['labels']=['self-hosted']
            if kind=='wrong_run': api.rows['/repos/'+h.REPOSITORY+'/actions/runs/123']['head_sha']='d'*40
            if kind=='result_exists': api.rows['/repos/'+h.REPOSITORY+'/git/ref/heads/'+h.EVIDENCE_REF]={}
            with self.subTest(kind=kind),self.assertRaises(ValueError): h.api_preflight(api,run,release_fixture())

    def test_nonunique_count_stops_without_another_page(self):
        class Pages:
            calls=0
            def get(self,path):
                self.calls+=1;return {'total_count':101,'items':list(range(100))}
        pages=Pages()
        with self.assertRaises(ValueError):h._unique_page(pages,'/fixed','items')
        self.assertEqual(pages.calls,1)

    def test_public_api_fixed_six_gets_and_actual_metadata_body_accounting(self):
        class Response:
            headers={}
            def __init__(self,url,oversize=False):self.url=url;self.oversize=oversize
            def read(self,n):return b'x'*n if self.oversize else b'{}'
            def geturl(self):return self.url
            def getcode(self):return 200
            def close(self):pass
        class Opener:
            calls=0;oversize=False
            def open(self,request,timeout):self.calls+=1;return Response(request.full_url,self.oversize)
        opener=Opener()
        with patch.object(h.urllib.request,'build_opener',return_value=opener):api=h.PublicAPI()
        for index in range(6):self.assertEqual(api.get('/repos/'+h.REPOSITORY+'/fixture-'+str(index)),{})
        self.assertEqual(api.accounting()['response_body_bytes'],12)
        self.assertEqual(api.accounting()['gets'],6)
        with self.assertRaises(ValueError):api.get('/repos/'+h.REPOSITORY+'/fixture-7')
        self.assertEqual(opener.calls,6)
        opener=Opener();opener.oversize=True
        with patch.object(h.urllib.request,'build_opener',return_value=opener):api=h.PublicAPI()
        with self.assertRaises(ValueError):api.get('/repos/'+h.REPOSITORY+'/fixture-too-large')
        self.assertEqual(api.accounting()['response_body_bytes'],512*1024+1)
        self.assertEqual(api.accounting()['gets'],1)

    def test_prior_history_is_independently_pinned(self):
        _,_,run=context(); release=release_fixture();release['prior_public_claim_history_sha256']='e'*64
        with self.assertRaises(ValueError): h.api_preflight(FakeAPI(),run,release)

    def test_source_expected_hash_comes_from_git_blob_not_working_file(self):
        _,_,run=context(); release=release_fixture()
        with patch.object(h,'git_read',side_effect=[b'c'*40+b'\n',b'',b'd'*40+b'\n',c.canonical({'git':'independent'})]), \
             patch.object(h,'read_bytes',return_value=c.canonical({'working':'different'})):
            with self.assertRaisesRegex(ValueError,'independent immutable blob'): h.source_authority(run,release)

    def test_dirty_or_different_git_head_blocks_source_authority(self):
        _,_,run=context()
        for values in ([b'e'*40+b'\n'],[b'c'*40+b'\n',b'?? unknown\n']):
            with patch.object(h,'git_read',side_effect=values),self.assertRaisesRegex(ValueError,'clean actual'): h.source_authority(run,release_fixture())

    def test_exact_once_claim_is_worker_compatible(self):
        _,_,run=context(); release=release_fixture(); pre=h.api_preflight(FakeAPI(),run,release)
        claim=h.source_claim(run,release,pre,'f'*64); raw=c.canonical(claim)
        self.assertEqual(c.validate_source_claim(raw,expected_sha256=c.digest(raw),release=release,source_freeze_sha256='f'*64,
            run_id='123',executed_head_sha='c'*40),claim)

    def test_immutable_input_authority_needed_before_reader(self):
        with patch('raw_data_reader.RawDataReader',side_effect=AssertionError('reader must not be reached')):
            with self.assertRaises(ValueError): h.input_authorities(release_fixture(),{})

    def test_policy_identity_is_exact_reviewed_source_bytes_before_reader(self):
        files,run=public_fixture();freeze=json.loads(files['authority/execution-freeze.json'])
        inp=json.loads(files['input-freeze.json']);release=json.loads(files['source/'+h.NEW+'execution-release.json'])
        authority=setup_binding()[1]['actual_authority']
        self.assertEqual(h.screening_policy_identity(freeze),c.identity(HERE/'asr18_metadata.py'))
        with patch.object(h,'read_json',side_effect=[inp,authority]):
            self.assertEqual(h.input_authorities(release,freeze),(inp,authority))
        changed=copy.deepcopy(inp);changed['actual_screening_receipt']['screening_policy_sha256']='e'*64
        admitted=copy.deepcopy(release)
        admitted['actual_screening_receipt_sha256']=c.digest(c.canonical(changed['actual_screening_receipt']))
        admitted['actual_input_freeze_sha256']=c.digest(c.canonical(changed))
        with patch.object(h,'read_json',side_effect=[changed,authority]), \
             patch('raw_data_reader.RawDataReader',side_effect=AssertionError('no download before policy binding')):
            with self.assertRaisesRegex(ValueError,'policy differs'):h.input_authorities(admitted,freeze)

    def test_policy_source_drift_fails_against_independent_freeze(self):
        files,_=public_fixture();freeze=json.loads(files['authority/execution-freeze.json'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/h.NEW/'asr18_metadata.py';path.parent.mkdir(parents=True)
            path.write_bytes(c.read_bytes(HERE/'asr18_metadata.py')+b'\n# synthetic source drift\n')
            with patch.object(h,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'source bytes drift'):h.screening_policy_identity(freeze)
        changed=copy.deepcopy(freeze);changed['inference_code'][h.NEW+'asr18_metadata.py']['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'one frozen'):h.screening_policy_identity(changed)

    def test_public_archive_policy_rechecks_pinned_algorithm_bytes(self):
        files,run=public_fixture()
        files['source/'+h.NEW+'asr18_metadata.py']+=b'\n# synthetic changed policy body\n'
        with self.assertRaisesRegex(ValueError,'verified scientific source byte identity'):h.validate_public_evidence(files,run)

    def test_download_passes_independent_policy_not_receipt_self_label(self):
        files,_=public_fixture();inp=json.loads(files['input-freeze.json'])
        release=json.loads(files['source/'+h.NEW+'execution-release.json'])
        raw,kwargs=setup_binding();verified=verify_screening_receipt(raw,**kwargs)
        reader=types.SimpleNamespace(read_complete=lambda *a,**k:{'archive':b'synthetic','accounting':{}})
        expected='d'*64
        with patch('raw_data_reader.RawDataReader',return_value=reader), \
             patch('input_binding.bind_complete_input_package',return_value={'verification':verified,'eligible_audio':{}}) as bind:
            h.verify_download(release,inp,kwargs['actual_authority'],expected_policy_sha256=expected)
        self.assertEqual(bind.call_args.kwargs['expected_policy_sha256'],expected)
        self.assertNotEqual(expected,inp['actual_screening_receipt']['screening_policy_sha256'])

    def test_zero_eligible_denominator_has_no_model_or_attempt(self):
        projection=invented_projection()
        for row in projection['cells']: row['disposition']='not_run_duplicate'
        blind=m.blind_metadata(projection)
        with tempfile.TemporaryDirectory() as temp:
            result=h.terminal_accounting(Path(temp),projection,blind['job'],blind['execution_plan'],context()[2],[])
        self.assertEqual(len(result['slots']),18);self.assertEqual(sum(x['attempts'] for x in result['slots']),0)
        self.assertTrue(all(x['status']=='not_run_duplicate' for x in result['slots']))

    def test_any_created_directory_consumes_even_without_started(self):
        projection=invented_projection();blind=m.blind_metadata(projection)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'output/qwen06'/blind['execution_plan']['audio_order'][0];folder.mkdir(parents=True)
            result=h.terminal_accounting(root,projection,blind['job'],blind['execution_plan'],context()[2],[])
        self.assertEqual(result['slots'][0],{'id':m.CELLS[0]+'-qwen06','status':'failed_no_retry','attempts':1})
        self.assertEqual(sum(x['attempts'] for x in result['slots']),1);self.assertTrue(result['accounting_valid'])

    def test_failed_completion_file_is_never_promoted_by_existence(self):
        projection=invented_projection();blind=m.blind_metadata(projection)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'output/qwen06'/blind['execution_plan']['audio_order'][0];folder.mkdir(parents=True)
            (folder/'completed.json').write_text('{}')
            with patch.object(h,'_completion',side_effect=AssertionError('failed current cannot validate into success')):
                result=h.terminal_accounting(root,projection,blind['job'],blind['execution_plan'],context()[2],[])
        self.assertEqual(result['slots'][0]['status'],'failed_no_retry')

    def test_valid_sequential_prefix_kept_and_later_all_notrun(self):
        projection=invented_projection();blind=m.blind_metadata(projection)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for clip in blind['execution_plan']['audio_order'][:2]: (root/'output/qwen06'/clip).mkdir(parents=True)
            with patch.object(h,'_completion',return_value={}):
                result=h.terminal_accounting(root,projection,blind['job'],blind['execution_plan'],context()[2],[])
        eligible=[x for x in result['slots'] if x['status'] not in ('not_run_quality','not_run_duplicate')]
        self.assertEqual([x['status'] for x in eligible[:3]],['complete','failed_no_retry','not_run'])
        self.assertEqual(sum(x['attempts'] for x in result['slots']),2)

    def test_collect_rejects_unknown_worker_member_not_prune(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'output/qwen06';folder.mkdir(parents=True)
            (folder/'secret.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unknown worker'):h.collect_public(root)

    def test_collect_never_walks_runtime_audio_or_private_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('input/audio/private.wav','asr-runtime/model.pt','private-history/private.json'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'private')
            self.assertEqual(h.collect_public(root),[])

    def test_preclaim_failure_keeps_source_and_all18_public_rows(self):
        files,run=public_fixture();self.assertTrue(h.validate_public_evidence(files,run))
        import evidence_transport as t
        t.validate_members(files,run)
        payload,manifest,_=t._zip_files(files,run)
        self.assertTrue(payload.startswith(b'PK'))
        t.validate_archive(payload,run)

    def test_public_receipt_separates_scientific_baseline_and_effective_window(self):
        files,run=public_fixture(); host=json.loads(files['host-receipt.json'])
        self.assertEqual(host['limits']['workflow_minutes'],70)
        self.assertEqual(host['limits_scope'],c.SCIENTIFIC_LIMITS_SCOPE)
        self.assertEqual(host['effective_execution_envelope'],
            {'workflow_minutes':80,'publisher_seconds':600,'host_seconds':3600})
        self.assertTrue(h.validate_public_evidence(files,run))
        for kind in ('missing_scope','wrong_scope','missing_effective','baseline_as_effective',
                     'publisher_expanded','host_expanded','bool_minutes','extra_effective'):
            value=json.loads(files['host-receipt.json'])
            if kind=='missing_scope':value.pop('limits_scope')
            elif kind=='wrong_scope':value['limits_scope']='current_execution_budget'
            elif kind=='missing_effective':value.pop('effective_execution_envelope')
            elif kind=='baseline_as_effective':value['effective_execution_envelope']['workflow_minutes']=70
            elif kind=='publisher_expanded':value['effective_execution_envelope']['publisher_seconds']=601
            elif kind=='host_expanded':value['effective_execution_envelope']['host_seconds']=3601
            elif kind=='bool_minutes':value['effective_execution_envelope']['workflow_minutes']=True
            else:value['effective_execution_envelope']['extra']=1
            altered=dict(files);altered['host-receipt.json']=c.canonical(value)
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):
                h.validate_public_evidence(altered,run)

    def test_extra_host_private_field_or_omitted_row_block_archive(self):
        for kind in ('extra','missing_row','forged_complete','claimed_success'):
            files,run=public_fixture()
            name='terminal-outcomes.json' if kind in ('missing_row','forged_complete') else 'host-receipt.json'
            value=json.loads(files[name])
            if kind=='extra':value['private_inventory']={'must':'not leak'}
            elif kind=='missing_row':value['slots'].pop()
            elif kind=='forged_complete':value['slots'][0].update(status='complete',attempts=1)
            else:value['status']='all_planned_native_complete'
            files[name]=c.canonical(value)
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):h.validate_public_evidence(files,run)

    def test_worker_envelope_exact_native_no_sensitive_mounts_or_env(self):
        run=context()[2]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(os,'getuid',return_value=1001):
                args=r.worker_arguments(name='k1-fixture-qwen',image='fixed',code=root/'code',runtime=root/'runtime',
                    output=root/'out',inputs=root/'input',family='qwen06',mode='prepare',order=root/'order',authority=root/'authority',
                    run_identity=run,freeze_sha='f'*64,claim_sha='e'*64)
        self.assertEqual(args[-2:],['qwen06','prepare']); self.assertIn('/code/research/k1-punctuation6-asr18-v1/native_worker.py',args)
        self.assertIn('KWS_SOURCE_CLAIM_SHA256='+'e'*64,args)
        self.assertFalse(any(any(secret in arg for secret in ('GITHUB_TOKEN','tts-source','private-history','seed')) for arg in args))
        self.assertTrue(any('dst=/order,readonly' in x for x in args));self.assertTrue(any('dst=/authority,readonly' in x for x in args))

    def test_retained_helpers_exact_static_node_bytes(self):
        source=(HERE.parent/'source-screen32-v1/hosted_run.py').read_text();tree=ast.parse(source)
        actual=r.__file__;new=Path(actual).read_text();newtree=ast.parse(new)
        def segments(text,tree):
            out={}
            for node in tree.body:
                name=node.name if isinstance(node,ast.FunctionDef) else node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else ''
                if name in r.RETAINED_NODE_SHA256:out[name]=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()
            return out
        self.assertEqual(segments(source,tree),r.RETAINED_NODE_SHA256);self.assertEqual(segments(new,newtree),r.RETAINED_NODE_SHA256)

    def test_fixed_candidate_workflow_matches_reviewed_template(self):
        text=(HERE/'workflow.yml').read_text()
        self.assertIn('timeout-minutes: 80',text);self.assertIn('persist-credentials: false',text)
        self.assertEqual(text.count('GITHUB_TOKEN:'),1)
        for value in ('workflow_dispatch:','schedule:','pull_request:','upload-artifact','self-hosted'):self.assertNotIn(value,text)
        self.assertEqual(c.read_bytes(h.ROOT/h.WORKFLOW),c.read_bytes(HERE/'workflow.yml'))


class NativeReceiptIntegration(unittest.TestCase):
    def test_real_synthetic_durable_completion_and_failed_current_reconciliation(self):
        from test_asr18_receipts import ReceiptFixture
        fixture=ReceiptFixture(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows=[]
        for index,cell in enumerate(m.CELLS):
            audio=fixture.input['audio'][index] if index<3 else {
                'wav_sha256':c.digest(('synthetic-extra-wav'+str(index)).encode()),
                'pcm_sha256':c.digest(('synthetic-extra-pcm'+str(index)).encode()),'audio_frames':1280}
            rows.append({'cell_id':cell,'status':'complete','attempts':1,
                'wav_sha256':audio['wav_sha256'],'pcm_sha256':audio['pcm_sha256'],
                'raw_wav_sha256':c.digest(('synthetic-raw-wav'+str(index)).encode()),
                'raw_float_sha256':c.digest(('synthetic-raw-float'+str(index)).encode()),
                'source_frames':1920,'frames':1280,'sample_rate_hz':16000,'channels':1,'sample_width_bytes':2,
                'eos_verified':True,'generation_iterations':2,'quality_flags':[]})
        projection=m.screen_metadata(rows,[{'wav_sha256':row['wav_sha256'],'pcm_sha256':row['pcm_sha256']} for row in rows[3:]])
        blind=m.blind_metadata(projection)
        self.assertEqual(blind['job'],fixture.job); self.assertEqual(blind['execution_plan'],fixture.plan)
        first=fixture.start();fixture.finish(first)
        second=fixture.start();fixture.finish(second) # A syntactically complete file can remain after an uncertain close.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output').mkdir();(root/'policy').mkdir()
            shutil.copytree(fixture.current['qwen06'],root/'output/qwen06')
            shutil.copyfile(fixture.policy_path,root/'policy/policy.json')
            slots=c.fixed_slots(projection,fixture.job,fixture.plan)
            checked=h._completion(root/'output/qwen06'/fixture.plan['audio_order'][0],slots[0],context()[2],c.digest(c.canonical(fixture.policy)))
            self.assertTrue(checked['decoder_completed'])
            ledger=h.terminal_accounting(root,projection,fixture.job,fixture.plan,context()[2],[])
            self.assertEqual([row['status'] for row in ledger['slots'][:3]],['complete','failed_no_retry','not_run'])
            self.assertEqual(sum(row['attempts'] for row in ledger['slots']),2)
            self.assertTrue(ledger['accounting_valid'])

    def test_source_bound_preclaim_failure_executes_no_runtime_and_retains_denominator(self):
        files,run=public_fixture(); frozen=json.loads(files['authority/execution-freeze.json'])
        inp=json.loads(files['input-freeze.json']); release=json.loads(files['source/'+h.NEW+'execution-release.json'])
        preflight=json.loads(files['run-preflight.json'])
        observed=[]
        def source_stage(roster,where):
            where.mkdir()
            for name in roster:
                path=where/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(files['source/'+name])
        def package(root,identity):
            captured={name:c.read_bytes(root/name) for name in h.collect_public(root)}
            h.validate_public_evidence(captured,identity)
            observed.append(captured)
            return b'synthetic-local-package',{}
        with tempfile.TemporaryDirectory() as temp, patch.object(h,'input_authorities',return_value=(inp,{})), \
             patch.object(h,'verify_download',side_effect=ValueError('synthetic rejected complete package')), \
             patch.object(r,'stage_source',side_effect=source_stage),patch.object(r,'setup_runtimes',side_effect=AssertionError('no setup')), \
             patch('resource_host.MeteredDocker',side_effect=AssertionError('no Docker')), \
             patch('evidence_transport.build_package',side_effect=package),patch('evidence_transport.write_package') as publish, \
             patch.object(h.signal,'signal'),patch.object(h.signal,'setitimer'),patch('builtins.print'):
            code=h.execute({'RUNNER_TEMP':temp},run,release,frozen,c.digest(c.canonical(frozen)),'d'*40,preflight)
        self.assertEqual(code,1); self.assertEqual(len(observed),1);publish.assert_called_once()
        self.assertNotIn('authority/source-claim.json',observed[0])
        self.assertEqual(len(json.loads(observed[0]['terminal-outcomes.json'])['slots']),18)

    def test_launch_inspect_rejects_token_env_and_substituted_mount(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(os,'getuid',return_value=1001):
            root=Path(temp)
            args=r.worker_arguments(name='synthetic',image='fixed',code=root/'code',runtime=root/'runtime',output=root/'out',
                inputs=root/'input',family='qwen06',mode='prepare',order=root/'order',authority=root/'authority',
                run_identity=context()[2],freeze_sha='f'*64,claim_sha='e'*64)
            mounts=[]
            for index,arg in enumerate(args):
                if arg=='--mount':
                    parts=args[index+1].split(',');spec=dict(p.split('=',1) for p in parts if '=' in p)
                    mounts.append({'Type':'bind','Destination':spec['dst'],'Source':spec['src'],'RW':'readonly' not in parts,'Propagation':'rprivate'})
            record={'Config':{'Image':'fixed','Cmd':args[args.index('fixed')+1:],'Entrypoint':None,
                'Env':[args[i+1] for i,arg in enumerate(args) if arg=='--env']},'HostConfig':{},'Mounts':mounts}
            with patch.object(r,'validate_container_inspect'):
                r.validate_launch(record,args,'qwen06')
                record['Config']['Env'].append('GITHUB_TOKEN=do-not-transmit')
                with self.assertRaisesRegex(ValueError,'credential'):r.validate_launch(record,args,'qwen06')
                record['Config']['Env'].pop();record['Mounts'][0]['Source']='/wrong'
                with self.assertRaisesRegex(ValueError,'substituted'):r.validate_launch(record,args,'qwen06')


class MockFlowTests(unittest.TestCase):
    def flow(self, fail_decode=False):
        files,run=public_fixture(); frozen=json.loads(files['authority/execution-freeze.json'])
        inp=json.loads(files['input-freeze.json']); release=json.loads(files['source/'+h.NEW+'execution-release.json'])
        preflight=json.loads(files['run-preflight.json']); events=[]; captured=[]
        job,plan,binding=inp['actual_blind_job'],inp['actual_opaque_order'],inp['actual_worker_binding']
        class Docker:
            def __init__(self,root,deadline):self.root=root;self.deadline=deadline;self.stage='setup'
            def finish(self):events.append('cleanup')
        def stage(roster,where):
            where.mkdir()
            for name in roster:
                path=where/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(files['source/'+name])
        def setup(root,freeze,docker,inputs,identity):
            events.append('setup')
            self.assertTrue((root/'authority/source-claim.json').exists())
            self.assertTrue((root/'order/execution-plan.json').exists())
            return {'runtimes':{family:root/'synthetic-runtime' for family in c.FAMILIES},
                'code':root/'code','image':'synthetic-image','seconds':0.0,'asr_body_bytes':0,'whisper':None}
        def supervise(docker,args,output,*,family,mode,deadline):
            events.append(mode+':'+family)
            self.assertEqual(args[-2:],[family,mode])
            if mode=='decode':
                for peer in c.FAMILIES:self.assertTrue((docker.root/'frozen'/peer/'worker-complete.json').exists())
                self.assertTrue((docker.root/'policy/budget-proof.json').exists())
                for audio_id in plan['audio_order']:
                    (output/audio_id).mkdir()
                    if fail_decode:raise OSError('synthetic current uncertain start')
            result={'schema':'k1-asr18-worker-result-v1','family':family,'mode':mode,
                'status':'prepared_no_recognition' if mode=='prepare' else 'all_planned_native_complete',
                'audio_forward_calls':0 if mode=='prepare' else len(job['clips']),
                'completed':0 if mode=='prepare' else len(job['clips']),
                'audio_ids':[] if mode=='prepare' else plan['audio_order'],
                'job_sha256':c.digest(c.canonical(job)),'execution_plan_sha256':c.digest(c.canonical(plan))}
            c.write_once(output/'worker-complete.json',result)
            if mode=='prepare':c.write_once(output/'prepared.json',result);c.write_once(output/'native-config.json',{})
        def proof(root,identity,**kw):
            events.append('proof')
            for family in c.FAMILIES:self.assertTrue((root/'output'/family/'prepared.json').exists())
            self.assertFalse(any((root/'output').rglob('clip-*')))
            return {'schema':'k1-asr18-predecode-package-budget-v1','status':'passed_before_any_audio_forward',
                'new_native_calls':0,'eligible_clip_count':len(job['clips']),'whisper_full_prefix_steps_per_clip':224,
                'transcript_reserve_characters_each_copy':32768,'transcript_copies_per_clip':2,'extra_json_reserve_bytes':c.MIB,
                'recognizer_policy_sha256':c.identity(kw['policy_path'])['sha256'],
                'worker_input_binding_sha256':c.digest(c.canonical(binding)),
                'execution_plan_sha256':c.digest(c.canonical(plan)),'job_sha256':c.digest(c.canonical(job)),
                'compressed_bytes':1,'expanded_member_bytes':1,'actual_prepared_members':1,'worst_fixture_members':1}
        def package(root,identity):
            events.append('package');captured.append(c.read_json(root/'terminal-outcomes.json'));return b'synthetic',{}
        verified={'verification':{'blind':{'eligible_cells':len(job['clips'])}},'eligible_audio':{}}
        with tempfile.TemporaryDirectory() as temp,patch.object(h,'input_authorities',return_value=(inp,{})), \
             patch.object(h,'verify_download',return_value=(verified,{})),patch('input_binding.stage_worker_inputs',return_value={}), \
             patch.object(r,'stage_source',side_effect=stage),patch.object(r,'setup_runtimes',side_effect=setup), \
             patch.object(r,'supervise',side_effect=supervise),patch('resource_host.MeteredDocker',Docker), \
             patch('worker_receipts.build_policy',return_value={}),patch('evidence_transport.package_budget_preflight',side_effect=proof), \
             patch('evidence_transport.build_package',side_effect=package),patch('evidence_transport.write_package'), \
             patch.object(h,'_completion',return_value={}),patch.object(os,'getuid',return_value=1001), \
             patch.object(h.shutil,'disk_usage',return_value=types.SimpleNamespace(free=12*c.GIB)), \
             patch.object(h.signal,'signal'),patch.object(h.signal,'setitimer'),patch('builtins.print'):
            code=h.execute({'RUNNER_TEMP':temp},run,release,frozen,c.digest(c.canonical(frozen)),'d'*40,preflight)
        return code,events,captured[0]

    def test_exact_three_prepare_then_proof_then_three_fresh_decode(self):
        code,events,terminal=self.flow()
        self.assertEqual(code,0)
        self.assertEqual(events,['setup',*[f'prepare:{f}' for f in c.FAMILIES],'proof',*[f'decode:{f}' for f in c.FAMILIES],'cleanup','package'])
        self.assertEqual(len(terminal['slots']),18);self.assertTrue(all(row['status']=='complete' for row in terminal['slots']))

    def test_first_uncertain_decode_stops_every_later_cell_and_family(self):
        code,events,terminal=self.flow(fail_decode=True)
        self.assertEqual(code,1)
        self.assertEqual(events,['setup',*[f'prepare:{f}' for f in c.FAMILIES],'proof','decode:qwen06','cleanup','package'])
        self.assertEqual(terminal['slots'][0]['status'],'failed_no_retry')
        self.assertEqual(sum(row['attempts'] for row in terminal['slots']),1)
        self.assertTrue(all(row['status']=='not_run' for row in terminal['slots'][1:]))


class CompleteSyntheticEndToEnd(unittest.TestCase):
    def run_case(self,fail_native=False):
        """Only acquisition/container/model constructors are fakes; archive path is real."""
        import evidence_transport as t
        import native_worker as nw
        import worker_receipts as wr
        from test_asr18_receipts import ReceiptFixture
        from asr_native_fixtures import native_fixture,rebind_native_fixture
        from test_asr18_binding import members_fixture
        files,run=public_fixture(); freeze=json.loads(files['authority/execution-freeze.json'])
        inp=json.loads(files['input-freeze.json']); release=json.loads(files['source/'+h.NEW+'execution-release.json'])
        preflight=json.loads(files['run-preflight.json'])
        raw,kwargs=setup_binding();verified=verify_screening_receipt(raw,**kwargs)
        wav_members=members_fixture()
        audio={row['wav_sha256']:wav_members['derived/'+row['cell_id']+'.wav'] for row in verified['projection']['cells']}
        bound={'verification':verified,'eligible_audio':audio}
        data={'gets':2,'response_body_bytes':100,'response_body_reservation_bytes':16*c.MIB,
            'retries':0,'redirects':0,'extra_source_gets':0,'failed':False,'execution_enabled':False}
        events=[];results=[]
        synthetic=object.__new__(ReceiptFixture);synthetic.captures={};synthetic.configs={}
        class Docker:
            def __init__(self,root,deadline):self.root=root;self.deadline=deadline;self.stage='setup'
            def finish(self):events.append('cleanup')
        def source_stage(roster,where):
            where.mkdir()
            for name in roster:
                path=where/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(files['source/'+name])
        def setup(root,*_args):
            events.append('setup')
            return {'runtimes':{f:root/'synthetic-runtime' for f in c.FAMILIES},'code':root/'synthetic-code',
                'image':'synthetic-image','seconds':0.0,'asr_body_bytes':0,'whisper':None}
        def construct(family,runtime,inputs,output,job,run_id,head):
            # Invented source/asset identities carry no real runtime/model claims.
            body=b'# invented model-free fixture source\n'
            source={'schema':'retained-asr-source-lock-v1','executed_head_sha':head,'workflow_run_id':run_id,
                'model_id':'invented-'+family,'empty_source_files':[],
                'files':[{'path':'output/'+family+'/retained-source/invented.py','bytes':len(body),
                    'sha256':c.digest(body),'role':'staged-public-code'}]}
            if family=='whisper':source.update(official_source_revision='invented',original_source_lock_canonical_sha256='1'*64)
            else:source['original_source_lock_sha256']='1'*64
            config,capture=native_fixture(wr.ADAPTERS[family],audio_frames=2560)
            assets=[{'path':'invented-'+family+'-unloaded-model','bytes':1,'sha256':c.digest(family.encode()),
                'body_retained':False,'kind':'invented-metadata-no-model'}]
            model={'schema':'retained-asr-model-lock-v1','assets':assets,'model_sha256':c.digest(c.canonical(assets)),
                'provider_id':'invented','model_id':'invented-'+family,'model_revision':'invented-v1',
                'model_family':config['model_family'],'asset_verification':'synthetic-metadata-only',
                'original_asset_lock_sha256':'2'*64}
            config['source_lock_sha256']=c.digest(c.canonical(source));config['asset_lock_sha256']=c.digest(c.canonical(model))
            capture['source_lock_sha256']=config['source_lock_sha256'];capture['asset_lock_sha256']=config['asset_lock_sha256']
            rebind_native_fixture(config,capture)
            synthetic.configs[family]=config;synthetic.captures[family]=capture
            (output/'retained-source').mkdir();(output/'retained-source/invented.py').write_bytes(body)
            for leaf,value in zip(wr.FILES,(source,model,config)):c.write_once(output/leaf,value)
            class Session:
                def bind(self,clip):return clip['audio_id']
                def infer(self,audio_id,directory):
                    events.append('synthetic-call:'+family+':'+audio_id)
                    if fail_native:raise OSError('synthetic native return fault')
                    row,value=synthetic.result(family)
                    def shaped(reference):
                        rule=t._RETAINED_SHAPES[reference];kind=rule[0]
                        if kind=='literal':return rule[1]
                        if kind=='str':return 'invented-synthetic-only'
                        if kind=='number':return 0
                        if kind=='bool':return False
                        if kind in ('null','scalar'):return None
                        if kind=='list':return []
                        if kind=='map':return {}
                        if kind=='object':return {key:shaped(ref) for key,ref in rule[1].items()}
                        raise AssertionError('unknown synthetic shape')
                    full=shaped(t._RETAINED_MEMBERS[family+'/clip/raw-native-output.json'][0])
                    for key,actual in value.items():
                        if type(actual) is dict and type(full.get(key)) is dict:
                            full[key].update(actual)
                        else:full[key]=actual
                    value=full
                    if family=='sensevoice':
                        decoded=value['raw_result'][0][0]['text']
                        value['raw_result']=[[{'key':'invented','text':decoded}],shaped('179')]
                    if family=='whisper':
                        c.write_once(directory/'raw-output.json',value)
                        termination=value['native_capture']['termination'];ids=list(termination['initial_token_ids'])
                        for step,token in enumerate(termination['generated_token_ids'],1):
                            ids.append(token);c.write_once(directory/f'update-{step:03d}.json',
                                {'step':step,'token_ids':list(ids),'completed':step==len(termination['generated_token_ids'])})
                    return row,value
            return Session()
        def supervise(docker,args,output,*,family,mode,deadline):
            events.append(mode+':'+family)
            kernel={'schema':'screen32-kernel-scope-v1','limits':{'memory.max':str(12*c.GIB),'memory.swap.max':'0',
                'cpu.max':'400000 100000','pids.max':'256'},'uid':1001,'private_cgroup_root':True,'read_only_root':True,
                'capabilities':'none','no_new_privileges':True,'seccomp_filter':True,'network_interfaces':['lo'],'network_disabled':True}
            guard={'schema':'qwen12-inference-process-guard-v1','family':family,'scope':'synthetic','clone3':'synthetic',
                'fork_vfork_nonthread_clone':'synthetic','rlimit_cpu_seconds':600,'scheduler_granularity_not_zero_overshoot':True}
            job=verified['blind']['job'];plan=verified['blind']['execution_plan'];binding=verified['worker_input_binding']
            for leaf,value in (('kernel-scope.json',kernel),('cpu-process-guard.json',guard),('blind-job.json',job),('execution-plan.json',plan)):
                c.write_once(output/leaf,value)
            nw._FAILURE_FAMILY,nw._FAILURE_MODE=family,mode
            try:
                with patch.object(nw,'_construct_native',side_effect=construct),patch.object(nw.signal,'setitimer'):
                    result=nw._execute_family(family,mode,runtime=docker.root/'synthetic-runtime',inputs=docker.root/'input',output=output,
                        frozen=docker.root/'frozen'/family,policy_path=docker.root/'policy/policy.json',job=job,execution_plan=plan,
                        input_binding=binding,run_id=str(run['run_id']),head=run['executed_head_sha'],
                        policy_sha256=c.identity(docker.root/'policy/policy.json')['sha256'] if mode=='decode' else '')
            except BaseException as error:
                c.write_once(output/'worker-failure.json',{'status':'failed_no_retry','error_type':nw._safe_error_type(error),
                    'diagnostic':nw._failure_diagnostic(error)})
                raise
            c.write_once(output/'worker-complete.json',result)
        # No collect_public, policy, native receipts, budget validator, compressor,
        # archive validator or publication-stash helper is replaced by this test.
        with tempfile.TemporaryDirectory() as temp,patch.object(h,'input_authorities',return_value=(inp,kwargs['actual_authority'])), \
             patch.object(h,'verify_download',return_value=(bound,data)),patch.object(r,'stage_source',side_effect=source_stage), \
             patch.object(r,'setup_runtimes',side_effect=setup),patch.object(r,'supervise',side_effect=supervise), \
             patch('resource_host.MeteredDocker',Docker),patch.object(os,'getuid',return_value=1001), \
             patch.object(h.shutil,'disk_usage',return_value=types.SimpleNamespace(free=12*c.GIB)), \
             patch.object(h.signal,'signal'),patch.object(h.signal,'setitimer'),patch('builtins.print'):
            code=h.execute({'RUNNER_TEMP':temp},run,release,freeze,c.digest(c.canonical(freeze)),'d'*40,preflight)
            root=Path(temp)/(c.EXPERIMENT+'-123')
            result=c.read_json(root/'host-receipt.json');terminal=c.read_json(root/'terminal-outcomes.json')
            # Surface a fixed diagnostic type if this assertion fails; no model/log text.
            self.assertEqual(code,int(fail_native),result.get('primary_error'))
            self.assertEqual([row['status'] for row in terminal['slots']],
                ['failed_no_retry']+['not_run']*17 if fail_native else ['complete']*18)
            proof=c.read_json(root/'policy/budget-proof.json')
            self.assertEqual(proof['eligible_clip_count'],6);self.assertGreater(proof['worst_fixture_members'],1300)
            payload=t.read_package(temp,run);manifest,header=t.validate_archive(payload,run)
            self.assertEqual(len(manifest['members']),len(h.collect_public(root)))
            self.assertTrue(all(not name.endswith('.wav') for name in manifest['members']))
            self.assertEqual(sum(name.endswith('/started.json') for name in manifest['members']),1 if fail_native else 18)
            self.assertEqual(sum(name.endswith('/completed.json') for name in manifest['members']),0 if fail_native else 18)
            self.assertEqual(sum(name.startswith('frozen/') and name.endswith('/prepared.json') for name in manifest['members']),3)
            mutated={name:c.read_bytes(root/name) for name in h.collect_public(root)}
            mutated['frozen/qwen06/worker-failure.json']=c.canonical({'synthetic':'conflicting prepare failure'})
            with self.assertRaisesRegex(ValueError,'prepare failure'):h.validate_public_evidence(mutated,run)
        self.assertEqual(sum(item.startswith('synthetic-call:') for item in events),1 if fail_native else 18)
        self.assertEqual([item for item in events if item.startswith(('prepare:','decode:'))],
            [*[f'prepare:{f}' for f in c.FAMILIES],*[f'decode:{f}' for f in (('qwen06',) if fail_native else c.FAMILIES)]])

    def test_real_prepare_proof_fresh18_receipts_and_complete_audiofree_package(self):
        self.run_case()

    def test_first_native_failure_retains_complete_source_and_all18_without_retry(self):
        self.run_case(fail_native=True)


if __name__=='__main__': unittest.main()
