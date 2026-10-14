"""Fixed source/Git/API/diagnostic integration. Stdlib; never ML or networking."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import prepare_run as h
import prepare_contract as pc
import prepare_observer as obs
import prepare_freeze_diagnostic as freezer
import diagnostic_transport as transport
import diagnostic_publisher as publisher

SECRET='PRIVATE_DIAGNOSTIC_SENTINEL_TOKEN_AUDIO_HISTORY'

def positive_release_fixture():
    # Explicit source-only positive fixture; independent of checked-in approval state.
    release=pc.decode(pc.regular(HERE/'prepare-release.json'))
    for key in ('enabled','independent_source_review','source_closure_ready'):release[key]=True
    release['prior_prepare_history_sha256']=pc.digest(pc.canonical(h.prior_history_identity()))
    return pc.validate_release(release)

@contextmanager
def explicit_disabled_source_copy():
    """Actual source/control files, with real release validators left intact."""
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)/'checkout';root.mkdir();fixed=root/pc.REL
        names=set(freezer.build(h.ROOT)['host_code'])|{pc.REL+'prepare-freeze.json'}
        for name in names:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(h.read_bytes(h.ROOT/name))
        for name,keys in (('prepare-release.json',('enabled','independent_source_review','source_closure_ready')),
                ('publication-release.json',('enabled','independent_source_review','root_publication_release'))):
            value=h.read_json(fixed/name)
            for key in keys:value[key]=False
            (fixed/name).write_bytes(pc.canonical(value))
        yield root,fixed


@contextmanager
def explicit_enabled_source_copy():
    """Temporary positive controls; no checked-in authority is changed."""
    with explicit_disabled_source_copy() as (root,fixed):
        (fixed/'prepare-release.json').write_bytes(pc.canonical(positive_release_fixture()))
        control=h.read_json(fixed/'publication-release.json')
        for key in ('enabled','independent_source_review','root_publication_release'):control[key]=True
        (fixed/'publication-release.json').write_bytes(pc.canonical(control))
        (fixed/'prepare-freeze.json').write_bytes(pc.canonical(freezer.build(root)))
        yield root,fixed


def context(head='c'*40):
    env={'GITHUB_REPOSITORY':h.REPOSITORY,'GITHUB_REF':'refs/heads/'+h.SOURCE_REF,'GITHUB_EVENT_NAME':'push',
        'GITHUB_RUN_ATTEMPT':'1','GITHUB_JOB':'prepare_diagnostic','RUNNER_ENVIRONMENT':'github-hosted',
        'RUNNER_OS':'Linux','RUNNER_ARCH':'X64','ImageOS':'ubuntu24','GITHUB_SHA':head,'GITHUB_RUN_ID':'12345',
        'GITHUB_WORKFLOW_REF':h.REPOSITORY+'/'+h.WORKFLOW+'@refs/heads/'+h.SOURCE_REF,'GITHUB_WORKFLOW_SHA':head}
    event={'repository':{'full_name':h.REPOSITORY,'private':False},'created':True,'deleted':False,'forced':False,
        'before':'0'*40,'after':head,'ref':env['GITHUB_REF'],'head_commit':{'timestamp':'2026-10-12T00:00:00Z'}}
    return env,event,h.event_identity(env,event)


class API:
    def __init__(self,run):
        base='/repos/'+h.REPOSITORY;head=run['executed_head_sha'];ident=run['run_id']
        current={'id':ident,'run_attempt':1,'status':'in_progress','conclusion':None,'event':'push',
            'head_branch':h.SOURCE_REF,'head_sha':head,'path':h.WORKFLOW,'workflow_id':8,
            'repository':{'full_name':h.REPOSITORY,'private':False},'head_repository':{'full_name':h.REPOSITORY}}
        self.rows={base+'/actions/runs/'+str(ident):current,base+'/actions/workflows/8':{'id':8,'path':h.WORKFLOW},
            base+'/actions/workflows/8/runs?branch='+h.urllib.parse.quote(h.SOURCE_REF,safe='')+'&per_page=100&page=1':{'total_count':1,'workflow_runs':[copy.deepcopy(current)]},
            base+'/actions/runs/'+str(ident)+'/attempts/1/jobs?per_page=100&page=1':{'total_count':1,'jobs':[{'id':999,'name':'prepare_diagnostic','run_id':ident,
                'status':'in_progress','conclusion':None,'head_sha':head,'labels':['ubuntu-24.04'],'runner_name':'GitHub Actions 1','runner_group_name':'GitHub Actions'}]},
            base+'/git/ref/heads/'+h.SOURCE_REF:{'ref':'refs/heads/'+h.SOURCE_REF,'object':{'type':'commit','sha':head}},
            base+'/git/ref/heads/'+h.EVIDENCE_REF:None}
        self.calls=[]
    def get(self,path,*,absent=False):
        if path in self.calls:raise AssertionError('no repeat GET')
        self.calls.append(path);value=self.rows[path]
        if absent:
            if value is not None:raise ValueError('occupied result')
        return copy.deepcopy(value)
    def accounting(self):return {'gets':len(self.calls),'response_body_bytes':1,'response_body_cap_bytes':h.c.MIB,
        'scope':'public_api_metadata_separate_from_model_audio_bodies','hard_wire_byte_cap':False}


class HostedTests(unittest.TestCase):
    def test_checked_in_prepare_closed_and_current_asr_authorized_with_pinned_prerequisite(self):
        release=pc.decode(pc.regular(HERE/'prepare-release.json'))
        control=h.read_json(HERE/'publication-release.json')
        flags=[release[k] for k in ('enabled','independent_source_review','source_closure_ready')]
        flags += [control[k] for k in ('enabled','independent_source_review','root_publication_release')]
        self.assertTrue(all(value is False for value in flags))
        with self.assertRaisesRegex(ValueError,'prepare_execution_disabled'):pc.release_first()
        with self.assertRaises(publisher.PublicationDisabled):publisher.publication_gate(control)
        self.assertEqual(release['max_new_asr_calls'],0);self.assertFalse(release['audio_forward_allowed'])
        self.assertNotEqual(release['prior_prepare_history_sha256'],'0'*64)
        self.assertEqual(release['prior_prepare_history_sha256'],pc.digest(pc.canonical(h.prior_history_identity())))
        history=h.prior_history_identity()
        self.assertEqual(history['observation_status'],'public_read_observed')
        self.assertRegex(history['observed_at'],r'^2026-10-12T[0-9:.]+Z$')
        self.assertEqual(history['prior_runs'],[])
        self.assertEqual(history['all_branch_push_history_count'],0)
        self.assertIs(history['source_ref_absent'],True);self.assertIs(history['evidence_ref_absent'],True)
        for name,keys in (('execution-release.json',('enabled','independent_source_review','root_execution_release')),
                          ('publication-release.json',('enabled','independent_source_review','root_publication_release'))):
            value=h.read_json(h.ASR/name)
            self.assertTrue(all(value[k] is True for k in keys))
            if name=='execution-release.json':
                self.assertEqual(value['experiment'],h.c.EXPERIMENT)
                self.assertEqual(h.retained_host.release_first(),value)
                self.assertEqual(pc.digest(pc.regular(h.ASR/'prepare-prerequisite.json')),value['prepare_prerequisite_sha256'])
        self.assertEqual([r['run_id'] for r in h.prior_history_identity()['sealed_asr_runs']],[38207026177,38219171568])
        self.assertTrue(all(r['actual_consumed_attempts'] is None and r['reusable_slots']==0 for r in h.prior_history_identity()['sealed_asr_runs']))

    def test_explicit_positive_fixture_api_preflight_and_history_rejections(self):
        release=positive_release_fixture();_,_,run=context();api=API(run)
        result=h.api_preflight(api,run,release)
        self.assertEqual(len(api.calls),6);self.assertEqual(result['numeric_job_id'],999)
        key=next(k for k in api.rows if '/runs?branch=' in k)
        for change in ('second','old','attempt2','consumed_prepare','sealed_r1','wrong_ref','old_r1_ref'):
            bad=API(run)
            if change=='second':bad.rows[key]['total_count']=2;bad.rows[key]['workflow_runs']*=2
            elif change=='old':bad.rows[key]['workflow_runs'][0]['id']=38219171568
            elif change=='consumed_prepare':bad.rows[key]['workflow_runs'][0]['id']=38234429437
            elif change=='sealed_r1':bad.rows[key]['workflow_runs'][0]['id']=38251170134
            elif change=='old_r1_ref':bad.rows[key]['workflow_runs'][0]['head_branch']='research/k1-punctuation6-prepare-diagnostic-r1'
            elif change=='wrong_ref':bad.rows[key]['workflow_runs'][0]['head_branch']='research/k1-punctuation6-prepare-diagnostic-v1'
            else:bad.rows[key]['workflow_runs'][0]['run_attempt']=2
            with self.subTest(change=change),self.assertRaises(ValueError):h.api_preflight(bad,run,release)
        wrong=dict(release,prior_prepare_history_sha256=pc.digest(pc.canonical(h.prior_history_identity())+b'\n'))
        with self.assertRaises(ValueError):h.api_preflight(API(run),run,wrong)

    def test_wrong_ref_old_run_and_attempt_reject_before_external_call(self):
        env,event,_=context()
        for changed in ({'GITHUB_REF':'refs/heads/research/k1-punctuation6-asr18-r1'},
                        {'GITHUB_REF':'refs/heads/research/k1-punctuation6-prepare-diagnostic-v1'},
                        {'GITHUB_REF':'refs/heads/research/k1-punctuation6-prepare-diagnostic-r1'},
                        {'GITHUB_RUN_ID':'38251170134'},
                        {'GITHUB_RUN_ID':'38234429437'}, {'GITHUB_RUN_ATTEMPT':'2'},
                        {'GITHUB_RUN_ID':'38219171568'},{'GITHUB_RUN_ID':'38207026177'},{'GITHUB_JOB':'asr18'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):h.event_identity({**env,**changed},event)

    def test_new_identity_retains_source_paths_workflow_and_historical_failure(self):
        self.assertEqual(pc.EXPERIMENT,'k1-punctuation6-prepare-diagnostic-r2')
        self.assertEqual(pc.REL,'research/k1-punctuation6-prepare-diagnostic-v1/')
        self.assertEqual(h.WORKFLOW,'.github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml')
        self.assertEqual(h.SOURCE_REF,'research/'+pc.EXPERIMENT)
        self.assertEqual(h.EVIDENCE_REF,'evidence/'+pc.EXPERIMENT)
        self.assertEqual(transport.SOURCE_REL,pc.REL)
        self.assertEqual(transport.POLICY['workflow'],h.WORKFLOW)
        self.assertEqual(transport.POLICY['source_branch'],h.SOURCE_REF)
        self.assertEqual(transport.POLICY['result_branch'],h.EVIDENCE_REF)
        self.assertEqual(transport.SOURCE_FREEZE,'source/'+pc.REL+'prepare-freeze.json')
        old=h.prior_history_identity()['sealed_prepare_runs']
        self.assertEqual(len(old),2)
        self.assertEqual(old[0]['run_id'],38234429437)
        self.assertEqual(old[0]['source_sha'],'54ee9ee80e88c172cd04611474ded2613d956745')
        self.assertEqual(old[0]['result_sha'],'97141cea41cc5cc18fac3f5a434777991bfd4864')
        self.assertTrue(old[0]['consumed']);self.assertFalse(old[0]['retry_authorized'])
        self.assertEqual(old[0]['reusable_slots'],0)
        self.assertEqual(old[0]['completed_constructor_receipts'],0)
        self.assertFalse(old[0]['prepare_completion_established'])
        self.assertFalse(old[0]['capacity_success_established'])
        self.assertIsNone(old[0]['asr_forward_calls'])
        r1=old[1]
        self.assertEqual(r1['run_id'],38251170134)
        self.assertEqual(r1['numeric_job_id'],114817557432)
        self.assertEqual(r1['source_sha'],pc.SEALED_PREPARE_R1_SOURCE)
        self.assertEqual(r1['result_sha'],pc.SEALED_PREPARE_R1_RESULT)
        self.assertEqual(r1['archive_sha256'],'a10c27861d69a78d694de37b7cfe244eea87b91404591019832e87c880628984')
        self.assertEqual(r1['conclusion'],'failure');self.assertEqual(r1['host_step'],'failure')
        self.assertEqual(r1['publisher_step'],'success');self.assertTrue(r1['identity_sealed'])
        self.assertFalse(r1['retry_authorized']);self.assertEqual(r1['reusable_slots'],0)
        self.assertEqual(r1['completed_constructor_receipts'],0)
        for key in ('consumed','constructor_entry_count','constructor_completion_count',
                    'asr_forward_calls','consumed_asr_attempts'):
            self.assertIsNone(r1[key])
        self.assertFalse(r1['prepare_completion_established']);self.assertFalse(r1['capacity_success_established'])
        failure=h.read_json(HERE/'FAILURE-38251170134.json')
        self.assertEqual(failure['run_id'],r1['run_id'])
        self.assertEqual(failure['source_commit'],r1['source_sha'])
        self.assertEqual(failure['result_commit'],r1['result_sha'])
        self.assertEqual(failure['archive_sha256'],r1['archive_sha256'])
        self.assertEqual(pc.digest((HERE/'FAILURE-38251170134.json').read_bytes()),'d1bfbe474402991908d6841ded360c10e64e867dd0401cfc64c94f7fad6ee224')
        text=(HERE/'workflow.yml').read_text()
        self.assertIn('      - '+h.SOURCE_REF,text)
        self.assertIn('  group: '+pc.EXPERIMENT,text)
        self.assertIn("github.ref == 'refs/heads/"+h.SOURCE_REF+"'",text)
        self.assertIn(pc.REL+'prepare_run.py',text)
        self.assertNotIn('      - research/k1-punctuation6-prepare-diagnostic-v1',text)

    def test_sealed_identity_rejected_by_contract_transport_observer_and_preflight(self):
        _,_,run=context()
        self.assertEqual(pc.validate_identity(run),run)
        self.assertEqual(obs.identity(run),run)
        transport.identity(run)
        changes=[{'run_id':n} for n in (38207026177,38219171568,38234429437,38251170134)]
        changes += [{'run_attempt':2},{'run_id':True},
            {'executed_head_sha':pc.SEALED_PREPARE_SOURCE},{'executed_head_sha':pc.SEALED_PREPARE_RESULT},
            {'executed_head_sha':pc.SEALED_PREPARE_R1_SOURCE},{'executed_head_sha':pc.SEALED_PREPARE_R1_RESULT}]
        for change in changes:
            bad={**run,**change}
            for validator in (pc.validate_identity,obs.identity,transport.identity):
                with self.subTest(change=change,validator=validator.__module__),self.assertRaises(ValueError):validator(bad)
            api=unittest.mock.Mock(side_effect=AssertionError('API touched before identity rejection'))
            with self.subTest(change=change),self.assertRaises(ValueError):h.api_preflight(api,bad,positive_release_fixture())
            self.assertEqual(api.mock_calls,[])

    def test_new_evidence_and_observer_workspace_names_are_separate(self):
        _,_,run=context()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            package=transport.package_path(root,run)
            self.assertTrue(package.name.startswith(pc.EXPERIMENT+'-evidence-12345-attempt-1-'))
            for role in ('host','publisher'):
                observer=obs.Observation(role)
                observer.attach(run,root)
                self.assertEqual(observer.receipt_dir,root/(pc.EXPERIMENT+'-observation-12345-'+role))

    def test_old_release_refs_and_pending_history_do_not_authorize_new_run(self):
        for key,value in (('experiment','k1-punctuation6-prepare-diagnostic-r1'),
            ('source_ref','research/k1-punctuation6-prepare-diagnostic-r1'),
            ('evidence_ref','evidence/k1-punctuation6-prepare-diagnostic-r1'),
            ('experiment','k1-punctuation6-prepare-diagnostic-v1'),
            ('source_ref','research/k1-punctuation6-prepare-diagnostic-v1'),
            ('evidence_ref','evidence/k1-punctuation6-prepare-diagnostic-v1'),
            ('workflow','.github/workflows/k1-punctuation6-prepare-diagnostic-r1.yml'),
            ('prior_prepare_history_sha256','0'*64)):
            release=positive_release_fixture();release[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):pc.validate_release(release)
        control=publisher.disabled_publication_release()
        for key in ('enabled','independent_source_review','root_publication_release'):control[key]=True
        publisher.publication_gate(control)
        for key,value in (('source_branch','research/k1-punctuation6-prepare-diagnostic-r1'),
            ('result_branch','evidence/k1-punctuation6-prepare-diagnostic-r1'),
            ('source_branch','research/k1-punctuation6-prepare-diagnostic-v1'),
            ('result_branch','evidence/k1-punctuation6-prepare-diagnostic-v1'),
            ('workflow','.github/workflows/k1-punctuation6-prepare-diagnostic-r1.yml')):
            with self.subTest(key=key),self.assertRaises(ValueError):publisher.publication_gate({**control,key:value})

    def check_rejected_cli(self, fixed, cwd, env, args=()):
        records={}
        for filename in ('prepare_run.py','diagnostic_publisher.py'):
            process=subprocess.run([sys.executable,'-I','-B',str(fixed/filename),*args],
                env=env,cwd=cwd,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(process.returncode,0);self.assertNotIn('Traceback',process.stderr)
            self.assertNotIn('ModuleNotFoundError',process.stderr);self.assertNotIn(SECRET,process.stdout+process.stderr)
            rows=[json.loads(x) for x in process.stdout.splitlines()]
            self.assertTrue(any('diagnostic' in row.get('schema','') for row in rows))
            record=next(row['record'] for row in rows if row.get('schema')=='k1-prepare-independent-diagnostic-envelope-v1')
            self.assertEqual(record['status'],'failed_or_unconfirmed')
            self.assertIsNone(record['identity'])
            records[filename]=record
        return records

    def test_explicit_disabled_isolated_cli_rejects_before_runtime(self):
        with explicit_disabled_source_copy() as (root,fixed):
            before={p.relative_to(root).as_posix() for p in root.rglob('*')}
            self.check_rejected_cli(fixed,root,{'PATH':'/usr/bin:/bin','GITHUB_TOKEN':SECRET,
                'RUNNER_TEMP':str(root/'forbidden-output')})
            self.assertEqual(before,{p.relative_to(root).as_posix() for p in root.rglob('*')})

    def test_actual_checked_in_disabled_cli_rejects_at_release_without_side_effects(self):
        # Closed prepare controls must reject before event/Git/API/runtime.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);event=root/'event.json';event.write_bytes(b'{}')
            env={'PATH':'/usr/bin:/bin','GITHUB_TOKEN':SECRET,'RUNNER_TEMP':str(root/'forbidden-output'),
                'GITHUB_EVENT_PATH':str(event),'GITHUB_REPOSITORY':'invalid/offline-fixture',
                'RUNNER_ENVIRONMENT':'self-hosted','GITHUB_ACTIONS':'false'}
            records=self.check_rejected_cli(HERE,root,env)
            self.assertEqual(records['prepare_run.py']['stage'],'release')
            self.assertEqual(records['diagnostic_publisher.py']['stage'],'publication_gate')
            self.assertEqual({p.name for p in root.iterdir()},{'event.json'})
            self.assertEqual(event.read_bytes(),b'{}')

    def test_explicit_enabled_copy_passes_real_gates_then_rejects_context_without_side_effects(self):
        # Preserve the former enabled-path coverage using real temporary controls.
        # Only the source path changes; no release validator is mocked.
        with explicit_enabled_source_copy() as (source,fixed),tempfile.TemporaryDirectory() as temp:
            release=h.read_json(fixed/'prepare-release.json')
            with patch.object(pc,'HERE',fixed):self.assertEqual(pc.release_first(),release)
            publisher.publication_gate(h.read_json(fixed/'publication-release.json'))
            before={p.relative_to(source).as_posix():p.read_bytes() for p in source.rglob('*') if p.is_file()}
            root=Path(temp);event=root/'event.json';event.write_bytes(b'{}')
            env={'PATH':'/usr/bin:/bin','GITHUB_TOKEN':SECRET,'RUNNER_TEMP':str(root/'forbidden-output'),
                'GITHUB_EVENT_PATH':str(event),'GITHUB_REPOSITORY':'invalid/offline-fixture',
                'RUNNER_ENVIRONMENT':'self-hosted','GITHUB_ACTIONS':'false',
                'KWS_PREPARE_FREEZE_SHA256':pc.digest((fixed/'prepare-freeze.json').read_bytes())}
            records=self.check_rejected_cli(fixed,root,env)
            self.assertEqual(records['prepare_run.py']['stage'],'context')
            self.assertEqual(records['diagnostic_publisher.py']['stage'],'source_release')
            self.assertEqual({p.name for p in root.iterdir()},{'event.json'})
            self.assertEqual(event.read_bytes(),b'{}')
            self.assertEqual(before,{p.relative_to(source).as_posix():p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_explicit_disabled_gates_precede_environment_api_git_and_runtime(self):
        class Poison(dict):
            def __getitem__(self,key):raise AssertionError('environment touched')
            def get(self,*args):raise AssertionError('environment touched')
            def __contains__(self,key):raise AssertionError('environment touched')
            def keys(self):raise AssertionError('environment iterated')
        with explicit_disabled_source_copy() as (_,fixed):
            with patch.object(pc,'HERE',fixed),patch.object(h.os,'environ',Poison()), \
                 patch.object(h,'PublicAPI',side_effect=AssertionError('API')), \
                 patch.object(h,'git_read',side_effect=AssertionError('Git')), \
                 patch.object(h,'execute',side_effect=AssertionError('model')),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'prepare_execution_disabled'):h.main([])
            with patch.object(publisher,'HERE',fixed),patch.object(publisher.os,'environ',Poison()), \
                 patch.object(publisher,'GitHubAPI',side_effect=AssertionError('API')), \
                 patch.object(publisher,'_source_git',side_effect=AssertionError('Git')),redirect_stdout(io.StringIO()):
                with self.assertRaises(publisher.PublicationDisabled):publisher.main([])

    def test_explicit_disabled_host_gate_precedes_api_git_runtime(self):
        release=pc.decode(pc.regular(HERE/'prepare-release.json'));release['enabled']=False
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'prepare-release.json').write_bytes(pc.canonical(release))
            with patch.object(pc,'HERE',folder),patch.object(h,'PublicAPI',side_effect=AssertionError('API')), \
                 patch.object(h,'git_read',side_effect=AssertionError('Git')),patch.object(h,'execute',side_effect=AssertionError('model')),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'prepare_execution_disabled'):h.main([])

    def test_exact_git_source_closure_and_real_authority_without_download(self):
        # Only Git is real. This source-only check cannot import/load a model.
        freeze=freezer.build(h.ROOT);self.assertEqual(pc.canonical(freeze),(HERE/'prepare-freeze.json').read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'checkout';root.mkdir();fixed=root/pc.REL;fixed.mkdir(parents=True)
            names=set(freeze['host_code'])|{pc.REL+'prepare-freeze.json',h.WORKFLOW}
            for name in names:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(h.read_bytes(h.ROOT/name))
            (fixed/'prepare-release.json').write_bytes(pc.canonical(positive_release_fixture()))
            fixture_freeze=freezer.build(root)
            (fixed/'prepare-freeze.json').write_bytes(pc.canonical(fixture_freeze))
            env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null',
                'GIT_AUTHOR_NAME':'Offline fixture','GIT_AUTHOR_EMAIL':'fixture@example.invalid','GIT_COMMITTER_NAME':'Offline fixture',
                'GIT_COMMITTER_EMAIL':'fixture@example.invalid','GIT_AUTHOR_DATE':'2026-10-12T00:00:00Z','GIT_COMMITTER_DATE':'2026-10-12T00:00:00Z'}
            def git(*args):return subprocess.run(['git','-C',str(root),*args],env=env,capture_output=True,check=True,timeout=10).stdout
            git('init','--quiet');git('add','.');git('commit','--quiet','-m','offline fixed source fixture')
            head=git('rev-parse','HEAD').decode().strip();_,_,run=context(head)
            with patch.object(h,'ROOT',root),patch.object(h,'HERE',fixed),patch.object(h,'git_read',side_effect=git):
                actual,identity,tree=h.source_authority(run,positive_release_fixture())
                self.assertEqual(actual,fixture_freeze);self.assertEqual(identity,pc.digest(pc.canonical(fixture_freeze)))
                self.assertEqual(tree,git('rev-parse','HEAD^{tree}').decode().strip())
                (fixed/'prepare_run.py').write_bytes(b'changed')
                with self.assertRaises(ValueError):h.source_authority(run,positive_release_fixture())
        inp,authority,policy=h.input_authorities(positive_release_fixture(),freeze)
        self.assertEqual(len(inp['actual_blind_job']['clips']),6)
        self.assertEqual(policy,freeze['host_code'][h.retained_host.NEW+'asr18_metadata.py']['sha256'])

    def test_observer_complete_stdout_even_local_write_fails_and_unknown_remains_null(self):
        out=io.StringIO();o=obs.Observation('host');o.receipt_dir=Path('/nonexistent')
        with redirect_stdout(out),patch.object(obs.c,'write_once',side_effect=OSError(5,SECRET)):
            o.stage('capacity_proof');o.failure(ValueError('private host path prohibited'));o.finish(False)
        rows=[json.loads(x) for x in out.getvalue().splitlines()]
        final=rows[-1];self.assertEqual(final['persistence']['status'],'local_record_failed')
        self.assertIsNone(final['record']['consumed_asr_attempts']);self.assertNotIn(SECRET,out.getvalue())
        self.assertEqual(final['record']['errors'][0]['detail']['validation_code'],'metadata_private_path')
        self.assertEqual(final['record_sha256'],pc.digest(pc.canonical(final['record'])))

    def test_observer_serializer_failure_fixed_literal_and_no_primary_text(self):
        out=io.StringIO();o=obs.Observation('publisher')
        with redirect_stdout(out):
            o.failure(ValueError(SECRET))
            with patch.object(obs.c,'canonical',side_effect=MemoryError):self.assertFalse(o.finish(False))
        self.assertEqual(json.loads(out.getvalue())['schema'],'k1-prepare-diagnostic-fallback-v1')
        self.assertNotIn(SECRET,out.getvalue())


if __name__=='__main__':unittest.main()
