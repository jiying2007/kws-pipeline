"""Fixed source/Git/API/diagnostic integration. Stdlib; never ML or networking."""
from contextlib import redirect_stdout
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
            base+'/actions/workflows/8/runs?branch='+h.urllib.parse.quote(h.SOURCE_REF,safe='')+'&per_page=100&page=1':{'total_count':1,'workflow_runs':[current]},
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
    def test_actual_controls_allow_only_prepare_and_all_sealed_asr_remains_disabled(self):
        release=pc.release_first()
        self.assertEqual(release['max_new_asr_calls'],0);self.assertFalse(release['audio_forward_allowed'])
        self.assertEqual(release['prior_prepare_history_sha256'],pc.digest(pc.canonical(h.prior_history_identity())))
        for name,keys in (('execution-release.json',('enabled','independent_source_review','root_execution_release')),
                          ('publication-release.json',('enabled','independent_source_review','root_publication_release'))):
            value=h.read_json(h.ASR/name)
            self.assertTrue(all(value[k] is False for k in keys))
        self.assertEqual([r['run_id'] for r in h.prior_history_identity()['sealed_asr_runs']],[38207026177,38219171568])
        self.assertTrue(all(r['actual_consumed_attempts'] is None and r['reusable_slots']==0 for r in h.prior_history_identity()['sealed_asr_runs']))

    def test_real_checked_in_release_api_preflight_positive_and_history_rejections(self):
        release=pc.release_first();_,_,run=context();api=API(run)
        result=h.api_preflight(api,run,release)
        self.assertEqual(len(api.calls),6);self.assertEqual(result['numeric_job_id'],999)
        key=next(k for k in api.rows if '/runs?branch=' in k)
        for change in ('second','old','attempt2'):
            bad=API(run)
            if change=='second':bad.rows[key]['total_count']=2;bad.rows[key]['workflow_runs']*=2
            elif change=='old':bad.rows[key]['workflow_runs'][0]['id']=38219171568
            else:bad.rows[key]['workflow_runs'][0]['run_attempt']=2
            with self.subTest(change=change),self.assertRaises(ValueError):h.api_preflight(bad,run,release)
        wrong=dict(release,prior_prepare_history_sha256=pc.digest(pc.canonical(h.prior_history_identity())+b'\n'))
        with self.assertRaises(ValueError):h.api_preflight(API(run),run,wrong)

    def test_wrong_ref_old_run_and_attempt_reject_before_external_call(self):
        env,event,_=context()
        for changed in ({'GITHUB_REF':'refs/heads/research/k1-punctuation6-asr18-r1'}, {'GITHUB_RUN_ATTEMPT':'2'},
                        {'GITHUB_RUN_ID':'38219171568'},{'GITHUB_RUN_ID':'38207026177'},{'GITHUB_JOB':'asr18'}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):h.event_identity({**env,**changed},event)

    def test_true_controls_actual_isolated_cli_rejects_before_runtime(self):
        for filename in ('prepare_run.py','diagnostic_publisher.py'):
            process=subprocess.run([sys.executable,'-I','-B',str(HERE/filename)],
                env={'PATH':'/usr/bin:/bin','GITHUB_TOKEN':SECRET,'RUNNER_TEMP':SECRET},capture_output=True,text=True,timeout=10)
            self.assertNotEqual(process.returncode,0);self.assertNotIn('Traceback',process.stderr)
            self.assertNotIn('ModuleNotFoundError',process.stderr);self.assertNotIn(SECRET,process.stdout+process.stderr)
            rows=[json.loads(x) for x in process.stdout.splitlines()]
            self.assertTrue(any('diagnostic' in row.get('schema','') for row in rows))
            self.assertFalse(Path(SECRET).exists())

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
            env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null',
                'GIT_AUTHOR_NAME':'Offline fixture','GIT_AUTHOR_EMAIL':'fixture@example.invalid','GIT_COMMITTER_NAME':'Offline fixture',
                'GIT_COMMITTER_EMAIL':'fixture@example.invalid','GIT_AUTHOR_DATE':'2026-10-12T00:00:00Z','GIT_COMMITTER_DATE':'2026-10-12T00:00:00Z'}
            def git(*args):return subprocess.run(['git','-C',str(root),*args],env=env,capture_output=True,check=True,timeout=10).stdout
            git('init','--quiet');git('add','.');git('commit','--quiet','-m','offline fixed source fixture')
            head=git('rev-parse','HEAD').decode().strip();_,_,run=context(head)
            with patch.object(h,'ROOT',root),patch.object(h,'HERE',fixed),patch.object(h,'git_read',side_effect=git):
                actual,identity,tree=h.source_authority(run,pc.release_first())
                self.assertEqual(actual,freeze);self.assertEqual(identity,pc.digest(pc.canonical(freeze)))
                self.assertEqual(tree,git('rev-parse','HEAD^{tree}').decode().strip())
                (fixed/'prepare_run.py').write_bytes(b'changed')
                with self.assertRaises(ValueError):h.source_authority(run,pc.release_first())
        inp,authority,policy=h.input_authorities(pc.release_first(),freeze)
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
