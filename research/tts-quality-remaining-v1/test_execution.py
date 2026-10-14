"""Offline integration tests; never call API, Docker, or inference runtimes."""
from pathlib import Path
import importlib.util,json,os,shutil,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('TQP_TEST_SOURCE_ROOT',str(HERE.parents[1])))
sys.path.insert(0,str(HERE));sys.path.insert(0,str(SOURCE/'research/source-screen32-v1'))
import remaining as r
import admission
import run_remaining as host

class Tests(unittest.TestCase):
    def test_metadata_rejects_sibling_lane(self):
        b=r.document(HERE/'binding-qwen06.json')
        run=dict(id=b['probe_run_id'],head_sha=b['probe_source_sha'],path=b['workflow_path'],event='push',run_number=1,run_attempt=1,status='completed',conclusion='success',repository={'full_name':admission.REPO})
        job=dict(id=b['probe_job_id'],run_id=b['probe_run_id'],head_sha=b['probe_source_sha'],run_attempt=1,name=b['probe_job_name'],status='completed',conclusion='success')
        artifact=dict(id=b['artifact_id'],name=b['artifact_name'],expired=False,digest='sha256:'+b['artifact_archive_sha256'],workflow_run={'id':b['probe_run_id'],'head_sha':b['probe_source_sha']})
        admission.verify_metadata(b,run,job,artifact)
        job['name']='probe (qwen17)'
        with self.assertRaises(ValueError):admission.verify_metadata(b,run,job,artifact)
    def test_partial_matrix_failure_allowed_only_for_successful_job(self):
        b=r.document(HERE/'binding-qwen06.json');b['probe_run_conclusion']='failure'
        r.binding_valid(b,'qwen06')
        b['probe_job_conclusion']='failure'
        with self.assertRaises(ValueError):r.binding_valid(b,'qwen06')
    def test_identity_recreated_or_wrong_source_rejected(self):
        env=dict(GITHUB_REPOSITORY=admission.REPO,GITHUB_REF='refs/heads/research/tts-quality-remaining-20261014-v1',GITHUB_EVENT_NAME='push',GITHUB_RUN_NUMBER='1',GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40)
        event=dict(created=True,deleted=False,after='a'*40,repository=dict(full_name=admission.REPO,private=False))
        host.identity(env,event)
        env['GITHUB_RUN_NUMBER']='2'
        with self.assertRaises(ValueError):host.identity(env,event)
    def test_workflow_token_only_admission(self):
        text=(HERE/'workflow.yml').read_text()
        self.assertEqual(text.count('GH_TOKEN:'),1)
        self.assertNotIn('GH_TOKEN', (HERE/'run_remaining.py').read_text())
        self.assertNotIn('GH_TOKEN', (HERE/'worker.py').read_text())
        self.assertIn('timeout-minutes: 110',text)
        self.assertIn('max-parallel: 1',text)
    def test_host_stages_immutable_probe_and_no_retry_on_container_failure(self):
        b=r.document(HERE/'binding-qwen06.json')
        archive=SOURCE/'research/effect-data-model-v2/evidence/probe-run-38554125374/qwen06.zip'
        self.assertTrue(archive.is_file(),'test requires retained public probe archive')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'repo';root.mkdir()
            for name in ('tts-quality-pilot-v1','source-screen32-v1','qwen6_tts','qwen6_asr'):
                shutil.copytree(SOURCE/'research'/name,root/'research'/name)
            workflow=Path('.github/workflows/tts-quality-probe-20261014-v1.yml')
            (root/workflow.parent).mkdir(parents=True);shutil.copyfile(SOURCE/workflow,root/workflow)
            here=root/'research/tts-quality-remaining-v1';shutil.copytree(HERE,here)
            work=root/'tqremaining-qwen06';(work/'artifact').mkdir(parents=True)
            (work/'artifact/admission-status.json').write_text(json.dumps(dict(status='admitted',engine='qwen06',archive_sha256=b['artifact_archive_sha256'])))
            inputs=work/'inputs';(inputs/'incoming').mkdir(parents=True);shutil.copyfile(archive,inputs/'probe.zip')
            for n,body in r.archive_members(archive,r.filenames('qwen06','tqp-qwen06-00'),None).items():(inputs/'incoming'/n).write_bytes(body)
            event=root/'event.json';event.write_text(json.dumps(dict(created=True,deleted=False,after='a'*40,repository=dict(full_name=admission.REPO,private=False))))
            env=dict(GITHUB_REPOSITORY=admission.REPO,GITHUB_REF='refs/heads/research/tts-quality-remaining-20261014-v1',GITHUB_EVENT_NAME='push',GITHUB_RUN_NUMBER='1',GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40,GITHUB_RUN_ID='123',GITHUB_EVENT_PATH=str(event))
            calls=[]
            def supervise(args,name,output,stage,deadline):
                calls.append((args,stage))
                if stage=='tts':raise RuntimeError('fixture container failure; no model called')
                return {'status':'complete'}
            with patch.object(host,'ROOT',root),patch.object(host,'HERE',here),patch.dict(os.environ,env),patch.object(sys,'argv',['run_remaining.py','qwen06']),patch.object(host.retained,'host_capacity',return_value={}),patch.object(host.retained,'pull_image'),patch.object(host.retained,'supervise_container',side_effect=supervise):
                with self.assertRaises(RuntimeError):host.main()
            self.assertEqual([s for _,s in calls],['setup','tts'])
            self.assertNotIn('GH_TOKEN',' '.join(calls[-1][0]))
            r.verify(b,'qwen06',work/'code/inputs/incoming',work/'code/probe-source',work/'code/inputs/probe.zip',here/'source-manifest.json')
            self.assertEqual(r.document(work/'artifact/host-receipt.json')['status'],'stopped_no_retry')
            self.assertEqual(r.document(work/'artifact/remaining-host-resource-status.json')['worker_measurements'],'unknown')
            self.assertEqual(len(r.document(work/'output/qwen06-receipt.json')['outcomes']),1)
    def test_metadata_redirect_never_replays_bearer(self):
        import urllib.error
        from unittest.mock import Mock
        opener=Mock();opener.open.side_effect=urllib.error.HTTPError('https://api.github.com',302,'redirect',{},None)
        with patch.object(admission.urllib.request,'build_opener',return_value=opener) as build:
            with self.assertRaises(urllib.error.HTTPError):admission.api('/actions/runs/123','fixture-token')
        self.assertEqual(opener.open.call_count,1)
        self.assertIsInstance(build.call_args.args[0],admission.NoRedirect)
        self.assertIsNone(admission.NoRedirect().redirect_request(None,None,None,None,None,None))
if __name__=='__main__':unittest.main()
