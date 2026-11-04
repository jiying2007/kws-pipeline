"""Invented-fixture source tests only. Never invokes Docker or ML."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import capacity
spec=importlib.util.spec_from_file_location('capacity_host',HERE/'host.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def env(root):
    value=dict(GITHUB_REPOSITORY='jiying2007/kws-pipeline',GITHUB_REF='refs/heads/research/'+runner.CAMPAIGN,
        GITHUB_EVENT_NAME='push',GITHUB_RUN_NUMBER='1',GITHUB_RUN_ATTEMPT='1',GITHUB_SHA='a'*40,
        GITHUB_RUN_ID='123',GITHUB_EVENT_PATH=str(root/'event.json'),GITHUB_WORKSPACE=str(root),
        GITHUB_ACTIONS='true',RUNNER_ENVIRONMENT='github-hosted',RUNNER_OS='Linux',RUNNER_ARCH='X64',
        GITHUB_WORKFLOW_SHA='a'*40,GITHUB_WORKFLOW_REF='jiying2007/kws-pipeline/.github/workflows/'+runner.CAMPAIGN+'.yml@refs/heads/research/'+runner.CAMPAIGN)
    (root/'event.json').write_text(json.dumps(dict(created=True,deleted=False,after='a'*40,before='0'*40,ref=value['GITHUB_REF'],
        repository=dict(private=False,full_name='jiying2007/kws-pipeline'))))
    return value

def observed(free):return dict(total_bytes=100*1024**3,used_bytes=100*1024**3-free,
    free_bytes=free,device_major=1,device_minor=2)

class CapacityTests(unittest.TestCase):
    def test_exact_new_identity(self):
        with tempfile.TemporaryDirectory() as d:
            values=env(Path(d));values['GITHUB_WORKSPACE']=str(runner.ROOT)
            runner.admission(values)
            for key,bad in [('GITHUB_REF','refs/heads/research/native-prepare-20261104-v1'),
                            ('GITHUB_RUN_ATTEMPT','2'),('GITHUB_RUN_NUMBER','2'),('GITHUB_EVENT_NAME','workflow_dispatch'),
                            ('GITHUB_WORKFLOW_SHA','b'*40),('RUNNER_ENVIRONMENT','self-hosted'),
                            ('GITHUB_WORKFLOW_REF','other'),('GITHUB_WORKSPACE','/wrong')]:
                changed=dict(values);changed[key]=bad
                with self.assertRaises(ValueError):runner.admission(changed)
    def test_zero_before_and_exact_event_ref(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root);values['GITHUB_WORKSPACE']=str(runner.ROOT)
            event=json.loads((root/'event.json').read_text())
            for key,bad in [('before','b'*40),('ref','refs/heads/other')]:
                changed=dict(event);changed[key]=bad;(root/'event.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):runner.admission(values)
    def test_oversized_event_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root);(root/'event.json').write_bytes(b' '*(1024**2+1))
            with self.assertRaisesRegex(ValueError,'bounded event'):runner.admission(values)
    def test_numeric_same_job_snapshot_and_fixed_stages(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root);values['PRIVATE_SECRET']='never-export'
            with patch.object(capacity,'disk',return_value=observed(30*1024**3)):
                c=capacity.Capacity(root,root,values,0);c.record('initial')
                with self.assertRaises(ValueError):c.record('initial')
                with self.assertRaises(ValueError):c.record('unbounded-user-path')
            raw=(root/'aggregate-only/capacity.json').read_text();data=json.loads(raw)
            self.assertNotIn('never-export',raw);self.assertNotIn(d,raw)
            self.assertEqual(data['required_initial_free_bytes'],25*1024**3)
            self.assertEqual(data['observations'][0]['workspace']['free_bytes'],30*1024**3)
            self.assertEqual(data['source_head'],'a'*40)
    def test_checkpoint_order(self):
        with tempfile.TemporaryDirectory() as d,patch.object(capacity,'disk',return_value=observed(30*1024**3)):
            root=Path(d);c=capacity.Capacity(root,root,env(root),0);c.record('initial');c.record('after_image')
            with self.assertRaises(ValueError):c.record('before_image')
    def test_low_disk_snapshot_persists_before_failure_no_docker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root)
            with patch.object(runner,'ROOT',root),patch.object(runner,'verify'),\
                 patch.object(capacity,'disk',return_value=observed(24*1024**3)),\
                 patch.object(runner.host,'stage',side_effect=AssertionError('must stop before staging')):
                with self.assertRaisesRegex(ValueError,'disk reserve'):runner.run(values)
            work=root/('native-private-'+runner.CAMPAIGN)
            self.assertEqual(json.loads((work/'aggregate-only/capacity.json').read_text())['observations'][0]['workspace']['free_bytes'],24*1024**3)
            status=json.loads((work/'aggregate-only/status.json').read_text())
            self.assertEqual(status['status'],'stopped_no_retry');self.assertEqual({r['status'] for r in status['models']},{'not_run'})
    def test_optional_target_failure_retains_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def measure(path):
                if Path(path)==root:return observed(30*1024**3)
                raise PermissionError(13,'private path must not be exported')
            with patch.object(capacity,'disk',side_effect=measure):
                c=capacity.Capacity(root,root,env(root),0);row=c.record('initial')
            self.assertEqual(row['workspace']['free_bytes'],30*1024**3)
            self.assertEqual(row['root'],{'status':'unavailable','errno':13})
            self.assertNotIn('private path',(root/'aggregate-only/capacity.json').read_text())
    def test_after_diagnostic_failure_preserves_stage_exception(self):
        class FailedAfter:
            def record(self,stage):
                if stage=='after_image':raise OSError('disk write failed')
        with self.assertRaisesRegex(RuntimeError,'original failure'):
            with capacity.observed(FailedAfter(),'before_image','after_image'):
                raise RuntimeError('original failure')
    def test_after_diagnostic_failure_stops_successful_stage(self):
        class FailedAfter:
            def record(self,stage):
                if stage=='after_image':raise OSError('disk write failed')
        with self.assertRaises(OSError):
            with capacity.observed(FailedAfter(),'before_image','after_image'):pass
    def test_no_overwrite_existing_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root);(root/('native-private-'+runner.CAMPAIGN)).mkdir()
            with patch.object(runner,'ROOT',root),patch.object(runner,'verify'):
                with self.assertRaises(FileExistsError):runner.run(values)
    def test_launch_never_deletes_prior_diagnostic(self):
        class Boundary:
            def host_capacity(self):pass
            def container_args(self,*args):return ['docker','image','command']
            def supervise_container(self,*args):raise AssertionError('existing diagnostic must stop')
        with tempfile.TemporaryDirectory() as d,patch.object(runner.time,'monotonic',return_value=0):
            root=Path(d);(root/'container-tts.json').write_text('keep')
            with self.assertRaises(ValueError):runner.launch(Boundary(),None,None,root/'output','image','test',[],1800)
            self.assertEqual((root/'container-tts.json').read_text(),'keep')
    def test_all_checkpoints_and_three_families(self):
        self.orchestration(False)
    def test_setup_failure_stops_all_models_and_records_after(self):
        self.orchestration(True)
    def test_exact_25gib_threshold_still_permitted(self):
        self.orchestration(False,25)
    def orchestration(self,fail,free_gib=30):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);values=env(root);code=root/'staged';code.mkdir()
            template=code/'research/native-asr-integration-20261104/sense-review-template.json';template.parent.mkdir(parents=True)
            template.write_text(json.dumps(dict(schema='reviewed-sense-runtime-capture-template-v1',dependencies={},sources=[])))
            image=code/'research/source-screen32-v1/container-lock.json';image.parent.mkdir(parents=True);image.write_text('{"image":"image"}')
            calls=[]
            boundary=NS(host_capacity=lambda:None,pull_image=lambda *args:None)
            def launch(*args):
                output=args[3];name=args[5];calls.append(name);output.mkdir(parents=True)
                if fail:raise RuntimeError('invented setup failure')
                if 'setup' not in name:
                    family=name[len(runner.CAMPAIGN)+1:]
                    (output/'zero-forward-proof.json').write_text(json.dumps(dict(family=family,status='prepared_zero_forward',
                        candidate_count=0,audio_inputs=0,observed_guarded_entrypoint_attempts=0,
                        native_config_sha256='b'*64,source_lock_sha256='c'*64,asset_lock_sha256='d'*64)))
                    (output/'identity.json').write_text('{}')
            with patch.object(runner,'ROOT',root),patch.object(runner,'verify'),\
                 patch.object(capacity,'disk',return_value=observed(free_gib*1024**3)),\
                 patch.object(runner.host,'stage',return_value=code),patch.object(runner,'launch',side_effect=launch),\
                 patch.object(runner.host,'freeze_policy',return_value=({},'e'*64)),patch.dict(sys.modules,boundary=boundary):
                if fail:
                    with self.assertRaises(RuntimeError):runner.run(values)
                else:runner.run(values)
            work=root/('native-private-'+runner.CAMPAIGN)
            stages=[r['stage'] for r in json.loads((work/'aggregate-only/capacity.json').read_text())['observations']]
            self.assertEqual(stages,list(capacity.STAGES[:5] if fail else capacity.STAGES))
            self.assertEqual(len(calls),1 if fail else 5)
    def test_budget_and_unchanged_worker(self):
        self.assertEqual(runner.SECONDS,1800)
        source=(HERE/'host.py').read_text()
        self.assertIn('/code/research/native-asr-integration-20261104/prepare_worker.py',source)
        self.assertNotIn('.unlink(',source);self.assertNotIn('rmtree(',source)

if __name__=='__main__':unittest.main()
