"""Host-only reconstruction tests. No network/model/setup commands."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from execution_common import claim, document, read, exact_equal
import admission
import launcher

class HostTests(unittest.TestCase):
    def test_report_rejects_alias_attempts_and_unknown_identity(self):
        import campaign
        from worker import frozen_plan
        plan=frozen_plan();cell=plan['slots'][0]
        for attempt in (True,1.0,'1'):
            with self.assertRaises(ValueError):
                campaign.report(plan,[dict(cell,status='completed',attempts=attempt,metrics={'technically_valid':True})],[],{},None)
        with self.assertRaises(ValueError):
            campaign.report(plan,[dict(cell,slot_id='unknown',status='NOT_RUN',attempts=0)],[],{},None)
        with self.assertRaises(ValueError):
            campaign.report(plan,[],[],{'qwen_asr':[dict(opaque_id='unbound',attempts=1,status='success')]},None)

    def test_exact_types(self):
        self.assertFalse(exact_equal({'x':[True]}, {'x':[1]}))
        self.assertFalse(exact_equal({'x':1.0}, {'x':1}))
        self.assertTrue(exact_equal({'x':[1]}, {'x':[1]}))
    def test_json_rejects_duplicate_and_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x'
            for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', ''):
                p.write_text(raw)
                with self.assertRaises(ValueError):document(p)
    def test_claim_never_overwrites_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'claim';p.touch()
            with self.assertRaises(FileExistsError):claim(p, {'attempt':1})
            self.assertEqual(p.read_bytes(),b'')
    def test_claim_never_overwrites_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'claim';claim(p,{'attempt':1})
            with self.assertRaises(FileExistsError):claim(p,{'attempt':1})
    def test_read_rejects_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(b'x');q=Path(tmp)/'alias';q.symlink_to(p)
            with self.assertRaises(ValueError):read(q)
    def test_global_clock_original_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'admission-clock.json'
            base=dict(schema='kw1-admission-clock-v1',boot_id='boot',monotonic=100.0,wall_time=1000.0,budget_seconds=18000)
            claim(p,base)
            with patch.object(launcher,'boot_id',return_value='boot'),patch.object(launcher.time,'monotonic',return_value=200.0),patch.object(launcher.time,'time',return_value=1300.0):
                self.assertEqual(launcher.remaining_deadline(tmp),17900.0)
            with patch.object(launcher,'boot_id',return_value='newboot'):
                with self.assertRaises(ValueError):launcher.remaining_deadline(tmp)
            with patch.object(launcher,'boot_id',return_value='boot'),patch.object(launcher.time,'monotonic',return_value=18101.0),patch.object(launcher.time,'time',return_value=19001.0):
                with self.assertRaises(ValueError):launcher.remaining_deadline(tmp)
    def test_admission_rejects_unenabled_and_boolean_budget(self):
        with self.assertRaises(ValueError):admission.validate({}, {}, {}, {}, {}, {}, '.')
        for tts in (True,16.0,'16'):
            config=dict(schema='kw1-execution-admission-v1',enabled=True,max_tts_calls=tts,max_asr_calls=48,retry=False)
            with self.assertRaises(ValueError):admission.validate(config, {}, {}, {}, {}, {}, '.')
    def test_server_admission_exact_identity_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'research/kw1-preset-supplement-v1';d.mkdir(parents=True)
            (d/'execution-admission.json').write_text('{}')
            head='a'*40
            config=dict(schema='kw1-execution-admission-v1',enabled=True,max_tts_calls=16,max_asr_calls=48,retry=False,source_files={})
            env=dict(GITHUB_REPOSITORY=admission.REPOSITORY,GITHUB_REF='refs/heads/'+admission.BRANCH,GITHUB_EVENT_NAME='push',GITHUB_RUN_ATTEMPT='1',GITHUB_RUN_NUMBER='1',GITHUB_SHA=head,GITHUB_RUN_ID='9')
            event=dict(created=True,deleted=False,after=head,ref=env['GITHUB_REF'],repository=dict(full_name=admission.REPOSITORY,private=False))
            run=dict(id=9,workflow_id=10,run_number=1,run_attempt=1,head_sha=head,head_branch=admission.BRANCH,event='push',repository=event['repository'])
            workflow=dict(id=10,path=admission.WORKFLOW,state='active')
            history=dict(total_count=1,workflow_runs=[dict(run)])
            with patch.object(admission,'required_sources',return_value=set()):
                self.assertTrue(admission.validate(config,env,event,run,workflow,history,root)['admitted'])
                for field in ('id','workflow_id','run_number','run_attempt'):
                    bad=copy.deepcopy(run);bad[field]=True
                    with self.assertRaises(ValueError):admission.validate(config,env,event,bad,workflow,history,root)
                for field in ('created','deleted'):
                    bad=copy.deepcopy(event);bad[field]=1 if field=='created' else 0
                    with self.assertRaises(ValueError):admission.validate(config,env,bad,run,workflow,history,root)
                bad=copy.deepcopy(history);bad['total_count']=2
                with self.assertRaises(ValueError):admission.validate(config,env,event,run,workflow,bad,root)
                bad=copy.deepcopy(workflow);bad['path']='other.yml'
                with self.assertRaises(ValueError):admission.validate(config,env,event,run,bad,history,root)

    def test_finalize_cannot_construct_backend(self):
        import ast
        tree=ast.parse(Path(launcher.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not args.finalize')
        self.assertIn('DockerBackend',ast.unparse(block))
        self.assertNotIn('DockerBackend',ast.unparse(ast.Module(body=[n for n in main.body if n is not block],type_ignores=[])))

if __name__=='__main__':unittest.main()
