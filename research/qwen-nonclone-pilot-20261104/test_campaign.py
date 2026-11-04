"""Invented fixtures only: no Torch, network, Docker, weights or speech."""
from pathlib import Path
from contextlib import nullcontext
import copy
import hashlib
import json
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign
import worker

class Samples(list):
    ndim=1
    def tobytes(self): return struct.pack('<%df' % len(self), *self)

class Sequence:
    def __getitem__(self,key): return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return [9]

class Model:
    def __init__(self,fail=None):
        owner=lambda: NS(training=False, parameters=lambda:[NS(device='cpu',dtype='torch.float32')])
        self.model=owner();self.model.tts_model_type='custom_voice';self.model.tts_model_size='0b6'
        self.model.speech_tokenizer=NS(model=owner())
        self.model.talker=NS(generate=lambda *a,**k:NS(sequences=Sequence()))
        self.model.config=NS(talker_config=NS(codec_eos_token_id=9))
        self.calls=[];self.fail=fail
    def get_supported_speakers(self): return ['Vivian']
    def generate_custom_voice(self,**kwargs):
        self.calls.append(kwargs)
        if len(self.calls)==self.fail: raise RuntimeError('invented failure')
        self.model.talker.generate()
        return [Samples([.1]*2400)],24000

TORCH=NS(manual_seed=lambda n:None,inference_mode=nullcontext)
NP=NS(random=NS(seed=lambda n:None),asarray=lambda x,dtype:x)
SF=NS(write=lambda path,data,rate,subtype:Path(path).write_bytes(b'invented-float-fixture'+data.tobytes()))

class Tests(unittest.TestCase):
    def event(self):
        env={'GITHUB_REPOSITORY':'jiying2007/kws-pipeline','GITHUB_REF':'refs/heads/'+campaign.BRANCH,
             'GITHUB_EVENT_NAME':'push','GITHUB_RUN_NUMBER':'1','GITHUB_RUN_ATTEMPT':'1',
             'GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123'}
        event={'created':True,'deleted':False,'after':'a'*40,
               'repository':{'full_name':'jiying2007/kws-pipeline','private':False}}
        return env,event
    def test_exact_four_no_cloning(self):
        rows=campaign.plan()
        self.assertEqual(len(rows),4)
        self.assertEqual(len({r['cell_id'] for r in rows}),4)
        self.assertEqual({r['speaker'] for r in rows},{'Vivian'})
        self.assertEqual([r['intended_keyword_id'] for r in rows],[1,2,None,None])
        self.assertTrue(all(r['training_admitted'] is False for r in rows))
        self.assertFalse(any('reference' in k for r in rows for k in r))
    def test_first_push(self):
        campaign.verify_event(*self.event())
    def test_old_namespace_and_rerun_rejected(self):
        for key,value in [('GITHUB_REF','refs/heads/research/tts-quality-probe-20261014-v1'),
                          ('GITHUB_RUN_NUMBER','2'),('GITHUB_RUN_ATTEMPT','2'),
                          ('GITHUB_EVENT_NAME','pull_request'),('GITHUB_REPOSITORY','other/repo')]:
            env,event=self.event();env[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): campaign.verify_event(env,event)
    def test_private_or_recreated_push_rejected(self):
        for key,value in [('created',False),('deleted',True),('after','b'*40)]:
            env,event=self.event();event[key]=value
            with self.assertRaises(ValueError):campaign.verify_event(env,event)
        env,event=self.event();event['repository']['private']=True
        with self.assertRaises(ValueError):campaign.verify_event(env,event)
    def generate(self,model,path):
        with patch.object(worker,'scope',lambda:None),patch.object(worker.signal,'alarm'):
            return worker.generate(model,TORCH,NP,SF,path)
    def test_four_calls_and_restored_talker(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();original=model.model.talker.generate
            result=self.generate(model,Path(d))
            self.assertEqual(len(model.calls),4)
            self.assertEqual(len(result['outcomes']),4)
            self.assertIs(model.model.talker.generate,original)
            self.assertTrue(all(r['status']=='generated' for r in result['outcomes']))
            self.assertEqual([k['text'] for k in model.calls],[r['text'] for r in campaign.plan()])
            self.assertFalse(result['training_admitted'])
    def test_failure_retains_attempt_and_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);model=Model(fail=2)
            with self.assertRaises(RuntimeError):self.generate(model,path)
            saved=json.loads((path/'generation.json').read_text())
            self.assertEqual(len(saved['planned']),4)
            self.assertEqual(len(saved['outcomes']),2)
            self.assertEqual(saved['status'],'stopped_no_retry')
            self.assertEqual(len(list(path.glob('*.claim'))),2)
            with self.assertRaisesRegex(ValueError,'ATTEMPT_ALREADY_STARTED'):self.generate(model,path)
            self.assertEqual(len(model.calls),2)
    def test_no_reference_model(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();model.model.tts_model_type='base'
            with self.assertRaisesRegex(ValueError,'EXACT_CUSTOMVOICE'):self.generate(model,Path(d))
            self.assertEqual(model.calls,[])
    def test_non_cpu_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();model.model.parameters=lambda:[NS(device='cuda:0',dtype='torch.float32')]
            with self.assertRaisesRegex(ValueError,'CPU_FP32'):self.generate(model,Path(d))
    def test_public_export_excludes_runtime_logs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'output').mkdir();(root/'runtime').mkdir()
            (root/'host-receipt.json').write_text('{}');(root/'output'/'x.wav').write_bytes(b'fake')
            (root/'output'/'stderr.log').write_text('not public')
            (root/'runtime'/'weights.bin').write_bytes(b'not public')
            campaign.export_public(root)
            names={p.relative_to(root/'artifact').as_posix() for p in (root/'artifact').rglob('*') if p.is_file()}
            self.assertEqual(names,{'host-receipt.json','output/x.wav','artifact-index.json'})
    def test_public_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'output').mkdir();(root/'secret').write_bytes(b'not public')
            (root/'output'/'alias.wav').symlink_to(root/'secret')
            with self.assertRaisesRegex(ValueError,'PUBLIC_FILE_TYPE'):campaign.export_public(root)
    def test_forced_interruption_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cell=campaign.plan()[0]
            (out/(cell['cell_id']+'.claim')).write_text(cell['cell_id'])
            (out/'generation.json').write_text(json.dumps({'campaign':campaign.CAMPAIGN,
                'planned':campaign.plan(),'status':'started',
                'outcomes':[dict(cell,status='attempt_started')]}))
            result=campaign.reconcile_terminal(out,'stopped_no_retry')
            self.assertEqual([r['status'] for r in result['outcomes']],['failed_no_retry']+['not_run']*3)
            self.assertEqual(result['generated_files_verified'],0)
    def test_empty_claim_is_consumed_not_retryable(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ident=campaign.plan()[0]['cell_id']
            (out/(ident+'.claim')).write_bytes(b'')
            result=campaign.reconcile_terminal(out,'stopped_no_retry')
            self.assertEqual(result['outcomes'][0]['status'],'failed_no_retry')
            self.assertEqual(result['claimed_attempts'],1)
    def test_success_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.generate(Model(),out)
            result=campaign.reconcile_terminal(out,'generated_pending_independent_screen')
            self.assertEqual(result['generated_files_verified'],4)
            self.assertTrue(result['all_four_technically_valid'])
            self.assertFalse(result['training_admitted'])
    def test_changed_generated_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.generate(Model(),out)
            next(out.glob('*.wav')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'GENERATED_WAV_CHANGED'):
                campaign.reconcile_terminal(out,'generated_pending_independent_screen')
    def test_setup_failure_all_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            result=campaign.reconcile_terminal(Path(d),'stopped_no_retry')
            self.assertTrue(all(r['status']=='not_run' for r in result['outcomes']))
            self.assertFalse(result['runtime_verification_summary_present'])
    def test_source_freeze(self):
        lock=campaign.verify_sources(campaign.ROOT)
        self.assertIn('research/qwen6_tts/runtime-lock.json',lock['files'])
        self.assertIn(campaign.WORKFLOW,lock['files'])
    def test_staging_includes_exact_model_lock(self):
        lock=campaign.verify_sources(campaign.ROOT)
        with tempfile.TemporaryDirectory() as d:
            code=Path(d)/'code';campaign.staged_sources(campaign.ROOT,code,lock)
            old=code/'research/tts-quality-pilot-v1'
            self.assertEqual((old/'qwen-model-lock.json').read_bytes(),(old/'qwen06-model-lock.json').read_bytes())
            self.assertFalse((code/'research/tts-quality-pilot-v1/worker.py').exists())
    def test_retained_host_import_is_source_only(self):
        sys.path.insert(0,str(campaign.ROOT/'research/source-screen32-v1'))
        import hosted_run
        self.assertTrue(callable(hosted_run.container_args))
        self.assertNotIn('torch',sys.modules)
    def test_workflow_budget_and_no_pr_model_trigger(self):
        text=(campaign.ROOT/campaign.WORKFLOW).read_text()
        self.assertIn('timeout-minutes: 45',text)
        self.assertIn('github.run_number == 1',text)
        self.assertNotIn('pull_request:',text)
        self.assertNotIn('workflow_dispatch:',text)
        self.assertNotIn('contents: write',text)

if __name__=='__main__':unittest.main()
