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
    def get_supported_speakers(self): return list(campaign.VOICES)
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
    def test_exact_64_no_cloning(self):
        rows=campaign.plan()
        self.assertEqual(len(rows),64)
        self.assertEqual(len({r['cell_id'] for r in rows}),64)
        self.assertEqual({r['speaker'] for r in rows},set(campaign.VOICES))
        self.assertEqual([r['intended_keyword_id'] for r in rows],([1]*2+[2]*2+[None]*12)*4)
        self.assertEqual([r['seed'] for r in rows],list(range(2026110400,2026110464)))
        self.assertEqual(len({(r['speaker'],r['text'],r['repeat']) for r in rows}),64)
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
    def generate(self,model,path,lane=0):
        with patch.object(worker,'scope',lambda:None),patch.object(worker.signal,'alarm'):
            return worker.generate(model,TORCH,NP,SF,path,lane)
    def test_16_calls_and_restored_talker(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();original=model.model.talker.generate
            result=self.generate(model,Path(d))
            self.assertEqual(len(model.calls),16)
            self.assertEqual(len(result['outcomes']),16)
            self.assertIs(model.model.talker.generate,original)
            self.assertTrue(all(r['status']=='generated' for r in result['outcomes']))
            self.assertEqual([k['text'] for k in model.calls],[r['text'] for r in campaign.lane_plan(0)])
            self.assertFalse(result['training_admitted'])
    def test_failure_retains_attempt_and_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);model=Model(fail=2)
            with self.assertRaises(RuntimeError):self.generate(model,path)
            saved=json.loads((path/'generation.json').read_text())
            self.assertEqual(len(saved['planned']),16)
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
    def test_public_export_excludes_runtime_logs_and_unknown_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'lane-0/output';out.mkdir(parents=True);(root/'runtime').mkdir()
            ident=campaign.plan()[0]['cell_id']
            (root/'host-receipt.json').write_text('{}');(out/(ident+'.raw-float.wav')).write_bytes(b'fake')
            (out/'stderr.log').write_text('not public');(out/'private.json').write_text('not public')
            (root/'runtime/weights.bin').write_bytes(b'not public')
            campaign.export_public(root)
            names={p.relative_to(root/'artifact').as_posix() for p in (root/'artifact').rglob('*') if p.is_file()}
            self.assertEqual(names,{'host-receipt.json','lane-0/output/'+ident+'.raw-float.wav','artifact-index.json'})
    def test_public_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'secret').write_bytes(b'not public')
            (root/'host-receipt.json').symlink_to(root/'secret')
            with self.assertRaisesRegex(ValueError,'PUBLIC_FILE_TYPE'):campaign.export_public(root)
    def test_forced_interruption_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cell=campaign.plan()[0]
            (out/(cell['cell_id']+'.claim')).write_text(cell['cell_id'])
            (out/'generation.json').write_text(json.dumps({'campaign':campaign.CAMPAIGN,
                'planned':campaign.lane_plan(0),'lane':0,'status':'started',
                'outcomes':[dict(cell,status='attempt_started')]}))
            result=campaign.reconcile_lane(out,'stopped_no_retry',0)
            self.assertEqual([r['status'] for r in result['outcomes']],['failed_no_retry']+['not_run']*15)
            self.assertEqual(result['generated_files_verified'],0)
    def test_empty_claim_is_consumed_not_retryable(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ident=campaign.plan()[0]['cell_id']
            (out/(ident+'.claim')).write_bytes(b'')
            result=campaign.reconcile_lane(out,'stopped_no_retry',0)
            self.assertEqual(result['outcomes'][0]['status'],'failed_no_retry')
            self.assertEqual(result['claimed_attempts'],1)
    def test_success_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.generate(Model(),out)
            result=campaign.reconcile_lane(out,'generated_pending_independent_screen',0)
            self.assertEqual(result['generated_files_verified'],16)
            self.assertTrue(result['all_lane_technically_valid'])
            self.assertFalse(result['training_admitted'])
    def test_changed_generated_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.generate(Model(),out)
            next(out.glob('*.wav')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'GENERATED_WAV_CHANGED'):
                campaign.reconcile_lane(out,'generated_pending_independent_screen',0)
    def test_setup_failure_all_not_run(self):
        with tempfile.TemporaryDirectory() as d:
            result=campaign.reconcile_lane(Path(d),'stopped_no_retry',0)
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
        self.assertIn('timeout-minutes: 90',text)
        self.assertIn('github.run_number == 1',text)
        self.assertNotIn('pull_request:',text)
        self.assertNotIn('workflow_dispatch:',text)
        self.assertNotIn('contents: write',text)

    def test_all_speakers_checked_before_any_claim(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();model.get_supported_speakers=lambda:['Vivian']
            with self.assertRaisesRegex(ValueError,'ALL_FOUR_PRESETS_REQUIRED'):self.generate(model,Path(d))
            self.assertEqual(model.calls,[]);self.assertEqual(list(Path(d).iterdir()),[])
    def test_invalid_lanes_and_negative_subsequence_rejected(self):
        for lane in (-1,4,True,'0'):
            with self.assertRaises(ValueError):campaign.lane_plan(lane)
        phrases=list(campaign.PHRASES);phrases[-1]=('你好小窝你好','near-miss-intent',None)
        with patch.object(campaign,'PHRASES',phrases),self.assertRaisesRegex(ValueError,'NEGATIVE_CONTAINS_WAKE'):
            campaign.plan()
    def test_fixed_64_terminal_when_setup_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for lane in range(4):(root/('lane-%d'%lane)/'output').mkdir(parents=True)
            result=campaign.reconcile_terminal(root,'stopped_no_retry')
            self.assertEqual(len(result['outcomes']),64)
            self.assertEqual(result['claimed_attempts'],0)
            self.assertFalse(result['all_64_technically_valid'])
    def test_all_four_lanes_mocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);seen=[]
            for lane in range(4):
                out=root/('lane-%d'%lane)/'output';out.mkdir(parents=True)
                model=Model();self.generate(model,out,lane);seen.extend(model.calls)
            result=campaign.reconcile_terminal(root,'generated_pending_independent_screen')
            self.assertEqual(len(seen),64);self.assertEqual(result['claimed_attempts'],64)
            self.assertTrue(result['all_64_technically_valid']);self.assertFalse(result['training_admitted'])
    def test_exact_generation_kwargs(self):
        with tempfile.TemporaryDirectory() as d:
            model=Model();self.generate(model,Path(d))
            for kwargs in model.calls:
                self.assertEqual(kwargs['max_new_tokens'],120)
                self.assertEqual(kwargs['instruct'],'');self.assertEqual(kwargs['language'],'Chinese')
                self.assertEqual(set(kwargs),{'text','language','speaker','instruct','do_sample','top_k','top_p',
                    'temperature','repetition_penalty','subtalker_dosample','subtalker_top_k',
                    'subtalker_top_p','subtalker_temperature','max_new_tokens','non_streaming_mode'})
    def test_python38_syntax(self):
        import ast
        for name in ('campaign.py','worker.py','test_campaign.py'):
            ast.parse((Path(__file__).parent/name).read_text(),feature_version=(3,8))

    def run_fake_host(self, fail_lane=None, expire_after_setup=False, expire_during=None):
        import shutil
        sys.path.insert(0,str(campaign.ROOT/'research/source-screen32-v1'))
        import hosted_run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'repo';shutil.copytree(campaign.ROOT,root)
            env,event=self.event();event_path=root/'event.json';event_path.write_text(json.dumps(event))
            env['GITHUB_EVENT_PATH']=str(event_path)
            stages=[];clock=[100.0];capacity_calls=[0]
            original_args=hosted_run.container_args
            def capacity():
                capacity_calls[0]+=1
                if expire_during == 'capacity' and capacity_calls[0] == 2:
                    clock[0]=4781.0  # 119 seconds remain after the slow check.
                return {}
            def arguments(*args):
                result=original_args(*args)
                if expire_during == 'argv':clock[0]=4781.0
                return result
            def supervise(args,name,output,stage,deadline):
                stages.append((stage,args,deadline))
                if stage == 'setup':
                    if expire_after_setup:clock[0]=4900.0
                else:
                    lane=int(args[-1]);self.generate(Model(),output,lane)
                    if lane == fail_lane:raise RuntimeError('invented ambiguous cleanup failure')
                return dict(status='complete',wall_seconds=1,exit_code=0)
            with patch.object(campaign,'ROOT',root),patch.object(campaign.shutil,'disk_usage',return_value=NS(free=32*1024**3)), \
                 patch.object(hosted_run,'host_capacity',side_effect=capacity),patch.object(hosted_run,'pull_image'), \
                 patch.object(hosted_run,'container_args',side_effect=arguments), \
                 patch.object(hosted_run,'supervise_container',side_effect=supervise), \
                 patch.object(hosted_run.os,'getuid',return_value=1001), \
                 patch.object(campaign.time,'monotonic',side_effect=lambda:clock[0]):
                if fail_lane is not None or expire_after_setup or expire_during:
                    with self.assertRaises((RuntimeError,ValueError)):campaign.run(env)
                else:campaign.run(env)
            work=root/('nonclone-'+campaign.CAMPAIGN)
            result=json.loads((work/'terminal.json').read_text())
            return stages,result
    def test_host_setup_once_four_isolated_lanes(self):
        stages,result=self.run_fake_host()
        self.assertEqual([s[0] for s in stages],['setup']+['tts']*4)
        self.assertEqual(result['generated_files_verified'],64)
        mounts=[]
        for stage,args,deadline in stages:
            self.assertLessEqual(deadline,4840)
            self.assertIn('--network',args)
            self.assertEqual(args[args.index('--network')+1],'bridge' if stage=='setup' else 'none')
            runtime=[a for a in args if 'dst=/runtime' in a][0]
            self.assertEqual(runtime.endswith(',readonly'),stage!='setup')
            mounts.append([a for a in args if 'dst=/output' in a][0])
        self.assertEqual(len(set(mounts)),5)
    def test_cleanup_failure_stops_later_lanes(self):
        stages,result=self.run_fake_host(fail_lane=1)
        self.assertEqual(len(stages),3)
        self.assertEqual(len(result['outcomes']),64)
        self.assertEqual(result['claimed_attempts'],32)
        self.assertEqual([r['status'] for r in result['outcomes'][32:]],['not_run']*32)
        self.assertEqual(result['host_status'],'stopped_no_retry')
    def test_absolute_deadline_prevents_new_lane(self):
        stages,result=self.run_fake_host(expire_after_setup=True)
        self.assertEqual(len(stages),1)
        self.assertEqual(result['claimed_attempts'],0)
    def test_reserve_rechecked_after_host_capacity(self):
        stages,result=self.run_fake_host(expire_during='capacity')
        self.assertEqual(stages,[])
        self.assertEqual(result['claimed_attempts'],0)
    def test_reserve_rechecked_after_argv_construction(self):
        stages,result=self.run_fake_host(expire_during='argv')
        self.assertEqual(stages,[])
        self.assertEqual(result['claimed_attempts'],0)
    def test_unknown_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'unknown.claim').write_text('unknown')
            with self.assertRaisesRegex(ValueError,'UNKNOWN_CLAIM'):
                campaign.reconcile_lane(out,'stopped_no_retry',0)

if __name__=='__main__':unittest.main()
