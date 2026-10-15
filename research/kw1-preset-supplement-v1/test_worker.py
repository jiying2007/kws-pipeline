"""Model-free worker, durability, schema and native signal tests."""
import copy
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import worker as w


def termination(**changes):
    return dict(dict(iterations=10,last_token_ids=[2150],eos=2150,ended_with_eos=True,hit_token_cap_without_eos=False), **changes)


def fixture(cell, samples=None, term=None):
    if samples is None:
        index = int(cell['slot_id'].split('-')[1])+1
        samples = [0.125*math.sin((i+index)*0.19) for i in range(240+index)]
    raw = w.float_wav(struct.pack('<'+'f'*len(samples), *samples))
    values, samples = w.native_samples(raw)
    term = termination() if term is None else term
    native = dict(sample_rate_hz=24000, channels=1, frames=len(samples), subtype='FLOAT', source_values_sha256=w.sha(values))
    row = dict(cell,status='completed',attempts=1,error_type=None,file=cell['slot_id']+'.raw-float.wav',
        raw_wav_sha256=w.sha(raw),native=native,metrics=w.signal_metrics(samples,term),termination=term)
    return raw, row


class FakeBackend:
    calls = []
    fail_at = None
    load_error = None
    signal = None
    term = None
    duplicate = False
    def __init__(self,engine,runtime,lock):
        self.engine = engine
        self.receipt = {'mock':'scientific loader not invoked'}
        self.runtime_identity = {'mock':'runtime not loaded'}
        self.source_pins = {'mock':'no source setup'}
    def __enter__(self):
        if self.load_error: raise self.load_error
        return self
    def __exit__(self,*_): return False
    def generate(self,cell):
        self.calls.append(cell['slot_id'])
        if len(self.calls) == self.fail_at: raise RuntimeError('secret error should not appear')
        chosen = dict(cell, slot_id='kw1s-00') if self.duplicate else cell
        raw, row = fixture(chosen,self.signal,self.term)
        values,_ = w.native_samples(raw)
        return values,row['termination']


def make_stage(root, engine, plan):
    """Create complete deterministic no-model receipts for host/export integration."""
    class FixtureBackend(FakeBackend):
        calls = []
        fail_at = None
        load_error = None
        signal = None
        term = None
        duplicate = False
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    return w.run(engine,plan,root,Path('/not-loaded'),Path('/not-loaded'),backend_factory=FixtureBackend)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.calls = []; FakeBackend.fail_at = None; FakeBackend.load_error = None
        FakeBackend.signal = None; FakeBackend.term = None; FakeBackend.duplicate = False
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name); self.plan = w.frozen_plan()
    def run_fake(self,engine='qwen06'):
        return w.run(engine,self.plan,self.output,Path('/not-loaded'),Path('/not-loaded'),backend_factory=FakeBackend)
    def test_sixteen_calls_in_two_fresh_stages(self):
        self.assertEqual(self.run_fake()['completed'],4)
        self.assertEqual([x['status'] for x in w.recover(self.output,'qwen06',self.plan)],['completed']*4)
        other = self.output/'second'; other.mkdir()
        self.assertEqual(w.run('qwen17',self.plan,other,Path('/none'),Path('/none'),backend_factory=FakeBackend)['completed'],12)
        self.assertEqual(FakeBackend.calls,[x['slot_id'] for x in self.plan['slots']])
    def test_failure_stops_second_call_and_never_retries(self):
        FakeBackend.fail_at=2
        summary=self.run_fake(); self.assertEqual(summary['attempted'],2); self.assertEqual(summary['status'],'stopped')
        outcomes=w.recover(self.output,'qwen06',self.plan)
        self.assertEqual([r['status'] for r in outcomes],['completed','FAILED','NOT_RUN','NOT_RUN'])
        with self.assertRaises(ValueError):self.run_fake()
        self.assertEqual(len(FakeBackend.calls),2)
        self.assertNotIn(b'secret error',b''.join(p.read_bytes() for p in self.output.rglob('*.json')))
    def test_load_failure_consumes_stage_no_call(self):
        FakeBackend.load_error=RuntimeError('secret path')
        summary=self.run_fake();self.assertEqual(summary['attempted'],0)
        self.assertTrue((self.output/'model-load-started.json').exists())
        self.assertTrue((self.output/'model-load-failed.json').exists())
        self.assertEqual(sum(r['attempts'] for r in w.recover(self.output,'qwen06',self.plan)),0)
        with self.assertRaises(ValueError):self.run_fake()
    def test_start_without_terminal_consumes_attempt(self):
        FakeBackend.fail_at=1; self.run_fake()
        (self.output/'cells/kw1s-00/failed.json').unlink()
        outcomes=w.recover(self.output,'qwen06',self.plan)
        self.assertEqual((outcomes[0]['attempts'],outcomes[0]['status'],outcomes[0]['error_type']),(1,'FAILED','interrupted'))
        self.assertEqual(sum(r['attempts'] for r in outcomes),1)
    def test_empty_started_receipt_is_unverified_not_zero(self):
        FakeBackend.fail_at=1; self.run_fake()
        p=self.output/'cells/kw1s-00/started.json';p.write_bytes(b'')
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_malformed_terminal_is_unverified(self):
        FakeBackend.fail_at=1;self.run_fake()
        p=self.output/'cells/kw1s-00/failed.json';p.write_bytes(b'{"secret":"never public"')
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_missing_load_parent_is_unverified(self):
        self.run_fake();(self.output/'model-load-started.json').unlink()
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_wrong_load_hash_is_unverified(self):
        self.run_fake();p=self.output/'cells/kw1s-00/started.json';v=w.decode(p.read_bytes());v['load_completed_sha256']='0'*64;p.write_bytes(w.canonical(v))
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_contradictory_terminal_is_unverified(self):
        self.run_fake();p=self.output/'cells/kw1s-00/failed.json';p.write_text('{}')
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_failed_then_later_call_is_unverified(self):
        self.run_fake();(self.output/'cells/kw1s-00/completed.json').unlink()
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_duplicate_waveform_stops_and_consumes(self):
        FakeBackend.duplicate=True;summary=self.run_fake()
        self.assertEqual(summary['attempted'],2)
        rows=w.recover(self.output,'qwen06',self.plan)
        self.assertEqual(rows[1]['error_type'],'duplicate_native')
    def test_silent_audio_preserves_failed_metrics(self):
        FakeBackend.signal=[0.0]*240
        self.assertEqual(self.run_fake()['attempted'],1)
        row=w.recover(self.output,'qwen06',self.plan)[0]
        self.assertEqual(row['metrics']['quality_flags'],['near_silent'])
        self.assertFalse(row['metrics']['technically_valid'])
    def test_clipping_stops_no_repair(self):
        FakeBackend.signal=[1.0,-1.0]*120
        self.assertEqual(self.run_fake()['attempted'],1)
        row=w.recover(self.output,'qwen06',self.plan)[0]
        self.assertEqual(row['metrics']['clipped_fraction'],1.0)
    def test_eos_at_cap_fails_even_with_eos(self):
        FakeBackend.term=termination(iterations=120)
        self.assertEqual(self.run_fake()['attempted'],1)
        row=w.recover(self.output,'qwen06',self.plan)[0]
        self.assertEqual(row['error_type'],'technical_quality')
        self.assertEqual(row['metrics']['quality_flags'],['termination_unverified'])
    def test_non_eos_fails(self):
        FakeBackend.term=termination(last_token_ids=[42],ended_with_eos=False)
        self.assertEqual(self.run_fake()['attempted'],1)
    def test_raw_native_values_and_sha(self):
        raw,row=fixture(self.plan['slots'][0])
        self.assertTrue(w.verify_audio_outcome(raw,row))
        tampered=raw[:-4]+struct.pack('<f',0.5)
        with self.assertRaises(ValueError):w.verify_audio_outcome(tampered,row)
    def test_nonfinite_samples_rejected(self):
        for v in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):w.native_samples(w.float_wav(struct.pack('<f',v)))
    def test_wav_chunk_truncation_extra_or_pcm_rejected(self):
        raw,_=fixture(self.plan['slots'][0])
        for bad in (raw[:-1],raw+b'junk',raw[:20]+struct.pack('<H',1)+raw[22:]):
            with self.assertRaises(ValueError):w.native_samples(bad)
    def test_forged_metrics_rejected(self):
        raw,row=fixture(self.plan['slots'][0]);row['metrics']['rms']=0.9
        with self.assertRaises(ValueError):w.verify_audio_outcome(raw,row)
    def test_unknown_outcome_body_is_not_projected(self):
        _,row=fixture(self.plan['slots'][0]);row['private']='secret'
        with self.assertRaises(ValueError):w.validate_outcome(row,self.plan['slots'][0],'completed')
    def test_bool_attempt_and_float_frames_rejected(self):
        _,row=fixture(self.plan['slots'][0]);row['attempts']=True
        with self.assertRaises(ValueError):w.validate_outcome(row,self.plan['slots'][0],'completed')
        row['attempts']=1;row['native']['frames']=float(row['native']['frames'])
        with self.assertRaises(ValueError):w.validate_outcome(row,self.plan['slots'][0],'completed')
    def test_orphan_wave_cannot_be_zero_attempt(self):
        self.run_fake()
        import shutil
        shutil.rmtree(self.output/'cells')
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_stage_boolean_attempt_is_unverified(self):
        self.run_fake();p=self.output/'stage-claim.json';v=w.decode(p.read_bytes());v['attempt']=True;p.write_bytes(w.canonical(v))
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_extra_cell_body_is_unverified(self):
        self.run_fake();(self.output/'cells/kw1s-00/secret.json').write_text('{"private":"secret"}')
        with self.assertRaises(ValueError):w.recover(self.output,'qwen06',self.plan)
    def test_receipt_symlink_and_hardlink_rejected(self):
        p=self.output/'source';p.write_bytes(b'abc');q=self.output/'link';q.symlink_to(p)
        with self.assertRaises(ValueError):w.read_regular(q)
        q.unlink();os.link(p,q)
        with self.assertRaises(ValueError):w.read_regular(q)
    def test_exclusive_empty_claim_cannot_be_replaced(self):
        p=self.output/'claim';p.touch()
        with self.assertRaises(FileExistsError):w.write_once(p,{'attempt':1})
        self.assertEqual(p.read_bytes(),b'')
    def test_import_is_stdlib_only(self):
        import ast
        tree=ast.parse(Path(w.__file__).read_text())
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertTrue(all(n.name not in ('torch','numpy','qwen_tts','transformers') for n in node.names))
    def test_retained_source_set_matches_backend(self):
        import docker_backend
        self.assertEqual(set(w.RETAINED_PINS)|{'research/kw1-preset-supplement-v1/worker.py'},set(docker_backend.TTS_FILES))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace as NS
        from contextlib import nullcontext
        self.seeds = []
        class Array:
            dtype='float32'; ndim=1; size=4
            def astype(self,dtype,copy=False):
                if dtype != '<f4' or copy: raise ValueError('unexpected conversion')
                return self
            def tobytes(self):return struct.pack('<ffff',0.1,0.2,-0.1,-0.2)
        class Seq:
            def __getitem__(self,_):return self
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [2150]
        attention=NS(training=False,q_proj=True,k_proj=True,v_proj=True,config=NS(_attn_implementation='eager'))
        parameter=NS(device='cpu',dtype='torch.float32')
        owner=NS(training=False,parameters=lambda:[parameter],named_modules=lambda:[('',attention)])
        talker=NS(generate=lambda *args,**kwargs:NS(sequences=Seq(),hidden_states=[None]*10))
        config=NS(talker_config=NS(codec_eos_token_id=2150,spk_id={k.lower():v for k,v in w.PRESET_IDS.items()},spk_is_dialect=dict(w.DIALECTS)))
        inner=NS(**vars(owner),tts_model_type='custom_voice',tts_model_size='1b7',config=config,talker=talker,
            speech_tokenizer=NS(model=owner))
        self.kwargs=[]
        def generate(**kwargs):
            self.kwargs.append(kwargs);talker.generate();return [Array()],24000
        model=NS(model=inner,generate_custom_voice=generate,get_supported_speakers=lambda:sorted(config.talker_config.spk_id))
        self.backend=object.__new__(w.RetainedBackend)
        self.backend.engine='qwen17';self.backend.model=model
        self.backend.torch=NS(manual_seed=lambda x:self.seeds.append(('torch',x)),inference_mode=nullcontext)
        self.backend.np=NS(float32='float32',asarray=lambda x:x,isfinite=lambda _:NS(all=lambda:True),
            random=NS(seed=lambda x:self.seeds.append(('numpy',x))))
        self.attention,self.parameter=attention,parameter
    def test_retained_generation_exact_arguments_and_three_rngs(self):
        cell=w.frozen_plan()['slots'][6]
        with patch.object(w,'verify_scope',return_value={}),patch.object(w.random,'seed',side_effect=lambda x:self.seeds.append(('python',x))):
            raw,term=self.backend.generate(cell)
        self.assertEqual(self.kwargs,[dict(text=cell['target_text'],speaker=cell['speaker'],language='Chinese',instruct=w.INSTRUCTION,**w.GENERATION)])
        self.assertEqual(self.seeds,[('python',cell['seed']),('numpy',cell['seed']),('torch',cell['seed'])])
        self.assertEqual(len(raw),16);self.assertTrue(w.validate_termination(term))
    def test_retained_wrong_dtype_device_and_eval_rejected(self):
        for obj,key,value in ((self.parameter,'dtype','torch.float16'),(self.parameter,'device','cuda:0'),(self.attention,'training',True)):
            original=getattr(obj,key);setattr(obj,key,value)
            with self.assertRaises(ValueError):self.backend._check_model()
            setattr(obj,key,original)
    def test_retained_wrong_attention_rejected(self):
        self.attention.config._attn_implementation='sdpa'
        with self.assertRaises(ValueError):self.backend._check_model()
    def test_retained_wrong_preset_and_eos_rejected(self):
        config=self.backend.model.model.config.talker_config
        config.spk_id['uncle_fu']=3009
        with self.assertRaises(ValueError):self.backend._check_model()
        config.spk_id['uncle_fu']=3010;config.codec_eos_token_id=42
        with self.assertRaises(ValueError):self.backend._check_model()
    def test_retained_wrong_dialect_route_rejected(self):
        self.backend.model.model.config.talker_config.spk_is_dialect['eric']=False
        with self.assertRaises(ValueError):self.backend._check_model()
    def test_retained_wrong_model_size_rejected(self):
        self.backend.model.model.tts_model_size='0b6'
        with self.assertRaises(ValueError):self.backend._check_model()
    def test_retained_missing_speaker_rejected(self):
        self.backend.model.get_supported_speakers=lambda:['uncle_fu','eric']
        with self.assertRaises(ValueError):self.backend._check_model()
    def test_acoustic_hook_restored_after_generation_error(self):
        original=self.backend.model.model.talker.generate
        def fail(**_):raise RuntimeError('mock')
        self.backend.model.generate_custom_voice=fail
        with patch.object(w,'verify_scope',return_value={}):
            with self.assertRaises(RuntimeError):self.backend.generate(w.frozen_plan()['slots'][6])
        self.assertIs(self.backend.model.model.talker.generate,original)
    def test_explicit_source_pins_match_recovered_bytes(self):
        root=Path(w.__file__).resolve().parents[2]
        for name,digest in w.RETAINED_PINS.items():self.assertEqual(w.sha(w.read_regular(root/name)),digest)



if __name__=='__main__': unittest.main()
