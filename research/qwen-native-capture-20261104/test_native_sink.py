"""Invented native identities/configs. Real strict validators, zero models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import asr_native_contracts as contracts
from generation_capture import canonical, EXPECTED
from native_sink import NativeEvidenceSink, sha, read_json
import capture_worker as worker

def fixture():
    body=b'invented source body'
    source=dict(schema='retained-asr-source-lock-v1',executed_head_sha='a'*40,
        files=[dict(path='invented.py',sha256=sha(body),bytes=len(body))])
    sr=canonical(source)
    raw=canonical(EXPECTED)
    identities=[];models=[]
    for i,(adapter,family) in enumerate(contracts.ADAPTER_FAMILIES.items()):
        assets=[dict(path='invented.weights',bytes=1,sha256=str(i+1)*64)]
        identity=dict(recognizer_id='recognizer'+str(i),provider_id='invented-provider'+str(i),
            model_id='invented-model'+str(i),model_revision='revision'+str(i),model_family=family,
            model_sha256=contracts.canonical_sha256(assets))
        model=dict(identity,schema='retained-asr-model-lock-v1',assets=assets)
        mr=canonical(model)
        if adapter==contracts.QWEN_ADAPTER:
            effective={k:copy.deepcopy(EXPECTED[k]) for k in EXPECTED if k not in ('do_sample','max_new_tokens','eos_token_id')}
            effective.update(execution=contracts.EXECUTION,native_request=contracts.QWEN_REQUEST,
                generation_mode='greedy_search',custom_logits_processors=[],custom_stopping_criteria=[],
                resolved_generation_config_sha256=sha(raw))
        elif adapter==contracts.SENSE_ADAPTER:
            effective=dict(execution=contracts.EXECUTION,native_request=contracts.SENSE_REQUEST,
                decoding='full_input_greedy_ctc',vad_used=False,rich_postprocess_applied=False,
                vocabulary_size=25055,lexical_unknown_id=0,metadata_prefix_tokens=4,
                tokenizer_sha256='a'*64,parser_contract_sha256='b'*64,frontend_config_sha256='c'*64)
        else:
            controls=dict(transcribe=1,translate=2,sot=3,sot_prev=4,sot_lm=5,no_speech=6,language=7,no_timestamps=8)
            effective=dict(execution=contracts.EXECUTION,native_request=contracts.WHISPER_REQUEST,
                decoder='greedy',n_text_ctx=448,effective_sample_len=224,tokenizer_sha256='a'*64,
                non_speech_token_ids=[11],control_token_ids=controls,resolved_suppress_token_ids=[1,2,3,4,5,6,11],
                space_token_ids=[10],initial_blank_suppress_token_ids=[9,10],eot_token_id=9,initial_token_ids=[3,7,1,8])
        config=dict(schema_version=2,evidence_class=contracts.CONFIG_CLASS,adapter_id=adapter,adapter_version=1,
            model_family=family,source_lock_sha256=sha(sr),asset_lock_sha256=sha(mr),
            conditioning_policy=contracts.CONDITIONING_POLICY,execution=contracts.EXECUTION,
            input_policy=dict(sample_rate=16000,channels=1,max_audio_frames=192000,
                pcm_format='float32_pcm16_scale_32768',preprocessing='zero_pad_only_480000' if i==2 else 'full_input'),
            native_request=contracts.REQUESTS[adapter],effective_config=effective,
            effective_config_sha256=contracts.canonical_sha256(effective))
        contracts.validate_native_config(config)
        identity.update(native_decode_config=config,native_decode_config_sha256=contracts.canonical_sha256(config))
        identities.append(identity);models.append(mr)
    policy=dict(schema_version=2,evidence_class='speech-three-asr-recognizer-policy-v2',
        normalization='fixed-punctuation-removal-only-v1',frozen_before_recognition=True,
        target_conditioning_allowed=False,recognizers=identities)
    return canonical(policy),models[0],sr,{'invented.py':body},raw

class SinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'native'
        self.policy,self.model,self.source,self.bodies,self.raw=fixture()
        self.assets=json.loads(self.model)['assets']
        self.bound=NS(opaque_id='clip-000001',descriptor=dict(
            operations=dict(mix=False,resample=False,trim=False),wav=dict(sha256='d'*64),
            raw_pcm16=dict(sha256='e'*64,frame_count=80,sample_rate_hz=16000,channels=1)))
    def sink(self):
        return NativeEvidenceSink(self.root,self.policy,self.model,self.source,'recognizer0',self.bodies,self.assets)
    def begin(self,sink):sink.begin(self.bound,'invented-source','invented-request')
    def start(self,sink):sink.before_continue(self.raw,contracts.EXECUTION,dict(custom_logits_processors=[],custom_stopping_criteria=[]))
    def evidence(self):
        return dict(input_binding=self.bound.descriptor,prompt_context='',force_language=None,
            transcript_repair_applied=False,metadata=dict(eos_terminated=True,generated_tokens=2),token_ids=[123,151645])
    def row(self):return dict(raw_text='invented transcript',completeness='complete',quality_flags=[])
    def test_started_precedes_decode_and_receipt_hash_chain(self):
        s=self.sink();self.begin(s)
        self.assertFalse((self.root/'clip-000001/started.json').exists())
        self.start(s)
        self.assertTrue((self.root/'clip-000001/started.json').exists())
        self.assertFalse((self.root/'clip-000001/completed.json').exists())
        result=s.complete(self.row(),self.evidence());d=self.root/'clip-000001'
        self.assertEqual(result['started_sha256'],sha((d/'started.json').read_bytes()))
        self.assertEqual(result['raw_output_sha256'],sha((d/'raw-output.json').read_bytes()))
        self.assertEqual(result['recognizer_policy_sha256'],contracts.canonical_sha256(json.loads(self.policy)))
        contracts.validate_native_capture(s.config,result['native_capture'])
    def test_missing_prefrozen_policy_rejected(self):
        self.policy=b'{}'
        with self.assertRaises(ValueError):self.sink()
    def test_snapshot_drift_blocks_start(self):
        s=self.sink();self.begin(s);self.raw+=b' '
        with self.assertRaises(ValueError):self.start(s)
        self.assertFalse((self.root/'clip-000001/started.json').exists())
    def test_source_body_drift_rejected(self):
        self.bodies['invented.py']=b'changed'
        with self.assertRaises(ValueError):self.sink()
    def test_asset_inventory_drift_rejected(self):
        self.assets[0]['sha256']='f'*64
        with self.assertRaises(ValueError):self.sink()
    def test_no_posthoc_start(self):
        s=self.sink();self.begin(s)
        with self.assertRaises(ValueError):s.complete(self.row(),self.evidence())
    def test_duplicate_start_rejected(self):
        s=self.sink();self.begin(s);self.start(s)
        with self.assertRaises(ValueError):self.start(s)
    def test_failure_never_completed(self):
        s=self.sink();self.begin(s);self.start(s);s.fail()
        with self.assertRaises(ValueError):s.complete(self.row(),self.evidence())
        self.assertFalse((self.root/'clip-000001/completed.json').exists())
    def test_missing_eos_rejected(self):
        s=self.sink();self.begin(s);self.start(s);e=self.evidence();e['token_ids']=[123,124]
        with self.assertRaises(ValueError):s.complete(self.row(),e)
    def test_audio_binding_drift_rejected(self):
        s=self.sink();self.begin(s);self.start(s);e=self.evidence();e['input_binding']={}
        with self.assertRaises(ValueError):s.complete(self.row(),e)
    def test_custom_controls_rejected_before_start(self):
        s=self.sink();self.begin(s)
        with self.assertRaises(ValueError):s.before_continue(self.raw,contracts.EXECUTION,{'custom_logits_processors':['custom'],'custom_stopping_criteria':[]})

    def test_actual_hook_start_exists_before_native_forward(self):
        s=self.sink()
        bound=self.bound
        bound.descriptor_sha256='f'*64
        config=NS(**copy.deepcopy(EXPECTED))
        config.to_dict=lambda:copy.deepcopy(EXPECTED)
        sink_root=self.root
        class Thinker:
            training=False
            def parameters(self):return iter([NS(device='cpu',dtype='torch.float32')])
            def _prepare_generation_config(self,*a,**kw):return config,{}
            def generate(self,inputs=None,generation_config=None,logits_processor=None,stopping_criteria=None,
                         prefix_allowed_tokens_fn=None,assistant_model=None,custom_generate=None,
                         negative_prompt_ids=None,negative_prompt_attention_mask=None,**kwargs):
                self._prepare_generation_config()
                require_started=(sink_root/'clip-000001/started.json').is_file()
                if not require_started:raise AssertionError('forward preceded native start')
                return 'fake forward only'
        thinker=Thinker();loaded=NS(wrapper=NS(model=NS(thinker=thinker)))
        cap=worker.BoundCapture(loaded,Path(self.temp.name)/'observations',1,
                               native_sink=s,request_map={'clip-000001':{'source_id':'invented-source','request_id':'invented-request'}})
        def inference(*args):
            thinker.generate(input_ids=NS(shape=(1,4)))
            return self.row(),self.evidence()
        row,evidence=cap.infer(inference,loaded,bound,'c'*64)
        self.assertIn('strict_completion_sha256',evidence['prospective_generation_capture'])
        self.assertTrue((self.root/'clip-000001/completed.json').is_file())
        cap.restore()

    def test_prepare_native_calls_resolver_without_forward(self):
        class Config:
            def to_dict(self):return copy.deepcopy(EXPECTED)
            def get_generation_mode(self,assistant):return NS(value='greedy_search')
        class Thinker:
            training=False
            def parameters(self):return iter([NS(device='cpu',dtype='torch.float32')])
            def _prepare_generation_config(self,*args,**kwargs):
                self.seen=(args,kwargs)
                return Config(),{}
            def generate(self,*args,**kwargs):raise AssertionError('prepare must never forward')
        thinker=Thinker();loaded=NS(wrapper=NS(model=NS(thinker=thinker)))
        with patch.object(worker,'verified_native_source'):
            cfg=worker.prepare_native_qwen(loaded,Path(self.temp.name)/'prepared',self.model,self.source)
        contracts.validate_native_config(cfg)
        self.assertEqual(thinker.seen[1]['return_dict_in_generate'],True)
        self.assertEqual(cfg['effective_config']['resolved_generation_config_sha256'],sha(self.raw))

    def test_unsafe_opaque_ids_rejected_before_directory_creation(self):
        s=self.sink()
        for value in ('/absolute','../escape','one/two','one\\two','.','..','clip-000001/..'):
            self.bound.opaque_id=value
            with self.assertRaises(ValueError):self.begin(s)
        self.assertIsNone(s.active)
        self.assertEqual(s.seen,set())

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):read_json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):read_json(b'{"nested":{"a":1,"a":2}}')

    def test_nonfinite_json_rejected(self):
        for value in (b'NaN',b'Infinity',b'-Infinity',b'1e999',b'-1e999'):
            with self.assertRaises(ValueError):read_json(b'{"x":'+value+b'}')

if __name__=='__main__':unittest.main()
