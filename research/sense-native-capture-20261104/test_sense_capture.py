"""Invented object fixtures only: no imports/install/model execution/network."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sense_capture as c

class Tensor:
    device='cpu';dtype='torch.float32';shape=(2,3)
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return [[.1,.2,.3],[1.,1.,1.]]

class WavFrontend:
    __module__='funasr.frontends.wav_frontend'
    def __init__(self):
        self.fs=16000;self.window='hamming';self.n_mels=80;self.frame_length=25;self.frame_shift=10
        self.filter_length_min=-1;self.filter_length_max=-1;self.lfr_m=7;self.lfr_n=6
        self.dither=1.;self.snip_edges=True;self.upsacle_samples=True;self.cmvn=Tensor()

class PCM:
    dtype='float32';ndim=1
    def __len__(self):return 2
    def tobytes(self,order='C'):return struct.pack('<2f',.1,.2)

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.paths={};self.reviewed={}
        for name in c.DEPENDENCIES:
            path=self.root/name;path.write_bytes(('invented '+name).encode())
            self.paths[name]=path;self.reviewed[name]=dict(bytes=path.stat().st_size,sha256=c.sha(path.read_bytes()))
        self.frontend=WavFrontend();self.frontend.cmvn_file=str(self.paths['cmvn_file'])
        self.tokenizer=NS(sp=NS(serialized_model_proto=lambda:self.paths['tokenizer_file'].read_bytes()))
        self.path_patch=patch.object(c,'dependency_paths',return_value=self.paths);self.path_patch.start()
        self.map_patch=patch.object(c,'mapped_binary',return_value=True);self.map_patch.start()
        self.original_calls=0
        def inference(**kw):self.original_calls+=1;return 'untouched-return'
        self.runtime=NS(wrapper=NS(model=NS(inference=inference),kwargs=dict(frontend=self.frontend,tokenizer=self.tokenizer)),tokenizer_file=self.paths['tokenizer_file'])
        self.original=inference
        pcm=PCM()
        self.bound=NS(opaque_id='clip-000001',descriptor_sha256='d'*64,descriptor=dict(wav=dict(sha256='a'*64),raw_pcm16=dict(sha256='b'*64),float32_pcm=dict(sha256=c.sha(pcm.tobytes()),frame_count=2)))
    def tearDown(self):self.path_patch.stop();self.map_patch.stop();self.temp.cleanup()
    def capture(self):return c.BoundSenseCapture(self.runtime,self.root/'out',self.paths['parser_source'],self.reviewed,2)
    def adapter(self,runtime,bound,decoder,model_root,state):
        result=runtime.wrapper.model.inference(data_in=PCM(),key=[bound.opaque_id],tokenizer=NS(_tokenizer=self.tokenizer),frontend=self.frontend,**c.REQUEST)
        self.assertEqual(result,'untouched-return')
        return {'status':'complete'},{'invented_evidence':True}
    def invoke(self,cap,adapter=None):return cap.infer(adapter or self.adapter,self.runtime,self.bound,'e'*64,self.root,{})
    def test_success_chain_and_restore(self):
        cap=self.capture();row,evidence=self.invoke(cap)
        self.assertEqual(self.original_calls,1)
        obs=evidence['prospective_frontend_capture'];self.assertFalse(obs['training_admitted'])
        snap=json.loads((cap.output/obs['snapshot_file']).read_text())
        self.assertEqual(snap['attributes']['lfr_m'],7)
        self.assertEqual(snap['binding']['opaque_id'],self.bound.opaque_id)
        self.assertEqual(c.sha((cap.output/obs['snapshot_file']).read_bytes()),obs['frontend_snapshot_sha256'])
        self.assertEqual(c.sha((cap.output/'claim-0001.json').read_bytes()),obs['claim_sha256'])
        self.assertTrue((cap.output/'completed-0001.json').is_file())
        cap.restore();self.assertIs(self.runtime.wrapper.model.inference,self.original)
    def test_snapshot_persisted_before_native_call(self):
        self.runtime.wrapper.model.inference=lambda **kw: self.assertTrue((self.root/'out/frontend-0001.json').is_file()) or 'untouched-return'
        self.invoke(self.capture())
    def test_missing_field_fails_before_call(self):
        del self.frontend.window;cap=self.capture()
        with self.assertRaises(AttributeError):self.invoke(cap)
        self.assertEqual(self.original_calls,0);self.assertTrue(cap.dead)
    def test_dependency_drift(self):
        self.paths['fbank_source'].write_bytes(b'changed');cap=self.capture()
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.invoke(cap)
        self.assertEqual(self.original_calls,0)
    def test_no_inference_without_claim(self):
        cap=self.capture()
        with self.assertRaises(ValueError):self.runtime.wrapper.model.inference()
        self.assertEqual(self.original_calls,0)
    def test_no_repeat(self):
        cap=self.capture();self.invoke(cap)
        with self.assertRaises(ValueError):self.invoke(cap)
        self.assertEqual(self.original_calls,1)
    def test_failure_poison_and_restore(self):
        def fail(**kw):raise RuntimeError('invented inference failure')
        self.runtime.wrapper.model.inference=fail;cap=self.capture()
        with self.assertRaises(RuntimeError):self.invoke(cap)
        with self.assertRaises(ValueError):self.invoke(cap)
        self.assertTrue((cap.output/'frontend-0001.json').exists())
        self.assertFalse((cap.output/'completed-0001.json').exists())
        cap.restore();self.assertIs(self.runtime.wrapper.model.inference,fail)
    def test_frontend_mutation_fails(self):
        def mutate(**kw):self.frontend.dither=0.;return 'untouched-return'
        self.runtime.wrapper.model.inference=mutate;cap=self.capture()
        with self.assertRaisesRegex(ValueError,'changed during inference'):self.invoke(cap)
    def test_wrong_pcm_fails_before_call(self):
        self.bound.descriptor['float32_pcm']['sha256']='f'*64;cap=self.capture()
        with self.assertRaisesRegex(ValueError,'PCM bytes mismatch'):self.invoke(cap)
        self.assertEqual(self.original_calls,0)
    def test_no_calls_is_failure(self):
        cap=self.capture()
        with self.assertRaisesRegex(ValueError,'exactly one'):self.invoke(cap,lambda *a:({},{}))
    def test_duplicate_native_call_fails(self):
        def twice(*args):self.adapter(*args);return self.adapter(*args)
        cap=self.capture()
        with self.assertRaises(ValueError):self.invoke(cap,twice)
        self.assertEqual(self.original_calls,1)
    def test_bad_request_fails_before_call(self):
        def bad(r,b,*args):
            kw=dict(c.REQUEST);kw['language']='zh'
            return r.wrapper.model.inference(data_in=PCM(),key=[b.opaque_id],tokenizer=NS(_tokenizer=self.tokenizer),frontend=self.frontend,**kw)
        cap=self.capture()
        with self.assertRaisesRegex(ValueError,'request mismatch'):self.invoke(cap,bad)
        self.assertEqual(self.original_calls,0)
    def test_snapshot_write_failure_no_model_call(self):
        cap=self.capture();(cap.output/'frontend-0001.json').write_text('existing')
        with self.assertRaises(FileExistsError):self.invoke(cap)
        self.assertEqual(self.original_calls,0)
    def test_changed_tokenizer_fails(self):
        self.tokenizer.sp.serialized_model_proto=lambda:b'wrong';cap=self.capture()
        with self.assertRaisesRegex(ValueError,'loaded tokenizer'):self.invoke(cap)
    def test_nan_field_fails(self):
        self.frontend.dither=float('nan');cap=self.capture()
        with self.assertRaisesRegex(ValueError,'finite'):self.invoke(cap)
    def test_wrong_cmvn_dtype_fails(self):
        self.frontend.cmvn.dtype='torch.float16';cap=self.capture()
        with self.assertRaisesRegex(ValueError,'CMVN'):self.invoke(cap)
    def test_symlink_dependency_fails(self):
        name='cmvn_file';source=self.paths[name];other=self.root/'other';source.rename(other);source.symlink_to(other)
        cap=self.capture()
        with self.assertRaisesRegex(ValueError,'alias'):self.invoke(cap)
    def test_python38_syntax(self):
        for name in ('sense_capture.py','test_sense_capture.py'):
            ast.parse((Path(__file__).parent/name).read_text(),feature_version=(3,8))
    def test_no_model_imports(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

    def test_wrapper_restores_success_and_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                root=self.root/str(fail);(root/'sensevoice').mkdir(parents=True)
                self.runtime.decoder_inputs=[1,2]
                original_loader=lambda *a,**k:self.runtime
                original_infer=self.adapter
                adapter=NS(infer_sense_once_after_approval=original_infer)
                legacy=NS(load_sense_after_approval=original_loader)
                def run(model,*args):
                    self.assertEqual(model,'sensevoice')
                    loaded=legacy.load_sense_after_approval()
                    if fail:raise RuntimeError('invented postload failure')
                    return adapter.infer_sense_once_after_approval(loaded,self.bound,'e'*64,self.root,{})
                legacy.run=run
                if fail:
                    with self.assertRaises(RuntimeError):c.run_wrapped(legacy,adapter,self.root,root,self.root,self.paths['parser_source'],self.reviewed)
                else:c.run_wrapped(legacy,adapter,self.root,root,self.root,self.paths['parser_source'],self.reviewed)
                self.assertIs(legacy.load_sense_after_approval,original_loader)
                self.assertIs(adapter.infer_sense_once_after_approval,original_infer)
                self.assertIs(self.runtime.wrapper.model.inference,self.original)
    def test_actual_dependency_resolution_no_glob(self):
        from types import ModuleType
        frontend_file=self.root/'frontend.py';fbank_file=self.root/'fbank.py'
        frontend_file.write_text('def forward(self): pass\n')
        fbank_file.write_text('def fbank(): pass\n')
        namespace={};exec(compile(fbank_file.read_text(),str(fbank_file),'exec'),namespace)
        binary_module=ModuleType('invented_native_backend');binary_module.__file__=str(self.root/'backend.so')
        backend=NS(__file__=str(self.root/'backend_init.py'),OnlineFbank=type('OnlineFbank',(),{'__module__':'invented_native_backend'}))
        kaldi=NS(_HAS_TORCHAUDIO=False,_knf=backend,fbank=namespace['fbank'])
        namespace2={'kaldi':kaldi};exec(compile(frontend_file.read_text(),str(frontend_file),'exec'),namespace2)
        front=type('Frontend',(),{'forward':namespace2['forward']})();front.cmvn_file=str(self.paths['cmvn_file'])
        self.path_patch.stop()
        try:
            with patch.dict(sys.modules,{'invented_native_backend':binary_module}):
                paths=c.dependency_paths(front,self.paths['tokenizer_file'],self.paths['parser_source'])
                self.assertEqual(set(paths),c.DEPENDENCIES)
                self.assertEqual(paths['backend_binary'],Path(binary_module.__file__))
                kaldi._HAS_TORCHAUDIO=True
                with self.assertRaisesRegex(ValueError,'backend required'):c.dependency_paths(front,self.paths['tokenizer_file'],self.paths['parser_source'])
        finally:self.path_patch.start()
    def test_wrong_capturing_tokenizer_fails(self):
        def wrong(r,b,*args):
            return r.wrapper.model.inference(data_in=PCM(),key=[b.opaque_id],tokenizer=NS(_tokenizer=object()),frontend=self.frontend,**c.REQUEST)
        with self.assertRaisesRegex(ValueError,'tokenizer identity'):self.invoke(self.capture(),wrong)
        self.assertEqual(self.original_calls,0)
    def test_cmvn_tensor_values_recorded_not_reconstructed(self):
        cap=self.capture();_,evidence=self.invoke(cap)
        snap=json.loads((cap.output/evidence['prospective_frontend_capture']['snapshot_file']).read_text())
        self.assertEqual(snap['cmvn']['values'],self.frontend.cmvn.tolist())
        self.assertEqual(snap['cmvn']['sha256'],c.sha(struct.pack('<6f',.1,.2,.3,1.,1.,1.)))

    def prepared(self):
        self.runtime.wrapper.kwargs['batch_size']=1
        self.runtime.wrapper.model.parameters=lambda:[NS(device='cpu',dtype='torch.float32')]
        self.tokenizer.sp.get_piece_size=lambda:25055;self.tokenizer.sp.unk_id=lambda:0
        return c.prepare_loaded(self.runtime,self.root/'prepare',self.paths['parser_source'],self.reviewed,
            b'{"original":"source lock"}',b'{"original":"asset lock"}',16000*12,[['lang'],['emotion'],['event'],['itn']])
    def test_prepare_zero_forward_and_private_bytes(self):
        result=self.prepared()
        self.assertEqual(self.original_calls,0);self.assertEqual(result['forward_calls'],0)
        self.assertEqual(result['native_config']['effective_config']['vocabulary_size'],25055)
        self.assertEqual(set(result['dependencies']),c.DEPENDENCIES|{'frontend_observation'})
        for ref in result['dependencies'].values():
            raw=Path(ref['path']).read_bytes();self.assertEqual(c.sha(raw),ref['sha256'])
        self.assertEqual(result['strict_resolver_dependency_membership'],'pending_review_for_runtime_derived_frontend')
    def test_prepare_unknown_batchsize_rejected(self):
        self.runtime.wrapper.kwargs['batch_size']=None
        with self.assertRaises(ValueError):c.prepare_loaded(self.runtime,self.root/'prepare',self.paths['parser_source'],self.reviewed,b'{}',b'{}',10,[[],[],[],[]])
    def test_prepared_config_mutation_before_forward(self):
        prepared=self.prepared();self.frontend.frame_shift=11
        cap=c.BoundSenseCapture(self.runtime,self.root/'out',self.paths['parser_source'],self.reviewed,2,prepared)
        with self.assertRaisesRegex(ValueError,'prepared frontend changed'):self.invoke(cap)
        self.assertEqual(self.original_calls,0)
    def test_strict_sink_order_and_native_contract(self):
        import importlib.util
        path=Path(__file__).resolve().parent/'retained_native_contracts.py'
        spec=importlib.util.spec_from_file_location('sense_test_native_contract',path)
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        prepared=self.prepared();native.validate_native_config(prepared['native_config'])
        events=[]
        class Sink:
            def begin(self,*args):events.append('begin')
            def start_native(self,config):
                native.validate_native_config(config);events.append('start')
            def complete_native(self,text,capture,evidence):
                native.validate_native_capture(prepared['native_config'],capture);events.append('complete')
            def fail(self):events.append('fail')
        def actual(**kw):events.append('forward');return 'untouched-return'
        self.runtime.wrapper.model.inference=actual
        cap=c.BoundSenseCapture(self.runtime,self.root/'out',self.paths['parser_source'],self.reviewed,2,
            prepared,Sink(),{self.bound.opaque_id:dict(source_id='synthetic-fixture',request_id='request-fixture')})
        def adapter(*args):
            self.adapter(*args)
            binding=dict(self.bound.descriptor)
            binding['operations']=dict(mix=False,resample=False,trim=False)
            binding['float32_pcm']=dict(binding['float32_pcm'],sample_rate_hz=16000,channels=1)
            token=dict(metadata_boundary_verified=True,raw_redecode_matches_capture=True,lexical_unknown_present=False,
                token_scope='Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits',
                raw_token_ids=[1,2,3,4,5],metadata_token_ids=[1,2,3,4],lexical_token_ids=[5],
                tokenizer=dict(sha256=self.reviewed['tokenizer_file']['sha256']),
                parser_contract_sha256=self.reviewed['parser_source']['sha256'])
            return dict(raw_text='invented'),dict(token_binding=token,rich_postprocess_applied=False,use_itn=False,input_binding=binding)
        self.invoke(cap,adapter)
        self.assertEqual(events,['begin','start','forward','complete'])
    def test_strict_requires_preparation_and_identity_map(self):
        with self.assertRaisesRegex(ValueError,'pre-frozen'):c.BoundSenseCapture(self.runtime,self.root/'out',self.paths['parser_source'],self.reviewed,2,strict_sink=object())

if __name__=='__main__':unittest.main()
