"""Stdlib fake-operation integration tests. Never imports a model framework."""
from pathlib import Path
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import safe_diagnostic as diagnostic

def require(ok,why):
    if not ok:raise ValueError(why)

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name);self.events=[];self.fail_at=None
        self.identity={'native_decode_config_sha256':'a'*64,'native_decode_config':{
            'source_lock_sha256':'b'*64,'asset_lock_sha256':'c'*64}}
        def operation(name,result=None):
            def run(*args,**kwargs):
                self.events.append((name,json.loads((self.out/diagnostic.FILENAME).read_text())['stage']))
                if self.fail_at==name:raise KeyError('PRIVATE SECRET MUST NOT APPEAR')
                return result
            return run
        self.op=operation
        class Module:
            def _call_impl(self):return 'original'
            def forward(self):return 'original'
        self.module_class=Module;self.original=Module.forward
        evidence=types.SimpleNamespace(infer_qwen_once_after_approval=lambda:None,infer_sense_once_after_approval=lambda:None)
        self.legacy=types.SimpleNamespace(THREAD_ENV={},OFFLINE_ENV={},verify_scope=operation('scope'),
            prepare_caches=operation('caches'),verify_sources=operation('sources'),import_helpers=lambda:(None,evidence))
        self.sense=types.SimpleNamespace(read_object=operation('review',{}),source_closure=operation('closure',{}),
            strict_identity=operation('identity',{'identity':self.identity}))
        def module(path,name):return self.sense if name=='zero_audio_sense' else self.legacy
        qworker=types.ModuleType('qwen_worker');qworker.module=module;qworker.require=require;qworker.source_bodies=operation('qwen_closure',{})
        class AutoModel:
            def generate(self):raise AssertionError('never execute')
            def inference(self):raise AssertionError('never execute')
        class Qwen:
            def transcribe(self):raise AssertionError('never execute')
        class GenerationMixin:
            def generate(self):raise AssertionError('never execute')
        def whisper_prepare(*args,**kwargs):
            operation('loader')();kwargs['on_loaded']();operation('identity')()
            return types.SimpleNamespace(identity=self.identity)
        whisper=types.SimpleNamespace(decode=lambda:None,transcribe=lambda:None,
            decoding=types.SimpleNamespace(DecodingTask=type('DecodingTask',(),{'run':lambda self:None})))
        modules={'qwen_worker':qworker,'torch':types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module)),
            'setup_adapter':types.SimpleNamespace(verify_runtime=operation('runtime')),
            'sense_adapter':types.SimpleNamespace(infer_sense_once_after_approval=lambda:None),
            'sense_binding':types.SimpleNamespace(_contract=lambda:{'SENSE_GROUPS':[]}),
            'funasr':types.SimpleNamespace(AutoModel=AutoModel),'qwen_asr':types.SimpleNamespace(Qwen3ASRModel=Qwen),
            'transformers':types.ModuleType('transformers'),'transformers.generation':types.ModuleType('transformers.generation'),
            'transformers.generation.utils':types.SimpleNamespace(GenerationMixin=GenerationMixin),
            'capture_worker':types.SimpleNamespace(prepare_strict_identity=operation('identity',{'identity':self.identity})),
            'whisper':whisper,'whisper_capture':types.SimpleNamespace(verify_prepare_imports=operation('runtime'),prepare=whisper_prepare)}
        mp=patch.dict(sys.modules,modules);mp.start();self.addCleanup(mp.stop)
        spec=importlib.util.spec_from_file_location('tested_overlay_prepare',HERE/'prepare_worker.py')
        self.worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.worker)
        def loader(legacy,family,runtime,callback):operation('loader')();return callback(object())
        self.worker.primary_loader=loader
    def execute(self,family='sensevoice'):
        return self.worker.execute(family,'d'*40,Path('/invented-code'),Path('/invented-runtime'),self.out)
    def receipt(self):return json.loads((self.out/diagnostic.FILENAME).read_text())
    def test_all_families_true_stage_order(self):
        for family in ('qwen06','sensevoice','whisper'):
            with self.subTest(family=family):
                self.events=[]
                with tempfile.TemporaryDirectory() as d:
                    self.out=Path(d);self.execute(family);row=self.receipt()
                    self.assertEqual(row['status'],'complete')
                    self.assertIn(('loader','load_model'),self.events)
                    self.assertIn(('identity','prepare_native_identity'),self.events)
                    self.assertIs(self.module_class.forward,self.original)
    def test_scope_error_reported_without_text(self):
        self.fail_at='scope'
        with self.assertRaises(KeyError):self.execute()
        self.assertEqual(self.receipt()['stage'],'verify_scope')
        self.assertNotIn('PRIVATE',(self.out/diagnostic.FILENAME).read_text())
    def test_runtime_error_stage(self):
        self.fail_at='runtime'
        with self.assertRaises(KeyError):self.execute()
        self.assertEqual(self.receipt()['stage'],'verify_runtime')
    def test_source_review_fails_before_loader(self):
        self.fail_at='closure'
        with self.assertRaises(KeyError):self.execute()
        self.assertEqual(self.receipt()['stage'],'review_native_sources')
        self.assertFalse(any(name=='loader' for name,stage in self.events))
        self.assertIs(self.module_class.forward,self.original)
    def test_loader_failure_restores_guard(self):
        self.fail_at='loader'
        with self.assertRaises(KeyError):self.execute()
        self.assertEqual(self.receipt()['stage'],'load_model')
        self.assertIs(self.module_class.forward,self.original)
    def test_identity_failure_stops_no_proof(self):
        self.fail_at='identity'
        with self.assertRaises(KeyError):self.execute()
        self.assertEqual(self.receipt()['stage'],'prepare_native_identity')
        self.assertFalse((self.out/'zero-forward-proof.json').exists())
        self.assertIs(self.module_class.forward,self.original)
    def test_guard_restoration_failure_cannot_complete(self):
        original=self.worker.ForwardFence.__exit__
        def broken(fence,*args):original(fence,*args);raise RuntimeError('private restoration failure')
        with patch.object(self.worker.ForwardFence,'__exit__',broken):
            with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.receipt()['status'],'failed')
        self.assertEqual(self.receipt()['stage'],'write_preparation_proof')
    def test_proof_write_failure_cannot_complete(self):
        original=self.worker.write
        def fail(path,value):
            if path.name=='zero-forward-proof.json':raise OSError('private write failure')
            original(path,value)
        with patch.object(self.worker,'write',fail):
            with self.assertRaises(OSError):self.execute()
        self.assertEqual(self.receipt()['status'],'failed')
        self.assertEqual(self.receipt()['stage'],'write_preparation_proof')
    def test_guarded_forward_attempt_fails_before_proof(self):
        def loaded(*args):self.module_class().forward()
        self.worker.primary_loader=loaded
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.receipt()['stage'],'load_model')
        self.assertIs(self.module_class.forward,self.original)
    def test_python38_syntax(self):
        for path in HERE.rglob('*.py'):ast.parse(path.read_text(),feature_version=(3,8))

if __name__=='__main__':unittest.main()
