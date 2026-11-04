"""Invented stdlib-only no-forward guard and no-audio host tests."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from zero_forward import ForwardFence,primary_loader
import prepare_host

class ZeroForwardTests(unittest.TestCase):
    def fixture(self):
        class Module:
            def _call_impl(self):return self.forward()
            def __call__(self):return self._call_impl()
        class Model(Module):
            def forward(self):return 'would infer'
            def generate(self):return 'would generate'
        return NS(nn=NS(Module=Module)),Model
    def test_module_call_refused_and_restored(self):
        torch,Model=self.fixture();original=Model.forward
        with ForwardFence(torch) as fence:
            with self.assertRaises(RuntimeError):Model()()
            with self.assertRaises(RuntimeError):fence.assert_zero()
        self.assertIs(Model.forward,original);self.assertEqual(Model()(),'would infer')
    def test_direct_forward_and_generate_refused(self):
        for name in ('forward','generate'):
            torch,Model=self.fixture()
            with ForwardFence(torch) as fence:
                with self.assertRaises(RuntimeError):getattr(Model(),name)()
                self.assertEqual(len(fence.attempts),1)
    def test_public_function_refused(self):
        torch,Model=self.fixture();module=NS(infer=lambda:'bad')
        with ForwardFence(torch,[(module,'infer')]) as fence:
            with self.assertRaises(RuntimeError):module.infer()
        self.assertEqual(module.infer(),'bad')
    def test_no_forward_prepare_passes(self):
        torch,Model=self.fixture()
        with ForwardFence(torch) as fence:
            loaded=Model();fence.assert_zero();self.assertTrue(fence.active)
        self.assertFalse(fence.active)
    def test_exception_restores(self):
        torch,Model=self.fixture();original=Model.generate
        with self.assertRaises(KeyError):
            with ForwardFence(torch):raise KeyError('load fail')
        self.assertIs(Model.generate,original)
    def test_static_and_class_entrypoints(self):
        torch,Model=self.fixture()
        class API:
            @staticmethod
            def decode():return 1
            @classmethod
            def transcribe(cls):return 2
        with ForwardFence(torch,[(API,'decode'),(API,'transcribe')]):
            with self.assertRaises(RuntimeError):API.decode()
            with self.assertRaises(RuntimeError):API.transcribe()
        self.assertEqual((API.decode(),API.transcribe()),(1,2))
    def primary_fixture(self,fail=None):
        import hashlib
        calls=[]
        canonical=lambda x:json.dumps(x,sort_keys=True).encode()
        digest=lambda raw:hashlib.sha256(raw).hexdigest()
        schema={'actual':'expected'}
        lock=dict(asset_lock={'files':'actual'},source_lock={'source':'actual'},asset_lock_sha256='a'*64,
                  source_lock_sha256='b'*64,expected_schema_sha256=digest(canonical(schema)))
        def checked(name):
            def call(*args):
                calls.append(name)
                if fail==name:raise ValueError(name)
                return {'checked':name}
            return call
        def require(ok,why):
            if not ok:raise ValueError(why)
        legacy=NS(RETAINED=Path('/retained'),MODEL_IDS={'qwen06':'Qwen'},decode=json.loads,
            read_regular=lambda *args:canonical({'qwen06':lock}),canonical=canonical,digest=digest,
            canonical_sha=lambda x:digest(canonical(x)),verify_scope=checked('scope'),require=require,
            qwen06_schema=lambda:schema,verify_assets=checked('assets'),require_runtime_versions=checked('runtime'),
            verify_source_members=checked('source'),_preload=object())
        def actual_loader(root,assets,assetsha,sources,sourcesha,declaration,declsha,raw):
            calls.append('load_body')
            self.assertEqual(declaration['candidate_count'],0);self.assertEqual(declaration['audio_inputs'],0)
            self.assertIs(declaration['inference_allowed'],False);self.assertEqual(raw,b'')
            prepared=legacy._preload('Qwen',root,assets,assetsha,sources,sourcesha,declaration,declsha,raw)
            self.assertEqual(prepared[-1],{})
            return NS(decoder_inputs={})
        legacy.load_qwen_after_approval=actual_loader
        return legacy,calls
    def test_candidate_zero_original_checks_before_prepare(self):
        legacy,calls=self.primary_fixture();original=legacy._preload
        result=primary_loader(legacy,'qwen06','/runtime',lambda loaded:calls.append('prepare') or 'identity')
        self.assertEqual(result,'identity')
        self.assertEqual(calls,['load_body','scope','assets','runtime','source','prepare'])
        self.assertIs(legacy._preload,original)
    def test_each_primary_guard_failure_restores_without_prepare(self):
        for failure in ('scope','assets','runtime','source'):
            legacy,calls=self.primary_fixture(failure);original=legacy._preload
            with self.assertRaises(ValueError):primary_loader(legacy,'qwen06','/runtime',lambda loaded:calls.append('prepare'))
            self.assertNotIn('prepare',calls);self.assertIs(legacy._preload,original)
    def test_primary_callback_failure_restores(self):
        legacy,calls=self.primary_fixture();original=legacy._preload
        def bad(loaded):raise RuntimeError('capture failed')
        with self.assertRaises(RuntimeError):primary_loader(legacy,'qwen06','/runtime',bad)
        self.assertIs(legacy._preload,original)
    def test_expired_launch_never_calls_capacity(self):
        class Boundary:
            def host_capacity(self):raise AssertionError('no Docker')
        with patch.object(prepare_host.time,'monotonic',return_value=100):
            with self.assertRaises(ValueError):prepare_host.launch(Boundary(),None,None,None,None,None,None,219)
    def test_no_audio_geometry_and_deadline(self):
        calls=[]
        class Boundary:
            def host_capacity(self):pass
            def container_args(self,*args):
                calls.append(args);return ['docker','image','oldcommand']
            def supervise_container(self,*args):calls.append(args)
        with tempfile.TemporaryDirectory() as d,patch.object(prepare_host.time,'monotonic',return_value=100):
            prepare_host.launch(Boundary(),Path('code'),Path('runtime'),Path(d)/'out','image','test',['new'],1800)
        self.assertEqual(calls[0][-3:],('tts','tts',None))
        self.assertEqual(calls[1][0][-1],'new');self.assertEqual(calls[1][-1],1300)
    def test_container_diagnostics_retained_per_family_even_failure(self):
        class Boundary:
            def host_capacity(self):pass
            def container_args(self,*args):return ['docker','image','old']
            def supervise_container(self,args,name,output,kind,deadline):
                (output.parent/('container-'+kind+'.json')).write_text(name)
                if name=='second':raise RuntimeError('controlled failure')
        with tempfile.TemporaryDirectory() as d,patch.object(prepare_host.time,'monotonic',return_value=100):
            root=Path(d)
            prepare_host.launch(Boundary(),None,None,root/'first','image','first',['new'],1800)
            with self.assertRaises(RuntimeError):prepare_host.launch(Boundary(),None,None,root/'second','image','second',['new'],1800)
            self.assertEqual((root/'first/container-private.json').read_text(),'first')
            self.assertEqual((root/'second/container-private.json').read_text(),'second')
    def test_export_drops_all_payload(self):
        proof=dict(family='qwen06',status='prepared_zero_forward',candidate_count=0,audio_inputs=0,
            observed_guarded_entrypoint_attempts=0,native_config_sha256='a'*64,source_lock_sha256='b'*64,
            asset_lock_sha256='c'*64,raw_text='NEVER',private_file='SECRET')
        with tempfile.TemporaryDirectory() as d:
            prepare_host.export(Path(d),'stopped_no_retry',{'qwen06':proof})
            raw=(Path(d)/'aggregate-only/status.json').read_text()
            self.assertNotIn('NEVER',raw);self.assertNotIn('SECRET',raw)
            self.assertEqual(len(json.loads(raw)['models']),3)
    def test_export_rejects_forward_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):prepare_host.export(Path(d),'stopped_no_retry',{'qwen06':dict(
                family='qwen06',status='prepared_zero_forward',candidate_count=0,audio_inputs=0,
                observed_guarded_entrypoint_attempts=1)})

if __name__=='__main__':unittest.main()
