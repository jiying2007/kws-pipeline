"""Standalone stdlib-only composition tests. No ML imports or real models."""
from pathlib import Path
import ast
import importlib.util
import inspect
import json
import sys
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'sense'),str(ROOT)]
import zero_forward as z
import sense_capture as c
fixture_path=ROOT.parent/'sense-native-capture-20261104/test_sense_capture.py'
spec=importlib.util.spec_from_file_location('retained_sense_fixtures',fixture_path)
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)

class Tests(unittest.TestCase):
    setUp=fixtures.Tests.setUp
    tearDown=fixtures.Tests.tearDown
    def actual_fixture(self):
        class Module:
            def _call_impl(self):return self.forward()
            def __call__(self):return self._call_impl()
        file=self.paths['frontend_source'];file.write_text('def forward(self):\n    raise AssertionError("never run fixture forward")\n')
        fbankfile=self.paths['fbank_source'];fbankfile.write_text('def fbank():\n    raise AssertionError("never run fixture fbank")\n')
        namespace={};exec(compile(fbankfile.read_text(),str(fbankfile),'exec'),namespace)
        old=self.paths['backend_binary'];new=self.root/'extension.so';new.write_bytes(old.read_bytes());self.paths['backend_binary']=new
        core=self.root/'lib/libkaldi-native-fbank-core.so';core.parent.mkdir();core.write_bytes(self.paths['backend_core_library'].read_bytes());self.paths['backend_core_library']=core
        binary=types.ModuleType('invented_fence_binary');binary.__file__=str(new)
        backend=types.SimpleNamespace(__file__=str(self.paths['backend_package_source']),OnlineFbank=type('OnlineFbank',(),{'__module__':'invented_fence_binary'}))
        kaldi=types.SimpleNamespace(_HAS_TORCHAUDIO=False,_knf=backend,fbank=namespace['fbank'])
        binary_patch=patch.dict(sys.modules,{'invented_fence_binary':binary});binary_patch.start();self.addCleanup(binary_patch.stop)
        globals_={'kaldi':kaldi};exec(compile(file.read_text(),str(file),'exec'),globals_)
        klass=type('WavFrontend',(fixtures.WavFrontend,Module),{'forward':globals_['forward'],'__module__':'funasr.frontends.wav_frontend'})
        self.frontend=klass();self.frontend.cmvn_file=str(self.paths['cmvn_file'])
        for key,path in self.paths.items():self.reviewed[key]=dict(bytes=path.stat().st_size,sha256=c.sha(path.read_bytes()))
        self.runtime.wrapper.kwargs.update(frontend=self.frontend,batch_size=1)
        self.runtime.wrapper.model.parameters=lambda:[types.SimpleNamespace(device='cpu',dtype='torch.float32')]
        self.tokenizer.sp.get_piece_size=lambda:25055;self.tokenizer.sp.unk_id=lambda:0
        return types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module)),binary,kaldi
    def test_baseline_composition_reproduces_keyerror_no_forward(self):
        torch,binary,kaldi=self.actual_fixture()
        self.path_patch.stop()
        try:
            with z.ForwardFence(torch) as fence:
                with self.assertRaisesRegex(KeyError,'kaldi'):c.dependency_paths(self.frontend,self.paths['tokenizer_file'],self.paths['parser_source'])
                self.assertEqual(fence.attempts,[])
        finally:self.path_patch.start()
    def test_registered_metadata_keeps_original_native_source(self):
        torch,binary,kaldi=self.actual_fixture();original=self.frontend.forward.__func__
        self.path_patch.stop()
        try:
            with patch.dict(sys.modules,{'invented_fence_binary':binary}),z.ForwardFence(torch) as fence:
                paths=c.dependency_paths(self.frontend,self.paths['tokenizer_file'],self.paths['parser_source'],fence.method_metadata)
                self.assertEqual(paths,self.paths)
                self.assertEqual(fence.attempts,[])
                self.assertIsNot(self.frontend.forward.__func__,original)
                with self.assertRaises(RuntimeError):self.frontend.forward()
        finally:self.path_patch.start()
        self.assertIs(self.frontend.forward.__func__,original)
    def test_real_prepare_loaded_composition_no_forward(self):
        torch,binary,kaldi=self.actual_fixture();self.path_patch.stop()
        try:
            with patch.dict(sys.modules,{'invented_fence_binary':binary}),z.ForwardFence(torch) as fence:
                result=c.prepare_loaded(self.runtime,self.root/'prepared',self.paths['parser_source'],self.reviewed,
                    b'original-source-lock',b'original-model-lock',192000,[[],[],[],[]],fence.method_metadata)
                fence.assert_zero()
                self.assertEqual(result['forward_calls'],0)
                self.assertEqual(result['native_config']['effective_config']['frontend_config_sha256'],result['frontend_snapshot_sha256'])
                self.assertEqual(result['dependencies']['frontend_source']['sha256'],self.reviewed['frontend_source']['sha256'])
        finally:self.path_patch.start()
    def test_arbitrary_wrapped_chain_not_trusted(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            def forged(self):pass
            forged.__wrapped__=self.frontend.forward.__func__
            fake=types.MethodType(forged,self.frontend)
            with self.assertRaisesRegex(ValueError,'not uniquely registered'):fence.method_metadata(fake,('kaldi',))
    def test_mutated_wrapper_wrapped_attribute_ignored(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            self.frontend.forward.__func__.__wrapped__=lambda:None
            metadata=fence.method_metadata(self.frontend.forward,('kaldi',))
            self.assertTrue(all(type(v) is str for v in metadata.values()))
            self.assertEqual(metadata['source_file'],str(self.paths['frontend_source']))
    def test_original_wrapped_attribute_ignored(self):
        torch,binary,kaldi=self.actual_fixture()
        self.frontend.forward.__func__.__wrapped__=lambda:None
        with z.ForwardFence(torch) as fence:self.assertEqual(fence.method_metadata(self.frontend.forward,('kaldi',))['fbank_source'],str(self.paths['fbank_source']))
    def test_cross_fence_capability_rejected(self):
        torch,binary,kaldi=self.actual_fixture()
        other=types.SimpleNamespace(nn=types.SimpleNamespace(Module=type('OtherModule',(),{})))
        with z.ForwardFence(torch),z.ForwardFence(other) as fence:
            with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,('kaldi',))
    def test_stale_capability_rejected(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:bound=self.frontend.forward
        with self.assertRaises(ValueError):fence.method_metadata(bound,('kaldi',))
    def test_replaced_owner_guard_rejected(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            bound=self.frontend.forward;type(self.frontend).forward=lambda self:None
            with self.assertRaisesRegex(ValueError,'guard method changed'):fence.method_metadata(bound,('kaldi',))
    def test_missing_global_and_unbound_rejected(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,('missing',))
            with self.assertRaises(ValueError):fence.method_metadata(type(self.frontend).forward,('kaldi',))
    def test_duplicate_guard_registration_preserves_original(self):
        torch,binary,kaldi=self.actual_fixture();owner=type(self.frontend);original=owner.forward
        with z.ForwardFence(torch,[(owner,'forward')]) as fence:
            self.assertEqual(sum(x[0] is owner and x[1]=='forward' for x in fence.saved),1)
            self.assertEqual(fence.method_metadata(self.frontend.forward,('kaldi',))['fbank_source'],str(self.paths['fbank_source']))
        self.assertIs(owner.forward,original)
    def test_current_prepare_worker_passes_guard_capability(self):
        tree=ast.parse((ROOT/'prepare_worker.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='strict_identity']
        self.assertEqual(len(calls),1)
        self.assertTrue(any(k.arg=='forward_metadata' and isinstance(k.value,ast.Attribute) and
            k.value.attr=='method_metadata' and isinstance(k.value.value,ast.Name) and
            k.value.value.id=='fence' for k in calls[0].keywords))
    def test_python38_syntax(self):
        for path in [ROOT/'zero_forward.py',ROOT/'prepare_worker.py',ROOT/'sense/sense_capture.py',ROOT/'sense/worker.py']:
            ast.parse(path.read_text(),feature_version=(3,8))
    def test_no_model_packages_imported(self):
        for name in ('torch','numpy','funasr','qwen_asr','transformers'):
            self.assertNotIn(name,sys.modules)

    def test_fence_cannot_reenter_with_stale_metadata_registry(self):
        torch,binary,kaldi=self.actual_fixture();fence=z.ForwardFence(torch)
        with fence:fence.assert_zero()
        with self.assertRaisesRegex(ValueError,'single-use'):
            with fence:pass

    def test_only_fixed_kaldi_request_allowed(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            for keys in ((),('forward',),('kaldi','fbank'),['kaldi'],('kaldi','kaldi')):
                with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,keys)
    def test_callable_global_rejected(self):
        torch,binary,kaldi=self.actual_fixture();function=self.frontend.forward.__func__
        function.__globals__['kaldi']=lambda:None
        with z.ForwardFence(torch) as fence:
            with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,('kaldi',))
    def test_callable_backend_rejected(self):
        torch,binary,kaldi=self.actual_fixture();kaldi._knf=lambda:None
        with z.ForwardFence(torch) as fence:
            with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,('kaldi',))
    def test_callable_path_rejected(self):
        torch,binary,kaldi=self.actual_fixture();kaldi._knf.__file__=lambda:None
        with z.ForwardFence(torch) as fence:
            with self.assertRaises(ValueError):fence.method_metadata(self.frontend.forward,('kaldi',))
    def test_only_plain_strings_returned_no_global_objects(self):
        torch,binary,kaldi=self.actual_fixture()
        with z.ForwardFence(torch) as fence:
            result=fence.method_metadata(self.frontend.forward,('kaldi',))
            self.assertEqual(set(result),{'source_file','fbank_source','backend_package_source','backend_binary'})
            self.assertTrue(all(type(value) is str and not callable(value) for value in result.values()))

if __name__=='__main__':unittest.main()
