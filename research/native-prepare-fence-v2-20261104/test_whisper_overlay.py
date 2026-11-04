"""Invented no-ML fixtures for genuine load/config diagnostic transition."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('overlay_whisper',HERE/'whisper_capture.py')
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)

class OverlayWhisperTests(unittest.TestCase):
    def fixture(self,root,events):
        loaded=NS()
        def constructor(runtime,diagnostics):
            events.append('actual_loader')
            (diagnostics/'load-receipt.json').write_text('{"audio_calls":0}')
            return loaded
        adapter=NS(FixedWhisper=constructor)
        return adapter
    def execute(self,callback,fail_loader=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=[];adapter=self.fixture(root,events)
            if fail_loader:
                def failed(*args):events.append('actual_loader');raise RuntimeError('load failure')
                adapter.FixedWhisper=failed
            def config(*args):events.append('native_config');return {'invented':True}
            def on_loaded():events.append('stage_transition');callback()
            original_identity=w.identity
            def identity(path,*args):
                return {'bytes':1,'sha256':w.MODEL_SHA} if Path(path).name=='small.pt' else original_identity(path,*args)
            with patch.object(w,'_source_snapshot',return_value=({}, {w.TOKENIZER_MEMBER:b'tokenizer'})),\
                 patch.object(w,'_modules',return_value=(None,None,None,adapter,None)),\
                 patch.object(w,'identity',side_effect=identity),\
                 patch.object(w,'_config',side_effect=config):
                try:result=w.prepare(root,root,root/'out','a'*40,root/'sink.py',root,maximum=16,on_loaded=on_loaded)
                except RuntimeError:return events,None
                return events,result.identity
    def test_callback_is_after_real_loader_before_config(self):
        events,identity=self.execute(lambda:None)
        self.assertEqual(events,['actual_loader','stage_transition','native_config'])
        self.assertIsNotNone(identity)
    def test_failed_loader_never_claims_config_stage(self):
        events,identity=self.execute(lambda:None,True)
        self.assertEqual(events,['actual_loader']);self.assertIsNone(identity)
    def test_failed_callback_stops_before_config(self):
        def failure():raise RuntimeError('diagnostic transition failure')
        events,identity=self.execute(failure)
        self.assertEqual(events,['actual_loader','stage_transition']);self.assertIsNone(identity)
    def test_noncallable_rejected_before_source_or_loader(self):
        with patch.object(w,'_source_snapshot',side_effect=AssertionError('must not start')):
            with self.assertRaises(ValueError):w.prepare('/code','/runtime','/output','a'*40,'/sink','/tools',on_loaded=True)
    def test_actual_source_snapshot_binds_exact_diagnostic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'code';runtime=root/'runtime';runtime.mkdir()
            integration=repo/'research/native-asr-integration-20261104';integration.mkdir(parents=True)
            for name in ('prepare_worker.py','zero_forward.py','safe_diagnostic.py'):(integration/name).write_bytes(('exact '+name).encode())
            retained=repo/'research/source-screen32-whisper-v1/source-lock.json';retained.parent.mkdir(parents=True)
            retained.write_text(json.dumps(dict(source_revision='86098128c0b4f24f0e2aa2994de830614b474227',files=[])))
            capture=repo/'research/whisper-native-capture-20261104';capture.mkdir();(capture/'worker.py').write_bytes(b'fixed worker')
            (capture/'whisper_capture.py').write_bytes((HERE/'whisper_capture.py').read_bytes())
            sink=repo/'research/qwen-native-capture-20261104';sink.mkdir()
            for name in ('native_sink.py','generation_capture.py'):(sink/name).write_bytes(b'fixed '+name.encode())
            tools=repo/'tools';tools.mkdir()
            for name in ('speech_asr_consensus.py','speech_label_admission.py'):(tools/name).write_bytes(b'fixed '+name.encode())
            for name in ('installed-inventory.json','asset-lock.json'):(runtime/name).write_bytes(b'{}')
            pins={retained.relative_to(repo).as_posix():w.sha(retained.read_bytes())}
            with patch.object(w,'PINS',pins),patch.object(w,'HERE',capture),patch.object(w,'__file__',str(capture/'whisper_capture.py')):
                paths,bodies=w._source_snapshot(repo,runtime,sink/'native_sink.py',tools)
                self.assertEqual(bodies['bindings/capture/safe_diagnostic.py'],b'exact safe_diagnostic.py')
                self.assertEqual(paths['bindings/capture/safe_diagnostic.py'],integration/'safe_diagnostic.py')
                (integration/'safe_diagnostic.py').unlink()
                with self.assertRaises(FileNotFoundError):w._source_snapshot(repo,runtime,sink/'native_sink.py',tools)
    def test_diagnostic_source_is_required_in_snapshot(self):
        import ast
        source=(HERE/'whisper_capture.py').read_text();tree=ast.parse(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_source_snapshot')
        tuples=[n for n in ast.walk(function) if isinstance(n,ast.Tuple)]
        self.assertTrue(any([getattr(v,'value',None) for v in n.elts]==['prepare_worker.py','zero_forward.py','safe_diagnostic.py'] for n in tuples))

if __name__=='__main__':unittest.main()
