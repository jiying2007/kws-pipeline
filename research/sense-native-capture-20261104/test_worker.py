"""Two-phase worker orchestration tests; invented loaded objects only."""
from pathlib import Path
import ast
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import worker as w
import test_sense_capture as fixtures

class Tests(unittest.TestCase):
    setUp=fixtures.Tests.setUp
    tearDown=fixtures.Tests.tearDown
    def ready(self):
        self.runtime.wrapper.kwargs['batch_size']=1
        self.runtime.wrapper.model.parameters=lambda:[NS(device='cpu',dtype='torch.float32')]
        self.tokenizer.sp.get_piece_size=lambda:25055;self.tokenizer.sp.unk_id=lambda:0
        self.runtime.decoder_inputs={'clip-000001':{}}
        self.runtime.receipt=dict(assets=dict(model_id='FunAudioLLM/SenseVoiceSmall',revision='3847d57b6bdf2dd8875cb1508d2af43d80a16bf7',verified_files=[dict(filename='model.pt',bytes=123,sha256='a'*64)]))
        self.review=dict(executed_head_sha='f'*40,dependencies=self.reviewed)
    def test_actual_zero_forward_identity_preparation(self):
        self.ready()
        result=w.strict_identity(self.runtime,self.root/'prepare',self.review,{'code.py':b'original source'},self.paths['parser_source'],[[],[],[],[]])
        self.assertEqual(result['identity']['model_family'],'sensevoice')
        self.assertEqual(self.original_calls,0)
        self.assertTrue((self.root/'prepare/identity.json').exists())
    def fake_legacy(self,do_infer=False):
        self.ready();out=self.root/'output';out.mkdir()
        loader=lambda *a,**k:self.runtime
        adapter=NS(infer_sense_once_after_approval=lambda *a:({'raw_text':'invented'},{}))
        legacy=NS(load_sense_after_approval=loader)
        def run(model,*args):
            (out/'sensevoice').mkdir()
            loaded=legacy.load_sense_after_approval()
            self.assertTrue(do_infer,'prepare must have escaped before forward boundary')
            adapter.infer_sense_once_after_approval(loaded,self.bound,'e'*64,self.root,{})
            return dict(status='complete')
        legacy.run=run
        return out,legacy,adapter,loader
    def test_prepare_sentinel_and_exact_restore(self):
        out,legacy,adapter,loader=self.fake_legacy()
        original=adapter.infer_sense_once_after_approval
        result=w.execute('prepare',legacy,adapter,self.review,{'code.py':b'original'},self.paths['parser_source'],[[],[],[],[]],self.root,out,self.root)
        self.assertEqual(result['status'],'prepared_zero_forward');self.assertEqual(self.original_calls,0)
        self.assertIs(legacy.load_sense_after_approval,loader);self.assertIs(adapter.infer_sense_once_after_approval,original)
    def test_infer_requires_frozen_policy_before_loader(self):
        out,legacy,adapter,loader=self.fake_legacy()
        with self.assertRaisesRegex(ValueError,'prefrozen'):w.execute('infer',legacy,adapter,self.review,{},self.paths['parser_source'],[],self.root,out,self.root)
        self.assertFalse((out/'sensevoice').exists())
    def test_infer_reprepares_then_binds_sink(self):
        out,legacy,adapter,loader=self.fake_legacy(True);events=[]
        class Sink:
            def __init__(self,*args):
                events.append('sink');Path(args[0]).mkdir()
        class Capture:
            def __init__(self,*args):events.append('capture')
            def infer(self,*args):events.append('inference');return {},{}
            def restore(self):events.append('restore')
        with patch.object(w,'BoundSenseCapture',Capture),patch.object(w,'persist_native_dependencies'):
            result=w.execute('infer',legacy,adapter,self.review,{'code.py':b'original'},self.paths['parser_source'],[[],[],[],[]],self.root,out,self.root,b'prefrozen policy',{'clip-000001':{}},Sink)
        self.assertEqual(result['status'],'complete');self.assertEqual(events,['sink','capture','inference','restore'])
        self.assertTrue((out/'sensevoice/prepare/proposed-native-config.json').is_file())
        self.assertEqual((out/'sensevoice/strict-native/code.py').read_bytes(),b'original')
    def test_review_closure_missing_or_unmounted_rejected(self):
        with self.assertRaises(ValueError):w.source_closure({})
        review=dict(schema='reviewed-sense-runtime-capture-v1',executed_head_sha='f'*40,dependencies=self.reviewed,sources=[dict(path='/private/secret',member='x',bytes=1,sha256='a'*64)])
        with self.assertRaisesRegex(ValueError,'fixed mounted root'):w.source_closure(review)
    def test_identity_map_exact_audio_association(self):
        path=self.root/'map.json';job=dict(clips=[dict(audio_id='clip-000001',wav_sha256='a'*64)])
        value=dict(schema='blind-asr-source-request-map-v1',clips=[dict(audio_id='clip-000001',source_id='synthetic',request_id='request',wav_sha256='a'*64)])
        path.write_text(json.dumps(value));self.assertEqual(w.identity_map(path,job),{'clip-000001':dict(source_id='synthetic',request_id='request')})
        value['clips'][0]['wav_sha256']='b'*64;path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):w.identity_map(path,job)
    def test_policy_raw_never_created_from_results(self):
        source=Path(w.__file__).read_text()
        self.assertIn("regular(policy/'recognizer-policy.json'",source)
        self.assertNotIn("write_once(policy",source)
    def test_python38_syntax(self):
        for name in ('worker.py','test_worker.py'):
            ast.parse((Path(__file__).parent/name).read_text(),feature_version=(3,8))

    def test_private_source_member_exact_bytes_no_overwrite(self):
        root=self.root/'strict';root.mkdir()
        w.persist_source_members(root,{'src/code.py':b'exact original'})
        self.assertEqual((root/'src/code.py').read_bytes(),b'exact original')
        with self.assertRaises(FileExistsError):w.persist_source_members(root,{'src/code.py':b'changed'})
        with self.assertRaises(ValueError):w.persist_source_members(root,{'../escape.py':b'bad'})

    def test_native_dependency_members_match_original_inventories(self):
        self.ready();sources=[];bodies={}
        for name,path in self.paths.items():
            raw=path.read_bytes()
            if name in ('cmvn_file','tokenizer_file'):
                self.runtime.receipt['assets']['verified_files'].append(dict(filename=path.name,bytes=len(raw),sha256=w.sha(raw)))
            else:
                member='source/'+name;sources.append(dict(path=str(path),member=member,bytes=len(raw),sha256=w.sha(raw)));bodies[member]=raw
        self.review['sources']=sources
        prepared=w.strict_identity(self.runtime,self.root/'prepare',self.review,bodies,self.paths['parser_source'],[[],[],[],[]])
        strict=self.root/'strict';strict.mkdir();w.persist_source_members(strict,bodies)
        metadata=w.persist_native_dependencies(strict,prepared,self.review)
        self.assertIsNone(metadata['archive_id'])
        self.assertEqual(metadata['dependencies']['tokenizer_file']['member'],'tokenizer_file')
        self.assertEqual(metadata['dependencies']['parser_source']['member'],'source/parser_source')
        self.assertEqual(metadata['dependencies']['frontend_observation']['member'],'frontend-observation.json')
        self.assertEqual((strict/'tokenizer_file').read_bytes(),self.paths['tokenizer_file'].read_bytes())
        self.assertNotIn('frontend-observation.json',bodies)

    def package_fixture(self):
        names=['funasr/auto/auto_model.py','funasr/models/sense_voice/model.py',
            'funasr/frontends/wav_frontend.py','funasr/utils/load_utils.py','funasr/utils/fbank.py',
            'funasr/models/ctc/ctc.py','funasr/tokenizer/sentencepiece_tokenizer.py']
        raw={n:('invented original '+n).encode() for n in names}
        rows=[dict(distribution='funasr',relative_path=n,sha256=w.sha(raw[n])) for n in names]
        model=dict(sensevoice=dict(source_lock=rows,source_lock_sha256=w.sha(w.canonical(rows))))
        lock=w.canonical(model)
        review=dict(sources=[dict(path='/code/research/qwen6_asr/model-locks.json',member='model-locks.json',bytes=len(lock),sha256=w.sha(lock))])
        bodies={'model-locks.json':lock}
        return raw,review,bodies
    def test_original_package_sources_verified_and_preserved(self):
        raw,review,bodies=self.package_fixture()
        dist=NS(locate_file=lambda n:Path('/runtime/venv/lib/python3.12/site-packages')/n)
        with patch('importlib.metadata.distribution',return_value=dist),patch.object(w,'regular',side_effect=lambda p,m:raw[str(p).split('site-packages/')[1]]):
            w.include_retained_package_sources(review,bodies)
        self.assertEqual(len(bodies),8)
        for name,body in raw.items():self.assertEqual(bodies['runtime-source/funasr/'+name],body)
    def test_original_package_overlap_exact_deduplication(self):
        raw,review,bodies=self.package_fixture();name='funasr/frontends/wav_frontend.py'
        path='/runtime/venv/lib/python3.12/site-packages/'+name
        review['sources'].append(dict(path=path,member='reviewed/frontend.py',bytes=len(raw[name]),sha256=w.sha(raw[name])))
        bodies['reviewed/frontend.py']=raw[name]
        dist=NS(locate_file=lambda n:Path('/runtime/venv/lib/python3.12/site-packages')/n)
        with patch('importlib.metadata.distribution',return_value=dist),patch.object(w,'regular',side_effect=lambda p,m:raw[str(p).split('site-packages/')[1]]):
            w.include_retained_package_sources(review,bodies)
        self.assertEqual(len(bodies),8);self.assertNotIn('runtime-source/funasr/'+name,bodies)
    def test_original_package_drift_rejected(self):
        raw,review,bodies=self.package_fixture()
        dist=NS(locate_file=lambda n:Path('/runtime/venv/lib/python3.12/site-packages')/n)
        with patch('importlib.metadata.distribution',return_value=dist),patch.object(w,'regular',return_value=b'drift'):
            with self.assertRaisesRegex(ValueError,'source bytes changed'):w.include_retained_package_sources(review,bodies)
    def test_original_lock_missing_rejected(self):
        with self.assertRaisesRegex(ValueError,'original Sense model lock'):w.include_retained_package_sources({'sources':[]},{})
    def test_isolated_cli_help_returns_zero(self):
        import subprocess
        result=subprocess.run([sys.executable,'-I','-B',str(Path(w.__file__).resolve()),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--phase',result.stdout);self.assertNotIn('stopped_no_retry',result.stdout)

if __name__=='__main__':unittest.main()
