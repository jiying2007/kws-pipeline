"""Anonymous PCM and fake recognizer tests only."""
from contextlib import ExitStack
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import blind_pcm as pcm
import blind_asr_worker as asr


def fixture(root, count=16):
    root=Path(root);root.mkdir();clips=[]
    for i in range(count):
        raw=struct.pack('<16h',*[(i+1)*j for j in range(1,17)]);digest=pcm.sha(raw)
        (root/(digest+'.pcm')).write_bytes(raw)
        clips.append(dict(opaque_id=digest,file=digest+'.pcm',sha256=digest,bytes=len(raw),frames=16,sample_rate_hz=16000,channels=1,sample_format='s16le'))
    job=dict(schema=pcm.JOB_SCHEMA,clips=sorted(clips,key=lambda x:x['opaque_id']))
    pcm.write_once(root/'job.json',job)
    return job

class FakeModel:
    calls=0
    result={'raw_text':'你好小窝','completeness':'complete','quality_flags':[]}
    def __init__(self,*args):self.receipt={'test_only':True}
    def decode(self,*args):
        type(self).calls+=1
        return copy.deepcopy(type(self).result), {'private_unreviewed':'CANARY_MUST_NOT_EXPORT'}

def fake_run(root,recognizer='qwen_asr',result=None):
    root=Path(root);inp=root/'input';fixture(inp);out=root/'output';out.mkdir();scratch=root/'scratch';scratch.mkdir()
    FakeModel.calls=0;FakeModel.result=result or {'raw_text':'你好小窝','completeness':'complete','quality_flags':[]}
    with ExitStack() as stack:
        for name,value in [('verify_sources',{}),('verify_scope',{'schema':'fake-scope'}),('prepare_environment',None),('verify_runtime',{'fake':True})]:
            stack.enter_context(patch.object(asr,name,return_value=value))
        stack.enter_context(patch.object(asr,'SCRATCH',scratch))
        stack.enter_context(patch.object(asr,'RetainedBackend',FakeModel))
        summary=asr.run(recognizer,inp,out,root/'runtime')
    return inp,out,summary

class BlindTests(unittest.TestCase):
    def test_all_three_recognizers_fixed_sixteen(self):
        for name in asr.RECOGNIZERS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                inp,out,summary=fake_run(tmp,name)
                self.assertEqual(FakeModel.calls,16);self.assertEqual(summary['success'],16)
                rows=asr.recover_outcomes(inp,out,name)
                self.assertEqual(len(rows),16)
                for path in out.glob('cells/*/decoder-evidence.json'):
                    self.assertNotIn(b'CANARY_MUST_NOT_EXPORT',path.read_bytes())
    def test_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp,out,summary=fake_run(tmp,result={'raw_text':'你好小窝','completeness':'incomplete','quality_flags':[]})
            self.assertEqual(FakeModel.calls,1);self.assertEqual(summary['attempted'],1);self.assertEqual(summary['not_run'],15)
            with self.assertRaises(ValueError):asr.run('qwen_asr',inp,out,Path(tmp)/'runtime')
    def test_no_producer_fields_in_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'input';job=fixture(inp)
            for key,value in [('speaker','Eric'),('target_text','你好小窝'),('seed',2026101401)]:
                bad=copy.deepcopy(job);bad['clips'][0][key]=value
                with self.assertRaises(ValueError):pcm.validate_job(pcm.canonical(bad))
            (inp/'plan.json').write_text('{}')
            with self.assertRaises(ValueError):pcm.bind_input(inp)
    def test_geometry_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=fixture(Path(tmp)/'input',1)
            for key in ('frames','bytes','channels','sample_rate_hz'):
                for value in (True,1.0,'1'):
                    bad=copy.deepcopy(job);bad['clips'][0][key]=value
                    with self.assertRaises(ValueError):pcm.validate_job(pcm.canonical(bad))
    def test_exact_result_schema(self):
        for bad in ({'raw_text':None,'completeness':'complete','quality_flags':[]},{'raw_text':'x','completeness':'complete','quality_flags':False},{'raw_text':'x','completeness':'complete','quality_flags':[],'target':'x'}):
            with self.assertRaises(ValueError):asr.validate_result(bad)
if __name__=='__main__':unittest.main()
