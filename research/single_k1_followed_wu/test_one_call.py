"""Pure mock checks: forbid inference imports, no model load or synthesis."""
import builtins
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'onnxruntime','onnx','torch','transformers','melo'}:
        raise AssertionError('INFERENCE_IMPORT_FORBIDDEN')
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
import numpy as np
import run_single_melo as runner
FIXTURES=json.loads((runner.ROOT/'frozen-one-input.json').read_bytes())['fixtures']
PLAN_SHA='a'*64

class FakeSession:
    def __init__(self, out, fail=False, bad=None):
        self.calls=0;self.out=out;self.fail=fail;self.bad=bad
    def run(self, outputs, arrays):
        self.calls+=1
        assert self.calls==1
        claim=json.loads((self.out/'single-k1-wu-001.started.json').read_bytes())
        assert claim['inputs']==runner.input_boundary(arrays)
        assert claim['attempt_consumed'] is True
        receipt=json.loads((self.out/'generation-receipt.json').read_bytes())
        assert receipt['attempts_consumed']==1
        assert outputs==['y'] and arrays['sid'].tolist()==[1]
        if self.fail:raise RuntimeError('PRIVATE_ERROR_TEXT')
        if self.bad is not None:return self.bad
        return [np.array([[[0.,.1,-.1,.2]*20]],dtype='float32')]

class OneCallTests(unittest.TestCase):
    def test_default_disabled(self):
        with patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(runner.main([]),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'disabled')
        self.assertNotIn('onnxruntime',sys.modules)
    def test_unarmed_child_refused(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(ValueError,'SUPERVISED_CHILD_REQUIRED'):
            runner.armed_child(Path('/none'),Path('/none'),Path('/none'),PLAN_SHA)
    def test_exact_one_write_ahead_and_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'generation';fake=FakeSession(out)
            result=runner.run_one(fake,FIXTURES,out,PLAN_SHA,np)
            self.assertEqual(fake.calls,1);self.assertEqual(result['status'],'one_candidate_generated')
            self.assertEqual(result['attempts_consumed'],1)
            self.assertEqual(len(list(out.glob('*.native-f32.wav'))),1)
            self.assertEqual(len(list(out.glob('*.pcm16.wav'))),1)
            with self.assertRaisesRegex(ValueError,'NO_RESUME'):runner.run_one(fake,FIXTURES,out,PLAN_SHA,np)
            self.assertEqual(fake.calls,1)
    def test_failed_call_consumed_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'generation';fake=FakeSession(out,fail=True)
            with self.assertRaises(RuntimeError):runner.run_one(fake,FIXTURES,out,PLAN_SHA,np)
            receipt=json.loads((out/'generation-receipt.json').read_bytes())
            self.assertEqual(fake.calls,1);self.assertEqual(receipt['attempts_consumed'],1)
            self.assertEqual([r['status'] for r in receipt['generation_rows']],['failed_no_retry'])
            self.assertNotIn('PRIVATE_ERROR_TEXT',(out/'generation-receipt.json').read_text())
    def test_wrong_text_or_count_before_call(self):
        wrong=copy.deepcopy(FIXTURES);wrong[0]['text']='你好小窝'
        for fixtures in [[],FIXTURES*2,wrong]:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'generation';fake=FakeSession(out)
                with self.assertRaisesRegex(ValueError,'EXACT_ONE_FIXTURE'):runner.run_one(fake,fixtures,out,PLAN_SHA,np)
                self.assertEqual(fake.calls,0)
    def test_invalid_output_consumes_and_saves_no_audio(self):
        bads=[[np.array([[[float('nan')]]],dtype='float32')],
              [np.zeros((1,1,1),dtype='float64')],[np.zeros((1,2,1),dtype='float32')],
              [np.zeros((1,1,0),dtype='float32')],[np.zeros((1,1,441001),dtype='float32')],[]]
        for bad in bads:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'generation';fake=FakeSession(out,bad=bad)
                with self.assertRaises(ValueError):runner.run_one(fake,FIXTURES,out,PLAN_SHA,np)
                self.assertEqual(fake.calls,1);self.assertFalse(list(out.glob('*.wav')))
    def test_finalize_abrupt_exit_preserves_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'generation';out.mkdir()
            runner.write_json(out/'single-k1-wu-001.started.json',{'attempt_consumed':True})
            result=runner.finalize(root,PLAN_SHA,{'returncode':-9},{'reason':'native_call_wall_deadline'})
            self.assertEqual(result['attempts_consumed'],1)
            self.assertEqual(result['generation_rows'][0]['status'],'failed_no_retry')
    def test_finalize_pre_call_failure_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=runner.finalize(Path(tmp),PLAN_SHA,{'returncode':1},{})
            self.assertEqual(result['attempts_consumed'],0)
            self.assertEqual([r['status'] for r in result['generation_rows']],['not_run'])
    def test_watchdog_kills_mock_native_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            try:
                runner.write_json(root/'child-identity.json',{'pid':child.pid,'pgid':child.pid})
                runner.write_json(root/'active-call.json',{'pid':child.pid,'source_id':'single-k1-wu-001','deadline_monotonic':time.monotonic()+.1})
                report={};stop=threading.Event();thread=threading.Thread(target=runner.watchdog,args=(root,stop,report));thread.start()
                self.assertEqual(child.wait(timeout=3),-signal.SIGKILL);thread.join(timeout=1)
                self.assertEqual(report['reason'],'native_call_wall_deadline')
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def test_caps_and_unknown_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'INDIVIDUAL_FILE_CAP'):runner.bounded_write(Path(tmp)/'large',b'x'*(2*runner.MiB+1))
            proc=Path(tmp)/'123';proc.mkdir();(proc/'stat').write_text('invalid')
            with self.assertRaisesRegex(RuntimeError,'PROC_OBSERVATION_UNKNOWN'):runner.observed_group(123,Path(tmp))
        r=runner.annotate_rss({'max_sampled':{'rss_bytes':0}},{'valid_member_samples':0,'empty_member_samples':1,'unknown_samples':0},{})
        self.assertIsNone(r['max_sampled']['rss_bytes']);self.assertEqual(r['rss_observation'],'unknown_not_zero')
    def test_plan_hash_and_exact_source_binding(self):
        path=runner.ROOT/'execution-plan.json'
        runner.validate_plan(path,runner.sha(path))
        with self.assertRaisesRegex(ValueError,'APPROVED_PLAN_HASH_MISMATCH'):runner.validate_plan(path,'0'*64)

if __name__=='__main__':unittest.main(verbosity=2)
