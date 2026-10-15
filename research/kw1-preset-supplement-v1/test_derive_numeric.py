"""Pinned numeric derivation fixtures; no synthesized/model audio."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import resample_poly
from worker import frozen_plan,float_wav,native_samples,sha,signal_metrics
from derive_native import derive


def native_fixture(root):
    root=Path(root);plan=frozen_plan();roots={};rows=[];arrays=[]
    term=dict(iterations=10,last_token_ids=[2150],eos=2150,ended_with_eos=True,hit_token_cap_without_eos=False)
    for engine in ('qwen06','qwen17'):
        roots[engine]=root/engine;roots[engine].mkdir()
    for i,cell in enumerate(plan['slots']):
        n=241+i;x=(0.1*np.sin(np.arange(n,dtype=np.float64)*(0.03+i*0.001))).astype('<f4')
        raw=float_wav(x.tobytes());values,samples=native_samples(raw)
        row=dict(cell,status='completed',attempts=1,error_type=None,file=cell['slot_id']+'.raw-float.wav',raw_wav_sha256=sha(raw),native=dict(sample_rate_hz=24000,channels=1,frames=n,subtype='FLOAT',source_values_sha256=sha(values)),metrics=signal_metrics(samples,term),termination=copy.deepcopy(term))
        (roots[cell['engine']]/row['file']).write_bytes(raw);rows.append(row);arrays.append(x)
    return plan,rows,roots,arrays

class NumericTests(unittest.TestCase):
    def test_exact_rational_resampling_and_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,rows,roots,arrays=native_fixture(root)
            hashes=[row['raw_wav_sha256'] for row in rows]
            output=root/'derived';mapping=derive(plan['slots'],rows,roots,output)
            self.assertEqual(len(mapping),16)
            for cell,row,entry,x,digest in zip(plan['slots'],rows,mapping,arrays,hashes):
                expected=np.floor(resample_poly(x,2,3,window=('kaiser',5.0),padtype='constant').astype(np.float64)*32768).astype('<i2').tobytes()
                self.assertEqual((output/entry['derived']['file']).read_bytes(),expected)
                self.assertEqual(entry['derived']['frames'],(2*len(x)+2)//3)
                self.assertEqual(sha((roots[cell['engine']]/row['file']).read_bytes()),digest)
    def test_duplicate_native_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,rows,roots,arrays=native_fixture(root)
            original=rows[0];cell=plan['slots'][1];rows[1]={**original,**cell,'file':cell['slot_id']+'.raw-float.wav'}
            (roots[cell['engine']]/rows[1]['file']).write_bytes((roots[original['engine']]/original['file']).read_bytes())
            with self.assertRaises(ValueError):derive(plan['slots'],rows,roots,root/'derived')
    def test_numeric_version_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,rows,roots,arrays=native_fixture(root)
            with patch.object(np,'__version__','0.0.0'):
                with self.assertRaises(ValueError):derive(plan['slots'],rows,roots,root/'derived')
            self.assertFalse((root/'derived').exists())
    def test_incomplete_native_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,rows,roots,arrays=native_fixture(root)
            with self.assertRaises(ValueError):derive(plan['slots'],rows[:-1],roots,root/'derived')
    def test_native_mutation_fails_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,rows,roots,arrays=native_fixture(root)
            p=roots['qwen06']/rows[0]['file'];p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):derive(plan['slots'],rows,roots,root/'derived')
class OrchestrationTests(unittest.TestCase):
    def run_case(self, failure):
        import campaign
        import worker
        from test_worker import FakeBackend
        from test_archive_faults import make_asr
        from execution_common import claim
        class Backend:
            def __init__(self):
                self.stages=[];self.calls=[]
            def tts(self, engine, plan, output):
                self.stages.append(engine);output.mkdir(parents=True)
                class Model(FakeBackend):
                    calls=self.calls
                    fail_at=2 if failure=='tts' else None
                    load_error=None;signal=None;term=None;duplicate=False
                worker.run(engine,plan,output,Path('/not-loaded'),Path('/not-loaded'),backend_factory=Model)
            def asr(self, name, inputs, output):
                self.stages.append(name)
                make_asr(inputs,output,name,count=1 if failure=='asr' else None)
            def public_summary(self):
                return dict(no_models_executed=True,synthetic_mock=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);claim(root/'campaign.claim.json',{'synthetic':True});backend=Backend()
            result=campaign.execute(worker.frozen_plan(),root,backend)
            return result,backend
    def test_full_pipeline_fixed_16_plus_48(self):
        result,backend=self.run_case(None)
        self.assertEqual(backend.stages,['qwen06','qwen17','qwen_asr','sensevoice','whisper_small'])
        self.assertEqual(backend.calls,[f'kw1s-{i:02d}' for i in range(16)])
        self.assertEqual(result['consumed_calls'],{'tts':16,'asr':48})
        self.assertEqual(result['target_qualified'],16)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(result['training_admitted'])
    def test_pipeline_tts_failure_consumes_two_then_stops(self):
        result,backend=self.run_case('tts')
        self.assertEqual(backend.stages,['qwen06'])
        self.assertEqual(backend.calls,['kw1s-00','kw1s-01'])
        self.assertEqual(result['consumed_calls'],{'tts':2,'asr':0})
        self.assertEqual(result['status'],'stopped_no_retry')
    def test_pipeline_partial_asr_stops_all_peers(self):
        result,backend=self.run_case('asr')
        self.assertEqual(backend.stages,['qwen06','qwen17','qwen_asr'])
        self.assertEqual(result['consumed_calls'],{'tts':16,'asr':1})
        self.assertEqual(result['target_qualified'],0)
        self.assertEqual(result['status'],'stopped_no_retry')

if __name__=='__main__':unittest.main()
