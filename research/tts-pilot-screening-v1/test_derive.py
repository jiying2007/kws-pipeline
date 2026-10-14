import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from derive import derive
from screen import manifest,validate_wav
from campaign import expected_plan,MODELS
class TestDerive(unittest.TestCase):
 def test_native_preserved_and_missing_slots(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);folder=root/'in/tqp-qwen06/artifact';folder.mkdir(parents=True)
   src=folder/'tqp-qwen06-00.raw-float.wav';samples=(.1*np.sin(np.arange(2400)*.1)).astype('float32');wavfile.write(src,24000,samples);original=src.read_bytes()
   plan=expected_plan()
   row=dict(plan['qwen06'][0],attempts=1,status='generated',raw_wav_sha256=hashlib.sha256(original).hexdigest(),metrics={'technically_valid':True},native={'sample_rate_hz':24000,'channels':1,'frames':len(samples),'subtype':'FLOAT','source_values_sha256':hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest()})
   (folder/'qwen06-receipt.json').write_text(json.dumps({'engine':'qwen06','model_id':MODELS['qwen06'][0],'revision':MODELS['qwen06'][1],'outcomes':[row]}))
   (root/'plan.json').write_text(json.dumps(plan));rows=derive(root/'plan.json',root/'in',root/'out')
   self.assertEqual(src.read_bytes(),original);self.assertEqual(len(rows),48);self.assertEqual(sum(r['status']=='success' for r in rows),1);self.assertEqual(sum(r['status']=='blocked' for r in rows),16)
   validate_wav(root/'out'/rows[0]['audio_relpath'],rows[0]['wav_sha256']);manifest(root/'out/manifest.json')

class TestDeriveBinding(unittest.TestCase):
 def prepare(self,root,frames=2401):
  folder=root/'in/tqp-qwen06/artifact';folder.mkdir(parents=True)
  plan=expected_plan();cell=plan['qwen06'][0];samples=(.1*np.sin(np.arange(frames)*.01)).astype('float32')
  src=folder/(cell['cell_id']+'.raw-float.wav');wavfile.write(src,24000,samples)
  row=dict(cell,attempts=1,status='generated',raw_wav_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),metrics={'technically_valid':True},native={'sample_rate_hz':24000,'channels':1,'frames':frames,'subtype':'FLOAT','source_values_sha256':hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest()})
  receipt={'engine':'qwen06','model_id':MODELS['qwen06'][0],'revision':MODELS['qwen06'][1],'outcomes':[row]}
  path=folder/'qwen06-receipt.json';path.write_text(json.dumps(receipt));(root/'plan.json').write_text(json.dumps(plan));return path,receipt,samples
 def test_wrong_engine_or_text_rejected(self):
  for field,value in [('engine','qwen17'),('text','小窝小窝'),('seed',0)]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
    root=Path(t);p,r,_=self.prepare(root);r['outcomes'][0][field]=value;p.write_text(json.dumps(r))
    with self.assertRaises(ValueError):derive(root/'plan.json',root/'in',root/'out')
 def test_native_mismatch_retains_failed_slot(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p,r,_=self.prepare(root);r['outcomes'][0]['native']['frames']=1;p.write_text(json.dumps(r))
   rows=derive(root/'plan.json',root/'in',root/'out');self.assertEqual(rows[0]['status'],'failed');self.assertEqual(len(rows),48)
 def test_frame_rounding_and_pcm_quantization(self):
  from scipy.signal import resample_poly
  for n in (1,2,3,2401,288000):
   with self.subTest(frames=n),tempfile.TemporaryDirectory() as t:
    root=Path(t);_,_,x=self.prepare(root,n);rows=derive(root/'plan.json',root/'in',root/'out');sr,y=wavfile.read(root/'out'/rows[0]['audio_relpath'])
    expected=resample_poly(x,2,3,window=('kaiser',5.0),padtype='constant').astype('float64')
    self.assertEqual(sr,16000);self.assertEqual(len(y),(2*n+2)//3);self.assertEqual(y.dtype,np.int16)
    error=expected-y.astype('float64')/32768;self.assertTrue(np.all(error>=0));self.assertTrue(np.all(error<1/32768))

if __name__=='__main__':unittest.main()
