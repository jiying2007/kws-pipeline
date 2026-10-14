import unittest,tempfile,json,hashlib,wave
from pathlib import Path
from screen import score,third_subset,pack,normalize

def data():
 from campaign import expected_plan
 slots=[dict(sample_id=c['cell_id'],engine=e,target_text=c['text'],status='success',audio_quality_pass=True) for e,cells in expected_plan().items() for c in cells]
 slots += [dict(sample_id=f'tqp-cosy3-{i:02d}',engine='cosy3',target_text=('你好小窝','小窝小窝')[(i//2)%2],status='blocked',audio_quality_pass=False) for i in range(16)]
 return {'schema':'tts-quality-pilot-asr-input-v1','slots':slots}

def r(s,f,text='你好小窝',status='success'):
 return dict(sample_id=s,family=f,raw_text=text,status=status,completeness='complete',quality_flags=[])
class TestScreen(unittest.TestCase):
 def test_fixed_denominator(self):
  x=score(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice')]);self.assertEqual(x['metrics']['overall_qualified']['denominator'],48);self.assertEqual(x['metrics']['overall_qualified']['numerator'],1);self.assertFalse(x['human_gold'])
 def test_denominator_layers(self):
  x=score(data(),[]);self.assertEqual(x['denominators'],{'design_slots':48,'authorized_executable_slots':32,'generated_decodable_slots':32,'blocked_reference_slots':16});self.assertEqual(x['executable_metrics']['audio_quality_pass']['denominator'],32)
 def test_consensus_named_correctly(self):
  x=score(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice')]);self.assertEqual(x['slots'][0]['consensus_kind'],'two-model-consensus')
 def test_third_disagreement_only(self):
  self.assertEqual(third_subset(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice','你好小我')]),['tqp-qwen06-00'])
  self.assertEqual(third_subset(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice')]),[])
 def test_failures_not_retried(self):
  self.assertEqual(third_subset(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice',None,'error')]),[])
 def test_majority_review_not_truth(self):
  x=score(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice','你好小我'),r('tqp-qwen06-00','whisper')]);self.assertEqual(x['slots'][0]['tier'],'review');self.assertFalse(x['slots'][0]['human_gold'])
 def test_forbidden_third(self):
  with self.assertRaises(ValueError):score(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice'),r('tqp-qwen06-00','whisper')])
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):score(data(),[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','qwen06')])
 def test_failed_quality_cannot_qualify(self):
  d=data();d['slots'][0]['audio_quality_pass']=False;self.assertFalse(score(d,[r('tqp-qwen06-00','qwen06'),r('tqp-qwen06-00','sensevoice')])['slots'][0]['overall_qualified'])
 def test_missing_completion_no_high_tier(self):
  a=r('tqp-qwen06-00','qwen06');a['completeness']='unknown';self.assertFalse(score(data(),[a,r('tqp-qwen06-00','sensevoice')])['slots'][0]['overall_qualified'])
 def test_no_homophone_mapping(self):self.assertNotEqual(normalize('你好小我'),normalize('你好小窝'))
 def test_blind_pack(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);wav=root/'target.wav'
   with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*1600)
   sha=hashlib.sha256(wav.read_bytes()).hexdigest();d=data();d['slots']=d['slots'][:1];d['slots'][0].update(audio_relpath='target.wav',wav_sha256=sha)
   pack(d,root,root/'blind');job=(root/'blind/qwen06/job.json').read_text();self.assertNotIn('你好',job);self.assertNotIn('target',job);self.assertNotIn('sample_id',job)
if __name__=='__main__':unittest.main()
