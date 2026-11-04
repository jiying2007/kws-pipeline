"""Invented dictionaries/JSON only; no live audio, ASR or training."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compatibility as c
class Contracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.mapping={'rows':[]};self.report={'planned_audio':4,'planned_asr_slots':12,'rows':[]}
  for i in range(4):
   row=dict(sample_id='sample%d'%i,opaque_id='clip-%06d'%(i+1),derived_sha256=str(i+1)*64,pcm_sha256=str(i+5)*64,requested_text='小窝小窝');self.mapping['rows'].append(row)
   obs={f:dict(opaque_id=row['opaque_id'],wav_sha256=row['derived_sha256'],status='success',completeness='complete',quality_flags=[],raw_text=row['requested_text']) for f in c.FAMILIES}
   self.report['rows'].append(dict(sample_id=row['sample_id'],observations=obs))
   for family in ('qwen06','sensevoice'):
    p=self.root/'primary'/family/family/(row['opaque_id']+'.decoder.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(opaque_id=row['opaque_id'],wav_sha256=row['derived_sha256'],evidence=dict(input_binding={},processor_tensor_evidence={},token_ids=[1],metadata={'eos_terminated':True},raw_result=[],token_binding={}))))
 def run_assess(self):return c.assess(self.report,self.mapping,self.root)
 def test_consensus_cannot_promote(self):
  r=self.run_assess();self.assertEqual(r['current_formal_admission_count'],0);self.assertFalse(r['training_admitted']);self.assertFalse(r['native_evidence_synthesized']);self.assertTrue(r['rows'][0]['machine_text_match']);self.assertIn('resolved_runtime_generation_config_not_captured',r['rows'][0]['blockers'])
 def test_caller_permissions_cannot_promote(self):
  self.report.update(training_admitted=True,reviewed_train_root=True,human_gold=True);self.assertFalse(self.run_assess()['training_admitted'])
 def test_oov_not_blank(self):
  self.mapping['rows'][0]['requested_text']='你好小屋'
  for o in self.report['rows'][0]['observations'].values():o['raw_text']='你好小屋'
  r=self.run_assess()['rows'][0];self.assertTrue(r['machine_text_match']);self.assertFalse(r['vocabulary_representable']);self.assertIn('requested_text_outside_reviewed_ctc_vocabulary',r['blockers'])
 def test_disagreement_preserved(self):
  self.report['rows'][0]['observations']['whisper']['raw_text']='小屋小屋';self.assertFalse(self.run_assess()['rows'][0]['machine_text_match'])
 def test_nonpolicy_punctuation_is_not_erased(self):
  self.report['rows'][0]['observations']['whisper']['raw_text']='小窝—小窝';self.assertFalse(self.run_assess()['rows'][0]['machine_text_match'])
 def test_wrong_hash_rejected(self):
  self.report['rows'][0]['observations']['qwen06']['wav_sha256']='f'*64
  with self.assertRaises(ValueError):self.run_assess()
 def test_wrong_decoder_hash_rejected(self):
  p=self.root/'primary/qwen06/qwen06/clip-000001.decoder.json';d=json.loads(p.read_text());d['wav_sha256']='f'*64;p.write_text(json.dumps(d))
  with self.assertRaises(ValueError):self.run_assess()
 def test_duplicate_pcm_rejected(self):
  self.mapping['rows'][0]['pcm_sha256']=self.mapping['rows'][1]['pcm_sha256']
  with self.assertRaises(ValueError):self.run_assess()
 def test_missing_family_rejected(self):
  self.report['rows'][0]['observations'].pop('whisper')
  with self.assertRaises(ValueError):self.run_assess()
 def test_failed_decoder_not_consensus(self):
  self.report['rows'][0]['observations']['sensevoice']['status']='failed_no_retry';self.assertFalse(self.run_assess()['rows'][0]['machine_text_match'])
 def test_missing_capture_not_fabricated(self):
  p=self.root/'primary/qwen06/qwen06/clip-000001.decoder.json';d=json.loads(p.read_text());d['evidence'].pop('processor_tensor_evidence');p.write_text(json.dumps(d));self.assertFalse(self.run_assess()['rows'][0]['declared_legacy_field_presence']['qwen_processor_tensor_hashes'])
if __name__=='__main__':unittest.main()
