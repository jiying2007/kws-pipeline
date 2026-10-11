import copy,json,pathlib,unittest
from compare import compare,normalized
D=json.loads((pathlib.Path(__file__).parent/'comparison-input.json').read_text())
def terminals():
 return [{'audio_id':r['audio_id'],'wav_sha256':r['wav_sha256'],'raw_text':r['sensevoice']['raw_text'],'status':'completed_native_eot','attempts':1,'stop_reason':'eot','validation_errors':[],'raw_output_sha256':'a'*64} for r in D['rows']]
class ComparisonTests(unittest.TestCase):
 def test_not_run(self):
  c=compare(D)['counts'];self.assertEqual(c['incomplete_three_way'],16);self.assertEqual(c['normalized_three_way_agreement'],0);self.assertEqual(c['qwen_sense_normalized_agreement'],13)
 def test_unanimity_separate_target(self):
  c=compare(D,terminals())['counts'];self.assertEqual(c['normalized_three_way_agreement'],13);self.assertEqual(c['normalized_three_way_agreement_matches_target'],8);self.assertEqual(c['normalized_three_way_agreement_differs_from_target'],5);self.assertEqual(c['normalized_three_way_disagreement'],3);self.assertEqual(c['literal_three_way_agreement'],0)
 def test_missing(self):
  c=compare(D,[])['counts'];self.assertEqual(c['incomplete_three_way'],16);self.assertEqual(c['fixed_denominator'],16)
 def test_failed_is_not_vote(self):
  t=terminals()
  for r in t:r['status']='failed_no_retry'
  self.assertEqual(compare(D,t)['counts']['third_valid_outputs'],0)
 def test_validation_error_is_not_vote(self):
  t=terminals();t[0]['validation_errors']=['ValueError'];self.assertEqual(compare(D,t)['counts']['third_valid_outputs'],15)
 def test_hash_mismatch(self):
  t=terminals();t[0]['wav_sha256']='0'*64
  with self.assertRaises(ValueError):compare(D,t)
 def test_duplicate_unknown(self):
  with self.assertRaises(ValueError):compare(D,terminals()+terminals()[:1])
  t=terminals();t[0]['audio_id']='unknown'
  with self.assertRaises(ValueError):compare(D,t)
 def test_conservative_normalization(self):
  self.assertEqual(normalized('你好，小屋。'),'你好小屋');self.assertNotEqual(normalized('小窝'),normalized('小屋'));self.assertNotEqual(normalized('小吳'),normalized('小吴'));self.assertNotEqual(normalized('A B'),normalized('AB'));self.assertEqual(normalized('OOV+'), 'OOV+')
 def test_original_failure_preserved(self):
  self.assertEqual(D['rows'][0]['original_sensevoice']['status'],'failed_no_retry');self.assertEqual(sum(r['original_sensevoice']['status']=='not_run' for r in D['rows']),15)
 def test_empty_not_vote(self):
  t=terminals();t[0]['raw_text']='。';self.assertEqual(compare(D,t)['counts']['third_valid_outputs'],15)
if __name__=='__main__':unittest.main()
