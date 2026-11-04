"""Invented native resolver objects only; no Torch, weights or inference."""
from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generation_capture as c
class CaptureTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.value=dict(c.EXPECTED,unknown_preserved_field='native value');self.config=SimpleNamespace(to_dict=lambda:self.value);self.result=(self.config,{'opaque':'unchanged'});self.calls=[]
  def resolve(*a,**k):self.calls.append((a,k));return self.result
  self.model=SimpleNamespace(_prepare_generation_config=resolve);self.recorder=c.ResolvedConfigRecorder(self.model,self.root/'observed',2);self.recorder.install()
 def test_preserves_native_objects_and_unrequested_fields(self):
  result=self.model._prepare_generation_config('config',do_sample=False);self.assertIs(result,self.result);self.assertEqual(self.calls,[(('config',),{'do_sample':False})]);self.assertEqual(json.loads((self.root/'observed/resolved-generation-config-0001.json').read_text()),self.value)
 def test_missing_field_not_defaulted(self):
  del self.value['num_beams']
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
  self.assertEqual(list((self.root/'observed').iterdir()),[])
 def test_non_greedy_rejected_before_return(self):
  self.value['num_beams']=2
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_boolean_alias_rejected(self):
  self.value['num_beams']=True
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_actual_two_token_eos_list_preserved(self):
  self.model._prepare_generation_config();self.assertEqual(json.loads((self.root/'observed/resolved-generation-config-0001.json').read_text())['eos_token_id'],[151645,151643])
 def test_scalar_eos_rejected(self):
  self.value['eos_token_id']=151645
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_float_eos_alias_rejected(self):
  self.value['eos_token_id']=[151645.0,151643]
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_nonfinite_rejected(self):
  self.value['unknown_preserved_field']=float('nan')
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_bound_prevents_additional_native_resolver_calls(self):
  self.model._prepare_generation_config();self.model._prepare_generation_config()
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
  self.assertEqual(len(self.calls),2)
 def test_existing_path_cannot_be_overwritten(self):
  p=self.root/'observed/resolved-generation-config-0001.json';p.write_text('prior')
  with self.assertRaises(FileExistsError):self.model._prepare_generation_config()
  self.assertEqual(p.read_text(),'prior')
 def test_change_in_full_config_rejected(self):
  self.model._prepare_generation_config();self.value['unknown_preserved_field']='changed'
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
 def test_failed_resolution_poison_prevents_retry(self):
  self.value['num_beams']=2
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
  self.value['num_beams']=1
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
  self.assertEqual(len(self.calls),1);self.assertEqual(self.recorder.calls,1);self.assertEqual(self.recorder.completed,0);self.assertTrue(self.recorder.poisoned)
 def test_partial_write_poison_prevents_retry(self):
  (self.root/'observed/resolution-0001.json').write_text('existing receipt')
  with self.assertRaises(FileExistsError):self.model._prepare_generation_config()
  with self.assertRaises(ValueError):self.model._prepare_generation_config()
  self.assertEqual(len(self.calls),1);self.assertEqual(self.recorder.completed,0)
 def test_restore_original(self):
  self.recorder.restore();self.assertIs(self.model._prepare_generation_config,self.recorder.original)
if __name__=='__main__':unittest.main()
