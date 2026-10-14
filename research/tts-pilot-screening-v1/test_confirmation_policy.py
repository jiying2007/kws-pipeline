"""Explicit, bounded third confirmation; no models or external actions."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from screen import score,third_subset,third_policy,validate_manifest
from test_screen import data,r
SID='tqp-qwen06-00'
def confirmed_data():
 d=data();d['third_policy']='training_candidate_confirmation';return d
def primary():return [r(SID,'qwen06'),r(SID,'sensevoice')]
class ConfirmationPolicyTests(unittest.TestCase):
 def test_default_unchanged(self):
  d=data();self.assertEqual(third_policy(d),'disagreement_only');self.assertEqual(third_subset(d,primary()),[])
  with self.assertRaises(ValueError):score(d,primary()+[r(SID,'whisper')])
 def test_explicit_selects_primary_candidate(self):
  self.assertEqual(third_subset(confirmed_data(),primary()),[SID])
  out=score(confirmed_data(),primary())['slots'][0];self.assertEqual(out['consensus_kind'],'two-model-consensus');self.assertEqual(out['tier'],'machine_high_confidence_candidate');self.assertFalse(out['three_model_training_candidate']);self.assertEqual(out['training_confirmation_status'],'not_run')
 def test_unknown_or_null_policy_rejected(self):
  for policy in ('all_clips',None,True):
   d=data();d['third_policy']=policy
   with self.assertRaises(ValueError):validate_manifest(d)
 def test_actual_three_confirmed_not_gold_or_admitted(self):
  x=score(confirmed_data(),primary()+[r(SID,'whisper')]);row=x['slots'][0]
  self.assertEqual(row['consensus_kind'],'three-model-consensus');self.assertEqual(row['tier'],'machine_three_asr_training_candidate');self.assertEqual(row['primary_tier'],'machine_high_confidence_candidate');self.assertTrue(row['three_model_training_candidate']);self.assertEqual(row['label_source'],'three_asr_machine_consensus');self.assertFalse(row['human_gold']);self.assertFalse(row['training_admitted']);self.assertEqual(x['metrics']['three_model_training_candidate']['denominator'],48)
 def test_third_contradiction_preserves_primary_tier(self):
  x=score(confirmed_data(),primary()+[r(SID,'whisper','你好小我')])['slots'][0]
  self.assertEqual(x['primary_tier'],'machine_high_confidence_candidate');self.assertEqual(x['tier'],'review');self.assertEqual(x['consensus_kind'],'three-model-disagreement');self.assertFalse(x['overall_qualified']);self.assertFalse(x['three_model_training_candidate']);self.assertEqual(x['training_confirmation_status'],'disagreed')
 def test_disagreement_third_majority_stays_review(self):
  rows=[r(SID,'qwen06'),r(SID,'sensevoice','你好小我'),r(SID,'whisper')]
  for d in (data(),confirmed_data()):
   x=score(d,rows)['slots'][0];self.assertEqual(x['tier'],'review');self.assertEqual(x['consensus_kind'],'three-model-disagreement');self.assertFalse(x['three_model_training_candidate']);self.assertEqual(x['third_reason'],'primary_disagreement')
 def test_failed_third_is_never_rescheduled(self):
  for status in ('error','timeout','failed_no_retry','not_run'):
   rows=primary()+[r(SID,'whisper',None,status)]
   self.assertEqual(third_subset(confirmed_data(),rows),[]);self.assertFalse(score(confirmed_data(),rows)['slots'][0]['three_model_training_candidate'])
 def test_duplicate_third_or_second_attempt_rejected(self):
  for extra in ([r(SID,'whisper'),r(SID,'whisper')],[dict(r(SID,'whisper'),attempt=2)]):
   with self.assertRaises(ValueError):score(confirmed_data(),primary()+extra)
 def test_complete_and_quality_required_for_confirmation(self):
  variants=[]
  d=confirmed_data();d['slots'][0]['audio_quality_pass']=False;variants.append((d,primary()))
  for field,value in [('completeness','unknown'),('quality_flags',['truncated']),('status','timeout')]:
   rows=primary();rows[0][field]=value;variants.append((confirmed_data(),rows))
  for d,rows in variants:self.assertEqual(third_subset(d,rows),[])
 def test_consistent_wrong_target_not_selected(self):
  rows=[r(SID,'qwen06','你好小我'),r(SID,'sensevoice','你好小我')];self.assertEqual(third_subset(confirmed_data(),rows),[])
 def test_incomplete_third_text_agreement_not_training(self):
  c=r(SID,'whisper');c['completeness']='unknown';x=score(confirmed_data(),primary()+[c])['slots'][0]
  self.assertEqual(x['consensus_kind'],'three-model-consensus');self.assertFalse(x['three_model_training_candidate']);self.assertEqual(x['training_confirmation_status'],'incomplete')
 def test_32_candidates_same_96_call_ceiling(self):
  d=confirmed_data();rows=[]
  for slot in d['slots']:
   if slot['engine']=='cosy3':continue
   rows += [r(slot['sample_id'],f,slot['target_text']) for f in ('qwen06','sensevoice')]
  selected=third_subset(d,rows);self.assertEqual(len(selected),32)
  targets={s['sample_id']:s['target_text'] for s in d['slots']}
  rows += [r(s,'whisper',targets[s]) for s in selected]
  x=score(d,rows);self.assertEqual(len(rows),96);self.assertEqual(x['max_asr_slots'],144);self.assertEqual(x['metrics']['three_model_training_candidate']['numerator'],32);self.assertEqual(third_subset(d,rows),[])
 def test_admission_propagation_requires_explicit_field(self):
  path=Path(__file__).parent/'activation-template/finalize_input.py';spec=importlib.util.spec_from_file_location('confirmation_finalize',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=root/'admission.json';p=root/'manifest.json';p.write_text(json.dumps(data()));a.write_text(json.dumps({'failed_probe_lanes':[]}))
   with self.assertRaises(ValueError):m.finalize(a,p)
   a.write_text(json.dumps({'failed_probe_lanes':[],'third_policy':'training_candidate_confirmation'}));out=m.finalize(a,p);self.assertEqual(out['third_policy'],'training_candidate_confirmation')
 def test_default_and_new_share_disagreement_budget(self):
  d=confirmed_data();rows=primary()+[r('tqp-qwen06-01','qwen06'),r('tqp-qwen06-01','sensevoice','你好小我')]
  self.assertEqual(set(third_subset(d,rows)),{SID,'tqp-qwen06-01'})
if __name__=='__main__':unittest.main()
