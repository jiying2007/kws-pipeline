"""Frozen scientific semantics and conservative label boundaries; stdlib only."""
import copy
import json
from pathlib import Path
import unittest
import plan_contract as p
import worker as w

class ContractTests(unittest.TestCase):
    def setUp(self):self.plan=w.frozen_plan()
    def test_on_disk_plan_matches_literal_contract(self):
        self.assertEqual(len(p.validate_plan(w.decode(w.read_regular(Path(w.__file__).with_name('plan.json'))))),16)
    def test_exact_slot_order(self):
        self.assertEqual([r['slot_id'] for r in self.plan['slots']],[f'kw1s-{i:02d}' for i in range(16)])
        self.assertEqual([r['engine'] for r in self.plan['slots']],['qwen06']*4+['qwen17']*12)
    def test_exact_text_and_seeds(self):
        self.assertEqual({r['target_text'] for r in self.plan['slots']},{'你好小窝'})
        self.assertEqual([r['seed'] for r in self.plan['slots']],[2026101401,2026101402]*8)
    def test_point_six_never_gets_instruction(self):
        self.assertTrue(all(r['instruction']=='' for r in self.plan['slots'][:4]))
        self.assertEqual(sum(bool(r['instruction']) for r in self.plan['slots']),6)
    def test_instruction_pairs_only_change_instruction_and_id(self):
        for first in (4,8,12):
            for offset in (0,1):
                a,b=self.plan['slots'][first+offset],self.plan['slots'][first+offset+2]
                self.assertEqual({k:v for k,v in a.items() if k not in ('slot_id','instruction')},{k:v for k,v in b.items() if k not in ('slot_id','instruction')})
    def test_cross_model_empty_pairs(self):
        for left,right in ((0,4),(1,5),(2,8),(3,9)):
            a,b=self.plan['slots'][left],self.plan['slots'][right]
            self.assertEqual({k:v for k,v in a.items() if k not in ('slot_id','engine')},{k:v for k,v in b.items() if k not in ('slot_id','engine')})
    def test_frozen_contract_rejects_any_new_instruction(self):
        self.plan['slots'][0]['instruction']='Read';self.assertRaises(ValueError,p.validate_plan,self.plan)
    def test_frozen_contract_rejects_adaptive_seed(self):
        self.plan['slots'][0]['seed']=42;self.assertRaises(ValueError,p.validate_plan,self.plan)
    def test_frozen_contract_rejects_param_change(self):
        self.plan['slots'][0]['generation_parameters']['temperature']=0.8;self.assertRaises(ValueError,p.validate_plan,self.plan)
    def test_frozen_contract_rejects_bool_numeric_equivalence(self):
        self.plan['slots'][0]['generation_parameters']['top_p']=True;self.assertRaises(ValueError,p.validate_plan,self.plan)
    def test_frozen_contract_rejects_missing_extra_or_reordered_slots(self):
        for slots in (self.plan['slots'][:-1],self.plan['slots']+[self.plan['slots'][0]],list(reversed(self.plan['slots']))):
            plan=copy.deepcopy(self.plan);plan['slots']=slots;self.assertRaises(ValueError,p.validate_plan,plan)
    def test_nfkc_whitespace_punctuation_only(self):
        self.assertEqual(p.normalize(' 你，好\n小窝。'),'你好小窝')
        self.assertEqual(p.normalize('ＡＢＣ！'),'ABC')
        self.assertNotEqual(p.normalize('你好小窝'),p.normalize('你好小屋'))
        self.assertNotEqual(p.normalize('你好小窝'),p.normalize('你好小沃'))
        self.assertNotEqual(p.normalize('你好小窝'),p.normalize('你好小我'))
        self.assertNotEqual(p.normalize('你好小窝'),p.normalize('你好小窝+'))
    def test_eric_only_does_not_supply_three_common_groups(self):
        result=p.coverage_only(['kw1s-02'])
        self.assertEqual(len(result['kw1_groups']),3);self.assertEqual(result['common_group_count'],2)
        self.assertFalse(result['three_role_coverage_necessary_condition'])
    def test_new_uncle_group_is_necessary_never_sufficient(self):
        result=p.coverage_only(['kw1s-00'])
        self.assertEqual(result['common_group_count'],3)
        self.assertTrue(result['three_role_coverage_necessary_condition'])
        self.assertFalse(result['training_admitted']);self.assertFalse(result['roles_assigned']);self.assertFalse(result['human_gold'])
    def test_repeats_do_not_make_new_groups(self):
        a=p.coverage_only(['kw1s-00']);b=p.coverage_only(['kw1s-00','kw1s-01','kw1s-04','kw1s-05','kw1s-06','kw1s-07'])
        self.assertEqual(a,b)
    def test_no_candidate_or_old_unknown_reuse(self):
        self.assertEqual(p.coverage_only([])['common_group_count'],2)
        for ids in (['C13'],['kw1s-00','kw1s-00']):self.assertRaises(ValueError,p.coverage_only,ids)
    def test_historical_execution_gate_always_rejects(self):
        with self.assertRaises(ValueError):p.check_execution(self.plan)
    def test_budget_limits_and_labels(self):
        self.assertEqual(self.plan['budget']['tts_calls'],16);self.assertEqual(self.plan['budget']['asr_calls'],48)
        self.assertEqual(self.plan['budget']['retries'],0);self.assertEqual(self.plan['budget']['money_usd'],0)
        self.assertTrue(all(v is False for v in self.plan['admission'].values()))

if __name__=='__main__':unittest.main()
