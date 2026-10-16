"""Synthetic-only tests: no Torch imports, numerical models, optimizer or audio services."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('grouped_run_under_test',HERE/'run_grouped_balance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
c=runner._contract
class ContractTests(unittest.TestCase):
    def test_archived_closure(self):
        spec=importlib.util.spec_from_file_location('grouped_archive',HERE/'verify_archived_result.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.verify(),{'result_files':6077,'observations':18,'optimizer_calls':1200,'model_forward_calls':1218,'scientific_validation':'DESCRIPTIVE_ONLY'})
        self.assertNotIn('torch',sys.modules)
    def test_closure_rejects_weak_types_and_budget_reuse(self):
        spec=importlib.util.spec_from_file_location('grouped_archive_negative',HERE/'verify_archived_result.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        closure=json.loads((HERE/'closure.json').read_bytes())
        for key,value in [('remaining_budget',False),('remaining_budget',0.0),('remaining_model_forward_budget',1),('retry_authorized',True),('execution_enabled',True),('execution_status','running'),('scientific_validation','PASS'),('training_admitted',0),('optimizer_calls_reserved',1200.0)]:
            altered=copy.deepcopy(closure);altered[key]=value
            with self.subTest(key=key,value=value):
                with self.assertRaises(ValueError):module.verify_consumed(altered)
        for raw in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}']:
            with self.assertRaises(ValueError):module.strict_json(raw)
        self.assertFalse(module.exact(True,1))
    def test_disabled_and_type_strict(self):
        p=json.loads((HERE/'plan.json').read_bytes());c.validate_plan(p,False)
        disabled=copy.deepcopy(p);disabled['execution_enabled']=False
        with self.assertRaises(ValueError):c.validate_plan(disabled)
        for field,value in [('max_optimizer_calls',1200.0),('execution_enabled',1),('resume',True),('max_model_forward_calls',1219),('seed',True)]:
            q=copy.deepcopy(p);q[field]=value
            with self.assertRaises(ValueError):c.validate_plan(q,False)
        p['execution_enabled']=True;c.validate_plan(p)
    def test_all_contract_fields_frozen(self):
        p=json.loads((HERE/'plan.json').read_bytes())
        for key in c.FIXED_PLAN:
            q=copy.deepcopy(p);q[key]=None
            with self.assertRaises(ValueError,msg=key):c.validate_plan(q,False)
    def test_weights(self):
        classes=['KW1']*4+['KW2']*8
        self.assertEqual(c.weights(classes,'A'),[1/12]*12)
        self.assertEqual(c.weights(classes,'B'),[1/8]*4+[1/16]*8)
        self.assertEqual(sum(c.weights(classes,'B')),1)
    def test_weights_cover_whole_loss(self):
        classes=['KW1']*4+['KW2']*8
        ctc=[float(i+1) for i in range(12)];anchor=[float(12-i) for i in range(12)]
        for arm in ('A','B'):
            w=c.weights(classes,arm)
            split=sum(a*x for a,x in zip(w,ctc))+sum(a*x for a,x in zip(w,anchor))
            whole=sum(a*(x+y) for a,x,y in zip(w,ctc,anchor))
            self.assertAlmostEqual(split,whole)
    def test_evidence_file_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'oversize.json'
            with self.assertRaises(ValueError):c.atomic_json(p,{'payload':'x'*(1024*1024)})
            self.assertFalse(p.exists())
    def test_total_evidence_limit_reserves_failure(self):
        with tempfile.TemporaryDirectory() as d:
            key=str(Path(d).absolute())
            c._WRITTEN_BYTES[key]=c.MAX_EVIDENCE_DIRECTORY_BYTES-1024*1024
            try:
                with self.assertRaises(ValueError):c.atomic_json(Path(d)/'payload.json',{'x':1})
                c.atomic_json(Path(d)/'failure.json',{'failed':True})
            finally:c._WRITTEN_BYTES.pop(key,None)
    def test_schedule(self):
        s=c.forward_schedule();self.assertEqual(len(s),1218)
        self.assertEqual(s[-2:],[('probe','A',600),('probe','B',600)])
        self.assertEqual(sum(x[0]=='training' for x in s),1200)
        self.assertEqual(sum(x[0]=='observation' for x in s),16)
    def test_full_synthetic_receipt_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b=c.Budget(root/'out',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
            for kind,stage,step in c.forward_schedule():
                b.before_forward(kind,stage,step)
                with self.assertRaises(ValueError):b.before_forward(kind,stage,step)
                b.after_forward(kind,stage,step)
                with self.assertRaises(ValueError):b.after_forward(kind,stage,step)
                if kind=='training':
                    b.before_step(stage,step)
                    with self.assertRaises(ValueError):b.before_step(stage,step)
                    b.after_step(stage,step,{'synthetic':True})
            b.complete();self.assertEqual(b.count,1200)
            with self.assertRaises(ValueError):b.before_forward('probe','B',600)
            with self.assertRaises(FileExistsError):c.Budget(root/'other',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
    def test_fail_closed_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b=c.Budget(root/'out',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
            with self.assertRaises(ValueError):b.before_step('A',1)
            with self.assertRaises(ValueError):b.before_forward('probe','A',600)
            b.before_forward('observation','A',0)
            with self.assertRaises(ValueError):b.after_forward('observation','B',0)
            with self.assertRaises(ValueError):b.complete()
    def test_prefix_synthetic(self):
        from array import array
        for values,expected in [([0]*1040,3),([32767]*400,0),([0]*400+[32767]*320,1)]:
            pcm=array('h',values)
            if sys.byteorder!='little':pcm.byteswap()
            self.assertEqual(runner.prefix_from_pcm(pcm.tobytes())[0],expected)
    def test_frozen_roster(self):
        f=json.loads((HERE/'cohort-prefix-freeze.json').read_bytes())['samples']
        train=[r for r in f if r['role']=='train'];probe=[r for r in f if r['role']=='probe']
        self.assertEqual(len(train),12);self.assertEqual(len(probe),4)
        self.assertEqual(sum(r['target_class']=='KW1' for r in train),4)
        self.assertEqual({r['preset_voice_id'] for r in probe},{'qwen-preset:vivian'})
        self.assertFalse({r['preset_voice_id'] for r in train}&{r['preset_voice_id'] for r in probe})
        self.assertEqual([r['record_id'] for r in train],c.FIXED_PLAN['train_record_ids'])
        for row in f:
            self.assertEqual(row['input_frames'],1+(row['frames']-400)//320)
            self.assertEqual(row['prefix_frames'],next((i for i,v in enumerate(row['rms']) if v>=1e-4),len(row['rms'])))
    def test_json_and_no_overwrite(self):
        for raw in ['{"a":1,"a":2}','{"a":NaN}']:
            with self.assertRaises(ValueError):c.strict_json(raw)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json';c.atomic_json(p,{'a':1})
            with self.assertRaises(FileExistsError):c.atomic_json(p,{'a':2})
            self.assertEqual(json.loads(p.read_bytes()),{'a':1})
    def test_pins_and_no_models_imported(self):
        spec=importlib.util.spec_from_file_location('grouped_preflight_under_test',HERE/'numeric_preflight.py')
        numeric_preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(numeric_preflight)
        numeric_preflight.main();self.assertNotIn('torch',sys.modules)
        p=HERE/'plan.json'
        with self.assertRaises(ValueError):runner.read_checked(p,'0'*64)
        with tempfile.TemporaryDirectory() as d:
            alias=Path(d)/'alias';alias.symlink_to(p)
            with self.assertRaises(ValueError):runner.read_checked(alias,c.sha(p.read_bytes()))
    def test_no_hidden_experimental_calls(self):
        import ast
        tree=ast.parse((HERE/'run_grouped_balance.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
        self.assertEqual(len(calls),1)
        self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
