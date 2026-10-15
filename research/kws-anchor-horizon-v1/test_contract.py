"""stdlib-only tests; no research/model/optimizer execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('anchor_run',HERE/'run_horizon.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r._contract
class Tests(unittest.TestCase):
    def setUp(self):self.plan=json.loads((HERE/'plan.json').read_text())
    def test_disabled(self):
        c.validate_plan(self.plan,False)
        disabled=copy.deepcopy(self.plan);disabled['execution_enabled']=False
        with self.assertRaises(ValueError):c.validate_plan(disabled)
        enabled=copy.deepcopy(self.plan);enabled['execution_enabled']=True
        c.validate_plan(enabled)
        self.assertNotIn('torch',sys.modules)
    def test_closed_types_and_values(self):
        for key,value in [('seed',True),('max_optimizer_calls',600.0),('export_weights',0),('prefix_frames',{'KW1':24,'KW2':10}),('campaign_id','old'),('unknown',0)]:
            p=copy.deepcopy(self.plan);p[key]=value
            with self.assertRaises(ValueError):c.validate_plan(p,False)
        for value in [1,'true',None]:
            p=copy.deepcopy(self.plan);p['execution_enabled']=value
            with self.assertRaises(ValueError):c.validate_plan(p,False)
    def test_json_rejections(self):
        for value in ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}']:
            with self.assertRaises(ValueError):c.strict_json(value)
    def test_code_closure(self):
        for op in ['missing','extra','invalid']:
            p=copy.deepcopy(self.plan)
            if op=='missing':p['diagnostic_code_sha256'].pop('contract.py')
            elif op=='extra':p['diagnostic_code_sha256']['extra.py']='a'*64
            else:p['diagnostic_code_sha256']['contract.py']='z'*64
            with self.assertRaises(ValueError):c.validate_plan(p,False)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';c.atomic_json(p,{'a':1})
            with self.assertRaises(FileExistsError):c.atomic_json(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})
    def test_budget(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);claim=d/'kws-anchor-horizon-20261015-v1.claim.json'
            b=c.Budget(d/'result',claim,c.FIXED_PLAN['campaign_id'])
            for stage,update in [('T',1),('C',1),('R',True),('R',2)]:
                with self.assertRaises(ValueError):b.before_step(stage,update)
            b.before_forward('observation','R',0)
            for i in range(1,601):
                b.before_forward('training','R',i)
                b.before_step('R',i)
                if i in [50,100,200,300,400,500,600]:b.before_forward('observation','R',i)
            self.assertEqual((b.count,b.forward_count),(600,608))
            with self.assertRaises(ValueError):b.before_step('R',601)
            with self.assertRaises(ValueError):b.before_forward('observation','R',600)
            with self.assertRaises(FileExistsError):c.Budget(d/'other',claim,c.FIXED_PLAN['campaign_id'])
    def test_trajectory_exact(self):
        baseline=json.loads((HERE/'historical-baseline.json').read_text())
        a=baseline['steps']['1'];self.assertTrue(c.compare_step(a,a))
        for key in ['finite_loss','finite_grad_norm','ctc_loss','anchor_loss']:
            b=copy.deepcopy(a);b[key]+=1e-12;self.assertFalse(c.compare_step(b,a))
        b=copy.deepcopy(a);b['samples'][0]['anchor_nll']+=1e-12
        self.assertFalse(c.compare_step(b,a))
        rows=baseline['observations']['0'];self.assertTrue(c.compare_observation(rows,rows))
        b=copy.deepcopy(rows);b[0]['log_probs'][0][0]+=1e-12
        self.assertFalse(c.compare_observation(b,rows))
    def test_exact_control(self):
        a=[{'x':1.0,'path':[1,0,1],'more':2}];b=[{'x':1.0,'path':[1,0,1]}]
        self.assertTrue(c.compare_control(a,b))
        for value in [1,1.00000000000001,True]:
            a[0]['x']=value;self.assertFalse(c.compare_control(a,b))
    def test_checked_read(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';p.write_bytes(b'test');digest=c.sha(b'test')
            self.assertEqual(r.read_checked(p,digest),b'test')
            with self.assertRaises(ValueError):r.read_checked(p,'0'*64)
            link=Path(td)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):r.read_checked(link,digest)
    def test_baseline(self):
        b=json.loads(r.read_checked(HERE/'historical-baseline.json',self.plan['control_baseline_sha256']))
        self.assertEqual(set(b['steps']),{str(i) for i in range(1,301)})
        self.assertEqual(set(b['observations']),{'0','50','100','200','300'})
    def test_source_guardrails(self):
        source=(HERE/'run_horizon.py').read_text()
        self.assertIn("loss=ctc_loss+anchor_loss",source)
        self.assertIn('ctc_loss=((nll/lengths)*w).sum()',source)
        self.assertLess(source.index('budget.before_step(stage_name,update)'),source.index('optimizer.step()'))
        self.assertNotIn('torch.save',source)
        self.assertNotIn('pair_gate_failed',source)
if __name__=='__main__':unittest.main()
