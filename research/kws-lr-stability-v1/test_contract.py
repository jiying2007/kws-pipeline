"""Synthetic-only contract tests; run in repository layout beside pinned grouped module."""
import ast,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lr_runner_test',HERE/'run_lr_stability.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);c=r._contract
class Tests(unittest.TestCase):
 def test_enabled_disabled(self):
  p=json.loads((HERE/'plan.json').read_bytes());c.validate_plan(p,False)
  disabled=copy.deepcopy(p);disabled['execution_enabled']=False
  with self.assertRaises(ValueError):c.validate_plan(disabled)
  enabled=copy.deepcopy(p);enabled['execution_enabled']=True;c.validate_plan(enabled)
 def test_closed_plan(self):
  p=json.loads((HERE/'plan.json').read_bytes())
  for key in c.FIXED_PLAN:
   q=copy.deepcopy(p);q[key]=None
   with self.assertRaises(ValueError,msg=key):c.validate_plan(q,False)
  for key,val in [('seed',True),('max_optimizer_calls',600.0),('execution_enabled',1),('unknown',False)]:
   q=copy.deepcopy(p);q[key]=val
   with self.assertRaises(ValueError):c.validate_plan(q,False)
 def test_lr_fixed_boundary(self):
  self.assertEqual([c.learning_rate(i) for i in (1,499,500,501,502,600)],[.003,.003,.003,.0003,.0003,.0003])
  for val in (True,0,601,500.0):
   with self.assertRaises(ValueError):c.learning_rate(val)
 def test_schedule_preserves_moments(self):
  from types import SimpleNamespace
  moments={'first':object(),'second':object()};op=SimpleNamespace(param_groups=[{'lr':.003}],state=moments)
  for update in range(1,501):self.assertFalse(c.prepare_update(op,update,[]))
  self.assertTrue(c.prepare_update(op,501,[0,50,100,200,300,400,500]));self.assertIs(op.state,moments)
  for update in range(502,601):self.assertFalse(c.prepare_update(op,update,[]))
  self.assertIs(op.state,moments);self.assertEqual(op.param_groups[0]['lr'],.0003)
 def test_failed500_blocks501_and_bad_schedule(self):
  from types import SimpleNamespace
  op=SimpleNamespace(param_groups=[{'lr':.003}])
  with self.assertRaises(ValueError):c.prepare_update(op,501,[0,50,100,200,300,400])
  self.assertEqual(op.param_groups[0]['lr'],.003)
  for update,lr in [(499,.0003),(500,.0003),(502,.003),(501,.0003)]:
   op=SimpleNamespace(param_groups=[{'lr':lr}])
   with self.assertRaises(ValueError):c.prepare_update(op,update,[0,50,100,200,300,400,500])
 def test_complete_schedule(self):
  s=c.forward_schedule();self.assertEqual(len(s),608)
  self.assertEqual(s[506],('observation','A',500));self.assertEqual(s[507],('training','A',501));self.assertEqual(s[-2:], [('training','A',600),('observation','A',600)])
  self.assertFalse(any(e[0]=='probe' or e[1]!='A' for e in s))
 def test_exact_receipt_sequence(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);b=c.Budget(root/'result',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
   for kind,arm,step in c.forward_schedule():
    b.before_forward(kind,arm,step)
    with self.assertRaises(ValueError):b.before_forward(kind,arm,step)
    b.after_forward(kind,arm,step)
    with self.assertRaises(ValueError):b.after_forward(kind,arm,step)
    if kind=='training':
     b.before_step(arm,step)
     with self.assertRaises(ValueError):b.before_step(arm,step)
     b.after_step(arm,step,{'synthetic':True})
   b.complete();self.assertEqual(b.count,600)
   with self.assertRaises(ValueError):b.before_forward('training','A',601)
   with self.assertRaises(ValueError):b.before_step('A',601)
   with self.assertRaises(FileExistsError):c.Budget(root/'other',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
 def test_pending_and_wrong_campaign(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaises(ValueError):c.Budget(root/'x',root/'old.claim.json','kws-grouped-balance-20261015-v1')
   b=c.Budget(root/'x',root/(c.FIXED_PLAN['campaign_id']+'.claim.json'),c.FIXED_PLAN['campaign_id'])
   with self.assertRaises(ValueError):b.before_step('A',1)
   with self.assertRaises(ValueError):b.before_forward('probe','A',0)
   with self.assertRaises(ValueError):b.before_forward('observation','B',0)
   with self.assertRaises(ValueError):b.complete()
 def sample_payload(self):
  x={k:0 for k in c.REPLAY_FIELDS};x.update(schema='grouped-forward-payload-v1',samples=[{'log_probs':[[-1.,-2.]],'target_tokens':[1,2],'weight':1/12}],stage='A');return x
 def test_replay_identity_ignored_science_exact(self):
  a=self.sample_payload();b=copy.deepcopy(a);b['schema']='lr-stability-forward-payload-v1';b['identity']='new';c.verify_replay(b,a)
  for key in c.REPLAY_FIELDS:
   changed=copy.deepcopy(b);changed[key]=None
   with self.assertRaises(ValueError,msg=key):c.verify_replay(changed,a)
  b['samples'][0]['log_probs'][0][0]+=.00000000001
  with self.assertRaises(ValueError):c.verify_replay(b,a)
 def test_replay_schema_and_types(self):
  a=self.sample_payload();b=copy.deepcopy(a);b['schema']='unknown'
  with self.assertRaises(ValueError):c.verify_replay(b,a)
  b=copy.deepcopy(a);b['update']=False
  with self.assertRaises(ValueError):c.verify_replay(b,a)
 def test_replay_precedes_optimizer(self):
  text=(HERE/'run_lr_stability.py').read_text();self.assertLess(text.index('historical_gradient_comparison'),text.index('budget.before_step'))
  self.assertLess(text.index('_contract.verify_replay(evidence'),text.index('return loss,evidence'))
  self.assertIn("_contract.prepare_update(optimizer,update,replay_observations)",text)
  tree=ast.parse(text)
  adam=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Adam'];self.assertEqual(len(adam),1)
  modelcalls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model'];self.assertEqual(len(modelcalls),1)
 def gate_nodes(self,phase):
  tree=ast.parse((HERE/'run_lr_stability.py').read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.If) and any(isinstance(x,ast.Constant) and x.value==phase for x in ast.walk(node)):
    # Only select the innermost comparison branch, not an outer control loop.
    if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id in ('step','update'):
     for parent in ast.walk(tree):
      body=getattr(parent,'body',None)
      if isinstance(body,list) and node in body:
       index=body.index(node);return [node,body[index+1]]
  raise AssertionError('comparison/reset block missing')
 def test_successful_payload_gate_clears_phase_before_io_failure(self):
  payload=self.sample_payload();env={'step':500,'budget':type('Budget',(),{'forward_count':507})(),'_contract':c,'evidence':payload,'history':lambda name:copy.deepcopy(payload),'failure_context':{}}
  nodes=self.gate_nodes('historical_payload_comparison')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_payload_gate','exec'),env)
  try:raise OSError('synthetic observation write failure')
  except OSError:self.assertEqual(env['failure_context'],{'phase':'evaluation'})
  env['history']=lambda name:{**payload,'samples':[]}
  with self.assertRaises(ValueError):exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_payload_gate','exec'),env)
  self.assertEqual(env['failure_context']['phase'],'historical_payload_comparison')
 def test_successful_gradient_gate_clears_phase_before_reservation_failure(self):
  evidence={'finite_loss':1.,'ctc_loss':.8,'anchor_loss':.2,'blank_lambda':1};old={'finite_grad_norm':2.,**evidence}
  env={'update':500,'norm':2.,'evidence':evidence,'history':lambda name:copy.deepcopy(old),'_contract':c,'failure_context':{}}
  nodes=self.gate_nodes('historical_gradient_comparison')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_gradient_gate','exec'),env)
  try:raise OSError('synthetic optimizer reservation failure')
  except OSError:self.assertEqual(env['failure_context'],{'phase':'optimizer_reservation'})
  env['norm']=3.
  with self.assertRaises(ValueError):exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_gradient_gate','exec'),env)
  self.assertEqual(env['failure_context']['phase'],'historical_gradient_comparison')
  self.assertNotEqual(env['failure_context']['observed'],env['failure_context']['expected'])
 def test_source_history_and_runtime_pins(self):
  spec=importlib.util.spec_from_file_location('lr_static',HERE/'numeric_preflight.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.main()
  pins=json.loads((HERE/'historical-input-pins.json').read_bytes());self.assertEqual(len(pins['files']),1017)
  self.assertIn('step-0500.json',pins['files']);self.assertIn('payload-0507.json',pins['files']);self.assertIn('A-0600.json',pins['files'])
  with self.assertRaises(ValueError):r.read_checked(HERE/'plan.json','0'*64)
  with self.assertRaises(ValueError):c.pinned_module('bad',HERE/'contract.py','0'*64)
  self.assertNotIn('torch',sys.modules)
 def test_evidence_limits(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.json'
   with self.assertRaises(ValueError):c.atomic_json(p,{'x':'x'*(1024**2)})
   c.atomic_json(p,{'ok':True})
   with self.assertRaises(FileExistsError):c.atomic_json(p,{'ok':False})
   key=str(Path(d).absolute());c._shared._WRITTEN_BYTES[key]=c._shared.MAX_EVIDENCE_DIRECTORY_BYTES-1024**2
   try:
    with self.assertRaises(ValueError):c.atomic_json(Path(d)/'payload.json',{'x':1})
    c.atomic_json(Path(d)/'failure.json',{'failed':True})
   finally:c._shared._WRITTEN_BYTES.pop(key,None)
if __name__=='__main__':unittest.main()
