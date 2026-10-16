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
class ArchiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  spec=importlib.util.spec_from_file_location('lr_archive',HERE/'verify_archived_result.py');cls.a=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.a)
 def test_saved_archive_integrity(self):self.assertEqual(self.a.verify()['archive_integrity'],'PASS');self.assertNotIn('torch',sys.modules)
 def test_consumed_budget_and_bool_corruptions(self):
  c=self.a.load(HERE,'closure.json')
  for key,val in [('remaining_budget',1),('remaining_model_forward_budget',1),('execution_enabled',True),('consumed',1),('retry_authorized',True),('successful_optimizer_steps',599),('model_forward_calls_returned',607)]:
   q=copy.deepcopy(c);q[key]=val
   with self.assertRaises(ValueError):self.a.consumed(q)
 def test_archive_part_corruptions(self):
  root=HERE/'result-20261016';m=self.a.load(root,'parts-manifest.json')
  for change in [lambda x:x['parts'][0].update(sha256='0'*64),lambda x:x['parts'][0].update(path='../escape'),lambda x:x.update(archive_sha256='0'*64),lambda x:x['parts'].pop(),lambda x:x['parts'][0].update(bytes=True)]:
   q=copy.deepcopy(m);change(q)
   with self.assertRaises(ValueError):self.a.reconstruct(root,q)
 def test_strict_json_and_roster(self):
  import io,zipfile
  for raw in [b'{"x":1,"x":2}',b'{"x":NaN}']:
   with self.assertRaises(ValueError):self.a.decode(raw)
  stream=io.BytesIO()
  with zipfile.ZipFile(stream,'w') as z:z.writestr('../escape','x')
  with zipfile.ZipFile(stream) as z:
   with self.assertRaises(ValueError):self.a.roster(z,{})
 def test_manifest_or_plan_corruption(self):
  from unittest.mock import patch
  original=self.a.read
  for target in ['plan.json','executed-plan.json','archive-manifest.json','executed-source-references.json']:
   def altered(root,name):
    raw=original(root,name)
    if name!=target:return raw
    obj=json.loads(raw)
    if target=='plan.json':obj['execution_enabled']=True
    elif target=='executed-plan.json':obj['seed']=123
    elif target=='executed-source-references.json':obj['files'][0]['sha256']='0'*64
    else:obj['files']['acceptance-summary.json']='0'*64
    return json.dumps(obj).encode()
   with patch.object(self.a,'read',altered):
    with self.assertRaises(ValueError):self.a.verify()

 def test_payload_hash_corruption(self):
  with self.assertRaises(ValueError):self.a.payload_hash(b'changed payload','0'*64)
  self.a.payload_hash(b'original payload',self.a.sha(b'original payload'))

 def test_semantic_bindings_after_manifest_refresh(self):
  from unittest.mock import patch
  a=self.a;root=HERE;result=HERE/'result-20261016';original=a.read
  cases=[('reproducibility/trusted-freeze.json','reviewed_source_commit','0'*40,'freeze commit'),('acceptance-summary.json','externally_supplied_source_commit','0'*40,'acceptance source commit'),('reproducibility/grouped-balance-result-verification/verify_result.py',None,None,'shared verifier pin'),('reproducibility/grouped-balance-result-verification/tolerances.json',None,None,'tolerances pin'),('reproducibility/trusted-freeze.json','tolerances_sha256','0'*64,'tolerances pin'),('result-hash-manifest.json','shared_verifier_sha256','0'*64,'manifest shared verifier'),('result-hash-manifest.json','verifier_sha256','0'*64,'manifest verifier'),('acceptance-summary.json','shared_verifier_sha256','0'*64,'acceptance shared verifier')]
  for target,key,value,label in cases:
   with self.subTest(target=target,key=key):
    changes={};raw=original(result,target)
    if key is None:changed=raw+b'\n'
    else:obj=json.loads(raw);obj[key]=value;changed=json.dumps(obj).encode()
    changes[result/target]=changed
    freeze_path=result/'reproducibility/trusted-freeze.json';hash_path=result/'result-hash-manifest.json'
    if freeze_path in changes:
     hm=json.loads(original(result,'result-hash-manifest.json'));hm['trusted_freeze_sha256']=a.sha(changes[freeze_path]);changes[hash_path]=json.dumps(hm).encode()
    if hash_path in changes:
     closure=json.loads(original(root,'closure.json'));closure['result_manifest_sha256']=a.sha(changes[hash_path]);changes[root/'closure.json']=json.dumps(closure).encode()
     manifest=json.loads(original(root,'source-manifest.json'));manifest['files']['closure.json']=a.sha(changes[root/'closure.json']);changes[root/'source-manifest.json']=json.dumps(manifest).encode()
    manifest=json.loads(original(result,'archive-manifest.json'))
    for path,data in changes.items():
     if path.is_relative_to(result):manifest['files'][path.relative_to(result).as_posix()]=a.sha(data)
    changes[result/'archive-manifest.json']=json.dumps(manifest).encode()
    def altered(base,name):
     path=base/name
     return changes[path] if path in changes else original(base,name)
    with patch.object(a,'read',altered):
     with self.assertRaisesRegex(ValueError,label):a.verify()

class SavedDiagnosticTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  spec=importlib.util.spec_from_file_location('saved_diagnostics_integrity',HERE/'verify_saved_diagnostics.py');cls.a=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.a)
 def test_saved_diagnostics_hashes_and_semantics(self):
  self.assertEqual(self.a.verify()['integrity'],'PASS');self.assertNotIn('torch',sys.modules)
 def test_synthetic_suppression_boundaries(self):
  f=self.a.suppression;self.assertEqual(f(False,0,0),(0,0));self.assertEqual(f(True,400,0),(1,19600));self.assertEqual(f(True,19599,19600),(2,19600));self.assertEqual(f(True,19600,19600),(1,38800));self.assertEqual(f(True,400,0),(1,19600))
  for args in [(1,400,0),(True,True,0),(True,400,-1)]:
   with self.assertRaises(ValueError):f(*args)
 def test_synthetic_prefix_math_without_saved_inputs(self):
  import itertools,math
  folder=HERE/self.a.DIRECTORY/self.a.PREFIX;spec=importlib.util.spec_from_file_location('saved_prefix_synthetic',folder/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  probs=[[.3,.2,.1,.25,.15],[.4,.1,.2,.2,.1],[.2,.3,.2,.1,.2],[.2,.1,.2,.2,.3]];rows=[[math.log(x) for x in p] for p in probs];state={():[0.,m.NEG]}
  for length in range(1,5):
   exact={}
   for path in itertools.product(range(5),repeat=length):
    sequence=m.collapse(path);exact[sequence]=exact.get(sequence,0.)+math.prod(probs[t][k] for t,k in enumerate(path))
   state=m.beam_step(state,rows[length-1],625);self.assertEqual(set(state),set(exact))
   for sequence,p in exact.items():
    self.assertLess(abs(m.logadd(*state[sequence])-math.log(p)),1e-12);expanded,forward=m.forward_init(sequence)
    for row in rows[:length]:forward,score=m.forward_step(expanded,forward,row)
    self.assertLess(abs(score-math.log(p)),1e-12)
  self.assertEqual(m.collapse([1,1,0,1]),(1,1));self.assertEqual(m.intervals([False,True,True,False,True]),[[1,2],[4,4]]);self.assertEqual(m.available_ms(0),25);self.assertEqual(m.available_ms(1),45)
 def test_saved_report_semantic_corruption(self):
  a=self.a;base=HERE/a.DIRECTORY;p=base/a.PREFIX;d=base/a.DECODER;pr=a.load(p,'report.json');pp=a.load(p,'analysis-plan.json');dr=a.load(d,'report.json');dp=a.load(d,'analysis-plan.json');claim=a.load(d,'execution-claim.json');complete=a.load(d,'execution-complete.json')
  for key,value in [('runtime_replay',True),('new_model_forwards',1),('status','PRODUCTION_PASS')]:
   q=copy.deepcopy(pr);q[key]=value
   with self.assertRaises(ValueError):a.prefix_semantics(q,pp)
  q=copy.deepcopy(pr);q['samples'][0]['trace'][0]['nominal_available_ms']=0
  with self.assertRaises(ValueError):a.prefix_semantics(q,pp)
  for key,value in [('speech_gate','real-VAD'),('threshold_calibrated_for_research_model',True),('saved_data_native_passes',2),('production_engine_executed',True),('new_model_forwards',True)]:
   q=copy.deepcopy(dr);q[key]=value
   with self.assertRaises(ValueError):a.decoder_semantics(q,dp,claim,complete)
  q=copy.deepcopy(dr);q['samples'][0]['events'][0]['mirror_outcome']='suppressed'
  with self.assertRaises(ValueError):a.decoder_semantics(q,dp,claim,complete)
  q=copy.deepcopy(complete);q['saved_data_passes_completed']=2
  with self.assertRaises(ValueError):a.decoder_semantics(dr,dp,claim,q)
 def test_source_and_result_bindings_after_manifest_refresh(self):
  from unittest.mock import patch
  a=self.a;root=HERE/a.DIRECTORY;original=a.read
  cases=[(a.PREFIX+'/source-addendum.json','actual_observation_writer',{'relative_path':'wrong','sha256':'0'*64},'actual LR writer identity'),(a.DECODER+'/execution-complete.json','report_sha256','0'*64,'completion report binding'),(a.DECODER+'/report.json','native_output_sha256','0'*64,'native numerical body binding'),(a.PREFIX+'/report.json','analyzer_sha256','0'*64,'prefix binding analyzer_sha256')]
  for target,key,value,label in cases:
   with self.subTest(target=target):
    obj=json.loads(original(root,target));obj[key]=value;raw=json.dumps(obj).encode();changes={root/target:raw};m=a.load(root,'archive-manifest.json');m['files'][target]['sha256']=a.sha(raw);m['files'][target]['bytes']=len(raw)
    if target==a.DECODER+'/report.json':
     cp=a.load(root/a.DECODER,'execution-complete.json');cp['report_sha256']=a.sha(raw);cr=json.dumps(cp).encode();path=a.DECODER+'/execution-complete.json';changes[root/path]=cr;m['files'][path]['sha256']=a.sha(cr);m['files'][path]['bytes']=len(cr)
    changes[root/'archive-manifest.json']=json.dumps(m).encode()
    def altered(base,name):return changes[base/name] if base/name in changes else original(base,name)
    with patch.object(a,'read',altered):
     with self.assertRaisesRegex(ValueError,label):a.verify()
 def test_strict_saved_metadata_and_no_replay_entry(self):
  for raw in [b'{"x":1,"x":2}',b'{"x":Infinity}']:
   with self.assertRaises(ValueError):self.a.decode(raw)
  tree=ast.parse((HERE/'verify_saved_diagnostics.py').read_text());imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))];self.assertFalse(any('subprocess' in ast.unparse(n) or 'torch' in ast.unparse(n) for n in imports))

if __name__=='__main__':unittest.main()
