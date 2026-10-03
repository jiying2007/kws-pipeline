import copy, importlib.util, json, pathlib, unittest
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('admit_run', HERE/'admit_run.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
class Admission(unittest.TestCase):
 def setUp(self):
  self.head='a'*40; self.source='b'*64; self.lock={'compressed_total_bytes':123}
  self.approval={'armed':True,'pr_number':999,'base_sha':a.BASE,'head_sha':self.head,'source_sha256':self.source,'nonce':a.NONCE,'scope':'one_cpu_qualification_then_six_plain_inpaint_pairs','maximum_clips':12,'dependency_compressed_bytes':123,'model_bytes':5427029103,'post_runtime_free_bytes':6635020288,'new_job_bytes_max':14000000000,'output_bytes_max':134217728,'artifact_retention_days':1,'zero_cost_artifact_verified':True}
  self.event={'action':'labeled','label':{'name':a.LABEL},'number':999,'repository':{'full_name':a.REPOSITORY,'private':False},'pull_request':{'number':999,'head':{'repo':{'full_name':a.REPOSITORY},'ref':a.BRANCH,'sha':self.head},'base':{'sha':a.BASE},'user':{'login':'jiying2007'},'body':a.PREFIX+json.dumps(self.approval)}}
  self.env={'GITHUB_EVENT_NAME':'pull_request','GITHUB_RUN_ATTEMPT':'1','GITHUB_ACTOR':'jiying2007'}
 def validate(self,event=None,env=None,paths=None):
  return a.validate(event or self.event,env or self.env,self.head,self.source,paths or ['research/cosyvoice3_pilot/pilot.py'],self.lock)
 def test_exact_approval(self): self.assertEqual(self.validate(),self.approval)
 def test_ordinary_events(self):
  for action in ['opened','reopened','synchronize','unlabeled']:
   x=copy.deepcopy(self.event);x['action']=action
   with self.assertRaises(RuntimeError):self.validate(x)
 def test_rerun_and_actor(self):
  for k,v in [('GITHUB_RUN_ATTEMPT','2'),('GITHUB_ACTOR','other'),('GITHUB_EVENT_NAME','workflow_dispatch')]:
   x=dict(self.env);x[k]=v
   with self.assertRaises(RuntimeError):self.validate(env=x)
 def test_approval_budget_source_cost_scope(self):
  for k,v in [('maximum_clips',13),('zero_cost_artifact_verified',False),('source_sha256','c'*64),('artifact_retention_days',2),('scope','other')]:
   x=copy.deepcopy(self.event);p=dict(self.approval);p[k]=v;x['pull_request']['body']=a.PREFIX+json.dumps(p)
   with self.assertRaises(RuntimeError):self.validate(x)
 def test_unrelated_files(self):
  with self.assertRaises(RuntimeError):self.validate(paths=['src/kws.c'])
 def test_private_or_unowned(self):
  for node,k,v in [('repository','private',True),('head','ref','other')]:
   x=copy.deepcopy(self.event);d=x['repository'] if node=='repository' else x['pull_request']['head'];d[k]=v
   with self.assertRaises(RuntimeError):self.validate(x)
 def history(self,state):
  def f(path):
   if '/workflows/' in path:return {'workflow_runs':[{'id':1,'head_branch':a.BRANCH},{'id':2,'head_branch':a.BRANCH}]}
   return {'jobs':[{'name':'one-time-pilot','conclusion':state}]}
  return f
 def test_skipped_static_run_allows(self):a.reject_prior_runs(self.history('skipped'),2)
 def test_any_previous_admitted_run_blocks(self):
  for state in [None,'success','failure','cancelled','timed_out']:
   with self.assertRaises(RuntimeError):a.reject_prior_runs(self.history(state),2)
 def test_history_saturation_blocks(self):
  with self.assertRaises(RuntimeError):a.reject_prior_runs(lambda p:{'workflow_runs':[{'id':2,'head_branch':a.BRANCH}]*100},2)
if __name__=='__main__':unittest.main(verbosity=2)
