#!/usr/bin/env python3
"""Pure mocked-driver tests: no real audio, model, runtime or network."""
import ast,contextlib,copy,json,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import pilot_common as c,pilot_worker as w,admit_run as a

class Activation(unittest.TestCase):
 def setUp(self):
  self.head='1'*40;self.parent='2'*40;self.payload='3'*64
  self.arm={'mode':'run-fixed30-once','reviewed_parent_sha':self.parent,'payload_sha256':self.payload,'batch_id':a.BATCH}
  self.event={'repository':{'full_name':a.REPOSITORY,'private':False},'ref':'refs/heads/'+a.BRANCH,'deleted':False,'before':self.parent,'after':self.head}
  self.env={'GITHUB_EVENT_NAME':'push','GITHUB_REF':'refs/heads/'+a.BRANCH,'GITHUB_RUN_ATTEMPT':'1','GITHUB_ACTOR':'jiying2007','GITHUB_SHA':self.head}
 def valid(self):return a.validate(self.event,self.env,self.head,self.parent,self.payload,[a.ARM],self.arm)
 def test_exact_activation(self):self.assertEqual(self.valid()['maximum_clips'],30)
 def test_default_disabled(self):
  arm={'mode':'disabled','reviewed_parent_sha':None,'payload_sha256':None,'batch_id':a.BATCH}
  self.assertFalse(a.check_activation(arm,self.payload))
 def test_payload_change(self):
  self.arm['payload_sha256']='4'*64
  with self.assertRaises(RuntimeError):self.valid()
 def test_wrong_event_ref_actor_or_rerun(self):
  for key,value in [('GITHUB_EVENT_NAME','pull_request'),('GITHUB_REF','refs/heads/main'),('GITHUB_ACTOR','other'),('GITHUB_RUN_ATTEMPT','2')]:
   with self.subTest(key=key),patch.dict(self.env,{key:value}):
    with self.assertRaises(RuntimeError):self.valid()
 def test_private_deleted_or_parent_change(self):
  for change in [{'repository':{'full_name':a.REPOSITORY,'private':True}},{'deleted':True},{'before':'4'*40}]:
   with self.subTest(change=change),patch.dict(self.event,change):
    with self.assertRaises(RuntimeError):self.valid()
 def test_only_arm_can_change(self):
  with self.assertRaises(RuntimeError):a.validate(self.event,self.env,self.head,self.parent,self.payload,[a.ARM,'other'],self.arm)
 def test_prior_heavy_job_consumes_attempt(self):
  def fetch(path):
   if '/workflows/' in path:return {'workflow_runs':[{'id':10,'head_branch':a.BRANCH}]}
   return {'jobs':[{'name':a.HEAVY_JOB,'conclusion':'failure'}]}
  with self.assertRaises(RuntimeError):a.reject_prior_runs(fetch,11)
 def test_prior_skipped_only_allowed(self):
  def fetch(path):
   if '/workflows/' in path:return {'workflow_runs':[{'id':10,'head_branch':a.BRANCH}]}
   return {'jobs':[{'name':a.HEAVY_JOB,'conclusion':'skipped'}]}
  a.reject_prior_runs(fetch,11)

class FrozenContract(unittest.TestCase):
 def test_exact_matrix(self):
  cfg=c.load_config();self.assertEqual(len(cfg['rows']),30)
  self.assertEqual([r['stock_voice'] for r in cfg['references']],['Eric','Serena','Vivian','Uncle_Fu','Dylan'])
  self.assertEqual([r['seed'] for r in cfg['rows']],list(range(610401,610431)))
  self.assertEqual(sum(r['split']=='train' for r in cfg['rows']),18)
  self.assertEqual(sum(r['split']=='development' for r in cfg['rows']),6)
  self.assertEqual(sum(r['split']=='sealed_recording_holdout' for r in cfg['rows']),6)
 def test_bad_split_seed_text_or_count(self):
  for kind in ('split','seed','text','count'):
   cfg=copy.deepcopy(c.load_config())
   if kind=='count':cfg['rows'].pop()
   elif kind=='split':cfg['rows'][-1]['split']='train'
   elif kind=='seed':cfg['rows'][0]['seed']+=1
   else:cfg['rows'][0]['intended_text']='你好小屋'
   with self.subTest(kind=kind),patch.object(c,'read_json',return_value=cfg):
    with self.assertRaises(c.GateError):c.load_config()
 def test_one_synthesis_call_site(self):
  tree=ast.parse(pathlib.Path(w.__file__).read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='inference_zero_shot']
  self.assertEqual(len(calls),1)
  self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in {'inference_instruct2','inference_cross_lingual','inference_sft'} for n in ast.walk(tree)))
 def test_runtime_lock_and_guards(self):
  import runtime_lock
  self.assertEqual(runtime_lock.RAW_SHA256,'c9a2a0db05af4d90861cbab7f06837007a74c6a3b82d255b9fad231e0bc00406')
  self.assertEqual(c.OUTPUT_LIMIT,134217728);self.assertEqual(c.RSS_LIMIT,12884901888);self.assertEqual(c.JOB_LIMIT,14000000000)

class Native:
 def astype(self,*args):return self
 def __pow__(self,n):return self
class Wave:
 dtype='float32';ndim=2;shape=(1,24000)
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return [Native()]
class Bool:
 def all(self):return self
 def item(self):return True

class Driver(unittest.TestCase):
 def exercise(self,fail_first=False):
  cfg=c.load_config();calls=[];seeds=[];refs=[]
  frontend=types.SimpleNamespace(allowed_special='all',spk2info={})
  class Model:
   sample_rate=24000
   def __init__(self):self.frontend=frontend
   def add_zero_shot_spk(self,prompt,path,pid):
    refs.append((prompt,path,pid));self.frontend.spk2info[pid]={};return True
   def inference_zero_shot(self,text,prompt,path,**kwargs):
    calls.append((text,prompt,path,kwargs,seeds[-1]))
    if fail_first:raise RuntimeError('intentional mock failure')
    yield {'tts_speech':Wave()}
  torch=types.SimpleNamespace(float32='float32',inference_mode=contextlib.nullcontext,manual_seed=seeds.append,isfinite=lambda x:Bool())
  np=types.ModuleType('numpy');np.random=types.SimpleNamespace(seed=lambda x:None);np.max=lambda x:.2;np.abs=lambda x:x;np.sqrt=lambda x:.1;np.mean=lambda x:.01;np.float64='float64';np.isfinite=lambda x:Bool();np.save=lambda path,x,allow_pickle=False:pathlib.Path(path).write_bytes(b'MOCK_NATIVE_NOT_AUDIO')
  sf=types.ModuleType('soundfile');sf.write=lambda path,x,rate,subtype:pathlib.Path(path).write_bytes(b'MOCK_PCM_NOT_AUDIO')
  ta=types.ModuleType('torchaudio');ta.transforms=types.SimpleNamespace(Resample=lambda **k:lambda x:x)
  cv=types.ModuleType('cosyvoice.cli.cosyvoice');cv.AutoModel=lambda **kwargs:Model()
  adapter=types.ModuleType('soundfile_adapter');adapter.bind_frontend=lambda path:None
  modules={'numpy':np,'soundfile':sf,'torchaudio':ta,'cosyvoice':types.ModuleType('cosyvoice'),'cosyvoice.cli':types.ModuleType('cosyvoice.cli'),'cosyvoice.cli.cosyvoice':cv,'soundfile_adapter':adapter}
  tokens={'clips':[dict(row,text=row['intended_text'],token_ids=[1,2]) for row in cfg['rows']]}
  with tempfile.TemporaryDirectory() as temp:
   root=pathlib.Path(temp);out=root/'outputs';out.mkdir()
   args=types.SimpleNamespace(source=str(root/'source'),model=str(root/'model'),reference=str(root/'refs'),output=str(out),state=str(root/'state.json'),result=str(root/'result.json'))
   with patch.dict(sys.modules,modules),patch.object(w,'cpu_environment',return_value=(torch,None)),patch.object(w,'strict_local_loaders'),patch.object(w,'assert_cpu_model'),patch.object(w,'token_contract',return_value=tokens):
    if fail_first:
     with self.assertRaises(RuntimeError):w.generate(args,[])
    else:w.generate(args,[])
   attempts=list(out.glob('*.attempt.json'));results=list(out.glob('*.result.json'))
   if not fail_first:
    self.assertEqual(json.loads((root/'result.json').read_text())['generated_clips'],30)
    for result in results:self.assertFalse(json.loads(result.read_text())['eos_proved'])
   return calls,seeds,refs,len(attempts),len(results)
 def test_exact30_no_hidden_warmup_and_five_references(self):
  calls,seeds,refs,attempts,results=self.exercise()
  self.assertEqual((len(calls),attempts,results),(30,30,30));self.assertEqual(seeds,list(range(610401,610431)))
  self.assertEqual(len(refs),5)
  for index,call in enumerate(calls):
   row=c.load_config()['rows'][index];self.assertEqual(call[0],row['intended_text']);self.assertEqual(call[-1],row['seed'])
   self.assertEqual(call[3],{'zero_shot_spk_id':'fixed30-'+row['stock_reference_voice'].lower(),'stream':False,'speed':1.0,'text_frontend':False})
 def test_first_eric_failure_stops_without_retry(self):
  calls,seeds,refs,attempts,results=self.exercise(fail_first=True)
  self.assertEqual((len(calls),attempts,results),(1,1,0));self.assertEqual(calls[0][0],'你好小窝');self.assertEqual(seeds,[610401])

if __name__=='__main__':unittest.main(verbosity=2)
