"""Offline synthetic fixtures only; never calls GitHub or any model."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,unittest,zipfile,io,email.message,urllib.response
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import launcher as m

class ActivationTests(unittest.TestCase):
 def disabled(self):
  with tempfile.TemporaryDirectory() as t,patch.object(m,'api',side_effect=AssertionError('network forbidden')):
   p=Path(t)/'disabled.json';p.write_text(json.dumps({'enabled':False}))
   with self.assertRaisesRegex(ValueError,'disabled'):m.binding(p)
 def test_disabled_binding_has_no_network(self):self.disabled()
 def test_enabled_with_null_evidence_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   value=json.loads((HERE/'activation.json').read_bytes());value.update(enabled=True,input_artifacts=None);p=Path(t)/'unbound.json';p.write_text(json.dumps(value))
   with self.assertRaises(ValueError):m.binding(p)
 def test_template_not_installed_workflow(self):
  self.assertNotIn('.github',HERE.parts);text=(HERE/'workflow.yml').read_text();self.assertNotIn('workflow_dispatch',text);self.assertIn('github.run_attempt == 1',text);self.assertIn('github.run_number == 1',text);self.assertIn('pilot-asr-public/',text)
 def run_fixture(self):
  env={'GITHUB_REPOSITORY':m.REPO,'GITHUB_EVENT_NAME':'push','GITHUB_REF':'refs/heads/'+m.BRANCH,'GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_NUMBER':'1','GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40}
  event={'created':True,'deleted':False,'repository':{'private':False}}
  run={'id':123,'workflow_id':789,'head_sha':'a'*40,'head_branch':m.BRANCH,'path':m.WORKFLOW,'run_attempt':1,'run_number':1,'event':'push'}
  return env,event,run
 def test_exact_current_identity(self):
  env,event,run=self.run_fixture();m.verify_current({'workflow_id':789},env,event,run)
  for key,value in [('GITHUB_RUN_ATTEMPT','2'),('GITHUB_RUN_NUMBER','2'),('GITHUB_REF','refs/heads/main')]:
   with self.subTest(key=key),self.assertRaises(ValueError):m.verify_current({'workflow_id':789},dict(env,**{key:value}),event,run)
 def test_created_required(self):
  env,event,run=self.run_fixture()
  with self.assertRaises(ValueError):m.verify_current({'workflow_id':789},env,dict(event,created=False),run)
 def artifact_fixture(self):
  a={'run_id':123,'head_sha':'a'*40,'head_branch':'research/test-only','workflow_path':'.github/workflows/test-only.yml','run_attempt':1,'event':'push','conclusion':'success','artifact_id':456,'artifact_name':'fixture-only','zip_sha256':'b'*64}
  run={'id':123,'head_sha':'a'*40,'head_branch':a['head_branch'],'path':a['workflow_path'],'run_attempt':1,'event':'push','status':'completed','conclusion':'success'}
  artifact={'id':456,'name':'fixture-only','expired':False,'workflow_run':{'id':123,'head_sha':'a'*40},'digest':'sha256:'+'b'*64,'size_in_bytes':100}
  return a,run,artifact
 def test_upstream_identity(self):
  a,run,artifact=self.artifact_fixture();m.verify_upstream(a,run,artifact)
  with self.assertRaises(ValueError):m.verify_upstream(a,dict(run,head_sha='c'*40),artifact)
  with self.assertRaises(ValueError):m.verify_upstream(a,run,dict(artifact,expired=True))
  with self.assertRaises(ValueError):m.verify_upstream(a,run,dict(artifact,digest='sha256:'+'d'*64))
 def test_exact_zip_only(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);raw=b'fixture-only';row={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'restore_name':'qwen06-receipt.json'};a={'members':{'qwen06-receipt.json':row}}
   with zipfile.ZipFile(root/'ok.zip','w') as z:z.writestr('qwen06-receipt.json',raw)
   m.restore(a,root/'ok.zip',root/'ok');self.assertEqual((root/'ok/qwen06-receipt.json').read_bytes(),raw)
   with zipfile.ZipFile(root/'bad.zip','w') as z:z.writestr('qwen06-receipt.json',raw);z.writestr('runtime/weights.bin',b'not allowed')
   with self.assertRaises(ValueError):m.restore(a,root/'bad.zip',root/'bad')
 def test_member_hash_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   with zipfile.ZipFile(root/'x.zip','w') as z:z.writestr('x',b'abc')
   with self.assertRaises(ValueError):m.restore({'members':{'x':{'sha256':'0'*64,'bytes':3,'restore_name':'qwen06-receipt.json'}}},root/'x.zip',root/'out')
 def test_probe_not_replaced(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);plan=m.expected_plan()
   for e in ('qwen06','qwen17'):
    probe=dict(plan[e][0],status='generated')
    for phase,rows in [('probe',[probe]),('remaining',[dict(probe,status='changed')])]:
     d=root/(e+'-'+phase);d.mkdir();(d/(e+'-receipt.json')).write_text(json.dumps({'engine':e,'outcomes':rows}))
   with self.assertRaises(ValueError):m.combine(root,root/'merged',[{'engine':e,'phase':phase,'state':'recorded_run'} for e in ('qwen06','qwen17') for phase in ('probe','remaining')])
 def test_failed_probe_lane_not_restored_or_inferred(self):
  from finalize_input import finalize
  from test_screen import data
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);artifacts=[{'engine':e,'phase':p,'state':'not_run' if p=='remaining' else 'recorded_run'} for e in ('qwen06','qwen17') for p in ('probe','remaining')]
   m.combine(root,root/'inputs',artifacts);self.assertFalse((root/'inputs').exists())
   d=data()
   for row in d['slots']:
    if row['engine']!='cosy3':row.update(status='not_run',audio_quality_pass=False)
   (root/'manifest.json').write_text(json.dumps(d));(root/'admitted.json').write_text(json.dumps({'failed_probe_lanes':['qwen06'],'third_policy':'disagreement_only'}))
   out=finalize(root/'admitted.json',root/'manifest.json');lane=[r for r in out['slots'] if r['engine']=='qwen06']
   self.assertEqual([r['status'] for r in lane],['failed']+['not_run']*15);self.assertFalse(any('attempts' in r for r in lane));self.assertEqual(len(out['slots']),48)
 def test_binding_failed_lane_requires_actual_failure(self):
  # Synthetic local identities only; never submitted to GitHub.
  required={m.WORKFLOW,'research/source-screen32-v1/container-lock.json','numeric-fixture.txt'}
  required.update('research/tts-pilot-screening-v1/'+p.name for p in m.BASE.iterdir() if p.is_file() and p.suffix in ('.py','.json'))
  required.update('research/tts-pilot-screening-v1/activation-template/'+n for n in ('launcher.py','finalize_input.py','test_activation.py'))
  artifacts=[]
  for engine in ('qwen06','qwen17'):
   for phase in ('probe','remaining'):
    if engine=='qwen06' and phase=='remaining':
     artifacts.append({'engine':engine,'phase':phase,'state':'not_run','reason':'verified_probe_failed'});continue
    artifacts.append({'engine':engine,'phase':phase,'state':'recorded_run','run_id':123,'artifact_id':456,'head_sha':'a'*40,'zip_sha256':'b'*64,'artifact_name':'synthetic-local-fixture','conclusion':'failure','lane_outcome':'failed' if engine=='qwen06' else 'succeeded','run_attempt':1,'event':'push','workflow_path':'.github/workflows/test-only.yml','head_branch':'research/test-only','members':{'host-receipt.json':{'sha256':'c'*64,'bytes':2,'restore_name':'host-receipt.json'}}})
  value={'schema':'tts-pilot-asr-activation-v1','third_policy':'disagreement_only','enabled':True,'repository':m.REPO,'branch':m.BRANCH,'workflow_path':m.WORKFLOW,'workflow_id':789,'max_asr_slots':144,'current_qwen_call_cap':96,'input_artifacts':artifacts,'source_files':{p:'d'*64 for p in required},'numeric_requirements_file':'numeric-fixture.txt'}
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'fixture.json';p.write_text(json.dumps(value));self.assertEqual(m.binding(p)['input_artifacts'][1]['state'],'not_run')
   value['input_artifacts'][0]['lane_outcome']='succeeded';p.write_text(json.dumps(value))
   with self.assertRaises(ValueError):m.binding(p)
 def test_shared_failed_run_can_have_successful_probe_lane(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);item={'engine':'qwen17','head_sha':'a'*40,'conclusion':'failure','lane_outcome':'succeeded'}
   (root/'host-receipt.json').write_text(json.dumps({'engine':'qwen17','sha':'a'*40,'campaign':'tts-quality-probe-20261014-v1','status':'probe_completed_pending_human_review'}))
   m.verify_probe_lane(item,root)
   with self.assertRaises(ValueError):m.verify_probe_lane(dict(item,lane_outcome='failed'),root)
 def test_workflow_id_cannot_reset(self):
  env,event,run=self.run_fixture()
  with self.assertRaises(ValueError):m.verify_current({'workflow_id':790},env,event,run)
 def test_api_bearer_redirects_are_rejected(self):
  # Exercise urllib's real redirect/error handlers with a fake HTTPS transport.
  # No socket opens and no real credential is used.
  original_builder=m.urllib.request.build_opener
  for code in (301,302,303,307,308):
   calls=[]
   class FixtureHTTPS(m.urllib.request.HTTPSHandler):
    def https_open(self,request):
     calls.append((request.full_url,request.get_header('Authorization')))
     headers=email.message.Message();headers['Location']='https://external.invalid/never-follow'
     response=urllib.response.addinfourl(io.BytesIO(b''),headers,request.full_url,code)
     response.msg='Fixture redirect';return response
   def build(*handlers):
    self.assertIn(m.NoRedirect,handlers)
    return original_builder(*handlers,FixtureHTTPS())
   with self.subTest(code=code),patch.object(m.urllib.request,'build_opener',side_effect=build),patch.object(m.urllib.request,'urlopen',side_effect=AssertionError('default redirecting opener forbidden')):
    with self.assertRaises(m.urllib.error.HTTPError) as error:m.api('/repos/'+m.REPO+'/actions/runs/123','synthetic-test-token')
    self.assertEqual(error.exception.code,code)
   self.assertEqual(calls,[('https://api.github.com/repos/'+m.REPO+'/actions/runs/123','Bearer synthetic-test-token')])
 def test_no_write_permission_or_api(self):
  text=(HERE/'workflow.yml').read_text();self.assertNotIn('contents: write',text);self.assertNotIn('/git/refs',(HERE/'launcher.py').read_text())
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(ActivationTests)
 if suite.countTestCases()!=15:raise RuntimeError('activation test count')
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not result.wasSuccessful())
