#!/usr/bin/env python3
"""One reviewed payload, one arming commit, one public CPU batch. No retries."""
import hashlib,json,os,pathlib,re,subprocess,sys,urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import runtime_lock
ROOT=pathlib.Path(__file__).resolve().parents[2]
REPOSITORY='jiying2007/kws-pipeline'
BRANCH='research/a20-cosy30-cross-voice-v1'
WORKFLOW='.github/workflows/research-cosyvoice3-cross-voice.yml'
DIRECTORY='research/cosyvoice3_cross_voice'
ARM=DIRECTORY+'/arm.json'
BASE='55a4e23379ad7072db507dbe419b38d8898e17fa'
BATCH='cosy30-cross-voice-20261004-v1'
HEAVY_JOB='one-time-cosy30'

def require(ok,message):
 if not ok:raise RuntimeError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def git(root,*args):return subprocess.check_output(['/usr/bin/git','-C',str(root),*args],text=True,timeout=30).strip()
def identity(root):
 files=[WORKFLOW]+sorted(str(p.relative_to(root)) for p in (root/DIRECTORY).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc') and str(p.relative_to(root))!=ARM)
 values={}
 for name in files:
  p=root/name;require(not p.is_symlink() and p.is_file(),'nonregular public payload')
  b=p.read_bytes();values[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 return hashlib.sha256(canonical(values)).hexdigest()
def check_activation(arm,payload):
 require(set(arm)=={'mode','reviewed_parent_sha','payload_sha256','batch_id'},'activation fields changed')
 require(arm['batch_id']==BATCH,'batch identity changed')
 if arm['mode']=='disabled':
  require(arm['reviewed_parent_sha'] is None and arm['payload_sha256'] is None,'disabled contract must not carry activation')
  return False
 require(arm['mode']=='run-fixed30-once','unknown activation mode')
 require(re.fullmatch('[0-9a-f]{40}',arm.get('reviewed_parent_sha') or '') is not None,'invalid reviewed parent')
 require(arm['payload_sha256']==payload,'reviewed payload changed')
 return True

def validate(event,env,head,parent,payload,paths,arm):
 require(check_activation(arm,payload),'generation disabled')
 require(env.get('GITHUB_EVENT_NAME')=='push' and env.get('GITHUB_REF')=='refs/heads/'+BRANCH,'wrong event/ref')
 require(env.get('GITHUB_RUN_ATTEMPT')=='1','reruns never admitted')
 repo=event.get('repository',{})
 require(repo.get('full_name')==REPOSITORY and repo.get('private') is False,'wrong/private repository')
 require(env.get('GITHUB_ACTOR')=='jiying2007','unexpected triggering actor')
 require(event.get('ref')=='refs/heads/'+BRANCH and event.get('deleted') is False,'wrong/deleted branch')
 require(event.get('before')==parent==arm['reviewed_parent_sha'],'activation parent mismatch')
 require(event.get('after')==head==env.get('GITHUB_SHA') and re.fullmatch('[0-9a-f]{40}',head),'activation head mismatch')
 require(paths==[ARM],'activation may change only arm.json')
 return {'batch_id':BATCH,'head_sha':head,'source_sha256':payload,'maximum_clips':30,'runtime_lock_sha256':runtime_lock.RAW_SHA256,'model_bytes':5427029103,'job_wall_seconds':3600,'controller_work_seconds':3300,'rss_limit_bytes':12884901888,'new_job_bytes_max':14000000000,'output_bytes_max':134217728,'artifact_retention_days':1,'cpu_capacity_bound_kind':'at-most4vCPU times60min; not aggregate process CPU accounting'}

def api(path,token):
 require(path.startswith('/repos/'+REPOSITORY+'/actions/'),'API scope mismatch')
 request=urllib.request.Request('https://api.github.com'+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'bounded-cosy30'})
 with urllib.request.urlopen(request,timeout=30) as response:data=response.read(8*1024**2+1)
 require(len(data)<=8*1024**2,'oversized API response')
 return json.loads(data)

def reject_prior_runs(fetch,current):
 for page in range(1,21):
  runs=fetch('/repos/'+REPOSITORY+'/actions/workflows/research-cosyvoice3-cross-voice.yml/runs?branch='+BRANCH+'&event=push&per_page=100&page='+str(page))['workflow_runs']
  for run in runs:
   if int(run['id'])==current:continue
   require(run.get('head_branch')==BRANCH,'prior branch mismatch')
   for jp in range(1,21):
    jobs=fetch('/repos/'+REPOSITORY+'/actions/runs/'+str(run['id'])+'/jobs?filter=all&per_page=100&page='+str(jp))['jobs']
    for job in jobs:
     if job.get('name')==HEAVY_JOB and job.get('conclusion')!='skipped':raise RuntimeError('Prior heavy job consumed the only batch; no automatic rerun')
    if len(jobs)<100:break
   else:raise RuntimeError('prior jobs exceed bounded audit')
  if len(runs)<100:return
 raise RuntimeError('prior run history exceeds bounded audit')

def main():
 arm=json.loads((ROOT/ARM).read_text());head=git(ROOT,'rev-parse','HEAD');parent=git(ROOT,'rev-parse','HEAD^')
 require(git(ROOT,'status','--porcelain')=='','working tree not clean')
 require(git(ROOT,'rev-list','--parents','-n','1',head).split()==[head,parent],'arming commit must have one parent')
 prior=json.loads(git(ROOT,'show',parent+':'+ARM))
 require(prior=={'mode':'disabled','reviewed_parent_sha':None,'payload_sha256':None,'batch_id':BATCH},'parent was not default-disarmed')
 git(ROOT,'fetch','--no-tags','--depth=1','origin',BASE)
 all_paths=git(ROOT,'diff','--name-only',BASE,head).splitlines()
 require(all_paths and all(p.startswith(DIRECTORY+'/') or p==WORKFLOW for p in all_paths),'base diff exceeds new experiment directory/workflow')
 event=json.loads(pathlib.Path(os.environ['GITHUB_EVENT_PATH']).read_text())
 activation=validate(event,os.environ,head,parent,identity(ROOT),git(ROOT,'diff','--name-only',parent,head).splitlines(),arm)
 token=os.environ.pop('GH_READ_TOKEN');reject_prior_runs(lambda path:api(path,token),int(os.environ['GITHUB_RUN_ID']))
 marker=pathlib.Path(os.environ['RUNNER_TEMP'])/'cosy30-admitted.json'
 with marker.open('x') as f:json.dump(activation,f,sort_keys=True)
 print(json.dumps({'batch_id':BATCH,'status':'admitted-once','head_sha':head,'payload_sha256':activation['source_sha256']},sort_keys=True))
if __name__=='__main__':main()
