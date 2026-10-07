#!/usr/bin/env python3
"""One fresh runner work tree, bounded dependency/source/model phases, no retries."""
import ctypes, hashlib, json, os, pathlib, shutil, signal, subprocess, sys, time, traceback
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import admit_run, runtime_lock
from bounded_log import BoundedLog
MAX_RSS=12*1024**3; MIN_AVAILABLE=2*1024**3; RESERVE=1024**3; TOTAL=14_000_000_000; LOG_MAX=8*1024**2

def require(x,message):
 if not x:raise RuntimeError(message)
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def host_available():return int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
SUBREAPER_PID=None
def enable_subreaper():
 global SUBREAPER_PID
 # Linux adopts orphaned descendants here, including a rapid setsid/double-fork
 # that completes between samples. Fail closed if the runner cannot enforce it.
 require(sys.platform=='linux','Whole-process-tree supervision requires Linux')
 libc=ctypes.CDLL(None,use_errno=True);libc.prctl.restype=ctypes.c_int
 require(libc.prctl(36,1,0,0,0)==0,'Cannot enable child subreaper: '+str(ctypes.get_errno()))
 value=ctypes.c_int(0)
 require(libc.prctl(37,ctypes.byref(value),0,0,0)==0 and value.value==1,'Cannot verify child subreaper')
 SUBREAPER_PID=os.getpid()
def process_snapshot():
 rows={}
 for path in pathlib.Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   fields=(path/'stat').read_text().rsplit(')',1)[1].split()
   rows[int(path.name)]={'parent':int(fields[1]),'start':fields[19],'state':fields[0],'rss':max(0,int(fields[21]))*os.sysconf('SC_PAGE_SIZE')}
  except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):continue
 return rows
def process_identity(pid):
 try:return pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):return None
def owned_processes(pid,observed=None,rows=None):
 # A recorded child remains owned after reparenting; PID identity must match.
 # Include its subsequent descendants using the same coherent /proc snapshot.
 rows=process_snapshot() if rows is None else rows
 known=observed or {};selected={p for p,start in known.items() if p in rows and rows[p]['start']==start}
 if pid in rows and (pid not in known or rows[pid]['start']==known[pid]):selected.add(pid)
 if SUBREAPER_PID==os.getpid():
  selected.update(p for p,row in rows.items() if row['parent']==SUBREAPER_PID)
 while True:
  enlarged=selected|{p for p,row in rows.items() if row['parent'] in selected}
  if enlarged==selected:break
  selected=enlarged
 return {p:rows[p]['start'] for p in selected}
def descendants(pid):return owned_processes(pid)
def tree_rss(pid,observed=None):
 rows=process_snapshot();owned=owned_processes(pid,observed,rows)
 return sum(rows[p]['rss'] for p in owned)
def live_owned(pid,observed):
 rows=process_snapshot();owned=owned_processes(pid,observed,rows)
 return {p:start for p,start in owned.items() if rows[p]['state'] not in {'Z','X'}}
def terminate(p,observed=None):
 owned=owned_processes(p.pid,observed)
 # Freeze the owned group and all identity-matched descendants before killing.
 try:os.killpg(p.pid,signal.SIGSTOP)
 except ProcessLookupError:pass
 for _ in range(3):
  owned.update(owned_processes(p.pid,owned))
  for pid,start in list(owned.items()):
   if process_identity(pid)==start:
    try:os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError:pass
 for pid,start in sorted(owned.items(),reverse=True):
  if process_identity(pid)==start:
   try:os.kill(pid,signal.SIGKILL)
   except ProcessLookupError:pass
 try:os.killpg(p.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 p.wait(timeout=10)
 deadline=time.monotonic()+2
 while live_owned(p.pid,owned):
  require(time.monotonic()<deadline,'Owned processes survived phase cleanup')
  time.sleep(.02)
 # Reap only verified owned adopted children; the subprocess leader was reaped above.
 for pid,start in owned.items():
  if process_identity(pid)==start:
   try:os.waitpid(pid,os.WNOHANG)
   except ChildProcessError:pass
 return True
class Controller:
 def __init__(self,root):
  self.root=root;root.mkdir(exist_ok=False);(root/'receipts').mkdir();(root/'logs').mkdir()
  self.free=shutil.disk_usage(root).free;self.started=time.monotonic();self.steps=[]
  self.state={'schema':'cosyvoice3.controller.v1','status':'running','baseline_free_bytes':self.free,'new_job_bytes_max':TOTAL,'started_monotonic':self.started,'steps':self.steps,'run_id':os.environ['GITHUB_RUN_ID'],'head_sha':admit_run.git(HERE.parents[1],'rev-parse','HEAD')}
  write(root/'controller-state.json',self.state)
 def check(self,p=None,log=None,observed=None):
  free=shutil.disk_usage(self.root).free;used=max(0,self.free-free)
  require(free>=RESERVE and used+RESERVE<=TOTAL,'Whole-job disk reserve/14GB budget exceeded')
  require(host_available()>=MIN_AVAILABLE,'Host available memory below 2GiB')
  rss=tree_rss(p.pid,observed) if p else 0;require(rss<=MAX_RSS,'Process tree exceeded12GiB RSS')
  require(time.monotonic()-self.started<55*60,'Overall 55-minute work deadline exceeded; publication reserve before60min job limit')
  if log:require(log.stat().st_size<=LOG_MAX,'Phase log exceeded8MiB')
  return rss,used,free
 def phase(self,name,cmd,timeout):
  enable_subreaper()
  require(not {p:start for p,start in live_owned(os.getpid(),{}).items() if p!=os.getpid()},'Previous phase left live owned children')
  self.check();start=time.monotonic();log=self.root/'logs'/(name+'.log')
  record={'phase':name,'status':'running','command':[str(x) for x in cmd],'max_rss_bytes':0,'peak_job_bytes':0}
  self.steps.append(record);write(self.root/'controller-state.json',self.state)
  env=dict(os.environ)
  for key in ['GH_READ_TOKEN','GITHUB_TOKEN','ACTIONS_RUNTIME_TOKEN','ACTIONS_ID_TOKEN_REQUEST_TOKEN','PYTHONPATH','PYTHONOPTIMIZE']:env.pop(key,None)
  env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PIP_NO_CACHE_DIR='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',WANDB_DISABLED='true')
  env['HOME']=str(self.root/'home');pathlib.Path(env['HOME']).mkdir(exist_ok=True)
  with log.open('xb') as f:
   p=subprocess.Popen([str(x) for x in cmd],cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
   observed=descendants(p.pid);capture=None
   try:
    capture=BoundedLog(p.stdout,f,LOG_MAX)
    while True:
     capture.pump(timeout=.2)
     require(not capture.truncated,'Phase log exceeded byte limit')
     observed.update(owned_processes(p.pid,observed))
     rss,used,free=self.check(p,log,observed);record['max_rss_bytes']=max(record['max_rss_bytes'],rss);record['peak_job_bytes']=max(record['peak_job_bytes'],used)
     require(time.monotonic()-start<=timeout,'Phase deadline exceeded: '+name)
     code=p.poll()
     if code is not None:break
     if capture.eof:time.sleep(.2)
    record['returncode']=code;require(code==0,'Phase failed: '+name+' (exit '+str(code)+')')
    record['status']='passed'
   except BaseException as exc:
    record['status']='failed';record['error']=str(exc);raise
   finally:
    try:
     try:record['cleanup_verified']=terminate(p,observed)
     finally:
      if capture is not None:capture.finish()
     require(capture is not None and not capture.truncated,'Phase log exceeded byte limit')
     require(capture.eof,'Phase log pipe remained open after cleanup')
    except BaseException as exc:
     record['status']='failed';record['cleanup_error']=str(exc);raise
    finally:
     record['elapsed_seconds']=time.monotonic()-start
     record['log_bytes']=log.stat().st_size;record['log_limit_bytes']=LOG_MAX;record['log_truncated']=capture.truncated if capture else False;record['log_observed_bytes']=capture.observed_bytes if capture else 0;record['log_eof']=capture.eof if capture else False;record['log_drain_timed_out']=capture.drain_timed_out if capture else False
     with log.open('rb') as final_log:record['log_sha256']=hashlib.file_digest(final_log,'sha256').hexdigest()
     write(self.root/'controller-state.json',self.state)
  print(json.dumps({'phase':name,'status':'passed','elapsed_seconds':record['elapsed_seconds'],'max_rss_bytes':record['max_rss_bytes']},sort_keys=True),flush=True)
 def execute(self):
  python=pathlib.Path(sys.executable); lock=runtime_lock.materialize(self.root/'runtime-input-lock.json'); runtime=self.root/'runtime'; receipts=self.root/'receipts'/'runtime'; receipt=receipts/'runtime-qualification.json'
  self.phase('acquire-source',[python,'-I','-S',HERE/'pilot.py','acquire-source','--job-root',self.root],600)
  self.phase('install-runtime',[python,'-I','-S',HERE/'install_runtime.py','--lock',lock,'--work',runtime,'--receipts',receipts],2700)
  vpython=runtime/'venv/bin/python'
  self.phase('qualify-runtime',[vpython,'-I',HERE/'qualify_runtime.py','--lock',lock,'--output',receipt],600)
  common=['--job-root',self.root,'--runtime-lock',lock,'--runtime-receipt',receipt]
  self.phase('qualify-source',[vpython,'-I',HERE/'pilot.py','qualify-source',*common],600)
  self.phase('acquire-model',[vpython,'-I',HERE/'pilot.py','acquire-model',*common],1800)
  self.phase('generate-fixed30',[vpython,'-I',HERE/'pilot.py','run',*common],3300)
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('GITHUB_REPOSITORY')==admit_run.REPOSITORY,'Public GitHub runner only')
 require(os.cpu_count()<=4,'Standard4CPU limit changed')
 repo=HERE.parents[1];approved=json.loads((pathlib.Path(os.environ['RUNNER_TEMP'])/'cosy30-admitted.json').read_text())
 require(approved['source_sha256']==admit_run.identity(repo),'Source changed after admission')
 root=pathlib.Path(os.environ['RUNNER_TEMP'])/'cosy30-job';c=Controller(root)
 try:c.execute();c.state['status']='completed'
 except BaseException as exc:c.state['status']='failed';c.state['error']=str(exc);c.state['traceback']=traceback.format_exc()
 finally:c.state['elapsed_seconds']=time.monotonic()-c.started;write(root/'controller-state.json',c.state)
 print(json.dumps({'status':c.state['status'],'error':c.state.get('error')},sort_keys=True));return 0 if c.state['status']=='completed' else 1
if __name__=='__main__':sys.exit(main())
