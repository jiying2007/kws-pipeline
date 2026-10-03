#!/usr/bin/env python3
"""Stage only synthetic clips admitted by pilot verifier plus bounded receipts."""
import hashlib,json,os,pathlib,shutil,stat,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LIMIT=128*1024**2
EXPLICIT=['controller-state.json','receipts/runtime/install-receipt.json','receipts/runtime/realized-wheel-lock.json','receipts/runtime/runtime-qualification.json']

def main():
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!='jiying2007/kws-pipeline':raise RuntimeError('Public runner only')
 job=pathlib.Path(os.environ['RUNNER_TEMP'])/'cosyvoice3-job'
 if not job.exists():
  print('No admitted pilot work tree exists; no artifact upload prepared');return 0
 destination=ROOT/'cosyvoice3-public-artifact'
 if destination.exists():raise RuntimeError('Artifact stage already exists; no overwrite/retry')
 # Preflight before the verifier or any staging copy writes bytes. The selected
 # generation receipts are a subset of all JSON receipts; raw dependency logs
 # are retained but not recursively published.
 retained=0;stage_upper=1024**2
 for folder in ['outputs','receipts','logs']:
  for f in (job/folder).rglob('*'):
   if f.is_symlink():raise RuntimeError('Linked retained output')
   if f.is_file():retained+=f.stat().st_size
 for folder in ['outputs','receipts/generation','logs']:
  for f in (job/folder).rglob('*'):
   if f.is_file() and (folder!='receipts/generation' or f.suffix=='.json'):stage_upper+=f.stat().st_size
 for name in EXPLICIT:
  f=job/name
  if f.exists():stage_upper+=f.stat().st_size
 if retained+2*stage_upper+1024**2>LIMIT:raise RuntimeError('Publication preflight exceeds128MiB envelope')
 p=subprocess.run([sys.executable,'-I','-S',str(HERE/'pilot.py'),'verify-output','--job-root',str(job),'--artifact-dir',str(destination)],timeout=120,check=False)
 if p.returncode:raise RuntimeError('Output allowlist verification failed; no publication')
 receipts=destination/'technical';receipts.mkdir(exist_ok=False)
 for name in EXPLICIT:
  source=job/name
  if not source.exists():continue
  st=source.lstat()
  if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size>2*1024**2:raise RuntimeError('Unsafe/oversized technical receipt')
  json.loads(source.read_text());shutil.copyfile(source,receipts/name.replace('/','__'))
 # Logs are from this isolated controller only; never copy model/runtime/source trees.
 for source in sorted((job/'logs').glob('*.log')):
  if source.name not in {'acquire-source.log','install-runtime.log','qualify-runtime.log','qualify-source.log','acquire-model.log','generate-fixed-pairs.log'}:raise RuntimeError('Unexpected phase log')
  st=source.lstat()
  if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size>8*1024**2:raise RuntimeError('Unsafe/oversized phase log')
  shutil.copyfile(source,receipts/source.name)
 manifest={};total=0
 for p in sorted(destination.rglob('*')):
  if p.is_symlink():raise RuntimeError('Artifact contains symlink')
  if p.is_dir():continue
  st=p.lstat()
  if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:raise RuntimeError('Artifact contains nonregular entry')
  b=p.read_bytes();total+=len(b);manifest[str(p.relative_to(destination))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 data=json.dumps({'schema':'cosyvoice3.public-artifact.v1','run_id':os.environ['GITHUB_RUN_ID'],'retention_days':1,'artifact_max_bytes':LIMIT,'payload_bytes':total,'files':manifest},sort_keys=True,indent=2).encode()+b'\n'
 if total+len(data)>LIMIT:raise RuntimeError('Combined publication exceeds128MiB')
 # Account for retained outputs/receipts/logs, this staging copy, and the
 # upload action's uncompressed ZIP, with1MiB explicit framing allowance.
 retained=0
 for folder in ['outputs','receipts','logs']:
  for f in (job/folder).rglob('*'):
   if f.is_symlink():raise RuntimeError('Linked retained output')
   if f.is_file():retained+=f.stat().st_size
 peak=retained+2*(total+len(data))+1024**2
 if peak>LIMIT:raise RuntimeError('Outputs, receipts, staging and ZIP exceed128MiB envelope')
 # Only a fully validated, final manifest enables the workflow upload step.
 with (destination/'manifest.json').open('xb') as f:f.write(data)
 print(json.dumps({'publication':'verified','files':len(manifest),'bytes':total+len(data),'retention_days':1},sort_keys=True))
 return 0
if __name__=='__main__':raise SystemExit(main())
