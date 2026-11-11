#!/usr/bin/env python3
"""Build the frozen cached-head predictor only; never execute it."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    os.umask(0o077)
    plan=json.loads((HERE/'PLAN.json').read_text())
    for name,expected in plan['source_files'].items():
        p=Path(name);need(p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'],'source changed')
    compiler=Path(plan['compiler']['path'])
    need(sha(compiler)==plan['compiler']['sha256'],'compiler changed')
    out=HERE/'build';out.mkdir(mode=0o700,exist_ok=False)
    binary=out/'active_head'
    sdk=Path(plan['generated_par4_sdk'])
    argv=[str(compiler),*plan['compiler']['flags'],'-I'+str(sdk/'core'),'-I'+str(HERE),
          str(HERE/'active_head.c'),str(sdk/'core/sha256.c'),'-lm','-o',str(binary)]
    receipt={'schema':'active113-cached-head-build-v1','status':'BUILDING',
             'declared_plan_sha256':sha(HERE/'PLAN.json'),'argv':argv,
             'source_files':plan['source_files'],'compiler':plan['compiler'],'model_calls':0}
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env['LC_ALL']='C'
    start=time.monotonic()
    try:
        with (out/'stdout.txt').open('xb') as so,(out/'stderr.txt').open('xb') as se:
            result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,timeout=60,check=False)
        need(result.returncode==0,'build failed; logs retained')
        receipt.update(status='BUILT_NOT_EXECUTED',binary={'path':str(binary),'bytes':binary.stat().st_size,'sha256':sha(binary)})
    except BaseException as exc:
        receipt.update(status='FAILED_BUILD_NOT_EXECUTED',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt['build_seconds']=time.monotonic()-start;save(out/'BUILD_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'binary':receipt['binary'],'model_calls':0},sort_keys=True))
if __name__=='__main__':main()
