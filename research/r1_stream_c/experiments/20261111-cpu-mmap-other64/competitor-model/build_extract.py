#!/usr/bin/env python3
"""Compile the fixed teacher extractor against the retained par4 static SDK; no inference."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SDK=ROOT/'matrix-build/sdk/par4'
LIB=ROOT/'matrix-build/par4/host/libfsmn_sdk_static.a'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    os.umask(0o077)
    plan=json.loads((HERE/'PLAN.json').read_text())
    for name,digest in plan['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('predeclared source changed: '+name)
    if sha(LIB)!=plan['static_sdk']['sha256']:raise ValueError('SDK library changed')
    if sha(Path('/usr/bin/gcc').resolve())!=plan['compiler']['sha256']:raise ValueError('compiler changed')
    for name,digest in plan['sdk_header_sha256'].items():
        if sha(SDK/name)!=digest:raise ValueError('SDK header changed')
    out=HERE/'build';out.mkdir(exist_ok=False)
    binary=out/'extract_teacher'
    command=['/usr/bin/gcc',*plan['compiler_flags'],'-I'+str(SDK/'include'),'-I'+str(SDK/'core'),
             str(HERE/'extract_teacher.c'),'-Wl,--wrap=r1dfsmn_step',str(LIB),'-lm','-o',str(binary)]
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    receipt={'schema':'r1-hidden-teacher-build-v1','status':'BUILDING','model_calls':0,
             'predeclared_plan_sha256':sha(HERE/'PLAN.json'),'source_sha256':plan['source_sha256'],
             'static_sdk_sha256':sha(LIB),'command':command,
             'compiler_sha256':sha(Path('/usr/bin/gcc').resolve())}
    try:
        with (out/'stdout.log').open('xb') as so,(out/'stderr.log').open('xb') as se:
            result=subprocess.run(command,stdout=so,stderr=se,stdin=subprocess.DEVNULL,env=env,timeout=60)
        if result.returncode:raise RuntimeError('teacher build failed; logs retained')
        receipt.update(status='BUILT_NO_INFERENCE',binary={'path':str(binary),'bytes':binary.stat().st_size,'sha256':sha(binary)})
    except BaseException as exc:
        receipt.update(status='FAILED_NO_INFERENCE',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        with (out/'BUILD_RECEIPT.json').open('x') as f:
            json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':receipt['status'],'binary':receipt['binary'],'model_calls':0},sort_keys=True))
if __name__=='__main__':main()
