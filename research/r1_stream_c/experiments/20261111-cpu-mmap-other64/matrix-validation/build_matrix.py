#!/usr/bin/env python3
"""Build only: unchanged PR787 probe plus an external validation wrapper; no inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SOURCE_HEAD = '13661a38064247d38bca88287e48db85bf2ccd79'
HERE = Path(__file__).resolve().parent
SOURCES = ('core/r1d_stream.c', 'core/r1d_pcm.c', 'core/reference64_fbank.c',
           'core/r1dfsmn.c', 'core/sha256.c', 'src/fsmn_sdk.c', 'src/fsmn_detector.c')
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
         '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard', '-fPIC']

def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')
        f.flush(); os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','toolchain','output'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    repo,toolchain,out=(getattr(args,k).resolve() for k in ('repo','toolchain','output'))
    os.umask(0o077)
    out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir(); (out/'sdk').mkdir()
    probe=repo/'research/r1_stream_c/quant_head8/target_resource_bench.c'
    wrapper=HERE/'trace_wrap.c'
    receipt={'schema':'r1-mmap-matrix-build-v1','source_head':SOURCE_HEAD,
             'status':'BUILDING','model_executions':0,'commands':[], 'builds':{},
             'source_probe':dict(identity(probe),path=str(probe)),
             'wrapper':dict(identity(wrapper),path=str(wrapper)),
             'build_script':identity(Path(__file__)),'generated_sources':{}}
    env=dict(os.environ); env.pop('LD_LIBRARY_PATH',None); env['PYTHONDONTWRITEBYTECODE']='1'
    def run(command,environment):
        command=[str(x) for x in command]
        index=len(receipt['commands']); receipt['commands'].append(command)
        with (out/'logs'/f'{index:03d}.stdout').open('xb') as so, (out/'logs'/f'{index:03d}.stderr').open('xb') as se:
            result=subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,
                                  stdout=so,stderr=se,timeout=90,check=False)
        if result.returncode: raise RuntimeError(f'build command {index} failed; logs retained')
        return (out/'logs'/f'{index:03d}.stdout').read_text()
    try:
        for kernel in ('par4','block2','block4'):
            sdk=out/'sdk'/kernel
            mod='borrowed_weights' if kernel=='par4' else 'cpu_opt'
            command=[sys.executable,'-I','-S','-B',repo/f'research/r1_stream_c/{mod}/materialize.py',
                     '--repo',repo,'--output',sdk]
            if kernel!='par4': command+=['--kernel',kernel]
            run(command,env)
            receipt['generated_sources'][kernel]={str(p.relative_to(sdk)):identity(p)
                         for p in sorted(sdk.rglob('*')) if p.is_file()}
            for arch in ('host','arm'):
                target=out/kernel/arch;target.mkdir(parents=True)
                localenv=dict(env); flags=list(FLAGS)
                if arch=='arm':
                    localenv['PATH']=str(toolchain/'usr/bin')+os.pathsep+localenv['PATH']
                    localenv['LD_LIBRARY_PATH']=str(toolchain/'usr/lib/x86_64-linux-gnu')
                    cc=toolchain/'usr/bin/arm-linux-gnueabihf-gcc-13'
                    ar=toolchain/'usr/bin/arm-linux-gnueabihf-ar'
                    readelf=toolchain/'usr/bin/arm-linux-gnueabihf-readelf'
                    flags+=['--sysroot='+str(toolchain),'-mcpu=cortex-a32','-mfpu=neon-vfpv4','-mfloat-abi=hard']
                else:
                    cc,ar,readelf=(Path('/usr/bin')/x for x in ('gcc','ar','readelf'))
                include=['-I'+str(sdk/'include'),'-I'+str(sdk/'core')]
                objects=[]
                for source in SOURCES:
                    obj=target/(Path(source).stem+'.o')
                    run([cc,*flags,*include,'-c',sdk/source,'-o',obj],localenv);objects.append(obj)
                lib=target/'libfsmn_sdk_static.a'
                run([ar,'rcsD',lib,*objects],localenv)
                binaries={}
                for kind in ('plain','capture'):
                    binary=target/('probe_'+kind)
                    command=[cc,*flags,*include,probe]
                    if kind=='capture':
                        command+=[wrapper,'-Wl,--wrap=fsmn_sdk_create',
                                  '-Wl,--wrap=fsmn_sdk_create_view','-Wl,--wrap=fsmn_sdk_destroy']
                    run([*command,lib,'-lm','-o',binary],localenv)
                    headers=run([readelf,'-h','-W',binary],localenv)
                    if arch=='arm' and not (re.search(r'Class:\s+ELF32',headers) and
                        re.search(r'Machine:\s+ARM\b',headers) and 'hard-float ABI' in headers):
                        raise ValueError('ARM ELF/ABI mismatch')
                    binaries[kind]=dict(identity(binary),path=str(binary))
                receipt['builds'][kernel+'_'+arch]={'binaries':binaries,'flags':flags,
                     'compiler':run([cc,'--version'],localenv).splitlines()[0],
                     'compiler_sha256':identity(cc)['sha256'],'static_library':identity(lib)}
        receipt['status']='BUILT_NO_MODEL_EXECUTION'
    except BaseException as exc:
        receipt['status']='FAILED_NO_MODEL_EXECUTION';receipt['failure']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'BUILD_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'builds':len(receipt['builds']),'model_executions':0}))

if __name__=='__main__': main()
