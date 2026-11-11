#!/usr/bin/env python3
"""Link four validation probes to already-built ACTIVE113 SDKs. No execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SDKROOT=ROOT/'competitor-active-sdk-v1'
SDK=SDKROOT/'sdk-v2'
PACK=SDKROOT/'pack-v2'
SDK_BUILD=SDKROOT/'build-v2'
TOOLCHAIN=Path('/workspace/scratch/1e1fe64498b2/r1-toolchain/toolchain')
FLAGS=['-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
       '-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard','-fPIC']
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ident(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def load(path):return json.loads(path.read_bytes())
def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):continue
        need(not path.is_symlink(),'linked SDK source member')
        if path.is_file():result[path.relative_to(root).as_posix()]=ident(path)
    need(result,'empty generated SDK source tree');return result
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('declaration-sha256','sdk-plan-sha256','sdk-build-sha256','pack-sha256'):ap.add_argument('--'+name,required=True)
    args=ap.parse_args();os.umask(0o077)
    for value in vars(args).values():need(re.fullmatch('[0-9a-f]{64}',value),'SHA256 arguments required')
    declared=HERE/'DECLARED_PLAN.json';need(sha(declared)==args.declaration_sha256,'validation declaration changed')
    declaration=load(declared)
    need(args.sdk_plan_sha256==declaration['sdk_bindings']['sdk_plan_sha256'] and
         args.sdk_build_sha256==declaration['sdk_bindings']['sdk_build_receipt_sha256'] and
         args.pack_sha256==declaration['sdk_bindings']['pack_receipt_sha256'],'declared final v2 SDK identity changed')
    for name,value in declaration['source_files'].items():need(ident(Path(name))==value,'validation source changed')
    watched={str(declared):args.declaration_sha256,str(SDKROOT/'PLAN.json'):args.sdk_plan_sha256,
             str(SDK_BUILD/'BUILD_RECEIPT.json'):args.sdk_build_sha256,str(PACK/'PACK.json'):args.pack_sha256}
    for name,value in watched.items():need(sha(Path(name))==value,'SDK build/pack binding changed')
    pack=load(PACK/'PACK.json');sdk_build=load(SDK_BUILD/'BUILD_RECEIPT.json')
    need(sdk_build['schema']=='active113-sdk-build-v1' and sdk_build['status']=='PASS_BUILD_AND_INVENTED_LOADER_CHECKS' and
         sdk_build['preparation_plan_sha256']==args.sdk_plan_sha256 and sdk_build['sdk_path']==str(SDK) and
         sdk_build['pack_path']==str(PACK) and sdk_build['neural_model_calls']==sdk_build['pcm_calls']==0,
         'SDK build is not an admitted source-only/loader-only build')
    materialized=load(SDK/'MATERIALIZATION.json')
    need(sha(SDK/'MATERIALIZATION.json')==sdk_build['materialization_manifest_sha256'] and
         materialized['pack_manifest_sha256']==args.pack_sha256 and
         materialized['preparation_plan_sha256']==args.sdk_plan_sha256,'materialization binding changed')
    for name,value in materialized['generated_sources'].items():need(ident(SDK/name)==value,'materialized source changed')
    watched[str(SDK/'MATERIALIZATION.json')]=sdk_build['materialization_manifest_sha256']
    need(pack['schema']=='r1-active113-compact-pack-v1' and pack['status']=='PACKED_NO_MODEL_EXECUTION' and
         pack['selection_sha256']==declaration['selection_sha256'] and
         pack['screen_result_sha256']==declaration['cached_screen_sha256'] and
         pack['preparation_plan_sha256']==args.sdk_plan_sha256,'compact pack is not admitted')
    need(pack['model']['path']=='model-active113.r1a113' and pack['model']['bytes']==1582624 and
         pack['model']['sha256']==declaration['sdk_bindings']['compact_model_sha256'] and
         pack['frontend']['path']=='frontend.f32le','compact model shape/path changed')
    for field in ('model','frontend'):
        descriptor=pack[field];path=PACK/descriptor['path']
        need(ident(path)=={k:descriptor[k] for k in ('bytes','sha256')},'packed bytes changed')
        watched[str(path)]=descriptor['sha256']
    sources=inventory(SDK)
    libraries={arch:ident(SDK_BUILD/arch/'libactive113_sdk_static.a') for arch in ('host','arm')}
    for arch,value in libraries.items():
        expected=sdk_build['builds'][arch]['static_library']
        need(expected['path']==str(SDK_BUILD/arch/'libactive113_sdk_static.a') and
             value=={k:expected[k] for k in ('bytes','sha256')},'SDK library receipt mismatch')
    out=HERE/'build';out.mkdir(mode=0o700,exist_ok=False);(out/'logs').mkdir()
    receipt={'schema':'active113-validation-probe-build-v1','status':'BUILDING_NO_PCM',
        'declaration_sha256':args.declaration_sha256,'sdk_plan_sha256':args.sdk_plan_sha256,
        'sdk_build_receipt_sha256':args.sdk_build_sha256,'pack_receipt_sha256':args.pack_sha256,
        'source_files':declaration['source_files'],'generated_sdk_sources':sources,'input_libraries':libraries,
        'pack':pack,'commands':[],'builds':{},'pcm_calls':0,'neural_calls':0}
    def run(argv,env):
        argv=[str(x) for x in argv];number=len(receipt['commands']);receipt['commands'].append(argv)
        stdout=out/'logs'/f'{number:03d}.stdout';stderr=out/'logs'/f'{number:03d}.stderr'
        with stdout.open('xb') as so,stderr.open('xb') as se:
            result=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,timeout=60,check=False)
        need(result.returncode==0,'probe build failed; logs retained');return stdout.read_text()
    try:
        for arch in ('host','arm'):
            target=out/arch;target.mkdir();flags=list(FLAGS);env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
            if arch=='host':cc=Path('/usr/bin/gcc');readelf=Path('/usr/bin/readelf')
            else:
                cc=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-gcc-13';readelf=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-readelf'
                env['PATH']=str(TOOLCHAIN/'usr/bin')+os.pathsep+env['PATH'];env['LD_LIBRARY_PATH']=str(TOOLCHAIN/'usr/lib/x86_64-linux-gnu')
                flags+=['--sysroot='+str(TOOLCHAIN),'-mcpu=cortex-a32','-mfpu=neon-vfpv4','-mfloat-abi=hard']
            need(flags==sdk_build['builds'][arch]['flags'] and sha(cc)==sdk_build['builds'][arch]['compiler_sha256'],
                 'SDK/probe compiler configuration mismatch')
            lib=SDK_BUILD/arch/'libactive113_sdk_static.a';binaries={}
            for kind in ('plain','capture'):
                binary=target/('probe_'+kind)
                argv=[cc,*flags,'-I'+str(SDK/'include'),'-I'+str(SDK/'core')]
                if kind=='capture':argv+=['-DACTIVE113_CAPTURE=1','-Wl,--wrap=r1active113_step']
                run([*argv,HERE/'probe_active.c',lib,'-lm','-o',binary],env)
                elf=run([readelf,'-h','-W',binary],env)
                if arch=='arm':need(re.search(r'Class:\s+ELF32',elf) and re.search(r'Machine:\s+ARM\b',elf) and
                    'hard-float ABI' in elf,'ARM ELF/ABI mismatch')
                versions=run([readelf,'--version-info','-W',binary],env)
                required=sorted(set(re.findall(r'GLIBC_[0-9]+\.[0-9]+(?:\.[0-9]+)?',versions)),
                                key=lambda s:tuple(int(n) for n in s.removeprefix('GLIBC_').split('.')))
                binaries[kind]={'path':str(binary),**ident(binary),'required_glibc_symbol_versions':required}
            receipt['builds'][arch]={'binaries':binaries,'compiler':str(cc),'compiler_sha256':sha(cc),
                'flags':flags,'input_static_library_sha256':libraries[arch]['sha256']}
        for name,value in watched.items():need(sha(Path(name))==value,'identity changed during build')
        need(inventory(SDK)==sources,'generated SDK source drift during probe build')
        for arch,value in libraries.items():need(ident(SDK_BUILD/arch/'libactive113_sdk_static.a')==value,'library changed during build')
        receipt['status']='BUILT_PROBES_NO_PCM_OR_NEURAL_EXECUTION'
    except BaseException as exc:
        receipt.update(status='FAILED_PROBE_BUILD_NO_PCM',failure=type(exc).__name__+': '+str(exc));raise
    finally:save(out/'BUILD_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'build_receipt_sha256':sha(out/'BUILD_RECEIPT.json'),
        'builds':{a:{k:v['sha256'] for k,v in b['binaries'].items()} for a,b in receipt['builds'].items()},'pcm_calls':0},sort_keys=True))
if __name__=='__main__':main()
