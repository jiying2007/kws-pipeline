#!/usr/bin/env python3
"""Build namespaced ACTIVE113 libraries and run invented loader checks only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from active_format import HERE,ROOT,base_binding,load,need,save,screen_gate,sha,source_gate

FLAGS=['-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic','-fno-fast-math',
       '-ffp-contract=off','-fexcess-precision=standard','-fPIC']
SOURCES=('core/r1a113_stream.c','core/r1a113_pcm.c','core/reference64_fbank.c',
         'core/r1active113.c','core/sha256.c','core/active113_layout.c',
         'src/active113_sdk.c','src/active113_detector.c')
TOOLCHAIN=Path('/workspace/scratch/1e1fe64498b2/r1-toolchain/toolchain')


def identity(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--sdk-manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sdk=a.sdk.resolve();out=a.output.resolve()
    need(not out.exists(),'fresh build directory required');source_gate(a.plan_sha256);base_binding()
    manifest=sdk/'MATERIALIZATION.json';need(sha(manifest)==a.sdk_manifest_sha256,'SDK manifest changed')
    material=load(manifest)
    need(material['status']=='MATERIALIZED_NO_MODEL_EXECUTION' and
         material['preparation_plan_sha256']==a.plan_sha256,'SDK preparation binding')
    screen_gate(ROOT/'competitor-active-v1/execution/AGGREGATE.json',material['screen_result_sha256'])
    def unchanged():
        need(sha(manifest)==a.sdk_manifest_sha256,'SDK manifest drift')
        for name,value in material['generated_sources'].items():need(sha(sdk/name)==value['sha256'],'generated source drift:'+name)
        need(sha(sdk/'MODEL_CARD.json')==material['model_card_sha256'],'model card drift')
    unchanged();original_build=load(ROOT/'matrix-build/BUILD_RECEIPT.json')
    execution=load(ROOT/'matrix-execution/PLAN.json');qemu=Path(execution['qemu']['path'])
    need(sha(qemu.resolve())==execution['qemu']['sha256'],'QEMU identity drift')
    os.umask(0o077);out.mkdir(parents=True);(out/'logs').mkdir()
    receipt={'schema':'active113-sdk-build-v1','status':'BUILDING','preparation_plan_sha256':a.plan_sha256,
        'materialization_manifest_sha256':a.sdk_manifest_sha256,'sdk_path':str(sdk),
        'pack_path':material['pack_path'],'compact_model_sha256':material['compact_model_sha256'],
        'screen_result_sha256':material['screen_result_sha256'],'selection_sha256':material['selection_sha256'],
        'policy':material['policy'],'builds':{},'commands':[],'neural_model_calls':0,'pcm_calls':0,
        'loader_fixture':'invented bytes only; no network, feature, frontend, or PCM execution',
        'model_promotion_authorized':False}
    def run(argv,env,timeout=90):
        argv=[str(x) for x in argv];i=len(receipt['commands']);receipt['commands'].append(argv)
        stdout=out/'logs'/f'{i:03d}.stdout';stderr=out/'logs'/f'{i:03d}.stderr'
        with stdout.open('xb') as so,stderr.open('xb') as se:
            process=subprocess.run(argv,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,timeout=timeout,check=False)
        need(process.returncode==0 and stderr.stat().st_size==0,'build/check command failed: '+str(i))
        return stdout.read_text()
    try:
        for arch in ('host','arm'):
            target=out/arch;target.mkdir();env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
            flags=list(FLAGS)
            if arch=='host':cc,ar,nm,readelf=(Path('/usr/bin')/x for x in ('gcc','ar','nm','readelf'))
            else:
                env['PATH']=str(TOOLCHAIN/'usr/bin')+os.pathsep+env['PATH']
                env['LD_LIBRARY_PATH']=str(TOOLCHAIN/'usr/lib/x86_64-linux-gnu')
                cc=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-gcc-13'
                ar=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-ar';nm=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-nm'
                readelf=TOOLCHAIN/'usr/bin/arm-linux-gnueabihf-readelf'
                flags+=['--sysroot='+str(TOOLCHAIN),'-mcpu=cortex-a32','-mfpu=neon-vfpv4','-mfloat-abi=hard']
            need(sha(cc.resolve())==original_build['builds']['par4_'+arch]['compiler_sha256'],'compiler changed')
            includes=['-I'+str(sdk/'include'),'-I'+str(sdk/'core')];objects=[]
            for name in SOURCES:
                obj=target/(Path(name).stem+'.o');run([cc,*flags,*includes,'-c',sdk/name,'-o',obj],env);objects.append(obj)
            library=target/'libactive113_sdk_static.a';run([ar,'rcsD',library,*objects],env)
            inventory=run([nm,'-g','--defined-only',library],env);symbols=[]
            for line in inventory.splitlines():
                parts=line.split()
                if len(parts)==3 and re.fullmatch('[A-Za-z]',parts[1]):symbols.append(parts[2])
            need(symbols and all(s.startswith(('active113_','r1active113_','r1a113_')) for s in symbols),
                 'non-namespaced exported backend symbol')
            check=target/'check_loader';run([cc,*flags,*includes,HERE/'check_loader.c',library,'-lm','-o',check],env)
            elf=run([readelf,'-h','-W',check],env)
            if arch=='arm':need(re.search(r'Class:\s+ELF32',elf) and re.search(r'Machine:\s+ARM\b',elf) and 'hard-float ABI' in elf,'ARM ABI')
            argv=[check] if arch=='host' else [qemu,'-L',execution['sysroot'],check]
            result=json.loads(run(argv,env,timeout=45))
            need(result['status']=='PASS_INVENTED_LOADER_AND_ID_CHECKS' and result['neural_model_calls']==result['pcm_calls']==0,
                 'invented checks did not pass')
            receipt['builds'][arch]={'static_library':identity(library),'loader_test_binary':identity(check),
                'compiler_sha256':sha(cc.resolve()),'flags':flags,'namespace_symbols':sorted(symbols),
                'loader_test_result':result,'objects':{x.name:identity(x) for x in objects}}
        unchanged();source_gate(a.plan_sha256);base_binding()
        receipt['status']='PASS_BUILD_AND_INVENTED_LOADER_CHECKS'
    except BaseException as e:
        receipt['status']='FAILED_NO_MODEL_EXECUTION';receipt['failure']=type(e).__name__+': '+str(e);raise
    finally:save(out/'BUILD_RECEIPT.json',receipt)
    print('PASS_BUILD_AND_INVENTED_LOADER_CHECKS '+sha(out/'BUILD_RECEIPT.json'))


if __name__=='__main__':main()
