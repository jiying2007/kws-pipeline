#!/usr/bin/env python3
"""Cross-build source-bound R1 output-block4 SDK and run synthetic ARM/QEMU parity.

Only invented head arrays: no private weights or original PCM are used here.
The emulator CPU time does not qualify physical SSC305 performance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def load(path,name):
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError('missing source module')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def checked(argv,timeout=150):
    env=dict(os.environ)
    env.pop('LD_LIBRARY_PATH',None)
    return subprocess.run([str(x) for x in argv],capture_output=True,text=True,
                          check=True,timeout=timeout,env=env).stdout

def elf(readelf,path):
    text=checked([readelf,'-h','-W',path],timeout=10)
    if not (re.search(r'Class:\s+ELF32',text) and
            re.search(r'Machine:\s+ARM\b',text) and
            'hard-float ABI' in text):
        raise ValueError('target is not ARM hard-float ELF32')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(compiler,readelf,emulator,sysroot):
    cpu=load(HERE/'materialize.py','r1_arm_block_materialize')
    borrowed=load(HERE.parent/'borrowed_weights/materialize.py','r1_arm_block_borrowed')
    fixtures=load(HERE/'test_contracts.py','r1_arm_block_fixture')
    with tempfile.TemporaryDirectory(prefix='r1-block-arm-test-') as tmp:
        root=Path(tmp)
        base=root/'base'
        borrowed.materialize(ROOT,base)
        candidates={}
        for kernel in ('block2','block4'):
            sdk=root/('sdk-'+kernel)
            receipt=cpu.materialize(ROOT,sdk,kernel=kernel)
            if receipt['private_model_packed'] or (sdk/'weights').exists():
                raise ValueError('private weight accidentally included')
            build=root/('build-'+kernel)
            checked(['cmake','-S',sdk,'-B',build,
                     '-DCMAKE_BUILD_TYPE=Release',
                     '-DCMAKE_C_COMPILER='+str(compiler),
                     '-DCMAKE_TOOLCHAIN_FILE='+str(sdk/'cmake/armv7-hardfloat.cmake'),
                     '-DBUILD_TESTING=OFF'],timeout=120)
            checked(['cmake','--build',build,'--parallel','2'],timeout=150)
            binary=build/'fsmn_run_view'
            library=build/'libfsmn_sdk.so.0.1.0'
            identities={'binary_sha256':elf(readelf,binary),
                        'shared_sha256':elf(readelf,library)}
            symbols=checked([readelf,'-W','--dyn-syms',library],timeout=10)
            for symbol in ('fsmn_sdk_create_view','fsmn_sdk_feed',
                           'fsmn_sdk_finish','fsmn_sdk_model_sha256'):
                if not re.search(r'\b'+symbol+r'\b',symbols):
                    raise ValueError('missing ARM SDK symbol: '+symbol)
            test=root/('head_fixture_'+kernel+'.c')
            test.write_text(fixtures.C_HEADER+'\n'
                +fixtures.kernel(base/'core/r1dfsmn.c','reference_head')+'\n'
                +fixtures.kernel(sdk/'core/r1dfsmn.c','candidate_head')+'\n'
                +fixtures.C_FOOTER)
            exe=root/('arm-head-fixture-'+kernel)
            checked([compiler,'-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
                     '-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard',
                     '-DR1_CPU_BENCH_REPS=4u',
                     '-mcpu=cortex-a32','-mfpu=neon-vfpv4','-mfloat-abi=hard',
                     test,'-lm','-o',exe],timeout=100)
            identities['fixture_elf_sha256']=elf(readelf,exe)
            result=json.loads(checked([emulator,'-L',sysroot,exe],timeout=120))
            if (result.get('bit_exact') is not True or result.get('fixture_cases')!=8
                or result.get('real_model') is not False or
                result.get('physical_ssc305') is not False):
                raise ValueError('ARM/QEMU invented-head fixture failed: '+kernel)
            candidates[kernel]={'source_compiled':True,
                'synthetic_arm_qemu_head_bit_exact':True,
                'fixture':result,'identities':identities,
                'cpu_gain_qualified':False}
        # New opt-in INT16 full-OTHER mechanism: two real ARM binaries, both
        # execute invented 2,601-row fixture comparisons against full par4.
        # Under QEMU these are functional tests only, never a CPU claim.
        i16_root = HERE.parent/'i16_certified_head'
        i16_results = {}
        original_result = None
        for variant, macros in (('scalar', []), ('neon', ['-DR1_I16_USE_NEON=1'])):
            test_binary = root/('arm-i16-'+variant)
            checked([compiler,'-std=c11','-O3','-Wall','-Wextra',
                     '-Werror','-pedantic','-fno-fast-math',
                     '-ffp-contract=off','-fexcess-precision=standard',
                     '-mcpu=cortex-a32','-mfpu=neon-vfpv4','-mfloat-abi=hard',
                     *macros,i16_root/'i16_certified_head.c',
                     i16_root/'test_i16_certified_head.c',
                     '-lm','-o',test_binary],timeout=100)
            binary_sha = elf(readelf,test_binary)
            raw = checked([emulator,'-L',sysroot,test_binary],timeout=120)
            record = json.loads(raw)
            if (record.get('status') != 'PASS_INVENTED_I16_CERTIFIED_OTHER' or
                record.get('frames') != 135 or
                record.get('model_calls') != 0 or
                record.get('pcm_calls') != 0 or
                record.get('decoder_calls') != 0):
                raise ValueError('ARM/QEMU invented i16 source parity failed: '+variant)
            if original_result is None:
                original_result = record
            elif record != original_result:
                raise ValueError('NEON vs scalar invented source receipts differ')
            i16_results[variant]={'binary_sha256':binary_sha,
                                  'synthetic_qemu_full_other_bitwise':True,
                                  'source_variant':variant,'receipt':record}
        return {'schema':'r1-head-blocks-arm-source-v2',
                'target':'cortex-a32-neon-vfpv4-hardfloat',
                'candidates':candidates,'i16_scalar_vs_neon':i16_results,'model_executions':0,
                'physical_ssc305':False,'real_model_parity':'NOT_RUN',
                'promote_model':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--compiler',type=Path,default=Path('/usr/bin/arm-linux-gnueabihf-gcc'))
    p.add_argument('--readelf',type=Path,default=Path('/usr/bin/readelf'))
    p.add_argument('--emulator',type=Path,required=True)
    p.add_argument('--sysroot',type=Path,required=True)
    args=p.parse_args()
    if not args.execute:raise SystemExit('explicit --execute required')
    if not all(p.exists() for p in (args.compiler,args.readelf,args.emulator,args.sysroot)):
        raise SystemExit('required cross toolchain/emulator/sysroot unavailable')
    print(json.dumps(execute(args.compiler,args.readelf,args.emulator,args.sysroot),
                     sort_keys=True))

if __name__=='__main__':
    main()
