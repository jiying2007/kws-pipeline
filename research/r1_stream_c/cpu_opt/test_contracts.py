#!/usr/bin/env python3
"""No-model CPU-block4 source, bit-exact head-only fixture and timing contracts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COMPILERS=['gcc','clang']
FLAGS=['-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
       '-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard']
C_HEADER=r'''#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define R1DFSMN_OUTPUTS 2601u
#ifndef R1_CPU_BENCH_REPS
#define R1_CPU_BENCH_REPS 96u
#endif
typedef struct {
    const int8_t *qhead;
    const double *scales;
    const float *biases;
    const float *precise_head;
} r1dfsmn_model;
'''
C_FOOTER=r'''static int8_t weights[2601u*140u];
static double scales[2601u], input[140u], refout[2601u], newout[2601u];
static float biases[2601u], precise[5u*140u];
static uint32_t seed=0xa341316cu;
static uint32_t rnd(void){seed=seed*1664525u+1013904223u;return seed;}
static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&ts)) return 0u;
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
}
int main(void) {
    r1dfsmn_model m={weights,scales,biases,precise};
    for (size_t i=0;i<sizeof(weights);i++)
        weights[i]=(int8_t)((int)(rnd()%255u)-127);
    for (unsigned o=0;o<2601u;o++) {
        scales[o]=0.001+(double)(o%29u)*0.01271;
        biases[o]=(float)((int)(rnd()%1999u)-999)*0.001f;
    }
    for (size_t i=0;i<sizeof(precise)/sizeof(precise[0]);i++)
        precise[i]=(float)((int)(rnd()%4095u)-2047)*0.0003f;
    for (unsigned trial=0;trial<8u;trial++) {
        for (unsigned i=0;i<140u;i++) {
            const double r=((double)(int)(rnd()%2001u)-1000.0)/2048.0;
            switch(trial) {
                case 0u:input[i]=0.0;break;
                case 1u:input[i]=i%2u ? -1.0 : 1.0;break;
                case 2u:input[i]=r;break;
                case 3u:input[i]=r*1e-7;break;
                case 4u:input[i]=r*1e5;break;
                case 5u:input[i]=i%7u ? r*1e-8 : r*1e7;break;
                case 6u:input[i]=(double)i*0.00001;break;
                default:input[i]=r*0.003;break;
            }
        }
        reference_head(&m,input,refout);
        candidate_head(&m,input,newout);
        for (unsigned o=0;o<2601u;o++) {
            if (memcmp(&refout[o],&newout[o],sizeof(double))) {
                fprintf(stderr,"bitwise head mismatch trial=%u row=%u\n",trial,o);
                return 9;
            }
        }
    }
    volatile double observed=0.0;
    uint64_t old=0u,neu=0u;
    for (unsigned rep=0;rep<R1_CPU_BENCH_REPS;rep++) {
        const int reverse=(rep%2u)!=0u;
        for (unsigned pass=0;pass<2u;pass++) {
            const int new_first=((pass==0u)==reverse);
            uint64_t begin=now_ns();
            if (!begin) return 8;
            if (new_first) candidate_head(&m,input,newout);
            else reference_head(&m,input,refout);
            uint64_t end=now_ns();
            if (end<=begin) return 7;
            if (new_first) {
                neu+=end-begin;
                observed+=newout[(rep*19u)%2601u];
            } else {
                old+=end-begin;
                observed+=refout[(rep*19u)%2601u];
            }
        }
    }
    if (memcmp(refout,newout,sizeof(refout))) return 6;
    printf("{\"schema\":\"r1-head-only-invented-cpu-v1\",\"fixture_cases\":8,"
           "\"bit_exact\":true,\"reference_cpu_ns\":%llu,"
           "\"candidate_cpu_ns\":%llu,\"observed\":%.9g,"
           "\"real_model\":false,\"physical_ssc305\":false}\n",
           (unsigned long long)old,(unsigned long long)neu,observed);
    return 0;
}
'''

def load(path,name):
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('missing source: '+str(path))
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def run(args,timeout=100):
    env=dict(os.environ)
    env.pop('LD_LIBRARY_PATH',None)
    return subprocess.run([str(arg) for arg in args],env=env,check=True,
                          capture_output=True,text=True,timeout=timeout)

def kernel(path,new_name):
    content=path.read_text()
    begin='static void affine_qhead('
    end='static int finite_values('
    if content.count(begin)!=1 or content.count(end)!=1:
        raise ValueError('unrecognized head boundary')
    start=content.index(begin)
    limit=content.index(end,start)
    return content[start:limit].replace(begin,'static void '+new_name+'(',1)

class CpuBlock4Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='r1-head-block4-test-')
        cls.root=Path(cls.tmp.name)
        cls.m=load(HERE/'materialize.py','r1_block4_materialize')
        cls.borrowed=load(HERE.parent/'borrowed_weights/materialize.py','r1_borrowed_source_baseline')
        cls.base=cls.root/'baseline'
        cls.sdk=cls.root/'block4'
        cls.borrowed.materialize(ROOT,cls.base)
        cls.report=cls.m.materialize(ROOT,cls.sdk)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_only_classifier_head_and_preparation_receipts_change(self):
        old={str(p.relative_to(self.base)) for p in self.base.rglob('*') if p.is_file()}
        now={str(p.relative_to(self.sdk)) for p in self.sdk.rglob('*') if p.is_file()}
        self.assertEqual(now-old,{'README-CPU-OPT.md','CPU_OPT_PREPARATION.json'})
        self.assertEqual(old-now,set())
        changed={p for p in old if (self.base/p).read_bytes()!=(self.sdk/p).read_bytes()}
        self.assertEqual(changed,{'core/r1dfsmn.c','BORROWED_PREPARATION.json'})
        old_code=(self.base/'core/r1dfsmn.c').read_text()
        new_code=(self.sdk/'core/r1dfsmn.c').read_text()
        q=load(HERE.parent/'quant_head8/materialize.py','r1_block4_quant_reference')
        self.assertEqual(old_code.count(q.Q8_HEAD_PAR4),1)
        self.assertEqual(new_code,old_code.replace(q.Q8_HEAD_PAR4,self.m.BLOCK4_HEAD,1))
        self.assertEqual(self.report['source_r1dfsmn_sha256_before'],
                         hashlib.sha256(old_code.encode()).hexdigest())
        self.assertEqual(self.report['source_r1dfsmn_sha256_after'],
                         hashlib.sha256(new_code.encode()).hexdigest())
        self.assertFalse(self.report['private_model_packed'])
        self.assertEqual(self.report['cpu_speedup'],'NOT_MEASURED')
        self.assertEqual(self.report['full_model_parity'],'NOT_RUN_FOR_THIS_BUILD')
        self.assertEqual(self.report['model_forwards'],0)
        self.assertFalse((self.sdk/'weights').exists())
        borrowed=json.loads((self.sdk/'BORROWED_PREPARATION.json').read_text())
        self.assertEqual(borrowed['changed_source_sha256']['core/r1dfsmn.c'],
                         hashlib.sha256(new_code.encode()).hexdigest())

    def test_fresh_output_and_unknown_assets_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.m.materialize(ROOT,self.sdk)
        root=self.root/'linked'
        root.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.m.materialize(ROOT,root/'nested')
        incomplete=self.root/'bad-model'
        invalid=self.root/'invalid.f32le'
        invalid.write_bytes(b'not weights')
        with self.assertRaises(ValueError):
            self.m.materialize(ROOT,incomplete,model=invalid,frontend=invalid)
        self.assertFalse(incomplete.exists())

    def test_compiler_bit_parity_and_invented_cpu_timing(self):
        old=kernel(self.base/'core/r1dfsmn.c','reference_head')
        new=kernel(self.sdk/'core/r1dfsmn.c','candidate_head')
        harness=self.root/'test_head.c'
        harness.write_text(C_HEADER+'\n'+old+'\n'+new+'\n'+C_FOOTER)
        for cc in COMPILERS:
            if shutil.which(cc) is None:
                raise RuntimeError('required test compiler unavailable: '+cc)
            for opt in ('-O2','-O3'):
                with self.subTest(compiler=cc,optimization=opt):
                    exe=self.root/(cc+'-'+opt[1:])
                    flags=[opt if flag=='-O2' else flag for flag in FLAGS]
                    run([cc,*flags,harness,'-lm','-o',exe])
                    receipt=json.loads(run([exe],timeout=40).stdout)
                    self.assertEqual(receipt['schema'],'r1-head-only-invented-cpu-v1')
                    self.assertEqual(receipt['fixture_cases'],8)
                    self.assertTrue(receipt['bit_exact'])
                    self.assertFalse(receipt['real_model'])
                    self.assertFalse(receipt['physical_ssc305'])
                    self.assertGreater(receipt['reference_cpu_ns'],0)
                    self.assertGreater(receipt['candidate_cpu_ns'],0)
                    print(json.dumps({'compiler':cc,'opt':opt,**receipt},sort_keys=True),flush=True)

    def test_complete_sdk_compiles_and_rejects_fake_model(self):
        fake=self.root/'invalid-model.r1hq8'
        fake.write_bytes(b'fake')
        for cc in COMPILERS:
            with self.subTest(compiler=cc):
                build=self.root/('sdk-build-'+cc)
                run(['cmake','-S',self.sdk,'-B',build,
                     '-DCMAKE_C_COMPILER='+cc,
                     '-DCMAKE_BUILD_TYPE=Release','-DBUILD_TESTING=OFF'],timeout=100)
                run(['cmake','--build',build,'--parallel','2'],timeout=120)
                exe=build/'fsmn_run_view'
                self.assertTrue(exe.is_file())
                result=subprocess.run([str(exe),'--research',str(fake),str(fake),
                    '--pcm',str(fake)],capture_output=True,timeout=10)
                self.assertNotEqual(result.returncode,0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc',action='append')
    args, remaining=p.parse_known_args()
    if args.cc:COMPILERS=args.cc
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
