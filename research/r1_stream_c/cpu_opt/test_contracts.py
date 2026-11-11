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
        cls.sdk2=cls.root/'block2'
        cls.report2=cls.m.materialize(ROOT,cls.sdk2,kernel='block2')

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

    def test_block2_exact_parity_and_lower_register_pressure(self):
        old=kernel(self.base/'core/r1dfsmn.c','reference_head')
        new=kernel(self.sdk2/'core/r1dfsmn.c','candidate_head')
        q=load(HERE.parent/'quant_head8/materialize.py','r1_block2_quant_reference')
        original=(self.base/'core/r1dfsmn.c').read_text()
        self.assertEqual((self.sdk2/'core/r1dfsmn.c').read_text(),
                         original.replace(q.Q8_HEAD_PAR4,self.m.BLOCK2_HEAD,1))
        self.assertEqual(self.report2['candidate_kernel'],
                         'qhead8_par4_output_block2_experimental')
        harness=self.root/'test_head_block2.c'
        harness.write_text(C_HEADER+'\n'+old+'\n'+new+'\n'+C_FOOTER)
        for cc in COMPILERS:
            for opt in ('-O2','-O3'):
                with self.subTest(compiler=cc,optimization=opt):
                    exe=self.root/('block2-'+cc+'-'+opt[1:])
                    flags=[opt if flag=='-O2' else flag for flag in FLAGS]
                    run([cc,*flags,harness,'-lm','-o',exe])
                    receipt=json.loads(run([exe],timeout=40).stdout)
                    self.assertTrue(receipt['bit_exact'])
                    self.assertEqual(receipt['fixture_cases'],8)
                    self.assertFalse(receipt['real_model'])
                    print(json.dumps({'kernel':'block2','compiler':cc,
                                      'opt':opt,**receipt},sort_keys=True),flush=True)

    def test_block2_full_sdk_gcc_build(self):
        build=self.root/'sdk-build-block2'
        run(['cmake','-S',self.sdk2,'-B',build,
             '-DCMAKE_C_COMPILER=gcc','-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_TESTING=OFF'],timeout=100)
        run(['cmake','--build',build,'--parallel','2'],timeout=120)
        self.assertTrue((build/'fsmn_run_view').is_file())

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


class DeviceReceiptContracts(unittest.TestCase):
    """Invented receipt-only measurements: no physical board, PCM or model."""

    def setUp(self):
        self.temp_receipts = tempfile.TemporaryDirectory(prefix='r1-cpu-evidence-')
        self.addCleanup(self.temp_receipts.cleanup)
        self.work = Path(self.temp_receipts.name)
        self.comparator = load(HERE/'device_compare.py', 'r1_cpu_receipt_comparator')
        self.fixture_id = 0

    @staticmethod
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def invent(self):
        self.fixture_id += 1
        root = self.work / ('fixture-' + str(self.fixture_id))
        root.mkdir()
        m = self.comparator
        plan = {
            'schema': m.PLAN_SCHEMA,
            'model_sha256': m.MODEL_SHA,
            'frontend_sha256': m.FRONTEND_SHA,
            'pcm_sha256': 'e' * 64, 'pcm_bytes': 64000,
            'input_mode': 'offline_pcm16k',
            'binary_sha256': {'par4': 'a' * 64, 'block2': 'b' * 64, 'block4': 'c' * 64},
            'board': {'id': 'INVENTED-NOT-AN-SSC305',
                      'firmware_sha256': 'd' * 64, 'machine': 'armv7l',
                      'governor': 'performance', 'frequency_khz': 800000,
                      'affinity': '0'},
            'rounds': [],
        }
        for i in range(9):
            order = list(m.ORDERS[i % 3])
            captures = {}
            for v in order:
                # Positive CPU contrast is intentionally fictional, for
                # classifier contract tests only (never a measured result).
                cpu_base = 0.4 + 0.002 * i
                cpu_s = {'par4': cpu_base, 'block2': cpu_base * 0.8,
                         'block4': cpu_base * 1.1}[v]
                wall = 0.8 + 0.001 * i
                receipt = {
                    'schema': m.BENCH_SCHEMA,
                    'device_qualified': False, 'live_afe': False,
                    'independent_far_frr': False, 'power_mw': None,
                    'stack_high_water_bytes': None,
                    'model_kind': 'source-hash-checked-by-sdk',
                    'arch': 'armv7l', 'kernel': '5.10.0-invented',
                    'pcm_bytes': 64000, 'audio_seconds': 2.0,
                    'sdk_model_frames': 60, 'keyword_events': 5,
                    'keyword1_events': 2, 'keyword2_events': 3,
                    'feed_chunk_bytes': 640, 'feed_calls': 100,
                    'feed_p95_histogram_upper_us': {'par4': 128, 'block2': 64, 'block4': 256}[v],
                    'feed_max_us': 200, 'model_init_wall_ms': 10.0,
                    'offline_feed_wall_seconds': wall,
                    'offline_feed_cpu_seconds': cpu_s,
                    'offline_rtf': wall / 2.0,
                    'cpu_seconds_per_audio_second': cpu_s / 2.0,
                    'process_peak_rss_kib': 3000,
                }
                name = 'receipts/%02d-%s.json' % (i, v)
                captures[v] = {'path': name, 'sha256': self.put(root / name, receipt)}
            plan['rounds'].append({'round': i, 'order': order, 'receipts': captures})
        self.put(root / 'PLAN.json', plan)
        return root, plan

    def modify_receipt(self, root, plan, index, variant, modify, bind=True):
        entry = plan['rounds'][index]['receipts'][variant]
        path = root / entry['path']
        receipt = json.loads(path.read_text())
        modify(receipt)
        digest = self.put(path, receipt)
        if bind:
            entry['sha256'] = digest
            self.put(root / 'PLAN.json', plan)

    def test_rotated_synthetic_cpu_screen_never_grants_model_authority(self):
        root, plan = self.invent()
        report = self.comparator.examine(root)
        self.assertEqual(report['rounds'], 9)
        self.assertEqual(len(report['receipt_identities']), 27)
        self.assertEqual(report['candidates']['block2']['screen'],
                         'RESOURCE_SCREEN_PASS_NOT_QUALIFIED')
        self.assertEqual(report['candidates']['block4']['screen'],
                         'NO_CREDIBLE_CPU_GAIN')
        self.assertEqual(report['candidates']['block2']['paired_cpu_wins'], 9)
        self.assertEqual(report['candidates']['block2']['p95_upper_nonregressions'], 9)
        self.assertEqual(report['reference']['variant'], 'par4')
        self.assertFalse(report['model_promotion_authorized'])
        self.assertFalse(report['device_qualified'])
        self.assertFalse(report['physical_device_identity_independently_verified'])
        self.assertFalse(report['model_and_binary_hashes_independently_verified'])
        self.assertEqual(report['real_model_output_parity'], 'NOT_CHECKED_BY_THIS_TOOL')
        self.assertEqual(report['analyzer_model_forwards'], 0)
        self.assertEqual(report['model_sha256'], self.comparator.MODEL_SHA)
        self.assertEqual(report['pcm_sha256'], 'e' * 64)
        self.assertTrue(all(row['variant'] in self.comparator.VARIANTS
                            and row['sha256'] for row in report['receipt_identities']))

    def test_missing_tampered_and_repeated_receipts_fail_closed(self):
        root, plan = self.invent()
        entry = plan['rounds'][0]['receipts']['par4']
        path = root / entry['path']
        path.write_text(path.read_text().replace('"process_peak_rss_kib": 3000',
                                                 '"process_peak_rss_kib": 3001'))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.comparator.examine(root)

        root, plan = self.invent()
        entry = plan['rounds'][0]['receipts']['par4']
        (root / entry['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'missing evidence'):
            self.comparator.examine(root)

        root, plan = self.invent()
        a = plan['rounds'][0]['receipts']['par4']
        b = plan['rounds'][1]['receipts']['par4']
        raw = (root / a['path']).read_bytes()
        (root / b['path']).write_bytes(raw)
        b['sha256'] = hashlib.sha256(raw).hexdigest()
        self.put(root / 'PLAN.json', plan)
        with self.assertRaisesRegex(ValueError, 'duplicate receipt payload'):
            self.comparator.examine(root)

    def test_reject_frame_event_cpu_histogram_and_qualification_forgery(self):
        experiments = (
            ('counts', lambda r: r.__setitem__('sdk_model_frames', 61), 'frame/event disagreement'),
            ('events', lambda r: r.__setitem__('keyword_events', 99), 'keyword-event sum'),
            ('ratio', lambda r: r.__setitem__('offline_feed_cpu_seconds', 7.0), 'inconsistent CPU'),
            ('p95-shape', lambda r: r.__setitem__('feed_p95_histogram_upper_us', 123), 'histogram'),
            ('p95-value', lambda r: r.__setitem__('feed_p95_histogram_upper_us', 1024), 'histogram'),
            ('explicit-pass', lambda r: r.__setitem__('device_qualified', True),
             'qualification boundary'),
            ('negative-cpu', lambda r: r.__setitem__('offline_feed_cpu_seconds', -1.0),
             'numeric range'),
            ('pcm-exposure', lambda r: r.__setitem__('pcm_bytes', 60000), 'PCM byte exposure'),
            ('machine', lambda r: r.__setitem__('arch', 'x86_64'), 'machine/kernel'),
        )
        for name, change, message in experiments:
            with self.subTest(name=name):
                root, plan = self.invent()
                self.modify_receipt(root, plan, 0, 'block2', change)
                with self.assertRaisesRegex(ValueError, message):
                    self.comparator.examine(root)

    def test_reject_plan_identity_order_and_sampling_claims(self):
        experiments = (
            ('model', lambda p: p.__setitem__('model_sha256', '0' * 64), 'pretrained R1'),
            ('source', lambda p: p.__setitem__('frontend_sha256', '0' * 64), 'pretrained R1'),
            ('unbalanced', lambda p: p['rounds'][1].__setitem__(
                'order', ['par4', 'block2', 'block4']), 'run order'),
            ('insufficient', lambda p: p.__setitem__('rounds', p['rounds'][:6]),
             'rotated repetitions'),
            ('same-binary', lambda p: p['binary_sha256'].__setitem__(
                'block2', 'a' * 64), 'variant binaries'),
            ('bad-governor', lambda p: p['board'].__setitem__(
                'governor', 'unknown'), 'governor'),
            ('false-pcm', lambda p: p.__setitem__('pcm_bytes', 64002), 'PCM byte exposure'),
        )
        for name, change, message in experiments:
            with self.subTest(name=name):
                root, plan = self.invent()
                change(plan)
                self.put(root / 'PLAN.json', plan)
                with self.assertRaisesRegex(ValueError, message):
                    self.comparator.examine(root)

    def test_reject_duplicate_json_and_evidence_symlinks(self):
        root, plan = self.invent()
        path = root / plan['rounds'][0]['receipts']['par4']['path']
        content = path.read_text()
        self.assertIn('"arch": "armv7l"', content)
        path.write_text(content.replace('"arch": "armv7l"',
                                        '"arch": "armv7l", "arch": "armv7l"', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.comparator.examine(root)

        root, plan = self.invent()
        entry = plan['rounds'][0]['receipts']['par4']
        original = root / entry['path']
        other = root / 'offsite-evidence.json'
        other.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.comparator.examine(root)

    def test_cli_receipt_creation_never_overwrites_or_grants_authority(self):
        root, plan = self.invent()
        out = self.work / 'first-comparison.json'
        argv = [sys.executable, '-I', '-S', '-B', HERE/'device_compare.py',
                '--root', root, '--output', out]
        result = run(argv, timeout=25)
        self.assertEqual(json.loads(result.stdout)['status'], 'ANALYZED_NOT_QUALIFIED')
        report = json.loads(out.read_text())
        self.assertFalse(report['device_qualified'])
        self.assertFalse(report['model_promotion_authorized'])
        identity = hashlib.sha256(out.read_bytes()).hexdigest()
        again = subprocess.run([str(x) for x in argv], capture_output=True,
                               timeout=25)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), identity)



    def test_mmap_and_copy_receipts_cannot_be_mixed_in_cpu_ab(self):
        root, plan = self.invent()
        for batch in plan['rounds']:
            for variant in self.comparator.VARIANTS:
                entry = batch['receipts'][variant]
                path = root / entry['path']
                row = json.loads(path.read_text())
                row['model_storage'] = 'read_only_mmap'
                entry['sha256'] = self.put(path, row)
        self.put(root / 'PLAN.json', plan)
        result = self.comparator.examine(root)
        self.assertEqual(result['model_storage'], 'read_only_mmap')
        self.assertFalse(result['device_qualified'])
        self.assertFalse(result['model_promotion_authorized'])
        self.modify_receipt(root, plan, 0, 'par4',
                            lambda row: row.__setitem__('model_storage','sdk_owned_copy'))
        with self.assertRaisesRegex(ValueError, 'mixed model storage modes'):
            self.comparator.examine(root)

        root, plan = self.invent()
        self.assertEqual(self.comparator.examine(root)['model_storage'],
                         'legacy_unspecified')
        self.modify_receipt(root, plan, 0, 'block2',
                            lambda row: row.__setitem__('model_storage','mmap_unverified'))
        with self.assertRaisesRegex(ValueError, 'invalid model storage'):
            self.comparator.examine(root)


    def test_i16_certified_full_other_mechanism(self):
        """Int16 screening with original par4 exact fallback; invented rows."""
        source=ROOT / 'research/r1_stream_c/i16_certified_head'
        with tempfile.TemporaryDirectory(prefix='r1-i16-oracle-') as temp:
            for cc in COMPILERS:
                with self.subTest(compiler=cc):
                    binary=Path(temp)/('test-i16-'+cc)
                    build=subprocess.run([cc,*FLAGS,
                        source/'i16_certified_head.c',
                        source/'test_i16_certified_head.c',
                        '-lm','-o',binary],capture_output=True,text=True,timeout=60)
                    self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                    run=subprocess.run([str(binary)],capture_output=True,text=True,
                                       timeout=60)
                    self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                    proof=json.loads(run.stdout)
                    self.assertEqual(proof['status'],'PASS_INVENTED_I16_CERTIFIED_OTHER')
                    self.assertEqual(proof['frames'],135)
                    self.assertGreater(proof['ten_or_fewer_exact_rows'],0)
                    self.assertEqual(proof['model_calls'],0)
                    self.assertEqual(proof['pcm_calls'],0)
                    self.assertEqual(proof['decoder_calls'],0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc',action='append')
    args, remaining=p.parse_known_args()
    if args.cc:COMPILERS=args.cc
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
