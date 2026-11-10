"""No-model, invented-fixture tests for private-owner ARM actual-replay driver."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent


def load_driver():
    sys.dont_write_bytecode=True
    s=importlib.util.spec_from_file_location('r1_owner_arm_replay',HERE/'owner_arm_replay.py')
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class OwnerArmSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=load_driver()

    def setUp(self):
        self.t=tempfile.TemporaryDirectory(prefix='r1-owner-arm-test-')
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)

    def test_indices_closed_and_preserve_order(self):
        self.assertEqual(self.r.parse_indices('0,100,133'),[0,100,133])
        self.assertEqual(self.r.parse_indices('all'),list(range(134)))
        self.assertEqual(self.r.parse_indices('000'),[0])

    def test_indices_reject_duplicates_and_out_of_range(self):
        for bad in ('','0,0','134','-1','1.0','1,','1,你好','0,0,133',','.join('0' for _ in range(135))):
            with self.subTest(input=bad),self.assertRaises(ValueError):
                self.r.parse_indices(bad)

    def test_frame_events_are_exact_not_nonempty_diagnostic(self):
        raw=b'F,0,720,960,0\nF,1,1200,1280,1027\nE,0,1,1,0,1200,1280\n'
        frames,events=self.r.parse_trace(raw)
        self.assertEqual(len(frames),2)
        self.assertEqual(events,[(0,1,1,0,1200,1280)])

    def test_reject_invalid_or_reordered_traces(self):
        for bad in (b'', b'F,1,720,960,0\n', b'F,0,960,720,0\n',
                    b'F,0,720,960,2601\n',b'E,0,1,0,0,720,960\n',
                    b'F,0,720,960,0\nE,0,3,0,0,720,960\n',
                    b'F,0,720,960,0\nE,0,1,1,0,720,960\n',
                    b'F,0,720,960,0\nE,0,1,0,1,720,960\n',
                    b'F,0,720,960,0\nFAKE\n'):
            with self.subTest(data=bad),self.assertRaises(ValueError):
                self.r.parse_trace(bad)

    def test_reject_unbound_or_missing_private_manifest(self):
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({'schema':'kws-r1-arm-input-recovery-v1','records':[]}))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            self.r.load_manifest(manifest)

    def test_private_input_hash_and_byte_guard(self):
        p=self.root/'model.f32le';p.write_bytes(b'a')
        with self.assertRaisesRegex(ValueError,'byte count'):
            self.r.verify_file(p,self.r.MODEL_SHA,3028860)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            self.r.verify_file(p,self.r.MODEL_SHA)

    def test_elf_target_guard_fails_closed(self):
        artifact=self.root/'binary'; artifact.write_bytes(b'fake ELF body')
        fake=self.root/'readelf';fake.write_text('#!/bin/sh\necho "  Class: ELF64"\necho "  Machine: Advanced Micro Devices X86-64"\n')
        fake.chmod(0o755)
        with self.assertRaisesRegex(ValueError,'ARM target'):
            self.r.verify_arm_elf(fake,artifact)

    def test_public_header_compiles_without_private_assets(self):
        repo=HERE.parents[2]
        archive=repo/'research/r1_stream_c/deploy/archive-20261110/KWS_FSMN_R1_SOURCE_20261110.zip'
        # In an official complete checkout. External owner weights never needed.
        with ZipFile(archive) as z:
            header=z.read('KWS_FSMN_R1_SOURCE_20261110/tree/research/r1_deploy_sdk/include/fsmn_sdk.h')
        inc=self.root/'fsmn_sdk.h';inc.write_bytes(header)
        for cc in ('gcc','clang'):
            if shutil.which(cc) is None:
                self.skipTest(f'{cc} is unavailable')
            subprocess.run([cc,'-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
                            '-fsyntax-only','-I'+str(self.root),str(HERE/'owner_arm_trace.c')],
                           timeout=20,capture_output=True,check=True)

    def test_qualification_flags_are_not_inferred_from_source(self):
        script=(HERE/'owner_arm_replay.py').read_text()
        self.assertIn("'physical_ssc305':False",script)
        self.assertIn("'frame_logit_numeric_tolerance':'NOT_MEASURED'",script)
        self.assertNotIn('requests.get(',script)
        self.assertNotIn('urllib.request',script)


class TargetResourceSourceTests(unittest.TestCase):
    """Standalone board resource source contracts only; zero model forwards."""

    def test_compiles_against_byte_pinned_sdk_header(self):
        repo=HERE.parents[2]
        archive=repo/'research/r1_stream_c/deploy/archive-20261110/KWS_FSMN_R1_SOURCE_20261110.zip'
        with ZipFile(archive) as z:
            header=z.read('KWS_FSMN_R1_SOURCE_20261110/tree/research/r1_deploy_sdk/include/fsmn_sdk.h')
        with tempfile.TemporaryDirectory(prefix='r1-target-resource-contract-') as t:
            inc=Path(t)/'fsmn_sdk.h'
            inc.write_bytes(header)
            for cc in ('gcc','clang'):
                self.assertIsNotNone(shutil.which(cc),cc+' compiler required in CI')
                subprocess.run(
                    [cc,'-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
                     '-fsyntax-only','-I'+t,str(HERE/'target_resource_bench.c')],
                    check=True,timeout=20,capture_output=True)

    def test_resource_receipt_is_complete_or_nonzero(self):
        """Invented no-model SDK isolates real stdio success and ENOSPC paths."""
        full = Path('/dev/full')
        if not full.is_char_device():
            self.skipTest('/dev/full is required for the Linux output-error case')
        cc = shutil.which('gcc')
        self.assertIsNotNone(cc, 'gcc compiler required in CI')
        repo = HERE.parents[2]
        archive = repo / ('research/r1_stream_c/deploy/archive-20261110/'
                          'KWS_FSMN_R1_SOURCE_20261110.zip')
        with ZipFile(archive) as z:
            header = z.read(
                'KWS_FSMN_R1_SOURCE_20261110/tree/research/r1_deploy_sdk/include/fsmn_sdk.h')
        with tempfile.TemporaryDirectory(prefix='r1-resource-output-') as t:
            root = Path(t)
            (root / 'fsmn_sdk.h').write_bytes(header)
            # These buffers satisfy file-size checks only. The stub deliberately
            # performs no model identity check, acoustic forward or callbacks.
            (root / 'model.bin').write_bytes(bytes(3028860))
            (root / 'frontend.bin').write_bytes(bytes(83840))
            (root / 'input.pcm').write_bytes(bytes(640))
            stub = root / 'no_model_sdk.c'
            stub.write_text(r'''
#include "fsmn_sdk.h"
struct fsmn_sdk { int unused; };
static struct fsmn_sdk instance;
fsmn_sdk *fsmn_sdk_create(const unsigned char *model,size_t model_bytes,
    const unsigned char *tables,size_t table_bytes,uint64_t origin,
    fsmn_event_fn event_fn,fsmn_frame_fn frame_fn,void *user) {
    (void)model; (void)model_bytes; (void)tables; (void)table_bytes;
    (void)origin; (void)event_fn; (void)frame_fn; (void)user;
    return &instance;
}
int fsmn_sdk_feed(fsmn_sdk *s,const unsigned char *pcm,size_t bytes,size_t *consumed) {
    (void)s; (void)pcm; *consumed=bytes; return FSMN_OK;
}
int fsmn_sdk_finish(fsmn_sdk *s) { (void)s; return FSMN_OK; }
int fsmn_sdk_destroy(fsmn_sdk *s) { (void)s; return FSMN_OK; }
''', encoding='utf-8')
            binary = root / 'resource_bench'
            subprocess.run(
                [cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-I' + t, str(HERE / 'target_resource_bench.c'), str(stub),
                 '-o', str(binary)], check=True, timeout=20, capture_output=True)
            command = [str(binary), str(root / 'model.bin'),
                       str(root / 'frontend.bin'), str(root / 'input.pcm')]
            successful = subprocess.run(command, capture_output=True,
                                        timeout=20, check=True)
            receipt = json.loads(successful.stdout)
            self.assertEqual(receipt['pcm_bytes'], 640)
            self.assertEqual(receipt['feed_calls'], 1)
            self.assertFalse(receipt['device_qualified'])
            with full.open('wb') as sink:
                failed = subprocess.run(command, stdout=sink, stderr=subprocess.PIPE,
                                        timeout=20, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(b'unable to write complete resource receipt', failed.stderr)

    def test_scientific_and_input_boundaries_explicit(self):
        source=(HERE/'target_resource_bench.c').read_text()
        guide=(HERE/'TARGET_RESOURCE_PROFILE.md').read_text()
        for marker in ('FSMN_MODEL_BYTES','FSMN_TABLE_BYTES','fsmn_sdk_create',
                       'fsmn_sdk_feed','fsmn_sdk_finish','getrusage(RUSAGE_SELF',
                       'CLOCK_PROCESS_CPUTIME_ID','PCM_LIMIT_BYTES',
                       'device_qualified','live_afe',
                       'power_mw','stack_high_water_bytes'):
            self.assertIn(marker,source)
        self.assertIn('not the strong R1 SDK format',guide)
        self.assertIn('NOT_RUN',guide)
        self.assertNotIn('curl ',guide)
        self.assertNotIn('requests.get(',source)


if __name__=='__main__': unittest.main()

