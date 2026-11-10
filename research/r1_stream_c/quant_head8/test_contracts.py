"""Source-only, no-weight R1 head8 tests; cannot invoke training or inference."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError('missing exact source module')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Head8SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=load(HERE/'materialize.py','r1_q8_source')
        cls.p=load(HERE/'pack_model.py','r1_q8_packer')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='r1-head8-contract-')
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)/'out'

    def test_archive_sha_pinned_and_no_network(self):
        p=REPO/self.m.SOURCE_ARCHIVE_REL
        self.assertTrue(p.is_file())
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),self.m.SOURCE_ARCHIVE_SHA)
        self.assertNotIn('http',self.m.SOURCE_ARCHIVE_REL)

    def test_scalar_source_only_and_no_weights(self):
        receipt=self.m.materialize_from_repository(REPO,self.out)
        self.assertEqual(receipt['kernel'],'scalar')
        self.assertFalse(receipt['private_model_packed'])
        self.assertEqual(receipt['original_fp32_numeric_gate'],'FAIL_UNCHANGED')
        self.assertFalse((self.out/'weights').exists())
        self.assertTrue((self.out/'core/r1dfsmn.c').is_file())
        self.assertIn('affine_qhead', (self.out/'core/r1dfsmn.c').read_text())

    def test_par4_only_changes_arithmetic(self):
        normal=Path(self.temp.name)/'normal'
        fast=Path(self.temp.name)/'fast'
        self.m.materialize_from_repository(REPO,normal,kernel='scalar')
        self.m.materialize_from_repository(REPO,fast,kernel='par4')
        names1={p.relative_to(normal).as_posix() for p in normal.rglob('*') if p.is_file()}
        names2={p.relative_to(fast).as_posix() for p in fast.rglob('*') if p.is_file()}
        self.assertEqual(names1,names2)
        altered=[n for n in sorted(names1) if (normal/n).read_bytes()!=(fast/n).read_bytes()]
        self.assertEqual(altered,['Q8_PREPARATION.json','core/r1dfsmn.c'])
        self.assertIn('s0+=x[i]',(fast/'core/r1dfsmn.c').read_text())
        self.assertNotIn('s0+=x[i]',(normal/'core/r1dfsmn.c').read_text())

    def test_unknown_kernel_rejected_without_output(self):
        with self.assertRaises(ValueError):
            self.m.materialize_from_repository(REPO,self.out,kernel='definitely-not-validated')
        self.assertFalse(self.out.exists())

    def test_existing_output_not_overwritten(self):
        self.out.mkdir()
        (self.out/'keep.txt').write_text('original invariant')
        with self.assertRaises(ValueError):
            self.m.materialize_from_repository(REPO,self.out)
        self.assertEqual((self.out/'keep.txt').read_text(),'original invariant')

    def test_fake_model_rejected_before_output(self):
        fake=Path(self.temp.name)/'bad.f32le';fake.write_bytes(b'not a model')
        with self.assertRaises(ValueError):
            self.m.materialize_from_repository(REPO,self.out,model=fake,frontend=fake)
        self.assertFalse(self.out.exists())

    def test_bad_archive_rejected_before_output(self):
        repo=Path(self.temp.name)/'repository'
        target=repo/self.m.SOURCE_ARCHIVE_REL
        target.parent.mkdir(parents=True)
        data=(REPO/self.m.SOURCE_ARCHIVE_REL).read_bytes()
        target.write_bytes(data[:-1]+bytes([data[-1]^1]))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            self.m.materialize_from_repository(repo,self.out)
        self.assertFalse(self.out.exists())

    def test_packer_rejects_unverified_input(self):
        with self.assertRaisesRegex(ValueError,'identity'):
            self.p.pack(b'\0'*64)
        with self.assertRaisesRegex(ValueError,'length'):
            self.p.validate(b'\0'*88)
        bad=bytearray(self.p.PACKAGE_BYTES)
        bad[:self.p.HEADER.size]=self.p.HEADER.pack(self.p.MAGIC,1,2601,140,5,self.p.BASE_SHA.encode())
        with self.assertRaisesRegex(ValueError,'identity'):
            self.p.validate(bytes(bad))

    def test_source_only_cmake_build_and_ctest(self):
        env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
        for compiler,kernel in (('gcc','scalar'),('clang','par4')):
            if not shutil.which(compiler):
                self.skipTest(compiler+' compiler missing')
            source=Path(self.temp.name)/('source-'+compiler)
            build=Path(self.temp.name)/('build-'+compiler)
            self.m.materialize_from_repository(REPO,source,kernel=kernel)
            for command in (
                ['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                 '-DBUILD_TESTING=ON','-DCMAKE_C_COMPILER='+compiler],
                ['cmake','--build',str(build),'--parallel','2'],
                ['ctest','--test-dir',str(build),'--output-on-failure']):
                subprocess.run(command,env=env,check=True,capture_output=True,timeout=90)

if __name__=='__main__':
    unittest.main()
