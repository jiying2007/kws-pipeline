"""Read-only-base numerical source contracts: no trained weights or inference."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

class NumericSourceContracts(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        path = self.repo / 'research/r1_stream_c/numerics.py'
        spec = importlib.util.spec_from_file_location('r1_numeric_contract', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.directory = tempfile.TemporaryDirectory(prefix='r1-numeric-contract-')
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'out'

    def test_reference_has_distinct_abi(self):
        self.module.materialize(self.repo,self.output,'reference64')
        header=(self.output/'source/r1d_stream.h').read_text()
        self.assertIn('const double *',header)
        self.assertIn('r1d_create',header)
        core=(self.output/'source/r1dfsmn.h').read_text()
        self.assertIn('const float *weights',core)
        self.assertIn('double cache[',core)

    def test_precision_state_identities_are_distinct(self):
        self.module.materialize(self.repo,self.output,'reference64')
        for name,pin in (('r1d_stream.c','0x52324436'),('r1dfsmn.c','0x5232444e')):
            self.assertIn(pin,(self.output/'source'/name).read_text())

    def test_reference_no_public_symbol_alias_to_float32(self):
        self.module.materialize(self.repo,self.output,'reference64')
        for name in ('r1d_stream.c','r1dfsmn.c','reference64_fbank.c','sha256.c'):
            text=(self.output/'source'/name).read_text()
            self.assertNotIn('kcf_fbank_frame(',text)
            self.assertNotIn('kws_sha256_memory_hex(',text)

    def test_wide32_keeps_fp32_cache_and_io(self):
        self.module.materialize(self.repo,self.output,'wide32')
        self.assertIn('float cache[',(self.output/'source/r1fsmn.h').read_text())
        self.assertIn('const float *',(self.output/'source/r1_stream.h').read_text())
        self.assertIn('double sum=0.0',(self.output/'source/r1fsmn.c').read_text())

    def test_scope_remains_unqualified(self):
        result=self.module.materialize(self.repo,self.output,'reference64')
        self.assertEqual(result['legacy_fp32_gate'],'FAIL_UNCHANGED')
        self.assertEqual(result['reference_gate'],'NOT_RUN_FOR_THIS_BUILD')
        self.assertFalse(result['acoustic_qualification'])
        self.assertFalse(result['device_qualification'])
        self.assertFalse(result['default_runtime_changed'])

    def test_source_only_never_synthesizes_weight_bytes(self):
        result=self.module.materialize(self.repo,self.output,'reference64')
        self.assertFalse(result['weights_present'])
        self.assertEqual(list((self.output/'weights').iterdir()),[])
        self.assertEqual(result['model_forwards'],0)

    def test_unsupported_mode_before_output(self):
        with self.assertRaises(ValueError):self.module.materialize(self.repo,self.output,'fp16')
        self.assertFalse(self.output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir();(self.output/'kept').write_text('kept')
        with self.assertRaises(ValueError):self.module.materialize(self.repo,self.output,'reference64')
        self.assertEqual((self.output/'kept').read_text(),'kept')

    def test_symlink_output_is_rejected(self):
        destination=Path(self.directory.name)/'real';destination.mkdir();self.output.symlink_to(destination)
        with self.assertRaises(ValueError):self.module.materialize(self.repo,self.output,'wide32')
        self.assertEqual(list(destination.iterdir()),[])

    def test_missing_source_is_rejected(self):
        with self.assertRaises(ValueError):self.module.materialize(Path(self.directory.name),self.output,'reference64')
        self.assertFalse(self.output.exists())

    def test_constant_identity_and_tamper_rejection(self):
        text=self.module.twiddles()
        self.assertIn('r1_fft_cos64[256]',text)
        self.module.TWIDDLE_SHA='0'*64
        with self.assertRaises(ValueError):self.module.twiddles()

    def test_source_shape_change_is_rejected(self):
        with self.assertRaises(ValueError):self.module.replace_once('a a','a','b')
        with self.assertRaises(ValueError):self.module.replace_once('c','a','b')
        self.assertEqual(self.module.replace_once('a','a','b'),'b')

    def test_both_modes_build_without_model_load(self):
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        for mode in ('wide32','reference64'):
            output=Path(self.directory.name)/mode
            self.module.materialize(self.repo,output,mode)
            subprocess.run(['bash',str(output/'build.sh')],env=env,check=True,
                           capture_output=True,timeout=30)
            expected='r1_pcm' if mode=='wide32' else 'r1d_pcm'
            self.assertTrue((output/'bin'/expected).is_file())
            self.assertEqual(list((output/'weights').iterdir()),[])

if __name__=='__main__':unittest.main()
