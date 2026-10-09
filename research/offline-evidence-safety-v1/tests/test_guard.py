import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from offline_safety import guard as g
from offline_safety.supervisor import capture
from offline_safety import preflight as pf
from synthetic import build

HELPER = Path(os.environ['PUBLIC_ADMISSION_PATH']).resolve()
EXPECTED_HELPER = 'f825a3e081723b8e153e6ec93a68b6a1d69acbc27c84937cefed9cf0d3dc900c'


class SavedAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = g.load_public_admission(HELPER, EXPECTED_HELPER)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.attempt, self.reference, self.identity, self.ticket = build(self.root, self.helper)
        self.native = {k: v.copy() for k, v in self.reference.items()}
        self.dest = self.root / 'evidence'
        self.approval = g.approval_binding(HELPER, self.ticket, self.identity)

    def run_review(self, **overrides):
        args = dict(helper_path=HELPER, helper_sha256=EXPECTED_HELPER,
                    ticket=self.ticket, work=self.root, attempt=self.attempt,
                    approval=self.approval, armed=True)
        args.update(overrides)
        return g.save_then_compare(self.dest, self.native, self.reference, self.identity, **args)

    def terminal(self):
        return json.loads((self.dest / 'TERMINAL.json').read_text())

    def test_guard_modules_have_no_optimization_sensitive_asserts(self):
        for name in ('guard.py', 'supervisor.py', 'preflight.py'):
            tree = ast.parse((Path(g.__file__).parent / name).read_text())
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_real_audit_and_complete_entry_pass(self):
        out = self.run_review()
        self.assertTrue(out['qualified'])
        self.assertTrue(out['complete'])
        self.assertEqual(out['model_calls'], 0)
        self.assertFalse(json.loads((self.dest / 'RAW-SAVED.json').read_text())['qualified'])

    def test_default_disarmed(self):
        with self.assertRaises(PermissionError): self.run_review(armed=False)
        self.assertTrue(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])

    def test_missing_approval(self):
        with self.assertRaises(PermissionError): self.run_review(approval=None)
        self.assertFalse(self.terminal()['qualified'])

    def test_wrong_approval(self):
        self.approval['policy_sha256'] = '0' * 64
        with self.assertRaises(PermissionError): self.run_review()

    def test_missing_ticket(self):
        self.ticket = None
        self.approval = g.approval_binding(HELPER, self.ticket, self.identity)
        with self.assertRaises(PermissionError): self.run_review()

    def test_source_drift_is_reaudited(self):
        (self.root / 'synthetic-source.txt').write_text('changed')
        with self.assertRaises(ValueError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])

    def test_input_fixture_drift_is_reaudited(self):
        path = self.attempt.parent / 'FIXTURES.json'
        value = json.loads(path.read_text())
        value['features'][0]['sha256'] = '0' * 64
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.run_review()

    def test_wrong_helper_hash(self):
        with self.assertRaises(PermissionError): self.run_review(helper_sha256='0' * 64)

    def test_bad_reference_gate_cannot_be_forged_pass(self):
        path = self.attempt / 'D20/features2-ragged.npz'
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays['stage0_7'][0, 0] = 1
        np.savez(path, **arrays)
        freeze = self.attempt / 'D20/REFERENCE-FROZEN.json'
        obj = json.loads(freeze.read_text())
        next(r for r in obj['records'] if r['name'] == 'features2-ragged')['sha256'] = g.sha(path)
        freeze.write_text(json.dumps(obj))
        with self.assertRaises(PermissionError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])

    def test_nan_is_saved_before_rejection(self):
        self.native['stage0_7'][0, 0] = np.nan
        with self.assertRaises(ValueError): self.run_review()
        with np.load(self.dest / 'native-raw.npz', allow_pickle=False) as data:
            self.assertTrue(np.isnan(data['stage0_7'][0, 0]))
        self.assertTrue(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])
        self.assertFalse((self.dest / 'validated/COMPARISON.json').exists())

    def test_wrong_shape_is_saved_before_rejection(self):
        self.native['logits0'] = np.zeros((1, 1), np.float32)
        with self.assertRaises(ValueError): self.run_review()
        with np.load(self.dest / 'native-raw.npz', allow_pickle=False) as data:
            self.assertEqual(data['logits0'].shape, (1, 1))
        self.assertTrue(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])

    def test_infinity_is_saved_and_rejected(self):
        self.native['logits0'][0, 0] = np.inf
        with self.assertRaises(ValueError): self.run_review()
        self.assertTrue(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])

    def test_wrong_dtype_is_saved_and_rejected(self):
        self.native['logits0'] = self.native['logits0'].astype(np.float64)
        with self.assertRaises(ValueError): self.run_review()
        with np.load(self.dest / 'native-raw.npz', allow_pickle=False) as data:
            self.assertEqual(data['logits0'].dtype, np.float64)
        self.assertFalse(self.terminal()['qualified'])

    def test_missing_member_is_saved_and_rejected(self):
        del self.native['stage0_7']
        with self.assertRaises(ValueError): self.run_review()
        self.assertTrue(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])

    def test_bad_counts_are_saved_and_rejected(self):
        self.native['counts'][0] = -1
        with self.assertRaises(ValueError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])

    def test_dependency_drift_is_reaudited(self):
        (self.root / 'reference-dependency-recovery-20261008/INSTALLED-BYTE-AUDIT.json').write_text('{}')
        with self.assertRaises(ValueError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])

    def test_unadmitted_reference_cannot_be_rebound(self):
        self.reference['stage0_7'][0, 0] = 1
        self.native['stage0_7'][0, 0] = 1
        self.identity['reference_sha256'] = self.helper.trace_digest(self.reference)
        self.approval = g.approval_binding(HELPER, self.ticket, self.identity)
        with self.assertRaises(PermissionError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])

    def test_numeric_failure_is_complete_not_qualified(self):
        self.native['stage0_7'][0, 0] = 1
        out = self.run_review()
        self.assertEqual(out['status'], 'FAIL')
        self.assertTrue(out['complete'])
        self.assertFalse(out['qualified'])

    def test_input_mismatch_rejected(self):
        self.native['input'][0, 0] = 1
        with self.assertRaises(ValueError): self.run_review()

    def test_unadmitted_fixture_rejected(self):
        self.identity['fixture'] = 'unadmitted'
        self.approval = g.approval_binding(HELPER, self.ticket, self.identity)
        with self.assertRaises(PermissionError): self.run_review()

    def test_no_overwrite(self):
        self.run_review()
        before = g.sha(self.dest / 'TERMINAL.json')
        with self.assertRaises(FileExistsError): self.run_review()
        self.assertEqual(before, g.sha(self.dest / 'TERMINAL.json'))

    def test_saved_raw_substitution_before_compare_is_rejected(self):
        self.native['stage0_7'][0, 0] = 1
        original_loader = g.load_public_admission
        def replace_after_snapshot(*args):
            helper = original_loader(*args)
            np.savez(self.dest / 'native-raw.npz', **self.reference)
            return helper
        with patch.object(g, 'load_public_admission', side_effect=replace_after_snapshot):
            with self.assertRaisesRegex(IOError, 'saved evidence drift'):
                self.run_review()
        self.assertFalse(self.terminal()['qualified'])
        self.assertFalse(self.terminal()['complete'])

    def test_missing_saved_evidence_clears_complete(self):
        original_loader = g.load_public_admission
        def remove_after_snapshot(*args):
            helper = original_loader(*args)
            (self.dest / 'native-raw.npz').unlink()
            return helper
        with patch.object(g, 'load_public_admission', side_effect=remove_after_snapshot):
            with self.assertRaises(FileNotFoundError): self.run_review()
        self.assertFalse(self.terminal()['qualified'])
        self.assertFalse(self.terminal()['complete'])

    def test_terminal_staging_failure_never_publishes_pass(self):
        original = g.write_new
        def fail_staging(path, value):
            if Path(path).name == 'TERMINAL.pending.json':
                Path(path).write_bytes(g.canonical(value))
                raise OSError('synthetic final receipt fsync failure')
            return original(path, value)
        with patch.object(g, 'write_new', side_effect=fail_staging):
            with self.assertRaisesRegex(OSError, 'synthetic final receipt'):
                self.run_review()
        self.assertFalse(self.terminal()['qualified'])
        self.assertEqual(self.terminal()['status'], 'REJECTED')

    def test_final_directory_fsync_error_is_not_masked_or_claimed_durable(self):
        original = g.sync_directory
        def fail_final_sync(path):
            if (Path(path) / 'TERMINAL.json').exists():
                raise OSError('synthetic final directory fsync failure')
            return original(path)
        with patch.object(g, 'sync_directory', side_effect=fail_final_sync):
            with self.assertRaisesRegex(OSError, 'synthetic final directory'):
                self.run_review()
        # Numerical verdict was published, but the call did not return success.
        self.assertEqual(self.terminal()['status'], 'PASS')
        self.assertFalse((self.dest / 'TERMINAL.pending.json').exists())

    def test_receipt_substitution_cannot_qualify(self):
        original_loader = g.load_public_admission
        def replace_after_snapshot(*args):
            helper = original_loader(*args)
            (self.dest / 'RAW-SAVED.json').write_text('{}')
            return helper
        with patch.object(g, 'load_public_admission', side_effect=replace_after_snapshot):
            with self.assertRaisesRegex(IOError, 'saved evidence drift'):
                self.run_review()
        self.assertFalse(self.terminal()['qualified'])
        self.assertFalse(self.terminal()['complete'])

    def test_partial_save_never_gets_complete_receipt(self):
        with patch.object(g.np, 'savez', side_effect=OSError('synthetic disk full')):
            with self.assertRaises(OSError): self.run_review()
        self.assertFalse(self.terminal()['complete'])
        self.assertFalse(self.terminal()['qualified'])
        self.assertFalse((self.dest / 'RAW-SAVED.json').exists())

    def test_minus_O_and_OO_fail_closed_at_actual_entry(self):
        # Subprocess receives only synthetic arrays/files; no model imports.
        np.savez(self.root / 'arrays.npz', **self.native)
        (self.root / 'args.json').write_text(json.dumps(dict(identity=self.identity,
             ticket=self.ticket, approval=self.approval)))
        code = '''
import json, numpy as np, sys
from pathlib import Path
from offline_safety.guard import save_then_compare
root=Path(sys.argv[1]); args=json.loads((root/'args.json').read_text())
with np.load(root/'arrays.npz', allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
try:
 save_then_compare(root/sys.argv[2], arrays, arrays, args['identity'], helper_path=sys.argv[3], helper_sha256=sys.argv[4], ticket=args['ticket'], work=root, attempt=root/'review/attempt', approval=args['approval'], armed=True)
except PermissionError as e:
 if 'optimized Python' not in str(e): raise
else: raise RuntimeError('optimization bypass')
'''
        for flag in ('-O', '-OO'):
            process = subprocess.run([sys.executable, flag, '-c', code, str(self.root), flag,
                                      str(HELPER), EXPECTED_HELPER], capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr.decode())
            terminal = json.loads((self.root / flag / 'TERMINAL.json').read_text())
            self.assertFalse(terminal['qualified'])
            self.assertTrue(terminal['complete'])


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'child.py'
        self.source.write_text("print('synthetic')")
        self.input = self.root / 'fixture.txt'
        self.input.write_text('synthetic')
        self.sources = {'child': self.source}
        self.inputs = {'fixture': self.input}
        self.approval = pf.preflight_binding(source_files=self.sources, input_files=self.inputs)
        self.spawns = 0

    def attempt_launch(self, approval, armed=True):
        pf.require_preflight(source_files=self.sources, input_files=self.inputs,
                             approval=approval, armed=armed)
        self.spawns += 1
        return subprocess.run([sys.executable, str(self.source)], capture_output=True, timeout=2)

    def test_valid_preflight_precedes_real_synthetic_spawn(self):
        result = self.attempt_launch(self.approval)
        self.assertEqual(self.spawns, 1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'synthetic\n')

    def test_missing_approval_spawns_zero(self):
        with self.assertRaises(PermissionError): self.attempt_launch(None)
        self.assertEqual(self.spawns, 0)

    def test_disarmed_spawns_zero(self):
        with self.assertRaises(PermissionError): self.attempt_launch(self.approval, armed=False)
        self.assertEqual(self.spawns, 0)

    def test_wrong_and_old_scope_spawn_zero(self):
        for scope in ('wrong', 'D20_D90_LOCAL_CERTIFICATE_V2_1916_ROWS_ONLY'):
            with self.subTest(scope=scope):
                approval = dict(self.approval, scope=scope)
                with self.assertRaises(PermissionError): self.attempt_launch(approval)
                self.assertEqual(self.spawns, 0)

    def test_numeric_true_is_not_exact_boolean_approval(self):
        approval = dict(self.approval, approved=1)
        with self.assertRaises(PermissionError): self.attempt_launch(approval)
        self.assertEqual(self.spawns, 0)

    def test_source_drift_spawns_zero(self):
        self.source.write_text("raise RuntimeError('changed')")
        with self.assertRaises(PermissionError): self.attempt_launch(self.approval)
        self.assertEqual(self.spawns, 0)

    def test_input_drift_spawns_zero(self):
        self.input.write_text('changed')
        with self.assertRaises(PermissionError): self.attempt_launch(self.approval)
        self.assertEqual(self.spawns, 0)

    def test_policy_drift_spawns_zero(self):
        approval = dict(self.approval, policy_sha256='0' * 64)
        with self.assertRaises(PermissionError): self.attempt_launch(approval)
        self.assertEqual(self.spawns, 0)

    def test_missing_source_spawns_zero(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError): self.attempt_launch(self.approval)
        self.assertEqual(self.spawns, 0)

    def test_optimized_preflight_spawns_zero(self):
        code = """
import sys
from offline_safety.preflight import require_preflight
spawns=0
try:
 require_preflight(source_files={},input_files={},approval={},armed=True)
 spawns+=1
except PermissionError as exc:
 if 'optimized Python' not in str(exc): raise
else: raise RuntimeError('preflight bypass')
if spawns != 0: raise RuntimeError('unexpected launch')
print('spawns=0')
"""
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, flag, '-c', code], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b'spawns=0\n')


class SyntheticProcessTests(unittest.TestCase):
    def run_child(self, code, **limits):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / 'capture'
        script = Path(self.tmp.name) / 'synthetic_child.py'
        script.write_text(code)
        input_file = Path(self.tmp.name) / 'synthetic-input.txt'
        input_file.write_text('synthetic fixture only')
        sources = {'child': script, 'supervisor': Path(g.__file__).with_name('supervisor.py')}
        inputs = {'fixture': input_file}
        approval = pf.preflight_binding(source_files=sources, input_files=inputs)
        pf.require_preflight(source_files=sources, input_files=inputs, approval=approval, armed=True)
        child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: child.kill() if child.poll() is None else None)
        return capture(child, self.dest, **limits)

    def test_success_captures_both_streams_but_never_qualifies(self):
        out = self.run_child("import sys; print('raw'); print('diagnostic', file=sys.stderr)")
        self.assertTrue(out['capture_ok'])
        self.assertTrue(out['complete'])
        self.assertFalse(out['qualified'])
        self.assertEqual((self.dest / 'stdout.bin').read_bytes(), b'raw\n')
        self.assertEqual((self.dest / 'stderr.bin').read_bytes(), b'diagnostic\n')

    def test_exit_failure_keeps_complete_raw_output(self):
        out = self.run_child("import sys; print('failed', flush=True); sys.exit(7)")
        self.assertEqual(out['reason'], 'exit_failure')
        self.assertEqual(out['returncode'], 7)
        self.assertTrue(out['complete'])
        self.assertFalse(out['capture_ok'])
        self.assertEqual((self.dest / 'stdout.bin').read_bytes(), b'failed\n')

    def test_timeout_retains_prefix_without_complete_claim(self):
        out = self.run_child("import time; print('prefix', flush=True); time.sleep(5)", timeout_seconds=.3)
        self.assertEqual(out['reason'], 'timeout')
        self.assertFalse(out['complete'])
        self.assertFalse(out['qualified'])
        self.assertEqual((self.dest / 'stdout.bin').read_bytes(), b'prefix\n')

    def test_output_limit_retains_bounded_truncated_evidence(self):
        out = self.run_child("import os; os.write(1,b'x'*10000)", max_output_bytes=97)
        self.assertEqual(out['reason'], 'output_limit')
        self.assertTrue(out['truncated'])
        self.assertEqual(out['retained_bytes'], 97)
        self.assertEqual((self.dest / 'stdout.bin').read_bytes(), b'x' * 97)
        self.assertFalse(out['complete'])

    def test_combined_stdout_stderr_limit(self):
        out = self.run_child("import os; os.write(1,b'a'*60); os.write(2,b'b'*60)", max_output_bytes=100)
        self.assertEqual(out['reason'], 'output_limit')
        self.assertEqual(sum((self.dest / n).stat().st_size for n in ('stdout.bin', 'stderr.bin')), 100)

    def test_closed_pipes_do_not_bypass_wall_deadline(self):
        out = self.run_child("import os,time; os.close(1); os.close(2); time.sleep(5)", timeout_seconds=.3)
        self.assertEqual(out['reason'], 'timeout')
        self.assertFalse(out['complete'])
        self.assertIsNotNone(out['returncode'])

    def test_exact_limit_and_eof_is_not_truncated(self):
        out = self.run_child("import os; os.write(1,b'x'*100)", max_output_bytes=100)
        self.assertTrue(out['capture_ok'])
        self.assertFalse(out['truncated'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
