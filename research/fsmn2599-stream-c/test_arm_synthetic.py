"""Artificial driver regressions; no toolchain, emulator, model or audio needed."""
import struct
import io
import json
import time
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import check_arm_synthetic as arm


class DriverTests(unittest.TestCase):
    def test_production_extraction_and_reference(self):
        source = (arm.M / 'fsmn2599.c').read_text()
        reference = (arm.M / 'affine_reference.inc').read_text()
        bodies = arm.extract(source, reference)
        self.assertNotEqual(bodies[True], bodies[False])
        self.assertIn('affine_block2(', bodies[True])
        with self.assertRaises(RuntimeError):
            arm.extract(source, reference.replace('0.0f', '1.0f'))
        with self.assertRaises(RuntimeError):
            arm.extract(source + '#ifdef KWS_AFFINE_BLOCK2\n', reference)

    def test_elf_geometry_and_calling_convention(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'elf'
            raw = bytearray(52); raw[:7] = b'\x7fELF\x01\x01\x01'
            struct.pack_into('<H', raw, 18, 40)
            struct.pack_into('<I', raw, 36, 0x05000400)
            p.write_bytes(raw)
            self.assertEqual(arm.check_elf(p)['elf_flags'], '0x5000400')
            for offset,value in ((4,2),(5,2),(18,62),(37,2)):
                changed = raw.copy(); changed[offset] = value; p.write_bytes(changed)
                with self.assertRaises(RuntimeError): arm.check_elf(p)
            host = raw + bytearray(32)
            struct.pack_into('<H', host, 18, 3)
            struct.pack_into('<I', host, 28, 52)
            struct.pack_into('<HH', host, 42, 32, 1)
            struct.pack_into('<I', host, 52, 1)
            p.write_bytes(host)
            pin = {'binary_bytes': len(host), 'binary_sha256': arm.sha(p)}
            arm.check_emulator(p, pin)
            struct.pack_into('<I', host, 52, 3)  # Dynamic host loader forbidden.
            p.write_bytes(host)
            with self.assertRaisesRegex(RuntimeError, 'pinned'): arm.check_emulator(p, pin)
            pin['binary_sha256'] = arm.sha(p)
            with self.assertRaisesRegex(RuntimeError, 'static'): arm.check_emulator(p, pin)
            host[4] = 2; p.write_bytes(host); pin['binary_sha256'] = arm.sha(p)
            with self.assertRaisesRegex(RuntimeError, 'ELF32'): arm.check_emulator(p, pin)

    def test_shared_budget_and_command_evidence(self):
        with patch.object(arm.guard, 'run_bounded', return_value=('ok', 2.0)) as run:
            b = arm.Budget(Path('/tmp')); b.run(['fixture'], 'synthetic-execution')
            self.assertEqual(b.cpu, 2.0); self.assertEqual(b.output, 2)
            self.assertEqual(b.commands[0]['argv'], ['fixture'])
            self.assertEqual(b.commands[0]['returncode'], 0)
            self.assertLessEqual(run.call_args.args[2], 30)
            self.assertEqual(run.call_args.kwargs['output_limit'], 8*1024**2)

    def test_exhausted_budgets_never_invoke_child(self):
        for kind in ('cpu','wall','output'):
            with patch.object(arm.guard, 'run_bounded') as run:
                b = arm.Budget(Path('/tmp'))
                if kind == 'cpu': b.cpu = 300
                if kind == 'wall': b.start -= 301
                if kind == 'output': b.output = arm.OUTPUT_BYTES
                with self.assertRaises(RuntimeError): b.run(['fixture'], 'test')
                run.assert_not_called()

    def test_numerical_failure_never_retried(self):
        with patch.object(arm.guard, 'run_bounded', side_effect=RuntimeError('mismatch')) as run:
            b = arm.Budget(Path('/tmp'))
            with self.assertRaisesRegex(RuntimeError, 'mismatch'): b.run(['fixture'], 'test')
            run.assert_called_once()
            self.assertNotIn('returncode', b.commands[0])
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaisesRegex(RuntimeError, 'structured evidence'):
            with arm.evidence_directory(time.monotonic(), 'bound-source') as (tmp, b):
                b.evidence['artifacts']['fixture'] = {'sha256': 'public-fixture'}
                with patch.object(arm.guard, 'run_bounded', side_effect=RuntimeError('mismatch')) as run:
                    try: b.run(['qemu', 'fixture'], 'synthetic-execution')
                    finally: run.assert_called_once()
        report = json.loads(stream.getvalue())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['commands'][0]['argv'], ['qemu', 'fixture'])
        self.assertEqual(report['commands'][0]['error'], 'mismatch')
        self.assertFalse(tmp.exists())

    def test_missing_tool_fails_without_skip(self):
        with patch.object(arm.guard, 'validate_sources', return_value='identity'), patch.object(arm.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'required tool missing'):
                arm.execute('/missing')

    def test_loader_override_rejected(self):
        with patch.dict(arm.os.environ, {'LD_AUDIT':'untrusted'}):
            with self.assertRaisesRegex(RuntimeError, 'loader overrides'):
                arm.execute('/missing')


if __name__ == '__main__':
    unittest.main()
