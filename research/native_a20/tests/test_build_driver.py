"""Offline driver contracts with invented ELF headers; no target binary runs."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build


def elf_bytes(elf_class=32, machine=40, endian='little', kind=1, osabi=0, abi_version=0, flags=0):
    size = 52 if elf_class == 32 else 64
    raw = bytearray(size)
    raw[:7] = b'\x7fELF'+bytes((1 if elf_class == 32 else 2, 1 if endian == 'little' else 2, 1))
    order = '<' if endian == 'little' else '>'
    struct.pack_into(order+'HHI', raw, 16, kind, machine, 1)
    struct.pack_into(order+'H', raw, 40 if elf_class == 32 else 52, size)
    raw[7], raw[8] = osabi, abi_version
    struct.pack_into(order+'I', raw, 36 if elf_class == 32 else 48, flags)
    if kind in (2, 3):
        phsize = 32 if elf_class == 32 else 56
        struct.pack_into(order+('I' if elf_class == 32 else 'Q'), raw, 28 if elf_class == 32 else 32, size)
        struct.pack_into(order+'HH', raw, 42 if elf_class == 32 else 54, phsize, 1)
        # One invented file-backed executable PT_LOAD; never run these bytes.
        if elf_class == 32:
            raw += struct.pack(order+'IIIIIIII', 1, 0, 0, 0, size+phsize, size+phsize, 5, 1)
        else:
            raw += struct.pack(order+'IIQQQQQQ', 1, 5, 0, 0, 0, size+phsize, size+phsize, 1)
    return bytes(raw)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)/'new-build'
        self.calls = []
        self.executions = []
        self.load_calls = []
        self.target = {'class': 32, 'machine': 40, 'endian': 'little'}
        self.replace_name = None
        self.replacement = None
        self.fail_compile = False
        self.raise_compile = None
        self.host = build.read_elf(Path(sys.executable).resolve())
        certificate = json.loads((build.ROOT/'twiddle-certificate-summary.json').read_text())
        certificate['certificates'] = []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(build.generate_twiddles, 'generate', side_effect=lambda: (
            (build.ROOT/'src/fft64_twiddles.h').read_text(), dict(certificate))).start()
        mock.patch.object(build.subprocess, 'run', side_effect=self.fake_run).start()
        mock.patch.object(build.ctypes_api, 'load', side_effect=self.fake_load).start()
        self.direct_dlopen = mock.patch.object(build.ctypes_api.C, 'CDLL',
                                              side_effect=AssertionError('unexpected dlopen')).start()

    def fake_load(self, path):
        self.load_calls.append(path)
        return None, {'stream_bytes': 64, 'stream_alignment': 8}

    def fake_run(self, command, **kwargs):
        self.calls.append(command)
        stdout = ''
        if command[0] == sys.executable:
            self.assertEqual(command[1:], ['-B', str(build.ROOT/'tests/check_manifest.py')])
        elif '-o' in command:
            if self.raise_compile is not None:
                raise self.raise_compile
            if self.fail_compile:
                return subprocess.CompletedProcess(command, 1, '', 'invented compiler failure')
            path = Path(command[command.index('-o')+1])
            kind = 1 if '-c' in command else (3 if '-shared' in command else 2)
            raw = elf_bytes(self.target['class'], self.target['machine'], self.target['endian'], kind,
                            self.target.get('osabi', 0), self.target.get('abi_version', 0),
                            self.target.get('flags', 0))
            if path.name == self.replace_name:
                raw = self.replacement
            path.write_bytes(raw)
        elif '--version' in command:
            stdout = 'invented offline compiler\n'
        elif '-dumpmachine' in command:
            # Always claim host, even while deliberately producing ARM ELF.
            stdout = 'x86_64-linux-gnu\n'
        elif Path(command[0]).parent == self.out:
            self.executions.append(command)
            # Every artifact must already exist before first target execution.
            self.assertTrue((self.out/'test_identity').exists())
            if Path(command[0]).name == 'resource_sizes':
                stdout = json.dumps({'model_cache_bytes': 22528, 'weights_bytes': 1565280})
            else:
                stdout = 'PASS invented target response\n'
        else:
            self.fail(f'unexpected subprocess: {command}')
        return subprocess.CompletedProcess(command, 0, stdout, '')

    def args(self, extra=()):
        return ['--output', str(self.out), '--cc', 'invented-cc', *extra]

    def cross(self, extra=()):
        return self.args(['--mode', 'compile-only', '--target-elf-class', '32',
                          '--target-machine', 'arm', '--target-endian', 'little', *extra])

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            build.main(args)

    def receipt(self):
        return json.loads((self.out/'receipt.json').read_text())

    def assert_not_run(self):
        self.assertEqual(self.executions, [])
        self.assertEqual(self.load_calls, [])
        self.direct_dlopen.assert_not_called()
        self.assertFalse((self.out/'resource-sizes.json').exists())

    def test_compile_only_all_artifacts_no_execution_or_dlopen(self):
        self.invoke(self.cross())
        receipt = self.receipt()
        self.assertEqual(len(receipt['artifacts']), 17)
        self.assertEqual(receipt['invented_tests'], [])
        self.assertEqual(receipt['compiled_invented_tests'], list(build.TESTS))
        for key in ('invented_tests_status', 'resource_sizes', 'ctypes_abi'):
            self.assertEqual(receipt[key], 'NOT_RUN')
        self.assertEqual(receipt['target_executions'], 0)
        self.assertEqual(receipt['target_dlopen_calls'], 0)
        self.assertFalse(receipt['asan_ubsan'])
        self.assertFalse(receipt['vendor_abi_qualified'])
        self.assertFalse(receipt['board_qualified'])
        self.assert_not_run()

    def test_compile_only_same_host_still_never_runs(self):
        self.target = self.host
        name = next(key for key, value in build.MACHINES.items() if value == self.host['machine'])
        self.invoke(self.args(['--mode', 'compile-only', '--target-elf-class', str(self.host['class']),
                               '--target-machine', name, '--target-endian', self.host['endian']]))
        self.assert_not_run()

    def test_native_default_keeps_resource_abi_and_five_tests(self):
        self.target = self.host
        self.invoke(self.args())
        self.assertEqual(len(self.executions), 6)
        self.assertEqual(len(self.load_calls), 1)
        self.assertEqual(self.receipt()['mode'], 'native-test')
        self.assertEqual(self.receipt()['invented_tests'], list(build.TESTS))
        self.assertEqual(self.receipt()['ctypes_abi'], 'PASS')

    def test_native_sanitize_keeps_prior_ctypes_skip(self):
        self.target = self.host
        self.invoke(self.args(['--sanitize']))
        self.assertEqual(len(self.executions), 6)
        self.assertEqual(self.load_calls, [])
        self.assertTrue(self.receipt()['asan_ubsan'])
        self.assertEqual(self.receipt()['ctypes_abi'], 'NOT_RUN')

    def test_cross_cc_default_fails_even_when_triple_claims_host(self):
        self.target = dict(self.host, machine=40 if self.host['machine'] != 40 else 62)
        with self.assertRaisesRegex(ValueError, 'cross compilers require'):
            self.invoke(self.args())
        self.assert_not_run()
        self.assertFalse((self.out/'receipt.json').exists())

    def test_compile_only_requires_complete_explicit_target(self):
        with self.assertRaises(SystemExit):
            self.invoke(self.args(['--mode', 'compile-only', '--target-machine', 'arm']))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.out.exists())

    def test_native_rejects_target_options(self):
        with self.assertRaises(SystemExit):
            self.invoke(self.args(['--target-flag=-mcpu=cortex-a32']))
        self.assertEqual(self.calls, [])

    def test_target_flags_apply_to_every_compile_and_link(self):
        sysroot = Path(self.temp.name)/'sysroot'; sysroot.mkdir()
        flags = ['-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard']
        self.invoke(self.cross([*['--target-flag='+flag for flag in flags], '--sysroot', str(sysroot)]))
        commands = [command for command in self.calls if '-o' in command]
        self.assertEqual(len(commands), 17)
        for command in commands:
            for flag in [*flags, '--sysroot='+str(sysroot), *build.FLAGS]:
                self.assertIn(flag, command)
        self.assert_not_run()

    def test_unsafe_flags_cannot_override_numerical_contract(self):
        for flag in ('-ffast-math', '-Ofast', '-ffp-contract=fast', '-funsafe-math-optimizations',
                     '-fplugin=anything', '@response', '-mcpu=x -ffast-math', '-mrecip'):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                self.invoke(self.cross(['--target-flag='+flag]))
        self.assertEqual(self.calls, [])

    def test_missing_sysroot_fails_before_compiler(self):
        with self.assertRaises(SystemExit):
            self.invoke(self.cross(['--sysroot', str(Path(self.temp.name)/'absent')]))
        self.assertEqual(self.calls, [])

    def test_late_wrong_elf_prevents_all_native_execution(self):
        self.target = self.host
        self.replace_name = 'test_identity'
        self.replacement = elf_bytes(32, 40 if self.host['machine'] != 40 else 62)
        with self.assertRaises(ValueError):
            self.invoke(self.args())
        self.assertTrue((self.out/'resource_sizes').exists())
        self.assert_not_run()
        self.assertFalse((self.out/'receipt.json').exists())

    def test_each_elf_identity_dimension_is_enforced(self):
        for field, value in [('class', 64), ('machine', 183), ('endian', 'big')]:
            with self.subTest(field=field):
                self.out = Path(self.temp.name)/('bad-'+field)
                self.target = {'class': 32, 'machine': 40, 'endian': 'little', field: value}
                with self.assertRaisesRegex(ValueError, 'ELF '+field):
                    self.invoke(self.cross())
                self.assertFalse((self.out/'receipt.json').exists())
        self.assert_not_run()

    def test_bad_shared_type_rejected(self):
        self.replace_name = 'liba20_fft64.so'
        self.replacement = elf_bytes(kind=1)
        with self.assertRaisesRegex(ValueError, 'ELF type'):
            self.invoke(self.cross())
        self.assert_not_run()

    def test_compiler_failure_logs_and_emits_no_success(self):
        self.fail_compile = True
        with self.assertRaises(RuntimeError):
            self.invoke(self.cross())
        commands = json.loads((self.out/'commands.json').read_text())
        self.assertEqual(commands[-1]['returncode'], 1)
        self.assertFalse((self.out/'receipt.json').exists())
        self.assert_not_run()

    def test_compiler_oserror_logs_and_emits_no_success(self):
        self.raise_compile = FileNotFoundError('invented missing compiler')
        with self.assertRaises(FileNotFoundError):
            self.invoke(self.cross())
        self.assertIn('error', json.loads((self.out/'commands.json').read_text())[-1])
        self.assertFalse((self.out/'receipt.json').exists())
        self.assert_not_run()

    def test_compiler_timeout_logs_and_emits_no_success(self):
        self.raise_compile = subprocess.TimeoutExpired(['invented-cc'], 120)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke(self.cross())
        self.assertIn('error', json.loads((self.out/'commands.json').read_text())[-1])
        self.assertFalse((self.out/'receipt.json').exists())
        self.assert_not_run()

    def test_stale_output_never_reused(self):
        self.out.mkdir()
        sentinel = self.out/'receipt.json'; sentinel.write_text('previous receipt')
        with self.assertRaises(FileExistsError):
            self.invoke(self.cross())
        self.assertEqual(sentinel.read_text(), 'previous receipt')
        self.assertEqual(self.calls, [])

    def test_output_cannot_modify_source_tree(self):
        for path in (build.ROOT, build.ROOT/'new-build'):
            with self.subTest(path=path), self.assertRaises(SystemExit):
                self.invoke(['--output', str(path)])
        self.assertEqual(self.calls, [])


    def test_same_machine_wrong_native_osabi_prevents_execution(self):
        self.target = dict(self.host, osabi=9)
        with self.assertRaisesRegex(ValueError, 'incompatible native ELF osabi'):
            self.invoke(self.args())
        self.assert_not_run()

    def test_header_only_native_shared_prevents_execution(self):
        self.target = self.host
        self.replace_name = 'liba20_fft64.so'
        raw = elf_bytes(self.host['class'], self.host['machine'], self.host['endian'], 3)
        self.replacement = raw[:52 if self.host['class'] == 32 else 64]
        with self.assertRaisesRegex(ValueError, 'program headers'):
            self.invoke(self.args())
        self.assert_not_run()


class ELFTests(unittest.TestCase):
    def test_invented_32_64_little_big_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'header'
            for bits in (32, 64):
                for endian in ('little', 'big'):
                    path.write_bytes(elf_bytes(bits, 40, endian))
                    header = build.read_elf(path)
                    self.assertEqual((header['class'], header['endian'], header['machine']),
                                     (bits, endian, 40))

    def test_native_arm_float_and_eabi_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'arm'
            expected = {'class': 32, 'machine': 40, 'endian': 'little',
                        'flags': 0x05000400, 'osabi': 0, 'abi_version': 0}
            for flags in (0x05000200, 0x04000400, 0x05000600):
                path.write_bytes(elf_bytes(kind=3, flags=flags))
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, 'ARM'):
                    build.check_elf(path, expected, (3,))
            path.write_bytes(elf_bytes(kind=3, flags=0x05000400))
            build.check_elf(path, expected, (3,))
            # Object EABI is still checked; hard/soft float is not encoded there.
            path.write_bytes(elf_bytes(flags=0x05000000))
            build.check_elf(path, expected, (1,))

    def test_bad_program_headers_and_segments_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid'
            good = elf_bytes(kind=3)
            variants = [good[:52], good[:83]]
            for offset, value, fmt in ((28, 1, 'I'), (42, 1, 'H'), (44, 0, 'H'),
                                       (52, 0, 'I'), (56, 9999, 'I'), (68, 9999, 'I'),
                                       (72, 1, 'I'), (76, 4, 'I'), (80, 3, 'I')):
                bad = bytearray(good); struct.pack_into('<'+fmt, bad, offset, value); variants.append(bad)
            for raw in variants:
                path.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    build.read_elf(path)

    def test_malformed_headers_fail_closed(self):
        good = elf_bytes()
        samples = [b'', b'not an ELF', good[:16], good[:51]]
        for offset, value in ((4, 0), (5, 0), (6, 0), (20, 0), (40, 0)):
            malformed = bytearray(good); malformed[offset] = value; samples.append(malformed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'header'
            for raw in samples:
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(ValueError):
                        build.read_elf(path)


if __name__ == '__main__':
    unittest.main()
