"""Standard-library command/validation contracts; never compile or run C."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import run_research_three_window_arm as arm
from test_research_three_window_c import PUBLIC_SYMBOLS

SMOKE = 'synthetic groups=8 storage_bytes=1; no acoustic admission\n'
COUNTS = dict(status='PASS_DECODER_THRESHOLD_SYNTHETIC_ONLY', fixtures_passed=8,
              completed_rows=569, step_attempts=572, beam_updates=949,
              windows=21, direct_merge_calls=11)
ELF_HEADER = """ELF Header:
  Class:                             ELF32
  Data:                              2's complement, little endian
  Type:                              DYN (Shared object file)
  Machine:                           ARM
"""


def symbols(names):
    rows = ["0: 00000000 0 NOTYPE LOCAL DEFAULT UND",
            "1: 00000000 0 FUNC GLOBAL DEFAULT UND kws3_undefined"]
    rows += [f"{index}: 00001234 12 FUNC GLOBAL DEFAULT 9 {name}"
             for index, name in enumerate(names, 2)]
    return (ELF_HEADER + f"Symbol table '.dynsym' contains {len(rows)} entries:\n"
            + "   Num: Value Size Type Bind Vis Ndx Name\n"
            + "\n".join("   " + row for row in rows) + "\n")


SYMBOLS = symbols(sorted(PUBLIC_SYMBOLS))


class ArmRunnerTests(unittest.TestCase):
    def test_fixed_cross_commands_and_bounds(self):
        self.assertEqual(arm.COMPILER, 'arm-linux-gnueabihf-gcc')
        self.assertEqual(arm.READELF, 'readelf')
        self.assertEqual(arm.FLAGS, [
            '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic',
            '-fno-fast-math', '-ffp-contract=off', '-fexcess-precision=standard',
            '-mcpu=cortex-a32', '-mfpu=neon-vfpv4', '-mfloat-abi=hard'])
        replies = ['', SMOKE, '', json.dumps(COUNTS), '', SYMBOLS]
        with patch.object(arm, 'run', side_effect=replies) as run:
            arm.run_arm(Path('/tmp/output'), '/usr/bin/qemu-arm-static', Path('/arm'))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([call.kwargs['timeout'] for call in run.call_args_list],
                         [45, 60, 45, 60, 45, 10])
        for index, source, name in ((0, 'test_synthetic.c', 'smoke'),
                                     (2, 'test_threshold.c', 'threshold')):
            self.assertEqual(commands[index], [arm.COMPILER] + arm.FLAGS +
                             ['-O1', '-g', source, '-lm', '-o', '/tmp/output/' + name])
            self.assertEqual(commands[index + 1],
                             ['/usr/bin/qemu-arm-static', '-L', '/arm', '/tmp/output/' + name])
        self.assertEqual(commands[4], [arm.COMPILER] + arm.FLAGS +
                         ['-O2', '-fPIC', '-shared', 'kws3.c', '-lm', '-o', '/tmp/output/libkws3.so'])
        self.assertEqual(commands[5], [arm.READELF, '-h', '-W', '--dyn-syms',
                                       '/tmp/output/libkws3.so'])
        self.assertEqual(run.call_args_list[5].kwargs['env']['LC_ALL'], 'C')
        self.assertNotIn('-DKWS3_TESTING', sum(commands, []))

    def test_missing_smoke_completion_stops(self):
        with patch.object(arm, 'run', side_effect=['', 'synthetic groups=7 ']) as run:
            with self.assertRaises(RuntimeError):
                arm.run_arm(Path('/tmp/output'), 'qemu', Path('/arm'))
        self.assertEqual(run.call_count, 2)

    def test_each_threshold_count_is_required(self):
        for key in COUNTS:
            broken = dict(COUNTS)
            del broken[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                arm.check_threshold_completion(json.dumps(broken))

    def test_threshold_failure_record_is_fatal(self):
        with self.assertRaises(RuntimeError):
            arm.check_threshold_completion('{"status":"FAIL"}\n' + json.dumps(COUNTS))

    def test_missing_extra_duplicate_or_hook_symbol_rejected(self):
        names = sorted(PUBLIC_SYMBOLS)
        invalid = (symbols(names[:-1]), symbols(names + ['kws3_extra']),
                   symbols(names + ['kws3_create']),
                   symbols(names + ['kws3_test_top3']),
                   SYMBOLS.replace('Machine:                           ARM',
                                   'Machine:                           X86-64'),
                   SYMBOLS.replace("'.dynsym'", "'.symtab'"),
                   SYMBOLS.replace('FUNC GLOBAL DEFAULT 9 kws3_create',
                                   'FUNC GLOBAL DEFAULT UND kws3_create'),
                   SYMBOLS.replace('FUNC GLOBAL DEFAULT 9 kws3_create',
                                   'FUNC GLOBAL HIDDEN 9 kws3_create'),
                   SYMBOLS.replace('FUNC GLOBAL DEFAULT 9 kws3_create',
                                   'FUNC LOCAL DEFAULT 9 kws3_create'),
                   SYMBOLS.rsplit('\n', 1)[0].rsplit('\n', 1)[0] + '\n')
        for text in invalid:
            with self.subTest(text=text[-100:]), self.assertRaises(RuntimeError):
                arm.check_arm_dynamic_symbols(text)
        arm.check_arm_dynamic_symbols(SYMBOLS)
        # Exported symbol versions are normalized as in the original nm contract.
        arm.check_arm_dynamic_symbols(SYMBOLS.replace(
            'FUNC GLOBAL DEFAULT 9 kws3_create',
            'FUNC GLOBAL DEFAULT 9 kws3_create@@API_1'))

    def test_timeout_or_subprocess_failure_has_no_fallback(self):
        for error in (subprocess.TimeoutExpired('gcc', 45), RuntimeError('compiler failure')):
            with patch.object(arm, 'run', side_effect=error) as run:
                with self.assertRaises(type(error)):
                    arm.run_arm(Path('/tmp/output'), 'qemu', Path('/arm'))
            self.assertEqual(run.call_count, 1)
        # Symbol reader timeout still fails under the unchanged 10-second ceiling.
        replies = ['', SMOKE, '', json.dumps(COUNTS), '',
                   subprocess.TimeoutExpired(arm.READELF, 10)]
        with patch.object(arm, 'run', side_effect=replies) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                arm.run_arm(Path('/tmp/output'), 'qemu', Path('/arm'))
        self.assertEqual(run.call_count, 6)

    def test_missing_sysroot_loader_stops_before_build(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(arm, 'run') as run:
            with self.assertRaises(SystemExit) as result:
                arm.main(['--emulator-sysroot', '/absent'])
            self.assertEqual(result.exception.code, 2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
