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
SYMBOLS = '\n'.join('0000 T ' + name for name in sorted(PUBLIC_SYMBOLS))


class ArmRunnerTests(unittest.TestCase):
    def test_fixed_cross_commands_and_bounds(self):
        self.assertEqual(arm.COMPILER, 'arm-linux-gnueabihf-gcc')
        self.assertEqual(arm.NM, 'arm-linux-gnueabihf-nm')
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
        self.assertEqual(commands[5], [arm.NM, '-D', '--defined-only', '/tmp/output/libkws3.so'])
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
        for text in (SYMBOLS.split('\n', 1)[1], SYMBOLS + '\n0000 T kws3_extra',
                     SYMBOLS + '\n0000 T kws3_create', SYMBOLS + '\n0000 T kws3_test_top3'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                arm.check_production_symbols(text)

    def test_timeout_or_subprocess_failure_has_no_fallback(self):
        for error in (subprocess.TimeoutExpired('gcc', 45), RuntimeError('compiler failure')):
            with patch.object(arm, 'run', side_effect=error) as run:
                with self.assertRaises(type(error)):
                    arm.run_arm(Path('/tmp/output'), 'qemu', Path('/arm'))
            self.assertEqual(run.call_count, 1)

    def test_missing_sysroot_loader_stops_before_build(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(arm, 'run') as run:
            with self.assertRaises(SystemExit) as result:
                arm.main(['--emulator-sysroot', '/absent'])
            self.assertEqual(result.exception.code, 2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
