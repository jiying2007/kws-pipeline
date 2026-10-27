"""Offline fake-APT contracts; never invoke apt, sudo, a compiler, or a model."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tools/install_arm_dependencies.py'
PACKAGES = ['gcc-arm-linux-gnueabihf', 'g++-arm-linux-gnueabihf', 'qemu-user-static']
PYTHON = [sys.executable, '-I', '-S', '-B'] + (['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else [])

FAKE_APT = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
args = sys.argv[2:]
kind = 'index' if 'update' in args else 'download' if '--download-only' in args else 'install'
log = root / 'calls.jsonl'
old = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
attempt = sum(row['kind'] == kind for row in old)
with log.open('a') as out:
    out.write(json.dumps(dict(kind=kind, args=args, pid=os.getpid())) + '\n')
plan = json.loads((root / 'plan.json').read_text())
actions = plan.get(kind, ['ok'])
action = actions[min(attempt, len(actions) - 1)]
if action.startswith('sleep:'):
    time.sleep(float(action.split(':')[1]))
elif action.startswith('fail:'):
    raise SystemExit(int(action.split(':')[1]))
elif action in ('hang', 'stubborn', 'child', 'orphan'):
    if action in ('child', 'orphan'):
        code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
        child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', code])
        with (root / 'children.jsonl').open('a') as out:
            out.write(json.dumps(dict(pid=child.pid)) + '\n')
        # Give the child time to install its signal handler.
        time.sleep(0.08)
        if action == 'orphan':
            raise SystemExit(0)
    if action in ('stubborn', 'child'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / 'ready').write_text(str(os.getpid()))
    time.sleep(60)
'''


def load_helper():
    spec = importlib.util.spec_from_file_location('arm_dependency_budget', HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WorkflowContractTests(unittest.TestCase):
    def test_bounded_helper_replaces_unbounded_apt_and_keeps_job_budget(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        arm = workflow.split('  armv7-cross:\n', 1)[1]
        dependency_step = arm.split('      - name: Install cross compiler', 1)[1].split(
            '      - name: Locate ARM sysroot', 1)[0]
        self.assertIn('timeout-minutes: 30', arm.split('    steps:', 1)[0])
        self.assertIn('timeout-minutes: 8', dependency_step)
        self.assertIn('sudo python3 -I -S -B tools/install_arm_dependencies.py', dependency_step)
        self.assertNotIn('apt-get', dependency_step)
        self.assertNotIn('continue-on-error', dependency_step)
        for mode in ('', '-O ', '-OO '):
            self.assertIn('python3 -I -S -B ' + mode + 'tests/test_arm_dependency_budget.py', workflow)
        self.assertTrue(HELPER.is_file())


class FakeAPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = load_helper()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fake-arm-apt-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fake = self.root / 'apt.py'
        self.fake.write_text(FAKE_APT)
        self.command = PYTHON + [str(self.fake), str(self.root)]
        self.output = io.StringIO()
        self.budget = self.arm.Budget(index=0.65, download=1.3, attempt=0.65,
                                      install=0.65, total=2.9, term=0.12, kill=0.12)

    def run_plan(self, plan, budget=None):
        (self.root / 'plan.json').write_text(json.dumps(plan))
        started = time.monotonic()
        code = self.arm.install(apt=self.command, budget=budget or self.budget, output=self.output)
        return code, time.monotonic() - started

    def calls(self):
        log = self.root / 'calls.jsonl'
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def kinds(self):
        return [row['kind'] for row in self.calls()]

    def assert_exited(self, pid):
        # A zombie has exited; it cannot perform work. Its adopter owns reaping.
        path = Path('/proc') / str(pid) / 'stat'
        if path.exists():
            try:
                state = path.read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                return
            self.assertEqual(state, 'Z', f'fake child {pid} is still running')

    def test_production_limits_and_no_extra_apt_retry_or_download_on_install(self):
        defaults = self.arm.Budget()
        self.assertEqual((defaults.index, defaults.download, defaults.attempt,
                          defaults.install, defaults.total), (60, 240, 120, 120, 440))
        code, _ = self.run_plan({})
        self.assertEqual(code, 0)
        self.assertEqual(self.kinds(), ['index', 'download', 'install'])
        for row in self.calls():
            args = row['args']
            for option in ('Acquire::Retries=0', 'Acquire::http::Timeout=20',
                           'Acquire::https::Timeout=20', 'APT::Get::Fix-Missing=false',
                           'APT::Get::AllowUnauthenticated=false'):
                self.assertIn(option, args)
            self.assertNotIn('--ignore-missing', args)
            self.assertNotIn('--allow-unauthenticated', args)
        self.assertIn('APT::Update::Error-Mode=any', self.calls()[0]['args'])
        self.assertIn('--download-only', self.calls()[1]['args'])
        self.assertIn('--no-download', self.calls()[2]['args'])
        for row in self.calls()[1:]:
            self.assertEqual(row['args'][-3:], PACKAGES)
        self.assertIn('result=success', self.output.getvalue())

    def test_download_failure_then_success_retries_only_download(self):
        code, _ = self.run_plan({'download': ['fail:100', 'ok']})
        self.assertEqual(code, 0)
        self.assertEqual(self.kinds(), ['index', 'download', 'download', 'install'])
        self.assertIn('phase=download-1', self.output.getvalue())
        self.assertIn('exit_code=100', self.output.getvalue())

    def test_download_exhaustion_propagates_without_install(self):
        code, _ = self.run_plan({'download': ['fail:100']})
        self.assertEqual(code, 100)
        self.assertEqual(self.kinds(), ['index', 'download', 'download'])
        self.assertNotIn('result=success', self.output.getvalue())

    def test_download_timeout_then_success_cleans_before_one_retry(self):
        code, elapsed = self.run_plan({'download': ['stubborn', 'ok']})
        self.assertEqual(code, 0)
        self.assertEqual(self.kinds(), ['index', 'download', 'download', 'install'])
        self.assertLess(elapsed, self.budget.download + 0.4)
        first_download = self.calls()[1]
        self.assert_exited(first_download['pid'])
        text = self.output.getvalue()
        self.assertLess(text.index('event=child_exit phase=download-1'),
                        text.index('event=phase_start phase=download-2'))

    def test_index_failure_signature_or_missing_index_is_fatal(self):
        for code in (1, 100):
            with self.subTest(code=code):
                if (self.root / 'calls.jsonl').exists():
                    (self.root / 'calls.jsonl').unlink()
                actual, _ = self.run_plan({'index': [f'fail:{code}']})
                self.assertEqual(actual, code)
                self.assertEqual(self.kinds(), ['index'])

    def test_missing_cached_package_or_install_failure_is_never_retried(self):
        code, _ = self.run_plan({'install': ['fail:100']})
        self.assertEqual(code, 100)
        self.assertEqual(self.kinds(), ['index', 'download', 'install'])
        self.assertNotIn('result=success', self.output.getvalue())

    def test_hung_index_terminates_before_budget(self):
        code, elapsed = self.run_plan({'index': ['hang']})
        self.assertEqual(code, 124)
        self.assertEqual(self.kinds(), ['index'])
        self.assertLess(elapsed, self.budget.index + 0.3)
        self.assertIn('signal=SIGTERM', self.output.getvalue())
        self.assertIn('child_exit_code=-15', self.output.getvalue())
        self.assert_exited(self.calls()[0]['pid'])

    def test_hung_downloads_share_budget_and_kill_stubborn_children(self):
        code, elapsed = self.run_plan({'download': ['child']})
        self.assertEqual(code, 124)
        self.assertEqual(self.kinds(), ['index', 'download', 'download'])
        self.assertLess(elapsed, self.budget.download + 0.4)
        self.assertIn('signal=SIGKILL', self.output.getvalue())
        self.assertIn('child_exit_code=-9', self.output.getvalue())
        for row in self.calls():
            self.assert_exited(row['pid'])
        children = [json.loads(line) for line in (self.root / 'children.jsonl').read_text().splitlines()]
        self.assertEqual(len(children), 2)
        for row in children:
            self.assert_exited(row['pid'])
        self.assertIn('live_pids=[]', self.output.getvalue())

    def test_parent_zero_exit_with_live_child_is_cleaned_and_never_success(self):
        code, _ = self.run_plan({'download': ['orphan']})
        self.assertEqual(code, 125)
        self.assertEqual(self.kinds(), ['index', 'download'])
        self.assertIn('orphaned_children', self.output.getvalue())
        self.assertIn('signal=SIGKILL', self.output.getvalue())
        for line in (self.root / 'children.jsonl').read_text().splitlines():
            self.assert_exited(json.loads(line)['pid'])

    def test_hung_install_is_not_retried_even_after_kill(self):
        code, elapsed = self.run_plan({'install': ['stubborn']})
        self.assertEqual(code, 124)
        self.assertEqual(self.kinds(), ['index', 'download', 'install'])
        self.assertLess(elapsed, self.budget.install + 0.4)
        self.assertNotIn('result=success', self.output.getvalue())

    def test_total_deadline_overrides_phase_budgets(self):
        budget = self.arm.Budget(index=0.65, download=1.3, attempt=0.65,
                                 install=0.65, total=0.65, term=0.12, kill=0.12)
        code, elapsed = self.run_plan({'index': ['sleep:0.2'], 'download': ['hang']}, budget)
        self.assertEqual(code, 124)
        # On a loaded runner index startup may consume the remaining launch
        # allowance. Either way no extra attempt/install may escape the total.
        self.assertEqual(self.kinds()[0], 'index')
        self.assertLessEqual(self.kinds().count('download'), 1)
        self.assertLessEqual(self.output.getvalue().count('event=phase_start phase=download-'), 1)
        self.assertNotIn('install', self.kinds())
        self.assertLess(elapsed, budget.total + 0.3)

    def test_absolute_global_deadline_and_exhausted_launch_with_controlled_clock(self):
        now = [0.0]
        seen = []
        real_phase = self.arm.run_phase

        def scripted_phase(command, name, deadline, budget, cancelled, log):
            seen.append((name, deadline))
            if name == 'index':
                now[0] = 2.0
                return 0, False
            if name == 'download-1':
                now[0] = 5.0
                return 124, False
            # Exercise the real pre-spawn deadline guard, not a fake success.
            return real_phase(command, name, deadline, budget, cancelled, log)

        budget = self.arm.Budget(total=5, term=1, kill=1)
        with patch.object(self.arm.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(self.arm, 'run_phase', side_effect=scripted_phase), \
                patch.object(self.arm.subprocess, 'Popen') as spawn:
            code = self.arm.install(apt=self.command, budget=budget, output=self.output)
        self.assertEqual(code, 124)
        self.assertEqual(seen, [('index', 5), ('download-1', 5), ('download-2', 5)])
        spawn.assert_not_called()
        self.assertIn('reason=no_budget_or_cancelled', self.output.getvalue())
        self.assertNotIn('result=success', self.output.getvalue())

    def test_spawn_and_cleanup_errors_fail_closed_without_retry(self):
        (self.root / 'plan.json').write_text('{}')
        with patch.object(self.arm.subprocess, 'Popen', side_effect=FileNotFoundError('fake missing')):
            self.assertEqual(self.arm.install(apt=self.command, budget=self.budget, output=self.output), 127)
        self.assertEqual(self.calls(), [])
        # Simulate permission failure without leaving a real process behind.
        proc = unittest.mock.Mock(pid=12345)
        proc.poll.return_value = None
        with patch.object(self.arm.subprocess, 'Popen', return_value=proc), \
                patch.object(self.arm, 'live_group_members', return_value=[12345]), \
                patch.object(self.arm.os, 'killpg', side_effect=PermissionError('denied')):
            code = self.arm.install(apt=self.command, budget=self.budget, output=self.output)
        self.assertEqual(code, 125)
        self.assertIn('cleanup_failed', self.output.getvalue())
        self.assertNotIn('phase=download', self.output.getvalue())

    def test_cleanup_that_never_confirms_exit_is_fatal(self):
        proc = unittest.mock.Mock(pid=12345, returncode=None)
        proc.poll.return_value = None
        with patch.object(self.arm.subprocess, 'Popen', return_value=proc) as spawn, \
                patch.object(self.arm, 'live_group_members', return_value=[12345]), \
                patch.object(self.arm.os, 'killpg') as kill:
            code = self.arm.install(apt=self.command, budget=self.budget, output=self.output)
        self.assertEqual(code, 125)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual([call.args[1] for call in kill.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL])
        self.assertIn('cleanup_failed', self.output.getvalue())

    def test_broken_log_pipe_during_cleanup_cannot_leave_child_or_retry(self):
        (self.root / 'plan.json').write_text(json.dumps({'index': ['stubborn']}))

        class BrokenOutput:
            def write(self, message):
                if 'event=cleanup' in message:
                    raise BrokenPipeError('synthetic closed log pipe')

            def flush(self):
                pass

        code = self.arm.install(apt=self.command, budget=self.budget, output=BrokenOutput())
        self.assertEqual(code, 125)
        self.assertEqual(self.kinds(), ['index'])
        self.assert_exited(self.calls()[0]['pid'])

    def test_unreadable_process_metadata_still_signals_then_fails_closed(self):
        with patch.object(self.arm, 'live_group_members', side_effect=PermissionError('unreadable')):
            code, _ = self.run_plan({'index': ['stubborn']})
        self.assertEqual(code, 125)
        self.assertEqual(self.kinds(), ['index'])
        self.assertIn('signal=SIGTERM', self.output.getvalue())
        self.assertIn('signal=SIGKILL', self.output.getvalue())
        self.assertIn('cleanup_failed', self.output.getvalue())
        self.assert_exited(self.calls()[0]['pid'])

    def test_external_term_cleans_process_group_and_propagates_143(self):
        self.external_signal(signal.SIGTERM, 143)

    def test_external_int_cleans_process_group_and_propagates_130(self):
        self.external_signal(signal.SIGINT, 130)

    def external_signal(self, sig, expected):
        (self.root / 'plan.json').write_text(json.dumps({'index': ['child']}))
        runner = self.root / 'runner.py'
        runner.write_text(
            'import importlib.util,sys\n'
            f'spec=importlib.util.spec_from_file_location("arm",{str(HELPER)!r})\n'
            'm=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)\n'
            f'raise SystemExit(m.install(apt={self.command!r},'
            'budget=m.Budget(index=4,download=4,attempt=2,install=4,total=8,term=.12,kill=.12)))\n')
        process = subprocess.Popen(PYTHON + [str(runner)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 3
            while not (self.root / 'ready').exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.root / 'ready').exists())
            process.send_signal(sig)
            output, _ = process.communicate(timeout=2)
            self.assertEqual(process.returncode, expected, output)
            self.assertIn('signal=SIGKILL', output)
            self.assertNotIn('result=success', output)
            self.assertEqual(self.kinds(), ['index'])
            self.assert_exited(self.calls()[0]['pid'])
            for line in (self.root / 'children.jsonl').read_text().splitlines():
                self.assert_exited(json.loads(line)['pid'])
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=2)

    def test_invalid_budget_rejected_without_execution(self):
        for value in (0, -1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.arm.Budget(index=value)
        with self.assertRaises(ValueError):
            self.arm.Budget(term=40, kill=40)

    def test_cli_rejects_nonroot_or_argument_overrides_without_apt(self):
        with patch.object(self.arm, 'install') as install, \
                patch.object(self.arm.sys, 'stderr', io.StringIO()):
            with patch.object(self.arm.sys, 'argv', [str(HELPER)]), \
                    patch.object(self.arm.os, 'geteuid', return_value=1000):
                self.assertEqual(self.arm.main(), 126)
            with patch.object(self.arm.sys, 'argv', [str(HELPER), '--apt=other']), \
                    patch.object(self.arm.os, 'geteuid', return_value=0):
                self.assertEqual(self.arm.main(), 126)
            install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
