"""Independent review-only checks; invented processes and in-memory observations."""
import ast
import builtins
import importlib.util
import json
import pathlib
import signal
import unittest
from unittest import mock

SOURCE_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src/launch_once.py'
SOURCE = SOURCE_PATH.read_text()
SPEC = importlib.util.spec_from_file_location('independently_reviewed_launcher', SOURCE_PATH)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)
PID = 987654321
STATUS = 'VmRSS: 1 kB\nThreads: 1\nVmHWM: 1 kB\nVmSize: 1 kB\n'


class InventedProcess:
    pid = PID

    def __init__(self, polls):
        self.polls = iter(polls)

    def poll(self):
        return next(self.polls)

    def wait(self):
        return 0


def read_memory(status=STATUS, children='', io='rchar: 0\n'):
    def read(path):
        p = str(path)
        value = status if p.endswith('/status') else children if p.endswith('/children') else io
        if isinstance(value, Exception):
            raise value
        return value
    return read


class IndependentBoundaryTests(unittest.TestCase):
    def setUp(self):
        forbidden = AssertionError('live operation forbidden in independent review')
        for target, member in ((pathlib.Path, 'read_text'), (launch.subprocess, 'Popen'),
                               (launch.os, 'killpg'), (launch.os, 'sched_setaffinity'),
                               (launch.resource, 'setrlimit'), (launch.signal, 'alarm')):
            patched = mock.patch.object(target, member, side_effect=forbidden).start()
            self.addCleanup(patched.assert_not_called)
        self.addCleanup(mock.patch.stopall)

    def test_poll_precedes_reads_and_repoll_follows_all_reads(self):
        events = []
        proc = InventedProcess([None, 0])
        original_poll = proc.poll
        def poll():
            events.append('poll')
            return original_poll()
        proc.poll = poll
        original_read = read_memory()
        def read(path):
            events.append(str(path).split('/')[-1])
            return original_read(path)
        state = launch.observe_process(proc, read)
        self.assertEqual(events, ['poll', 'status', 'children', 'io', 'poll'])
        self.assertEqual(state['returncode'], 0)

    def test_mixed_disappearance_and_permission_denial_never_relaxes(self):
        missing = FileNotFoundError(2, 'invented missing', '/proc/invented/status')
        denied = PermissionError(13, 'invented denied', '/proc/invented/children')
        state = launch.observe_process(InventedProcess([None, 0]), read_memory(missing, denied))
        self.assertEqual(state['critical_monitoring'], 'FAILED_UNAVAILABLE')
        self.assertEqual(launch.guard_reason(state, 1, 0), 'CRITICAL_MONITORING_UNAVAILABLE')

    def test_known_crossing_survives_disappearance_and_exit_in_monitor(self):
        missing = FileNotFoundError(2, 'invented missing', '/proc/invented/children')
        kill = mock.Mock(side_effect=AssertionError('exited child must not be killed'))
        result = launch.monitor_process(
            InventedProcess([None, 0, 0]), 0, lambda: 0,
            read_text=read_memory(STATUS.replace('VmRSS: 1 kB', 'VmRSS: 262145 kB'), missing),
            clock=lambda: 1, sleep=mock.Mock(), killpg=kill)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['guard_stop_reason'], 'RSS_LIMIT')
        kill.assert_not_called()

    def test_each_counter_retains_its_own_last_successful_sample(self):
        state1 = {'returncode': None, 'critical_monitoring': 'OBSERVED',
                  'io': launch.unknown_io('invented unobserved')}
        state1['io']['rchar'] = {'status': 'AVAILABLE', 'value': 0}
        state2 = {'returncode': 0, 'critical_monitoring': 'OBSERVED',
                  'io': launch.unknown_io('invented unobserved')}
        state2['io']['wchar'] = {'status': 'AVAILABLE', 'value': 7}
        with mock.patch.object(launch, 'observe_process', side_effect=[state1, state2]):
            record = launch.monitor_process(InventedProcess([]), 0, lambda: 0,
                                            clock=mock.Mock(side_effect=[1, 2]),
                                            sleep=mock.Mock(), killpg=mock.Mock())
        self.assertEqual(record['io_last_successful_observation']['rchar'],
                         {'status': 'AVAILABLE', 'value': 0, 'wall_s': 1})
        self.assertEqual(record['io_last_successful_observation']['wchar'],
                         {'status': 'AVAILABLE', 'value': 7, 'wall_s': 2})
        self.assertEqual(record['io_final_observation']['rchar']['status'], 'UNKNOWN')

    def test_exact_serialized_byte_budget_and_one_byte_under(self):
        original = {'status': 'RAW_COMPLETE_PENDING_AUDIT', 'guard_stop_reason': None,
                    'proc_samples': [], 'unicode': '\u4f60\u597d'}
        _, data = launch.encode_resource_record(original, 0)
        cap = launch.LIMITS['output_bytes']
        record, exact = launch.encode_resource_record(original, cap - len(data))
        self.assertEqual(exact, data)
        self.assertEqual(record['status'], 'RAW_COMPLETE_PENDING_AUDIT')
        record, insufficient = launch.encode_resource_record(original, cap - len(data) + 1)
        self.assertEqual(record['status'], 'FAILED_NO_RETRY')
        self.assertTrue(insufficient is None or len(insufficient) <= len(data) - 1)
        self.assertEqual(original['status'], 'RAW_COMPLETE_PENDING_AUDIT')

    def test_per_file_limit_and_unreducible_evidence_return_no_bytes(self):
        original = {'status': 'RAW_COMPLETE_PENDING_AUDIT', 'guard_stop_reason': None,
                    'proc_samples': [], 'message': 'x' * launch.PER_FILE_BYTES}
        record, data = launch.encode_resource_record(original, 0)
        self.assertEqual(record['status'], 'FAILED_NO_RETRY')
        self.assertEqual(record['guard_stop_reason'], 'RESOURCE_RECORD_OUTPUT_LIMIT')
        self.assertIsNone(data)

    def test_gate_is_first_statement_and_rejects_before_any_io(self):
        run = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        self.assertIsInstance(run.body[0], ast.Raise)
        with mock.patch.object(builtins, 'open', side_effect=AssertionError('file I/O forbidden')) as opened, \
             mock.patch.object(launch.os, 'open', side_effect=AssertionError('file I/O forbidden')) as os_open, \
             mock.patch.object(launch.pathlib.Path, 'read_bytes', side_effect=AssertionError('file I/O forbidden')) as read_bytes, \
             mock.patch.object(launch, 'load', side_effect=AssertionError('load forbidden')) as load:
            with self.assertRaisesRegex(RuntimeError, '^FUTURE_DRAFT_NOT_RELEASED:'):
                launch.run(object())
        opened.assert_not_called()
        os_open.assert_not_called()
        read_bytes.assert_not_called()
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
