import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import controller
from controller import Controller, Rejected, CONTROL_RESERVE, DISK_FLOOR, zero_bounds

HERE = Path(__file__).resolve().parent

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'state'
        self.free = DISK_FLOOR + 16 * 1024**2

    def create(self, ids=None, budget=4096):
        return Controller.create(self.root, ids or ['a', 'b'],
                                 total_bytes=CONTROL_RESERVE + budget,
                                 disk_free=lambda: self.free)

    def complete(self, c):
        c.execute(lambda item, w: w.write('output', 'x', b'ab'), per_item_bounds=zero_bounds(output=2))

    def test_actual_roster_16_no_legacy_eight(self):
        c = self.create([str(x) for x in range(16)])
        seen = []
        c.execute(lambda item, w: seen.append(item), per_item_bounds=zero_bounds())
        self.assertEqual(len(seen), 16)
        self.assertEqual(c.status()['counts'], dict(planned=16, consumed_or_unknown=16,
                                                 completed=16, unknown=0, not_started=0))
        self.assertEqual(c.status()['delivery']['status'], 'pending')
        with self.assertRaises(Rejected):
            c.execute(lambda *a: self.fail('called twice'), per_item_bounds=zero_bounds())

    def test_all_five_categories_fail_before_delivery_callback(self):
        c = self.create(budget=64)
        self.complete(c)
        # Four local bytes plus all five declared increments, not just output.
        with self.assertRaises(Rejected):
            c.deliver(lambda w: self.fail('delivery called'), peak_bounds=zero_bounds(
                output=1, archive=16, upload=16, readback=16, temp=16))
        self.assertEqual(c.status()['execution']['status'], 'completed')
        self.assertEqual(c.status()['delivery']['status'], 'pending')

    def test_total_execution_bound_prevents_any_callback(self):
        c = self.create(budget=10)
        with self.assertRaises(Rejected):
            c.execute(lambda *a: self.fail('called'), per_item_bounds=zero_bounds(output=6))
        self.assertEqual(c.status()['counts']['consumed_or_unknown'], 0)

    def test_disk_floor_checked_before_callback_and_each_write(self):
        c = self.create()
        self.free = DISK_FLOOR + CONTROL_RESERVE + 3
        with self.assertRaises(Rejected):
            self.complete(c)
        self.assertEqual(c.status()['execution']['status'], 'pending')
        self.free += 100
        def callback(item, writer):
            self.free = DISK_FLOOR + CONTROL_RESERVE + 1
            writer.write('output', 'x', b'ab')
        with self.assertRaises(Rejected):
            c.execute(callback, per_item_bounds=zero_bounds(output=2))
        self.assertFalse((self.root / 'output' / 'exec-0000' / 'x').exists())
        self.assertEqual(c.status()['counts']['unknown'], 1)

    def test_category_cap_and_failed_artifact_is_retained(self):
        c = self.create()
        def callback(item, w):
            w.write('output', 'first', b'a')
            w.write('output', 'too-much', b'ab')
        with self.assertRaises(Rejected):
            c.execute(callback, per_item_bounds=zero_bounds(output=2))
        self.assertEqual(c.status()['storage_bytes']['output'], 1)
        self.assertEqual(c.status()['counts'], dict(planned=2, consumed_or_unknown=1,
                                                 completed=0, unknown=1, not_started=1))
        with self.assertRaises(Rejected):
            self.complete(c)

    def test_delivery_unknown_reconciliation_retry_never_reexecutes(self):
        c = self.create()
        self.complete(c)
        def failed(w):
            w.write('archive', 'partial', b'keep')
            raise RuntimeError('local fixture failed before network')
        with self.assertRaises(RuntimeError):
            c.deliver(failed, peak_bounds=zero_bounds(archive=4))
        self.assertEqual(c.status()['delivery']['status'], 'unknown')
        with self.assertRaises(Rejected):
            c.deliver(lambda w: self.fail('unreconciled retry'), peak_bounds=zero_bounds())
        c.reconcile_delivery(remote_absence_verified=True, evidence='test fixture performed no network calls')
        def done(w):
            for category in ('archive', 'upload', 'readback', 'temp'):
                w.write(category, 'copy', b'new')
            return {'verified': True, 'scope': 'fixture only'}
        c.deliver(done, peak_bounds=zero_bounds(archive=3, upload=3, readback=3, temp=3))
        self.assertEqual(c.status()['storage_bytes']['archive'], 7)
        self.assertEqual(c.status()['counts']['completed'], 2)
        self.assertEqual(len(c.status()['delivery']['attempts']), 2)
        self.assertEqual(c.status()['delivery']['status'], 'completed')
        with self.assertRaises(Rejected):
            c.deliver(done, peak_bounds=zero_bounds())

    def test_failed_execution_can_deliver_failure_evidence(self):
        c = self.create()
        def failed(item, writer):
            raise RuntimeError('fixture')
        with self.assertRaises(RuntimeError):
            c.execute(failed, per_item_bounds=zero_bounds())
        c.deliver(lambda w: {'verified': True, 'scope': 'failure evidence'}, peak_bounds=zero_bounds())
        self.assertEqual(c.status()['execution']['status'], 'failed')
        self.assertEqual(c.status()['delivery']['status'], 'completed')

    def test_incomplete_delivery_receipt_is_not_success(self):
        c = self.create()
        self.complete(c)
        with self.assertRaises(Rejected):
            c.deliver(lambda w: {'verified': False}, peak_bounds=zero_bounds())
        self.assertEqual(c.status()['delivery']['status'], 'unknown')

    def test_input_validation_and_no_root_reset(self):
        with self.assertRaises(Rejected):
            self.create(['a', 'a'])
        c = self.create()
        with self.assertRaises(Rejected):
            self.create()
        with self.assertRaises(Rejected):
            c.execute(lambda *a: None, per_item_bounds={'output': 0})
        with self.assertRaises(Rejected):
            c.execute(lambda *a: None, per_item_bounds=zero_bounds(output=True))

    def test_path_escape_and_symlink_rejected(self):
        c = self.create()
        with self.assertRaises(Rejected):
            c.execute(lambda item, w: w.write('output', '../escape', b'x'),
                      per_item_bounds=zero_bounds(output=1))
        self.assertFalse((self.root / 'output' / 'escape').exists())
        (self.root / 'output' / 'link').symlink_to(self.root / 'state.json')
        with self.assertRaises(Rejected):
            c.status()

    def test_writer_cannot_be_used_after_callback(self):
        c = self.create()
        saved = []
        c.execute(lambda item, w: saved.append(w), per_item_bounds=zero_bounds(output=2))
        with self.assertRaises(Rejected):
            saved[0].write('output', 'late', b'x')

    def test_crash_after_consumption_blocks_execution_on_reopen(self):
        c = self.create()
        code = ('import os; from controller import Controller,zero_bounds; '
                'c=Controller(' + repr(str(self.root)) + '); '
                'c.execute(lambda *a: os._exit(17),per_item_bounds=zero_bounds())')
        child = subprocess.run([sys.executable, '-B', '-c', code], cwd=HERE, capture_output=True)
        self.assertEqual(child.returncode, 17, child.stderr)
        reopened = Controller(self.root, disk_free=lambda: self.free)
        self.assertEqual(reopened.status()['counts']['unknown'], 1)
        with self.assertRaises(Rejected):
            self.complete(reopened)
        with self.assertRaises(Rejected):
            reopened.seal_failed_execution(processes_stopped_verified=False, evidence='not checked')
        sealed = subprocess.run([sys.executable, '-B', str(HERE / 'controller.py'), 'seal-failed',
                                 '--root', str(self.root), '--processes-stopped-verified',
                                 '--evidence', 'test subprocess exited 17, no descendants were launched'],
                                capture_output=True, text=True)
        self.assertEqual(sealed.returncode, 0, sealed.stderr)
        result = json.loads(sealed.stdout)
        self.assertEqual(result['execution']['status'], 'failed')
        self.assertEqual(result['counts']['unknown'], 1)
        self.assertEqual(result['counts']['not_started'], 1)
        self.assertEqual(result['counts']['completed'], 0)
        reopened.deliver(lambda w: {'verified': True, 'scope': 'crash evidence fixture'},
                         peak_bounds=zero_bounds())
        self.assertEqual(reopened.status()['delivery']['status'], 'completed')
        with self.assertRaises(Rejected):
            self.complete(reopened)

    def test_seal_rejects_active_lock_and_preserves_consumption(self):
        c = self.create()
        with c._locked():
            state = c._load()
            state['execution']['status'] = 'started'
            state['execution']['items']['a'] = 'started'
            c._save(state)
            with self.assertRaisesRegex(Rejected, 'controller is active'):
                c.seal_failed_execution(processes_stopped_verified=True, evidence='fixture claim')
            self.assertEqual(c._load()['execution']['status'], 'started')
        c.seal_failed_execution(processes_stopped_verified=True, evidence='fixture has no worker process')
        self.assertEqual(c.status()['counts']['consumed_or_unknown'], 1)
        self.assertEqual(c.status()['counts']['unknown'], 1)
        with self.assertRaises(Rejected):
            c.seal_failed_execution(processes_stopped_verified=True, evidence='already terminal')

    def test_streaming_bounds_read_size_and_retains_failed_prefix(self):
        c = self.create(['one'], budget=300000)
        class Source:
            remaining = 140000
            requests = []
            def read(self, n):
                self.requests.append(n)
                take = min(n, self.remaining)
                self.remaining -= take
                return b'x' * take
        source = Source()
        c.execute(lambda item, w: w.write_stream('output', 'large', source, expected_bytes=140000),
                  per_item_bounds=zero_bounds(output=140000))
        self.assertLessEqual(max(source.requests), 65536)
        self.assertEqual(c.status()['storage_bytes']['output'], 140000)
        def bad(w):
            w.write_stream('archive', 'partial', io.BytesIO(b'longer'), expected_bytes=3)
        with self.assertRaises(Rejected):
            c.deliver(bad, peak_bounds=zero_bounds(archive=3))
        self.assertEqual(c.status()['storage_bytes']['archive'], 3)
        self.assertEqual((self.root / 'archive' / 'delivery-0001' / 'partial').read_bytes(), b'lon')

    def test_prefix_parent_is_synced_before_file_and_completion(self):
        c = self.create(['one'])
        events = []
        original = controller.sync_dir
        def observed(path):
            events.append((Path(path).name, (self.root / 'output' / 'exec-0000' / 'x').exists()))
            return original(path)
        with patch('controller.sync_dir', side_effect=observed):
            self.complete(c)
        self.assertIn(('output', False), events)
        self.assertIn(('exec-0000', True), events)
        self.assertLess(events.index(('output', False)), events.index(('exec-0000', True)))
        self.assertEqual(c.status()['execution']['status'], 'completed')

    def test_parent_sync_failure_blocks_artifact_and_preserves_unknown(self):
        c = self.create(['one'])
        original = controller.sync_dir
        def fail_parent(path):
            if Path(path) == self.root / 'output':
                raise OSError('injected prefix directory persistence failure')
            return original(path)
        with patch('controller.sync_dir', side_effect=fail_parent):
            with self.assertRaises(OSError):
                self.complete(c)
        self.assertFalse((self.root / 'output' / 'exec-0000' / 'x').exists())
        self.assertEqual(c.status()['counts']['unknown'], 1)
        with self.assertRaises(Rejected):
            self.complete(c)

    def test_real_cli_fixture_and_status(self):
        command = [sys.executable, '-B', str(HERE / 'controller.py')]
        done = subprocess.run(command + ['fixture', '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(done.stdout)
        self.assertEqual(result['counts']['completed'], 16)
        self.assertEqual(result['delivery']['status'], 'completed')
        for category in ('output', 'archive', 'upload', 'readback', 'temp'):
            self.assertGreater(result['storage_bytes'][category], 0)
        readback = subprocess.run(command + ['status', '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(readback.returncode, 0, readback.stderr)
        self.assertEqual(json.loads(readback.stdout), result)

if __name__ == '__main__':
    unittest.main()
