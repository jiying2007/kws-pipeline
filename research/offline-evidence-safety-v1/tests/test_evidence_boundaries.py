"""Deterministic deadline and raw-serialization boundaries; no producer launches."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from offline_safety import guard, supervisor


class SnapshotBoundaryTests(unittest.TestCase):
    def test_reserved_names_and_npy_aliases_preserve_exact_bytes(self):
        arrays = {key: np.array([index, np.nan], dtype=np.float64)
                  for index, key in enumerate(('file', 'allow_pickle', 'args', 'kwds',
                                               'arr_0', 'x', 'x.npy'))}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.npz'
            guard._snapshot_arrays(path, arrays)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {key + '.npy' for key in arrays})
                for key, original in arrays.items():
                    with archive.open(key + '.npy') as member:
                        saved = np.lib.format.read_array(member, allow_pickle=False)
                    self.assertEqual(saved.dtype, original.dtype)
                    self.assertEqual(saved.shape, original.shape)
                    self.assertEqual(saved.tobytes(), original.tobytes())

    def test_reserved_names_saved_before_disarmed_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'evidence'
            arrays = {'file': np.array([np.inf]), 'allow_pickle': np.zeros((0, 2))}
            with self.assertRaises(PermissionError):
                guard.save_then_compare(dest, arrays, arrays, {}, helper_path='unused',
                                        helper_sha256='', ticket=None, work=tmp, attempt=tmp)
            terminal = json.loads((dest / 'TERMINAL.json').read_text())
            self.assertTrue(terminal['complete'])
            self.assertFalse(terminal['qualified'])
            self.assertTrue((dest / 'RAW-SAVED.json').is_file())

    def test_object_dtype_still_refused_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(np.lib.format, 'write_array') as write:
                with self.assertRaises(TypeError):
                    guard._snapshot_arrays(Path(tmp) / 'raw.npz',
                                           {'file': np.array([object()], dtype=object)})
                write.assert_not_called()


class DeadlineBoundaryTests(unittest.TestCase):
    def capture_at(self, initial, final):
        now = [10.0]
        waits = []
        entries = {}
        pipes = [io.BytesIO(), io.BytesIO()]
        for index, pipe in enumerate(pipes):
            pipe.fileno = lambda index=index: index + 50
        process = SimpleNamespace(stdout=pipes[0], stderr=pipes[1], returncode=0,
                                  poll=lambda: 0, wait=lambda timeout: 0)
        def register(pipe, events, name):
            entries[name] = SimpleNamespace(fileobj=pipe, data=name)
            now[0] = 10.0 + initial
        def select(timeout):
            waits.append(timeout)
            now[0] = 10.0 + final
            return [(key, 1) for key in entries.values()]
        selector = SimpleNamespace(register=register, get_map=lambda: entries,
                                   select=select, close=lambda: None,
                                   unregister=lambda pipe: entries.pop(
                                       'stdout' if pipe is pipes[0] else 'stderr'))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(supervisor.selectors, 'DefaultSelector', return_value=selector), \
                patch.object(supervisor.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(supervisor.os, 'set_blocking'), \
                patch.object(supervisor.os, 'read', return_value=b''):
            result = supervisor.capture(process, Path(tmp) / 'capture', timeout_seconds=1)
            terminal = json.loads((Path(tmp) / 'capture/TERMINAL.json').read_text())
            self.assertEqual(result, terminal)
        return result, waits

    def test_final_eof_at_and_after_deadline_never_completes(self):
        for final in (1.0, 1.001):
            with self.subTest(final=final):
                result, waits = self.capture_at(.995, final)
                self.assertAlmostEqual(waits[0], .005)
                self.assertEqual(result['reason'], 'timeout')
                self.assertFalse(result['complete'])
                self.assertFalse(result['capture_ok'])
                self.assertFalse(result['qualified'])

    def test_final_eof_just_before_deadline_completes(self):
        result, waits = self.capture_at(.995, .999)
        self.assertAlmostEqual(waits[0], .005)
        self.assertTrue(result['complete'])
        self.assertTrue(result['capture_ok'])

    def test_deadline_already_reached_never_selects(self):
        result, waits = self.capture_at(1.0, 1.0)
        self.assertEqual(waits, [])
        self.assertEqual(result['reason'], 'timeout')


if __name__ == '__main__':
    unittest.main()
