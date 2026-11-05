#!/usr/bin/env python3
"""Offline invocation-boundary tests; no model imports or generation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import active_tts_smoke as smoke


def plan_fixture():
    return {'cell_layout': 'd1-interleaved-32-v1', 'references': [{} for _ in range(8)],
            'cells': [{'cell_id': 'c' + str(i), 'reference_id': 'r' + str(i % 8),
                       'source_family_id': 'f' + str(i % 8),
                       'role': 'train-candidate' if i % 8 < 6 else 'development-only'}
                      for i in range(256)]}


class FakeClone:
    def __init__(self):
        self.calls = []

    def _model(self):
        return self

    def _prompt(self, ref, audio_root):
        return {'reference': ref['reference_id']}

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        return [[0.1, -0.1]], 24000


class TTSSmokeContract(unittest.TestCase):
    def test_selection_keeps_one_per_source_and_both_roles(self):
        plan = plan_fixture()
        self.assertEqual(smoke.selection(plan), ['c' + str(i) for i in range(8)])
        plan['cells'][7]['reference_id'] = 'r0'
        with self.assertRaisesRegex(ValueError, 'SMOKE_SOURCE_COVERAGE'):
            smoke.selection(plan)
        plan = plan_fixture()
        plan['cells'][7]['role'] = 'train-candidate'
        with self.assertRaisesRegex(ValueError, 'SMOKE_SOURCE_COVERAGE'):
            smoke.selection(plan)

    def test_new_and_repeat_cells_cannot_consume_extra_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            delegate = FakeClone()
            backend = smoke.BoundedBackend(delegate, directory,
                {'selected_cells': ['c0'], 'max_calls': 1, 'generation': smoke.GENERATION},
                native_writer=lambda folder, audio, rate: smoke.once(folder / 'raw-saved.json', {'rate': rate}))
            cell = {'cell_id': 'c0', 'seed': 100, 'prompt': '你好小窝'}
            ref = {'reference_id': 'r0', 'source_family_id': 'f0', 'wav_sha256': 'a' * 64,
                   'pcm_sha256': 'b' * 64, 'reference_text_status': 'dataset-provided-content-checked'}
            with patch.dict(sys.modules, {'torch': SimpleNamespace(manual_seed=lambda seed: None)}):
                backend.synthesize(cell, ref, directory)
                for candidate in (cell, dict(cell, cell_id='c1')):
                    with self.assertRaisesRegex(ValueError, 'SMOKE_CALL_BOUNDARY'):
                        backend.synthesize(candidate, ref, directory)
            self.assertEqual(len(delegate.calls), 1)
            self.assertEqual(delegate.calls[0]['max_new_tokens'], 256)
            self.assertEqual(delegate.calls[0]['language'], 'Chinese')
            self.assertTrue((Path(directory) / 'native/c0/raw-saved.json').is_file())
            call = smoke.read(Path(directory) / 'native/c0/call.json')
            self.assertEqual(call['reference_mode'], 'ICL')
            self.assertTrue(call['reference_text_passed'])

    def test_raw_failure_keeps_consumed_call_and_never_retries(self):
        def fail(folder, audio, rate):
            smoke.once(folder / 'partial-raw.json', {'rate': rate})
            raise ValueError('synthetic output retention failure')
        with tempfile.TemporaryDirectory() as directory:
            delegate = FakeClone()
            backend = smoke.BoundedBackend(delegate, directory,
                {'selected_cells': ['c0'], 'max_calls': 1, 'generation': smoke.GENERATION}, fail)
            cell = {'cell_id': 'c0', 'seed': 100, 'prompt': '天气不错'}
            ref = {'reference_id': 'r0', 'source_family_id': 'f0', 'wav_sha256': 'a' * 64,
                   'pcm_sha256': 'b' * 64, 'reference_text_status': 'dataset-provided'}
            with patch.dict(sys.modules, {'torch': SimpleNamespace(manual_seed=lambda seed: None)}):
                with self.assertRaisesRegex(ValueError, 'retention failure'):
                    backend.synthesize(cell, ref, directory)
                with self.assertRaisesRegex(ValueError, 'SMOKE_CALL_BOUNDARY'):
                    backend.synthesize(cell, ref, directory)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(len(delegate.calls), 1)
            folder = Path(directory) / 'native/c0'
            self.assertTrue((folder / 'partial-raw.json').is_file())
            self.assertTrue((folder / 'error.json').is_file())
            self.assertEqual(smoke.read(folder / 'call.json')['reference_mode'], 'x-vector-only')

    def test_invocation_claim_is_create_only_and_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invocation.json'
            smoke.once(path, {'selected_cells': ['a'], 'calls_allowed': 8})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                smoke.once(path, {'selected_cells': ['b'], 'calls_allowed': 8})
            self.assertEqual(path.read_bytes(), original)

    def test_paths_reject_escape(self):
        for path in ('../source', '/tmp/source', 'a\\source'):
            with self.assertRaisesRegex(ValueError, 'UNSAFE_RELATIVE_PATH'):
                smoke.relative(path)


if __name__ == '__main__':
    unittest.main()
