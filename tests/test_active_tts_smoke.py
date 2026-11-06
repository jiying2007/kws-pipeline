#!/usr/bin/env python3
"""Offline invocation-boundary tests; no model imports or generation."""
import copy
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


def import_receipt_fixture():
    versions = {'torch': '2.11.0+cpu', 'numpy': '1.26.4', 'scipy': '1.17.1', 'soundfile': '0.13.1'}
    setup = {'status': 'complete', 'active_python': '3.12.16', 'installed_packages': versions}
    result = {'python': '3.12.16', **versions, 'qwen_class': 'Qwen3TTSModel', 'cuda_compiled': None,
              'model_objects_constructed': 0, 'model_calls': 0}
    receipt = {'schema': 'kws-runtime-import-preflight-v1', 'role': 'tts',
               'argv': ['/fixture/tts/bin/python', '-B', '-I', '-c', 'fictional import-only command'],
               'returncode': 0, 'stdout': 'Dependency warning retained\n' + json.dumps(result) + '\n',
               'stderr': 'Original dependency diagnostic\n', 'elapsed_seconds': 0.25,
               'recorded_at': '2026-11-06T00:00:00+00:00',
               'model_objects_constructed': 0, 'model_calls': 0,
               'network_environment': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}}
    return setup, receipt


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

    def test_legacy_preflight_interface_preserves_opaque_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{"legacy-format": true}\n'
            (root / 'preflight.json').write_bytes(original)
            raw, explicit_source = smoke.preflight_input(SimpleNamespace(runtime=root))
            self.assertEqual(raw, original)
            self.assertIsNone(explicit_source)
            self.assertEqual((root / 'preflight.json').read_bytes(), original)

    def test_explicit_receipt_is_copied_verbatim_and_source_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, data = root / 'runtime', root / 'data'
            runtime.mkdir()
            setup, receipt = import_receipt_fixture()
            (runtime / 'setup.json').write_text(json.dumps(setup))
            original = (json.dumps(receipt, indent=3) + '\n\n').encode()
            receipt_path = root / 'retained-import.json'
            receipt_path.write_bytes(original)
            sources = root / 'sources.json'
            sources.write_text('{}\n')
            for name in smoke.SOURCE_FILES:
                path = data / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture producer\n')
            fake = SimpleNamespace(LOCK_SCHEMA='fixture-lock', asset_lock=lambda *args: None,
                p=SimpleNamespace(read_json=lambda path: {}, validate_sources=lambda *args: [],
                    plan=lambda *args, **kwargs: plan_fixture(), D1_LAYOUT='d1-interleaved-32-v1'))
            args = SimpleNamespace(output=root / 'prepared', runtime=runtime, data_root=data,
                audio_root=root, sources=sources, batch_id='fictional-first-eight', runtime_preflight=receipt_path)
            with patch.object(smoke, 'load_bulk', return_value=fake), \
                    patch.object(smoke.subprocess, 'check_output', return_value='a' * 40 + '\n'):
                result = smoke.prepare(args)
            recipe = smoke.read(result['recipe'])
            self.assertEqual((args.output / 'runtime-preflight.json').read_bytes(), original)
            self.assertEqual(recipe['runtime_preflight_source'], {'path': str(receipt_path), **smoke.record(receipt_path)})
            self.assertEqual(recipe['runtime_preflight'], smoke.record(receipt_path))
            self.assertFalse((runtime / 'preflight.json').exists())
            self.assertEqual(receipt_path.read_bytes(), original)
            receipt_path.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'TTS_IMPORT_PREFLIGHT_SOURCE_CHANGED'):
                smoke.verify_recipe(recipe)
            self.assertEqual((args.output / 'runtime-preflight.json').read_bytes(), original)

    def test_explicit_failed_or_incompatible_receipts_are_rejected(self):
        setup, good = import_receipt_fixture()
        cases = []
        for key, value in [('role', 'asr'), ('returncode', 1), ('model_calls', 1),
                           ('model_objects_constructed', False), ('elapsed_seconds', -1),
                           ('network_environment', {})]:
            bad = copy.deepcopy(good)
            bad[key] = value
            cases.append(bad)
        for key, value in [('model_calls', 1), ('numpy', 'different-version'), ('cuda_compiled', '12.0')]:
            bad = copy.deepcopy(good)
            payload = json.loads(good['stdout'].splitlines()[-1])
            payload[key] = value
            bad['stdout'] = json.dumps(payload) + '\n'
            cases.append(bad)
        bad = copy.deepcopy(good)
        bad['stdout'] += good['stdout'].splitlines()[-1] + '\n'
        cases.append(bad)
        for bad in cases:
            with self.subTest(receipt=bad), self.assertRaises(ValueError):
                smoke.validate_import_preflight(json.dumps(bad).encode(), setup)

    def test_bad_explicit_receipt_fails_before_preparation_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            runtime.mkdir()
            setup, receipt = import_receipt_fixture()
            (runtime / 'setup.json').write_text(json.dumps(setup))
            source = root / 'invalid.json'
            receipt['returncode'] = 1
            source.write_text(json.dumps(receipt))
            args = SimpleNamespace(output=root / 'prepared', runtime=runtime, runtime_preflight=source)
            with patch.object(smoke, 'load_bulk') as loader:
                with self.assertRaisesRegex(ValueError, 'NOT_SUCCESSFUL_IMPORT_ONLY'):
                    smoke.prepare(args)
                loader.assert_not_called()
            self.assertFalse(args.output.exists())
            alias = root / 'receipt-alias.json'
            alias.symlink_to(source)
            args.runtime_preflight = alias
            with self.assertRaisesRegex(ValueError, 'EXPECTED_REGULAR_FILE'):
                smoke.prepare(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
