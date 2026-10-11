"""Offline adversarial metadata/orchestration tests. Never install or load models."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tests'))
from common import (MODEL_SHA, JOB_SHA, LIMITS, BOUNDARY, read_json, write_once,
                    file_hash, identity, verify_inventory, bind_input, wav_bytes, canonical)
import common
import adapter
import hosted_run as host
from asr_native_fixtures import native_fixture
from asr_native_contracts import WHISPER_ADAPTER


class ContractTests(unittest.TestCase):
    def test_no_model_import_at_module_import(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('whisper', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_frozen_request(self):
        from asr_native_contracts import WHISPER_REQUEST
        self.assertEqual(WHISPER_REQUEST['prompt'], None)
        self.assertEqual(WHISPER_REQUEST['prefix'], None)
        self.assertEqual(WHISPER_REQUEST['beam_size'], None)
        self.assertEqual(WHISPER_REQUEST['temperature'], 0.0)
        self.assertEqual(WHISPER_REQUEST['language'], 'zh')
        self.assertFalse(WHISPER_REQUEST['fp16'])

    def test_scope_policy(self):
        self.assertEqual(LIMITS['clips'], 16)
        self.assertLessEqual(LIMITS['job_seconds'], 2700)
        for key in ('new_tts_calls', 'new_qwen_calls', 'new_sensevoice_calls'):
            self.assertEqual(LIMITS[key], 0)
        self.assertEqual(BOUNDARY['source_role'], 'source_screen_only')
        self.assertFalse(BOUNDARY['training_admitted'])
        self.assertFalse(BOUNDARY['human_gold'])

    def test_real_retained_input_exact_bytes(self):
        old = host.historical_host()
        sys.path.insert(0, str(ROOT / 'research/source-screen32-v1'))
        import sense_continuation
        lock = sense_continuation.load_lock(old)
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / 'input'
            expected = sense_continuation.prepare_input(dest, lock, old)
            self.assertEqual(bind_input(dest), expected)
            self.assertEqual(file_hash(dest / 'job.json'), JOB_SHA)
            clip = expected['clips'][0]
            pcm, frames = wav_bytes(dest / clip['audio_path'], clip['wav_sha256'])
            self.assertEqual(len(pcm), frames * 2)
            self.assertEqual(frames, 44800)
            with (dest / 'job.json').open('ab') as stream: stream.write(b' ')
            with self.assertRaises(ValueError): bind_input(dest)

    def effective(self):
        return native_fixture(WHISPER_ADAPTER)[0]['effective_config']

    def trace(self, generated, completed_last=True):
        eff = self.effective(); initial = eff['initial_token_ids']
        return [{'step': i, 'token_ids': initial + generated[:i],
                 'completed': bool(completed_last and i == len(generated))}
                for i in range(1, len(generated) + 1)]

    def test_real_eot_only_pre_finalize(self):
        eff = self.effective(); eot = eff['eot_token_id']
        result = adapter.termination(self.trace([50, eot]), eff)
        self.assertEqual(result['stop_reason'], 'eot')
        self.assertFalse(result['truncated'])
        self.assertEqual(result['generated_token_ids'], [50, eot])

    def test_no_appended_finalization_eot_forgery(self):
        eff = self.effective()
        trace = self.trace([50], completed_last=False)
        with self.assertRaises(ValueError): adapter.termination(trace, eff)
        trace[0]['completed'] = True
        with self.assertRaises(ValueError): adapter.termination(trace, eff)

    def test_cap_is_incomplete_even_if_public_decode_returns(self):
        eff = self.effective()
        trace = self.trace([50] * eff['effective_sample_len'], completed_last=False)
        result = adapter.termination(trace, eff)
        self.assertEqual(result['stop_reason'], 'sample_limit')
        self.assertTrue(result['truncated'])
        self.assertFalse(result['completed'])

    def test_eot_last_allowed_step_is_real(self):
        eff = self.effective(); eot = eff['eot_token_id']
        trace = self.trace([50] * (eff['effective_sample_len'] - 1) + [eot])
        self.assertTrue(adapter.termination(trace, eff)['completed'])

    def test_trace_mutation_rejected(self):
        eff = self.effective(); eot = eff['eot_token_id']
        for mutation in ('extra', 'missing', 'changed_prefix', 'after_eot'):
            trace = self.trace([50, eot])
            if mutation == 'extra': trace[0]['untrusted'] = True
            if mutation == 'missing': trace[1]['step'] = 3
            if mutation == 'changed_prefix': trace[1]['token_ids'][0] = 55
            if mutation == 'after_eot': trace = self.trace([eot, 50, eot]); trace[0]['completed'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                adapter.termination(trace, eff)

    def test_entry_dormant_without_reviewed_admission(self):
        with self.assertRaises(ValueError): host.admission({})

    def test_container_tail_only_changes_command(self):
        old = host.historical_host()
        with mock.patch.object(old.os, 'getuid', return_value=1001), mock.patch.object(old.os, 'getgid', return_value=1001):
            for stage in ('setup', 'sensevoice'):
                args = host.container_command(old, 'fixed', 'image@sha256:' + 'a' * 64,
                    Path('/code'), Path('/runtime'), Path('/output'), stage, Path('/input'))
                self.assertEqual(args[args.index('--memory') + 1], str(12 * 1024**3))
                self.assertEqual(args[args.index('--network') + 1], 'bridge' if stage == 'setup' else 'none')
                self.assertIn('no-new-privileges', args)
                self.assertIn('--read-only', args)
                self.assertTrue(args[-1].endswith('setup_runtime.py' if stage == 'setup' else 'worker.py'))

    def test_whole_runtime_inventory_and_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'a').write_text('safe')
            expected = {'a': identity(root / 'a')}
            verify_inventory(root, expected)
            (root / 'b').write_text('extra')
            with self.assertRaises(ValueError): verify_inventory(root, expected)
            (root / 'b').unlink(); (root / 'a').write_text('changed')
            with self.assertRaises(ValueError): verify_inventory(root, expected)

    def test_input_over_30s_rejected_before_frontend(self):
        # Fake metadata only: no new audio is synthesized or rendered for this test.
        class Header:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def getnframes(self): return 480001
            def getframerate(self): return 16000
            def getnchannels(self): return 1
            def getsampwidth(self): return 2
            def getcomptype(self): return 'NONE'
            def readframes(self, count): raise AssertionError('must reject before reading audio')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fake-header'; path.write_bytes(b'metadata fixture, not audio')
            with mock.patch.object(common.wave, 'open', return_value=Header()):
                with self.assertRaises(ValueError): wav_bytes(path, file_hash(path))

    def test_no_transcribe_or_pad_trim_or_model_alternative(self):
        source = (HERE / 'adapter.py').read_text(); tree = ast.parse(source)
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('transcribe', calls)
        self.assertNotIn('pad_or_trim', calls)
        self.assertEqual(calls.count('load_model'), 1)
        self.assertNotIn('base.pt', source)
        self.assertNotIn('intended_text', source)

    def test_native_finalize_three_dimensional_shape(self):
        class Tokens:
            def __init__(self, shape): self.shape = shape
            def __getitem__(self, index):
                if index != (0, 0): raise AssertionError('must index audio and group')
                return self
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [103, 107, 101, 108, 50, 100]
        self.assertEqual(adapter.flatten_pre_finalize(Tokens((1, 1, 6))), [103, 107, 101, 108, 50, 100])
        for shape in ((1, 6), (2, 1, 6), (1, 2, 6), (6,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                adapter.flatten_pre_finalize(Tokens(shape))

    def test_first_public_hosted_branch_and_source_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); here = root / 'research/source-screen32-whisper-v1'; here.mkdir(parents=True)
            freeze = read_json(HERE / 'execution-freeze.json')
            for name in freeze['files']:
                destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            shutil.copyfile(HERE / 'execution-freeze.json', here / 'execution-freeze.json')
            release = read_json(HERE / 'execution-release.json'); release['approved'] = True
            write_once(here / 'execution-release.json', release)
            active = root / '.github/workflows/source-screen32-whisper-third-v1.yml'; active.parent.mkdir(parents=True)
            shutil.copyfile(HERE / 'workflow.yml', active)
            env = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + common.BRANCH,
                   'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
                   'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux',
                   'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '123', 'GITHUB_EVENT_PATH': str(root / 'event.json')}
            write_once(root / 'event.json', {'created': True, 'deleted': False, 'after': 'a' * 40,
                       'repository': {'private': False, 'full_name': 'jiying2007/kws-pipeline'}})
            with mock.patch.object(host, 'ROOT', root), mock.patch.object(host, 'HERE', here):
                self.assertEqual(host.admission(env), freeze)
                for key, value in (('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_NUMBER', '2'),
                                   ('RUNNER_ENVIRONMENT', 'self-hosted'), ('GITHUB_EVENT_NAME', 'workflow_dispatch')):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        host.admission({**env, key: value})
                with (here / 'adapter.py').open('a') as stream: stream.write('\n# drift\n')
                with self.assertRaises(ValueError): host.admission(env)

    def test_source_and_runtime_pin_completeness(self):
        source = read_json(HERE / 'source-lock.json')
        self.assertEqual(source['source_revision'], common.SOURCE_REVISION)
        names = {r['path'] for r in source['files']}
        for name in ('whisper/decoding.py', 'whisper/audio.py', 'whisper/model.py', 'whisper/timing.py',
                     'whisper/assets/multilingual.tiktoken', 'whisper/assets/mel_filters.npz', 'LICENSE'):
            self.assertIn(name, names)
        wheels = read_json(HERE / 'runtime-lock.json')['files']
        self.assertIn('more-itertools', {r['name'] for r in wheels})
        self.assertNotIn('triton', {r['name'] for r in wheels})
        self.assertEqual(next(r for r in wheels if r['name'] == 'torch')['version'], '2.12.1+cpu')
        from setup_runtime import HOSTS
        from urllib.parse import urlsplit
        self.assertTrue(all(urlsplit(r['url']).hostname in HOSTS for r in wheels + source['files']))

    def test_finalizer_preserves_unattempted_and_started_denominator(self):
        old = host.historical_host()
        import sense_continuation
        lock = sense_continuation.load_lock(old)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); inputs = root / 'input'; output = root / 'output'; output.mkdir()
            job = sense_continuation.prepare_input(inputs, lock, old)
            rows = host.terminal_rows(inputs, output)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r['attempts'] == 0 and r['status'] == 'not_run' for r in rows))
            first = output / job['clips'][0]['audio_id']; first.mkdir()
            # Even a kill during the started marker write conservatively consumes the call.
            (first / 'started.json').write_bytes(b'{')
            rows = host.terminal_rows(inputs, output)
            self.assertEqual(rows[0]['status'], 'failed_no_retry')
            self.assertEqual(rows[0]['attempts'], 1)
            self.assertTrue(all(r['status'] == 'not_run' for r in rows[1:]))
            second = output / job['clips'][1]['audio_id']; second.mkdir()
            (second / 'started.json').write_bytes(b'{}')
            rows = host.terminal_rows(inputs, output)
            self.assertEqual(len(rows), 16)
            self.assertIn('attempt_after_failed_or_untouched_clip', rows[1]['validation_errors'])
            (first / 'raw-output.json').write_bytes(b'{')
            (output / 'native-config.json').write_bytes(b'{')
            rows = host.terminal_rows(inputs, output)
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows[0]['status'], 'failed_no_retry')
            self.assertTrue(rows[0]['validation_errors'])

    def test_public_artifact_never_copies_runtime_or_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('runtime', 'input', 'whisper'):
                (root / name).mkdir()
            (root / 'runtime/small.pt').write_bytes(b'not a real checkpoint')
            (root / 'input/private.txt').write_text('test fixture only')
            write_once(root / 'host-receipt.json', {'status': 'stopped_no_retry'})
            write_once(root / 'whisper/worker-failure.json', {'status': 'failed_no_retry'})
            host.collect_public(root)
            names = read_json(root / 'artifact/artifact-freeze.json')['files']
            self.assertEqual(set(names), {'host-receipt.json', 'whisper/worker-failure.json'})

    def test_workflow_is_nonactive_template(self):
        workflow = (HERE / 'workflow.yml').read_text()
        self.assertIn('timeout-minutes: 45', workflow)
        self.assertIn('runs-on: ubuntu-24.04', workflow)
        self.assertNotIn('workflow_dispatch:', workflow)
        self.assertIn('github.event.created == true', workflow)
        self.assertIn('github.run_attempt == 1', workflow)
        self.assertIn('github.event.repository.private == false', workflow)
        self.assertFalse(read_json(HERE / 'execution-release.json')['approved'])


if __name__ == '__main__':
    unittest.main()
