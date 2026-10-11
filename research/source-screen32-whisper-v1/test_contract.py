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


class SetupDiagnosticTests(unittest.TestCase):
    """Injected byte fixtures and exceptions only; no network/install/model calls."""
    def setUp(self):
        import setup_runtime
        self.setup = setup_runtime
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.output = self.root / 'output'; self.output.mkdir()
        self.diag = setup_runtime.SetupDiagnostics(self.output)
        self.diag.enter('source_download')

    def row(self, body=b'fixture', **overrides):
        import hashlib
        return {'url': 'https://raw.githubusercontent.com/official/pinned/fixture',
                'sha256': hashlib.sha256(body).hexdigest(), 'size_bytes': len(body), **overrides}

    def download(self, row, response=None, error=None, index=1):
        import io
        self.diag.begin_asset('source', index, row)
        downloader = self.setup.Downloader(self.diag)
        downloader.opener = mock.Mock()
        downloader.opener.open.return_value = io.BytesIO(b'fixture') if response is None else response
        if error is not None: downloader.opener.open.side_effect = error
        return downloader, self.root / ('asset-' + str(index))

    def assert_safe(self, sentinel='DO_NOT_DISCLOSE'):
        raw = b''.join(p.read_bytes() for p in self.output.iterdir() if p.is_file())
        self.assertNotIn(sentinel.encode(), raw)
        return read_json(self.output / 'setup-failure.json')

    def test_download_success_has_observed_bytes_and_verified_identity(self):
        row = self.row(); downloader, target = self.download(row)
        self.assertEqual(downloader.get(row, target), {'bytes': 7, 'sha256': row['sha256']})
        progress = read_json(self.output / 'setup-progress.json')
        self.assertEqual(progress['response_body_bytes_observed'], 7)
        self.assertEqual(progress['verified_assets'], 1)
        self.assertEqual(progress['asset']['observed_sha256'], row['sha256'])
        self.assertEqual(progress['asset']['status'], 'verified')
        self.assertFalse(progress['network_transfer_total_known'])
        self.assertEqual(downloader.opener.open.call_count, 1)

    def test_hash_failure_retains_actual_hash_without_promoting_verification(self):
        row = self.row(sha256='0' * 64); downloader, target = self.download(row)
        with self.assertRaises(self.setup.SetupFault) as caught: downloader.get(row, target)
        self.diag.fail(caught.exception); failure = self.assert_safe()
        self.assertEqual(failure['failure']['reason_code'], 'sha256_mismatch')
        self.assertEqual(failure['response_body_bytes_observed'], 7)
        self.assertEqual(failure['verified_assets'], 0)
        self.assertNotEqual(failure['asset']['observed_sha256'], row['sha256'])
        self.assertEqual(downloader.opener.open.call_count, 1)

    def test_length_and_byte_cap_fail_closed(self):
        for code, row, limit in (('declared_length', self.row(size_bytes=8), LIMITS['download_bytes']),
                                 ('byte_cap', self.row(), 3)):
            with self.subTest(code=code):
                # Distinct fixture destination and fresh ledger per independent test case.
                out = self.root / code; out.mkdir()
                self.diag = self.setup.SetupDiagnostics(out); self.output = out
                self.diag.enter('source_download')
                downloader, target = self.download(row)
                target = target.with_name(code + '.fixture')
                with mock.patch.dict(self.setup.LIMITS, download_bytes=limit):
                    with self.assertRaises(self.setup.SetupFault) as caught: downloader.get(row, target)
                self.diag.fail(caught.exception)
                self.assertEqual(self.assert_safe()['failure']['reason_code'], code)
                self.assertEqual(self.diag.state['verified_assets'], 0)

    def test_partial_read_timeout_is_durable_and_not_retried(self):
        class Partial:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, maximum):
                if not hasattr(self, 'read_once'):
                    self.read_once = True; return b'part'
                raise TimeoutError('DO_NOT_DISCLOSE credential in transport context')
        row = self.row(size_bytes=10); downloader, target = self.download(row, Partial())
        with self.assertRaises(TimeoutError) as caught: downloader.get(row, target)
        # Progress was durable even before the exception handler ran.
        self.assertEqual(read_json(self.output / 'setup-progress.json')['response_body_bytes_observed'], 4)
        self.diag.fail(caught.exception); failure = self.assert_safe()
        self.assertEqual(failure['failure']['reason_code'], 'timeout')
        self.assertEqual(failure['asset']['response_body_bytes_observed'], 4)
        self.assertEqual(failure['retry_count'], 0)
        self.assertEqual(downloader.opener.open.call_count, 1)

    def test_http_error_exposes_only_bounded_status(self):
        import urllib.error
        error = urllib.error.HTTPError('https://DO_NOT_DISCLOSE/path?token=DO_NOT_DISCLOSE', 403,
                                       'DO_NOT_DISCLOSE', {'Cookie': 'DO_NOT_DISCLOSE'}, None)
        row = self.row(); downloader, target = self.download(row, error=error)
        with self.assertRaises(urllib.error.HTTPError) as caught: downloader.get(row, target)
        self.diag.fail(caught.exception); failure = self.assert_safe()
        self.assertEqual(failure['failure'], {'reason_code': 'http_error', 'http_status': 403})
        self.assertEqual(failure['response_body_bytes_observed'], 0)
        self.assertEqual(failure['asset']['http_status'], 403)
        self.assertEqual(failure['asset']['expected_sha256'], row['sha256'])
        self.assertNotIn('url', failure['asset'])

    def test_exception_categories_never_copy_messages_commands_or_headers(self):
        import ssl
        import subprocess
        import urllib.error
        cases = [(urllib.error.URLError('DO_NOT_DISCLOSE'), 'url_error'),
                 (urllib.error.URLError(ssl.SSLError('DO_NOT_DISCLOSE')), 'tls_error'),
                 (subprocess.CalledProcessError(1, ['DO_NOT_DISCLOSE'], 'DO_NOT_DISCLOSE', 'DO_NOT_DISCLOSE'), 'subprocess_exit'),
                 (subprocess.TimeoutExpired(['DO_NOT_DISCLOSE'], 1, 'DO_NOT_DISCLOSE'), 'timeout'),
                 (PermissionError(13, 'DO_NOT_DISCLOSE', '/DO_NOT_DISCLOSE'), 'permission_denied'),
                 (ValueError('DO_NOT_DISCLOSE'), 'validation_rejected'),
                 (RuntimeError('DO_NOT_DISCLOSE'), 'runtime_error'),
                 (Exception('DO_NOT_DISCLOSE'), 'unexpected_exception')]
        for error, expected in cases:
            with self.subTest(expected=expected):
                value = self.setup.public_reason(error)
                self.assertEqual(value['reason_code'], expected)
                self.assertNotIn('DO_NOT_DISCLOSE', json.dumps(value))

    def test_unapproved_redirect_and_url_still_rejected(self):
        import urllib.request
        handler = self.setup.FixedRedirect()
        request = urllib.request.Request('https://raw.githubusercontent.com/official/pinned')
        with self.assertRaises(self.setup.SetupFault) as caught:
            handler.redirect_request(request, None, 302, 'DO_NOT_DISCLOSE', {},
                                     'https://DO_NOT_DISCLOSE.example/private')
        self.assertEqual(caught.exception.code, 'unapproved_redirect')
        row = self.row(url='https://DO_NOT_DISCLOSE.example/private')
        downloader, target = self.download(row)
        with self.assertRaises(self.setup.SetupFault) as caught: downloader.get(row, target)
        self.diag.fail(caught.exception)
        self.assertEqual(self.assert_safe()['failure']['reason_code'], 'unapproved_url')
        downloader.opener.open.assert_not_called()
        self.assertEqual(self.setup.HOSTS, {'raw.githubusercontent.com', 'files.pythonhosted.org',
            'download.pytorch.org', 'download-r2.pytorch.org', 'openaipublic.azureedge.net'})

    def test_every_phase_failure_emits_fixed_phase_without_running_setup(self):
        import contextlib
        import io
        for index, stage in enumerate(sorted(self.setup.STAGES)):
            with self.subTest(stage=stage):
                output = self.root / str(index); output.mkdir()
                diag = self.setup.SetupDiagnostics(output)
                def injected(diagnostic):
                    diagnostic.enter(stage)
                    raise RuntimeError('DO_NOT_DISCLOSE')
                with mock.patch.object(self.setup, 'SetupDiagnostics', return_value=diag), \
                     mock.patch.object(self.setup, 'execute_setup', side_effect=injected), \
                     mock.patch.object(self.setup.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(self.setup.main(), 1)
                value = read_json(output / 'setup-failure.json')
                self.assertEqual(value['stage'], stage)
                self.assertEqual(value['model_calls'], 0)
                self.assertEqual(value['status'], 'failed_no_retry')
                self.assertNotIn('DO_NOT_DISCLOSE', stream.getvalue())
                self.assertNotIn('DO_NOT_DISCLOSE', json.dumps(value))

    def test_diagnostic_write_failure_does_not_emit_primary_text(self):
        import contextlib
        import io
        with mock.patch.object(self.setup, 'SetupDiagnostics', return_value=self.diag), \
             mock.patch.object(self.setup, 'execute_setup', side_effect=self.setup.SetupFault('sha256_mismatch')), \
             mock.patch.object(self.diag, 'fail', side_effect=OSError('DO_NOT_DISCLOSE')), \
             mock.patch.object(self.setup.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.setup.main(), 1)
        self.assertIn('sha256_mismatch', output.getvalue())
        self.assertIn("'diagnostic_saved': False", output.getvalue())
        self.assertNotIn('DO_NOT_DISCLOSE', output.getvalue())

    def test_constructor_signal_and_first_save_failures_are_sanitized(self):
        import contextlib
        import io
        cases = ('constructor', 'signal', 'first_save', 'atomic_save')
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                output = self.root / ('startup-' + str(index)); output.mkdir()
                diagnostic = self.setup.SetupDiagnostics(output)
                if case == 'atomic_save': diagnostic.enter('entry')
                def injected(diag): diag.enter('entry')
                with contextlib.ExitStack() as stack:
                    constructor = stack.enter_context(mock.patch.object(self.setup, 'SetupDiagnostics', return_value=diagnostic))
                    signals = stack.enter_context(mock.patch.object(self.setup.signal, 'signal'))
                    execute = stack.enter_context(mock.patch.object(self.setup, 'execute_setup', side_effect=injected))
                    stream = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if case == 'constructor': constructor.side_effect = RuntimeError('DO_NOT_DISCLOSE')
                    elif case == 'signal': signals.side_effect = ValueError('DO_NOT_DISCLOSE')
                    elif case == 'first_save': stack.enter_context(mock.patch.object(self.setup, 'write_once', side_effect=OSError('DO_NOT_DISCLOSE')))
                    else: stack.enter_context(mock.patch.object(self.setup, 'atomic', side_effect=OSError('DO_NOT_DISCLOSE')))
                    self.assertEqual(self.setup.main(), 1)
                    self.assertNotIn('DO_NOT_DISCLOSE', stream.getvalue())
                    if case in ('constructor', 'signal'): execute.assert_not_called()
                self.assertFalse((output / 'setup-receipt.json').exists())

    def test_lock_identity_http_headers_and_completed_stages_are_fixed_metadata(self):
        self.diag.bind_locks({'source-lock.json': 'a' * 64, 'runtime-lock.json': 'b' * 64})
        self.diag.enter('model_download')
        self.diag.enter('wheel_download')
        self.assertIn('source_download', self.diag.state['completed_stages'])
        self.assertIn('model_download', self.diag.state['completed_stages'])
        self.assertNotIn('wheel_download', self.diag.state['completed_stages'])
        row = self.row(); downloader, target = self.download(row)
        self.diag.http_response(200)
        self.assertEqual(read_json(self.output / 'setup-progress.json')['asset']['http_status'], 200)
        self.assertEqual(self.diag.state['lock_sha256']['runtime-lock.json'], 'b' * 64)
        with self.assertRaises(self.setup.SetupFault): self.diag.bind_locks({'DO_NOT_DISCLOSE': 'a' * 64})

    def test_read_connection_reset_and_incomplete_read_are_not_filesystem_claims(self):
        import http.client
        for index, (error, reason) in enumerate(((ConnectionResetError('DO_NOT_DISCLOSE'), 'connection_error'),
                (http.client.IncompleteRead(b'DO_NOT_DISCLOSE'), 'incomplete_read')), 1):
            with self.subTest(reason=reason):
                output = self.root / ('transport-' + str(index)); output.mkdir()
                self.output = output; self.diag = self.setup.SetupDiagnostics(output)
                self.diag.enter('source_download')
                class Response:
                    status = 200
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def read(self, maximum): raise error
                row = self.row(); downloader, target = self.download(row, response=Response())
                with self.assertRaises(type(error)) as caught: downloader.get(row, target.with_name(reason))
                self.diag.fail(caught.exception); failure = self.assert_safe()
                self.assertEqual(failure['failure']['reason_code'], reason)
                self.assertEqual(failure['asset']['http_status'], 200)
                self.assertEqual(failure['response_body_bytes_observed'], 0)
                self.assertFalse(failure['network_transfer_total_known'])

    def test_existing_progress_cannot_be_overwritten_as_fresh_setup(self):
        original = (self.output / 'setup-progress.json').read_bytes()
        other = self.setup.SetupDiagnostics(self.output)
        with self.assertRaises(FileExistsError): other.enter('entry')
        self.assertEqual((self.output / 'setup-progress.json').read_bytes(), original)

    def test_first_failed_run_public_evidence_is_immutable_and_zero_attempt(self):
        directory = HERE / 'evidence/run-38105471896'
        provenance = read_json(directory / 'provenance.json')
        self.assertEqual(provenance['zip_sha256'], '04b6b23c85f25e0b29520972654c2d59f18f22506cf1b77c9fd1056bd357b0ff')
        self.assertEqual(provenance['zip_bytes'], 11518)
        self.assertEqual(len(provenance['files']), 7)
        for name, expected in provenance['files'].items():
            self.assertEqual(identity(directory / name), expected)
        receipt = read_json(directory / 'host-receipt.json')
        self.assertEqual(receipt['source_freeze_sha256'], 'fdfb8a9040a10ec380ac7a658662d2d2935296301ee4269ac0ce79309cc48c2e')
        self.assertEqual(receipt['stages']['whisper'], 'not_run')
        rows = read_json(directory / 'terminal-outcomes.json')
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r['attempts'] == 0 and r['status'] == 'not_run' for r in rows))
        freeze = read_json(HERE / 'execution-freeze.json')
        self.assertFalse(any('source-screen32-whisper-v1/evidence/' in p for p in freeze['files']))
        self.assertFalse(any('/evidence/' in p for p in freeze['blind_code']))


if __name__ == '__main__':
    unittest.main()
