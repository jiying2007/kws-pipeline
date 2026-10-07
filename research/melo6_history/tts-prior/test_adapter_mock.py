"""Pure fake-session tests. Never install/import ORT, load a model or synthesize."""
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

# Every test fails immediately if any inference runtime import is attempted.
original_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'onnxruntime', 'onnx', 'torch', 'transformers', 'melo'}:
        raise AssertionError('INFERENCE_RUNTIME_IMPORT_FORBIDDEN_IN_MOCK_TESTS')
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
import numpy as np
import run_melo6 as runner
import pack_blind

FIXTURES = json.loads((runner.ROOT / 'frozen-six-inputs.json').read_bytes())['fixtures']
PLAN_SHA = 'a' * 64


class FakeSession:
    def __init__(self, generation, fail_at=None, bad_output=None):
        self.calls = 0
        self.generation = generation
        self.fail_at = fail_at
        self.bad_output = bad_output

    def run(self, outputs, arrays):
        self.calls += 1
        source = f'melo6-{self.calls:03d}'
        claim = self.generation / (source + '.started.json')
        assert claim.is_file(), 'write-ahead claim must predate every fake call'
        record = json.loads(claim.read_bytes())
        assert record['inputs'] == runner.input_boundary(arrays)
        assert outputs == ['y']
        assert arrays['sid'].tolist() == [1]
        if self.calls == self.fail_at:
            raise RuntimeError('SECRET ERROR TEXT MUST NOT BE RETAINED')
        if self.bad_output is not None:
            return self.bad_output
        # Six distinct tiny synthetic vectors. These are never TTS outputs.
        return [np.array([[[0., .01 * self.calls, -.01 * self.calls,
                            .02 * self.calls] * 20]], dtype='float32')]


class AdapterMockTests(unittest.TestCase):
    def test_default_is_disabled_without_runtime_or_output(self):
        with patch('sys.stdout', new_callable=io.StringIO) as stream:
            self.assertEqual(runner.main([]), 0)
        self.assertEqual(json.loads(stream.getvalue())['status'], 'disabled')
        self.assertNotIn('onnxruntime', sys.modules)

    def test_child_cannot_start_without_supervised_arming(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'SUPERVISED_CHILD_REQUIRED'):
                runner.armed_child(Path('/not-read'), Path('/not-written'), Path('/not-read'), PLAN_SHA)

    def test_six_exact_calls_write_ahead_and_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'generation'
            fake = FakeSession(out)
            result = runner.run_six(fake, FIXTURES, out, PLAN_SHA, np)
            self.assertEqual(fake.calls, 6)
            self.assertEqual(result['status'], 'six_candidates_generated')
            self.assertEqual(result['attempts_consumed'], 6)
            self.assertFalse(result['deterministic_waveform_claim'])
            self.assertEqual(len(list(out.glob('*.native-f32.wav'))), 6)
            self.assertEqual(len(list(out.glob('*.pcm16.wav'))), 6)
            with self.assertRaisesRegex(ValueError, 'NO_RESUME'):
                runner.run_six(fake, FIXTURES, out, PLAN_SHA, np)
            self.assertEqual(fake.calls, 6)

    def test_failed_attempt_consumed_and_remaining_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'generation'
            fake = FakeSession(out, fail_at=3)
            with self.assertRaises(RuntimeError):
                runner.run_six(fake, FIXTURES, out, PLAN_SHA, np)
            receipt = json.loads((out / 'generation-receipt.json').read_bytes())
            self.assertEqual(fake.calls, 3)
            self.assertEqual(receipt['attempts_consumed'], 3)
            self.assertEqual([r['status'] for r in receipt['generation_rows']],
                             ['generated_candidate'] * 2 + ['failed_no_retry'] + ['not_run'] * 3)
            self.assertNotIn('SECRET', (out / 'generation-receipt.json').read_text())
            self.assertFalse((out / 'melo6-004.started.json').exists())

    def test_wrong_order_refused_before_any_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'generation'
            fake = FakeSession(out)
            with self.assertRaisesRegex(ValueError, 'EXACT_SIX_FIXTURES'):
                runner.run_six(fake, list(reversed(FIXTURES)), out, PLAN_SHA, np)
            self.assertEqual(fake.calls, 0)

    def test_invalid_outputs_stop_without_native_save(self):
        bad_outputs = [[np.array([[[float('nan')]]], dtype='float32')],
                       [np.zeros((1, 1, 1), dtype='float64')],
                       [np.zeros((1, 2, 1), dtype='float32')],
                       [np.zeros((1, 1, 0), dtype='float32')], []]
        for bad in bad_outputs:
            with self.subTest(shape=[getattr(a, 'shape', None) for a in bad]):
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / 'generation'
                    fake = FakeSession(out, bad_output=bad)
                    with self.assertRaises(ValueError):
                        runner.run_six(fake, FIXTURES, out, PLAN_SHA, np)
                    self.assertEqual(fake.calls, 1)
                    self.assertFalse(list(out.glob('*.wav')))

    def test_external_watchdog_kills_blocked_mock_child(self):
        # A sleeping subprocess stands in for a native call that cannot deliver
        # a Python signal callback. No runtime or model is involved.
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
            try:
                runner.write_json(runtime / 'child-identity.json', {'pid': child.pid, 'pgid': child.pid})
                runner.write_json(runtime / 'active-call.json',
                                  {'pid': child.pid, 'source_id': 'melo6-001',
                                   'deadline_monotonic': time.monotonic() + .1})
                report = {}
                stopped = threading.Event()
                thread = threading.Thread(target=runner.watchdog, args=(runtime, stopped, report))
                thread.start()
                self.assertEqual(child.wait(timeout=3), -signal.SIGKILL)
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(report['reason'], 'native_call_wall_deadline')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()

    def test_finalize_consumes_claim_after_abrupt_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            out = runtime / 'generation'
            out.mkdir()
            runner.write_json(out / 'melo6-001.started.json', {'attempt_consumed': True})
            receipt = runner.finalize(runtime, PLAN_SHA, {'returncode': -9},
                                      {'reason': 'native_call_wall_deadline'})
            self.assertEqual(receipt['generation_rows'][0]['status'], 'failed_no_retry')
            self.assertTrue(receipt['generation_rows'][0]['attempt_consumed'])
            self.assertEqual(receipt['attempts_consumed'], 1)
            self.assertEqual([r['status'] for r in receipt['generation_rows'][1:]], ['not_run'] * 5)

    def test_finalize_pre_call_failure_records_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = runner.finalize(Path(tmp), PLAN_SHA, {'returncode': 1}, {})
            self.assertEqual(receipt['attempts_consumed'], 0)
            self.assertEqual([r['status'] for r in receipt['generation_rows']], ['not_run'] * 6)

    def test_signature_rejects_same_rank_shape_drift(self):
        metadata = json.loads((runner.ROOT / 'onnx-static-metadata.json').read_bytes())
        def nodes(rows):
            return [SimpleNamespace(name=r['name'], type='tensor(int64)' if r['element_type'] == 'INT64'
                                    else 'tensor(float)', shape=[d.get('symbol', d.get('value'))
                                                                for d in r['shape']]) for r in rows]
        inputs = nodes(metadata['graph']['inputs'])
        outputs = nodes(metadata['graph']['outputs'])
        session = SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'],
                                  get_inputs=lambda: inputs, get_outputs=lambda: outputs,
                                  get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata['metadata']))
        runner.validate_loaded_session(session, metadata)
        inputs[3].shape = [2]
        with self.assertRaisesRegex(ValueError, 'LOADED_INPUT_SIGNATURE'):
            runner.validate_loaded_session(session, metadata)

    def make_complete(self, runtime):
        out = runtime / 'generation'
        runner.run_six(FakeSession(out), FIXTURES, out, PLAN_SHA, np)
        runner.finalize(runtime, PLAN_SHA, {'returncode': 0}, {})

    def test_blind_schema_content_addressing_no_intent_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            self.make_complete(runtime)
            with patch.object(pack_blind, 'validate_plan', return_value=({}, FIXTURES, {})):
                result = pack_blind.pack(runtime, runtime / 'blind', Path('unused'), PLAN_SHA)
            self.assertEqual(result['clip_count'], 6)
            self.assertFalse(result['asr_run_performed'])
            with zipfile.ZipFile(runtime / 'blind' / 'blind-inputs.zip') as archive:
                names = archive.namelist()
                self.assertEqual(len(names), 7)
                job_bytes = archive.read('job.json')
                job = json.loads(job_bytes)
                self.assertEqual(set(job), {'schema', 'clips'})
                self.assertEqual(job['schema'], 'blind-asr-job-v1')
                self.assertEqual([c['audio_id'] for c in job['clips']],
                                 [f'clip-{i:06d}' for i in range(1, 7)])
                for clip in job['clips']:
                    raw = archive.read(clip['audio_path'])
                    self.assertEqual(clip['wav_sha256'], hashlib.sha256(raw).hexdigest())
                    self.assertEqual(clip['audio_path'], 'audio/' + clip['wav_sha256'] + '.wav')
                    pack_blind.check_pcm(raw)
                for word in ['melo', 'qwen', 'speaker', 'intended_text', 'sid', '你好']:
                    self.assertNotIn(word, job_bytes.decode())
            freeze = json.loads((runtime / 'blind' / 'blind-input-freeze.json').read_bytes())
            self.assertEqual(freeze['schema'], 'qwen6-blind-input-freeze-v1')

    def test_blind_refuses_missing_or_mutated_wavs_and_no_partial_pack(self):
        for operation in ['missing', 'mutated']:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                runtime = Path(tmp)
                self.make_complete(runtime)
                path = runtime / 'generation' / 'melo6-001.pcm16.wav'
                if operation == 'missing':
                    path.unlink()
                else:
                    path.write_bytes(path.read_bytes() + b'x')
                with patch.object(pack_blind, 'validate_plan', return_value=({}, FIXTURES, {})):
                    with self.assertRaises(ValueError):
                        pack_blind.pack(runtime, runtime / 'blind', Path('unused'), PLAN_SHA)
                self.assertFalse((runtime / 'blind').exists())

    def test_cap_rejects_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'too-large.bin'
            with self.assertRaisesRegex(ValueError, 'INDIVIDUAL_FILE_CAP'):
                runner.bounded_write(path, b'x' * (2 * runner.MiB + 1))
            self.assertFalse(path.exists())

    def test_diagnostics_retain_own_location_code_without_raw_text(self):
        try:
            runner.require(False, 'MODEL_BYTES_MISMATCH')
        except ValueError as error:
            diagnostic = runner.safe_diagnostic(error)
        self.assertEqual(diagnostic['validation_code'], 'MODEL_BYTES_MISMATCH')
        self.assertEqual(diagnostic['source_frames'][0]['module'], 'run_melo6.py')
        self.assertEqual(diagnostic['source_frames'][0]['function'], 'require')
        self.assertGreater(diagnostic['source_frames'][0]['line'], 0)
        self.assertNotIn('/workspace', json.dumps(diagnostic))
        secret = runner.safe_diagnostic(RuntimeError('/secret/path/token'))
        self.assertNotIn('secret', json.dumps(secret))

    def test_unreadable_proc_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            (proc / '123').mkdir()
            (proc / '123' / 'stat').write_text('unparseable mock stat')
            with self.assertRaisesRegex(RuntimeError, 'PROC_OBSERVATION_UNKNOWN'):
                runner.observed_group(123, proc)

    def test_unknown_or_empty_rss_samples_never_publish_zero(self):
        for observations, watcher in [
                ({'valid_member_samples': 0, 'empty_member_samples': 1, 'unknown_samples': 0}, {}),
                ({'valid_member_samples': 1, 'empty_member_samples': 0, 'unknown_samples': 0},
                 {'reason': 'proc_observation_unknown'})]:
            receipt = runner.annotate_rss({'max_sampled': {'rss_bytes': 0}}, observations, watcher)
            self.assertIsNone(receipt['max_sampled']['rss_bytes'])
            self.assertEqual(receipt['rss_observation'], 'unknown_not_zero')
        good = runner.annotate_rss({'max_sampled': {'rss_bytes': 8192}},
                                  {'valid_member_samples': 1, 'empty_member_samples': 0,
                                   'unknown_samples': 0}, {})
        self.assertEqual(good['max_sampled']['rss_bytes'], 8192)

    def test_frozen_plan_wrong_hash_refused_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'APPROVED_PLAN_HASH_MISMATCH'):
                runner.validate_plan(path, '0' * 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
