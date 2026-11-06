#!/usr/bin/env python3
"""Offline continuation contracts using protocol fixtures, never an ML runtime."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import active_tts_expand as expand
import research_queue as controller


def cell(index):
    return {'cell_id': 'bulk-' + format(index, '024x'), 'reference_id': 'reference-1',
            'source_family_id': 'source-1', 'seed': index, 'prompt': 'synthetic fixture',
            'intended_kind': 'ordinary-negative'}


REF = {'reference_id': 'reference-1', 'source_family_id': 'source-1',
       'wav_sha256': 'a' * 64, 'pcm_sha256': 'b' * 64}
BINDING = {'plan_canonical_sha256': 'c' * 64, 'plan_file_sha256': 'd' * 64,
           'generation': dict(expand.common.GENERATION)}
OWNER = {'stage_sha256': 'e' * 64, 'job_id': 'chunk-001', 'job_recipe_sha256': 'f' * 64}


def raw_fixture(output, row, ref, *, failed=False):
    """Small stdlib FLOAT/PCM WAVs; this fixture does not imitate inference."""
    name = row['cell_id']; native = output / 'native' / name; generated = output / 'generation/cells' / name
    samples = struct.pack('<8f', *([0.125, -0.125] * 4))
    fmt = struct.pack('<HHIIHH', 3, 1, 24000, 96000, 4, 32)
    wav = b'RIFF' + struct.pack('<I', 36 + len(samples)) + b'WAVEfmt ' + struct.pack('<I', 16) + fmt
    wav += b'data' + struct.pack('<I', len(samples)) + samples
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (8,), }"
    header += b' ' * ((16 - (10 + len(header) + 1) % 16) % 16) + b'\n'
    array = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + samples
    expand.atomic_bytes(native / 'returned-array.npy', array)
    expand.atomic_bytes(native / 'returned-native.wav', wav)
    expand.once(native / 'native.json', {'schema': 'active-tts-native-return-v1',
        'array': expand.common.record(native / 'returned-array.npy'),
        'wav': expand.common.record(native / 'returned-native.wav'), 'dtype': '<f4', 'shape': [8],
        'sample_rate_hz': 24000, 'samples_sha256': hashlib.sha256(samples).hexdigest(),
        'all_finite': True, 'technical_gate_applied': False})
    expand.once(native / 'call.json', {'schema': 'active-tts-generation-call-v1', 'cell_id': name,
        'seed': row['seed'], 'reference_wav_sha256': ref['wav_sha256'],
        'reference_pcm_sha256': ref['pcm_sha256'], 'generation': BINDING['generation'], 'attempt': 1})
    expand.once(native / 'returned.json', {'generation_seconds': 0.0, 'fixture': True})
    expand.once(generated / 'started.json', expand.expected_start(row, ref, BINDING['plan_canonical_sha256']))
    if failed:
        status = 'FAILED_NO_RETRY'; native_path = 'native/' + name + '/returned-native.wav'
        expand.once(generated / 'failed.json', {'schema': 'kws-bulk-clone-failure-v1', 'cell_id': name,
            'status': status, 'error_type': 'ValueError', 'training_admitted': False})
    else:
        status = 'GENERATED_PENDING_ASR'; native_path = 'generation/cells/' + name + '/native-f32.wav'
        expand.atomic_bytes(generated / 'native-f32.wav', wav)
        pcm_buffer = io.BytesIO()
        with wave.open(pcm_buffer, 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes(b'\x00\x10' * 5)
        expand.atomic_bytes(generated / 'audio-16k.wav', pcm_buffer.getvalue())
        expand.once(generated / 'completed.json', {'schema': 'kws-bulk-clone-complete-v1', 'cell_id': name,
            'status': status, 'technical': {'fixture': True},
            'audio': {'native_sha256': expand.common.sha_file(generated / 'native-f32.wav'),
                      'wav_sha256': expand.common.sha_file(generated / 'audio-16k.wav')},
            'source_family_id': row['source_family_id'], 'intended_kind': row['intended_kind'],
            'actual_text': None, 'training_admitted': False, 'qualification_allowed': False})
    files = {p.relative_to(output).as_posix(): expand.common.record(p)
             for folder in (native, generated) for p in folder.rglob('*') if p.is_file()}
    return {'status': status, 'technical_failure_reason': 'fixture quality failure' if failed else None,
            'output_root': str(output), 'generation_root': str(output / 'generation'),
            'files': files, 'native_wav': native_path}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = self.root / 'ledger'
        self.output = self.root / 'outputs'
        self.plan = {'cells': [cell(i) for i in range(256)], 'references': [REF]}
        self.addCleanup(self.temp.cleanup)

    def consume(self, index=8, *, failed=False, callback=None):
        return expand.consume_cell(self.ledger, cell(index), REF, BINDING, OWNER,
            callback or (lambda row, ref: raw_fixture(self.output, row, ref, failed=failed)))

    def test_first_eight_reused_then_two_28_chunks_and_remaining_tail(self):
        complete = {cell(i)['cell_id'] for i in range(8)}
        chunks = expand.partition_cells(self.plan, complete, 64)
        self.assertEqual(list(map(len, chunks)), [28, 28])
        self.assertEqual([name for group in chunks for name in group], [cell(i)['cell_id'] for i in range(8, 64)])
        for done, target, sizes in [(64, 120, [28, 28]), (232, 256, [24]), (256, 256, [])]:
            self.assertEqual(list(map(len, expand.partition_cells(self.plan, {cell(i)['cell_id'] for i in range(done)}, target))), sizes)
        for args in [(complete, 65, 28), (complete, 64, 29), (complete, 64, 10), (complete, 257, 28)]:
            with self.assertRaises(ValueError):
                expand.partition_cells(self.plan, *args)
        duplicate = {'cells': self.plan['cells'][:-1] + [cell(0)]}
        with self.assertRaisesRegex(ValueError, 'UNIQUE_256'):
            expand.partition_cells(duplicate, complete, 64)

    def test_terminal_reuse_never_invokes_callback_again(self):
        first, called = self.consume()
        self.assertTrue(called)
        before = {p: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.rglob('*') if p.is_file()}
        second, called = self.consume(callback=lambda *_: self.fail('cached cell called model'))
        self.assertFalse(called); self.assertEqual(first, second)
        self.assertEqual(before, {p: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in before})

    def test_claim_precedes_callback_and_unknown_cannot_be_retried(self):
        def crash(row, _):
            self.assertEqual(expand.common.read(self.ledger / 'claims' / (row['cell_id'] + '.json'))['attempt'], 1)
            raise RuntimeError('simulated interruption after durable claim')
        with self.assertRaises(RuntimeError):
            self.consume(callback=crash)
        with self.assertRaisesRegex(ValueError, 'CONSUMED_WITHOUT_TERMINAL'):
            self.consume(callback=lambda *_: self.fail('unknown call retried'))
        self.assertFalse((self.ledger / 'cells' / (cell(8)['cell_id'] + '.json')).exists())

    def test_quality_failure_retains_raw_and_is_terminal(self):
        row, invoked = self.consume(failed=True)
        self.assertTrue(invoked); self.assertEqual(row['status'], 'FAILED_NO_RETRY')
        self.assertTrue((self.output / row['native_wav']).is_file())
        self.assertFalse((self.output / 'generation/cells' / cell(8)['cell_id'] / 'completed.json').exists())
        _, invoked = self.consume(callback=lambda *_: self.fail('quality failure retried'))
        self.assertFalse(invoked)

    def test_native_writer_saves_abnormal_array_without_inventing_wav(self):
        folder = self.root / 'native'; folder.mkdir()
        array = SimpleNamespace(ndim=2, dtype=SimpleNamespace(kind='f', str='<f4'), shape=(2, 4),
                                tobytes=lambda: b'original-returned-array')
        numpy = SimpleNamespace(asarray=lambda value: value,
            save=lambda stream, value, allow_pickle: stream.write(value.tobytes()),
            isfinite=lambda _: SimpleNamespace(all=lambda: True))
        with patch.dict(sys.modules, {'numpy': numpy}):
            result = expand.retain_extension_native(folder, array, 24000)
        self.assertEqual((folder / 'returned-array.npy').read_bytes(), array.tobytes())
        self.assertIsNone(result['wav'])
        self.assertEqual(result['wav_unavailable_reason'], 'NATIVE_ARRAY_NOT_ONE_DIMENSIONAL_FLOAT')
        self.assertFalse((folder / 'returned-native.wav').exists())

    def test_asr_eligibility_keeps_quiet_failure_but_excludes_bad_geometry_or_samples(self):
        value, _ = self.consume(failed=True)
        self.assertTrue(expand.asr_eligibility(value)['eligible'])
        path = self.output / value['native_wav']; original = path.read_bytes()
        long_data = b'\x00\x00\x00\x00' * (24000 * 31)
        long_wav = original[:4] + struct.pack('<I', 36 + len(long_data)) + original[8:40]
        long_wav += struct.pack('<I', len(long_data)) + long_data
        nan_wav = original[:44] + struct.pack('<f', float('nan')) + original[48:]
        for raw, reason in ((long_wav, 'ASR_FRAME_BOUND'), (nan_wav, 'ASR_NONFINITE_SAMPLE')):
            path.write_bytes(raw); value['files'][value['native_wav']] = expand.common.record(path)
            result = expand.asr_eligibility(value)
            self.assertFalse(result['eligible']); self.assertEqual(result['reason'], reason)
        value['native_wav'] = None
        self.assertEqual(expand.asr_eligibility(value), {'eligible': False, 'reason': 'NO_DECODABLE_NATIVE_WAV'})

    def test_failed_array_without_wav_is_a_terminal_not_a_retry(self):
        def array_only(row, ref):
            value = raw_fixture(self.output, row, ref, failed=True)
            native = self.output / 'native' / row['cell_id']
            (native / 'returned-native.wav').unlink()
            raw = expand.common.read(native / 'native.json')
            raw.update(wav=None, wav_unavailable_reason='NATIVE_ARRAY_NOT_ONE_DIMENSIONAL_FLOAT')
            (native / 'native.json').write_bytes(expand.common.canonical(raw) + b'\n')
            value['files'].pop(value['native_wav']); value['native_wav'] = None
            value['files']['native/' + row['cell_id'] + '/native.json'] = expand.common.record(native / 'native.json')
            return value
        value, called = self.consume(callback=array_only)
        self.assertTrue(called); self.assertIsNone(value['native_wav'])
        _, called = self.consume(callback=lambda *_: self.fail('invalid array retried'))
        self.assertFalse(called)

    def test_native_deletion_or_corruption_rejects_cache_without_calls(self):
        for mode in ('missing', 'changed'):
            with self.subTest(mode=mode):
                index = 8 if mode == 'missing' else 9
                self.consume(index)
                path = self.output / 'native' / cell(index)['cell_id'] / 'returned-native.wav'
                path.unlink() if mode == 'missing' else path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    self.consume(index, callback=lambda *_: self.fail('damaged raw retried'))

    def test_origin_cannot_hide_missing_claim_or_import_foreign_cell(self):
        self.consume()
        path = self.ledger / 'cells' / (cell(8)['cell_id'] + '.json')
        value = expand.common.read(path); value['origin'] = 'reused-first-eight'; value['new_generation_calls'] = 0
        path.write_bytes(expand.common.canonical(value) + b'\n')
        with self.assertRaisesRegex(ValueError, 'IMPORTED_CELL_CLAIM_CONTRADICTION'):
            self.consume(callback=lambda *_: self.fail('origin rewrite called'))
        (self.ledger / 'claims' / (cell(8)['cell_id'] + '.json')).unlink()
        expand.once(self.ledger / 'base.json', {'schema': 'active-tts-plan-first-eight-v1',
                                             'base_terminal_sha256': 'a' * 64, 'records': {}})
        with self.assertRaisesRegex(ValueError, 'IMPORTED_CELL_NOT_IN_BASELINE'):
            self.consume(callback=lambda *_: self.fail('foreign imported cell called'))

    def test_cross_stage_race_publishes_only_one_callback(self):
        read_fd, write_fd = os.pipe(); children = []
        log = self.root / 'callback.log'
        for _ in range(2):
            child = os.fork()
            if child == 0:
                os.close(write_fd); os.read(read_fd, 1)
                def invoke(row, ref):
                    with log.open('ab') as stream:
                        stream.write(b'called\n'); stream.flush(); os.fsync(stream.fileno())
                    return raw_fixture(self.output, row, ref)
                try:
                    self.consume(callback=invoke)
                except (FileExistsError, ValueError):
                    pass
                except BaseException:
                    os._exit(2)
                os._exit(0)
            children.append(child)
        os.close(read_fd); os.write(write_fd, b'xx'); os.close(write_fd)
        for child in children:
            self.assertEqual(os.waitpid(child, 0)[1], 0)
        self.assertEqual(log.read_bytes(), b'called\n')
        _, called = self.consume(callback=lambda *_: self.fail('race produced second call'))
        self.assertFalse(called)

    def test_symlink_ledger_lock_and_unsafe_cell_rejected(self):
        self.ledger.mkdir(); outside = self.root / 'outside'; outside.write_bytes(b'unmodified')
        (self.ledger / 'writer.lock').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'PATH_ALIAS'):
            with expand.ledger_lock(self.ledger):
                self.fail('aliased lock accepted')
        self.assertEqual(outside.read_bytes(), b'unmodified')
        bad = cell(8); bad['cell_id'] = '../wrong'
        with self.assertRaisesRegex(ValueError, 'SAFE_CELL'):
            expand.consume_cell(self.ledger, bad, REF, BINDING, OWNER, lambda *_: self.fail('unsafe cell called'))

    def queue_fixture(self, extra=None):
        prepared = self.root / 'prepared'; prepared.mkdir()
        stage = {'stage_id': 'fixture-next56', 'prepared': str(prepared), 'execution_root': str(ROOT),
                 'runtime_python': sys.executable, 'heavy_lock': str(self.root / 'heavy.lock'),
                 'queue_root': str(prepared / 'queue'), 'chunks': [[cell(i)['cell_id'] for i in range(8, 36)],
                                                               [cell(i)['cell_id'] for i in range(36, 64)]]}
        stage.update(extra or {})
        expand.once(prepared / 'stage.json', stage)
        for i in range(2):
            recipe = expand.job_recipe(stage, i)
            expand.once(prepared / 'jobs' / (recipe['job_id'] + '.json'), recipe)
        request = expand.queue_request(stage)
        expand.once(prepared / 'queue-request.json', request)
        return stage, request

    def create_queue(self, stage, request):
        root = Path(stage['queue_root']); root.mkdir()
        expand.atomic_bytes(root / 'request.json', (Path(stage['prepared']) / 'queue-request.json').read_bytes())
        expand.once(root / 'active-plan.json', {'fixture': True})
        expand.once(root / 'queue.json', {'schema': request['schema'],
            'request_sha256': expand.common.sha_file(root / 'request.json'), 'heavy_lock': stage['heavy_lock'],
            'automatic_retries': 0, 'historical_jobs_imported': 0,
            'active_plan_sha256': expand.common.sha_file(root / 'active-plan.json')})
        for job in request['jobs']:
            (root / 'jobs' / job['job_id'] / 'outputs').mkdir(parents=True)
            expand.atomic_bytes(root / 'recipes' / (job['job_id'] + '.json'), Path(job['recipe']).read_bytes())

    def claim_queue(self, stage, definition, *, prior=None, allowed=600_000_000_000):
        root = Path(stage['queue_root']); folder = root / 'jobs' / definition['job_id']
        dependencies = {} if prior is None else {'chunk-001': {'receipt': str(root / 'jobs/chunk-001/result.json'),
            'receipt_sha256': expand.common.sha_file(root / 'jobs/chunk-001/result.json'),
            'outputs_root': str(root / 'jobs/chunk-001/outputs'), 'outputs': prior['outputs']}}
        claim = {'schema': 'kws-local-research-job-claim-v1', 'job_id': definition['job_id'],
            'queue_sha256': expand.common.sha_file(root / 'queue.json'),
            'job_definition_sha256': controller.sha(controller.canonical(definition)), 'recipe_sha256': definition['recipe_sha256'],
            'attempt': 1, 'automatic_retry': False, 'allowed_runner_nanoseconds': allowed,
            'started_monotonic_ns': 1, 'deadline_monotonic_ns': 1 + allowed,
            'global_actual_before_nanoseconds': 0 if prior is None else prior['actual_elapsed_ns'],
            'dependencies_sha256': controller.sha(controller.canonical(dependencies))}
        expand.once(folder / 'claim.json', claim); expand.once(folder / 'dependencies.json', dependencies)
        return claim

    def terminal_queue(self, stage, definition, claim, *, status='completed', elapsed=15):
        folder = Path(stage['queue_root']) / 'jobs' / definition['job_id']; files = {}
        logs = {}
        for name in ('stdout.log', 'stderr.log'):
            expand.atomic_bytes(folder / name, b'fixture log\n'); logs[name] = expand.common.record(folder / name)
        if status == 'completed':
            for name in expand.OUTPUTS:
                expand.atomic_bytes(folder / 'outputs' / name, b'fixture\n')
                files[name] = expand.common.record(folder / 'outputs' / name)
        value = {'schema': 'kws-local-research-job-result-v1',
            **{key: claim[key] for key in ('job_id', 'queue_sha256', 'job_definition_sha256', 'recipe_sha256')},
            'claim_sha256': expand.common.sha_file(folder / 'claim.json'), 'status': status,
            'actual_elapsed_ns': elapsed, 'exit_code': 0 if status == 'completed' else 1, 'outputs': files, 'logs': logs}
        expand.once(folder / 'result.json', value)
        return value

    def test_queue_request_cannot_raise_limits_or_remove_dependency(self):
        stage, request = self.queue_fixture()
        self.assertEqual(request['global_budget_minutes'], 20)
        self.assertEqual([j['timeout_minutes'] for j in request['jobs']], [10, 10])
        self.assertEqual(request['jobs'][1]['depends_on'], ['chunk-001'])
        path = Path(stage['prepared']) / 'queue-request.json'
        for mutate in ('budget', 'dependency'):
            changed = json.loads(json.dumps(request))
            if mutate == 'budget': changed['global_budget_minutes'] = 21
            else: changed['jobs'][1]['depends_on'] = []
            path.write_bytes(expand.common.canonical(changed) + b'\n')
            with self.assertRaisesRegex(ValueError, 'STAGE_QUEUE_REQUEST_CHANGED'):
                expand.queue_state(stage)

    def test_bulk_and_queue_canonical_conventions_are_distinct(self):
        value = {'fixture': '字节'}
        self.assertEqual(expand.queue_digest(value), controller.sha(controller.canonical(value)))
        self.assertNotEqual(expand.digest(value), expand.queue_digest(value))
        self.assertEqual(expand.digest(value), hashlib.sha256(expand.common.canonical(value)).hexdigest())

    def completed_history_fixture(self):
        """Use the real controller's zero-model claim/result, not mirrored JSON."""
        recipe = self.root / 'base-recipe.json'
        expand.once(recipe, {'fixture': 'completed-base-history-only'})
        queue_root = self.root / 'base-queue'
        command = "from pathlib import Path;import sys;Path(sys.argv[1]).joinpath('probe.json').write_text('{}\\n')"
        request = {'schema': controller.SCHEMA, 'queue_id': 'completed-history-fixture',
            'global_budget_minutes': 10, 'jobs': [{'job_id': 'first-eight', 'recipe': str(recipe),
                'recipe_sha256': expand.common.sha_file(recipe),
                'argv': [sys.executable, '-I', '-S', '-B', '-c', command, '{output_dir}'],
                'cwd': str(self.root), 'depends_on': [], 'timeout_minutes': 10, 'outputs': ['probe.json']}]}
        request_path = self.root / 'base-request.json'; expand.once(request_path, request)
        argv = [sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_queue.py')]
        for args in (['init', '--request', str(request_path), '--root', str(queue_root),
                      '--heavy-lock', str(self.root / 'base-heavy.lock')], ['run', '--root', str(queue_root)]):
            result = subprocess.run(argv + args, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        job = queue_root / 'jobs/first-eight'
        return job, expand.common.sha_file(job / 'result.json'), expand.common.sha_file(recipe)

    def test_completed_base_history_uses_actual_queue_duration_only(self):
        job, result_sha, recipe_sha = self.completed_history_fixture()
        before = {p: p.read_bytes() for p in job.rglob('*') if p.is_file()}
        history = expand.freeze_completed_base_history(job, result_sha, recipe_sha)
        actual = expand.common.read(job / 'result.json')['actual_elapsed_ns']
        self.assertGreater(actual, 0)
        self.assertEqual(history['schema'], 'active-tts-completed-base-history-v1')
        self.assertEqual(history['scope'], 'verified-first-eight-job-only')
        self.assertEqual(history['historical_known_job_nanoseconds'], actual)
        self.assertEqual(history['historical_actual_total_nanoseconds'], actual)
        self.assertTrue(history['historical_actual_total_complete'])
        self.assertFalse(history['project_or_runtime_lifetime_total_claimed'])
        self.assertEqual(history['unknown_conservative_reservation_seconds'], 0)
        self.assertNotIn('old_attempt', history)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_completed_history_refuses_unknown_missing_duration_or_changed_binding(self):
        job, result_sha, recipe_sha = self.completed_history_fixture()
        path = job / 'result.json'; original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'COMPLETED_BASE_TERMINAL_CHANGED'):
            expand.freeze_completed_base_history(job, '0' * 64, recipe_sha)
        for changes in ({'status': 'unknown', 'actual_elapsed_ns': None},
                        {'actual_duration_known': False}, {'actual_elapsed_ns': True},
                        {'recipe_sha256': '0' * 64}, {'claim_sha256': '0' * 64}):
            with self.subTest(changes=changes):
                row = json.loads(original); row.update(changes)
                path.write_bytes(expand.common.canonical(row) + b'\n')
                with self.assertRaises(ValueError):
                    expand.freeze_completed_base_history(job, expand.common.sha_file(path), recipe_sha)
        path.write_bytes(original)
        claim_path = job / 'claim.json'; claim = expand.common.read(claim_path)
        claim['allowed_runner_nanoseconds'] += 1
        claim_path.write_bytes(expand.common.canonical(claim) + b'\n')
        with self.assertRaises(ValueError):
            expand.freeze_completed_base_history(job, result_sha, recipe_sha)

    def test_old_unknown_history_is_separate_and_never_rewritten_as_zero(self):
        old = self.root / 'old/jobs/smoke8/result.json'
        expand.once(old, {'status': 'unknown', 'actual_elapsed_ns': None})
        known = self.root / 'known/jobs/success/result.json'
        expand.once(known, {'status': 'completed', 'actual_elapsed_ns': 123, 'recipe_sha256': 'a' * 64})
        budget = {'schema': 'active-tts-cross-attempt-budget-v1', 'actual_total_complete': False,
            'old_attempt': {'status': 'unknown', 'actual_elapsed_seconds': None,
                'queue': str(self.root / 'old'), 'terminal_receipt_sha256': expand.common.sha_file(old),
                'conservative_reservation_seconds': 600},
            'new_attempt': {'queue': str(self.root / 'known'),
                'jobs': [{'id': 'success', 'recipe_sha256': 'a' * 64}]},
            'total_conservative_allocation_seconds': 1200}
        path = self.root / 'old-budget.json'; expand.once(path, budget)
        history = expand.freeze_history(path)
        self.assertEqual(history['schema'], 'active-tts-extension-history-v1')
        self.assertIsNone(history['historical_actual_total_nanoseconds'])
        self.assertFalse(history['historical_actual_total_complete'])
        self.assertEqual(history['unknown_conservative_reservation_seconds'], 600)
        self.assertEqual(history['historical_known_job_nanoseconds'], 123)
        budget['actual_total_complete'] = True
        path.write_bytes(expand.common.canonical(budget) + b'\n')
        with self.assertRaisesRegex(ValueError, 'HISTORICAL_UNKNOWN_MUST_REMAIN_UNKNOWN'):
            expand.freeze_history(path)

    def test_invalid_history_is_rejected_before_any_shared_ledger_write(self):
        bad = self.root / 'bad-history.json'; expand.once(bad, {'schema': 'not-a-history'})
        args = SimpleNamespace(stage_id='fresh-history', target_count=64, base_recipe=self.root / 'recipe.json',
            base_recipe_sha256='a' * 64, base_job=self.root / 'base', base_result_sha256='b' * 64,
            completed_base_history=False, prior_budget=bad, prior_stage=None,
            prior_stage_sha256=None, prior_result_sha256=None, heavy_lock=self.root / 'unused.lock',
            output=self.root / 'new-stage')
        with patch.object(expand, 'verify_parent', return_value=({'runtime': str(self.root / 'runtime')},
                self.root / 'prepared', self.plan, None, BINDING)), \
                patch.object(expand, 'verify_base_job', return_value={}):
            with self.assertRaisesRegex(ValueError, 'HISTORICAL_UNKNOWN_MUST_REMAIN_UNKNOWN'):
                expand.prepare(args)
        self.assertFalse((self.root / 'runtime').exists())
        self.assertFalse(args.output.exists())

    def test_recovery_budget_is_fixed_and_never_allows_arbitrary_minutes(self):
        stage, _ = self.queue_fixture()
        stage.update(stage_budget_minutes=20, infrastructure_attempt=2, queue_budget_minutes=19.99,
                     recovery={'fixture': True})
        self.assertEqual(expand.queue_request(stage)['global_budget_minutes'], 19.99)
        for attempt, minutes in ((2, 20), (2, 19.5), (3, 19.99), (1, 19.99)):
            stage.update(infrastructure_attempt=attempt, queue_budget_minutes=minutes)
            with self.assertRaisesRegex(ValueError, 'FIXED_STAGE_AND_RECOVERY_QUEUE_BUDGET'):
                expand.queue_request(stage)

    def test_real_queue_cli_claims_and_dependencies_pass_zero_model_verifier(self):
        # The real controller writes its own canonical identities and receipts.
        # A stdlib probe invokes the production queue_state inside both actual
        # commands, exercising the first claim and the second prerequisite.
        prepared = self.root / 'real-cli-stage'; prepared.mkdir()
        project = self.root / 'probe-source'; (project / 'tools').mkdir(parents=True)
        probe = project / 'tools/active_tts_expand.py'
        probe.write_text("""import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0, TOOLS)
import active_tts_expand as e
p=argparse.ArgumentParser(); p.add_argument('command')
for n in ('recipe','output','heavy-lock'): p.add_argument('--'+n,required=True)
a=p.parse_args(); job=e.common.read(a.recipe); stage=e.common.read(job['stage'])
_,states=e.queue_state(stage,require_created=True)
if states[job['job_id']]['status']!='claimed-without-terminal': raise RuntimeError('missing live claim')
with e.heavy_model_slot(a.heavy_lock) as lease:
 if not lease.supervised: raise RuntimeError('missing supervision')
 for name in e.OUTPUTS:
  e.once(Path(a.output)/name,{'fixture':True,'job':job['job_id'],'pgid':os.getpgrp()})
if any(name in sys.modules for name in ('torch','numpy','onnxruntime','qwen_tts')): raise RuntimeError('ML import')
""".replace('TOOLS', repr(str(ROOT / 'tools'))))
        stage = {'stage_id': 'real-cli-fixture', 'prepared': str(prepared), 'execution_root': str(project),
            'runtime_python': sys.executable, 'heavy_lock': str(self.root / 'probe-heavy.lock'),
            'queue_root': str(prepared / 'queue'), 'chunks': [[cell(8)['cell_id']], [cell(9)['cell_id']]],
            'stage_budget_minutes': 20, 'infrastructure_attempt': 2, 'queue_budget_minutes': 19.99,
            'recovery': {'fixture': 'recovery evidence is separately tested; this probe checks queue protocol'}}
        expand.once(prepared / 'stage.json', stage)
        for index in range(2):
            job = expand.job_recipe(stage, index)
            expand.once(prepared / 'jobs' / (job['job_id'] + '.json'), job)
        expand.once(prepared / 'queue-request.json', expand.queue_request(stage))
        argv = [sys.executable, '-I', '-S', '-B', str(ROOT / 'tools/research_queue.py')]
        commands = [argv + ['init', '--request', str(prepared / 'queue-request.json'),
                           '--root', stage['queue_root'], '--heavy-lock', stage['heavy_lock']],
                    argv + ['run', '--root', stage['queue_root']]]
        for command in commands:
            result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        _, states = expand.queue_state(stage, require_created=True)
        self.assertEqual([row['status'] for row in states.values()], ['completed', 'completed'])
        self.assertGreater(states['chunk-001']['actual_elapsed_ns'], 0)
        self.assertEqual(states['chunk-002']['claim']['global_actual_before_nanoseconds'],
                         states['chunk-001']['actual_elapsed_ns'])
        terminal = json.loads(result.stdout)
        self.assertEqual(terminal['global_budget_minutes'], 19.99)
        self.assertEqual(terminal['remaining_budget_nanoseconds'] + terminal['actual_known_job_nanoseconds'],
                         expand.RECOVERY_QUEUE_NS)

    def recovery_fixture(self):
        stage, request = self.queue_fixture({'schema': expand.SCHEMA, 'stage_budget_minutes': 20,
            'source_revision': 'a' * 40, 'execution_source': {name: expand.common.record(ROOT / name) for name in expand.OWN_SOURCES},
            'ledger': str(self.ledger), 'binding': BINDING,
            'target_cells': [cell(i)['cell_id'] for i in range(64)]})
        for name in expand.OWN_SOURCES:
            expand.atomic_bytes(Path(stage['prepared']) / 'execution-source' / name, (ROOT / name).read_bytes())
        self.create_queue(stage, request)
        claim = self.claim_queue(stage, request['jobs'][0])
        self.terminal_queue(stage, request['jobs'][0], claim, status='failed', elapsed=275_788_912)
        stage_path = Path(stage['prepared']) / 'stage.json'
        result_path = Path(stage['queue_root']) / 'jobs/chunk-001/result.json'
        return stage, stage_path, result_path

    def test_recovery_charges_actual_and_rejects_prior_outputs_or_cell_claims(self):
        stage, stage_path, result_path = self.recovery_fixture()
        args = (stage_path, expand.common.sha_file(stage_path), expand.common.sha_file(result_path))
        recovery = expand.freeze_recovery(*args)
        self.assertEqual(recovery['actual_elapsed_nanoseconds'], 275_788_912)
        self.assertEqual(recovery['maximum_combined_nanoseconds'], 1_199_675_788_912)
        new_stage = dict(stage, infrastructure_attempt=2, queue_budget_minutes=19.99, recovery=recovery)
        registration = self.ledger / 'recoveries' / (recovery['prior_stage_sha256'] + '.json')
        expand.once(registration, expand.recovery_registration(new_stage))
        expand.verify_recovery(new_stage)
        with self.assertRaisesRegex(ValueError, 'RECOVERY_ALREADY_ASSIGNED_NO_THIRD_ATTEMPT'):
            expand.freeze_recovery(*args)
        registration.unlink()  # Independent boundary fixture below, not a real recovery ledger.
        output = Path(stage['queue_root']) / 'jobs/chunk-001/outputs/invocation.json'
        expand.once(output, {'fixture': True})
        with self.assertRaisesRegex(ValueError, 'PRIOR_MODEL_OR_OUTPUT_ACTIVITY'):
            expand.freeze_recovery(*args)
        expand.once(registration, expand.recovery_registration(new_stage))
        with self.assertRaisesRegex(ValueError, 'PRIOR_OUTPUTS_CHANGED'):
            expand.verify_recovery(new_stage)
        registration.unlink()
        output.unlink()
        expand.once(self.ledger / 'claims' / (cell(8)['cell_id'] + '.json'), {'fixture': 'consumed'})
        with self.assertRaisesRegex(ValueError, 'PRIOR_CELL_CONSUMPTION'):
            expand.freeze_recovery(*args)

    def test_recovery_cannot_hide_elapsed_over_combined_limit(self):
        _, stage_path, result_path = self.recovery_fixture()
        result = expand.common.read(result_path); result['actual_elapsed_ns'] = 600_000_001
        result_path.write_bytes(expand.common.canonical(result) + b'\n')
        with self.assertRaisesRegex(ValueError, 'RECOVERY_REQUIRES_BOUNDED_PRE_MODEL_FAILURE'):
            expand.freeze_recovery(stage_path, expand.common.sha_file(stage_path), expand.common.sha_file(result_path))

    def test_queue_accumulates_actual_and_unknown_remains_null(self):
        stage, request = self.queue_fixture(); self.create_queue(stage, request)
        first = self.claim_queue(stage, request['jobs'][0])
        result = self.terminal_queue(stage, request['jobs'][0], first, elapsed=172_000_000_000)
        second = self.claim_queue(stage, request['jobs'][1], prior=result)
        self.terminal_queue(stage, request['jobs'][1], second, status='unknown', elapsed=None)
        _, states = expand.queue_state(stage)
        self.assertEqual(states['chunk-001']['actual_elapsed_ns'], 172_000_000_000)
        self.assertIsNone(states['chunk-002']['actual_elapsed_ns'])
        self.assertEqual(states['chunk-002']['status'], 'unknown')
        path = Path(stage['queue_root']) / 'jobs/chunk-002/claim.json'
        changed = expand.common.read(path); changed['global_actual_before_nanoseconds'] = 0
        path.write_bytes(expand.common.canonical(changed) + b'\n')
        with self.assertRaisesRegex(ValueError, 'QUEUE_CUMULATIVE_BUDGET_CHANGED'):
            expand.queue_state(stage)

    def test_queue_rejects_overbudget_claim_before_any_model(self):
        stage, request = self.queue_fixture(); self.create_queue(stage, request)
        self.claim_queue(stage, request['jobs'][0], allowed=601_000_000_000)
        with self.assertRaisesRegex(ValueError, 'QUEUE_CLAIM_IDENTITY_OR_BUDGET_CHANGED'):
            expand.queue_state(stage)

    def test_outcomes_mirror_pinned_ledger_and_claim_for_data_consumer(self):
        stage, _ = self.queue_fixture()
        stage.update(ledger=str(self.ledger), binding=BINDING)
        job = expand.job_recipe(stage, 0)
        owner = {'stage_sha256': job['stage_sha256'], 'job_id': job['job_id'],
                 'job_recipe_sha256': expand.common.sha_file(Path(stage['prepared']) / 'jobs/chunk-001.json')}
        for index in range(8, 36):
            expand.consume_cell(self.ledger, cell(index), REF, BINDING, owner,
                lambda row, ref: raw_fixture(self.output, row, ref, failed=row['seed'] == 35))
        value = expand.outcomes(job, stage, job['cells'], [])
        self.assertEqual(value['schema'], 'active-tts-stage-outcomes-v1')
        self.assertEqual(len(value['rows']), 28)
        for row in value['rows']:
            self.assertEqual(hashlib.sha256(expand.common.canonical(row['record']) + b'\n').hexdigest(), row['ledger_record_sha256'])
            self.assertEqual(hashlib.sha256(expand.common.canonical(row['claim']) + b'\n').hexdigest(), row['record']['claim_sha256'])
        self.assertEqual(value['rows'][-1]['record']['status'], 'FAILED_NO_RETRY')
        self.assertTrue(value['rows'][-1]['record']['native_wav'].endswith('/returned-native.wav'))

    def test_completed_entry_cache_never_acquires_model_slot_and_checks_outcome_pin(self):
        stage, _ = self.queue_fixture()
        stage.update(ledger=str(self.ledger), binding=BINDING)
        job = expand.job_recipe(stage, 0)
        self.output = Path(stage['queue_root']) / 'jobs/chunk-001/outputs'
        recipe_path = Path(stage['prepared']) / 'jobs/chunk-001.json'
        owner = {'stage_sha256': job['stage_sha256'], 'job_id': job['job_id'],
                 'job_recipe_sha256': expand.common.sha_file(recipe_path)}
        for index in range(8, 36):
            expand.consume_cell(self.ledger, cell(index), REF, BINDING, owner,
                                lambda row, ref: raw_fixture(self.output, row, ref))
        result = expand.outcomes(job, stage, job['cells'], [])
        expand.once(self.output / 'outcomes.json', result)
        expand.once(self.output / 'invocation.json', {'fixture': True})
        expand.once(self.output / 'inventory.json', {'files': {p.relative_to(self.output).as_posix(): expand.common.record(p)
                    for p in self.output.rglob('*') if p.is_file()}})
        summary = {'schema': 'active-tts-extension-result-v1', 'status': 'completed',
            'job_recipe_sha256': expand.common.sha_file(recipe_path), 'stage_sha256': job['stage_sha256'],
            'inventory': expand.common.record(self.output / 'inventory.json'),
            'outcomes': expand.common.record(self.output / 'outcomes.json'),
            'new_call_ids': job['cells'], 'reused_cell_ids': [], 'new_generation_calls': 28,
            'cells': [{'cell_id': row['cell_id'], 'status': row['record']['status'],
                       'ledger_record_sha256': row['ledger_record_sha256']} for row in result['rows']]}
        expand.once(self.output / 'summary.json', summary)
        args = Namespace(recipe=recipe_path, output=self.output, heavy_lock=Path(stage['heavy_lock']))
        with patch.object(expand, 'verify_job', return_value=(job, stage, {}, self.root, self.plan, None)), \
                patch.object(expand, 'queue_state', return_value=({}, {'chunk-001': {'status': 'completed'}})), \
                patch.object(expand, 'heavy_model_slot', side_effect=RuntimeError('model slot reached')), \
                patch.dict(os.environ, {'KWS_RESEARCH_RECIPE_SHA256': expand.common.sha_file(recipe_path)}):
            value = expand.run(args)
            self.assertEqual(value['new_model_calls'], 0)
            (self.output / 'outcomes.json').write_bytes(b'changed outcome')
            with self.assertRaisesRegex(ValueError, 'CACHED_JOB_RECEIPTS_CHANGED'):
                expand.run(args)

    def test_export_only_new_roster_preserves_failed_raw_and_all_reused_bindings(self):
        prepared = self.root / 'export-stage'; prepared.mkdir()
        stage = {'prepared': str(prepared), 'ledger': str(self.ledger), 'binding': BINDING,
                 'queue_root': str(prepared / 'queue'), 'chunks': [[cell(8)['cell_id'], cell(9)['cell_id']], [cell(10)['cell_id']]],
                 'target_count': 11, 'target_cells': [cell(i)['cell_id'] for i in range(11)],
                 'initial_reused_cells': [cell(i)['cell_id'] for i in range(8)]}
        expand.once(prepared / 'stage.json', stage)
        records = {}
        for index in list(range(8)) + [9]:
            records[cell(index)['cell_id']] = self.consume(index)[0]
        for job_index, index in enumerate((8, 10)):
            name = 'chunk-' + str(job_index + 1).zfill(3)
            output = Path(stage['queue_root']) / 'jobs' / name / 'outputs'
            records[cell(index)['cell_id']] = self.consume(index, callback=lambda row, ref: raw_fixture(output, row, ref, failed=index == 10))[0]
            expand.once(output / 'summary.json', {'new_call_ids': [cell(index)['cell_id']]})
            expand.once(output / 'outcomes.json', {'fixture': True})
            expand.once(prepared / 'jobs' / (name + '.json'), {'fixture': True})
        state = {'pending_cells': [], 'unknown_cells': [], 'records': records,
                 'jobs': {'chunk-001': {'status': 'completed'}, 'chunk-002': {'status': 'completed'}}}
        with patch.object(expand, 'load_stage', return_value=(stage, {}, prepared, self.plan, None)), \
                patch.object(expand, 'inspect_stage', return_value=state):
            result = expand.export_asr(prepared / 'stage.json')
            repeated = expand.export_asr(prepared / 'stage.json')
        self.assertEqual(result, repeated)
        self.assertEqual(result['reused_cells_without_new_asr_request'], [cell(i)['cell_id'] for i in list(range(8)) + [9]])
        self.assertEqual(len(result['reused_bindings']), 9)
        self.assertEqual(result['asr_calls_executed'], 0)
        self.assertFalse(result['paired_cases_included'])
        for partition, index in zip(result['partitions'], (8, 10)):
            with zipfile.ZipFile(partition['archive']) as stream:
                self.assertEqual(stream.namelist(), ['cells/' + cell(index)['cell_id'] + '/native-f32.wav'])
                record = records[cell(index)['cell_id']]
                self.assertEqual(stream.read(stream.namelist()[0]), (Path(record['output_root']) / record['native_wav']).read_bytes())
        self.assertEqual(result['partitions'][1]['technical_failed_cells'], [cell(10)['cell_id']])
        self.assertEqual(result['asr_eligible_new_cell_count'], 2)
        self.assertEqual(result['excluded_cells'], [])
        state['unknown_cells'] = [cell(11)['cell_id']]
        with patch.object(expand, 'load_stage', return_value=(stage, {}, prepared, self.plan, None)), \
                patch.object(expand, 'inspect_stage', return_value=state), \
                self.assertRaisesRegex(ValueError, 'COMPLETE_STAGE_REQUIRED_FOR_EXPORT'):
            expand.export_asr(prepared / 'stage.json')

    def test_export_excluded_new_cell_does_not_become_reused_or_empty_asr_batch(self):
        prepared = self.root / 'invalid-export'; prepared.mkdir()
        stage = {'prepared': str(prepared), 'ledger': str(self.ledger), 'binding': BINDING,
                 'queue_root': str(prepared / 'queue'), 'chunks': [[cell(8)['cell_id']]],
                 'target_count': 1, 'target_cells': [cell(8)['cell_id']], 'initial_reused_cells': []}
        expand.once(prepared / 'stage.json', stage)
        output = Path(stage['queue_root']) / 'jobs/chunk-001/outputs'
        value, _ = self.consume(callback=lambda row, ref: raw_fixture(output, row, ref, failed=True))
        raw_path = output / value['native_wav']; body = raw_path.read_bytes()
        raw_path.write_bytes(body[:44] + struct.pack('<f', float('nan')) + body[48:])
        value['files'][value['native_wav']] = expand.common.record(raw_path)
        record_path = self.ledger / 'cells' / (cell(8)['cell_id'] + '.json')
        record_path.write_bytes(expand.common.canonical(value) + b'\n')
        expand.once(output / 'summary.json', {'new_call_ids': [cell(8)['cell_id']]})
        expand.once(output / 'outcomes.json', {'fixture': True})
        expand.once(prepared / 'jobs/chunk-001.json', {'fixture': True})
        state = {'pending_cells': [], 'unknown_cells': [], 'records': {cell(8)['cell_id']: value},
                 'jobs': {'chunk-001': {'status': 'completed'}}}
        with patch.object(expand, 'load_stage', return_value=(stage, {}, prepared, self.plan, None)), \
                patch.object(expand, 'inspect_stage', return_value=state):
            result = expand.export_asr(prepared / 'stage.json')
        self.assertEqual(result['partitions'], [])
        self.assertEqual(result['reused_cells_without_new_asr_request'], [])
        self.assertEqual(result['new_generation_cell_count'], 1)
        self.assertEqual(result['asr_excluded_new_cell_count'], 1)
        self.assertEqual(result['excluded_cells'][0]['reason'], 'ASR_NONFINITE_SAMPLE')
        self.assertIsNone(result['excluded_cells'][0]['observed_text'])
        self.assertEqual(result['job_outcomes'][0]['job_id'], 'chunk-001')

    def test_cli_isolated_without_machine_learning_imports(self):
        code = "import runpy,sys; sys.argv=[sys.argv[1],'--help']; "
        code += "\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit: pass\n"
        code += "if any(x in sys.modules for x in ('torch','numpy','onnxruntime','qwen_tts')): raise RuntimeError('ML import')"
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(ROOT / 'tools/active_tts_expand.py')],
                                cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
