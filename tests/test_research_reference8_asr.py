#!/usr/bin/env python3
"""No-model sequence controls using the real batch controller and fake workers."""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('KWS_REFERENCE8_TEST_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'tools'))
import research_reference8_asr as stage


class SequenceControlTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='reference8-control-fixture-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'native'
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.prep = self.root / 'preparation-checkpoint'
        self.prep.mkdir()
        stage.batch.write_once(self.prep / 'ack.json', {'synthetic_fixture': True})
        self.inputs_path = self.root / 'inputs.json'
        clips = []
        for i in range(8):
            pcm = self.root / ('clip-%03d.f32' % i)
            pcm.write_bytes(struct.pack('<16f', *([i / 16] * 16)))
            clips.append({'id': 'clip-%03d' % i, 'original_member': 'original/clip-%03d.wav' % i,
                'original_wav': {'sha256': '%064x' % (i+1)}, 'original_pcm': {'sha256': '%064x' % (i+100)},
                'derived_pcm': {'path': pcm.name, 'sha256': stage.batch.sha_file(pcm),
                    'sample_rate': 16000, 'channels': 1, 'dtype': '<f4', 'frames': 16}})
        self.inputs = {'schema': 'active-asr-inputs-v1', 'clips': clips}
        stage.batch.write_once(self.inputs_path, self.inputs)
        self.recipe = {'runtime': str(self.runtime), 'output': str(self.output),
            'inputs': stage.cp.pin(self.inputs_path), 'input_checkpoint': str(self.root/'input-checkpoint'),
            'preparation_checkpoint': str(self.prep), 'input_ack': {'synthetic_fixture': True},
            'models': list(stage.batch.MODELS), 'phases': stage.phase_contract(clips)}
        self.request = self.root / 'request.json'
        stage.batch.write_once(self.request, self.recipe)
        stage.batch.write_once(self.prep / 'capsule.json', {'context': {'request': stage.cp.pin(self.request)}})
        self.wall = 1000.0
        self.calls = []
        self.expire_after_smoke = False
        self.incomplete_first_model = False
        self.supervised = True
        self.stdout = io.StringIO()

    def identity(self, model, runtime, inputs, threads):
        return {'model': model, 'input_manifest_sha256': stage.batch.sha_file(inputs), 'threads': threads}

    @contextmanager
    def lease(self, path):
        self.assertEqual(Path(path), self.runtime / 'heavy-model.lock')
        yield SimpleNamespace(inherited=True, supervised=self.supervised)

    def worker(self, command, environment, log, runtime, remaining):
        # Keep the real batch's roster, persisted job and completed-result reuse.
        # Only the model subprocess is replaced by create-only synthetic evidence.
        def value(flag):
            return command[command.index(flag) + 1]
        model, attempt = value('--model'), value('--attempt')
        selected = value('--ids').split(',')
        identity = self.identity(model, runtime, self.inputs_path, 4)
        job = stage.batch.read_json(self.output / 'job.json')
        entry = {'model': model, 'attempt': attempt, 'selected': selected,
            'new': [], 'reused': [], 'job_sha256': stage.batch.sha_file(self.output / 'job.json'),
            'deadline': job['deadline_unix'], 'remaining': remaining}
        self.calls.append(entry)
        by_id = {x['id']: x for x in self.inputs['clips']}
        for identifier in selected:
            clip = by_id[identifier]
            status, key = stage.batch.existing_result(self.output, model, clip, identity)
            if status == 'complete':
                entry['reused'].append(identifier)
                continue
            self.assertEqual(status, 'pending')
            claim = {'schema': 'active-asr-call-claim-v1', 'identity_sha256': key,
                'model': model, 'clip_id': identifier, 'input': clip,
                'model_identity': identity, 'attempt': attempt}
            stage.batch.write_once(self.output / 'claims' / model / (identifier + '.json'), claim)
            entry['new'].append(identifier)
            if self.incomplete_first_model and attempt == 'smoke-001' and model == stage.batch.MODELS[0]:
                return 17
            native_path = self.output / 'native' / model / (identifier + '.json')
            stage.batch.write_once(native_path, {'schema': 'active-asr-native-output-v1',
                'identity_sha256': key, 'model': model, 'clip_id': identifier,
                'native': {'synthetic_fixture': True}})
            stage.batch.write_once(self.output / 'results' / model / (identifier + '.json'),
                {**claim, 'schema': 'active-asr-result-v1', 'status': 'complete',
                 'native_output_sha256': stage.batch.sha_file(native_path)})
        self.wall += 5
        if self.expire_after_smoke and attempt == 'smoke-001' and model == stage.batch.MODELS[-1]:
            self.wall = job['deadline_unix'] + 1
        log.write('synthetic worker completed; no model loaded\n')
        return 0

    def invoke(self, request=None):
        request = request or self.request
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage, 'verify_request', return_value=(self.recipe, self.inputs)))
            stack.enter_context(patch.object(stage.cp, 'verify_ack', return_value=None))
            stack.enter_context(patch.object(stage, 'heavy_model_slot', side_effect=self.lease))
            stack.enter_context(patch.object(stage.batch.research_runtime, 'load_manifest', return_value={}))
            stack.enter_context(patch.object(stage.batch.research_runtime, 'check_interpreter', return_value=None))
            stack.enter_context(patch.object(stage.batch, 'model_identity', side_effect=self.identity))
            stack.enter_context(patch.object(stage.batch, 'run_worker', side_effect=self.worker))
            stack.enter_context(patch.object(stage.batch.time, 'time', side_effect=lambda: self.wall))
            stack.enter_context(patch.dict(os.environ, {'KWS_RESEARCH_RECIPE_SHA256': stage.batch.sha_file(request)}))
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('network forbidden')))
            stack.enter_context(redirect_stdout(self.stdout))
            return stage.run(request)

    def result(self):
        return stage.batch.read_json(self.root / 'sequence-result.json')

    def test_real_batch_smoke_first_last_then_reuses_six_with_one_deadline(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.calls), 6)
        ids = [x['id'] for x in self.inputs['clips']]
        for entry in self.calls[:3]:
            self.assertEqual(entry['selected'], [ids[0], ids[-1]])
            self.assertEqual(entry['new'], [ids[0], ids[-1]])
        for entry in self.calls[3:]:
            self.assertEqual(entry['selected'], ids)
            self.assertEqual(entry['reused'], [ids[0], ids[-1]])
            self.assertEqual(entry['new'], ids[1:-1])
        self.assertEqual([len(x['new']) for x in self.calls], [2, 2, 2, 6, 6, 6])
        self.assertEqual({x['deadline'] for x in self.calls}, {1540.0})
        self.assertEqual(len({x['job_sha256'] for x in self.calls}), 1)
        result = self.result()
        self.assertEqual((result['status'], result['claims'], result['results']), ('complete', 24, 24))
        self.assertEqual([x['new_claims'] for x in result['phases']], [6, 18])
        self.assertTrue(result['deadline_was_not_reset'])
        self.assertNotIn('torch', sys.modules)

    def test_exhausted_smoke_deadline_does_not_start_full_or_reset_job(self):
        self.expire_after_smoke = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual({x['attempt'] for x in self.calls}, {'smoke-001'})
        result = self.result()
        self.assertEqual((result['status'], result['claims'], result['results']), ('incomplete', 6, 6))
        self.assertIn('ONE_DEADLINE_EXHAUSTED', result['error']['message'])
        self.assertEqual(stage.batch.read_json(self.output / 'job.json')['deadline_unix'], 1540.0)

    def test_consumed_output_refuses_reentry_without_any_new_worker(self):
        self.output.mkdir()
        retained = self.output / 'partial-evidence.json'
        retained.write_bytes(b'{"retained":true}\n')
        with self.assertRaisesRegex(ValueError, 'CONSUMED_OR_EXISTING_NATIVE_OUTPUT'):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(retained.read_bytes(), b'{"retained":true}\n')
        self.assertFalse((self.root / 'sequence-claim.json').exists())

    def test_incomplete_smoke_preserves_consumed_call_and_never_starts_full(self):
        self.incomplete_first_model = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual({x['attempt'] for x in self.calls}, {'smoke-001'})
        result = self.result()
        self.assertEqual((result['status'], result['claims'], result['results']), ('incomplete', 5, 4))
        clip = self.inputs['clips'][0]
        state, _ = stage.batch.existing_result(self.output, stage.batch.MODELS[0], clip,
            self.identity(stage.batch.MODELS[0], self.runtime, self.inputs_path, 4))
        self.assertEqual(state, 'consumed_without_complete_result')
        before = len(self.calls)
        with self.assertRaisesRegex(ValueError, 'CONSUMED_OR_EXISTING_NATIVE_OUTPUT'):
            self.invoke()
        self.assertEqual(len(self.calls), before)

    def test_same_bytes_at_queue_copy_path_need_the_original_ack_binding(self):
        copied = self.root / 'queue-copy.json'
        copied.write_bytes(self.request.read_bytes())
        with self.assertRaisesRegex(ValueError, 'ACTUAL_PREPARATION_ACK_MUST_BIND_THIS_REQUEST'):
            self.invoke(copied)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'sequence-claim.json').exists())

    def test_unsupervised_lease_cannot_claim_or_start_model_workers(self):
        self.supervised = False
        with self.assertRaisesRegex(ValueError, 'SUPERVISED_SHARED_QUEUE_SLOT_REQUIRED'):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'sequence-claim.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
