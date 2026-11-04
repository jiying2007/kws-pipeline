"""Two-phase CLI/state tests with invented objects, never model execution."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'phase'
        self.events = []
        self.args = NS(phase='prepare', retained_repo='/invented/code', runtime='/invented/runtime',
            output=str(self.output), executed_head='f'*40, sink_source='/invented/sink.py',
            consensus_tools='/invented/tools', recognizer_id='whisper-small', input=None,
            job_sha256=None, identity_map=None, policy=None, policy_sha256=None)
        self.prepared = NS(identity={'fixture': True}, sink=None, failed=False,
            bind_policy=lambda *a: self.events.append('policy'), infer=self.infer)
        self.failed_clip = None
        def prepare(*a, **k):
            self.events.append('prepare')
            self.output.mkdir()
            return self.prepared
        @contextmanager
        def deadline(value):
            self.events.append(('budget', value))
            yield
        self.patches = [patch.object(worker, 'prepare', prepare), patch.object(worker, 'deadline', deadline)]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
    def infer(self, **clip):
        self.events.append(('infer', clip['opaque_id']))
        if clip['opaque_id'] == self.failed_clip:
            raise TimeoutError('invented timeout')
        return dict(schema='invented-only', opaque_id=clip['opaque_id'])
    def make_input(self):
        self.args.phase = 'infer'
        root = self.root/'input'; (root/'audio').mkdir(parents=True)
        clips, mapping = [], []
        for index in range(1, 3):
            raw = ('invented wav %d' % index).encode()
            digest = worker.sha(raw)
            (root/'audio'/('%s.wav' % digest)).write_bytes(raw)
            clips.append(dict(audio_id='clip-%06d' % index, audio_path='audio/%s.wav' % digest, wav_sha256=digest))
            mapping.append(dict(audio_id='clip-%06d' % index, source_id='source-%d' % index,
                                request_id='request-%d' % index, wav_sha256=digest))
        job = worker.canonical(dict(schema='blind-source-screen32-job-v1', clips=clips))
        (root/'job.json').write_bytes(job)
        identity = self.root/'identity-map.json'
        identity.write_bytes(worker.canonical(dict(schema='blind-asr-source-request-map-v1', clips=mapping)))
        policy = self.root/'policy.json'; policy.write_bytes(b'{"invented_policy":true}')
        self.args.input, self.args.job_sha256 = str(root), worker.sha(job)
        self.args.identity_map, self.args.policy = str(identity), str(policy)
        self.args.policy_sha256 = worker.sha(policy.read_bytes())
    def test_prepare_no_audio_calls(self):
        result = worker.run(self.args)
        self.assertEqual(result['actual_decode_calls'], 0)
        self.assertEqual(self.events, [('budget', 1200), 'prepare'])
        self.assertEqual(json.loads((self.root/'progress.json').read_bytes())['stage'], 'load')
    def test_prepare_rejects_input_before_loader(self):
        self.args.input = '/invented/audio'
        with self.assertRaises(ValueError): worker.run(self.args)
        self.assertNotIn('prepare', self.events)
    def test_infer_reprepare_policy_then_each_clip_deadline(self):
        self.make_input()
        result = worker.run(self.args)
        self.assertEqual(result['completed_count'], 2)
        self.assertEqual(self.events, [('budget', 1200), 'prepare', ('budget', 1200), 'policy',
            ('budget', 300), ('infer', 'clip-000001'), ('budget', 300), ('infer', 'clip-000002')])
        self.assertEqual(json.loads((self.root/'progress.json').read_bytes())['stage'], 'cell')
    def test_failure_stops_without_later_clip(self):
        self.make_input(); self.failed_clip = 'clip-000001'
        with self.assertRaises(TimeoutError): worker.run(self.args)
        self.assertNotIn(('infer', 'clip-000002'), self.events)
        self.assertTrue(self.prepared.failed)
        self.assertFalse((self.output/'worker-terminal.json').exists())
        failure = json.loads((self.root/'phase-failed.json').read_bytes())
        self.assertEqual((failure['status'], failure['active_index']), ('failed_no_retry', 1))
    def test_map_cannot_receive_target(self):
        self.make_input()
        path = Path(self.args.identity_map); value = json.loads(path.read_bytes())
        value['clips'][0]['target_text'] = 'invented target'
        path.write_bytes(worker.canonical(value))
        with self.assertRaises(ValueError): worker.run(self.args)
        self.assertNotIn('prepare', self.events)
    def test_frozen_job_hash_required_before_loading(self):
        self.make_input(); self.args.job_sha256 = '0'*64
        with self.assertRaises(ValueError): worker.run(self.args)
        self.assertNotIn('prepare', self.events)
    def test_extra_input_file_rejected(self):
        self.make_input(); (Path(self.args.input)/'identity-map.json').write_bytes(b'{}')
        with self.assertRaises(ValueError): worker.run(self.args)
        self.assertNotIn('prepare', self.events)
    def test_duplicate_request_rejected(self):
        self.make_input()
        path = Path(self.args.identity_map); value = json.loads(path.read_bytes())
        value['clips'][1]['request_id'] = value['clips'][0]['request_id']
        path.write_bytes(worker.canonical(value))
        with self.assertRaises(ValueError): worker.run(self.args)
        self.assertNotIn('prepare', self.events)
    def test_isolated_cli_help(self):
        result = subprocess.run([sys.executable, '-I', '-B', worker.__file__, '--help'],
            text=True, capture_output=True, check=True)
        self.assertIn('--phase', result.stdout)

    def test_cli_and_python38_syntax(self):
        argv = ['--phase', 'prepare', '--retained-repo', '/code', '--runtime', '/runtime',
            '--output', '/output/whisper-prepare', '--executed-head', 'f'*40,
            '--sink-source', '/code/sink.py', '--consensus-tools', '/code/tools']
        args = worker.parser().parse_args(argv)
        self.assertEqual(args.phase, 'prepare')
        for path in (Path(worker.__file__), Path(__file__)):
            ast.parse(path.read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
