"""Zero-model fault tests for publisher checkpoints; all execution/network mocked."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
R = HERE.parent.parent
S = HERE if (HERE / 'campaign_contract.py').is_file() else R / 'phase3-asr-license-repair/checkout/research/k1-punctuation6-asr18-v1'
sys.path.insert(0, str(S))
import campaign_contract as c
import evidence_transport as t

def local_module(name, leaf):
    spec = importlib.util.spec_from_file_location(name, HERE / leaf)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

o = local_module('failure_observer', 'failure_observer.py')
p = local_module('observed_evidence_publisher', 'evidence_publisher.py')
RUN = {'repository': t.REPOSITORY, 'run_id': 123, 'run_attempt': 1, 'job_key': 'asr18', 'executed_head_sha': 'c' * 40}
SENTINEL = 'ghs_SYNTHETIC_SECRET_/private/audio.wav_你好小窝'



class PublisherObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch('subprocess.Popen',side_effect=AssertionError('no real process')))
        self.stack.enter_context(mock.patch('socket.socket',side_effect=AssertionError('no network')))
        self.stdout = io.StringIO(); self.stack.enter_context(redirect_stdout(self.stdout))

    def patch(self, obj, name, **kw):
        return self.stack.enter_context(mock.patch.object(obj, name, **kw))

    def main_setup(self):
        control = p.disabled_publication_release()
        for key in ('enabled', 'root_publication_release', 'independent_source_review'): control[key] = True
        self.patch(c, 'read_json', side_effect=lambda path, **kwargs: control if path.name == 'publication-release.json' else {})
        self.stack.enter_context(mock.patch.dict(p.os.environ, {
            'GITHUB_EVENT_PATH': str(self.root / 'synthetic-event.json'), 'RUNNER_TEMP': str(self.root),
            'GITHUB_TOKEN': SENTINEL}, clear=True))
        self.patch(p, 'bound_release', return_value={'purpose': 'asr18'})
        self.patch(p, 'context', return_value=RUN)
        self.patch(t, 'read_package', return_value=b'SYNTHETIC_NOT_A_REAL_ARCHIVE')
        self.patch(p, 'GitHubAPI', return_value=object())
        self.patch(p, 'run_once', return_value={'synthetic_return': True})

    def final(self):
        lines = [json.loads(line) for line in self.stdout.getvalue().splitlines()]
        self.assertNotIn(SENTINEL, self.stdout.getvalue())
        self.assertNotIn('private/audio', self.stdout.getvalue())
        checkpoints = [v for v in lines if v['schema'] == 'k1-asr18-safe-checkpoint-v1']
        self.assertTrue(all(len(c.canonical(v)) <= 1024 and v['stage'] in o.STAGES for v in checkpoints))
        envelope = lines[-1]
        self.assertEqual(envelope['schema'], 'k1-asr18-independent-process-diagnostic-envelope-v1')
        self.assertEqual(envelope['record_sha256'], c.digest(c.canonical(envelope['record'])))
        self.assertLessEqual(envelope['record_bytes'], 16384)
        return envelope

    def test_disabled_gate_diagnostic_without_environment_token_or_write(self):
        class Forbidden(dict):
            def get(self, *args): raise AssertionError('environment read')
            def __contains__(self, key): raise AssertionError('environment read')
            def __getitem__(self, key): raise AssertionError('environment read')
        self.stack.enter_context(mock.patch.object(p.os, 'environ', Forbidden()))
        self.patch(c, 'read_json', return_value=p.disabled_publication_release())
        mkdir = self.patch(c, 'durable_mkdir', side_effect=AssertionError('write before gate'))
        api = self.patch(p, 'GitHubAPI', side_effect=AssertionError('token before gate'))
        with self.assertRaises(p.PublicationDisabled): p.main([])
        record = self.final()['record']
        self.assertEqual(record['stage'], 'publication_gate')
        self.assertIsNone(record['identity'])
        self.assertIsNone(record['claims']['consumed_attempts'])
        mkdir.assert_not_called(); api.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_event_read_error_has_stage_and_no_content(self):
        self.main_setup()
        self.patch(c, 'read_json', side_effect=[c.read_json(S / 'publication-release.json'), ValueError(SENTINEL)])
        with self.assertRaises(ValueError): p.main([])
        record = self.final()['record']
        self.assertEqual(record['stage'], 'event_read')
        self.assertEqual(record['errors'][0]['detail']['error_type'], 'ValueError')
        self.assertIsNone(record['claims']['consumed_attempts'])

    def test_package_missing_durable_diagnostic_without_writer_entry(self):
        self.main_setup()
        self.patch(t, 'read_package', side_effect=FileNotFoundError(SENTINEL))
        with self.assertRaises(FileNotFoundError): p.main([])
        envelope = self.final(); record = envelope['record']
        self.assertEqual(record['stage'], 'package_read')
        self.assertIsNone(record['claims']['consumed_attempts'])
        self.assertEqual(envelope['persistence']['status'], 'durable_local_record_written')
        self.assertTrue((self.root / 'k1-asr18-diagnostic-123-publisher/process-diagnostic.json').is_file())
        p.run_once.assert_not_called(); p.GitHubAPI.assert_not_called()

    def test_invalid_archive_same_stage_no_native_or_publish(self):
        self.main_setup(); self.patch(t, 'read_package', side_effect=ValueError(SENTINEL))
        with self.assertRaises(ValueError): p.main([])
        self.assertEqual(self.final()['record']['stage'], 'package_read')
        p.run_once.assert_not_called()

    def test_token_admission_error_reports_no_secret(self):
        self.main_setup(); self.patch(p, 'GitHubAPI', side_effect=ValueError(SENTINEL))
        with self.assertRaises(ValueError): p.main([])
        self.assertEqual(self.final()['record']['stage'], 'token_admission')
        p.run_once.assert_not_called()

    def test_receipt_directory_error_has_distinct_stage(self):
        self.main_setup(); original = c.durable_mkdir
        def mkdir(path):
            if path.name == 'k1-asr18-publication-123': raise PermissionError(SENTINEL)
            return original(path)
        self.patch(c, 'durable_mkdir', side_effect=mkdir)
        with self.assertRaises(PermissionError): p.main([])
        self.assertEqual(self.final()['record']['stage'], 'receipt_directory')
        p.run_once.assert_not_called()

    def test_success_summary_does_not_make_native_or_consumption_claim(self):
        self.main_setup()
        self.assertEqual(p.main([]), {'synthetic_return': True})
        record = self.final()['record']
        self.assertEqual(record['status'], 'returned_success')
        self.assertFalse(record['native_success_established'])
        self.assertFalse(record['retry_authorized'])
        self.assertIsNone(record['claims']['consumed_attempts'])

    def test_diagnostic_failure_never_replaces_original_error(self):
        self.main_setup(); original = FileNotFoundError(SENTINEL)
        self.patch(t, 'read_package', side_effect=original)
        self.patch(o.Observation, 'failure', side_effect=RuntimeError('diagnostic failure'))
        self.patch(o.Observation, 'finish', side_effect=RuntimeError('diagnostic failure'))
        with self.assertRaises(FileNotFoundError) as got: p.main([])
        self.assertIs(got.exception, original)

    def test_started_receipt_fault_is_independently_reported(self):
        self.main_setup()
        # Undo only mocked run_once so main traverses its real started boundary.
        self.stack.close(); self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(self.stdout))
        original_once = p.run_once
        self.main_setup(); self.patch(p, 'run_once', side_effect=original_once)
        writer = self.patch(p, 'write_once', side_effect=OSError(SENTINEL))
        publish = self.patch(p, 'publish', side_effect=AssertionError('remote touched'))
        with self.assertRaises(OSError): p.main([])
        self.assertEqual(self.final()['record']['stage'], 'publication_started')
        self.assertEqual(writer.call_count, 1); publish.assert_not_called()

    def test_no_observer_direct_run_once_preserves_legacy_silence(self):
        self.patch(p, 'context', return_value=RUN)
        self.patch(p, 'write_once', return_value=None)
        self.patch(p, 'publish', return_value={'synthetic': True})
        self.assertEqual(p.run_once({}, {}, {'purpose': 'asr18'}, b'fake', None, None, self.root), {'synthetic': True})
        self.assertEqual(self.stdout.getvalue(), '')

    def test_existing_caps_and_route_constants_unchanged(self):
        self.assertEqual((p.MAX_REQUEST, p.MAX_RESPONSE, p.MAX_API_REQUESTS, p.MAX_API_BODY, p.MAX_API_UPLOAD, p.TIMEOUT),
                         (512 * 1024, 1024 * 1024, 64, 8 * 1024 ** 2, 18 * 1024 ** 2, 180))
        self.assertEqual(p._CONTEXT_KEYS, ('GITHUB_ACTIONS','GITHUB_REPOSITORY','GITHUB_EVENT_NAME','GITHUB_EVENT_PATH','GITHUB_REF',
            'GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','GITHUB_JOB','GITHUB_SHA','GITHUB_WORKFLOW_REF','GITHUB_WORKFLOW_SHA','RUNNER_TEMP','KWS_SOURCE_FREEZE_SHA256'))


if __name__ == '__main__': unittest.main(verbosity=2)
