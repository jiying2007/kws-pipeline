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


class TransferObservationTests(unittest.TestCase):
    def progress(self, count=1):
        files={t.chunk_path(i):bytes([i%256])*32 for i in range(count)}
        files[t.manifest_path()]=b'{}'
        rows=[{'path':name,'bytes':len(raw),'sha256':t.sha(raw),'git_blob_sha':t.git_sha(raw)} for name,raw in sorted(files.items())]
        return {'stage':'create_blobs','planned_blobs':rows,'attempted_blob_count':0,'confirmed_blob_count':0,
            'archive_sha256':'a'*64,'archive_bytes':count*32,'expected_result_commit':'b'*40,'expected_result_tree':'d'*40,
            'publication_confirmed':False,'readback_confirmed':False}

    def test_plan_and_terminal_are_full_bounded_safe_records(self):
        progress=self.progress(48);out=io.StringIO()
        progress['archive_bytes']=t.MAX_ARCHIVE
        for row in progress['planned_blobs']:
            if row['path']!=t.manifest_path():row['bytes']=t.MAX_CHUNK
        with redirect_stdout(out):
            self.assertTrue(p._emit_transfer(object(),RUN,progress,'upload_plan_prepared'))
            progress['attempted_blob_count']=1
            self.assertTrue(p._emit_transfer(object(),RUN,progress,'publication_failed_or_unconfirmed_no_retry',p.PublicationError(429)))
        values=[json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(values),2)
        for v in values:
            self.assertEqual(v['schema'],'k1-asr18-publisher-transfer-envelope-v1')
            self.assertEqual(v['record_bytes'],len(t.canonical(v['record'])))
            self.assertEqual(v['record_sha256'],t.sha(t.canonical(v['record'])))
            self.assertLessEqual(v['record_bytes'],16*1024)
            self.assertEqual(len(v['record']['planned_blobs']),49)
            self.assertFalse(v['record']['native_success_established'])
        self.assertEqual(values[1]['record']['http_status'],429)
        self.assertEqual(values[1]['record']['attempted_blob_count'],1)
        self.assertEqual(values[1]['record']['confirmed_blob_count'],0)
        self.assertTrue(values[1]['record']['unconfirmed_request_may_have_written'])
        self.assertEqual(values[1]['record']['attempted_blob_count_scope'],'blob_api_invocations_not_confirmed_http_sends')
        self.assertEqual(values[1]['record']['observation_delivery'],'best_effort_stdout')

    def test_missing_observations_are_null_not_zero(self):
        r=p.transfer_record(RUN,{'stage':'validate_local_package'},'publication_failed_or_unconfirmed_no_retry',ValueError(SENTINEL))
        for key in ('planned_blobs','attempted_blob_count','confirmed_blob_count','archive_sha256','expected_result_tree','http_status'):
            self.assertIsNone(r[key])
        self.assertNotIn(SENTINEL,t.canonical(r).decode())

    def test_no_arbitrary_paths_bodies_errors_or_field_values(self):
        bad=[]
        for key,val in (('path','/private/sentinel'),('bytes',True),('sha256',SENTINEL),('git_blob_sha',SENTINEL)):
            v=self.progress();v['planned_blobs'][0][key]=val;bad.append(v)
        v=self.progress();v['planned_blobs'][0]['secret']=SENTINEL;bad.append(v)
        for key,val in (('stage',SENTINEL),('archive_bytes',True),('archive_sha256',SENTINEL),('confirmed_blob_count',1),('attempted_blob_count',True)):
            v=self.progress();v[key]=val;bad.append(v)
        for v in bad:
            out=io.StringIO()
            with self.subTest(v=v),redirect_stdout(out):
                self.assertFalse(p._emit_transfer(object(),RUN,v,'publication_failed_or_unconfirmed_no_retry',ValueError(SENTINEL)))
            self.assertEqual(out.getvalue(),'')
        class HostileError(ValueError):
            def __str__(self):raise AssertionError('rendered exception')
        r=p.transfer_record(RUN,self.progress(),'publication_failed_or_unconfirmed_no_retry',HostileError(SENTINEL))
        self.assertEqual(r['error_type'],'OtherException');self.assertNotIn(SENTINEL,t.canonical(r).decode())

    def test_local_plan_record_precedes_stdout_and_no_replace(self):
        out=io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with redirect_stdout(out):
                self.assertTrue(p._emit_transfer(object(),RUN,self.progress(),'upload_plan_prepared',directory=root))
                self.assertTrue(p._emit_transfer(object(),RUN,self.progress(),'upload_plan_prepared',directory=root))
            records=[json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual((root/'publication.plan.json').read_bytes(),t.canonical(records[0]['record']))
            self.assertEqual(records[0]['persistence']['status'],'local_exclusive_write_and_fsync_completed')
            self.assertEqual(records[1]['persistence']['status'],'local_write_failed_or_unconfirmed')
            self.assertEqual(records[1]['persistence']['error_type'],'FileExistsError')

    def test_api_local_rejection_count_does_not_claim_sent_http(self):
        progress=self.progress();progress['attempted_blob_count']=1
        r=p.transfer_record(RUN,progress,'publication_failed_or_unconfirmed_no_retry',ValueError('API deadline/count cap'))
        self.assertEqual((r['attempted_blob_count'],r['confirmed_blob_count'],r['http_status']),(1,0,None))
        self.assertEqual(r['attempted_blob_count_scope'],'blob_api_invocations_not_confirmed_http_sends')
        class EvilPublication(p.PublicationError):
            @property
            def status(self):raise AssertionError('status property evaluated')
            def __str__(self):raise AssertionError('error rendered')
        error=BaseException.__new__(EvilPublication);BaseException.__init__(error,SENTINEL)
        r=p.transfer_record(RUN,progress,'publication_failed_or_unconfirmed_no_retry',error)
        self.assertEqual(r['error_type'],'OtherException');self.assertIsNone(r['http_status'])

    def test_output_failure_is_no_throw_and_no_observer_is_silent(self):
        with mock.patch('builtins.print',side_effect=OSError(SENTINEL)):
            self.assertFalse(p._emit_transfer(object(),RUN,self.progress(),'upload_plan_prepared'))
        with mock.patch('builtins.print') as printer:
            self.assertFalse(p._emit_transfer(None,RUN,self.progress(),'upload_plan_prepared'));printer.assert_not_called()

    def test_original_failure_receipt_output_precedes_failed_local_save(self):
        out=io.StringIO();primary=p.PublicationError(503);observer=mock.Mock()
        def publish(env,event,release,payload,api,raw,progress,observer=None,transfer_directory=None):
            progress.update(self.progress());progress.update(attempted_blob_count=1)
            raise primary
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(out),mock.patch.object(p,'context',return_value=RUN),\
             mock.patch.object(p,'publish',side_effect=publish),mock.patch.object(p,'write_once',side_effect=lambda root,name,value: None if name=='publication.started.json' else (_ for _ in ()).throw(OSError(SENTINEL))):
            with self.assertRaises(p.PublicationError) as caught:
                p.run_once({}, {}, {'purpose':'asr18'},b'fixture',None,None,Path(tmp),observer=observer)
        self.assertIs(caught.exception,primary)
        values=[json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(values),1);r=values[0]['record']
        self.assertEqual((r['http_status'],r['confirmed_blob_count'],r['attempted_blob_count']),(503,0,1))
        self.assertEqual(values[0]['persistence']['status'],'local_write_failed_or_unconfirmed')
        self.assertNotIn(SENTINEL,out.getvalue());self.assertEqual(observer.failure.call_count,2)


if __name__ == '__main__': unittest.main(verbosity=2)
