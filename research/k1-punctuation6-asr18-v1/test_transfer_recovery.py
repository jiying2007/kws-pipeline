"""Offline original-checkpoint recovery tests; real ASR validators, no network/model."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer_recovery as r
import evidence_transport as t
import evidence_publisher as p
import campaign_contract as c
from test_asr18_transport import tree_response, bound_test_fixture


class FakeAPI:
    def __init__(self, fixture):
        self.fixture = fixture
        self.rows = {}; self.calls = []; self.raw_calls = []; self.final = None; self.response_bodies = {}
        self.deadline = time.monotonic() + 600
        self.routes = r.endpoints(fixture.run, fixture.source, fixture.authority)
        self.rows[self.routes['repo']] = {'full_name': t.REPOSITORY, 'private': False}
        for prefix, identity, branch, workflow, job, tree, source_rows in (
            ('recovery', fixture.run, r.SOURCE_BRANCH, r.WORKFLOW, 987, fixture.source['source_tree'], fixture.recovery_rows),
            ('producer', fixture.original, t.POLICY['source_branch'], t.POLICY['workflow'], 456, fixture.authority['source_tree'], fixture.original_rows)):
            terminal = fixture.authority['terminal'] if prefix == 'producer' else None
            run = {'id': identity['run_id'], 'run_attempt': 1, 'head_sha': identity['executed_head_sha'],
                'head_branch': branch, 'event': 'push', 'path': workflow,
                'status': 'completed' if terminal else 'in_progress', 'conclusion': terminal['run_conclusion'] if terminal else None,
                'repository': {'full_name': t.REPOSITORY, 'private': False}}
            self.rows[self.routes[prefix + '_run']] = run
            self.rows[self.routes[prefix + '_history']] = {'total_count': 1, 'workflow_runs': [copy.deepcopy(run)]}
            value = {'id': job, 'run_id': identity['run_id'], 'run_attempt': 1, 'name': 'asr18' if terminal else r.JOB_KEY,
                'head_sha': identity['executed_head_sha'], 'head_branch': branch, 'run_url': 'https://api.github.com' + self.routes[prefix + '_run'],
                'status': 'completed' if terminal else 'in_progress', 'conclusion': terminal['job_conclusion'] if terminal else None}
            if terminal:
                value['steps'] = [{'name': name, 'status': 'completed', 'conclusion': terminal[key]} for name, key in (
                    (r.HOST_STEP, 'host_step_conclusion'), (r.PUBLISHER_STEP, 'publisher_step_conclusion'))]
            self.rows[self.routes[prefix + '_jobs']] = {'total_count': 1, 'jobs': [value]}
            self.rows[self.routes[prefix + '_source']] = {'sha': identity['executed_head_sha'], 'tree': {'sha': tree},
                'committer': {'date': '2026-10-12T00:00:00Z'}}
            self.rows[self.routes[prefix + '_tree']] = tree_response(source_rows)
        if hasattr(fixture,'members') and 'admission.json' in fixture.members:
            owner=t.decode(fixture.members['admission.json'])['owner']['identity']
            run=self.rows[self.routes['producer_run']]
            run.update(c.bound_run_row(owner),head_repository={'full_name':t.REPOSITORY,'private':False})
            later=copy.deepcopy(run);later.update(id=owner['run_id']+1,run_number=2,status='queued',conclusion=None)
            route=p.producer_history_route(owner)
            self.routes['producer_history']=route
            self.rows[route]={'total_count':2,'workflow_runs':[copy.deepcopy(run),later]}
            self.rows[self.routes['producer_source']]['committer']={'date':fixture.authority['source_commit_timestamp']}
        a = fixture.authority
        self.rows[self.routes['checkpoint_ref']] = {'ref': a['checkpoint_ref'], 'object': {'type': 'commit', 'sha': a['checkpoint_commit']}}
        body, digest = p.commit_body(a['checkpoint_tree'], r.original_release(a))
        self.rows[self.routes['checkpoint_commit']] = {'sha': digest, 'message': body['message'], 'tree': {'sha': body['tree']},
            'parents': [{'sha': parent} for parent in body['parents']], 'author': body['author'], 'committer': body['committer']}
        self.rows[self.routes['checkpoint_tree']] = tree_response(fixture.complete_rows)

    def __call__(self, method, route, body=None, expected=200):
        value=self._request(method,route,body,expected)
        if value is not None:self.response_bodies[route]=json.dumps(value,indent=2).encode()+b'\n'
        return value

    def _request(self, method, route, body=None, expected=200):
        self.calls.append((method, route, copy.deepcopy(body), expected))
        if len(self.calls) > r.MAX_API_REQUESTS:
            raise AssertionError('request budget exceeded')
        if route == self.routes['final_ref']:
            if expected == 404:
                if self.final is not None: raise p.PublicationError(200)
                return None
            return self.final
        if method == 'POST':
            if route != p.BASE + '/git/refs' or body != {'ref': 'refs/heads/' + t.POLICY['result_branch'],
                    'sha': self.fixture.authority['checkpoint_commit']} or self.final is not None:
                raise AssertionError('unexpected write')
            self.final = {'ref': body['ref'], 'object': {'type': 'commit', 'sha': body['sha']}}
            return self.final
        value = self.rows[route]
        if isinstance(value, Exception): raise value
        return copy.deepcopy(value)

    def raw_factory(self, commit, *, deadline):
        if commit != self.fixture.authority['checkpoint_commit'] or deadline != self.deadline:
            raise AssertionError('nonimmutable or unbounded raw read')
        def read(name):
            if name in self.raw_calls: raise AssertionError('second full read')
            self.raw_calls.append(name)
            return self.fixture.files[name]
        return read


class RecoveryFixture:
    def __init__(self):
        bound=bound_test_fixture()
        self.members,self.original=bound['files'],bound['identity']
        self.original_rows = [{'path': name[len('source/'):], 'mode': '100644', 'type': 'blob', 'sha': t.git_sha(raw)}
            for name, raw in self.members.items() if name.startswith('source/')]
        source_tree = t.tree_identity(self.original_rows)[0]
        host = t.decode(self.members['host-receipt.json']); host['executed_tree_sha'] = source_tree
        self.members['host-receipt.json'] = t.canonical(host)
        self.payload = t._zip_files(self.members, self.original)[0]
        freeze = t.sha(self.members['authority/execution-freeze.json'])
        self.files = t.build_file_set(self.payload, self.original, freeze, 456)
        self.complete_rows = self.original_rows + [{'path': name, 'mode': '100644', 'type': 'blob', 'sha': t.git_sha(raw)}
            for name, raw in sorted(self.files.items())]
        self.authority = {'schema': 'k1-asr18-transfer-recovery-authority-v1', 'producer_identity': self.original,
            'numeric_job_id': 456, 'source_tree': source_tree, 'source_closure_sha256': freeze,
            'source_commit_timestamp': bound['source_commit_timestamp'], 'archive_sha256': t.sha(self.payload),
            'archive_bytes': len(self.payload), 'outer_manifest_sha256': t.sha(self.files[t.manifest_path()]),
            'checkpoint_ref': 'refs/heads/' + t.POLICY['result_branch'] + '-checkpoint', 'checkpoint_commit': None,
            'checkpoint_tree': t.tree_identity(self.complete_rows)[0], 'terminal': {'run_conclusion': 'failure',
                'job_conclusion': 'failure', 'host_step_conclusion': 'failure', 'publisher_step_conclusion': 'failure',
                'host_receipt_status': 'stopped_no_retry'}}
        self.authority['checkpoint_commit'] = p.commit_body(self.authority['checkpoint_tree'], r.original_release(self.authority))[1]
        self.control = r.disabled_release()
        for key in r.APPROVALS: self.control[key] = True
        self.control['authority'] = self.authority
        self.recovery_rows = [{'path': r.WORKFLOW, 'mode': '100644', 'type': 'blob', 'sha': t.git_sha(b'# invented fixture')}]
        self.source = {'source_sha': 'f' * 40, 'source_tree': t.tree_identity(self.recovery_rows)[0],
            'source_closure_sha256': 'a' * 64, 'commit_timestamp': '2026-10-12T00:00:00Z'}
        self.env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': t.REPOSITORY, 'GITHUB_EVENT_NAME': 'push',
            'GITHUB_REF': 'refs/heads/' + r.SOURCE_BRANCH, 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '789',
            'GITHUB_SHA': self.source['source_sha'], 'GITHUB_JOB': r.JOB_KEY,
            'GITHUB_WORKFLOW_REF': t.REPOSITORY + '/' + r.WORKFLOW + '@refs/heads/' + r.SOURCE_BRANCH,
            'GITHUB_WORKFLOW_SHA': self.source['source_sha'], 'RUNNER_ENVIRONMENT': 'github-hosted',
            'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
        self.event = {'repository': {'full_name': t.REPOSITORY, 'private': False}, 'ref': self.env['GITHUB_REF'],
            'created': True, 'deleted': False, 'forced': False, 'before': '0' * 40, 'after': self.source['source_sha']}
        self.run = r.context(self.env, self.event, self.control)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch('http.client.HTTPSConnection', side_effect=AssertionError('no live network')):
            cls.fixture = RecoveryFixture()

    def setUp(self):
        self.f = copy.deepcopy(self.fixture)
        self.api = FakeAPI(self.f)
        self.guard = mock.patch('http.client.HTTPSConnection', side_effect=AssertionError('no live network'))
        self.guard.start(); self.addCleanup(self.guard.stop)

    def recover(self):
        return r.recover(self.f.env, self.f.event, self.f.control, self.f.source, self.api, self.api.raw_factory)

    def no_posts(self):
        self.assertFalse(any(call[0] == 'POST' for call in self.api.calls))

    def test_original_real_archive_native_source_validators_and_only_final_post(self):
        with mock.patch.object(t, 'validate_archive', wraps=t.validate_archive) as archive, \
             mock.patch.object(t, 'validate_source_binding', wraps=t.validate_source_binding) as binding, \
             mock.patch.object(t._host(), 'validate_public_evidence', wraps=t._host().validate_public_evidence) as native:
            receipt = self.recover()
        self.assertGreaterEqual(archive.call_count, 1); self.assertEqual(binding.call_count, 1); self.assertGreater(native.call_count, 0)
        self.assertTrue(all(call.args[1] == self.f.original for call in archive.call_args_list))
        self.assertEqual(binding.call_args.args[1], self.f.original)
        self.assertEqual(binding.call_args.args[4], 456)
        self.assertEqual(len(self.api.calls), 17)
        self.assertEqual([call[1] for call in self.api.calls if call[0] == 'POST'], [p.BASE + '/git/refs'])
        self.assertEqual(set(self.api.raw_calls), set(self.f.files))
        self.assertEqual(len(self.api.raw_calls), len(self.f.files))
        self.assertEqual(receipt['original_terminal'], self.f.authority['terminal'])
        self.assertEqual(receipt['original_run_status'], 'completed')
        self.assertEqual(receipt['original_job_status'], 'completed')
        self.assertEqual(receipt['producer_identity'], self.f.original)
        self.assertEqual(receipt['recovery_identity']['job_key'], r.JOB_KEY)
        for key in ('original_publisher_reclassified', 'native_success_established', 'new_native_calls_authorized', 'retry_authorized'):
            self.assertIs(receipt[key], False)
        self.assertNotEqual(receipt['status'], 'same_identity_readback_verified')
        self.assertNotIn('payload', receipt)

    def test_disabled_null_pins_old_lost_sealed_and_unknown_authorities_rejected(self):
        for mutation in ('disabled', 'null', 'missing', 'old_run', 'old_source', 'lost_r2_run', 'lost_r2_source', 'attempt2', 'wrong_ref', 'wrong_commit', 'successful_publisher'):
            control = copy.deepcopy(self.f.control)
            if mutation == 'disabled': control = r.disabled_release()
            elif mutation == 'null': control['authority'] = None
            elif mutation == 'missing': del control['authority']['archive_sha256']
            elif mutation == 'old_run': control['authority']['producer_identity']['run_id'] = next(iter(c.SEALED_ASR_RUNS))
            elif mutation == 'old_source': control['authority']['producer_identity']['executed_head_sha'] = next(iter(c.SEALED_ASR_SOURCES))
            elif mutation == 'lost_r2_run': control['authority']['producer_identity']['run_id'] = r.LOST_R2_RUN
            elif mutation == 'lost_r2_source': control['authority']['producer_identity']['executed_head_sha'] = r.LOST_R2_SOURCE
            elif mutation == 'attempt2': control['authority']['producer_identity']['run_attempt'] = 2
            elif mutation == 'wrong_ref': control['authority']['checkpoint_ref'] = 'refs/heads/foreign-checkpoint'
            elif mutation == 'wrong_commit': control['authority']['checkpoint_commit'] = '1' * 40
            elif mutation == 'successful_publisher': control['authority']['terminal']['publisher_step_conclusion'] = 'success'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): r.recovery_gate(control)
        self.no_posts()

    def test_successful_host_requires_complete_receipt_but_failed_host_need_not(self):
        for status in ('stopped_no_retry', 'cleanup_unconfirmed'):
            value = copy.deepcopy(self.f.authority)
            value['terminal'].update(host_step_conclusion='success', host_receipt_status=status)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'complete native host receipt'):
                r.validate_authority(value)
        value['terminal']['host_receipt_status'] = 'all_planned_native_complete'
        r.validate_authority(value)
        for status in ('stopped_no_retry', 'cleanup_unconfirmed', 'all_planned_native_complete'):
            value['terminal'].update(host_step_conclusion='failure', host_receipt_status=status)
            r.validate_authority(value)

    def test_foreign_recovery_context_attempt2_and_original_identity_rejected_without_api(self):
        for key, value in (('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_JOB', 'asr18'), ('GITHUB_REF', 'refs/heads/foreign'),
            ('GITHUB_WORKFLOW_SHA', '1' * 40), ('GITHUB_RUN_ID', str(self.f.original['run_id'])),
            ('GITHUB_SHA', self.f.original['executed_head_sha']), ('GITHUB_REPOSITORY', 'elsewhere/repo'), ('GITHUB_EVENT_NAME', 'workflow_dispatch')):
            original = self.f.env[key]; self.f.env[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.recover()
            self.f.env[key] = original
        self.assertEqual(self.api.calls, [])

    def test_missing_checkpoint_and_checkpoint_target_or_tree_drift_stop(self):
        for kind in ('missing', 'target', 'commit_parent', 'tree'):
            self.api = FakeAPI(self.f)
            if kind == 'missing': self.api.rows[self.api.routes['checkpoint_ref']] = p.PublicationError(404)
            elif kind == 'target': self.api.rows[self.api.routes['checkpoint_ref']]['object']['sha'] = '1' * 40
            elif kind == 'commit_parent': self.api.rows[self.api.routes['checkpoint_commit']]['parents'] = [{'sha': self.f.run['executed_head_sha']}]
            elif kind == 'tree': self.api.rows[self.api.routes['checkpoint_tree']]['truncated'] = True
            with self.subTest(kind=kind), self.assertRaises((ValueError, p.PublicationError)): self.recover()
            self.no_posts()

    def test_original_terminal_history_step_job_or_run_drift_stop(self):
        for prefix, suffix, mutation in (('producer', 'run', 'attempt'), ('producer', 'run', 'head'),
            ('producer', 'history', 'foreign'), ('recovery', 'history', 'extra'), ('producer', 'jobs', 'job'),
            ('producer', 'jobs', 'publisher'), ('producer', 'jobs', 'host'), ('recovery', 'run', 'terminal')):
            self.api = FakeAPI(self.f)
            record = self.api.rows[self.api.routes[prefix + '_' + suffix]]
            if mutation == 'attempt': record['run_attempt'] = 2
            elif mutation == 'head': record['head_sha'] = '1' * 40
            elif mutation == 'foreign': record['workflow_runs'][0]['id'] += 1
            elif mutation == 'extra': record['total_count'] = 2
            elif mutation == 'job': record['jobs'][0]['id'] += 1
            elif mutation == 'publisher': record['jobs'][0]['steps'][1]['conclusion'] = 'success'
            elif mutation == 'host': record['jobs'][0]['steps'][0]['status'] = 'in_progress'
            elif mutation == 'terminal': record['status'] = 'completed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.recover()
            self.no_posts()

    def test_original_source_body_mismatch_is_rejected_by_real_source_validator(self):
        rows = copy.deepcopy(self.f.original_rows)
        rows[0]['sha'] = '1' * 40
        wrong = tree_response(rows)
        self.f.authority['source_tree'] = wrong['sha']
        self.api = FakeAPI(self.f)
        self.api.rows[self.api.routes['producer_tree']] = wrong
        with self.assertRaisesRegex(ValueError, 'archived public source equals actual Git source'):
            self.recover()
        self.no_posts()

    def test_existing_final_even_same_target_is_never_reused_or_updated(self):
        for target in (self.f.authority['checkpoint_commit'], '1' * 40):
            self.api = FakeAPI(self.f)
            self.api.final = {'ref': 'refs/heads/' + t.POLICY['result_branch'], 'object': {'type': 'commit', 'sha': target}}
            with self.assertRaises(p.PublicationError): self.recover()
            self.no_posts()

    def test_missing_or_modified_chunk_and_wrong_outer_stop(self):
        for kind in ('chunk', 'outer', 'missing'):
            self.api = FakeAPI(self.f)
            original = self.f.files.copy()
            if kind == 'chunk': self.f.files[t.chunk_path(0)] += b'corruption'
            elif kind == 'outer': self.f.files[t.manifest_path()] += b'\n'
            else: del self.f.files[t.chunk_path(0)]
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)): self.recover()
            self.no_posts(); self.f.files = original

    def test_private_payload_uses_actual_archive_rejection_before_post(self):
        members = dict(self.f.members, **{'private/audio.wav': b'RIFFxxxxWAVEprivate'})
        payload, member_manifest, expanded = t._zip_files(members, self.f.original)
        outer = t.decode(self.f.files[t.manifest_path()])
        chunks = [payload[i:i + t.MAX_CHUNK] for i in range(0, len(payload), t.MAX_CHUNK)]
        outer.update(archive_bytes=len(payload), archive_sha256=t.sha(payload), member_count=len(members),
            manifest_sha256=t.sha(t.canonical(member_manifest)), uncompressed_bytes=expanded, chunk_count=len(chunks),
            chunks=[{'index': i, 'path': t.chunk_path(i), 'bytes': len(raw), 'sha256': t.sha(raw)} for i, raw in enumerate(chunks)])
        outer_raw = t.canonical(outer)
        self.f.files = {t.manifest_path(): outer_raw, **{t.chunk_path(i): raw for i, raw in enumerate(chunks)}}
        self.f.authority.update(archive_bytes=len(payload), archive_sha256=t.sha(payload), outer_manifest_sha256=t.sha(outer_raw))
        with self.assertRaisesRegex(ValueError, 'audio bytes prohibited'): self.recover()
        self.no_posts()

    def test_final_unknown_write_is_not_retried(self):
        api = self.api
        def uncertain(method, route, body=None, expected=200):
            result = api(method, route, body, expected)
            if method == 'POST': raise p.PublicationError(504)
            return result
        uncertain.deadline = api.deadline
        uncertain.response_bodies = api.response_bodies
        with self.assertRaises(p.PublicationError):
            r.recover(self.f.env, self.f.event, self.f.control, self.f.source, uncertain, api.raw_factory)
        self.assertEqual(sum(call[0] == 'POST' for call in api.calls), 1)
        self.assertIsNotNone(api.final)

    def test_recovery_http_is_shared_exact17_bounded_no_other_writes(self):
        api = r.RecoveryAPI('ghs_TEST_ONLY', self.f.run, self.f.source, self.f.control)
        self.assertEqual(api.max_requests, 17); self.assertEqual(api.max_body, 8 * 1024 ** 2)
        self.assertEqual(api.max_upload, 512 * 1024); self.assertTrue(api.final_ref_only)
        self.assertEqual({route for method, route in api.allowed if method == 'POST'}, {p.BASE + '/git/refs'})
        for route, body in ((p.BASE + '/git/blobs', {'content': '', 'encoding': 'base64'}),
                (p.BASE + '/git/refs', {'ref': self.f.authority['checkpoint_ref'], 'sha': self.f.authority['checkpoint_commit']}),
                (p.BASE + '/git/refs', {'ref': 'refs/heads/' + t.POLICY['result_branch'], 'sha': '1' * 40})):
            with self.assertRaises(ValueError): api('POST', route, body, expected=201)
        api.count = 17
        with self.assertRaises(ValueError): api('GET', p.BASE)
        self.assertEqual(r.MAX_RAW_BODY, 32 * 1024 ** 2); self.assertEqual(r.TIMEOUT, 600)


class RecoveryEntryTests(unittest.TestCase):
    def test_real_cli_disabled_safe_failure_all_optimization_modes(self):
        self.assertEqual(c.read_json(HERE / 'transfer-recovery-release.json'), r.disabled_release())
        self.assertFalse((HERE.parents[1] / r.WORKFLOW).exists())
        for optimize in ([], ['-O'], ['-OO']):
            result = subprocess.run([sys.executable, '-I', '-B', *optimize, str(HERE / 'transfer_recovery.py')],
                capture_output=True, timeout=15, env={'PATH': '/usr/bin:/bin', 'GITHUB_TOKEN': 'ghs_PRIVATE_SENTINEL'})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'ASR transfer recovery disabled or review absent.\n')

    def test_disabled_gate_has_no_environment_token_process_network_or_output(self):
        class Forbidden(dict):
            def __getitem__(self, key): raise AssertionError('environment before gate')
            def get(self, *args): raise AssertionError('environment before gate')
            def __contains__(self, key): raise AssertionError('environment before gate')
        output = io.StringIO()
        with mock.patch.object(r.os, 'environ', Forbidden()), mock.patch('subprocess.Popen', side_effect=AssertionError('process')), \
                mock.patch('http.client.HTTPSConnection', side_effect=AssertionError('network')), redirect_stdout(output):
            with self.assertRaises(r.RecoveryDisabled): r.main([])
        self.assertEqual(output.getvalue(), '')

    def test_disabled_missing_authority_and_wrong_context_never_access_token(self):
        fixture = RecoveryFixture()
        class TokenGuard(dict):
            def get(self, key, *args):
                if key == 'GITHUB_TOKEN': raise AssertionError('token read before admission')
                return super().get(key, *args)
        for failure in ('null_authority', 'foreign_job'):
            control = copy.deepcopy(fixture.control)
            environment = dict(fixture.env)
            if failure == 'null_authority': control['authority'] = None
            else: environment['GITHUB_JOB'] = 'asr18'
            with self.subTest(failure=failure), mock.patch.object(c, 'read_json', side_effect=[control, fixture.event]), \
                    mock.patch.object(r.sys, 'flags', type('Flags', (), {'isolated': True})()), \
                    mock.patch.object(r.sys, 'dont_write_bytecode', True), mock.patch.object(r.os, 'environ', TokenGuard(environment)), \
                    mock.patch.object(p, '_source_git', side_effect=AssertionError('Git before context')), \
                    mock.patch.object(r, 'RecoveryAPI', side_effect=AssertionError('token admitted')):
                with self.assertRaises(ValueError): r.main([])

    def test_context_and_dirty_source_before_token_even_with_enabled_mock_control(self):
        fixture = RecoveryFixture()
        with mock.patch.object(c, 'read_json', side_effect=[fixture.control, fixture.event]), \
                mock.patch.object(r.sys, 'flags', type('Flags', (), {'isolated': True})()), mock.patch.object(r.sys, 'dont_write_bytecode', True), \
                mock.patch.dict(r.os.environ, fixture.env, clear=True), \
                mock.patch.object(p, '_source_git', return_value='dirty'), \
                mock.patch.object(r, 'RecoveryAPI', side_effect=AssertionError('token admitted')) as api:
            with self.assertRaisesRegex(ValueError, 'clean recovery source'): r.main([])
            api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
