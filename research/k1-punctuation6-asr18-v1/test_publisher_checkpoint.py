"""Offline production-wire fault matrix; invented bytes, no network or models."""
import base64
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evidence_publisher as p
import evidence_transport as t
import test_asr18_transport as fixtures

TOKEN = 'ghs_INVENTED_ONLY_TOKEN_SENTINEL'


class Response(io.BytesIO):
    def __init__(self, status, value):
        super().__init__(value if type(value) is bytes else t.canonical(value))
        self.status = status


class Wire:
    """The real production client talks only to this in-memory HTTPS fixture."""
    def __init__(self, env, event, release, source):
        self.env = env; self.event = event; self.release = release; self.source = source
        self.identity = p.context(env, event, release); self.routes = p.endpoints(self.identity, release)
        self.policy = t.policy(release['purpose']); self.calls = []; self.faults = {}
        self.refs = {}; self.blobs = {}; self.result_rows = None; self.commit = None
        self.raw_override = {}; self.raw_calls = []; self.response_reads = []
        self.repo = {'full_name': t.REPOSITORY, 'private': False}
        self.run = {'id': self.identity['run_id'], 'run_attempt': 1, 'head_sha': release['source_sha'],
            'head_branch': self.policy['source_branch'], 'event': 'push', 'path': self.policy['workflow'],
            'status': 'in_progress', 'repository': self.repo}
        self.job = {'id': 456, 'run_id': self.identity['run_id'], 'head_sha': release['source_sha'],
            'head_branch': self.policy['source_branch'], 'name': self.policy['job_key'],
            'run_url': 'https://api.github.com' + self.routes['run'], 'status': 'in_progress', 'conclusion': None}

    def connection(self, host, timeout, context):
        owner = self
        class Connection:
            def request(self, method, route, body=None, headers=None):
                self.query = (host, method, route, body, copy.deepcopy(headers), timeout)
                owner.calls.append(self.query)
            def getresponse(self):
                _, method, route, raw, _, _ = self.query
                body = None if raw is None else t.decode(raw)
                fault = owner.faults.get((method, route))
                if fault:
                    value = fault.pop(0)
                    if isinstance(value, BaseException): raise value
                    if callable(value): value = value(method, route, body)
                    status, result = value
                else: status, result = owner.handle(host, method, route, body)
                response = Response(status, result); read = response.read
                def bounded_read(cap):
                    owner.response_reads.append(cap)
                    return read(cap)
                response.read = bounded_read
                return response
            def close(self): pass
        return Connection()

    def handle(self, host, method, route, body):
        if host == 'raw.githubusercontent.com':
            self.raw_calls.append(route)
            prefix = '/' + t.REPOSITORY + '/' + self.commit['sha'] + '/'
            if not route.startswith(prefix): raise AssertionError('raw identity changed')
            name = route[len(prefix):]
            if name in self.raw_override: return 200, self.raw_override[name]
            row = next(row for row in self.result_rows if row['path'] == name)
            return 200, self.blobs[row['sha']]
        if host != 'api.github.com': raise AssertionError('host changed')
        if method == 'GET':
            if route == self.routes['repo']: return 200, self.repo
            if route == self.routes['run']: return 200, self.run
            if route == self.routes['history']: return 200, {'total_count': 1, 'workflow_runs': [self.run]}
            if route == self.routes['jobs']: return 200, {'total_count': 1, 'jobs': [self.job]}
            if route == self.routes['source']: return 200, {'sha': self.release['source_sha'], 'tree': {'sha': self.release['source_tree']}}
            if route == self.routes['source_tree']: return 200, fixtures.tree_response(self.source)
            if route.startswith(p.BASE + '/git/ref/heads/'):
                ref = 'refs/heads/' + route.split('/git/ref/heads/', 1)[1]
                return (200, self.refs[ref]) if ref in self.refs else (404, {'message': 'not found'})
            if route.startswith(p.BASE + '/git/blobs/'):
                digest = route.rsplit('/', 1)[1]
                if digest not in self.blobs: return 404, {'message': 'not found'}
                raw = self.blobs[digest]
                return 200, {'sha': digest, 'encoding': 'base64', 'size': len(raw), 'content': base64.b64encode(raw).decode() + '\n'}
            if route == p.BASE + '/git/commits/' + self.commit['sha']: return 200, self.commit
            if route == p.BASE + '/git/trees/' + self.commit['tree']['sha'] + '?recursive=1': return 200, fixtures.tree_response(self.result_rows)
        if method == 'POST':
            if route == p.BASE + '/git/blobs':
                raw = base64.b64decode(body['content'], validate=True); digest = t.git_sha(raw)
                self.blobs[digest] = raw; return 201, {'sha': digest}
            if route == p.BASE + '/git/trees':
                self.result_rows = self.source + body['tree']; return 201, fixtures.tree_response(self.result_rows)
            if route == p.BASE + '/git/commits':
                _, digest = p.commit_body(body['tree'], self.release)
                self.commit = {'sha': digest, 'message': body['message'], 'tree': {'sha': body['tree']},
                    'parents': [{'sha': value} for value in body['parents']], 'author': body['author'], 'committer': body['committer']}
                return 201, self.commit
            if route == p.BASE + '/git/refs':
                if body['ref'] in self.refs: return 422, {'message': 'already exists'}
                value = {'ref': body['ref'], 'object': {'type': 'commit', 'sha': body['sha']}}
                self.refs[body['ref']] = value; return 201, value
        raise AssertionError(('unexpected fixture request', method, route))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.env, self.event, self.release, self.source = fixtures.setup()
        self.identity = p.context(self.env, self.event, self.release)
        self.net = mock.patch.object(socket, 'socket', side_effect=AssertionError('network forbidden'))
        self.net.start(); self.addCleanup(self.net.stop)
        self.ssl = mock.patch.object(p.ssl, 'create_default_context', return_value=object())
        self.ssl.start(); self.addCleanup(self.ssl.stop)
        self.configure()

    def configure(self):
        self.wire = Wire(self.env, self.event, self.release, self.source)
        patch = mock.patch.object(p.http.client, 'HTTPSConnection', side_effect=self.wire.connection)
        patch.start(); self.addCleanup(patch.stop)
        self.api = p.GitHubAPI(TOKEN, self.identity, self.release)
        self.progress = {}

    def synthetic_files(self, chunks=2, full=False):
        size = t.MAX_CHUNK if full else 16
        return {t.manifest_path(): b'{"synthetic_fixture_only":true}',
            **{t.chunk_path(i): bytes([i]) * size for i in range(chunks)}}

    def write(self, files=None, callback=None, checkpoint_only=False, preflight=False):
        files = files or self.synthetic_files()
        rows = [{'path': name, 'type': 'blob', 'mode': '100644', 'sha': t.git_sha(raw)} for name, raw in files.items()]
        tree, _ = t.tree_identity(self.source + rows); body, commit = p.commit_body(tree, self.release)
        if preflight:
            for key in ('repo', 'run', 'jobs', 'history', 'source', 'source_tree'): self.api('GET', self.wire.routes[key])
        return p.write_checked_files(api=self.api, raw_get=lambda sha: p.RawReader(sha, 'asr18'), files=files,
            original=self.source, source_tree=self.release['source_tree'], expected_body=body, expected_commit=commit,
            result_branch=t.POLICY['result_branch'], checkpoint_branch=t.POLICY['result_branch'] + '-checkpoint',
            verify_readback=callback or (lambda values: values == files), progress=self.progress, checkpoint_only=checkpoint_only)

    def posts(self, noun=None):
        return [call for call in self.wire.calls if call[1] == 'POST' and (noun is None or call[2] == p.BASE + '/git/' + noun)]

    def test_full_checkpoint_precedes_same_commit_final_and_reads_raw_once(self):
        files = self.synthetic_files(); result = self.write(files)
        refs = [t.decode(call[3]) for call in self.posts('refs')]
        self.assertEqual([value['ref'] for value in refs], ['refs/heads/' + t.POLICY['result_branch'] + '-checkpoint', 'refs/heads/' + t.POLICY['result_branch']])
        self.assertEqual(len({value['sha'] for value in refs}), 1)
        self.assertTrue(result['checkpoint_verified'] and result['checkpoint_preserved'] and result['publication_confirmed'] and result['readback_confirmed'])
        self.assertEqual(result['raw_gets'], len(files)); self.assertEqual(len(self.wire.raw_calls), len(files))
        finalpost = self.posts('refs')[-1]
        last_raw_index = max(i for i, call in enumerate(self.wire.calls) if call[0] == 'raw.githubusercontent.com')
        self.assertLess(last_raw_index, self.wire.calls.index(finalpost))
        self.assertEqual(self.wire.commit['parents'], [{'sha': self.release['source_sha']}])

    def test_checkpoint_only_has_no_final_ref_and_never_scientific_success(self):
        result = self.write(checkpoint_only=True)
        self.assertTrue(result['checkpoint_preserved']); self.assertFalse(result['publication_confirmed'])
        self.assertFalse(result['readback_confirmed']); self.assertEqual(len(self.posts('refs')), 1)
        self.assertNotIn('schema', result); self.assertNotIn('native_success', result)

    def test_either_existing_ref_stops_before_first_post(self):
        for suffix in ('', '-checkpoint'):
            with self.subTest(suffix=suffix):
                self.configure(); name = 'refs/heads/' + t.POLICY['result_branch'] + suffix
                self.wire.refs[name] = {'ref': name, 'object': {'sha': 'e' * 40, 'type': 'commit'}}
                with self.assertRaises(p.PublicationError): self.write()
                self.assertFalse(self.posts())

    def test_callback_must_be_exact_true_before_final(self):
        for value in (False, None, 1, {'success': True}):
            with self.subTest(value=value):
                self.configure()
                with self.assertRaises(ValueError): self.write(callback=lambda _, value=value: value)
                self.assertEqual(len(self.posts('refs')), 1); self.assertFalse(self.progress['checkpoint_verified'])

    def test_corrupt_missing_extra_tree_or_checkpoint_ref_never_finalizes(self):
        for kind in ('wrong_ref', 'wrong_commit', 'wrong_tree', 'raw_byte'):
            with self.subTest(kind=kind):
                self.configure()
                if kind == 'wrong_ref':
                    self.wire.faults[('GET', self.wire.routes['checkpoint_ref'])] = [(404, {}), (200, {'ref': 'other', 'object': {'type': 'commit', 'sha': 'e' * 40}})]
                elif kind == 'raw_byte': self.wire.raw_override[t.chunk_path(0)] = b'corrupt'
                else:
                    original = self.wire.handle
                    def corrupt(host, method, route, body, kind=kind):
                        status, value = original(host, method, route, body)
                        if method == 'GET' and self.wire.commit and route == p.BASE + '/git/commits/' + self.wire.commit['sha'] and kind == 'wrong_commit':
                            value = {**value, 'parents': []}
                        if method == 'GET' and self.wire.commit and route == p.BASE + '/git/trees/' + self.wire.commit['tree']['sha'] + '?recursive=1' and kind == 'wrong_tree':
                            value = {**value, 'tree': value['tree'][:-1]}
                        return status, value
                    self.wire.handle = corrupt
                with self.assertRaises(ValueError): self.write()
                self.assertEqual(len(self.posts('refs')), 1); self.assertFalse(self.progress['checkpoint_preserved'])

    def test_failure_after_verified_checkpoint_keeps_checkpoint_proof(self):
        self.wire.faults[('POST', p.BASE + '/git/refs')] = [lambda m, r, b: self.wire.handle('api.github.com', m, r, b), (503, {})]
        with self.assertRaises(p.PublicationError): self.write()
        self.assertTrue(self.progress['checkpoint_preserved']); self.assertFalse(self.progress['publication_confirmed'])
        self.assertEqual(len(self.posts('refs')), 2)

    def test_persisted_ref_with_lost_response_is_unknown_never_false(self):
        for target in ('checkpoint', 'final'):
            with self.subTest(target=target):
                self.configure()
                def persist_then_drop(method, route, body):
                    answer = self.wire.handle('api.github.com', method, route, body)
                    checkpoint = body['ref'].endswith('-checkpoint')
                    if checkpoint == (target == 'checkpoint'): raise ConnectionResetError(TOKEN)
                    return answer
                self.wire.faults[('POST', p.BASE + '/git/refs')] = [persist_then_drop, persist_then_drop]
                with self.assertRaises(p.PublicationError): self.write()
                branch = 'refs/heads/' + t.POLICY['result_branch'] + ('-checkpoint' if target == 'checkpoint' else '')
                self.assertIn(branch, self.wire.refs)
                self.assertIsNone(self.progress['checkpoint_created'] if target == 'checkpoint' else self.progress['publication_confirmed'])
                self.assertIs(self.progress['checkpoint_verified'], target == 'final')
                self.assertIs(self.progress['checkpoint_preserved'], target == 'final')
                self.assertFalse(self.progress['readback_confirmed'])
                self.assertEqual(len(self.posts('refs')), 1 if target == 'checkpoint' else 2)

    def test_unverified_checkpoint_post_response_keeps_creation_unknown(self):
        def persist_then_wrong_response(method, route, body):
            self.wire.handle('api.github.com', method, route, body)
            return 201, {'ref': body['ref'], 'object': {'type': 'commit', 'sha': 'e' * 40}}
        self.wire.faults[('POST', p.BASE + '/git/refs')] = [persist_then_wrong_response]
        with self.assertRaises(ValueError): self.write()
        self.assertIsNone(self.progress['checkpoint_created'])
        self.assertFalse(self.progress['checkpoint_verified']); self.assertFalse(self.progress['checkpoint_preserved'])
        self.assertEqual(len(self.posts('refs')), 1)

    def test_final_get_failure_keeps_created_final_and_verified_checkpoint(self):
        self.wire.faults[('GET', self.wire.routes['ref'])] = [(404, {}), (503, {})]
        with self.assertRaises(p.PublicationError): self.write()
        self.assertTrue(self.progress['checkpoint_preserved'] and self.progress['publication_confirmed'])
        self.assertFalse(self.progress['readback_confirmed']); self.assertEqual(self.progress['stage'], 'immutable_postpublish_readback')

    def test_blob_500_502_503_504_or_no_status_found_exact_is_confirmed_without_repost(self):
        for failure in (ConnectionResetError('private exception'), (500, {}), (502, {}), (503, {}), (504, {})):
            with self.subTest(failure=type(failure).__name__ if isinstance(failure, Exception) else failure[0]):
                self.configure(); files = self.synthetic_files(); raw = files[sorted(files)[0]]; self.wire.blobs[t.git_sha(raw)] = raw
                self.wire.faults[('POST', p.BASE + '/git/blobs')] = [failure]
                result = self.write(files)
                self.assertEqual((result['blob_api_attempt_count'], result['confirmed_blob_count'], result['blob_recovery_get_count'], result['blob_retry_count']), (3, 3, 1, 0))

    def test_uncertain_blob_actual_404_reposts_exact_bytes_once(self):
        self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {})]
        result = self.write()
        posts = self.posts('blobs'); self.assertEqual(posts[0][3], posts[1][3])
        self.assertEqual((result['blob_api_attempt_count'], result['confirmed_blob_count'], result['blob_recovery_get_count'], result['blob_retry_count']), (4, 3, 1, 1))
        reads = [call for call in self.wire.calls if call[1] == 'GET' and '/git/blobs/' in call[2]]
        self.assertEqual(len(reads), 1)
        raw = base64.b64decode(t.decode(posts[0][3])['content'])
        self.assertEqual(reads[0][2], p.BASE + '/git/blobs/' + t.git_sha(raw))
        self.assertEqual(reads[0][4]['Authorization'], 'Bearer ' + TOKEN)

    def test_hostile_publication_subclass_never_reads_properties_or_recovers(self):
        class Hostile(p.PublicationError):
            @property
            def status(self): raise AssertionError('status property read')
            @property
            def body_bounded(self): raise AssertionError('body property read')
            def __str__(self): raise AssertionError('exception text read')
        error = BaseException.__new__(Hostile); BaseException.__init__(error, TOKEN)
        self.wire.faults[('POST', p.BASE + '/git/blobs')] = [error]
        with self.assertRaises(Hostile): self.write()
        self.assertEqual((self.api.blob_recovery_get_count, self.api.blob_retry_count), (0, 0))
        self.assertEqual(len(self.posts('blobs')), 1)

    def test_found_blob_uses_global_uncertainty_budget_so_second_failure_stops(self):
        files = self.synthetic_files(); raw = files[sorted(files)[0]]; self.wire.blobs[t.git_sha(raw)] = raw
        self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {}), (503, {})]
        with self.assertRaises(p.PublicationError): self.write(files)
        self.assertEqual((self.api.blob_recovery_get_count, self.api.blob_retry_count), (1, 0))
        self.assertEqual(self.progress['confirmed_blob_count'], 1)
        self.assertEqual(len(self.posts('blobs')), 2)

    def test_nonretryable_401_403_429_and_others_never_get_blob_or_retry(self):
        for status in (301, 400, 401, 403, 404, 409, 422, 429, 501):
            with self.subTest(status=status):
                self.configure(); self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(status, {})]
                with self.assertRaises(p.PublicationError): self.write()
                self.assertEqual(len(self.posts('blobs')), 1)
                self.assertEqual((self.api.blob_recovery_get_count, self.api.blob_retry_count), (0, 0))
                self.assertEqual(self.progress['confirmed_blob_count'], 0)

    def test_one_global_uncertainty_and_second_uncertain_retry_stops(self):
        for faults in ([(503, {}), (503, {})], [(503, {}), lambda m, r, b: self.wire.handle('api.github.com', m, r, b), (503, {})]):
            self.configure(); self.wire.faults[('POST', p.BASE + '/git/blobs')] = faults
            with self.assertRaises(p.PublicationError): self.write()
            self.assertEqual((self.api.blob_recovery_get_count, self.api.blob_retry_count), (1, 1))
            self.assertFalse(self.posts('trees'))

    def test_nontransport_exception_without_http_status_does_not_retry(self):
        self.wire.faults[('POST', p.BASE + '/git/blobs')] = [ValueError('private local error')]
        with self.assertRaises(p.PublicationError): self.write()
        self.assertEqual((self.api.blob_recovery_get_count, self.api.blob_retry_count), (0, 0))
        self.assertEqual(len(self.posts('blobs')), 1)

    def test_failed_recovery_get_does_not_repost(self):
        raw = self.synthetic_files()[sorted(self.synthetic_files())[0]]
        route = p.BASE + '/git/blobs/' + t.git_sha(raw)
        for value in ((401, {}), (403, {}), (429, {}), (503, {}), ConnectionResetError('private')):
            with self.subTest(value=type(value).__name__):
                self.configure(); self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {})]
                self.wire.faults[('GET', route)] = [value]
                with self.assertRaises(p.PublicationError): self.write()
                self.assertEqual(len(self.posts('blobs')), 1); self.assertEqual(self.api.blob_retry_count, 0)

    def test_recovery_blob_sha_size_encoding_and_bytes_strict(self):
        raw = self.synthetic_files()[sorted(self.synthetic_files())[0]]; digest = t.git_sha(raw)
        valid = {'sha': digest, 'encoding': 'base64', 'size': len(raw), 'content': base64.b64encode(raw).decode()}
        for field, value in (('sha', 'e' * 40), ('encoding', 'utf-8'), ('size', True), ('size', len(raw) + 1),
                ('content', base64.b64encode(b'wrong').decode()), ('content', 'invalid!'), ('content', valid['content'] + '\r')):
            with self.subTest(field=field, value=value):
                self.configure(); self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {})]
                self.wire.faults[('GET', p.BASE + '/git/blobs/' + digest)] = [(200, {**valid, field: value})]
                with self.assertRaises(ValueError): self.write()
                self.assertEqual(len(self.posts('blobs')), 1); self.assertEqual(self.api.blob_confirmed_count, 0)

    def test_exact_two_refs_only_and_each_create_consumed_once(self):
        result = self.write()
        for suffix in ('', '-checkpoint'):
            body = {'ref': 'refs/heads/' + t.POLICY['result_branch'] + suffix, 'sha': result['result_commit']}
            with self.assertRaises(ValueError): self.api('POST', p.BASE + '/git/refs', body, expected=201)
        for body in ({'ref': 'refs/heads/other', 'sha': result['result_commit']},
                {'ref': 'refs/heads/' + t.POLICY['result_branch'], 'sha': 'e' * 40}):
            with self.assertRaises(ValueError): self.api('POST', p.BASE + '/git/refs', body, expected=201)
        self.assertEqual(len(self.posts('refs')), 2)

    def test_maximum_49_files_is_65_requests_and_67_with_one_retry(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                self.configure()
                if retry: self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {})]
                result = self.write(self.synthetic_files(48, full=True), preflight=True)
                self.assertEqual(self.api.count, 67 if retry else 65)
                self.assertEqual(result['raw_gets'], 49); self.assertLessEqual(self.api.request_bytes, 18 * 1024 ** 2)
                self.assertLessEqual(self.api.body_bytes, 8 * 1024 ** 2)
                self.assertEqual(result['confirmed_blob_count'], 49)

    def test_request_response_total_body_count_deadline_caps(self):
        for attr, value in (('count', 67), ('request_bytes', p.MAX_API_UPLOAD), ('deadline', 0)):
            with self.subTest(attr=attr):
                self.configure(); setattr(self.api, attr, value)
                call = lambda: self.api('POST', p.BASE + '/git/blobs', {'content': 'YQ==', 'encoding': 'base64'}, expected=201)
                with self.assertRaises(ValueError): call()
                self.assertEqual(self.wire.calls, [])
        for status in (200, 503):
            self.configure(); self.wire.faults[('GET', p.BASE)] = [(status, b'x' * (p.MAX_RESPONSE + 1))]
            with self.assertRaises(p.PublicationError): self.api('GET', p.BASE)
            self.assertEqual(self.wire.response_reads, [p.MAX_RESPONSE + 1])
        self.configure(); self.api.body_bytes = p.MAX_API_BODY
        with self.assertRaises(ValueError): self.api('GET', p.BASE)
        self.assertEqual(self.wire.response_reads, [])
        self.configure()
        with self.assertRaises(ValueError): self.write(self.synthetic_files(49))
        self.assertFalse(self.wire.calls)

    def test_oversized_404_probe_never_releases_repost(self):
        raw = self.synthetic_files()[sorted(self.synthetic_files())[0]]
        self.wire.faults[('POST', p.BASE + '/git/blobs')] = [(503, {})]
        self.wire.faults[('GET', p.BASE + '/git/blobs/' + t.git_sha(raw))] = [(404, b'x' * (p.MAX_RESPONSE + 1))]
        with self.assertRaises(p.PublicationError): self.write()
        self.assertEqual(len(self.posts('blobs')), 1); self.assertEqual(self.api.blob_retry_count, 0)

    def test_headers_wire_bodies_fixed_and_no_credential_in_receipt(self):
        result = self.write()
        for host, method, route, body, headers, timeout in self.wire.calls:
            self.assertGreater(timeout, 0); self.assertLessEqual(timeout, 15)
            if host == 'api.github.com':
                self.assertEqual(headers, {'Authorization': 'Bearer ' + TOKEN, 'Accept': 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'k1-asr18-bounded-evidence-publisher', 'Content-Type': 'application/json'})
                if body: self.assertLessEqual(len(body), p.MAX_REQUEST)
            else:
                self.assertEqual(headers, {'User-Agent': 'k1-asr18-bounded-evidence-reader'})
            self.assertNotIn(TOKEN, route + repr(body))
        self.assertNotIn(TOKEN, repr(result) + repr(self.progress))
        for token in ('ghs_bad\n', 'ghs_bad\r', 'ghs_\x00', 'ghs_私有'):
            with self.assertRaises(ValueError): p.GitHubAPI(token, self.identity, self.release)

    def test_raw_redirect_deadline_body_and_path_caps(self):
        self.write(); commit = self.wire.commit['sha']
        for name in ('private/file', '../secret'):
            reader = p.RawReader(commit, 'asr18')
            with self.assertRaises(ValueError): reader(name)
        reader = p.RawReader(commit, 'asr18'); reader.deadline = 0
        with self.assertRaises(ValueError): reader(t.chunk_path(0))
        for size in (t.MAX_CHUNK + 1,):
            reader = p.RawReader(commit, 'asr18'); self.wire.raw_override[t.chunk_path(0)] = b'x' * size
            with self.assertRaises(p.PublicationError): reader(t.chunk_path(0))
        self.wire.raw_override = {}; reader = p.RawReader(commit, 'asr18'); reader.body_bytes = t.MAX_BODY
        with self.assertRaises(ValueError): reader(t.chunk_path(0))
        reader = p.RawReader(commit, 'asr18'); reader(t.chunk_path(0))
        with self.assertRaises(ValueError): reader(t.chunk_path(0))
        reader = p.RawReader(commit, 'asr18'); route = reader.prefix + t.chunk_path(0)
        self.wire.faults[('GET', route)] = [(302, b'')]
        with self.assertRaises(p.PublicationError): reader(t.chunk_path(0))

    def test_receipt_symlink_and_exclusive_write_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / 'target'; target.mkdir(); alias = root / 'alias'; alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError): p.write_once(alias, 'receipt.json', {'safe': True})
            leaf = target / 'receipt.json'; leaf.symlink_to(root / 'outside')
            with self.assertRaises(FileExistsError): p.write_once(target, 'receipt.json', {'safe': True})
            self.assertFalse((root / 'outside').exists())
            p.write_once(target, 'safe.json', {'safe': True})
            with self.assertRaises(FileExistsError): p.write_once(target, 'safe.json', {'safe': False})
            self.assertEqual(t.decode((target / 'safe.json').read_bytes()), {'safe': True})

    def test_final_only_scope_has_no_blob_tree_commit_or_checkpoint_posts(self):
        files = self.synthetic_files(); tree, _ = t.tree_identity(self.source + [
            {'path': name, 'mode': '100644', 'type': 'blob', 'sha': t.git_sha(raw)} for name, raw in files.items()])
        body, commit = p.commit_body(tree, self.release); api = p.GitHubAPI.__new__(p.GitHubAPI)
        api._initialize(TOKEN, get_routes=set(), source_sha=self.release['source_sha'], source_tree=self.release['source_tree'],
            result_branch=t.POLICY['result_branch'], paths={name: t.MAX_CHUNK for name in files}, expected_body=body,
            max_requests=17, final_ref_only=True)
        api.bind_result(commit, tree)
        for noun, value in (('blobs', {'encoding': 'base64', 'content': 'YQ=='}), ('trees', {}), ('commits', body),
                ('refs', {'ref': 'refs/heads/' + t.POLICY['result_branch'] + '-checkpoint', 'sha': commit})):
            with self.assertRaises(ValueError): api('POST', p.BASE + '/git/' + noun, value, expected=201)
        self.assertEqual(api.count, 0); self.assertFalse(self.wire.calls)

    def test_normal_local_controls_remain_disabled(self):
        value = t.decode((HERE / 'publication-release.json').read_bytes())
        self.assertEqual(value, p.disabled_publication_release())
        execution = t.decode((HERE / 'execution-release.json').read_bytes())
        self.assertFalse(execution['enabled'])


class WorkflowTransferScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.here = self.root / 'research/k1-punctuation6-asr18-v1'; self.here.mkdir(parents=True)
        self.actual = self.root / t.POLICY['workflow']; self.actual.parent.mkdir(parents=True)
        self.original = (HERE / 'workflow.yml').read_bytes()
        self.actual.write_bytes(self.original); (self.here / 'workflow.yml').write_bytes(self.original)
        for name, value in (('ROOT', self.root), ('HERE', self.here)):
            patch = mock.patch.object(p, name, value); patch.start(); self.addCleanup(patch.stop)

    def test_current_effective_eighty_and_historical_baseline_seventy_are_distinct(self):
        p._validate_workflow_transfer_scope()
        self.assertEqual(p.c.LIMITS['workflow_minutes'], 70)
        self.assertEqual(p.c.EFFECTIVE_EXECUTION_ENVELOPE, {'workflow_minutes': 80, 'publisher_seconds': 600, 'host_seconds': 3600})
        self.assertEqual(p.TIMEOUT, p.c.EFFECTIVE_EXECUTION_ENVELOPE['publisher_seconds'])
        control = p.disabled_publication_release()
        self.assertEqual(control['effective_execution_envelope'], p.c.EFFECTIVE_EXECUTION_ENVELOPE)
        self.assertEqual(control['scientific_campaign_baseline_minutes'], 70)
        self.assertEqual(control['limited_blob_retry_cap'], 1)

    def test_exact_step_limits_and_total_mutations_fail_even_when_both_files_match(self):
        for old, new in ((b'    timeout-minutes: 80', b'    timeout-minutes: 70'),
                (b'        timeout-minutes: 11', b'        timeout-minutes: 4'),
                (b'        timeout-minutes: 61', b'        timeout-minutes: 60'),
                (b'        timeout-minutes: 61', b'        timeout-minutes: 62'),
                (b'        timeout-minutes: 5', b'        timeout-minutes: 4'),
                (b'        timeout-minutes: 2', b'        timeout-minutes: 3')):
            with self.subTest(old=old, new=new):
                changed = self.original.replace(old, new)
                self.assertNotEqual(changed, self.original)
                self.actual.write_bytes(changed); (self.here / 'workflow.yml').write_bytes(changed)
                with self.assertRaises(ValueError): p._validate_workflow_transfer_scope()

    def test_actual_template_mismatch_rejected(self):
        self.actual.write_bytes(self.original + b'\n# unreviewed actual-only change\n')
        with self.assertRaisesRegex(ValueError, 'exact actual workflow and template bytes'):
            p._validate_workflow_transfer_scope()

    def test_duplicate_or_extra_timeout_rejected(self):
        for extra in (b'    timeout-minutes: 80\n', b'        timeout-minutes: 11\n', b'      - name: extra\n        timeout-minutes: 1\n'):
            changed = self.original + extra
            self.actual.write_bytes(changed); (self.here / 'workflow.yml').write_bytes(changed)
            with self.assertRaises(ValueError): p._validate_workflow_transfer_scope()

    def test_publication_control_requires_exact_two_level_scope(self):
        control = p.disabled_publication_release()
        for key in ('enabled', 'root_publication_release', 'independent_source_review'): control[key] = True
        p.publication_gate(control)
        for key, value in (('effective_execution_envelope', 70),
                ('effective_execution_envelope', True), ('scientific_campaign_baseline_minutes', 80),
                ('scientific_campaign_baseline_minutes', True), ('limited_blob_retry_cap', 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                p.publication_gate({**control, key: value})
        for field in ('workflow_minutes', 'publisher_seconds', 'host_seconds'):
            for value in (0, True, float(control['effective_execution_envelope'][field])):
                changed = copy.deepcopy(control); changed['effective_execution_envelope'][field] = value
                with self.assertRaises(ValueError): p.publication_gate(changed)
        for key in ('effective_execution_envelope', 'scientific_campaign_baseline_minutes'):
            changed = dict(control); del changed[key]
            with self.assertRaises(ValueError): p.publication_gate(changed)

    def test_pure_workflow_scope_accepts_disabled_flags_without_requiring_authority(self):
        control = p.disabled_publication_release()
        p.validate_transfer_workflow(self.original, control)
        with self.assertRaises(p.PublicationDisabled): p.publication_gate(control)
        changed = copy.deepcopy(control); changed['effective_execution_envelope']['publisher_seconds'] = 180
        with self.assertRaises(ValueError): p.validate_transfer_workflow(self.original, changed)
        with self.assertRaises(ValueError): p.validate_transfer_workflow(self.original.decode('ascii'), control)

    def test_bound_release_checks_workflow_only_after_disabled_gate(self):
        with mock.patch.object(p.c, 'read_json', return_value=p.disabled_publication_release()), \
                mock.patch.object(p, '_validate_workflow_transfer_scope', side_effect=AssertionError('too early')) as check:
            with self.assertRaises(p.PublicationDisabled): p.bound_release({}, {})
            check.assert_not_called()
        import ast
        tree = ast.parse((HERE / 'evidence_publisher.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'bound_release')
        names = [node.value.func.id for node in function.body if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)]
        self.assertLess(names.index('publication_gate'), names.index('_validate_workflow_transfer_scope'))



class NormalAdmissionTests(unittest.TestCase):
    configure = CheckpointTests.configure
    posts = CheckpointTests.posts
    # Use only the pre-existing scientifically shaped invented fixtures. Calling
    # the common writer directly cannot stand in for normal admission tests.
    def setUp(self):
        self.materialize = lambda: fixtures.TransportTests.materialize(self)
        fixtures.TransportTests.setUp(self)
        self.ssl = mock.patch.object(p.ssl, 'create_default_context', return_value=object())
        self.ssl.start(); self.addCleanup(self.ssl.stop)
        self.configure()

    def test_normal_context_archive_actual_run_source_binding_all_precede_post(self):
        events = []
        package = fixtures.TransportTests.package(self)
        original_archive = t.validate_archive; original_source = t.validate_source_binding; original_run = p.verify_run
        def archive(*args, **kwargs): events.append('archive'); return original_archive(*args, **kwargs)
        def source(*args, **kwargs): events.append('source'); return original_source(*args, **kwargs)
        def actual(*args, **kwargs): events.append('run'); return original_run(*args, **kwargs)
        original = self.wire.handle
        def handle(host, method, route, body):
            if method == 'POST':
                self.assertIn('archive', events); self.assertIn('run', events); self.assertIn('source', events)
                self.assertEqual([call[1] for call in self.wire.calls[:8]], ['GET'] * 8)
            return original(host, method, route, body)
        self.wire.handle = handle
        with mock.patch.object(t, 'validate_archive', side_effect=archive), mock.patch.object(t, 'validate_source_binding', side_effect=source), mock.patch.object(p, 'verify_run', side_effect=actual):
            receipt = p.publish(self.env, self.event, self.release, package, self.api, lambda sha, purpose: p.RawReader(sha, purpose))
        self.assertEqual(receipt['status'], 'same_identity_readback_verified')
        self.assertTrue(receipt['semantic_inference_success_not_established_by_publisher'])
        self.assertFalse(receipt['retry_authorized'])
        self.assertEqual(receipt['archive_sha256'], t.sha(package))

    def test_uncertain_persisted_refs_remain_null_in_durable_receipt_and_stdout_envelope(self):
        package = fixtures.TransportTests.package(self)
        for target in ('checkpoint', 'final'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                self.configure()
                def persist_then_drop(method, route, body):
                    answer = self.wire.handle('api.github.com', method, route, body)
                    checkpoint = body['ref'].endswith('-checkpoint')
                    if checkpoint == (target == 'checkpoint'): raise ConnectionResetError(TOKEN)
                    return answer
                self.wire.faults[('POST', p.BASE + '/git/refs')] = [persist_then_drop, persist_then_drop]
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(p.PublicationError):
                    p.run_once(self.env, self.event, self.release, package, self.api,
                        lambda sha, purpose: p.RawReader(sha, purpose), Path(tmp), observer=mock.Mock())
                receipt = t.decode((Path(tmp) / 'publication.receipt.json').read_bytes())
                records = [json.loads(line)['record'] for line in output.getvalue().splitlines()]
                terminal = next(row for row in records if row['status'] == 'publication_failed_or_unconfirmed_no_retry')
                self.assertEqual(t.decode((Path(tmp) / 'publication.transfer.json').read_bytes()), terminal)
                field = 'checkpoint_created' if target == 'checkpoint' else 'publication_confirmed'
                for value in (receipt, terminal):
                    self.assertIsNone(value[field])
                    self.assertIs(value['checkpoint_verified'], target == 'final')
                    self.assertIs(value['checkpoint_preserved'], target == 'final')
                    self.assertFalse(value['readback_confirmed'])
                branch = 'refs/heads/' + t.POLICY['result_branch'] + ('-checkpoint' if target == 'checkpoint' else '')
                self.assertIn(branch, self.wire.refs)
                self.assertNotIn(TOKEN, output.getvalue() + repr(receipt))

    def test_normal_bad_archive_context_run_or_source_never_posts(self):
        package = fixtures.TransportTests.package(self)
        for mode in ('context', 'archive', 'run', 'source'):
            with self.subTest(mode=mode):
                self.configure(); env = dict(self.env); payload = package
                if mode == 'context': env['GITHUB_RUN_ATTEMPT'] = '2'
                if mode == 'archive': payload = b'not a ZIP'
                if mode == 'run': self.wire.run['run_attempt'] = 2
                if mode == 'source': self.wire.source = []
                with self.assertRaises(ValueError): p.publish(env, self.event, self.release, payload, self.api, lambda sha, purpose: p.RawReader(sha, purpose))
                self.assertFalse(self.posts())

if __name__ == '__main__': unittest.main(verbosity=2)
