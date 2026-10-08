"""Offline API-budget regressions using real GitHub read/pagination/cache code.

Coordinator fixtures replace only request(). Diagnostic fixtures instead replace
the urllib opener, exercising the real request/body/header/error paths. Git object
restoration and the one Git write boundary are synthetic; no network is used.
"""
from collections import Counter
from contextlib import redirect_stdout
import copy
from datetime import datetime, timezone
from email.message import Message
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import urllib.error
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'git_atomic_prune_20261008.py').exists():
    sys.path.insert(0, str(ROOT.parent / 'tools'))
import archive_branches_once_20261007 as A
import git_atomic_prune_20261008 as G
import test_archive_branches_once_20261007 as LEGACY


class RequestFixture(A.GitHub):
    """Exercise every inherited client method; replace only its HTTP request seam."""
    def __init__(self, plan, own='kws-pipeline', remaining=60, extra_tags=130):
        super().__init__('jiying2007/' + own, 'offline-fixture-not-a-credential')
        self.plan, self.own = plan, own
        self.remaining, self.limit = remaining, 60
        self.calls = []
        self.phase = 'before_push'
        self.rate_checks = 0
        self.prepush_forced_remaining = None
        self.external_quota_spend = 0
        self.refs = {name: {row['name']: row['before_oid'] for row in A.frozen_rows(repo)}
                     for name, repo in plan['repositories'].items()}
        for refs in self.refs.values():
            refs.update({'refs/heads/main': LEGACY.SOURCE, 'refs/heads/unrelated': '8' * 40})
        self.tags = {name: {A.TAG: repo['archive_commit_oid'], **{
            f'refs/tags/unrelated-{n:03}': '7' * 40 for n in range(extra_tags)}}
            for name, repo in plan['repositories'].items()}
        self.policies = {name: copy.deepcopy(repo['protection_snapshot']) for name, repo in plan['repositories'].items()}
        self.notes = {name: {'refs/notes/unrelated': '6' * 40} for name in plan['repositories']}

    def request(self, method, path, payload=None, authenticated=True):
        if method != 'GET' or payload is not None:
            raise AssertionError('HTTP mutation attempted in offline fixture')
        if authenticated and not (path == '/repos/' + self.repository or path.startswith('/repos/' + self.repository + '/')):
            raise AssertionError('token addressed a foreign repository')
        self.calls.append({'method': method, 'path': path, 'authenticated': authenticated, 'phase': self.phase})
        if path == '/rate_limit':
            if authenticated:
                raise AssertionError('public rate check used token')
            self.rate_checks += 1
            if self.rate_checks == 2 and self.prepush_forced_remaining is not None:
                self.external_quota_spend += self.remaining - self.prepush_forced_remaining
                self.remaining = self.prepush_forced_remaining
            return {'resources': {'core': {'remaining': self.remaining, 'limit': self.limit, 'reset': 1800000000}}}
        if not authenticated:
            self.remaining -= 1
            if self.remaining < 0:
                raise A.ApiError(403, [])
        parsed = urlsplit(path)
        query = parse_qs(parsed.query)
        parts = parsed.path.split('/')
        if len(parts) < 4 or parts[1:3] != ['repos', 'jiying2007']:
            raise AssertionError('unexpected fixture route ' + path)
        name = parts[3]
        repo = self.plan['repositories'][name]
        suffix = '/' + '/'.join(parts[4:]) if len(parts) > 4 else ''
        if suffix == '':
            value = {'id': int(repo['repository_id']), 'owner': {'id': 33591504}, 'full_name': repo['full_name'],
                     'default_branch': 'main', 'visibility': 'public', 'archived': False, 'node_id': 'fixture-' + name}
        elif suffix == '/branches/main':
            value = {'commit': {'sha': self.refs[name]['refs/heads/main']},
                     'protected': self.policies[name]['branch_protected'],
                     'protection': self.policies[name]['legacy_protection_summary']}
        elif suffix == '/branches':
            rows = [{'name': ref.removeprefix('refs/heads/'), 'commit': {'sha': oid}} for ref, oid in self.refs[name].items()]
            page = int(query.get('page', ['1'])[0]); size = int(query.get('per_page', ['100'])[0])
            value = rows[(page - 1) * size:page * size]
        elif suffix == '/rulesets':
            rows = self.policies[name]['rulesets']
            page = int(query.get('page', ['1'])[0]); size = int(query.get('per_page', ['100'])[0])
            value = [{'id': row['id']} for row in rows[(page - 1) * size:page * size]]
        elif suffix.startswith('/rulesets/'):
            value = next(row for row in self.policies[name]['rulesets'] if str(row['id']) == suffix.split('/')[-1])
        elif suffix == '/pulls':
            value = []
        elif suffix == '/actions/runs':
            if query.get('event') == ['push']:
                value = {'workflow_runs': [
                    {'id': 1000 + n, 'path': workflow, 'head_sha': LEGACY.SOURCE, 'head_branch': 'main',
                     'event': 'push', 'run_number': 10, 'run_attempt': 1, 'status': 'completed', 'conclusion': 'success'}
                    for n, workflow in enumerate(repo['required_success_workflows'])]}
            else:
                value = {'workflow_runs': []}
        elif suffix in ('/git/matching-refs/', '/git/matching-refs/tags/'):
            if query:
                raise AssertionError('matching-refs is nonpaginated: ' + path)
            inventory = self.tags[name] if suffix.endswith('/tags/') else {**self.refs[name], **self.tags[name], **self.notes[name]}
            value = [{'ref': ref, 'object': {'type': 'commit', 'sha': oid}} for ref, oid in inventory.items()]
        elif suffix.startswith('/git/ref/'):
            ref = 'refs/' + suffix.removeprefix('/git/ref/')
            value = {'ref': ref, 'object': {'type': 'commit', 'sha': self.tags[name][ref]}}
        elif suffix.startswith('/git/commits/'):
            value = {'sha': repo['archive_commit_oid'], 'tree': {'sha': repo['archive_tree_oid']},
                     'parents': [{'sha': oid} for oid in repo['ordered_archive_parents']]}
        elif suffix.startswith('/git/trees/'):
            value = {'truncated': False, 'tree': [
                {'path': row['path'], 'mode': row['mode'], 'type': row['type'], 'sha': row['git_blob_oid']}
                for row in repo['archive_docs']]}
        elif suffix.startswith('/git/blobs/'):
            value = {'encoding': 'base64', 'content': 'b2ZmbGluZSBmaXh0dXJl'}
        else:
            raise AssertionError('unexpected fixture route ' + path)
        return copy.deepcopy(value)

    def counts(self, phase=None):
        calls = [call for call in self.calls if phase is None or call['phase'] == phase]
        return {'all_requests': len(calls),
                'public_charged': sum(not call['authenticated'] and call['path'] != '/rate_limit' for call in calls),
                'authenticated': sum(call['authenticated'] for call in calls),
                'rate_checks': sum(call['path'] == '/rate_limit' for call in calls)}


class RestorationFixture:
    def __init__(self):
        self.bares = {'jiying2007/' + name: Path('/offline-unreachable-' + name) for name in A.REPOS}
    def __call__(self, repo, plan):
        return LEGACY.proof(repo, plan)


class BudgetCoordinatorTests(unittest.TestCase):
    def make(self, own='kws-pipeline', remaining=60, extra_tags=130):
        plan = LEGACY.ready()
        context = LEGACY.context(own, 'git-prune')
        api = RequestFixture(plan, own, remaining, extra_tags)
        coordinator = A.Coordinator(plan, context, api, RestorationFixture())
        # The frozen archive document bodies are intentionally not duplicated in
        # this API-quota fixture. Exercise exactly the same immutable object reads
        # as verify_server_archive: one commit, one tree, and two pinned blobs.
        def verify_server_archive(repo):
            api.read(repo['full_name'], '/git/commits/' + repo['archive_commit_oid'])
            api.read(repo['full_name'], '/git/trees/' + repo['archive_tree_oid'])
            for row in repo['archive_docs']:
                api.read(repo['full_name'], '/git/blobs/' + row['git_blob_oid'])
        coordinator.verify_server_archive = verify_server_archive
        attestation = {'contract': 'git-atomic-prune-owner-attested-v1', 'own_repository': context['repository'],
                       'source_sha': LEGACY.SOURCE, 'observed_at': datetime.now(timezone.utc).isoformat(),
                       'repository_ids': {name: r['repository_id'] for name, r in plan['repositories'].items()},
                       'main_oids': {name: LEGACY.SOURCE for name in plan['repositories']},
                       'branch_inventories_sha256': {name: A.sha256(A.canonical(refs)) for name, refs in api.refs.items()},
                       'anchor_oids': {name: None for name in plan['repositories']},
                       'policies_sha256': A.sha256(A.canonical({name: r['protection_snapshot'] for name, r in plan['repositories'].items()}))}
        env = {'ARCHIVE_CTX_OWNER_POLICY_ATTESTATION': json.dumps(attestation),
               'ATOMIC_GUARD_SHA256': 'a' * 64, 'ATOMIC_ASKPASS_SHA256': 'b' * 64}
        return plan, api, coordinator, env

    def execute(self, own='kws-pipeline', remaining=60, force_pre=None):
        plan, api, coordinator, env = self.make(own, remaining)
        api.prepush_forced_remaining = force_pre
        before_push = {}
        def push(bare, repo, token):
            before_push.update(api.counts())
            before_push['remaining'] = api.remaining
            api.phase = 'after_push'
            coordinator.git_transport.invoked = True
            coordinator.git_transport.acknowledged = True
            for row in repo['delete_refs']:
                del api.refs[own][row['name']]
            api.tags[own][G.ANCHOR] = repo['archive_commit_oid']
            return {'status': 'OFFLINE_FIXTURE_ACKNOWLEDGED'}
        # Fixture observations must not look like real runtime quota receipts in CI logs.
        with patch.dict(os.environ, env), patch.object(G.GitAtomicTransport, 'push', side_effect=push) as mocked, redirect_stdout(io.StringIO()):
            try:
                result = G.run_git_prune(coordinator, 'offline-fixture-not-a-credential', ROOT)
            except BaseException:
                self.last_api, self.last_push_count = api, mocked.call_count
                raise
        return result, api, before_push, mocked.call_count

    def test_both_own_modes_with_real_client_reads_fit_public_60_budget(self):
        observed = {}
        for own in A.REPOS:
            with self.subTest(own=own):
                result, api, pre, count = self.execute(own)
                self.assertEqual(count, 1)
                self.assertEqual(result['status'], 'OWNER_POSTCHECK_REQUIRED')
                self.assertGreaterEqual(pre['remaining'], 30)
                self.assertEqual(pre['remaining'], 60 - pre['public_charged'])
                self.assertLessEqual(api.counts()['public_charged'], 60)
                self.assertGreaterEqual(api.remaining, 0)
                self.assertEqual(result['public_api_budgets']['initial']['remaining'], 60)
                self.assertEqual(result['public_api_budgets']['prepush']['remaining'], pre['remaining'])
                observed[own] = {'before_push': pre, 'after_push': api.counts('after_push'),
                                 'total': api.counts(), 'final_public_remaining': api.remaining}
                # >100 tags proves matching-refs must be a single unpaginated
                # request per snapshot, not an infinite 100-item page loop.
                matching = [call for call in api.calls if '/git/matching-refs/' in call['path']]
                self.assertTrue(matching)
                self.assertTrue(all('?' not in call['path'] for call in matching))
                counts = Counter((call['phase'], call['path']) for call in matching)
                self.assertLessEqual(max(counts.values()), 4)
                self.assertTrue(all(len(tags) > 100 for tags in api.tags.values()))
                object_reads = [call for call in api.calls if any('/git/' + kind + '/' in call['path'] for kind in ('commits', 'trees', 'blobs'))]
                self.assertEqual(len(object_reads), 8)
                self.assertTrue(all(call['phase'] == 'before_push' for call in object_reads))
        print('OFFLINE_PUBLIC_API_COUNTS=' + json.dumps(observed, sort_keys=True))

    def test_exact_initial_and_prewrite_budget_floors_are_accepted(self):
        for own in A.REPOS:
            with self.subTest(own=own):
                result, api, pre, count = self.execute(own, remaining=58, force_pre=30)
                self.assertEqual(count, 1)
                self.assertEqual(result['public_api_budgets']['initial']['remaining'], 58)
                self.assertEqual(result['public_api_budgets']['prepush']['remaining'], 30)
                self.assertEqual(pre['remaining'], 30)
                self.assertGreaterEqual(api.remaining, 0)

    def test_initial_budget_below_58_stops_before_any_push_or_other_read(self):
        with patch('builtins.print'):
            with self.assertRaises(A.Stop):
                self.execute(remaining=57)
        self.assertEqual(self.last_push_count, 0)
        self.assertEqual(self.last_api.counts(), {'all_requests': 1, 'public_charged': 0, 'authenticated': 0, 'rate_checks': 1})

    def test_prewrite_budget_below_30_stops_before_push(self):
        for own in A.REPOS:
            with self.subTest(own=own), patch('builtins.print'):
                with self.assertRaises(A.Stop):
                    self.execute(own, force_pre=29)
                self.assertEqual(self.last_push_count, 0)
                self.assertEqual(self.last_api.remaining, 29)
                self.assertEqual(self.last_api.rate_checks, 2)
                self.assertGreater(self.last_api.external_quota_spend, 0)


class ImmutableReadCacheTests(unittest.TestCase):
    def setUp(self):
        self.plan = LEGACY.ready()
        self.api = RequestFixture(self.plan)
        self.peer = 'jiying2007/kws-data'
        self.repo = self.plan['repositories']['kws-data']

    def test_exact_oid_commits_trees_blobs_cache_once_and_return_deep_copies(self):
        suffixes = ['/git/commits/' + self.repo['archive_commit_oid'], '/git/trees/' + self.repo['archive_tree_oid']]
        suffixes += ['/git/blobs/' + row['git_blob_oid'] for row in self.repo['archive_docs']]
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                before = len(self.api.calls)
                first = self.api.read(self.peer, suffix)
                first['injected'] = {'malicious': True}
                second = self.api.read(self.peer, suffix)
                self.assertNotIn('injected', second)
                if 'tree' in second and isinstance(second['tree'], dict):
                    first['tree']['sha'] = 'f' * 40
                    self.assertNotEqual(self.api.read(self.peer, suffix)['tree']['sha'], 'f' * 40)
                self.assertEqual(len(self.api.calls) - before, 1)

    def test_cache_is_scoped_to_repository_and_exact_lowercase_oid_route(self):
        oid = self.repo['archive_commit_oid']
        suffix = '/git/commits/' + oid
        self.api.read(self.peer, suffix)
        self.api.read('jiying2007/kws-pipeline', suffix)
        self.assertEqual(len(self.api.calls), 2)
        for suffix in ('/git/commits/main', '/git/commits/' + oid.upper(),
                       '/git/trees/' + self.repo['archive_tree_oid'] + '?recursive=1',
                       '/git/blobs/' + self.repo['archive_docs'][0]['git_blob_oid'] + '/'):
            with self.subTest(suffix=suffix):
                before = len(self.api.calls)
                self.api.read(self.peer, suffix)
                self.api.read(self.peer, suffix)
                self.assertEqual(len(self.api.calls) - before, 2)

    def test_mutable_refs_and_matching_inventories_are_always_fresh(self):
        suffix = '/git/ref/tags/archive/branches-2026-10-07'
        first = self.api.read(self.peer, suffix)
        self.api.tags['kws-data'][A.TAG] = 'c' * 40
        second = self.api.read(self.peer, suffix)
        self.assertNotEqual(first['object']['sha'], second['object']['sha'])
        self.assertEqual(len(self.api.calls), 2)
        before = self.api.read(self.peer, '/git/matching-refs/')
        self.api.notes['kws-data']['refs/notes/new'] = 'd' * 40
        after = self.api.read(self.peer, '/git/matching-refs/')
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(len(self.api.calls), 4)
        self.assertGreater(len(after), 100)
        self.assertTrue(all('?' not in call['path'] for call in self.api.calls))

    def test_mutable_main_and_full_protection_are_always_fresh(self):
        repo = 'jiying2007/kws-pipeline'
        first = self.api.protection_snapshot(repo)
        calls = len(self.api.calls)
        self.api.policies['kws-pipeline']['rulesets'][0]['enforcement'] = 'disabled'
        self.api.policies['kws-pipeline']['branch_protected'] = False
        self.api.refs['kws-pipeline']['refs/heads/main'] = 'c' * 40
        second = self.api.protection_snapshot(repo)
        self.assertEqual(len(self.api.calls), calls * 2)
        self.assertNotEqual(first, second)
        self.assertFalse(second['branch_protected'])
        self.assertEqual(second['rulesets'][0]['enforcement'], 'disabled')
        self.assertEqual(self.api.read(repo, '/branches/main')['commit']['sha'], 'c' * 40)

    def test_public_workflow_and_active_work_reads_use_no_token_and_stay_fresh(self):
        for _ in range(2):
            self.api.assert_successful_workflows(self.peer, LEGACY.SOURCE, self.repo['required_success_workflows'], '123')
            self.api.assert_no_active_work(self.peer, [row['name'] for row in self.repo['delete_refs']])
        self.assertEqual(len(self.api.calls), 14)
        self.assertTrue(all(not call['authenticated'] for call in self.api.calls))
        self.assertEqual(self.api.remaining, 46)


class PublicBudgetObservationTests(unittest.TestCase):
    """Offline real-transport tests; no API request method is replaced here."""
    HEADER_KEYS = {'x-ratelimit-limit', 'x-ratelimit-remaining', 'x-ratelimit-used',
                   'x-ratelimit-reset', 'x-ratelimit-resource', 'retry-after'}
    BODY_KEYS = {'remaining', 'limit', 'used', 'reset'}
    TOKEN = 'offline-token-never-log-562'
    SECRET = 'private-server-text-never-log-731'
    RESET = 1800000000
    RESET_UTC = '2027-01-15T08:00:00+00:00'

    def make_api(self):
        opener = Mock(spec=['open'])
        with patch.object(A.urllib.request, 'build_opener', return_value=opener):
            api = A.GitHub('jiying2007/kws-pipeline', self.TOKEN)
        return api, opener

    def headers(self, remaining=58, reset=RESET, **extra):
        headers = Message()
        values = {'X-RateLimit-Limit': '60', 'X-RateLimit-Remaining': str(remaining),
                  'X-RateLimit-Used': str(60 - remaining), 'X-RateLimit-Reset': str(reset),
                  'X-RateLimit-Resource': 'core', 'Retry-After': '17'}
        values.update(extra)
        for key, value in values.items():
            headers[key] = value
        return headers

    def body(self, remaining=58, **extra):
        value = {'remaining': remaining, 'limit': 60, 'used': 60 - remaining if type(remaining) is int else 0,
                 'reset': self.RESET}
        value.update(extra)
        return {'resources': {'core': value}}

    def response(self, body, headers=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        response = io.BytesIO(raw)
        response.headers = headers if headers is not None else Message()
        response.status = 200
        return response

    def observe(self, api, minimum=58):
        output = io.StringIO()
        result = error = None
        before = datetime.now(timezone.utc)
        with redirect_stdout(output):
            try:
                result = api.unauthenticated_rate_budget(minimum)
            except Exception as exc:
                error = exc
        after = datetime.now(timezone.utc)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1, 'Each budget check emits exactly one observation, without retry.')
        observation = json.loads(lines[0])
        self.assertEqual(set(observation), {'status', 'observed_at', 'required_remaining', 'body',
                                             'body_reset_utc', 'headers', 'header_reset_utc'})
        self.assertEqual(observation['status'], 'PUBLIC_API_BUDGET_OBSERVATION')
        self.assertEqual(observation['required_remaining'], minimum)
        stamp = datetime.fromisoformat(observation['observed_at'])
        self.assertIsNotNone(stamp.utcoffset())
        self.assertEqual(stamp.utcoffset().total_seconds(), 0)
        self.assertLessEqual(before, stamp)
        self.assertLessEqual(stamp, after)
        self.assertEqual(set(observation['body']), self.BODY_KEYS)
        self.assertEqual(set(observation['headers']), self.HEADER_KEYS)
        self.assertNotIn(self.TOKEN, output.getvalue())
        self.assertNotIn(self.SECRET, output.getvalue())
        return result, error, observation

    def assert_public_request(self, opener):
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.github.com/rate_limit')
        self.assertEqual(request.get_method(), 'GET')
        self.assertIsNone(request.data)
        self.assertNotIn('authorization', {key.lower() for key, _ in request.header_items()})
        self.assertEqual(opener.open.call_args.kwargs, {'timeout': 45})

    def test_exact_58_and_30_body_floors_preserve_return_shape_and_utc_diagnostics(self):
        for minimum in (58, 30):
            with self.subTest(minimum=minimum):
                api, opener = self.make_api()
                opener.open.return_value = self.response(self.body(minimum), self.headers(minimum))
                result, error, observation = self.observe(api, minimum)
                self.assertIsNone(error)
                self.assertEqual(result, {'remaining': minimum, 'limit': 60, 'reset': self.RESET})
                self.assertEqual(observation['body'], {'remaining': minimum, 'limit': 60,
                                                      'used': 60 - minimum, 'reset': self.RESET})
                self.assertEqual(observation['headers'], {'x-ratelimit-limit': 60,
                    'x-ratelimit-remaining': minimum, 'x-ratelimit-used': 60 - minimum,
                    'x-ratelimit-reset': self.RESET, 'x-ratelimit-resource': 'core', 'retry-after': 17})
                self.assertEqual(observation['body_reset_utc'], self.RESET_UTC)
                self.assertEqual(observation['header_reset_utc'], self.RESET_UTC)
                self.assert_public_request(opener)

    def test_57_and_29_stop_only_after_diagnostics_even_if_headers_claim_sufficient(self):
        for minimum in (58, 30):
            with self.subTest(minimum=minimum):
                api, opener = self.make_api()
                opener.open.return_value = self.response(self.body(minimum - 1), self.headers(60))
                result, error, observation = self.observe(api, minimum)
                self.assertIsNone(result)
                self.assertIsInstance(error, A.Stop)
                self.assertEqual(str(error), 'insufficient public API read budget; no write permitted')
                self.assertEqual(observation['body']['remaining'], minimum - 1)
                self.assertEqual(observation['headers']['x-ratelimit-remaining'], 60)
                self.assert_public_request(opener)

    def test_insufficient_headers_do_not_replace_sufficient_original_body_gate(self):
        for minimum in (58, 30):
            with self.subTest(minimum=minimum):
                api, opener = self.make_api()
                opener.open.return_value = self.response(self.body(minimum), self.headers(0, self.RESET + 60))
                result, error, observation = self.observe(api, minimum)
                self.assertIsNone(error)
                self.assertEqual(result['remaining'], minimum)
                self.assertEqual(observation['body']['remaining'], minimum)
                self.assertEqual(observation['headers']['x-ratelimit-remaining'], 0)
                self.assertEqual(observation['header_reset_utc'], '2027-01-15T08:01:00+00:00')
                self.assert_public_request(opener)

    def test_http_403_and_429_log_whitelisted_headers_without_body_read_or_retry(self):
        for status in (403, 429):
            with self.subTest(status=status):
                api, opener = self.make_api()
                body = Mock(spec=['read', 'close'])
                error = urllib.error.HTTPError('https://api.github.com/rate_limit', status, self.SECRET,
                    self.headers(0, **{'Authorization': self.TOKEN, 'Set-Cookie': self.SECRET,
                                      'X-Arbitrary-Server-Text': self.SECRET}), body)
                opener.open.side_effect = error
                result, raised, observation = self.observe(api)
                self.assertIsNone(result)
                self.assertIsInstance(raised, A.ApiError)
                self.assertEqual(raised.status, status)
                self.assertEqual(raised.errors, [])
                self.assertNotIn(self.SECRET, str(raised))
                self.assertEqual(observation['body'], dict.fromkeys(self.BODY_KEYS))
                self.assertIsNone(observation['body_reset_utc'])
                self.assertEqual(observation['headers']['x-ratelimit-remaining'], 0)
                self.assertEqual(observation['headers']['retry-after'], 17)
                self.assertEqual(observation['header_reset_utc'], self.RESET_UTC)
                body.read.assert_not_called()
                self.assert_public_request(opener)

    def test_transport_failures_emit_empty_safe_observation_and_do_not_retry(self):
        for error in (urllib.error.URLError(self.SECRET), TimeoutError(self.SECRET), OSError(self.SECRET)):
            with self.subTest(kind=type(error).__name__):
                api, opener = self.make_api()
                opener.open.side_effect = error
                result, raised, observation = self.observe(api)
                self.assertIsNone(result)
                self.assertIsInstance(raised, A.Stop)
                self.assertEqual(str(raised), 'Transport failure; write outcome may be unknown. Do not retry.')
                self.assertEqual(observation['body'], dict.fromkeys(self.BODY_KEYS))
                self.assertEqual(observation['headers'], dict.fromkeys(self.HEADER_KEYS))
                self.assertIsNone(observation['body_reset_utc'])
                self.assertIsNone(observation['header_reset_utc'])
                self.assert_public_request(opener)

    def test_previous_response_headers_are_not_reused_after_next_transport_failure(self):
        api, opener = self.make_api()
        opener.open.side_effect = [self.response(self.body(), self.headers()), urllib.error.URLError(self.SECRET)]
        self.assertIsNone(self.observe(api)[1])
        result, error, observation = self.observe(api)
        self.assertIsNone(result)
        self.assertIsInstance(error, A.Stop)
        self.assertEqual(observation['headers'], dict.fromkeys(self.HEADER_KEYS))
        self.assertEqual(opener.open.call_count, 2)

    def test_malformed_json_and_response_shapes_log_once_then_fail_closed(self):
        malformed = [b'not-json-' + self.SECRET.encode(), b'{"resources":NaN}',
                     b'{"resources":{},"resources":{}}', {}, None, [],
                     {'resources': None}, {'resources': []}, {'resources': {}},
                     {'resources': {'core': None}}, {'resources': {'core': []}},
                     {'resources': {'core': {}}}]
        for value in malformed:
            with self.subTest(value=value):
                api, opener = self.make_api()
                opener.open.return_value = self.response(value, self.headers())
                result, error, observation = self.observe(api)
                self.assertIsNone(result)
                self.assertIsInstance(error, (A.Stop, ValueError, TypeError, KeyError))
                self.assertEqual(observation['body'], dict.fromkeys(self.BODY_KEYS))
                self.assertEqual(observation['headers']['x-ratelimit-limit'], 60)
                self.assert_public_request(opener)

    def test_malformed_remaining_is_redacted_and_cannot_pass_either_real_floor(self):
        for minimum in (58, 30):
            for value in (None, True, False, '58', 58.0, -1, [], {}, self.SECRET):
                with self.subTest(minimum=minimum, value=value):
                    api, opener = self.make_api()
                    opener.open.return_value = self.response(self.body(value), self.headers())
                    result, error, observation = self.observe(api, minimum)
                    self.assertIsNone(result)
                    self.assertIsInstance(error, A.Stop)
                    self.assertIsNone(observation['body']['remaining'])
                    self.assert_public_request(opener)

    def test_missing_return_fields_still_raise_after_body_observation(self):
        for missing in ('remaining', 'limit', 'reset'):
            with self.subTest(missing=missing):
                api, opener = self.make_api()
                value = self.body()
                del value['resources']['core'][missing]
                opener.open.return_value = self.response(value, self.headers())
                result, error, observation = self.observe(api)
                self.assertIsNone(result)
                self.assertIsInstance(error, KeyError)
                self.assertIsNone(observation['body'][missing])
                self.assert_public_request(opener)

    def test_body_and_header_unknown_fields_never_enter_success_diagnostics(self):
        api, opener = self.make_api()
        value = self.body(note=self.SECRET, authorization=self.TOKEN)
        value['private'] = {'cookie': self.SECRET}
        opener.open.return_value = self.response(value, self.headers(**{
            'Authorization': self.TOKEN, 'Cookie': self.SECRET, 'Set-Cookie': self.SECRET,
            'X-GitHub-Request-Id': self.SECRET, 'X-Arbitrary-Server-Text': self.SECRET}))
        result, error, observation = self.observe(api)
        self.assertIsNone(error)
        self.assertEqual(result, {'remaining': 58, 'limit': 60, 'reset': self.RESET})
        self.assertEqual(set(observation['body']), self.BODY_KEYS)
        self.assertEqual(set(observation['headers']), self.HEADER_KEYS)
        self.assert_public_request(opener)

    def test_invalid_allowed_header_values_are_redacted_and_cannot_change_body_gate(self):
        malformed = ('', '-1', '+60', '60.0', ' 60 ', '1e2', '9' * 13, '\u0666\u0660',
                     '60\r\nAuthorization:' + self.TOKEN, self.SECRET)
        for value in malformed:
            with self.subTest(value=value):
                api, opener = self.make_api()
                headers = Message()
                for key in self.HEADER_KEYS:
                    headers[key] = value
                opener.open.return_value = self.response(self.body(), headers)
                result, error, observation = self.observe(api)
                self.assertIsNone(error)
                self.assertEqual(result['remaining'], 58)
                self.assertEqual(observation['headers'], dict.fromkeys(self.HEADER_KEYS))
                self.assertIsNone(observation['header_reset_utc'])
                self.assert_public_request(opener)

    def test_invalid_body_metadata_is_redacted_without_changing_existing_return_values(self):
        for value in (True, False, -1, 1.5, '60', None, [], {}, self.SECRET, 10**12):
            with self.subTest(value=value):
                api, opener = self.make_api()
                body = self.body(limit=value, used=value, reset=value)
                opener.open.return_value = self.response(body, self.headers())
                result, error, observation = self.observe(api)
                self.assertIsNone(error)
                self.assertEqual(result, {'remaining': 58, 'limit': value, 'reset': value})
                self.assertEqual(observation['body'], {'remaining': 58, 'limit': None, 'used': None, 'reset': None})
                self.assertIsNone(observation['body_reset_utc'])
                self.assert_public_request(opener)

    def test_out_of_range_utc_conversion_is_safe_and_zero_is_unix_epoch(self):
        for reset, expected in ((0, '1970-01-01T00:00:00+00:00'), (999999999999, None)):
            with self.subTest(reset=reset):
                api, opener = self.make_api()
                opener.open.return_value = self.response(self.body(reset=reset), self.headers(reset=reset))
                result, error, observation = self.observe(api)
                self.assertIsNone(error)
                self.assertEqual(result['reset'], reset)
                self.assertEqual(observation['body']['reset'], reset)
                self.assertEqual(observation['headers']['x-ratelimit-reset'], reset)
                self.assertEqual(observation['body_reset_utc'], expected)
                self.assertEqual(observation['header_reset_utc'], expected)
                self.assert_public_request(opener)

    def test_existing_integer_gate_is_not_redefined_by_diagnostic_sanitization(self):
        # The original gate uses isinstance(value, int); telemetry must not
        # silently introduce a stricter gate, even for these unusual fixtures.
        for remaining, minimum in ((10**12, 58), (True, 1)):
            with self.subTest(remaining=remaining, minimum=minimum):
                api, opener = self.make_api()
                opener.open.return_value = self.response(self.body(remaining), self.headers())
                result, error, observation = self.observe(api, minimum)
                self.assertIsNone(error)
                self.assertEqual(result['remaining'], remaining)
                self.assertIsNone(observation['body']['remaining'])
                self.assert_public_request(opener)

    def test_error_after_response_headers_preserves_headers_without_raw_error_text(self):
        api, opener = self.make_api()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.headers = self.headers(29)
        response.read.side_effect = OSError(self.SECRET)
        opener.open.return_value = response
        result, error, observation = self.observe(api, 30)
        self.assertIsNone(result)
        self.assertIsInstance(error, A.Stop)
        self.assertNotIn(self.SECRET, str(error))
        self.assertEqual(observation['body'], dict.fromkeys(self.BODY_KEYS))
        self.assertEqual(observation['headers']['x-ratelimit-remaining'], 29)
        self.assertEqual(observation['header_reset_utc'], self.RESET_UTC)
        self.assert_public_request(opener)

    def test_non_budget_success_and_http_error_do_not_even_inspect_response_headers(self):
        api, opener = self.make_api()
        headers = Mock(spec=['get'])
        headers.get.side_effect = AssertionError('Non-budget headers must not be inspected.')
        opener.open.side_effect = [self.response({'full_name': 'jiying2007/kws-pipeline'}, headers),
            urllib.error.HTTPError('https://api.github.com/repos/jiying2007/kws-pipeline', 403,
                                   self.SECRET, headers, io.BytesIO(self.SECRET.encode()))]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(api.read('jiying2007/kws-pipeline'), {'full_name': 'jiying2007/kws-pipeline'})
            with self.assertRaises(A.ApiError) as caught:
                api.read('jiying2007/kws-pipeline')
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(opener.open.call_count, 2)
        headers.get.assert_not_called()

    def test_ordinary_own_read_auth_and_peer_public_routing_are_unchanged(self):
        api, opener = self.make_api()
        opener.open.side_effect = [self.response({'full_name': 'jiying2007/kws-pipeline'}, self.headers()),
                                   self.response({'full_name': 'jiying2007/kws-data'}, self.headers())]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(api.read('jiying2007/kws-pipeline'), {'full_name': 'jiying2007/kws-pipeline'})
            self.assertEqual(api.read('jiying2007/kws-data'), {'full_name': 'jiying2007/kws-data'})
            with self.assertRaisesRegex(A.Stop, 'cross-repository token use rejected'):
                api.request('GET', '/repos/jiying2007/kws-data', authenticated=True)
            with self.assertRaisesRegex(A.Stop, 'cross-repository token use rejected'):
                api.request('GET', '/rate_limit', authenticated=True)
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(opener.open.call_count, 2)
        own, peer = [call.args[0] for call in opener.open.call_args_list]
        self.assertEqual(own.full_url, 'https://api.github.com/repos/jiying2007/kws-pipeline')
        self.assertEqual(peer.full_url, 'https://api.github.com/repos/jiying2007/kws-data')
        self.assertEqual({key.lower(): value for key, value in own.header_items()}['authorization'], 'Bearer ' + self.TOKEN)
        self.assertNotIn('authorization', {key.lower() for key, _ in peer.header_items()})
        for call in opener.open.call_args_list:
            self.assertEqual(call.kwargs, {'timeout': 45})
            self.assertEqual(call.args[0].get_method(), 'GET')
            self.assertIsNone(call.args[0].data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
