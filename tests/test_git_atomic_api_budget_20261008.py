"""Offline API-budget regressions using real GitHub read/pagination/cache code.

Only request() is replaced with deterministic HTTP-response fixtures. Git object
restoration and the one Git write boundary are synthetic; no network is used.
"""
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
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
        with patch.dict(os.environ, env), patch.object(G.GitAtomicTransport, 'push', side_effect=push) as mocked:
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
