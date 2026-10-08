"""Offline full precondition tests; mutation boundary is always mocked."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parent
if not (ROOT / 'git_atomic_prune_20261008.py').exists():
    sys.path.insert(0, str(ROOT.parent / 'tools'))
import archive_branches_once_20261007 as A
import git_atomic_prune_20261008 as G
import test_archive_branches_once_20261007 as T

class FakeAPI(T.FakeAPI):
    def __init__(self, p, own='kws-pipeline'):
        super().__init__(p, own)
        self.alltags = {name: {A.TAG: repo['archive_commit_oid'], 'refs/tags/unrelated': '7' * 40}
                        for name, repo in p['repositories'].items()}
    def read(self, repository, suffix=''):
        if suffix.startswith('/git/matching-refs/'):
            return self.pages(repository, suffix)
        return super().read(repository, suffix)
    def pages(self, repository, path):
        if path == '/git/matching-refs/':
            name = repository.split('/')[-1]
            return [{'ref': k, 'object': {'sha': v}} for k, v in {**self.refs[name], **self.alltags[name], 'refs/notes/fixture':'6'*40}.items()]
        if path == '/git/matching-refs/tags/':
            return [{'ref': k, 'object': {'sha': v}} for k, v in self.alltags[repository.split('/')[-1]].items()]
        return super().pages(repository, path)
    def unauthenticated_rate_budget(self, minimum):
        if self.fail_mode == 'rate': raise A.Stop('public API budget unavailable')
        return {'remaining':60,'limit':60,'reset':0}
    def protection_snapshot(self, repository, main_branch=None):
        data = super().protection_snapshot(repository)
        if self.fail_mode == 'policy':
            data['branch_protected'] = not data['branch_protected']
        return data

class Restore:
    def __init__(self):
        self.bares = {name: Path('/unreachable-test-fixture') for name in ('jiying2007/kws-pipeline', 'jiying2007/kws-data')}
        self.failure = None
    def __call__(self, repo, plan):
        value = T.proof(repo, plan)
        if self.failure and repo['full_name'].endswith(self.failure[0]):
            value[self.failure[1]] = self.failure[2]
        return value

class Tests(unittest.TestCase):
    def setUp(self):
        self.plan = T.ready(); self.ctx = T.context(phase='git-prune')
        self.api = FakeAPI(self.plan); self.restore = Restore()
        self.coordinator = A.Coordinator(self.plan, self.ctx, self.api, self.restore)
        self.coordinator.verify_server_archive = lambda r: None
        self.attestation = {'contract': 'git-atomic-prune-owner-attested-v1',
            'own_repository': self.ctx['repository'], 'source_sha': T.SOURCE,
            'repository_ids': {name: r['repository_id'] for name, r in self.plan['repositories'].items()},
            'main_oids': {name: T.SOURCE for name in self.plan['repositories']},
            'branch_inventories_sha256': {name: A.sha256(A.canonical(refs)) for name, refs in self.api.refs.items()},
            'anchor_oids': {name: None for name in self.plan['repositories']},
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'policies_sha256': A.sha256(A.canonical({name: r['protection_snapshot'] for name, r in self.plan['repositories'].items()}))}
    def run_with(self, effect=None):
        env = {'ARCHIVE_CTX_OWNER_POLICY_ATTESTATION': json.dumps(self.attestation),
               'ATOMIC_GUARD_SHA256': 'a' * 64, 'ATOMIC_ASKPASS_SHA256': 'b' * 64}
        def dispatch(bare, repo, token):
            self.coordinator.git_transport.invoked = True
            if isinstance(effect, BaseException): raise effect
            value = effect(bare, repo, token) if effect else None
            self.coordinator.git_transport.acknowledged = True
            return value
        with patch.dict(os.environ, env), patch.object(G.GitAtomicTransport, 'push', side_effect=dispatch) as push:
            try:
                out = G.run_git_prune(self.coordinator, 'fixture-not-credential', ROOT)
                return out, push.call_count
            finally:
                self.push_count = push.call_count
    def success(self, bare, repo, token):
        for r in repo['delete_refs']:
            del self.api.refs[self.api.own][r['name']]
        self.api.alltags[self.api.own][G.ANCHOR] = repo['archive_commit_oid']
        return {'status': 'fixture'}
    def test_success_requires_owner_postcheck(self):
        out, count = self.run_with(self.success)
        self.assertEqual(count, 1)
        self.assertEqual(out['status'], 'OWNER_POSTCHECK_REQUIRED')
        self.assertEqual(self.api.writes, [])
    def test_bad_ci_policy_or_active_work_zero_push(self):
        for failure in ('ci', 'policy', 'active', 'rate'):
            self.setUp(); self.api.fail_mode = failure
            with self.assertRaises(RuntimeError): self.run_with(self.success)
            self.assertEqual(self.push_count, 0)
    def test_each_both_repo_restoration_field_zero_push(self):
        fields = {'status': 'FAIL', 'archive_oid': 'f'*40, 'restored_heads': 0,
                  'full_object_identity': False, 'restored_trees': False, 'tag_only_fetch': False,
                  'external_payload_gap': True, 'pointer_zip': False}
        for name in ('kws-pipeline', 'kws-data'):
            for field, value in fields.items():
                if name == 'kws-data' and field == 'pointer_zip': continue
                self.setUp(); self.restore.failure = (name, field, value)
                with self.assertRaises(RuntimeError): self.run_with(self.success)
                self.assertEqual(self.push_count, 0)
    def test_stale_or_wrong_owner_attestation_zero_push(self):
        for key, value in [('source_sha', 'f'*40), ('own_repository','jiying2007/kws-data'),
                           ('policies_sha256','f'*64), ('contract','archive-only'),
                           ('observed_at',(datetime.now(timezone.utc)-timedelta(minutes=11)).isoformat()),
                           ('observed_at',(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat())]:
            self.setUp(); self.attestation[key] = value
            with self.assertRaises(RuntimeError): self.run_with(self.success)
            self.assertEqual(self.push_count, 0)
    def test_existing_own_anchor_zero_push(self):
        for value in (self.plan['repositories']['kws-pipeline']['archive_commit_oid'], 'f'*40):
            self.setUp(); self.api.alltags['kws-pipeline'][G.ANCHOR] = value
            with self.assertRaises(RuntimeError): self.run_with(self.success)
            self.assertEqual(self.push_count, 0)
    def test_hidden_field_explicitly_unavailable_only_in_new_contract(self):
        repo = self.plan['repositories']['kws-pipeline']
        actual = copy.deepcopy(repo['protection_snapshot'])
        actual['rulesets'][0]['bypass_actors'] = {'observation':'UNAVAILABLE'}
        result = G.validate_git_protection(repo, actual)
        self.assertFalse(result['hidden_actor_list_observed'])
        self.assertTrue(result['owner_postcheck_required'])
        with self.assertRaises(A.Stop): A.validate_protection_observation(repo, actual, 'prune')
        actual['rulesets'][0]['conditions']['ref_name']['include'] = ['~ALL']
        with self.assertRaises(G.Stop): G.validate_git_protection(repo, actual)
    def test_unknown_outcome_is_one_push_and_no_graphql(self):
        with self.assertRaises(G.Stop): self.run_with(G.Stop('unknown fixture outcome'))
        self.assertEqual(self.push_count, 1); self.assertEqual(self.api.writes, [])
    def test_git_context_identity_source_and_attempt_rejected(self):
        for key, value in [('actor','other'),('triggering_actor','other'),('actor_id','7'),
                           ('repository_id','7'),('repository_owner_id','7'),('run_attempt','2'),
                           ('sha','f'*40),('workflow_sha','f'*40),('reviewed_source_sha','f'*40),
                           ('event_name','push'),('ref','refs/heads/other')]:
            context = dict(self.ctx); context[key] = value
            with patch.object(G.GitAtomicTransport, 'push') as push:
                with self.assertRaises(A.Stop): A.Coordinator(self.plan, context, self.api, self.restore)
                push.assert_not_called()
    def test_post_ack_failures_reconcile_once_without_retry(self):
        for failure in ('ref', 'policy', 'proof', 'api'):
            self.setUp()
            def effect(bare, repo, token):
                result = self.success(bare, repo, token)
                if failure == 'ref': self.api.refs[self.api.own]['refs/heads/unrelated-new'] = '4'*40
                if failure == 'policy': self.api.fail_mode = 'policy'
                if failure == 'proof': self.restore.failure = ('kws-data','full_object_identity',False)
                if failure == 'api': self.api.pages = lambda *args: (_ for _ in ()).throw(A.Stop('read unavailable'))
                return result
            with patch('builtins.print') as output:
                with self.assertRaises(RuntimeError): self.run_with(effect)
            self.assertEqual(self.push_count, 1)
            receipts = [json.loads(call.args[0]) for call in output.call_args_list]
            self.assertEqual(len(receipts), 1)
            self.assertFalse(receipts[0]['automatic_retry'])
            self.assertEqual(receipts[0]['transport_disposition'], 'ACKNOWLEDGED')
            self.assertTrue(receipts[0]['status'].startswith('READ_ONLY_RECONCILIATION'))
    def test_pipeline_then_data_accepts_fully_completed_peer(self):
        first, count = self.run_with(self.success)
        self.assertEqual(count, 1)
        self.api.own = 'kws-data'
        self.ctx = T.context('kws-data', 'git-prune')
        self.coordinator = A.Coordinator(self.plan, self.ctx, self.api, self.restore)
        self.coordinator.verify_server_archive = lambda r: None
        self.attestation['own_repository'] = self.ctx['repository']
        self.attestation['branch_inventories_sha256'] = {name: A.sha256(A.canonical(refs)) for name, refs in self.api.refs.items()}
        self.attestation['anchor_oids']['kws-pipeline'] = self.plan['repositories']['kws-pipeline']['archive_commit_oid']
        second, count = self.run_with(self.success)
        self.assertEqual(count, 1)
        self.assertEqual(second['states_before']['kws-pipeline']['state'], 'FULLY_COMPLETED')
        self.assertEqual(second['states_before']['kws-data']['state'], 'UNTOUCHED_READY')
        self.assertEqual(len(self.api.refs['kws-pipeline']), 5)
        self.assertEqual(len(self.api.refs['kws-data']), 2)
    def test_peer_partial_or_bad_anchor_states_zero_push(self):
        for mode in ('missing_one', 'anchor_with_live_candidates', 'absent_all_without_anchor', 'held_drift'):
            self.setUp()
            # Use pipeline as peer because it has kept refs.
            self.api.own = 'kws-data'; self.ctx = T.context('kws-data','git-prune')
            self.coordinator = A.Coordinator(self.plan,self.ctx,self.api,self.restore)
            self.coordinator.verify_server_archive = lambda r: None
            repo = self.plan['repositories']['kws-pipeline']
            if mode == 'missing_one': del self.api.refs['kws-pipeline'][repo['delete_refs'][0]['name']]
            elif mode == 'anchor_with_live_candidates': self.api.alltags['kws-pipeline'][G.ANCHOR] = repo['archive_commit_oid']
            elif mode == 'held_drift': self.api.refs['kws-pipeline'][repo['keep_refs'][0]['name']] = 'f'*40
            else:
                for row in repo['delete_refs']: del self.api.refs['kws-pipeline'][row['name']]
            self.attestation['own_repository'] = self.ctx['repository']
            self.attestation['branch_inventories_sha256'] = {name: A.sha256(A.canonical(refs)) for name, refs in self.api.refs.items()}
            self.attestation['anchor_oids'] = {name: values.get(G.ANCHOR) for name, values in self.api.alltags.items()}
            with self.assertRaises(RuntimeError): self.run_with(self.success)
            self.assertEqual(self.push_count, 0)
    def test_unconfirmed_cleanup_defers_reads_until_executor_exit(self):
        def effect(*args):
            self.coordinator.git_transport.cleanup_complete = False
            self.api.read = lambda *args: (_ for _ in ()).throw(AssertionError('must not reconcile before cleanup'))
            raise G.Stop('unknown outcome')
        with patch('builtins.print') as output:
            with self.assertRaises(G.Stop): self.run_with(effect)
        self.assertEqual(self.push_count, 1)
        receipt = json.loads(output.call_args.args[0])
        self.assertEqual(receipt['status'], 'CLEANUP_UNCONFIRMED')
        self.assertEqual(receipt['transport_disposition'], 'UNKNOWN_OUTCOME')
        self.assertTrue(receipt['owner_reconciliation_required_after_executor_exit'])
        self.assertNotIn('repositories', receipt)
    def test_oversize_attestation_is_prewrite_failure(self):
        self.attestation['untrusted_padding'] = 'x'*17000
        with patch('builtins.print') as output:
            with self.assertRaises(G.Stop): self.run_with(self.success)
        self.assertEqual(self.push_count, 0)
        self.assertEqual(json.loads(output.call_args.args[0])['transport_disposition'], 'NOT_INVOKED')
    def test_git_mode_cannot_fall_into_graphql(self):
        with self.assertRaises(A.Stop): self.coordinator.run()
        self.assertEqual(self.api.writes, [])

if __name__ == '__main__': unittest.main()
