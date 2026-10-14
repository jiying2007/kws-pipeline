"""Offline pure contract fixtures. Every digest is generated from actual bytes.

No source activation, network, credential, ref write or inference is performed.
The public observation tests deliberately do not assert private permit issuance.
"""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import campaign_contract as c


def git_sha(kind, raw):
    return hashlib.sha1(kind.encode() + b' ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def fixture(*, count=2, release=None, release_raw=None, control=None, control_raw=None,
            run=None, source_tree=None, source_timestamp='2026-10-13T00:00:00Z',
            source_freeze_sha256=None, numeric_job_id=81001, workflow_id=71001):
    """Shared proof factory; optional caller inputs are copied, never activated.

    Omitted controls are explicit zero-model fixture materializations. Supplied
    release/control bytes must already be coherent approved fixture copies. Raw
    overrides bind real source/archive bytes; no canonicalization impersonates
    original input. source_freeze_sha256, when supplied, is caller-established.
    """
    if control_raw is not None:
        decoded = c.decode_json(control_raw, require_canonical=False)
        c.require(control is None or c.exact_json(control, decoded), 'fixture control raw mismatch')
        control = decoded
    if control is None:
        control = c.disabled_claim_control()
        control.update({key: True for key in c.CLAIM_APPROVALS})
        control.update(campaign='fixture-k1-campaign', source_ref='research/fixture-k1-campaign',
            evidence_ref='evidence/fixture-k1-campaign', claim_ref='refs/heads/claims/fixture-k1-campaign',
            workflow='.github/workflows/fixture-k1-campaign.yml')
    control = copy.deepcopy(control)
    if control_raw is None:
        control_raw = json.dumps(control, indent=2).encode() + b'\n'
    if release_raw is not None:
        decoded = c.decode_json(release_raw, require_canonical=False)
        c.require(release is None or c.exact_json(release, decoded), 'fixture release raw mismatch')
        release = decoded
    if release is None:
        release = c.current_disabled_release(control_raw)
        release.update({key: True for key in c.RELEASE_APPROVALS})
        release.update({key: c.digest(('fixture-original-' + key).encode()) for key in c.RELEASE_AUTHORITIES})
        release['prior_public_claim_history_sha256'] = c.digest(c.canonical(c.bound_prior_history_identity(control)))
    release = copy.deepcopy(release)
    if release_raw is None:
        release_raw = json.dumps(release, indent=2).encode() + b'\n'
    source_tree = git_sha('tree', b'') if source_tree is None else source_tree
    source_sha = git_sha('commit', ('tree ' + source_tree + '\n\nfixture source\n').encode())
    freeze_raw = c.canonical({'schema': 'fixture-committed-freeze', 'source_tree': source_tree})
    source_freeze_sha256 = c.digest(freeze_raw) if source_freeze_sha256 is None else source_freeze_sha256
    run = {'repository': c.BOUND_REPOSITORY, 'run_id': 91001, 'run_attempt': 1,
        'job_key': 'asr18', 'executed_head_sha': source_sha} if run is None else copy.deepcopy(run)
    source_sha = run['executed_head_sha']
    identity = c.sourcebound_identity(run, numeric_job_id, release, source_tree, source_timestamp,
        source_freeze_sha256, c.digest(release_raw), control, workflow_id)
    rows = []
    for i in range(count):
        row = c.bound_run_row(identity)
        row.update(id=identity['run_id'] + i, run_number=i + 1)
        row.update(repository={'full_name': c.BOUND_REPOSITORY, 'private': False},
            head_repository={'full_name': c.BOUND_REPOSITORY, 'private': False},
            status='in_progress' if i == 0 else 'queued', conclusion=None)
        rows.append(row)
    page = {'total_count': len(rows), 'workflow_runs': rows}
    history_raw = json.dumps(page, indent=2).encode() + b'\n'
    raw_sha256 = c.digest(history_raw)
    acquisition = c.api_bound_history(identity, page, 'acquisition', raw_sha256=raw_sha256)
    owner = c.owner_record(identity, acquisition, c.digest(b'fixture nonce bytes'))
    body, commit_sha = c.owner_commit(owner)
    ref = {'ref': identity['claim_ref'], 'node_id': 'REF_fixture-node',
        'url': 'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/' + identity['claim_ref'],
        'object': {'type': 'commit', 'sha': commit_sha,
            'url': 'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/commits/' + commit_sha}}
    commit = {'sha': commit_sha, 'node_id': 'C_fixture-node', 'message': body['message'],
        'author': body['author'], 'committer': body['committer'],
        'url': 'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/commits/' + commit_sha,
        'html_url': 'https://github.com/' + c.BOUND_REPOSITORY + '/commit/' + commit_sha,
        'tree': {'sha': body['tree'], 'url': 'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/trees/' + body['tree']},
        'parents': [{'sha': body['parents'][0],
            'url': 'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/commits/' + body['parents'][0],
            'html_url': 'https://github.com/' + c.BOUND_REPOSITORY + '/commit/' + body['parents'][0]}],
        'verification': {'verified': False, 'reason': 'unsigned', 'signature': None, 'payload': None, 'verified_at': None}}
    root = '/repos/' + c.BOUND_REPOSITORY + '/git/'
    routes = {'ref_create': root + 'refs', 'ref_readback': root + 'ref/' + identity['claim_ref'][5:],
        'owner_readback': root + 'commits/' + commit_sha}
    responses = {key: c.response_receipt(201 if key == 'ref_create' else 200,
        json.dumps(commit if key == 'owner_readback' else ref, indent=2).encode() + b'\n', route=routes[key])
        for key in routes}
    admission = {'owner': owner, 'commit': {'sha': commit_sha, 'body': body},
        'acquisition_history': acquisition, 'acquisition_responses': responses,
        'host_history': c.api_bound_history(identity, page, 'host', raw_sha256=raw_sha256),
        'consumed': {'schema': 'k1-asr18-consumed-observation-v2', 'owner_commit': commit_sha,
            'identity': identity, 'permanent_marker_durable': True, 'model_consumption': None}}
    claim = c.make_source_claim(run, release, release_raw, control_raw, identity['job_id'], source_tree,
        identity['source_timestamp'], identity['freeze_sha256'], identity['workflow_id'], page, admission)
    return locals()


def validate(f, claim=None):
    raw = c.canonical(f['claim'] if claim is None else claim)
    return c.bind_source_claim(raw, f['run'], f['release'], f['identity']['job_id'], f['source_tree'],
        release_raw=f['release_raw'], control_raw=f['control_raw'], source_freeze_sha256=f['identity']['freeze_sha256'])


class BoundClaimContractTests(unittest.TestCase):
    def test_complete_history2_and_exact_raw_source_binding(self):
        f = fixture()
        claim = validate(f)
        self.assertEqual(claim['source_branch_workflow_runs'], 2)
        self.assertEqual(claim['admission']['host_history']['total_count'], 2)
        self.assertEqual(claim['release_sha256'], hashlib.sha256(f['release_raw']).hexdigest())
        self.assertNotEqual(claim['release_sha256'], c.digest(c.canonical(f['release'])))
        self.assertNotIn('job_id', claim['admission']['host_history']['rows'][1])
        self.assertIsNone(claim['admission']['consumed']['model_consumption'])

    def test_history1_still_valid_v2(self):
        self.assertEqual(validate(fixture(count=1))['source_branch_workflow_runs'], 1)

    def test_raw_http_whitespace_preserved_before_parse(self):
        f = fixture()
        receipt = f['responses']['ref_create']
        self.assertTrue(receipt['raw_body_utf8'].endswith('\n'))
        self.assertIn('\n  ', receipt['raw_body_utf8'])
        self.assertEqual(c.validate_acquisition_responses(f['responses'], f['owner'])['ref_create'], receipt['receipt_sha256'])
        self.assertEqual(receipt['body_bytes'], len(receipt['raw_body_utf8'].encode()))

    def test_real_api_message_terminal_lf_and_utc_representation(self):
        f = fixture()
        for message in (f['body']['message'], f['body']['message'][:-1]):
            value = copy.deepcopy(f['commit']); value['message'] = message
            value['author']['date'] = value['author']['date'][:-1] + '+00:00'
            self.assertEqual(c.verify_owner_commit(value, f['owner']), f['commit_sha'])
        for suffix in ('\n\n', ' ', '\r\n'):
            value = copy.deepcopy(f['commit']); value['message'] = f['body']['message'].rstrip('\n') + suffix
            with self.assertRaises(ValueError): c.verify_owner_commit(value, f['owner'])

    def test_fake_flat_ref_or_body_wrapper_rejected(self):
        f = fixture()
        with self.assertRaises(ValueError): c.validate_claim_ref({'ref': f['identity']['claim_ref'], 'sha': f['commit_sha']}, f['owner'])
        with self.assertRaises(ValueError): c.verify_owner_commit({'sha': f['commit_sha'], 'body': f['body']}, f['owner'])

    def test_http_status_route_bytes_and_hash_tamper(self):
        f = fixture()
        for key, value in (('status', 200), ('route', '/repos/' + c.BOUND_REPOSITORY + '/git/ref/heads/foreign'),
                ('body_bytes', True), ('body_sha256', c.digest(b'tamper')), ('raw_body_utf8', '{}')):
            responses = copy.deepcopy(f['responses']); responses['ref_create'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_acquisition_responses(responses, f['owner'])

    def test_recaptured_foreign_ref_and_commit_rejected(self):
        f = fixture()
        for target in ('ref_create', 'ref_readback', 'owner_readback'):
            responses = copy.deepcopy(f['responses']); receipt = responses[target]
            value = json.loads(receipt['raw_body_utf8'])
            if target == 'owner_readback': value['tree']['sha'] = git_sha('tree', b'foreign')
            else: value['object']['sha'] = git_sha('commit', b'foreign')
            responses[target] = c.response_receipt(receipt['status'], json.dumps(value).encode(), route=receipt['route'])
            with self.subTest(target=target), self.assertRaises(ValueError): c.validate_acquisition_responses(responses, f['owner'])

    def test_duplicate_json_in_original_http_rejected(self):
        f = fixture(); responses = copy.deepcopy(f['responses']); receipt = responses['ref_create']
        raw = receipt['raw_body_utf8'].encode().replace(b'{', b'{"ref":"foreign",', 1)
        responses['ref_create'] = c.response_receipt(201, raw, route=receipt['route'])
        with self.assertRaises(ValueError): c.validate_acquisition_responses(responses, f['owner'])

    def test_owner_source_tree_release_input_and_freeze_tamper(self):
        f = fixture()
        for key in ('source_sha', 'source_tree', 'freeze_sha256', 'release_sha256', 'input_sha256',
                    'prerequisite_sha256', 'prior_history_sha256', 'claim_control_sha256', 'job_id', 'workflow_id'):
            claim = copy.deepcopy(f['claim']); identity = claim['admission']['owner']['identity']
            identity[key] = identity[key] + 1 if type(identity[key]) is int else c.digest(('tamper-' + key).encode())[:len(identity[key])]
            with self.subTest(key=key), self.assertRaises(ValueError): validate(f, claim)

    def test_raw_release_reserialization_is_not_original_member(self):
        f = fixture(); claim = copy.deepcopy(f['claim'])
        claim['release_raw_utf8'] = c.canonical(f['release']).decode()
        claim['release_sha256'] = c.digest(claim['release_raw_utf8'].encode())
        with self.assertRaises(ValueError): validate(f, claim)
        raw = c.canonical(f['claim'])
        with self.assertRaises(ValueError): c.bind_source_claim(raw, f['run'], f['release'], f['identity']['job_id'], f['source_tree'],
            release_raw=c.canonical(f['release']), control_raw=f['control_raw'], source_freeze_sha256=f['identity']['freeze_sha256'])

    def test_independent_job_tree_freeze_and_run_tamper(self):
        f = fixture(); raw = c.canonical(f['claim'])
        args = dict(release_raw=f['release_raw'], control_raw=f['control_raw'], source_freeze_sha256=f['identity']['freeze_sha256'])
        for job, tree in ((f['identity']['job_id'] + 1, f['source_tree']), (f['identity']['job_id'], git_sha('tree', b'bad'))):
            with self.assertRaises(ValueError): c.bind_source_claim(raw, f['run'], f['release'], job, tree, **args)
        for name, value in (('source_freeze_sha256', c.digest(b'bad')), ('run_id', '91002'), ('executed_head_sha', git_sha('commit', b'bad'))):
            params = dict(expected_sha256=c.digest(raw), release=f['release'], source_freeze_sha256=f['identity']['freeze_sha256'],
                run_id=str(f['run']['run_id']), executed_head_sha=f['source_sha']); params[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError): c.validate_source_claim(raw, **params)

    def test_history2_truthful_count_and_all_rows_required(self):
        f = fixture()
        for mutation in ('count', 'truncate', 'duplicate_id', 'duplicate_ordinal', 'invented_job', 'source', 'workflow', 'attempt'):
            page = copy.deepcopy(f['page'])
            if mutation == 'count': page['total_count'] = 1
            elif mutation == 'truncate': page['workflow_runs'].pop()
            elif mutation == 'duplicate_id': page['workflow_runs'][1]['id'] = page['workflow_runs'][0]['id']
            elif mutation == 'duplicate_ordinal': page['workflow_runs'][1]['run_number'] = 1
            elif mutation == 'source': page['workflow_runs'][1]['head_sha'] = git_sha('commit', b'foreign')
            elif mutation == 'workflow': page['workflow_runs'][1]['workflow_id'] += 1
            elif mutation == 'attempt': page['workflow_runs'][1]['run_attempt'] = 2
            else:
                rows = copy.deepcopy(f['acquisition']['rows']); rows[1]['job_id'] = 44
                with self.assertRaises(ValueError): c.bound_history(f['identity'], rows, 'host', raw_sha256=f['raw_sha256'])
                continue
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): c.api_bound_history(f['identity'], page, 'host', raw_sha256=f['raw_sha256'])

    def test_ordinal2_and_attempt2_cannot_be_owner(self):
        f = fixture()
        for key, value in (('run_number', 2), ('run_attempt', 2), ('job_id', True), ('workflow_id', True)):
            identity = dict(f['identity']); identity[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.owner_record(identity, f['acquisition'], c.digest(b'nonce'))

    def test_source_selected_v2_rejects_legacy_downgrade(self):
        f = fixture(); raw = c.canonical(f['claim']); release = c.disabled_release()
        with self.assertRaises(ValueError): c.validate_source_claim(raw, expected_sha256=c.digest(raw), release=release,
            source_freeze_sha256=f['identity']['freeze_sha256'], run_id=str(f['run']['run_id']), executed_head_sha=f['source_sha'])
        release = dict(f['release']); release['claim_admission_mode'] = 'legacy-v1'
        with self.assertRaises(ValueError): c.validate_release_shape(release)
        claim = copy.deepcopy(f['claim']); claim['schema'] = 'k1-asr18-once-source-claim-v1'
        with self.assertRaises(ValueError): validate(f, claim)

    def test_legacy_exact_validator_retains_history1_only(self):
        f = fixture(); release = c.disabled_release()
        release.update({key: True for key in c.RELEASE_APPROVALS})
        release.update({key: c.digest(key.encode()) for key in c.RELEASE_AUTHORITIES})
        claim = {'schema': 'k1-asr18-once-source-claim-v1', 'identity': f['run'],
            'source_freeze_sha256': f['identity']['freeze_sha256'], 'release_sha256': c.digest(c.canonical(release)),
            'source_branch': release['source_ref'], 'result_branch': release['evidence_ref'], 'workflow': release['workflow'],
            'numeric_job_id': f['identity']['job_id'], 'prior_public_claim_history_sha256': release['prior_public_claim_history_sha256'],
            'api_preflight_verified': True, 'source_branch_workflow_runs': 1, 'maximum_new_asr_calls': 18,
            'old_calls_allowed': 0, 'new_audio_forwards_at_claim': 0}
        def check():
            raw = c.canonical(claim)
            return c.validate_legacy_source_claim(raw, expected_sha256=c.digest(raw), release=release,
                source_freeze_sha256=f['identity']['freeze_sha256'], run_id=str(f['run']['run_id']), executed_head_sha=f['source_sha'])
        self.assertEqual(check(), claim)
        claim['source_branch_workflow_runs'] = 2
        with self.assertRaises(ValueError): check()

    def test_public_observation_does_not_establish_model_use_or_private_permit(self):
        f = fixture(); proof = c.validate_bound_admission(f['admission'], f['identity'])
        self.assertFalse(proof['owner']['nonrecognition']['claim_proves_model_consumption'])
        self.assertIsNone(proof['consumed']['model_consumption'])
        self.assertNotIn('nonce', c.canonical(proof).decode().replace('nonce_sha256', ''))
        for key, value in (('model_consumption', 0), ('permanent_marker_durable', False)):
            admission = copy.deepcopy(proof); admission['consumed'][key] = value
            with self.assertRaises(ValueError): c.validate_bound_admission(admission, f['identity'])

    def test_owner_commit_deterministic_and_same_tree_sole_parent(self):
        f = fixture(); body, sha = c.owner_commit(f['owner'])
        self.assertEqual(c.owner_commit(f['owner']), (body, sha))
        self.assertEqual(body['tree'], f['source_tree']); self.assertEqual(body['parents'], [f['source_sha']])
        owner = copy.deepcopy(f['owner']); owner['nonce_sha256'] = c.digest(b'other nonce')
        self.assertNotEqual(c.owner_commit(owner)[1], sha)

    def test_checked_in_current_controls_closed_and_actual_workflow_id_unbound(self):
        here = Path(c.__file__).parent
        control_raw = (here / c.CLAIM_CONTROL_FILE).read_bytes(); control = c.decode_json(control_raw)
        release = c.read_json(here / 'execution-release.json')
        self.assertTrue(all(control[key] is False for key in c.CLAIM_APPROVALS))
        scope=c.current_scope()
        for key,field in (('campaign','campaign'),('source_ref','source_branch'),
                ('evidence_ref','result_branch'),('claim_ref','claim_ref'),('workflow','workflow'),('job_key','job_key')):
            self.assertEqual(control[key],scope[field])
        with self.assertRaisesRegex(ValueError,'claim admission closed'):c.require_current_scope(control)
        self.assertIsNone(control['workflow_id'])
        self.assertEqual(release['claim_admission_control_sha256'], c.digest(control_raw))
        self.assertEqual(c.validate_claim_control(control),control)
        with self.assertRaisesRegex(ValueError,'claim admission closed'):
            c.require_claim_admission(release, control_raw)
        with self.assertRaisesRegex(ValueError,'execution release disabled'):
            c.require_execution_release(release)

    def test_campaign_experiment_and_derived_refs_cannot_drift(self):
        f = fixture()
        release = dict(f['release']); release['experiment'] = 'other-campaign'
        with self.assertRaises(ValueError): c.require_claim_admission(release, f['control_raw'])
        for key, value in (('source_ref', 'research/unrelated'), ('evidence_ref', 'evidence/unrelated'),
                           ('claim_ref', 'refs/heads/claims/unrelated')):
            control = dict(f['control']); control[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_claim_control(control, require_enabled=True)

    def test_history_requires_original_raw_hash_and_preserves_observed_state(self):
        f = fixture()
        self.assertEqual(f['acquisition']['raw_response_sha256'], c.digest(f['history_raw']))
        self.assertEqual(f['acquisition']['rows'][1]['status'], 'queued')
        self.assertEqual(f['acquisition']['rows'][0]['repository'], {'full_name': c.BOUND_REPOSITORY, 'private': False})
        with self.assertRaises(ValueError): c.api_bound_history(f['identity'], f['page'], 'host')
        for key in ('status', 'conclusion', 'repository', 'head_repository'):
            page = copy.deepcopy(f['page']); del page['workflow_runs'][1][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.api_bound_history(f['identity'], page, 'host', raw_sha256=f['raw_sha256'])

    def test_shared_fixture_overrides_preserve_actual_raw_bytes(self):
        f = fixture()
        rebuilt = fixture(count=2, release_raw=f['release_raw'], control_raw=f['control_raw'],
            run=f['run'], source_tree=f['source_tree'], source_timestamp=f['identity']['source_timestamp'],
            source_freeze_sha256=f['identity']['freeze_sha256'], numeric_job_id=f['identity']['job_id'],
            workflow_id=f['identity']['workflow_id'])
        self.assertEqual(rebuilt['claim'], f['claim'])
        self.assertEqual(c.current_disabled_release(f['control_raw'])['experiment'], f['control']['campaign'])

    def test_v2_seals_both_r3_runs_and_source_without_relabeling_consumption(self):
        f = fixture()
        for run_id in (38385589303, 38385623017):
            run = {**f['run'], 'run_id': run_id}
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, 'sealed source owner'):
                fixture(run=run)
        run = {**f['run'], 'executed_head_sha': 'b012fca6bec8e0c2c1cdd5741fe8e43f0ca195b3'}
        with self.assertRaisesRegex(ValueError, 'sealed source owner'):
            fixture(run=run)
        self.assertEqual(c.SEALED_ASR_RUNS, (38207026177, 38219171568, 38286042984))
        self.assertEqual(len(c.SEALED_ASR_SOURCES), 3)
        self.assertIsNone(f['claim']['admission']['consumed']['model_consumption'])

    def test_public_raw_http_unknown_private_fields_reject_at_every_layer(self):
        f = fixture()
        for target, path in (('ref_create', ()), ('ref_readback', ('object',)),
                ('owner_readback', ()), ('owner_readback', ('tree',)),
                ('owner_readback', ('parents', 0)), ('owner_readback', ('author',)),
                ('owner_readback', ('committer',)), ('owner_readback', ('verification',))):
            for key, extra in (('token', 'SYNTHETIC_TOKEN_SENTINEL'), ('private_path', '/home/fixture/private'),
                               ('unknown', {'nested': {'nonce': 'SYNTHETIC_PRIVATE_NONCE'}})):
                responses = copy.deepcopy(f['responses']); receipt = responses[target]
                value = json.loads(receipt['raw_body_utf8']); item = value
                for component in path: item = item[component]
                item[key] = extra
                raw = json.dumps(value, indent=2).encode() + b'\n'
                responses[target] = c.response_receipt(receipt['status'], raw, route=receipt['route'])
                self.assertEqual(responses[target]['raw_body_utf8'].encode(), raw)
                with self.subTest(target=target,path=path,key=key), self.assertRaises(ValueError):
                    c.validate_acquisition_responses(responses, f['owner'])

    def test_public_git_optional_locators_are_exact_same_repository_only(self):
        f = fixture()
        for target, path in (('ref_create', ('url',)), ('ref_create', ('object', 'url')),
                ('owner_readback', ('url',)), ('owner_readback', ('html_url',)),
                ('owner_readback', ('tree', 'url')), ('owner_readback', ('parents', 0, 'url')),
                ('owner_readback', ('parents', 0, 'html_url'))):
            for bad in ('file:///home/fixture/private', 'https://example.invalid/private',
                        'https://api.github.com/repos/other/repository/git/commits/' + f['commit_sha'],
                        'https://api.github.com/repos/' + c.BOUND_REPOSITORY + '/git/commits/' + f['commit_sha'] + '?token=SYNTHETIC'):
                responses = copy.deepcopy(f['responses']); receipt = responses[target]
                value = json.loads(receipt['raw_body_utf8']); item = value
                for component in path[:-1]: item = item[component]
                item[path[-1]] = bad
                responses[target] = c.response_receipt(receipt['status'], json.dumps(value).encode(), route=receipt['route'])
                with self.subTest(target=target,path=path,bad=bad), self.assertRaises(ValueError):
                    c.validate_acquisition_responses(responses, f['owner'])

    def test_owner_unsigned_verification_cannot_carry_signature_or_private_payload(self):
        f = fixture()
        for key, value in (('signature', 'SYNTHETIC_SIGNATURE'), ('payload', '/home/fixture/private'),
                           ('verified', True), ('reason', 'valid'), ('verified_at', '2026-10-13T00:00:00Z')):
            commit = copy.deepcopy(f['commit']); commit['verification'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.verify_owner_commit(commit, f['owner'])
        for bad in ('/home/fixture/private', 'Bearer SYNTHETIC', 'ghp_SYNTHETIC', 'github_pat_SYNTHETIC'):
            value = copy.deepcopy(f['ref']); value['node_id'] = bad
            with self.subTest(node_id=bad), self.assertRaises(ValueError): c.validate_claim_ref(value, f['owner'])

    def test_bound_prior_is_unknown_source_scoped_and_r3_is_one_eighteen_slot_campaign(self):
        import hosted_run as host
        control = c.disabled_claim_control()
        prior = c.bound_prior_history_identity(control)
        self.assertEqual(host.bound_prior_history_identity(control), prior)
        self.assertIsNone(prior['prior_runs'])
        self.assertEqual(prior['prior_runs_status'], 'unobserved')
        for key in ('campaign', 'source_ref', 'evidence_ref', 'claim_ref', 'workflow', 'workflow_id'):
            self.assertIsNone(prior['scope'][key])
        legacy = host.prior_history_identity()
        self.assertEqual(c.digest(c.canonical(legacy)), '80670524b9a53bfcc5f296331c72ee9c1f8b060ba6ad76f292e4354ca7d97b1f')
        self.assertEqual(prior['sealed_prior_campaigns'][:3], legacy['sealed_prior_campaigns'])
        self.assertEqual(len(prior['sealed_prior_campaigns']), 4)
        r3 = prior['sealed_prior_campaigns'][3]
        self.assertEqual((r3['experiment'], r3['authorized_campaign_count'], r3['planned_slots']),
            ('k1-punctuation6-asr18-r3', 1, 18))
        self.assertEqual([row['run_id'] for row in r3['observed_runs']], [38385589303, 38385623017])
        self.assertTrue(all(row['executed_head_sha'] == 'b012fca6bec8e0c2c1cdd5741fe8e43f0ca195b3'
            and row['run_attempt'] == 1 for row in r3['observed_runs']))
        for row in prior['sealed_prior_campaigns']:
            self.assertIsNone(row['actual_consumed_attempts'])
            self.assertEqual(row['reusable_slots'], 0)
        release = c.read_json(Path(c.__file__).parent / 'execution-release.json')
        actual_control=c.read_json(Path(c.__file__).parent / c.CLAIM_CONTROL_FILE)
        actual_prior=c.bound_prior_history_identity(actual_control)
        self.assertEqual(actual_prior['sealed_prior_campaigns'],prior['sealed_prior_campaigns'])
        self.assertIsNone(actual_prior['prior_runs'])
        self.assertEqual(release['prior_public_claim_history_sha256'], c.digest(c.canonical(actual_prior)))

    def test_bound_prior_unknown_or_r3_budget_cannot_be_rewritten_even_rehashed(self):
        f = fixture(); prior = c.bound_prior_history_identity(f['control'])
        changes = []
        for value in ([], 0):
            changed = copy.deepcopy(prior); changed['prior_runs'] = value; changes.append(changed)
        for index in range(4):
            changed = copy.deepcopy(prior); changed['sealed_prior_campaigns'][index]['actual_consumed_attempts'] = 0
            changes.append(changed)
        for key, value in (('planned_slots', 36), ('authorized_campaign_count', 2), ('reusable_slots', 18)):
            changed = copy.deepcopy(prior); changed['sealed_prior_campaigns'][3][key] = value; changes.append(changed)
        changed = copy.deepcopy(prior); changed['sealed_prior_campaigns'].append(copy.deepcopy(changed['sealed_prior_campaigns'][3]))
        changes.append(changed)
        for changed in changes:
            release = {**f['release'], 'prior_public_claim_history_sha256': c.digest(c.canonical(changed))}
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'source-bound prior history'):
                c.require_claim_admission(release, f['control_raw'])

    def test_bound_prior_pin_cannot_move_to_another_coherent_campaign_scope(self):
        f = fixture(); control = copy.deepcopy(f['control'])
        campaign = 'fixture-other-campaign'
        control.update(campaign=campaign, source_ref='research/' + campaign, evidence_ref='evidence/' + campaign,
            claim_ref='refs/heads/claims/' + campaign, workflow='.github/workflows/' + campaign + '.yml')
        control_raw = c.canonical(control)
        release = {**f['release'], 'experiment': campaign,
            **{key:control[key] for key in ('source_ref', 'evidence_ref', 'workflow')},
            'claim_admission_control_sha256': c.digest(control_raw)}
        with self.assertRaisesRegex(ValueError, 'source-bound prior history'):
            c.require_claim_admission(release, control_raw)
        release['prior_public_claim_history_sha256'] = c.digest(c.canonical(c.bound_prior_history_identity(control)))
        self.assertEqual(c.require_claim_admission(release, control_raw), control)

    def test_pure_owner_parser_stays_flexible_but_current_scope_rejects_foreign_campaign(self):
        foreign = fixture()
        self.assertEqual(validate(foreign)['schema'], c.BOUND_CLAIM_SCHEMA)
        for value in (foreign['control'], foreign['identity']):
            with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
                c.require_current_scope(value)
        scope = c.current_scope(); control = c.disabled_claim_control()
        control.update({key:True for key in c.CLAIM_APPROVALS})
        control.update(campaign=scope['campaign'], source_ref=scope['source_branch'],
            evidence_ref=scope['result_branch'], claim_ref=scope['claim_ref'], workflow=scope['workflow'])
        current = fixture(control=control)
        for value in (control, current['identity'], scope):
            self.assertEqual(c.require_current_scope(value), scope)
        altered = {**scope, 'directory': scope['directory'] + '-foreign'}
        with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
            c.require_current_scope(altered)

    def test_compiled_campaign_and_workflow_are_single_current_source_with_fixed_legacy(self):
        import hosted_run as host
        legacy = c.disabled_release(); history = host.prior_history_identity()
        campaign = 'k1-punctuation6-asr18-future-unit-fixture'
        workflow = '.github/workflows/k1-asr18-future-unit-fixture.yml'
        with patch.object(c, 'EXPERIMENT', campaign), patch.object(c, 'CURRENT_WORKFLOW', workflow):
            scope = c.current_scope()
            self.assertEqual(scope, {'campaign':campaign, 'source_branch':'research/'+campaign,
                'result_branch':'evidence/'+campaign, 'claim_ref':'refs/heads/claims/'+campaign,
                'directory':'research/k1-punctuation6-asr18-v1/evidence/asr18-future-unit-fixture',
                'workflow':workflow, 'job_key':'asr18'})
            release = c.current_disabled_release()
            self.assertEqual((release['experiment'], release['source_ref'], release['evidence_ref'], release['workflow']),
                (campaign, scope['source_branch'], scope['result_branch'], workflow))
            self.assertEqual(c.disabled_release(), legacy)
            self.assertEqual(host.prior_history_identity(), history)
            c.validate_release_shape(legacy)
            self.assertTrue(all(release[key] is False for key in c.RELEASE_APPROVALS))
        self.assertEqual(c.digest(c.canonical(history)), '80670524b9a53bfcc5f296331c72ee9c1f8b060ba6ad76f292e4354ca7d97b1f')

    def test_source_claim_cap_remains_64k(self):
        f = fixture(); claim = copy.deepcopy(f['claim']); claim['unbounded'] = 'x' * 65536
        with self.assertRaises(ValueError): validate(f, claim)


if __name__ == '__main__':
    unittest.main()
