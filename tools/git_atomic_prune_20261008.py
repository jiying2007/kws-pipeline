#!/usr/bin/env python3
"""Exactly one standard Git atomic push, without fallback or automatic retry."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

ANCHOR = 'refs/tags/archive/branches-2026-10-07-prune-anchor'
LOCAL = 'refs/transfer/archive-target'
ZERO = '0' * 40

class Stop(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise Stop(message)

def validate_hook(hooks, expected_sha256):
    path = Path(hooks) / 'pre-push'
    require(path.is_file() and not path.is_symlink(), 'hook missing or symlink')
    mode = path.stat().st_mode
    require(stat.S_ISREG(mode) and mode & stat.S_IXUSR and not mode & (stat.S_IWGRP | stat.S_IWOTH), 'hook mode')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, 'hook hash')
    require(path.read_bytes().startswith(b'#!/usr/bin/python3\n'), 'hook interpreter')
    require(sorted(p.name for p in Path(hooks).iterdir()) == ['pre-push'], 'unexpected hooks')

def _canonical_command(bare, hooks, binding):
    require(binding['anchor'] == ANCHOR and binding['local_ref'] == LOCAL, 'anchor constants')
    rows = binding['delete_refs']
    require(rows and len({r['name'] for r in rows}) == len(rows), 'duplicate deletion')
    for row in rows:
        require(row['name'].startswith('refs/heads/') and row['name'] != 'refs/heads/main'
                and not any(c in row['name'] for c in (' ', '\n', '\r', '\t', '..', '@{', '\\', ':', '*', '?', '['))
                and re.fullmatch('[0-9a-f]{40}', row['before_oid']) and row['before_oid'] != ZERO, 'deletion binding')
    prefix = ['git', '-c', 'protocol.allow=never', '-c', 'protocol.https.allow=always',
              '-c', 'credential.helper=', '-c', 'credential.useHttpPath=true',
              '-c', 'http.followRedirects=false', '-c', 'push.followTags=false',
              '-c', 'push.recurseSubmodules=no', '-c', 'core.hooksPath=' + str(hooks),
              '--git-dir=' + str(bare)]
    return prefix + ['push', '--atomic', '--porcelain', '--no-follow-tags', '--recurse-submodules=no',
                     '--force-with-lease=' + ANCHOR + ':'] + [
        '--force-with-lease=' + r['name'] + ':' + r['before_oid'] for r in rows
    ] + [binding['remote'], LOCAL + ':' + ANCHOR] + [':' + r['name'] for r in rows]

def build_command(bare, hooks, binding):
    return _canonical_command(bare, hooks, binding)

def validate_command(argv, bare, hooks, binding):
    require(argv == _canonical_command(bare, hooks, binding), 'noncanonical push command or hook override')


def parse_porcelain(raw, binding):
    require(len(raw) <= 65536, 'push output size')
    lines = raw.decode('ascii').splitlines()
    require(lines and lines[0] == 'To ' + binding['remote'] and lines[-1] == 'Done', 'push output framing')
    actual = set()
    for line in lines[1:-1]:
        fields = line.split('\t')
        require(len(fields) == 3, 'push output fields')
        actual.add(tuple(fields))
    expected = {('*', LOCAL + ':' + ANCHOR, '[new tag]')}
    expected |= {('-', ':' + row['name'], '[deleted]') for row in binding['delete_refs']}
    require(len(lines) - 2 == len(expected) and actual == expected, 'push output incomplete or unexpected')
    return {'status': 'GIT_ATOMIC_ACKNOWLEDGED', 'deleted_refs': len(binding['delete_refs']), 'anchor': ANCHOR}

class GitAtomicTransport:
    def __init__(self, hook_path, hook_sha256, askpass_path, askpass_sha256):
        self.hook_path, self.hook_sha256 = Path(hook_path), hook_sha256
        self.askpass_path, self.askpass_sha256 = Path(askpass_path), askpass_sha256
        self.invoked = False
        self.acknowledged = False
        self.cleanup_complete = True

    def push(self, bare, repo, token):
        require(not self.invoked, 'automatic retry forbidden')
        require(repo['full_name'] in ('jiying2007/kws-pipeline', 'jiying2007/kws-data'), 'own repo binding')
        count = 57 if repo['full_name'].endswith('/kws-pipeline') else 22
        require(len(repo['delete_refs']) == count, 'frozen deletion count')
        require(bool(token) and '\n' not in token and '\r' not in token, 'ephemeral credential missing')
        binding = {'remote': 'https://github.com/' + repo['full_name'] + '.git',
                   'anchor': ANCHOR, 'archive_oid': repo['archive_commit_oid'],
                   'local_ref': LOCAL, 'delete_refs': [
                       {'name': r['name'], 'before_oid': r['before_oid']} for r in repo['delete_refs']]}
        with tempfile.TemporaryDirectory(prefix='kws-atomic-private-') as directory:
            root = Path(directory); hooks = root / 'hooks'; hooks.mkdir(mode=0o700)
            require(not self.hook_path.is_symlink() and not self.askpass_path.is_symlink(), 'source symlink')
            hook = hooks / 'pre-push'; hook.write_bytes(self.hook_path.read_bytes()); hook.chmod(0o700)
            validate_hook(hooks, self.hook_sha256)
            askpass = root / 'askpass'; askpass.write_bytes(self.askpass_path.read_bytes()); askpass.chmod(0o700)
            require(hashlib.sha256(askpass.read_bytes()).hexdigest() == self.askpass_sha256
                    and askpass.read_bytes().startswith(b'#!/usr/bin/python3\n'), 'askpass hash/interpreter')
            manifest = root / 'binding.json'; raw = json.dumps(binding, sort_keys=True).encode(); manifest.write_bytes(raw)
            receipt = root / 'hook-receipt.json'
            env = {'PATH': '/usr/bin:/bin', 'HOME': str(root), 'LANG': 'C', 'LC_ALL': 'C',
                   'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
                   'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1',
                   'GIT_ASKPASS': str(askpass), 'KWS_GIT_TOKEN': token,
                   'KWS_GIT_REPOSITORY': repo['full_name'], 'KWS_GIT_BINDING': str(manifest),
                   'KWS_GIT_BINDING_SHA256': hashlib.sha256(raw).hexdigest(),
                   'KWS_GIT_HOOK_RECEIPT': str(receipt)}
            local_env = {k:v for k,v in env.items() if k != 'KWS_GIT_TOKEN'}
            # Only a fresh, fully verified archive repository is admitted. No remote,
            # rewrite, helper, per-URL HTTP settings, or inherited push policy.
            result = subprocess.run(['git', '--git-dir=' + str(bare), 'config', '--local', '--list'],
                                    env=local_env, capture_output=True, check=True, timeout=10)
            require(set(result.stdout.decode().splitlines()) == {
                'core.repositoryformatversion=0', 'core.filemode=true', 'core.bare=true'}, 'nonfresh Git config')
            require(not (Path(bare) / 'shallow').exists() and not (Path(bare) / 'objects/info/alternates').exists(), 'incomplete object store')
            subprocess.run(['git', '--git-dir=' + str(bare), 'update-ref', LOCAL, repo['archive_commit_oid'], ZERO],
                           env=local_env, capture_output=True, check=True, timeout=10)
            argv = build_command(bare, hooks, binding)
            validate_command(argv, bare, hooks, binding)
            validate_hook(hooks, self.hook_sha256)
            self.invoked = True  # Once started, every outcome consumes this transport.
            try:
                process = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                try:
                    stdout, stderr = process.communicate(timeout=180)
                except BaseException:
                    import signal
                    self.cleanup_complete = False
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.communicate()
                    self.cleanup_complete = True
                    raise
                require(process.returncode == 0, 'atomic push not acknowledged; reconcile read-only; never retry')
                require(receipt.is_file() and json.loads(receipt.read_text()) == {
                    'anchor_old_oid': ZERO, 'outgoing_rows': count + 1}, 'hook receipt absent or incomplete; reconcile read-only')
                acknowledgement = parse_porcelain(stdout, binding)
                self.acknowledged = True
                return acknowledgement
            except (Stop, subprocess.SubprocessError, OSError, UnicodeError, ValueError) as exc:
                # Never expose subprocess command/environment/output with credentials.
                raise Stop('atomic outcome unknown; reconcile all refs and anchors read-only; never retry') from None

# A separate reviewed mode. The archive-only visibility exception is not inherited.
def validate_git_protection(repo, actual):
    import archive_branches_once_20261007 as base
    expected = repo['protection_snapshot']
    require(isinstance(actual, dict) and set(actual) == set(expected), 'Git-prune policy shape')
    require(isinstance(actual['rulesets'], list) and len(actual['rulesets']) == len(expected['rulesets']), 'Git-prune ruleset inventory')
    unavailable = []
    for observed, frozen in zip(actual['rulesets'], expected['rulesets']):
        require(isinstance(observed, dict) and set(observed) == set(frozen), 'Git-prune ruleset fields')
        if observed['bypass_actors'] == base.UNAVAILABLE_BYPASS_ACTORS:
            require(repo['full_name'] == 'jiying2007/kws-pipeline' and observed['id'] == 22507875
                    and observed['target'] == 'branch' and observed['source_type'] == 'Repository'
                    and observed['source'] == repo['full_name'] and repo['default_branch'] == 'main'
                    and observed['conditions'] == {'ref_name': {'include': ['~DEFAULT_BRANCH'], 'exclude': []}},
                    'Git-prune unapproved policy visibility gap')
            require({k:v for k,v in observed.items() if k != 'bypass_actors'} ==
                    {k:v for k,v in frozen.items() if k != 'bypass_actors'}, 'Git-prune visible rules changed')
            unavailable.append({'ruleset_id': 22507875, 'field': 'bypass_actors', 'observation': 'UNAVAILABLE'})
        else:
            require(isinstance(observed['bypass_actors'], list) and observed == frozen, 'Git-prune full rules changed')
    require(all(actual[k] == expected[k] for k in expected if k != 'rulesets'), 'Git-prune legacy protection changed')
    return {'contract': 'git-atomic-prune-owner-attested-v1', 'visible_policy_matches': True,
            'hidden_actor_list_observed': not unavailable, 'unavailable_fields': unavailable,
            'owner_postcheck_required': True}


def verify_owner_attestation(plan, context, raw, now=None):
    from datetime import datetime, timezone
    import archive_branches_once_20261007 as base
    require(isinstance(raw, str) and len(raw.encode('utf-8')) <= 16384, 'owner attestation size')
    value = base.decode(raw)
    require(set(value) == {'contract', 'own_repository', 'source_sha', 'observed_at', 'policies_sha256', 'main_oids', 'repository_ids', 'branch_inventories_sha256', 'anchor_oids'}, 'owner attestation fields')
    expected = {name: repo['protection_snapshot'] for name, repo in plan['repositories'].items()}
    require(value['contract'] == 'git-atomic-prune-owner-attested-v1'
            and value['own_repository'] == context['repository']
            and value['source_sha'] == context['reviewed_source_sha']
            and value['policies_sha256'] == base.sha256(base.canonical(expected)), 'owner full-policy attestation binding')
    require(value['repository_ids'] == {name: repo['repository_id'] for name, repo in plan['repositories'].items()}, 'owner repository identities')
    require(set(value['main_oids']) == set(plan['repositories']) and all(base.valid_oid(oid) for oid in value['main_oids'].values())
            and value['main_oids'][context['repository'].split('/')[-1]] == context['reviewed_source_sha'], 'owner main identity binding')
    require(set(value['branch_inventories_sha256']) == set(plan['repositories'])
            and all(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) for digest in value['branch_inventories_sha256'].values()), 'owner branch inventory hashes')
    require(set(value['anchor_oids']) == set(plan['repositories']) and all(value['anchor_oids'][name] in (None, repo['archive_commit_oid'])
            for name, repo in plan['repositories'].items()), 'owner anchor identities')
    observed = datetime.fromisoformat(value['observed_at'].replace('Z', '+00:00'))
    require(observed.tzinfo is not None, 'owner observation timezone')
    age = ((now or datetime.now(timezone.utc)) - observed).total_seconds()
    require(0 <= age <= 600, 'owner full-policy attestation not fresh')
    return value

class RetainedGitRestorer:
    """Keep verified own archive objects until the single push and post-verification finish."""
    def __init__(self, root, source_sha):
        import archive_branches_once_20261007 as base
        self.impl = base.GitRestorer(source_sha)
        self.root = Path(root)
        self.bares = {}
        self.sequence = 0
    def __call__(self, repo, plan):
        self.sequence += 1
        directory = self.root / ('proof-' + str(self.sequence)); directory.mkdir(mode=0o700)
        result = self.impl._restore(directory, repo, plan)
        self.bares[repo['full_name']] = directory / 'archive.git'
        return result


def run_git_prune(coordinator, token, source_root):
    coordinator.git_transport = None
    try:
        return _run_git_prune(coordinator, token, source_root)
    except BaseException:
        transport = coordinator.git_transport
        if transport is None or not transport.invoked:
            print(json.dumps({'status': 'PRE_WRITE_FAILURE', 'transport_disposition': 'NOT_INVOKED', 'automatic_retry': False}, sort_keys=True))
        raise

def _run_git_prune(coordinator, token, source_root):
    import archive_branches_once_20261007 as base
    own = coordinator.repo
    require(coordinator.ctx['operation'] == 'git-prune', 'Git mode required')
    attestation_raw = os.environ.get('ARCHIVE_CTX_OWNER_POLICY_ATTESTATION', '')
    attestation = verify_owner_attestation(coordinator.plan, coordinator.ctx, attestation_raw)
    def tags(repo):
        rows = coordinator.api.read(repo['full_name'], '/git/matching-refs/tags/')
        require(isinstance(rows, list) and len(rows) <= 10000, 'tag inventory bounds')
        require(len({row['ref'] for row in rows}) == len(rows), 'duplicate tags')
        result = {row['ref']: row['object']['sha'] for row in rows}
        require(all(ref.startswith('refs/tags/') for ref in result), 'non-tag response')
        return result
    def all_refs(repo):
        rows = coordinator.api.read(repo['full_name'], '/git/matching-refs/')
        require(isinstance(rows, list) and len(rows) <= 10000, 'ref inventory bounds')
        require(len({row['ref'] for row in rows}) == len(rows), 'duplicate ref inventory')
        require(all(row['ref'].startswith('refs/') for row in rows), 'invalid ref inventory')
        return {row['ref']: row['object']['sha'] for row in rows}
    latest_state_refs = {}
    def classify_states(check_owner=True):
        result = {}
        for name, repo in coordinator.plan['repositories'].items():
            inventory = all_refs(repo); latest_state_refs[name] = inventory
            current = {key:value for key,value in inventory.items() if key.startswith('refs/heads/')}
            tag = inventory.get(ANCHOR)
            require(inventory.get(base.TAG) == repo['archive_commit_oid'], 'original archive identity changed')
            require(all(current.get(row['name']) == row['before_oid'] for row in repo['keep_refs']), 'kept original head drift')
            ready = tag is None and all(current.get(row['name']) == row['before_oid'] for row in repo['delete_refs'])
            complete = tag == repo['archive_commit_oid'] and all(row['name'] not in current for row in repo['delete_refs'])
            require(ready or complete, 'partial or inconsistent peer/own cleanup state')
            if repo['full_name'] == own['full_name']:
                require(ready, 'own atomic cleanup already consumed or inconsistent')
            if check_owner:
                require(tag == attestation['anchor_oids'][name]
                        and base.sha256(base.canonical(current)) == attestation['branch_inventories_sha256'][name], 'owner live-state observation changed')
            result[name] = {'state': 'UNTOUCHED_READY' if ready else 'FULLY_COMPLETED',
                            'anchor_oid': tag, 'branch_inventory_sha256': base.sha256(base.canonical(current))}
        return result
    def check_both_sources():
        observations = {}
        for name, repo in coordinator.plan['repositories'].items():
            meta = coordinator.api.read(repo['full_name'])
            require(str(meta['id']) == repo['repository_id'] and str(meta['owner']['id']) == '33591504'
                    and meta['full_name'] == repo['full_name'] and meta['default_branch'] == 'main'
                    and meta['visibility'] == 'public' and not meta['archived'], 'peer repository identity')
            branch = coordinator.api.read(repo['full_name'], '/branches/main')
            require(branch['commit']['sha'] == attestation['main_oids'][name]
                    and branch['protected'] is repo['main_protected_expected'], 'peer main identity/protection drift')
            observations[name] = validate_git_protection(repo, coordinator.api.protection_snapshot(repo['full_name'], main_branch=branch))
            coordinator.api.assert_successful_workflows(repo['full_name'], attestation['main_oids'][name],
                                                       repo['required_success_workflows'], coordinator.ctx['run_id'])
        coordinator.protection_observation = observations[coordinator.own]
        return observations
    initial_rate_budget = coordinator.api.unauthenticated_rate_budget(58)
    policy_observations = check_both_sources()
    before = coordinator.branches(own)
    # Initial classify_states validates every frozen head; expensive active-work
    # listings are reserved for the fresh final pre-write check after restoration.
    tags_before = tags(own); refs_before = all_refs(own); states_before = classify_states()
    peer_refs_before = {name: latest_state_refs[name] for name, repo in coordinator.plan['repositories'].items() if repo['full_name'] != own['full_name']}
    require(all(refs_before.get(k) == v for k, v in {**before, **tags_before}.items()), 'inconsistent all-ref snapshot')
    proofs = coordinator.prove_both()
    # Re-read every precondition after potentially long full-object restoration.
    verify_owner_attestation(coordinator.plan, coordinator.ctx, attestation_raw)
    policy_observations = check_both_sources()
    require(coordinator.branches(own) == before and tags(own) == tags_before and all_refs(own) == refs_before, 'ref inventory changed before push')
    coordinator.check_original_heads(before)
    require(classify_states() == states_before, 'peer cleanup state drift')
    require(all(latest_state_refs[name] == refs for name, refs in peer_refs_before.items()), 'peer refs changed before push')
    for repo in coordinator.plan['repositories'].values():
        require(coordinator.tag(repo) == repo['archive_commit_oid'], 'original archive changed before push')
    prepush_rate_budget = coordinator.api.unauthenticated_rate_budget(30)
    verify_owner_attestation(coordinator.plan, coordinator.ctx, attestation_raw)
    transport = GitAtomicTransport(source_root / 'git_atomic_guard_20261008.py', os.environ['ATOMIC_GUARD_SHA256'],
                                   source_root / 'git_atomic_askpass_20261008.py', os.environ['ATOMIC_ASKPASS_SHA256'])
    coordinator.git_transport = transport
    try:
        receipt = transport.push(coordinator.restore.bares[own['full_name']], own, token)
        expected = {k:v for k,v in before.items() if k not in {r['name'] for r in own['delete_refs']}}
        expected_refs = {k:v for k,v in refs_before.items() if k not in {r['name'] for r in own['delete_refs']}}
        expected_refs[ANCHOR] = own['archive_commit_oid']
        require(all_refs(own) == expected_refs, 'post-push unrelated ref drift; reconcile read-only')
        require(coordinator.branches(own) == expected, 'post-push branch drift; reconcile read-only')
        require(tags(own) == {**tags_before, ANCHOR: own['archive_commit_oid']}, 'post-push tag drift; reconcile read-only')
        post = coordinator.prove_both()
        require(coordinator.branches(own) == expected and tags(own) == {**tags_before, ANCHOR: own['archive_commit_oid']},
                'post-restoration ref drift; reconcile read-only')
        require(all_refs(own) == expected_refs, 'post-restoration unrelated ref drift; reconcile read-only')
        require(all(all_refs(coordinator.plan['repositories'][name]) == refs for name, refs in peer_refs_before.items()), 'peer refs changed during own transaction')
        policy_observations = check_both_sources()
        return {'phase': 'git-prune', 'status': 'OWNER_POSTCHECK_REQUIRED',
                'owner_precheck_attestation': attestation, 'states_before': states_before, 'peer_refs_unchanged': True, 'atomicity': 'OWN_REPOSITORY_ONLY',
                'transport': receipt, 'transport_disposition': 'ACKNOWLEDGED', 'before_proofs': proofs, 'after_proofs': post,
                'protection_observations': policy_observations,
                'original_archive_unchanged': True, 'non_target_refs_unchanged': True,
                'public_api_budgets': {'initial': initial_rate_budget, 'prepush': prepush_rate_budget}}
    except BaseException:
        # Reads only. Failure here cannot trigger a second push or change the original error.
        if not transport.invoked:
            raise
        reconciliation = {'status': 'READ_ONLY_RECONCILIATION', 'automatic_retry': False,
                          'transport_disposition': 'ACKNOWLEDGED' if transport.acknowledged else 'UNKNOWN_OUTCOME',
                          'process_cleanup_complete': transport.cleanup_complete,
                          'authoritative_final_state': False}
        if not transport.cleanup_complete:
            reconciliation['status'] = 'CLEANUP_UNCONFIRMED'
            reconciliation['owner_reconciliation_required_after_executor_exit'] = True
            print(json.dumps(reconciliation, sort_keys=True))
            raise
        try:
            reconciliation['repositories'] = {
                name: {'branches': coordinator.branches(repo), 'refs': all_refs(repo)}
                for name, repo in coordinator.plan['repositories'].items()}
        except Exception:
            reconciliation['status'] = 'READ_ONLY_RECONCILIATION_INCOMPLETE'
            reconciliation['owner_authenticated_reconciliation_required'] = True
        print(json.dumps(reconciliation, sort_keys=True))
        raise
