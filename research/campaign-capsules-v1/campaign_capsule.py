"""Versioned, read-only ASR18 preparation metadata, never execution authority.

No legacy module is imported and no release, claim, ref, network operation or
model can be started here. The schema deliberately admits only closed drafts.
The original runtime and posterior remain revision-bound, independent callers.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re

REPOSITORY = 'jiying2007/kws-pipeline'
SHA = re.compile(r'[0-9a-f]{64}\Z')
CAMPAIGN = re.compile(r'k1-punctuation6-asr18-r([1-9][0-9]{0,5})\Z')
INPUT_AUTHORITIES = (
    'actual_input_freeze_sha256', 'actual_tts_authority_sha256',
    'actual_screening_receipt_sha256', 'actual_worker_binding_sha256',
    'actual_opaque_order_sha256', 'prior_public_claim_history_sha256',
    'prepare_prerequisite_sha256',
)
LIMIT_FIELDS = {
    'clips', 'campaign_slots', 'old_calls_allowed', 'new_starts',
    'workspace_bytes', 'initial_free_bytes', 'acquisition_reserved_bytes',
    'acquisition_cap_bytes', 'output_bytes', 'setup_seconds', 'engine_seconds',
    'clip_seconds', 'host_seconds', 'workflow_minutes', 'cpu_threshold_seconds',
    'cpu_total_threshold_seconds', 'cpu_poll_seconds', 'kill_command_seconds',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(raw):
    require(type(raw) is bytes, 'exact bytes required')
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    require(type(raw) is bytes and len(raw) <= 1024 * 1024, 'bounded JSON bytes required')
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON field')
            value[key] = item
        return value
    def nonfinite(_):
        raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def exact(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(exact(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
    return value == expected


def fields(value, names, label):
    require(type(value) is dict and set(value) == set(names), 'exact ' + label + ' fields')


def sha(value):
    require(type(value) is str and SHA.fullmatch(value), 'lowercase SHA256 required')
    return value


def relative(value):
    require(type(value) is str and 0 < len(value) <= 240 and '\\' not in value and
            all(p not in ('', '.', '..') for p in value.split('/')) and
            str(PurePosixPath(value)) == value and not value.startswith('/'), 'safe relative member')
    return value


def identity(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def validate_identity(value):
    fields(value, {'bytes', 'sha256'}, 'member identity')
    require(type(value['bytes']) is int and 0 <= value['bytes'] <= 8 * 1024 * 1024,
            'bounded integer member length')
    sha(value['sha256'])


def validate_index(value):
    fields(value, {'schema', 'repository', 'active_campaign', 'sealed_campaigns', 'policies'}, 'index')
    require(value['schema'] == 'k1-asr18-capsule-index-v1' and value['repository'] == REPOSITORY,
            'fixed index version and repository')
    require(value['active_campaign'] is None, 'no active campaign is supported by preparation schema')
    sealed = value['sealed_campaigns']
    require(type(sealed) is list and sealed and all(type(x) is str for x in sealed), 'sealed identities')
    require(len(sealed) == len(set(sealed)) and sealed[0] == 'k1-punctuation6-asr18-v1', 'unique sealed catalog')
    require(all(CAMPAIGN.fullmatch(x) for x in sealed[1:]), 'sealed campaign shape')
    numbers = [int(CAMPAIGN.fullmatch(x).group(1)) for x in sealed[1:]]
    require(numbers and numbers == list(range(1, len(numbers) + 1)), 'contiguous sealed campaign catalog')
    policies = value['policies']
    require(type(policies) is list and 0 < len(policies) <= 32, 'bounded policy catalog')
    names, paths = set(), set()
    for row in policies:
        fields(row, {'path', 'bytes', 'sha256', 'policy_id', 'scientific_sha256'}, 'policy catalog row')
        path = relative(row['path'])
        require(path.startswith('policies/') and path.endswith('.json'), 'policy catalog path')
        require(type(row['policy_id']) is str and re.fullmatch(r'k1-asr18-scientific-v[1-9][0-9]*', row['policy_id']),
                'versioned policy identity')
        require(row['policy_id'] not in names and path not in paths, 'unique policies')
        names.add(row['policy_id']); paths.add(path)
        validate_identity({k: row[k] for k in ('bytes', 'sha256')})
        sha(row['scientific_sha256'])
    return value


def validate_policy(raw, row):
    require(exact(identity(raw), {k: row[k] for k in ('bytes', 'sha256')}), 'catalog-pinned policy bytes')
    value = decode(raw)
    require(raw == canonical(value) + b'\n', 'canonical policy document required')
    fields(value, {'schema', 'policy_id', 'scientific_projection', 'effective_execution_envelope',
                   'claim_admission_mode'}, 'policy')
    require(value['schema'] == 'k1-asr18-capsule-policy-v1' and value['policy_id'] == row['policy_id'] and
            value['claim_admission_mode'] == 'git-owner-v2', 'fixed policy version and claim mode')
    science = value['scientific_projection']
    fields(science, {'schema', 'metadata_experiment', 'families', 'cells', 'limits', 'source_files'}, 'science')
    require(science['schema'] == 'k1-asr18-scientific-source-binding-v1' and
            science['metadata_experiment'] == 'k1-punctuation6-v1', 'original scientific binding schema')
    require(exact(science['families'], ['qwen06', 'sensevoice', 'whisper']) and
            exact(science['cells'], ['k1p6-%03d' % n for n in range(1, 7)]), 'fixed six-by-three science')
    limits = science['limits']; fields(limits, LIMIT_FIELDS, 'scientific budget')
    for name in LIMIT_FIELDS - {'cpu_poll_seconds', 'cpu_threshold_seconds'}:
        require(type(limits[name]) is int and 0 <= limits[name] <= 128 * 1024**3, 'integer scientific budget')
    require(type(limits['cpu_poll_seconds']) is float and 0 < limits['cpu_poll_seconds'] <= 1,
            'finite positive CPU polling period')
    cpu = limits['cpu_threshold_seconds']; fields(cpu, {'setup', *science['families']}, 'CPU budget')
    require(all(type(v) is int and 0 < v <= 86400 for v in cpu.values()), 'integer CPU thresholds')
    require(limits['clips'] == 6 and limits['campaign_slots'] == limits['new_starts'] == 18 and
            limits['old_calls_allowed'] == 0 and sum(cpu.values()) == limits['cpu_total_threshold_seconds'],
            'fixed18 scientific budget arithmetic')
    source = science['source_files']
    require(type(source) is dict and 0 < len(source) <= 256, 'bounded explicit scientific roster')
    for path, item in source.items():
        relative(path); validate_identity(item)
        require(path.startswith(('research/', 'tools/')) and not any(part in path.split('/') for part in
                ('evidence', 'output', 'results')) and not path.endswith(('.wav', '.pt', '.bin', '.safetensors')),
                'public scientific source members only')
    require(digest(canonical(science)) == row['scientific_sha256'], 'unchanged original scientific identity')
    envelope = value['effective_execution_envelope']
    fields(envelope, {'workflow_minutes', 'publisher_seconds', 'host_seconds'}, 'execution envelope')
    require(all(type(v) is int and v > 0 for v in envelope.values()), 'positive integer execution envelope')
    require(envelope['host_seconds'] == limits['host_seconds'] and
            envelope['workflow_minutes'] * 60 == limits['workflow_minutes'] * 60 + envelope['publisher_seconds'],
            'separate publication budget arithmetic')
    return value


def scientific_identity(policy):
    """Identical canonical projection/digest to the retained prerequisite verifier."""
    return digest(canonical(policy['scientific_projection']))


def verify_scientific_members(policy, read):
    """Caller supplies already-approved public local reads; hashes never supply bodies."""
    for path, expected in policy['scientific_projection']['source_files'].items():
        require(exact(identity(read(path)), expected), 'scientific source bytes drift: ' + path)
    return scientific_identity(policy)


def scope(campaign):
    require(type(campaign) is str and CAMPAIGN.fullmatch(campaign), 'fixed ASR18 future campaign shape')
    return {'campaign': campaign, 'repository': REPOSITORY,
            'source_branch': 'research/' + campaign, 'result_branch': 'evidence/' + campaign,
            'claim_ref': 'refs/heads/claims/' + campaign,
            'workflow': '.github/workflows/' + campaign + '.yml', 'job_key': 'asr18'}


def new_draft(campaign, index, *, policy_id):
    validate_index(index)
    selected = [row for row in index['policies'] if row['policy_id'] == policy_id]
    require(len(selected) == 1, 'known reviewed policy required')
    scope(campaign)
    require(campaign not in index['sealed_campaigns'], 'sealed campaign cannot be reopened')
    require(int(CAMPAIGN.fullmatch(campaign).group(1)) > len(index['sealed_campaigns']) - 1,
            'new campaign ordinal required')
    row = selected[0]
    return {'schema': 'k1-asr18-campaign-capsule-v1', 'status': 'draft_closed',
            'identity': scope(campaign), 'policy_id': policy_id, 'policy_sha256': row['sha256'],
            'scientific_sha256': row['scientific_sha256'], 'execution_authority': 'absent',
            'input_authorities': {key: None for key in INPUT_AUTHORITIES},
            'source_binding': None, 'claim_owner': None}


def validate_draft(value, index, *, expected_campaign):
    """Expected identity is selected by the caller, never by an untrusted owner proof."""
    require(type(value) is dict and type(value.get('policy_id')) is str, 'capsule object with policy')
    expected = new_draft(expected_campaign, index, policy_id=value['policy_id'])
    require(exact(value, expected), 'exact closed draft identity, budget policy and absent authority required')
    return value


def disabled_controls(value, index, policy, *, expected_campaign):
    """Four deterministic CLOSED projections; not a source freeze or authorization.

    These are legacy-format preparation artifacts. Current live scope validation
    intentionally rejects a future campaign until a separately reviewed adapter
    and new-version entrypoints are delivered.
    """
    validate_draft(value, index, expected_campaign=expected_campaign)
    selected = next(row for row in index['policies'] if row['policy_id'] == value['policy_id'])
    validate_policy(canonical(policy) + b'\n', selected)
    require(policy['policy_id'] == value['policy_id'] and
            scientific_identity(policy) == value['scientific_sha256'], 'selected scientific policy')
    ident = value['identity']; campaign = ident['campaign']
    control = {'schema': 'k1-asr18-claim-admission-control-v2', 'mode': 'git-owner-v2',
               'enabled': False, 'independent_source_review': False, 'root_execution_release': False,
               'campaign': campaign, 'repository': REPOSITORY, 'source_ref': ident['source_branch'],
               'evidence_ref': ident['result_branch'], 'claim_ref': ident['claim_ref'],
               'workflow': ident['workflow'], 'job_key': 'asr18', 'workflow_id': None,
               'workflow_id_source': 'actual_run_api_plus_path'}
    release = {'schema': 'k1-asr18-execution-release-v2', 'experiment': campaign,
               'enabled': False, 'root_execution_release': False, 'independent_source_review': False,
               'historical_screening_reviewed': False, **dict(value['input_authorities']),
               'max_new_asr_calls': 18, 'old_calls_allowed': 0, 'retries': 0, 'run_attempt': 1,
               'source_ref': ident['source_branch'], 'evidence_ref': ident['result_branch'],
               'workflow': ident['workflow'], 'job_key': 'asr18', 'claim_admission_mode': 'git-owner-v2',
               'claim_admission_control_sha256': digest(canonical(control))}
    publication = {'schema': 'k1-asr18-publication-control-v2', 'enabled': False,
                   'independent_source_review': False, 'root_publication_release': False,
                   'source_branch': ident['source_branch'], 'result_branch': ident['result_branch'],
                   'workflow': ident['workflow'], 'job_key': 'asr18', 'run_attempt': 1,
                   'retry_authorized': False, 'limited_blob_retry_cap': 1,
                   'same_repository_ephemeral_token_only': True, 'claim_admission_mode': 'git-owner-v2',
                   'scientific_campaign_baseline_minutes': policy['scientific_projection']['limits']['workflow_minutes'],
                   'effective_execution_envelope': deepcopy(policy['effective_execution_envelope'])}
    recovery = {'schema': 'k1-asr18-transfer-recovery-control-v1', 'enabled': False,
                'independent_source_review': False, 'root_transfer_recovery_release': False,
                'authority': None, 'result_branch': ident['result_branch'], 'retry_authorized': False,
                'new_native_calls_authorized': False, 'same_repository_ephemeral_token_only': True,
                'job_key': 'transfer_recovery', 'run_attempt': 1,
                'source_branch': 'research/k1-asr18-transfer-recovery-v1',
                'workflow': '.github/workflows/k1-asr18-transfer-recovery-v1.yml'}
    return {'claim-admission-control.json': control, 'execution-release.json': release,
            'publication-release.json': publication, 'transfer-recovery-release.json': recovery}
