"""K1's fixed six-cell/eighteen-slot contract. No old cohort or execution grant.

Filesystem primitives fail closed and never delete an uncertain write. Local
claims cannot prevent fresh-workspace replay; the fixed release must separately
establish first-run authority and consumed-claim history before runtime setup.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone

import asr18_metadata as metadata

EXPERIMENT = 'k1-punctuation6-asr18-r5'
CURRENT_WORKFLOW = '.github/workflows/k1-punctuation6-asr18-r5.yml'
LEGACY_V1_SCOPE = {'campaign': 'k1-punctuation6-asr18-r3',
    'source_branch': 'research/k1-punctuation6-asr18-r3', 'result_branch': 'evidence/k1-punctuation6-asr18-r3',
    'directory': 'research/k1-punctuation6-asr18-v1/evidence/asr18-r3',
    'workflow': '.github/workflows/k1-punctuation6-asr18-v1.yml', 'job_key': 'asr18'}
SEALED_ASR_RUNS = (38207026177, 38219171568, 38286042984)
SEALED_ASR_SOURCES = ('0f3cc89a3235c4b3af294dc6b5816ff9877c0f42',
                      '054213c551eb83494b9a4ba8cd6d3ca4c1f488dd',
                      'd3b7ce54c8909df8e962d8547a546c3d460dcbc8')
FAMILIES = ('qwen06', 'sensevoice', 'whisper')
CELLS = metadata.CELLS
MIB, GIB = 1024 ** 2, 1024 ** 3
LIMITS = {'clips': 6, 'campaign_slots': 18, 'old_calls_allowed': 0, 'new_starts': 18,
    'workspace_bytes': 10 * GIB, 'initial_free_bytes': 12 * GIB,
    'acquisition_reserved_bytes': 4 * GIB + 1536 * MIB + 128 * MIB + 16 * MIB,
    'acquisition_cap_bytes': 6 * GIB, 'output_bytes': 32 * MIB,
    'setup_seconds': 1800, 'engine_seconds': 600, 'clip_seconds': 120,
    'host_seconds': 3600, 'workflow_minutes': 70,
    'cpu_threshold_seconds': {'setup': 3600, 'qwen06': 600, 'sensevoice': 600, 'whisper': 1800},
    'cpu_total_threshold_seconds': 6600, 'cpu_poll_seconds': 0.2, 'kill_command_seconds': 5}
# LIMITS.workflow_minutes is the exact historical scientific admission baseline.
# The separate publication-only window gives the current job its effective total.
EFFECTIVE_EXECUTION_ENVELOPE = {'workflow_minutes': 80, 'publisher_seconds': 600, 'host_seconds': 3600}
SCIENTIFIC_LIMITS_SCOPE = 'original_prepare_scientific_admission_baseline'
BOUNDARY = {'scope': EXPERIMENT, 'training_admitted': False, 'human_gold': False,
    'shipping_approved': False, 'acoustic_verbatim_completeness': 'unknown'}
SHA = re.compile(r'[0-9a-f]{64}\Z')
HEAD = re.compile(r'[0-9a-f]{40}\Z')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(raw):
    require(type(raw) is bytes, 'byte digest required')
    return hashlib.sha256(raw).hexdigest()


def exact_json(value, expected):
    if type(value) is not type(expected):
        return False
    if type(value) is dict:
        return set(value) == set(expected) and all(exact_json(value[k], expected[k]) for k in expected)
    if type(value) is list:
        return len(value) == len(expected) and all(exact_json(a, b) for a, b in zip(value, expected))
    return value == expected


def sha(value):
    require(type(value) is str and SHA.fullmatch(value), 'lowercase SHA256 required')
    return value


def decode_json(raw, *, require_canonical=True):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON field')
            out[key] = value
        return out
    require(type(raw) is bytes, 'JSON bytes required')
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, 'nonfinite JSON'))
    require(not require_canonical or canonical(value) == raw, 'canonical JSON required')
    return value


def safe_path(path):
    path = Path(path)
    require('..' not in path.parts, 'parent path traversal prohibited')
    return Path(os.path.abspath(path))


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _open_directory(path):
    """Walk every component through verified no-follow directory descriptors."""
    path = safe_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            try:
                require(stat.S_ISDIR(os.fstat(child).st_mode), 'directory required')
                os.close(fd)
            except BaseException:
                os.close(child)
                raise
            fd = child
        require(_directory_identity(os.fstat(fd)) == _directory_identity(path.stat(follow_symlinks=False)),
                'directory path changed')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_mode


def read_bytes(path, *, cap=32 * MIB, readonly=False):
    path = safe_path(path)
    parent = _open_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= cap,
                'bounded unaliased regular file required')
        if readonly:
            require(not before.st_mode & 0o222 or bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY),
                    'read-only frozen file required')
        chunks, length = [], 0
        while length <= cap:
            chunk = os.read(fd, min(65536, cap + 1 - length))
            if not chunk:
                break
            chunks.append(chunk); length += len(chunk)
        body = b''.join(chunks)
        require(len(body) == before.st_size and _stamp(os.fstat(fd)) == _stamp(before), 'file changed during read')
        require(_stamp(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == _stamp(before),
                'file path changed during read')
        require(_directory_identity(os.fstat(parent)) == _directory_identity(path.parent.stat(follow_symlinks=False)),
                'parent path changed during read')
        return body
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def read_json(path, cap=32 * MIB):
    # Retained official lock files use existing pretty JSON; receipts separately
    # require exact canonical bytes. Both reject duplicate/nonfinite fields.
    return decode_json(read_bytes(path, cap=cap), require_canonical=False)


def identity(path):
    body = read_bytes(path)
    return {'bytes': len(body), 'sha256': digest(body)}


def durable_mkdir(path):
    """Create exactly one directory and persist its entry in the grandparent.

    Existing paths are always terminal; no directory reuse, resume or rollback.
    """
    path = safe_path(path)
    parent = _open_directory(path.parent)
    child = None
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        child = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent)
        require(_directory_identity(os.fstat(child)) == _directory_identity(path.stat(follow_symlinks=False)),
                'new directory path drift')
        os.fsync(parent)
        require(_directory_identity(os.fstat(parent)) == _directory_identity(path.parent.stat(follow_symlinks=False)),
                'new directory parent drift')
    finally:
        try:
            if child is not None:
                os.close(child)
        finally:
            os.close(parent)
    return path


def write_once(path, value):
    """Exclusive/no-follow canonical write, then file AND verified parent fsync.

    A created or uncertain record is never removed even after any failure. The
    caller must consume the attempt and stop on every exception, including close.
    """
    path = safe_path(path)
    body = canonical(value)
    require(len(body) <= LIMITS['output_bytes'], 'JSON output bound')
    parent = _open_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'unaliased new regular file required')
        offset = 0
        while offset < len(body):
            length = os.write(fd, body[offset:])
            require(type(length) is int and 0 < length <= len(body) - offset, 'complete canonical write required')
            offset += length
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        require(os.fstat(fd).st_size == len(body), 'complete canonical file required')
        require(_stamp(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == _stamp(os.fstat(fd)),
                'written file path drift')
        os.fsync(parent)
        require(_directory_identity(os.fstat(parent)) == _directory_identity(path.parent.stat(follow_symlinks=False)),
                'written parent path drift')
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)
    require(read_bytes(path, readonly=True) == body, 'durable retained output drift')
    return digest(body)


def validate_blind_job(job, execution_plan):
    require(type(job) is dict and set(job) == {'schema', 'clips'} and
            job['schema'] == 'blind-source-screen32-job-v1', 'audio-only blind job required')
    clips = job['clips']
    require(type(clips) is list and 1 <= len(clips) <= 6, 'one through six eligible clips required')
    hashes = []
    for index, row in enumerate(clips, 1):
        require(type(row) is dict and set(row) == {'audio_id', 'audio_path', 'wav_sha256'}, 'blind clip fields only')
        sha(row['wav_sha256'])
        require(row['audio_id'] == f'clip-{index:06d}' and
                row['audio_path'] == 'audio/' + row['wav_sha256'] + '.wav', 'anonymous fixed clip identity')
        hashes.append(row['wav_sha256'])
    require(hashes == sorted(set(hashes)), 'unique SHA-sorted eligible roster required')
    fields = {'schema', 'execution_enabled', 'job_sha256', 'family_order', 'audio_order'}
    require(type(execution_plan) is dict and set(execution_plan) == fields and
        execution_plan['schema'] == 'k1-asr18-anonymous-order-metadata-proposal-v1' and
        execution_plan['execution_enabled'] is False and execution_plan['job_sha256'] == digest(canonical(job)) and
        execution_plan['family_order'] == list(FAMILIES), 'fixed separate anonymous order required')
    order = execution_plan['audio_order']
    by_id = {row['audio_id']: row for row in clips}
    require(type(order) is list and all(type(x) is str for x in order) and
            len(order) == len(by_id) and set(order) == set(by_id), 'complete unique anonymous order required')
    return by_id


def planned_clips(job, execution_plan, family):
    require(family in FAMILIES, 'fixed recognizer family required')
    clips = validate_blind_job(job, execution_plan)
    return [dict(clips[audio_id]) for audio_id in execution_plan['audio_order']]


def validate_input_binding(value, job, execution_plan):
    clips = validate_blind_job(job, execution_plan)
    require(type(value) is dict and set(value) == {'schema', 'job_sha256', 'execution_plan_sha256',
        'screening_receipt_sha256', 'tts_authority_sha256', 'audio'} and
        value['schema'] == 'k1-asr18-worker-input-binding-v1', 'opaque worker input binding only')
    for key in ('job_sha256', 'execution_plan_sha256', 'screening_receipt_sha256', 'tts_authority_sha256'):
        sha(value[key])
    require(value['job_sha256'] == digest(canonical(job)) and
        value['execution_plan_sha256'] == digest(canonical(execution_plan)), 'frozen job/order identity drift')
    audio = value['audio']
    require(type(audio) is list and len(audio) == len(clips), 'complete opaque input identity roster')
    for audio_id, row in zip(execution_plan['audio_order'], audio):
        require(type(row) is dict and set(row) == {'audio_id', 'wav_sha256', 'pcm_sha256', 'audio_frames'} and
                row['audio_id'] == audio_id and row['wav_sha256'] == clips[audio_id]['wav_sha256'],
                'exact opaque sequence and WAV binding')
        sha(row['pcm_sha256'])
        require(type(row['audio_frames']) is int and 0 < row['audio_frames'] < 192000,
                'strict bounded derived frame identity')
    return value


def validate_start_slot(job, execution_plan, input_binding, family, *, audio_id, measured, source_id, request_id, run_id):
    require(family in FAMILIES and type(run_id) is str and re.fullmatch(r'[1-9][0-9]{0,19}', run_id),
            'actual fixed family/run identity required')
    validate_input_binding(input_binding, job, execution_plan)
    found = [row for row in input_binding['audio'] if row['audio_id'] == audio_id]
    require(len(found) == 1, 'eligible opaque slot required')
    row = found[0]
    require(exact_json(measured, {'file_sha256': row['wav_sha256'], 'pcm_sha256': row['pcm_sha256'],
        'audio_frames': row['audio_frames']}), 'actual WAV/PCM/frame identity drift')
    require(source_id == 'k1asr18-' + row['wav_sha256'] and
        request_id == 'k1asr18-' + run_id + '-' + family + '-' + audio_id,
        'single fixed source/request identity; no retry or relabel')
    return row


def fixed_slots(projection, job, execution_plan):
    blind = metadata.blind_metadata(projection)
    require(exact_json(job, blind['job']) and exact_json(execution_plan, blind['execution_plan']),
            'host six-cell screening and anonymous worker order differ')
    validate_blind_job(job, execution_plan)
    cells = {row['cell_id']: row for row in projection['cells']}
    opaque = {row['cell_id']: row['audio_id'] for row in blind['host_only_cell_map']}
    return [{**slot, 'disposition': cells[slot['cell_id']]['disposition'],
        'audio_id': opaque.get(slot['cell_id']), 'wav_sha256': cells[slot['cell_id']]['wav_sha256'],
        'pcm_sha256': cells[slot['cell_id']]['pcm_sha256'], 'audio_frames': cells[slot['cell_id']]['frames']}
        for slot in metadata.slots()]


def initial_ledger(projection):
    return metadata.initial_ledger(projection)


def validate_ledger(projection, ledger):
    return metadata.validate_ledger(projection, ledger)


RELEASE_APPROVALS = ('enabled', 'root_execution_release', 'independent_source_review',
                     'historical_screening_reviewed')
RELEASE_AUTHORITIES = ('actual_input_freeze_sha256', 'actual_tts_authority_sha256',
    'actual_screening_receipt_sha256', 'actual_worker_binding_sha256',
    'actual_opaque_order_sha256', 'prior_public_claim_history_sha256', 'prepare_prerequisite_sha256')


def disabled_release():
    """Public control-plane template, with every unestablished authority null."""
    return {'schema': 'k1-asr18-execution-release-v1', 'experiment': LEGACY_V1_SCOPE['campaign'],
        **{key: False for key in RELEASE_APPROVALS}, **{key: None for key in RELEASE_AUTHORITIES},
        'max_new_asr_calls': 18, 'old_calls_allowed': 0, 'retries': 0, 'run_attempt': 1,
        'source_ref': LEGACY_V1_SCOPE['source_branch'],
        'evidence_ref': LEGACY_V1_SCOPE['result_branch'],
        'workflow': LEGACY_V1_SCOPE['workflow'], 'job_key': LEGACY_V1_SCOPE['job_key']}


def _validate_legacy_release_shape(value):
    baseline = disabled_release()
    require(type(value) is dict and set(value) == set(baseline), 'exact fixed release fields required')
    for key in set(baseline) - set(RELEASE_APPROVALS) - set(RELEASE_AUTHORITIES):
        require(exact_json(value[key], baseline[key]), 'fixed one-use release scope drift')
    for key in RELEASE_APPROVALS:
        require(type(value[key]) is bool, 'explicit review/release boolean required')
    for key in RELEASE_AUTHORITIES:
        require(value[key] is None or (type(value[key]) is str and SHA.fullmatch(value[key])),
                'actual external authority digest or null required')
    return value


def _require_execution_release(release=None, source_freeze=None, *, expected_freeze_sha256=None,
                              prepare_prerequisite_raw=None, source_root=None, source_members=None):
    """Gate fixed source-local control plane; no CLI path or self-trusted digest.

    Tests can exercise this pure check with explicit synthetic values. Production
    entrypoints must read only source-local files and pass an independently
    established expected source-freeze digest, never its own freshly made hash.
    Cross-workspace once-only history remains a separate trusted host obligation.
    """
    here = Path(__file__).resolve().parent
    release = read_json(here / 'execution-release.json') if release is None else release
    validate_release_shape(release)
    require(all(release[key] is True for key in RELEASE_APPROVALS), 'execution release disabled or review absent')
    require(all(release[key] is not None for key in RELEASE_AUTHORITIES), 'actual execution authority unavailable')
    sha(expected_freeze_sha256)
    source_freeze = read_json(here / 'execution-freeze.json') if source_freeze is None else source_freeze
    require(digest(canonical(source_freeze)) == expected_freeze_sha256,
            'independently approved complete source-freeze identity required')
    import prepare_prerequisite
    require(Path(prepare_prerequisite.__file__).resolve() == here / 'prepare_prerequisite.py',
            'fixed prerequisite verifier origin')
    raw = read_bytes(here / 'prepare-prerequisite.json', cap=65536) if prepare_prerequisite_raw is None else prepare_prerequisite_raw
    prepare_prerequisite.validate_receipt(raw, release, source_freeze=source_freeze,
        source_root=source_root, source_members=source_members)
    return release


def validate_worker_admission(release, input_binding, job, execution_plan):
    """Bind opaque worker files to already released digests, not global authority.

    The host must separately verify the actual source/run and one-use claim;
    this function is deliberately only the anonymous worker-binding check.
    """
    validate_release_shape(release)
    require(all(release[key] is True for key in RELEASE_APPROVALS) and
            all(release[key] is not None for key in RELEASE_AUTHORITIES), 'approved real release required')
    validate_input_binding(input_binding, job, execution_plan)
    expected = {'actual_worker_binding_sha256': digest(canonical(input_binding)),
        'actual_screening_receipt_sha256': input_binding['screening_receipt_sha256'],
        'actual_tts_authority_sha256': input_binding['tts_authority_sha256'],
        'actual_opaque_order_sha256': digest(canonical(execution_plan))}
    require(all(release[key] == value for key, value in expected.items()), 'released anonymous input authority drift')
    return input_binding


def validate_decode_proof(proof, *, job, execution_plan, input_binding, policy_path, family):
    """Check the trusted host's pre-forward actual-closure budget proof binding.

    Bounds and identity checks do not reproduce the packaging operation. The
    host must actually retain the complete prepared closure and worst-prefix
    capacity fixture before passing this read-only proof into a decode worker.
    """
    require(family in FAMILIES, 'fixed decode family required')
    validate_input_binding(input_binding, job, execution_plan)
    fixed = {'schema': 'k1-asr18-predecode-package-budget-v1',
        'status': 'passed_before_any_audio_forward', 'new_native_calls': 0,
        'eligible_clip_count': len(job['clips']), 'whisper_full_prefix_steps_per_clip': 224,
        'transcript_reserve_characters_each_copy': 32768, 'transcript_copies_per_clip': 2,
        'extra_json_reserve_bytes': MIB}
    hashes = {'recognizer_policy_sha256': digest(read_bytes(policy_path, cap=8 * MIB, readonly=True)),
        'worker_input_binding_sha256': digest(canonical(input_binding)),
        'execution_plan_sha256': digest(canonical(execution_plan)), 'job_sha256': digest(canonical(job))}
    bounded = {'compressed_bytes': 12 * MIB, 'expanded_member_bytes': 32 * MIB,
               'actual_prepared_members': 4096, 'worst_fixture_members': 4096}
    require(type(proof) is dict and set(proof) == set(fixed) | set(hashes) | set(bounded),
            'exact complete prepared evidence budget proof required')
    require(all(exact_json(proof[key], value) for key, value in fixed.items()), 'fixed pre-forward budget semantics')
    require(all(proof[key] == value for key, value in hashes.items()), 'prepared budget input/policy identity drift')
    require(all(type(proof[key]) is int and 1 <= proof[key] <= cap for key, cap in bounded.items()),
            'whole evidence closure exceeds pre-forward publication bounds')
    return proof


def validate_legacy_source_claim(claim_raw, *, expected_sha256, release, source_freeze_sha256,
                          run_id, executed_head_sha):
    """Validate a readonly host claim against independent worker context.

    This does not query Actions, prove prior history or create a release. The
    trusted host must first verify the actual API run/complete branch history,
    absent evidence ref and clean Git source, then durably create the claim.
    """
    require(type(claim_raw) is bytes and len(claim_raw) <= 65536 and
            digest(claim_raw) == sha(expected_sha256), 'actual immutable source claim bytes required')
    claim = decode_json(claim_raw)
    _validate_legacy_release_shape(release)
    require(all(release[key] is True for key in RELEASE_APPROVALS) and
            all(release[key] is not None for key in RELEASE_AUTHORITIES), 'released source authority required')
    sha(source_freeze_sha256)
    require(type(run_id) is str and re.fullmatch(r'[1-9][0-9]{0,19}', run_id), 'actual run identity required')
    require(type(executed_head_sha) is str and HEAD.fullmatch(executed_head_sha), 'actual executed head required')
    require(int(run_id) not in SEALED_ASR_RUNS and executed_head_sha not in SEALED_ASR_SOURCES,
            'sealed earlier ASR identity is not reusable')
    fields = {'schema', 'identity', 'source_freeze_sha256', 'release_sha256', 'source_branch',
        'result_branch', 'workflow', 'numeric_job_id', 'prior_public_claim_history_sha256',
        'api_preflight_verified', 'source_branch_workflow_runs', 'maximum_new_asr_calls',
        'old_calls_allowed', 'new_audio_forwards_at_claim'}
    require(type(claim) is dict and set(claim) == fields, 'exact once-source-claim fields required')
    expected = {'schema': 'k1-asr18-once-source-claim-v1',
        'identity': {'repository': 'jiying2007/kws-pipeline', 'executed_head_sha': executed_head_sha,
                     'run_id': int(run_id), 'run_attempt': 1, 'job_key': 'asr18'},
        'source_freeze_sha256': source_freeze_sha256, 'release_sha256': digest(canonical(release)),
        'source_branch': release['source_ref'], 'result_branch': release['evidence_ref'],
        'workflow': release['workflow'],
        'prior_public_claim_history_sha256': release['prior_public_claim_history_sha256'],
        'api_preflight_verified': True, 'source_branch_workflow_runs': 1,
        'maximum_new_asr_calls': 18, 'old_calls_allowed': 0, 'new_audio_forwards_at_claim': 0}
    require(all(exact_json(claim[key], value) for key, value in expected.items()),
            'actual run/source/release/once-claim binding drift')
    require(type(claim['numeric_job_id']) is int and 1 <= claim['numeric_job_id'] < 10 ** 20,
            'actual positive numeric job identity required')
    return claim


# This source selects v2. Archived v1 is a separately named pure verifier, never
# a fallback chosen by a claim. These helpers do no network, token or model work.
CURRENT_CLAIM_MODE = 'git-owner-v2'
BOUND_RELEASE_SCHEMA = 'k1-asr18-execution-release-v2'
CLAIM_CONTROL_SCHEMA = 'k1-asr18-claim-admission-control-v2'
BOUND_CLAIM_SCHEMA = 'k1-asr18-once-source-claim-v2'
CLAIM_CONTROL_FILE = 'claim-admission-control.json'
MAX_BOUND_ADMISSION = 65536
MAX_HTTP_RESPONSE = 512 * 1024
BOUND_REPOSITORY = 'jiying2007/kws-pipeline'
# Later sealed r3 and r4 identities remain ineligible under v2; no consumption count
# is inferred, and original archived-v1 history semantics stay unchanged.
BOUND_SEALED_ASR_RUNS = SEALED_ASR_RUNS + (38385589303, 38385623017, 38499530026)
BOUND_SEALED_ASR_SOURCES = SEALED_ASR_SOURCES + ('b012fca6bec8e0c2c1cdd5741fe8e43f0ca195b3',
    '270364bd2c953ae312198b326f465b819d294cb9')
BOUND_AUTHOR = {'name': 'K1 ASR18 bounded claim', 'email': 'noreply@github.com'}
BOUND_NONRECOGNITION = {'existing_slots_at_most': 18, 'issuer_model_calls': 0,
    'old_calls_allowed': 0, 'automatic_retries_allowed': 0, 'claim_proves_model_consumption': False}
CLAIM_APPROVALS = ('enabled', 'independent_source_review', 'root_execution_release')
CLAIM_SELECTORS = ('campaign', 'source_ref', 'evidence_ref', 'claim_ref', 'workflow')
BOUND_API_ROW_KEYS = frozenset(('id', 'run_number', 'run_attempt', 'head_sha', 'head_branch',
    'path', 'workflow_id', 'event'))


def current_scope():
    """One compiled current campaign; no source-control read or future authority."""
    require(type(EXPERIMENT) is str and re.fullmatch(r'[A-Za-z0-9_-]+', EXPERIMENT), 'compiled campaign identifier')
    require(type(CURRENT_WORKFLOW) is str and re.fullmatch(r'\.github/workflows/[A-Za-z0-9_-]+\.ya?ml', CURRENT_WORKFLOW),
        'compiled workflow path')
    return {'campaign': EXPERIMENT, 'source_branch': 'research/' + EXPERIMENT,
        'result_branch': 'evidence/' + EXPERIMENT, 'claim_ref': 'refs/heads/claims/' + EXPERIMENT,
        'directory': 'research/k1-punctuation6-asr18-v1/evidence/' + EXPERIMENT.removeprefix('k1-punctuation6-'),
        'workflow': CURRENT_WORKFLOW, 'job_key': 'asr18'}


def require_current_scope(value):
    """Live/full-source boundary only; the pure archived owner parser stays flexible."""
    expected = current_scope()
    require(type(value) is dict, 'source-compiled current scope object')
    if value.get('schema') == CLAIM_CONTROL_SCHEMA:
        validate_claim_control(value, require_enabled=True)
        actual = {'campaign': value['campaign'], 'source_branch': value['source_ref'],
            'result_branch': value['evidence_ref'], 'claim_ref': value['claim_ref'],
            'workflow': value['workflow'], 'job_key': value['job_key']}
        fixed = {key: expected[key] for key in actual}
    elif set(value) == set(expected):
        actual, fixed = value, expected
    else:
        validate_bound_identity(value)
        actual = {key: value[key] for key in expected if key != 'directory'}
        fixed = {key: expected[key] for key in actual}
    require(exact_json(actual, fixed), 'control or owner differs from sole compiled current scope')
    return expected


def disabled_claim_control():
    return {'schema': CLAIM_CONTROL_SCHEMA, 'mode': CURRENT_CLAIM_MODE,
        **{key: False for key in CLAIM_APPROVALS}, **{key: None for key in CLAIM_SELECTORS},
        'repository': BOUND_REPOSITORY, 'job_key': 'asr18', 'workflow_id': None,
        'workflow_id_source': 'actual_run_api_plus_path'}


def validate_claim_control(control, *, require_enabled=False):
    baseline = disabled_claim_control()
    require(type(control) is dict and set(control) == set(baseline), 'exact source claim control fields')
    for key in set(baseline) - set(CLAIM_APPROVALS) - set(CLAIM_SELECTORS):
        require(exact_json(control[key], baseline[key]), 'source claim control scope drift')
    for key in CLAIM_APPROVALS:
        require(type(control[key]) is bool, 'explicit claim control approval')
    for key in CLAIM_SELECTORS:
        value = control[key]
        require(value is None or (type(value) is str and 1 <= len(value) <= 200 and
            re.fullmatch(r'[A-Za-z0-9_./-]+', value) and '..' not in value and
            '//' not in value and not value.endswith(('/', '.', '.lock'))), 'bounded fixed claim selector or null')
    if control['campaign'] is not None:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', control['campaign']), 'fixed campaign identifier')
        for key, prefix in (('source_ref', 'research/'), ('evidence_ref', 'evidence/'), ('claim_ref', 'refs/heads/claims/')):
            require(control[key] is None or control[key] == prefix + control['campaign'], 'campaign-derived fixed ref')
    if control['workflow'] is not None:
        require(re.fullmatch(r'\.github/workflows/[A-Za-z0-9_-]+\.ya?ml', control['workflow']),
            'fixed workflow path')
    if control['claim_ref'] is not None:
        require(control['claim_ref'].startswith('refs/heads/') and
            control['claim_ref'][11:] not in (control['source_ref'], control['evidence_ref']),
            'distinct campaign claim ref')
    if require_enabled:
        require(all(control[key] is True for key in CLAIM_APPROVALS) and
            all(control[key] is not None for key in CLAIM_SELECTORS), 'claim admission closed or future authority absent')
    return control


def bound_prior_history_identity(control):
    """Source-bound prior observations, never an assertion of an empty live history."""
    validate_claim_control(control)
    sealed = []
    for experiment, run_id, source in zip(('k1-punctuation6-asr18-v1', 'k1-punctuation6-asr18-r1',
            'k1-punctuation6-asr18-r2'), SEALED_ASR_RUNS, SEALED_ASR_SOURCES):
        sealed.append({'observation': 'frozen_existing_public_failure_observation', 'experiment': experiment,
            'source_branch': 'research/' + experiment, 'result_branch': 'evidence/' + experiment,
            'run_id': run_id, 'run_attempt': 1, 'executed_head_sha': source,
            'planned_slots': 18, 'actual_consumed_attempts': None, 'reusable_slots': 0,
            'status': 'failed_consumption_unknown_sealed'})
    sealed.append({'observation': 'frozen_existing_public_failure_observation',
        'experiment': 'k1-punctuation6-asr18-r3', 'source_branch': 'research/k1-punctuation6-asr18-r3',
        'result_branch': 'evidence/k1-punctuation6-asr18-r3', 'authorized_campaign_count': 1,
        'planned_slots': 18, 'actual_consumed_attempts': None, 'reusable_slots': 0,
        'observed_runs': [{'run_id': run_id, 'run_attempt': 1,
            'executed_head_sha': 'b012fca6bec8e0c2c1cdd5741fe8e43f0ca195b3'}
            for run_id in (38385589303, 38385623017)], 'status': 'failed_consumption_unknown_sealed'})
    sealed.append({'observation': 'frozen_existing_public_failure_observation',
        'experiment': 'k1-punctuation6-asr18-r4', 'source_branch': 'research/k1-punctuation6-asr18-r4',
        'result_branch': 'evidence/k1-punctuation6-asr18-r4', 'run_id': 38499530026, 'run_attempt': 1,
        'executed_head_sha': '270364bd2c953ae312198b326f465b819d294cb9',
        'planned_slots': 18, 'actual_consumed_attempts': None, 'reusable_slots': 0,
        'status': 'failed_consumption_unknown_sealed'})
    return {'schema': 'k1-asr18-prior-public-claim-history-v2', 'repository': control['repository'],
        'scope': {key: control[key] for key in ('campaign', 'source_ref', 'evidence_ref', 'claim_ref',
            'workflow', 'workflow_id', 'workflow_id_source', 'job_key')},
        'prior_runs': None, 'prior_runs_status': 'unobserved',
        'prior_runs_scope': 'only_source_bound_campaign_branch_and_workflow', 'sealed_prior_campaigns': sealed}


def require_execution_release(release=None, source_freeze=None, *, expected_freeze_sha256=None,
                              prepare_prerequisite_raw=None, source_root=None, source_members=None):
    release = read_json(Path(__file__).resolve().parent / 'execution-release.json') if release is None else release
    require(is_bound_release(release), 'source-selected v2 execution release required')
    return _require_execution_release(release, source_freeze, expected_freeze_sha256=expected_freeze_sha256,
        prepare_prerequisite_raw=prepare_prerequisite_raw, source_root=source_root, source_members=source_members)


def require_legacy_execution_release(release, source_freeze=None, *, expected_freeze_sha256=None,
                                     prepare_prerequisite_raw=None, source_root=None, source_members=None):
    """Archived v1 verification only. Live entrypoints use require_execution_release."""
    _validate_legacy_release_shape(release)
    return _require_execution_release(release, source_freeze, expected_freeze_sha256=expected_freeze_sha256,
        prepare_prerequisite_raw=prepare_prerequisite_raw, source_root=source_root, source_members=source_members)


def current_disabled_release(control_raw=None):
    """Closed v2 template; materialization requires actual source-control bytes."""
    control_raw = canonical(disabled_claim_control()) if control_raw is None else control_raw
    control = validate_claim_control(decode_json(control_raw, require_canonical=False))
    value = disabled_release()
    scope = current_scope()
    value.update(experiment=scope['campaign'], source_ref=scope['source_branch'],
        evidence_ref=scope['result_branch'], workflow=scope['workflow'], job_key=scope['job_key'])
    value.update(schema=BOUND_RELEASE_SCHEMA, claim_admission_mode=CURRENT_CLAIM_MODE,
        claim_admission_control_sha256=digest(control_raw))
    if control['campaign'] is not None:
        value['experiment'] = control['campaign']
    for key in ('source_ref', 'evidence_ref', 'workflow'):
        if control[key] is not None:
            value[key] = control[key]
    return value


def validate_release_shape(value):
    if type(value) is dict and value.get('schema') == 'k1-asr18-execution-release-v1':
        return _validate_legacy_release_shape(value)
    baseline = disabled_release()
    require(type(value) is dict and set(value) == set(baseline) |
        {'claim_admission_mode', 'claim_admission_control_sha256'}, 'exact bound release fields')
    require(value['schema'] == BOUND_RELEASE_SCHEMA and value['claim_admission_mode'] == CURRENT_CLAIM_MODE,
        'explicit source-bound claim policy required')
    sha(value['claim_admission_control_sha256'])
    legacy = {key: value[key] for key in baseline}
    legacy['schema'] = baseline['schema']
    for key in ('experiment', 'source_ref', 'evidence_ref', 'workflow'):
        require(type(legacy[key]) is str and 1 <= len(legacy[key]) <= 200, 'bound release selector')
        legacy[key] = baseline[key]
    _validate_legacy_release_shape(legacy)
    return value


def is_bound_release(release):
    validate_release_shape(release)
    return release['schema'] == BOUND_RELEASE_SCHEMA


def require_claim_admission(release, control_raw):
    require(is_bound_release(release), 'source-bound v2 release required')
    require(type(control_raw) is bytes and len(control_raw) <= 8192 and
        digest(control_raw) == release['claim_admission_control_sha256'], 'actual source claim control bytes')
    control = validate_claim_control(decode_json(control_raw, require_canonical=False), require_enabled=True)
    require(all(release[key] is True for key in RELEASE_APPROVALS) and
        all(release[key] is not None for key in RELEASE_AUTHORITIES), 'released source authority required')
    require(release['experiment'] == control['campaign'] and
        all(exact_json(release[key], control[key]) for key in ('source_ref', 'evidence_ref', 'workflow')),
        'source release and admission scope differ')
    require(release['prior_public_claim_history_sha256'] == digest(canonical(bound_prior_history_identity(control))),
        'source-bound prior history and unknown consumption pin')
    return control


def _positive(value):
    return type(value) is int and 0 < value < 10 ** 20


def _timestamp(value):
    require(type(value) is str and re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z', value),
        'actual UTC source commit timestamp')
    return int(datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp())


def sourcebound_identity(run, numeric_job_id, release, source_tree, source_timestamp,
        source_freeze_sha256, release_sha256, control, workflow_id, run_number=1):
    require(is_bound_release(release), 'source-bound v2 identity release')
    validate_claim_control(control, require_enabled=True)
    require(type(run) is dict and set(run) == {'repository', 'run_id', 'run_attempt', 'job_key', 'executed_head_sha'},
        'exact independently supplied run identity')
    expected = {'repository': BOUND_REPOSITORY, 'run_id': run['run_id'], 'run_attempt': 1,
        'job_key': control['job_key'], 'executed_head_sha': run['executed_head_sha']}
    require(exact_json(run, expected), 'actual fixed eligible owner run')
    require(release['experiment'] == control['campaign'] and
        all(release[key] == control[key] for key in ('source_ref', 'evidence_ref', 'workflow')),
        'source control scope mismatch')
    identity = {'campaign': control['campaign'], 'repository': BOUND_REPOSITORY,
        'source_branch': control['source_ref'], 'result_branch': control['evidence_ref'],
        'claim_ref': control['claim_ref'], 'workflow': control['workflow'], 'workflow_id': workflow_id,
        'job_key': control['job_key'], 'run_attempt': 1, 'event': 'push',
        'source_sha': run['executed_head_sha'], 'source_tree': source_tree, 'source_timestamp': source_timestamp,
        'freeze_sha256': source_freeze_sha256, 'release_sha256': release_sha256,
        'input_sha256': release['actual_input_freeze_sha256'],
        'prerequisite_sha256': release['prepare_prerequisite_sha256'],
        'prior_history_sha256': release['prior_public_claim_history_sha256'],
        'claim_control_sha256': release['claim_admission_control_sha256'],
        'run_id': run['run_id'], 'job_id': numeric_job_id, 'run_number': run_number}
    return validate_bound_identity(identity)


def validate_bound_identity(identity):
    fixed = {'repository': BOUND_REPOSITORY, 'job_key': 'asr18', 'run_attempt': 1, 'run_number': 1, 'event': 'push'}
    selectors = {'campaign', 'source_branch', 'result_branch', 'claim_ref', 'workflow'}
    hashes = {'freeze_sha256', 'release_sha256', 'input_sha256', 'prerequisite_sha256',
        'prior_history_sha256', 'claim_control_sha256'}
    fields = set(fixed) | selectors | hashes | {'source_sha', 'source_tree', 'source_timestamp', 'run_id', 'job_id', 'workflow_id'}
    require(type(identity) is dict and set(identity) == fields, 'exact source-bound owner identity')
    require(all(exact_json(identity[key], value) for key, value in fixed.items()), 'eligible attempt1 ordinal1 owner only')
    require(all(_positive(identity[key]) for key in ('run_id', 'job_id', 'workflow_id')), 'actual numeric run/job/workflow')
    require(identity['run_id'] not in BOUND_SEALED_ASR_RUNS and identity['source_sha'] not in BOUND_SEALED_ASR_SOURCES,
        'sealed source owner is not reusable')
    for key in hashes:
        sha(identity[key])
    for key in ('source_sha', 'source_tree'):
        require(type(identity[key]) is str and HEAD.fullmatch(identity[key]), 'actual Git source identity')
    _timestamp(identity['source_timestamp'])
    control = disabled_claim_control()
    control.update({key: True for key in CLAIM_APPROVALS})
    control.update(campaign=identity['campaign'], source_ref=identity['source_branch'],
        evidence_ref=identity['result_branch'], claim_ref=identity['claim_ref'], workflow=identity['workflow'])
    validate_claim_control(control, require_enabled=True)
    return identity


def bound_run_row(identity):
    validate_bound_identity(identity)
    return {'id': identity['run_id'], 'run_number': identity['run_number'], 'run_attempt': identity['run_attempt'],
        'head_sha': identity['source_sha'], 'head_branch': identity['source_branch'],
        'path': identity['workflow'], 'workflow_id': identity['workflow_id'], 'event': identity['event']}


def _observed_run_fields(row, identity):
    require(type(row) is dict and BOUND_API_ROW_KEYS | {'status', 'conclusion', 'repository', 'head_repository'} <= set(row),
        'required real API row identity and observed state')
    for key in ('repository', 'head_repository'):
        require(type(row.get(key)) is dict and row[key].get('full_name') == identity['repository'] and
            row[key].get('private') is False, 'public same-repository run history')
    require(row.get('status') in ('queued', 'in_progress', 'completed', 'waiting', 'requested', 'pending')
        and row.get('conclusion') in (None, 'success', 'failure', 'cancelled', 'timed_out', 'neutral',
        'skipped', 'action_required', 'stale', 'startup_failure'), 'observed status and conclusion')
    return {**{key: row[key] for key in BOUND_API_ROW_KEYS},
        'status': row['status'], 'conclusion': row['conclusion'],
        **{key: {'full_name': row[key]['full_name'], 'private': row[key]['private']}
           for key in ('repository', 'head_repository')}}


def bound_history(identity, rows, phase, *, raw_sha256=None):
    expected = bound_run_row(identity)
    sha(raw_sha256)
    require(phase in ('acquisition', 'host', 'publication', 'posterior'), 'fixed history observation phase')
    require(type(rows) is list and 1 <= len(rows) <= 100, 'complete bounded history rows')
    ids, ordinals = set(), set()
    for row in rows:
        require(exact_json(row, _observed_run_fields(row, identity)), 'real bounded API history fields only')
        for key, value in expected.items():
            require(_positive(row[key]) if key in ('id', 'run_number') else exact_json(row[key], value),
                'fixed complete history scope')
        require(row['id'] not in ids and row['run_number'] not in ordinals, 'unique history IDs and ordinals')
        ids.add(row['id']); ordinals.add(row['run_number'])
    require(sum(all(exact_json(row[key], value) for key, value in expected.items()) for row in rows) == 1,
        'exact ordinal1 owner occurs once')
    rows = sorted((dict(row) for row in rows), key=lambda row: row['id'])
    projection = {'total_count': len(rows), 'workflow_runs': rows, 'complete': True}
    return {'schema': 'k1-asr18-complete-history-v2', 'phase': phase,
        'scope': {key: identity[key] for key in ('repository', 'workflow', 'source_branch')},
        'total_count': len(rows), 'rows': rows, 'projection_sha256': digest(canonical(projection)),
        'raw_response_sha256': raw_sha256}


def validate_bound_history(identity, observation, phase):
    require(type(observation) is dict and exact_json(observation,
        bound_history(identity, observation.get('rows'), phase, raw_sha256=observation.get('raw_response_sha256'))),
        'truthful complete history observation')
    return observation


def api_bound_history(identity, page, phase, *, raw_sha256=None):
    require(type(page) is dict and type(page.get('total_count')) is int and
        type(page.get('workflow_runs')) is list and 1 <= page['total_count'] == len(page['workflow_runs']) <= 100,
        'untruncated counted complete API history')
    rows = [_observed_run_fields(row, identity) for row in page['workflow_runs']]
    return bound_history(identity, rows, phase, raw_sha256=raw_sha256)


def owner_record(identity, acquisition_history, nonce_sha256):
    validate_bound_identity(identity)
    validate_bound_history(identity, acquisition_history, 'acquisition')
    sha(nonce_sha256)
    return {'schema': 'k1-asr18-owner-v2', 'claim_ref': identity['claim_ref'], 'identity': dict(identity),
        'acquisition_history_sha256': digest(canonical(acquisition_history)), 'nonce_sha256': nonce_sha256,
        'nonrecognition': dict(BOUND_NONRECOGNITION)}


def validate_owner(owner, identity, acquisition_history):
    require(type(owner) is dict and exact_json(owner, owner_record(identity, acquisition_history,
        owner.get('nonce_sha256'))), 'exact bound owner and acquisition history')
    return owner


def owner_commit(owner):
    require(type(owner) is dict and set(owner) == {'schema', 'claim_ref', 'identity',
        'acquisition_history_sha256', 'nonce_sha256', 'nonrecognition'}, 'exact owner commit fields')
    identity = validate_bound_identity(owner['identity'])
    require(owner['schema'] == 'k1-asr18-owner-v2' and owner['claim_ref'] == identity['claim_ref'] and
        exact_json(owner['nonrecognition'], BOUND_NONRECOGNITION), 'owner commit policy')
    sha(owner['acquisition_history_sha256']); sha(owner['nonce_sha256'])
    message = canonical(owner).decode('utf-8') + '\n'
    require(len(message.encode('utf-8')) <= 15360, 'bounded owner commit message')
    author = {**BOUND_AUTHOR, 'date': identity['source_timestamp']}
    body = {'message': message, 'tree': identity['source_tree'], 'parents': [identity['source_sha']],
        'author': author, 'committer': dict(author)}
    person = author['name'] + ' <' + author['email'] + '> ' + str(_timestamp(author['date'])) + ' +0000'
    raw = ('tree ' + body['tree'] + '\nparent ' + identity['source_sha'] + '\nauthor ' + person +
        '\ncommitter ' + person + '\n\n' + message).encode('utf-8')
    commit_sha = hashlib.sha1(b'commit ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    return body, commit_sha


def _git_api_fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
        'public Git API fields only; unknown or private fields prohibited')


def _git_api_metadata(value, *, url=None, html_url=None):
    if 'node_id' in value:
        node = value['node_id']
        require(type(node) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_+=/-]{0,255}', node)
            and not node.startswith(('ghp_', 'github_pat_', 'Bearer')), 'bounded public Git node identifier')
    for key, expected in (('url', url), ('html_url', html_url)):
        if key in value:
            require(type(value[key]) is str and expected is not None and value[key] == expected,
                'exact same-repository public Git locator')


def validate_claim_ref(value, owner):
    _, commit_sha = owner_commit(owner)
    _git_api_fields(value, ('ref', 'object'), ('node_id', 'url'))
    _git_api_fields(value['object'], ('type', 'sha'), ('url',))
    root = 'https://api.github.com/repos/' + owner['identity']['repository'] + '/git/'
    _git_api_metadata(value, url=root + owner['claim_ref'])
    _git_api_metadata(value['object'], url=root + 'commits/' + commit_sha)
    require(value['ref'] == owner['claim_ref'] and value['object']['type'] == 'commit' and
        value['object']['sha'] == commit_sha, 'actual nested Git claim ref exact owner')
    return commit_sha


def verify_owner_commit(value, owner):
    body, commit_sha = owner_commit(owner)
    _git_api_fields(value, ('sha', 'message', 'tree', 'parents', 'author', 'committer'),
        ('node_id', 'url', 'html_url', 'verification'))
    root = 'https://api.github.com/repos/' + owner['identity']['repository'] + '/git/'
    web = 'https://github.com/' + owner['identity']['repository'] + '/commit/'
    _git_api_metadata(value, url=root + 'commits/' + commit_sha, html_url=web + commit_sha)
    _git_api_fields(value['tree'], ('sha',), ('url',))
    _git_api_metadata(value['tree'], url=root + 'trees/' + body['tree'])
    require(value['sha'] == commit_sha and value['tree']['sha'] == body['tree'], 'actual Git owner commit tree/SHA')
    require(type(value['parents']) is list and len(value['parents']) == 1, 'actual Git owner sole source parent')
    parent = value['parents'][0]
    _git_api_fields(parent, ('sha',), ('url', 'html_url'))
    _git_api_metadata(parent, url=root + 'commits/' + body['parents'][0], html_url=web + body['parents'][0])
    require(parent['sha'] == body['parents'][0], 'actual Git owner sole source parent')
    # GitHub's commit API may omit the one terminal LF. No general strip/replace.
    require(value['message'] in (body['message'], body['message'][:-1]), 'exact owner message and terminal LF')
    for key in ('author', 'committer'):
        person = value[key]
        _git_api_fields(person, ('name', 'email', 'date'))
        require(all(person[k] == body[key][k] for k in ('name', 'email')) and
            person['date'] in (body[key]['date'], body[key]['date'][:-1] + '+00:00'), 'actual deterministic Git owner person')
    if 'verification' in value:
        verification = value['verification']
        _git_api_fields(verification, ('verified', 'reason', 'signature', 'payload'), ('verified_at',))
        require(verification['verified'] is False and verification['reason'] == 'unsigned' and
            verification['signature'] is None and verification['payload'] is None and
            verification.get('verified_at') is None, 'fixed self-created unsigned owner verification only')
    return commit_sha


def response_receipt(status, raw, *, route):
    require(type(status) is int and 100 <= status <= 599 and type(raw) is bytes and
        len(raw) <= MAX_HTTP_RESPONSE and type(route) is str and route.startswith('/repos/' + BOUND_REPOSITORY + '/'),
        'bounded actual HTTP body and exact repository route')
    value = {'schema': 'k1-asr18-http-response-receipt-v2', 'provenance': 'actual-http-response-body',
        'route': route, 'status': status, 'raw_body_utf8': raw.decode('utf-8'),
        'body_bytes': len(raw), 'body_sha256': digest(raw)}
    return {**value, 'receipt_sha256': digest(canonical(value))}


def validate_response_receipt(receipt, status, route):
    require(type(receipt) is dict and type(receipt.get('raw_body_utf8')) is str, 'original HTTP receipt object')
    raw = receipt['raw_body_utf8'].encode('utf-8')
    require(exact_json(receipt, response_receipt(status, raw, route=route)), 'original HTTP bytes/status/route/hash')
    return decode_json(raw, require_canonical=False)


def validate_acquisition_responses(responses, owner):
    require(type(responses) is dict and set(responses) == {'ref_create', 'ref_readback', 'owner_readback'},
        'exact original HTTP response roster')
    _, commit_sha = owner_commit(owner)
    root = '/repos/' + owner['identity']['repository'] + '/git/'
    routes = {'ref_create': root + 'refs', 'ref_readback': root + 'ref/' + owner['claim_ref'][5:],
        'owner_readback': root + 'commits/' + commit_sha}
    for key in responses:
        value = validate_response_receipt(responses[key], 201 if key == 'ref_create' else 200, routes[key])
        (verify_owner_commit if key == 'owner_readback' else validate_claim_ref)(value, owner)
    return {key: responses[key]['receipt_sha256'] for key in responses}


def validate_bound_admission(admission, identity):
    """Public observations are evidence, never private nonce/marker authorization."""
    validate_bound_identity(identity)
    require(type(admission) is dict and set(admission) == {'owner', 'commit', 'acquisition_history',
        'acquisition_responses', 'host_history', 'consumed'} and len(canonical(admission)) <= MAX_BOUND_ADMISSION,
        'complete bounded consumed admission evidence')
    validate_owner(admission['owner'], identity, admission['acquisition_history'])
    validate_bound_history(identity, admission['host_history'], 'host')
    body, commit_sha = owner_commit(admission['owner'])
    require(exact_json(admission['commit'], {'sha': commit_sha, 'body': body}), 'exact deterministic owner Git commit')
    validate_acquisition_responses(admission['acquisition_responses'], admission['owner'])
    require(exact_json(admission['consumed'], {'schema': 'k1-asr18-consumed-observation-v2',
        'owner_commit': commit_sha, 'identity': identity, 'permanent_marker_durable': True,
        'model_consumption': None}), 'sanitized consumption observation does not establish model use')
    return admission


def make_source_claim(run, release, release_raw, control_raw, numeric_job_id, source_tree,
        source_timestamp, source_freeze_sha256, workflow_id, complete_api_history, admission):
    control = require_claim_admission(release, control_raw)
    require(type(release_raw) is bytes and len(release_raw) <= 16384 and
        exact_json(decode_json(release_raw, require_canonical=False), release), 'actual release bytes and parsed source control')
    identity = sourcebound_identity(run, numeric_job_id, release, source_tree, source_timestamp,
        source_freeze_sha256, digest(release_raw), control, workflow_id)
    validate_bound_admission(admission, identity)
    history = api_bound_history(identity, complete_api_history, 'host',
        raw_sha256=admission['host_history']['raw_response_sha256'])
    require(exact_json(history, admission['host_history']), 'true complete host history retained')
    claim = {'schema': BOUND_CLAIM_SCHEMA, 'identity': run, 'source_freeze_sha256': source_freeze_sha256,
        'release_sha256': digest(release_raw), 'release_raw_utf8': release_raw.decode('utf-8'),
        'control_raw_utf8': control_raw.decode('utf-8'), 'source_tree': source_tree,
        'source_timestamp': source_timestamp, 'numeric_job_id': numeric_job_id, 'workflow_id': workflow_id,
        'source_branch_workflow_runs': history['total_count'],
        'complete_api_history': {'total_count': history['total_count'], 'workflow_runs': history['rows']},
        'admission': admission, 'maximum_new_asr_calls': 18, 'old_calls_allowed': 0, 'new_audio_forwards_at_claim': 0}
    require(len(canonical(claim)) <= 65536, 'existing immutable source claim cap')
    return claim


def validate_source_claim(claim_raw, *, expected_sha256, release, source_freeze_sha256,
                          run_id, executed_head_sha):
    """Pure worker-compatible v2 parser. Host/archive separately bind RAW members.

    The unchanged worker independently receives run/head/freeze and parsed release.
    Numeric job/tree and exact raw release membership are independently checked by
    host, publisher and posterior callers through bind_source_claim. Public claim
    validity alone never permits constructing or consuming a private handoff.
    """
    require(CURRENT_CLAIM_MODE == 'git-owner-v2' and is_bound_release(release), 'source-selected v2 only; no downgrade')
    require(type(claim_raw) is bytes and len(claim_raw) <= 65536 and digest(claim_raw) == sha(expected_sha256),
        'actual immutable bound source claim bytes')
    claim = decode_json(claim_raw)
    require(type(claim) is dict and type(claim.get('release_raw_utf8')) is str and
        type(claim.get('control_raw_utf8')) is str and type(claim.get('identity')) is dict,
        'complete bound source claim')
    require(type(run_id) is str and re.fullmatch(r'[1-9][0-9]{0,19}', run_id) and
        claim['identity'].get('run_id') == int(run_id) and claim['identity'].get('executed_head_sha') == executed_head_sha and
        claim.get('source_freeze_sha256') == sha(source_freeze_sha256), 'independent actual run/head/freeze binding')
    expected = make_source_claim(claim['identity'], release, claim['release_raw_utf8'].encode('utf-8'),
        claim['control_raw_utf8'].encode('utf-8'), claim.get('numeric_job_id'), claim.get('source_tree'),
        claim.get('source_timestamp'), source_freeze_sha256, claim.get('workflow_id'),
        claim.get('complete_api_history'), claim.get('admission'))
    require(exact_json(claim, expected), 'exact source-bound v2 claim fields')
    return claim


def bind_source_claim(claim_raw, run, release, numeric_job_id, source_tree, *,
                      release_raw, control_raw, source_freeze_sha256):
    claim = validate_source_claim(claim_raw, expected_sha256=digest(claim_raw), release=release,
        source_freeze_sha256=source_freeze_sha256, run_id=str(run['run_id']),
        executed_head_sha=run['executed_head_sha'])
    require(exact_json(claim['identity'], run) and type(numeric_job_id) is int and
        claim['numeric_job_id'] == numeric_job_id and claim['source_tree'] == source_tree,
        'independent current numeric job and actual source tree')
    require(type(release_raw) is bytes and claim['release_raw_utf8'].encode('utf-8') == release_raw and
        type(control_raw) is bytes and claim['control_raw_utf8'].encode('utf-8') == control_raw,
        'immutable source/archive RAW release and control members')
    return claim
