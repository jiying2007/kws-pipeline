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

import asr18_metadata as metadata

EXPERIMENT = 'k1-punctuation6-asr18-r2'
SEALED_ASR_RUNS = (38207026177, 38219171568)
SEALED_ASR_SOURCES = ('0f3cc89a3235c4b3af294dc6b5816ff9877c0f42',
                      '054213c551eb83494b9a4ba8cd6d3ca4c1f488dd')
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
    return {'schema': 'k1-asr18-execution-release-v1', 'experiment': EXPERIMENT,
        **{key: False for key in RELEASE_APPROVALS}, **{key: None for key in RELEASE_AUTHORITIES},
        'max_new_asr_calls': 18, 'old_calls_allowed': 0, 'retries': 0, 'run_attempt': 1,
        'source_ref': 'research/k1-punctuation6-asr18-r2',
        'evidence_ref': 'evidence/k1-punctuation6-asr18-r2',
        'workflow': '.github/workflows/k1-punctuation6-asr18-v1.yml', 'job_key': 'asr18'}


def validate_release_shape(value):
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


def require_execution_release(release=None, source_freeze=None, *, expected_freeze_sha256=None,
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


def validate_source_claim(claim_raw, *, expected_sha256, release, source_freeze_sha256,
                          run_id, executed_head_sha):
    """Validate a readonly host claim against independent worker context.

    This does not query Actions, prove prior history or create a release. The
    trusted host must first verify the actual API run/complete branch history,
    absent evidence ref and clean Git source, then durably create the claim.
    """
    require(type(claim_raw) is bytes and len(claim_raw) <= 65536 and
            digest(claim_raw) == sha(expected_sha256), 'actual immutable source claim bytes required')
    claim = decode_json(claim_raw)
    validate_release_shape(release)
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
