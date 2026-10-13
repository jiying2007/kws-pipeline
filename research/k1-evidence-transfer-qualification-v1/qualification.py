"""Two fixed hosted transfer-only phases. Synthetic bytes never enter ASR.

The first phase reads a single immutable public fixture, retains a new checkpoint
using the production bounded HTTP writer, verifies it, and deliberately exits.
The second phase requires a new independently reviewed static source pin and
reconstructs only from that checkpoint. No input generator or model is imported.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASR = ROOT / 'research/k1-punctuation6-asr18-v1'
for name in (str(HERE), str(ASR)):
    if name not in sys.path:
        sys.path.insert(0, name)
import source_authority as authority
import evidence_publisher as publisher
import evidence_transport as transport

if (Path(authority.__file__).resolve() != HERE / 'source_authority.py'
        or Path(publisher.__file__).resolve() != ASR / 'evidence_publisher.py'
        or Path(transport.__file__).resolve() != ASR / 'evidence_transport.py'):
    raise ValueError('fixed reviewed transfer module origins')

require = authority.require
canonical = authority.canonical
REPOSITORY = 'jiying2007/kws-pipeline'
BASE = '/repos/' + REPOSITORY
WORKFLOW = authority.WORKFLOW
PHASES = {
    'checkpoint': 'research/k1-hosted-transfer-checkpoint-v1',
    'recovery': 'research/k1-hosted-transfer-recovery-v1',
}
CHECKPOINT = 'evidence/k1-hosted-transfer-qualification-v1-checkpoint'
FINAL = 'evidence/k1-hosted-transfer-qualification-v1-recovered'
DIRECTORY = 'evidence/k1-hosted-transfer-qualification-v1'
MANIFEST = DIRECTORY + '/manifest.json'
INPUT_COMMIT = '8b805105810518f849a42fcc6db5d2fd331b8e33'
INPUT_TREE = 'aeea9ae50c067b11b1d2cabf5bb9000a31d2d1f2'
INPUT_MANIFEST_SHA256 = 'c30b9d8ecd2b7b2d58bff5a1a14fcc436f72f02e3e7ea0f21881cb5c0b9acc58'
ARCHIVE_BYTES = 5_884_370
ARCHIVE_SHA256 = '5a75f40a03388dffa6366aa5a39b740582f4b6ea9f36f0d7212d17a1ad8ab901'
CHUNKS = 23
CHUNK_CAP = 256 * 1024
MANIFEST_CAP = 16 * 1024
WALL_SECONDS = 600
API_CAPS = {'checkpoint': 40, 'recovery': 17}
JOB_KEY = 'transfer_qualification'
FALSE_FLAGS = ('model_execution_enabled', 'asr_enabled', 'tts_enabled', 'training_enabled',
               'new_credentials_enabled', 'artifact_enabled', 'payload_logging_enabled',
               'model_download_enabled', 'model_load_enabled', 'native_forward_enabled', 'decode_enabled', 'native_retry_enabled')
CONTEXT_KEYS = ('GITHUB_ACTIONS', 'GITHUB_REPOSITORY', 'GITHUB_EVENT_NAME', 'GITHUB_EVENT_PATH',
    'GITHUB_REF', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB', 'GITHUB_SHA',
    'GITHUB_WORKFLOW_REF', 'GITHUB_WORKFLOW_SHA', 'KWS_TRANSFER_SOURCE_FREEZE_SHA256')
FAILURE_STAGES = frozenset(('arguments', 'control_read', 'control_gate', 'event_read', 'source_admission',
    'token_admission', 'api_preflight', 'verify_actual_run_source', 'input_fixture_read',
    'verify_destination_absent', 'create_blobs', 'create_tree', 'create_commit', 'create_checkpoint_ref',
    'immutable_checkpoint_readback', 'checkpoint_verified', 'verify_original_producer',
    'verify_checkpoint', 'checkpoint_body_read', 'create_ref', 'immutable_postpublish_readback'))


class QualificationDisabled(ValueError):
    pass


class InjectedCheckpointStop(Exception):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('fixed checkpoint readback fault injection')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, cap=65536):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary fixed JSON file')
    raw = path.read_bytes()
    require(len(raw) <= cap, 'bounded JSON file')
    value = transport.decode(raw)
    require(type(value) is dict and canonical(value) == raw, 'canonical JSON object')
    return value


def identity(value, phase):
    require(phase in PHASES and type(value) is dict and set(value) == {
        'repository', 'run_id', 'run_attempt', 'job_key', 'executed_head_sha'}, 'qualification identity fields')
    require(value['repository'] == REPOSITORY and value['job_key'] == JOB_KEY,
            'fixed qualification repository/job')
    require(type(value['run_id']) is int and 0 < value['run_id'] < 10 ** 20,
            'actual numeric qualification run')
    require(type(value['run_attempt']) is int and value['run_attempt'] == 1, 'attempt one')
    require(type(value['executed_head_sha']) is str and transport.HEAD.fullmatch(value['executed_head_sha']),
            'exact executed source')
    return value


def validate_pin(pin):
    fields = {'identity', 'numeric_job_id', 'source_tree', 'source_freeze_sha256',
              'commit_timestamp', 'checkpoint_commit', 'checkpoint_tree', 'manifest_sha256',
              'archive_bytes', 'archive_sha256', 'checkpoint_readback_verified',
              'producer_terminal_conclusion', 'injected_stop_verified'}
    require(type(pin) is dict and set(pin) == fields, 'root-reviewed exact producer pin required')
    identity(pin['identity'], 'checkpoint')
    require(type(pin['numeric_job_id']) is int and pin['numeric_job_id'] > 0, 'original numeric job pin')
    for key in ('source_tree', 'checkpoint_commit', 'checkpoint_tree'):
        require(type(pin[key]) is str and transport.HEAD.fullmatch(pin[key]), 'exact Git pin')
    for key in ('source_freeze_sha256', 'manifest_sha256'):
        require(type(pin[key]) is str and transport.SHA.fullmatch(pin[key]), 'exact SHA256 pin')
    require(pin['archive_bytes'] == ARCHIVE_BYTES and type(pin['archive_bytes']) is int
            and pin['archive_sha256'] == ARCHIVE_SHA256, 'same immutable fixture archive')
    require(pin['checkpoint_readback_verified'] is True and pin['injected_stop_verified'] is True
            and pin['producer_terminal_conclusion'] == 'failure', 'root accepted deliberate producer stop')
    require(type(pin['commit_timestamp']) is str and re.fullmatch(
        r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z', pin['commit_timestamp']), 'exact producer UTC timestamp')
    datetime.strptime(pin['commit_timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    return pin


def control_gate(control, phase):
    require(phase in PHASES and type(control) is dict and set(control) == {
        'schema', 'checkpoint_enabled', 'recovery_enabled', 'root_transfer_release',
        'independent_source_review', 'producer_pin', *FALSE_FLAGS}, 'fixed qualification control fields')
    require(control['schema'] == 'k1-hosted-transfer-qualification-control-v1', 'qualification control schema')
    require(all(control[k] is False for k in FALSE_FLAGS), 'all model/storage/credential flags remain false')
    require(all(type(control[k]) is bool for k in ('checkpoint_enabled', 'recovery_enabled',
        'root_transfer_release', 'independent_source_review')), 'strict qualification approval booleans')
    require(not (control['checkpoint_enabled'] and control['recovery_enabled']), 'separate reviewed source phases')
    if not all(control[k] is True for k in (phase + '_enabled', 'root_transfer_release', 'independent_source_review')):
        raise QualificationDisabled('qualification source is disabled')
    if phase == 'checkpoint':
        require(control['producer_pin'] is None and control['recovery_enabled'] is False,
                'checkpoint source cannot activate an unobserved recovery')
    else:
        validate_pin(control['producer_pin'])
        require(control['checkpoint_enabled'] is False, 'producer cannot restart from recovery source')


def context(env, event, phase):
    require(phase in PHASES and type(env) is dict and type(event) is dict, 'fixed phase context')
    require(env.get('GITHUB_ACTIONS') == 'true' and env.get('GITHUB_REPOSITORY') == REPOSITORY,
            'canonical Actions repository')
    repo = event.get('repository')
    require(type(repo) is dict and repo.get('full_name') == REPOSITORY and repo.get('private') is False,
            'public repository only')
    require(env.get('GITHUB_EVENT_NAME') == 'push' and event.get('created') is True
            and event.get('deleted') is False and event.get('forced') is False
            and event.get('before') == '0' * 40, 'fresh branch creation only')
    ref = 'refs/heads/' + PHASES[phase]
    require(event.get('ref') == ref == env.get('GITHUB_REF'), 'fixed phase source ref')
    require(env.get('GITHUB_RUN_ATTEMPT') == '1' and type(env.get('GITHUB_RUN_ID')) is str
            and re.fullmatch(r'[1-9][0-9]{0,19}', env['GITHUB_RUN_ID']), 'actual run attempt one')
    result = {'repository': REPOSITORY, 'run_id': int(env['GITHUB_RUN_ID']), 'run_attempt': 1,
              'job_key': env.get('GITHUB_JOB'), 'executed_head_sha': env.get('GITHUB_SHA')}
    identity(result, phase)
    require(event.get('after') == result['executed_head_sha'], 'same exact push source')
    require(env.get('GITHUB_WORKFLOW_SHA') == result['executed_head_sha']
            and env.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/' + WORKFLOW + '@' + ref,
            'executed exact workflow source')
    return result


def run_routes(who, phase):
    identity(who, phase)
    run = str(who['run_id'])
    return (BASE + '/actions/runs/' + run,
            BASE + '/actions/workflows/' + WORKFLOW.split('/')[-1] + '/runs?event=push&branch=' + PHASES[phase] + '&per_page=100',
            BASE + '/actions/runs/' + run + '/attempts/1/jobs?per_page=100')


def verify_run(api, who, phase, *, terminal=False):
    """Verify live consumer or pinned failed producer without replacing identity."""
    routes = run_routes(who, phase)
    expected = {'id': who['run_id'], 'run_attempt': 1, 'head_sha': who['executed_head_sha'],
                'head_branch': PHASES[phase], 'event': 'push', 'path': WORKFLOW,
                'status': 'completed' if terminal else 'in_progress'}
    if terminal:
        expected['conclusion'] = 'failure'
    run = api('GET', routes[0])
    publisher._equal(run, expected, 'exact actual qualification run')
    publisher._equal(run.get('repository'), {'full_name': REPOSITORY, 'private': False}, 'actual public run')
    history = api('GET', routes[1])
    require(type(history.get('total_count')) is int and history['total_count'] == 1
            and type(history.get('workflow_runs')) is list and len(history['workflow_runs']) == 1,
            'one fresh source run only')
    publisher._equal(history['workflow_runs'][0], expected, 'complete phase history')
    jobs = api('GET', routes[2])
    require(type(jobs.get('total_count')) is int and jobs['total_count'] == 1
            and type(jobs.get('jobs')) is list and len(jobs['jobs']) == jobs['total_count'],
            'fixed nonmatrix workflow jobs')
    matches = [row for row in jobs['jobs'] if type(row) is dict and row.get('name') == JOB_KEY]
    require(len(matches) == 1, 'one exact phase job')
    job = matches[0]
    publisher._equal(job, {'run_id': who['run_id'], 'head_sha': who['executed_head_sha'],
        'head_branch': PHASES[phase], 'run_url': 'https://api.github.com' + routes[0],
        'status': 'completed' if terminal else 'in_progress',
        'conclusion': 'failure' if terminal else None}, 'actual qualification numeric job')
    require(type(job.get('id')) is int and job['id'] > 0, 'positive numeric job')
    if 'run_attempt' in job:
        require(type(job['run_attempt']) is int and job['run_attempt'] == 1, 'job attempt one')
    return job['id']


def verify_source(api, sha, tree):
    value = api('GET', BASE + '/git/commits/' + sha)
    publisher._equal(value, {'sha': sha}, 'exact source commit')
    publisher._equal(value.get('tree'), {'sha': tree}, 'exact source tree')
    return publisher.verify_tree(api('GET', BASE + '/git/trees/' + tree + '?recursive=1'), tree)


def source_release(env, event, phase, control):
    control_gate(control, phase)
    who = context(env, event, phase)
    for filename in ('execution-release.json', 'publication-release.json'):
        value = transport.decode((ASR / filename).read_bytes())
        root_flag = 'root_execution_release' if filename == 'execution-release.json' else 'root_publication_release'
        require(value.get('enabled') is False and value.get('independent_source_review') is False
                and value.get(root_flag) is False,
                'original ASR source remains disabled')
    require(publisher._source_git('status', '--porcelain', '--untracked-files=all') == '', 'clean source checkout')
    head = publisher._source_git('rev-parse', 'HEAD')
    tree = publisher._source_git('rev-parse', 'HEAD^{tree}')
    require(head == who['executed_head_sha'] and transport.HEAD.fullmatch(tree), 'clean exact source identity')
    freeze = (HERE / 'source-freeze.json').read_bytes()
    require(len(freeze) <= 128 * 1024 and digest(freeze) == env.get('KWS_TRANSFER_SOURCE_FREEZE_SHA256')
            and canonical(authority.build(ROOT)) == freeze, 'independently reviewed fixed source closure')
    return {'identity': who, 'source_tree': tree, 'source_freeze_sha256': digest(freeze),
            'commit_timestamp': publisher._source_timestamp(event, head)}


def fixture_lock():
    value = read_json(HERE / 'fixture-lock.json')
    require(set(value) == {'schema', 'repository', 'commit', 'tree', 'archive_bytes', 'archive_sha256', 'files'}
        and value['schema'] == 'k1-hosted-transfer-fixed-input-v1' and value['repository'] == REPOSITORY
        and value['commit'] == INPUT_COMMIT and value['tree'] == INPUT_TREE
        and value['archive_bytes'] == ARCHIVE_BYTES and value['archive_sha256'] == ARCHIVE_SHA256,
        'fixed original synthetic fixture only')
    rows = value['files']
    require(type(rows) is list and len(rows) == CHUNKS + 1, 'fixed 24 input files')
    expected = [f'evidence/k1-transfer-qualification-v1/chunks/{i:04d}.bin' for i in range(CHUNKS)]
    expected += ['evidence/k1-transfer-qualification-v1/manifest.json']
    require([row.get('path') for row in rows] == expected, 'exact ordered input paths')
    for index, row in enumerate(rows):
        require(type(row) is dict and set(row) == {'path', 'bytes', 'mode', 'sha', 'sha256', 'type'}
            and row['mode'] == '100644' and row['type'] == 'blob', 'fixed ordinary input blob')
        require(type(row['bytes']) is int and 0 < row['bytes'] <= CHUNK_CAP
            and type(row['sha']) is str and transport.HEAD.fullmatch(row['sha'])
            and type(row['sha256']) is str and transport.SHA.fullmatch(row['sha256']), 'bounded input identity')
    require(rows[-1]['bytes'] == 5552 and rows[-1]['sha256'] == INPUT_MANIFEST_SHA256,
            'immutable original manifest identity')
    return value


def output_paths():
    return {MANIFEST: MANIFEST_CAP, **{f'{DIRECTORY}/chunks/{i:04d}.bin': CHUNK_CAP for i in range(CHUNKS)}}


def validate_payload(payload):
    require(type(payload) is bytes and len(payload) == ARCHIVE_BYTES and digest(payload) == ARCHIVE_SHA256,
            'exact synthetic qualification bytes; not an ASR archive')


def read_fixture(read):
    lock = fixture_lock()
    values = {}
    for row in lock['files']:
        raw = read(row['path'])
        require(type(raw) is bytes and len(raw) == row['bytes'] and digest(raw) == row['sha256']
                and transport.git_sha(raw) == row['sha'], 'exact immutable input body')
        values[row['path']] = raw
    payload = b''.join(values[row['path']] for row in lock['files'][:-1])
    validate_payload(payload)
    return payload


def build_files(payload, release, numeric_job_id):
    validate_payload(payload)
    identity(release['identity'], 'checkpoint')
    require(type(numeric_job_id) is int and numeric_job_id > 0, 'actual producer job identity')
    files = {}; chunks = []
    for index in range(CHUNKS):
        name = f'{DIRECTORY}/chunks/{index:04d}.bin'
        raw = payload[index * CHUNK_CAP:(index + 1) * CHUNK_CAP]
        files[name] = raw
        chunks.append({'path': name, 'bytes': len(raw), 'sha256': digest(raw), 'git_blob': transport.git_sha(raw)})
    manifest = {'schema': 'k1-hosted-transfer-checkpoint-v1',
        'purpose': 'synthetic hosted transport qualification only; not ASR evidence',
        'producer_identity': dict(release['identity']), 'producer_numeric_job_id': numeric_job_id,
        'source_tree': release['source_tree'], 'source_freeze_sha256': release['source_freeze_sha256'],
        'original_fixture_commit': INPUT_COMMIT, 'original_fixture_manifest_sha256': INPUT_MANIFEST_SHA256,
        'archive_bytes': ARCHIVE_BYTES, 'archive_sha256': ARCHIVE_SHA256, 'chunks': chunks,
        'asr_schema_accepted': False, **{key: False for key in FALSE_FLAGS}}
    files[MANIFEST] = canonical(manifest)
    require(len(files[MANIFEST]) <= MANIFEST_CAP, 'bounded qualification manifest')
    return files


def verify_files(files, release, numeric_job_id):
    require(type(files) is dict and set(files) == set(output_paths()), 'complete exact checkpoint files')
    payload = b''.join(files[f'{DIRECTORY}/chunks/{i:04d}.bin'] for i in range(CHUNKS))
    require(files == build_files(payload, release, numeric_job_id), 'same original producer and complete bytes')
    return True


def commit_body(tree, release):
    author = {**publisher.AUTHOR, 'date': release['commit_timestamp']}
    source = release['identity']['executed_head_sha']
    message = 'Retain synthetic hosted K1 transfer checkpoint v1\n'
    body = {'message': message, 'tree': tree, 'parents': [source], 'author': author, 'committer': author}
    stamp = int(datetime.strptime(release['commit_timestamp'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp())
    person = publisher.AUTHOR['name'] + ' <' + publisher.AUTHOR['email'] + '> ' + str(stamp) + ' +0000'
    raw = ('tree ' + tree + '\nparent ' + source + '\nauthor ' + person + '\ncommitter ' + person + '\n\n' + message).encode('utf-8')
    return body, transport.git_sha(raw, 'commit')


def ref_route(branch):
    require(branch in (CHECKPOINT, FINAL), 'fixed checkpoint/result ref only')
    return BASE + '/git/ref/heads/' + branch


def exact_ref(value, branch, sha):
    publisher._equal(value, {'ref': 'refs/heads/' + branch}, 'fixed ref readback')
    publisher._equal(value.get('object'), {'type': 'commit', 'sha': sha}, 'same immutable checkpoint commit')


def raw_reader(commit, paths, deadline):
    reader = publisher.RawReader.__new__(publisher.RawReader)
    reader._initialize(commit, paths=paths, shared_deadline=deadline)
    return reader


def bounded_raw_factory(phase, deadline, pin=None):
    """Exact phase-local two-pass/one-pass raw budget, never producer disk input."""
    require(phase in PHASES, 'fixed raw phase')
    if phase == 'recovery':
        validate_pin(pin)
    counts = {'requests': 0, 'reserved_bytes': 0, 'body_bytes': 0}
    opened = set()
    def factory(commit, paths):
        require(time.monotonic() < deadline and commit not in opened, 'immutable read pass once before deadline')
        if phase == 'checkpoint' and not opened:
            require(commit == INPUT_COMMIT and paths == {r['path']: r['bytes'] for r in fixture_lock()['files']},
                    'first pass only fixed original fixture')
        else:
            require(paths == output_paths() and commit != INPUT_COMMIT, 'only complete new checkpoint pass')
            if phase == 'recovery':
                require(commit == pin['checkpoint_commit'], 'recovery can read only root-pinned checkpoint')
            else:
                require(len(opened) == 1, 'one producer input and one immutable checkpoint pass')
        opened.add(commit)
        read = raw_reader(commit, paths, deadline)
        def get(path):
            require(path in paths and counts['requests'] < (48 if phase == 'checkpoint' else 24)
                    and counts['reserved_bytes'] + paths[path] + 1 <= 16 * 1024 ** 2,
                    'global phase raw request/body cap')
            counts['requests'] += 1
            # RawReader may read one excess byte to detect a cap violation.
            # Reserve that probe too; even an error cannot expand the aggregate.
            counts['reserved_bytes'] += paths[path] + 1
            measured = type(read) is publisher.RawReader
            before = read.__dict__['body_bytes'] if measured else None
            try:
                raw = read(path)
            finally:
                # The fixed reader charges a returned response body before
                # rejecting its status/size. Preserve that observed delta even
                # when read() raises; do not add successful bytes a second time.
                # Socket bytes not returned by response.read remain unobserved.
                if measured:
                    counts['body_bytes'] += read.__dict__['body_bytes'] - before
            if not measured:  # Explicit in-memory test readers have no HTTP counter.
                counts['body_bytes'] += len(raw)
            require(counts['body_bytes'] <= counts['reserved_bytes'], 'raw bodies within reserved byte budget')
            return raw
        return get
    factory.counts = counts
    return factory


def transport_observation(api, raw_get):
    """Measured fixed counters only; no request/response body or credential."""
    keys = ('count', 'request_bytes', 'body_bytes', 'blob_api_attempt_count',
            'blob_recovery_get_count', 'blob_retry_count')
    counters = {key: getattr(api, key) for key in keys}
    require(all(type(value) is int and value >= 0 for value in counters.values()), 'actual nonnegative transport counters')
    raw = raw_get.counts
    require(type(raw) is dict and set(raw) == {'requests', 'reserved_bytes', 'body_bytes'}
        and all(type(value) is int and value >= 0 for value in raw.values()), 'actual raw counters')
    return {'api': counters, 'raw': dict(raw), 'payload_body_logged': False,
            'scope': 'explicit client requests and observed returned read bodies; unreturned socket partial bytes unknown; HTTP framing excluded'}


def failure_record(state, error):
    """Only fixed safe projections; unknown state stays null, never error text."""
    phase = state.get('phase')
    phase = phase if phase in PHASES else None
    who = state.get('identity')
    try:
        identity(who, phase)
        who = dict(who)
    except BaseException:
        who = None
    progress = state.get('progress')
    progress = progress if type(progress) is dict else {}
    stage = progress.get('stage', state.get('stage'))
    stage = stage if stage in FAILURE_STAGES else None
    code = next((known.__name__ for known in (ValueError, TypeError, OSError, TimeoutError,
        publisher.PublicationError, QualificationDisabled) if type(error) is known), 'OtherException')
    status = None
    if type(error) is publisher.PublicationError:
        fields = BaseException.__dict__['__dict__'].__get__(error)
        candidate = dict.get(fields, 'status')
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
    checkpoint = {}
    for key in ('checkpoint_commit', 'checkpoint_tree'):
        value = progress.get(key)
        checkpoint[key] = value if type(value) is str and transport.HEAD.fullmatch(value) else None
    for key in ('checkpoint_created', 'checkpoint_verified', 'checkpoint_preserved'):
        value = progress.get(key)
        checkpoint[key] = value if type(value) is bool else None
    api = state.get('api')
    observed = {}
    fields = api.__dict__ if type(api) is publisher.GitHubAPI else {}
    for key in ('count', 'request_bytes', 'body_bytes', 'blob_api_attempt_count',
                'blob_recovery_get_count', 'blob_retry_count'):
        value = fields.get(key)
        observed[key] = value if type(value) is int and value >= 0 else None
    raw_factory = state.get('raw_get')
    raw_fields = raw_factory.__dict__.get('counts') if type(raw_factory) is type(failure_record) else None
    raw_fields = raw_fields if type(raw_fields) is dict else {}
    raw = {}
    for key in ('requests', 'reserved_bytes', 'body_bytes'):
        value = raw_fields.get(key)
        raw[key] = value if type(value) is int and value >= 0 else None
    return {'schema': 'k1-hosted-transfer-qualification-failure-v1',
        'status': 'failed_or_unconfirmed_no_automatic_retry', 'phase': phase, 'identity': who,
        'stage': stage, 'error_type': code, 'http_status': status, **checkpoint,
        'api': observed, 'raw': raw, 'payload_body_included': False, 'error_message_included': False,
        'body_byte_scope': 'observed returned read bodies; unreturned socket partial bytes unknown; HTTP framing excluded',
        'native_success_established': False}


def api_routes(release, phase, pin=None):
    who = release['identity']
    routes = {BASE, *run_routes(who, phase), BASE + '/git/commits/' + who['executed_head_sha'],
              BASE + '/git/trees/' + release['source_tree'] + '?recursive=1', ref_route(CHECKPOINT), ref_route(FINAL)}
    if phase == 'recovery':
        validate_pin(pin)
        routes.update(run_routes(pin['identity'], 'checkpoint'))
        routes.update((BASE + '/git/commits/' + pin['identity']['executed_head_sha'],
            BASE + '/git/trees/' + pin['source_tree'] + '?recursive=1',
            BASE + '/git/commits/' + pin['checkpoint_commit'],
            BASE + '/git/trees/' + pin['checkpoint_tree'] + '?recursive=1'))
    return routes


def checkpoint_phase(api, raw_get, release, progress=None):
    progress = {} if progress is None else progress
    progress['stage'] = 'verify_actual_run_source'
    publisher._equal(api('GET', BASE), {'full_name': REPOSITORY, 'private': False}, 'public standard-runner repository')
    numeric_job_id = verify_run(api, release['identity'], 'checkpoint')
    original = verify_source(api, release['identity']['executed_head_sha'], release['source_tree'])
    require(not any(row['path'] == DIRECTORY or row['path'].startswith(DIRECTORY + '/')
        or DIRECTORY.startswith(row['path'] + '/') for row in original), 'new qualification directory only')
    progress['stage'] = 'input_fixture_read'
    payload = read_fixture(raw_get(INPUT_COMMIT, {r['path']: r['bytes'] for r in fixture_lock()['files']}))
    files = build_files(payload, release, numeric_job_id)
    rows = [{'path': path, 'mode': '100644', 'type': 'blob', 'sha': transport.git_sha(raw)}
            for path, raw in sorted(files.items())]
    tree, _ = transport.tree_identity(original + rows)
    body, sha = commit_body(tree, release)
    # Scope is immutable before the first write; shared writer binds its two refs.
    api.expected_body = body
    receipt = publisher.write_checked_files(api=api,
        raw_get=lambda commit: raw_get(commit, output_paths()), files=files, original=original,
        source_tree=release['source_tree'], expected_body=body, expected_commit=sha,
        result_branch=FINAL, checkpoint_branch=CHECKPOINT,
        verify_readback=lambda values: verify_files(values, release, numeric_job_id),
        checkpoint_only=True, progress=progress)
    require(receipt.get('checkpoint_verified') is True, 'shared writer confirmed checkpoint readback')
    result = {'schema': 'k1-hosted-transfer-qualification-receipt-v1', 'phase': 'checkpoint',
        'status': 'checkpoint_readback_verified_injected_stop', 'identity': release['identity'],
        'numeric_job_id': numeric_job_id, 'source_tree': release['source_tree'],
        'source_freeze_sha256': release['source_freeze_sha256'], 'commit_timestamp': release['commit_timestamp'],
        'checkpoint_commit': sha, 'checkpoint_tree': tree, 'manifest_sha256': digest(files[MANIFEST]),
        'archive_bytes': ARCHIVE_BYTES, 'archive_sha256': ARCHIVE_SHA256,
        'checkpoint_readback_verified': True, 'final_ref_created': False, 'native_success_established': False,
        **{key: False for key in FALSE_FLAGS}}
    raise InjectedCheckpointStop(result)


def recovery_phase(api, raw_get, release, pin, progress=None):
    progress = {} if progress is None else progress
    progress['stage'] = 'verify_actual_run_source'
    validate_pin(pin)
    publisher._equal(api('GET', BASE), {'full_name': REPOSITORY, 'private': False}, 'public standard-runner repository')
    recovery_job_id = verify_run(api, release['identity'], 'recovery')
    verify_source(api, release['identity']['executed_head_sha'], release['source_tree'])
    progress['stage'] = 'verify_original_producer'
    require(verify_run(api, pin['identity'], 'checkpoint', terminal=True) == pin['numeric_job_id'],
            'same failed producer numeric job')
    original = verify_source(api, pin['identity']['executed_head_sha'], pin['source_tree'])
    progress.update(stage='verify_checkpoint', checkpoint_commit=pin['checkpoint_commit'],
                    checkpoint_tree=pin['checkpoint_tree'])
    exact_ref(api('GET', ref_route(CHECKPOINT)), CHECKPOINT, pin['checkpoint_commit'])
    progress['checkpoint_created'] = True
    original_release = {key: pin[key] for key in ('identity', 'source_tree', 'source_freeze_sha256', 'commit_timestamp')}
    body, sha = commit_body(pin['checkpoint_tree'], original_release)
    require(sha == pin['checkpoint_commit'], 'pinned exact producer commit object')
    publisher.verify_commit(api('GET', BASE + '/git/commits/' + sha), sha, body)
    tree = publisher.verify_tree(api('GET', BASE + '/git/trees/' + pin['checkpoint_tree'] + '?recursive=1'), pin['checkpoint_tree'])
    progress['stage'] = 'checkpoint_body_read'
    read = raw_get(sha, output_paths())
    files = {path: read(path) for path in sorted(output_paths())}
    require(digest(files[MANIFEST]) == pin['manifest_sha256'], 'root pinned manifest bytes')
    verify_files(files, original_release, pin['numeric_job_id'])
    rows = [{'path': path, 'mode': '100644', 'type': 'blob', 'sha': transport.git_sha(raw)}
            for path, raw in sorted(files.items())]
    require(sorted(tree, key=lambda row: row['path']) == sorted(original + rows, key=lambda row: row['path']),
            'entire original source plus exact retained checkpoint only')
    progress.update(stage='checkpoint_verified', checkpoint_verified=True, checkpoint_preserved=True)
    # The recovery consumer can create only one fixed ref, never new blobs,
    # source, tree, commit, payload generator, producer temp files or model calls.
    progress['stage'] = 'verify_destination_absent'
    api('GET', ref_route(FINAL), expected=404)
    api.expected_body = body
    api.bind_result(sha, pin['checkpoint_tree'])
    progress['stage'] = 'create_ref'
    publisher.finalize_exact_ref(api, FINAL, sha, progress=progress)
    return {'schema': 'k1-hosted-transfer-qualification-receipt-v1', 'phase': 'recovery',
        'status': 'same_original_checkpoint_recovered', 'producer_identity': pin['identity'],
        'producer_numeric_job_id': pin['numeric_job_id'], 'recovery_identity': release['identity'],
        'recovery_numeric_job_id': recovery_job_id, 'checkpoint_commit': sha, 'final_commit': sha,
        'archive_bytes': ARCHIVE_BYTES, 'archive_sha256': ARCHIVE_SHA256,
        'producer_terminal_conclusion': 'failure', 'native_success_established': False,
        **{key: False for key in FALSE_FLAGS}}


def _run(argv, state):
    state['stage'] = 'arguments'
    require((sys.argv[1:] if argv is None else argv) == [], 'fixed qualification entry takes no arguments')
    state['stage'] = 'control_read'
    control = read_json(HERE / 'qualification-control.json')
    # No environment, token, event, source/temp directory or HTTP access while disabled.
    if control.get('checkpoint_enabled') is not True and control.get('recovery_enabled') is not True:
        raise QualificationDisabled('qualification source is disabled')
    phase = 'checkpoint' if control.get('checkpoint_enabled') is True else 'recovery'
    state.update(phase=phase, stage='control_gate')
    control_gate(control, phase)
    env = {key: os.environ[key] for key in CONTEXT_KEYS if key in os.environ}
    state['stage'] = 'event_read'
    event = transport.decode(publisher.c.read_bytes(Path(env.get('GITHUB_EVENT_PATH', '')), cap=1024 ** 2))
    state['stage'] = 'source_admission'
    release = source_release(env, event, phase, control)
    state['identity'] = dict(release['identity'])
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'separate bounded qualification process')
    deadline = time.monotonic() + WALL_SECONDS
    state['stage'] = 'token_admission'
    api = publisher.GitHubAPI.__new__(publisher.GitHubAPI)
    api._initialize(os.environ.get('GITHUB_TOKEN'), get_routes=api_routes(release, phase, control['producer_pin']),
        source_sha=release['identity']['executed_head_sha'] if phase == 'checkpoint' else control['producer_pin']['identity']['executed_head_sha'],
        source_tree=release['source_tree'] if phase == 'checkpoint' else control['producer_pin']['source_tree'],
        result_branch=FINAL, checkpoint_branch=CHECKPOINT, paths=output_paths(), expected_body=None,
        max_requests=API_CAPS[phase], timeout=WALL_SECONDS, max_upload=9 * 1024 ** 2,
        max_body=8 * 1024 ** 2, final_ref_only=(phase == 'recovery'))
    state['api'] = api
    api.deadline = min(api.deadline, deadline)
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('qualification deadline')))
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        raw_get = bounded_raw_factory(phase, deadline, control['producer_pin'])
        state.update(raw_get=raw_get, stage='api_preflight')
        if phase == 'checkpoint':
            try:
                return checkpoint_phase(api, raw_get, release, state['progress'])
            except InjectedCheckpointStop as stopped:
                stopped.receipt['transport_observation'] = transport_observation(api, raw_get)
                raise
        result = recovery_phase(api, raw_get, release, control['producer_pin'], state['progress'])
        result['transport_observation'] = transport_observation(api, raw_get)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main(argv=None):
    state = {'phase': None, 'identity': None, 'stage': 'arguments', 'progress': {}, 'api': None, 'raw_get': None}
    try:
        return _run(argv, state)
    except InjectedCheckpointStop:
        raise
    except BaseException as error:
        # Observation is best effort and must never replace the primary error.
        try:
            raw = canonical(failure_record(state, error))
            require(len(raw) <= 4096, 'bounded fixed failure envelope')
            print(raw.decode('ascii'), end='', flush=True)
        except BaseException:
            pass
        raise


if __name__ == '__main__':
    try:
        result = main()
        print(canonical(result).decode('ascii'), end='', flush=True)
    except InjectedCheckpointStop as stopped:
        print(canonical(stopped.receipt).decode('ascii'), end='', flush=True)
        raise SystemExit(86) from None
    except QualificationDisabled:
        raise SystemExit('Hosted transfer qualification disabled.') from None
    except BaseException:
        raise SystemExit('Hosted transfer qualification blocked or unconfirmed; no retry.') from None
