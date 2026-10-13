"""Source-bound, default-CLOSED production metadata issuer and one-use host entry.

The issuer is a separate process: only it receives GITHUB_TOKEN. It uses an
existing source tree, two create-only POSTs, exact own-201/readbacks, and a
private durable local handoff. Public proof cannot authorize entry. No network,
model, filesystem-output or credential access occurs at the disabled CLI gate.
"""
from contextlib import contextmanager
import hashlib
import http.client
import os
from pathlib import Path
import re
import secrets
import signal
import ssl
import stat
import sys
import time
import urllib.parse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import campaign_contract as c
ROOT = HERE.parents[1]
REPOSITORY = 'jiying2007/kws-pipeline'
BASE = '/repos/' + REPOSITORY
JOB_KEY = c.current_scope()['job_key']
MAX_FILE = 16384
MAX_PUBLIC = 65536
MAX_RESPONSE = 512 * 1024
MAX_RESPONSE_TOTAL = 2 * 1024 * 1024
MAX_SECONDS = 45
GET_TICKETS = ('run', 'workflow', 'history', 'jobs', 'source_ref', 'final404',
    'checkpoint404', 'claim404', 'sourcecommit', 'claimreadback', 'ownercommitreadback')
POST_TICKETS = ('create_commit', 'create_ref')
TICKET_ORDER = GET_TICKETS[:9] + POST_TICKETS + GET_TICKETS[9:]
_PRIVATE_SEAL = object()


def need(condition, code):
    c.require(condition, code)


def _positive(value):
    return type(value) is int and 0 < value < 10**20


def _credential_key(key):
    return re.search(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|API_KEY|ACCESS_KEY', key.upper()) is not None


def host_environment():
    # Check names before copying values: an unexpected credential is not read.
    need(not any(_credential_key(key) for key in os.environ), 'claim_host_credentials_present')
    return dict(os.environ)


def issuer_environment():
    # Never copy the token during early ordinal/event/source checks.
    keys = ('GITHUB_ACTIONS', 'GITHUB_REPOSITORY', 'GITHUB_RUN_NUMBER', 'GITHUB_RUN_ATTEMPT',
        'GITHUB_EVENT_NAME', 'GITHUB_JOB', 'GITHUB_REF', 'GITHUB_RUN_ID', 'GITHUB_SHA',
        'GITHUB_WORKFLOW_REF', 'GITHUB_WORKFLOW_SHA', 'RUNNER_ENVIRONMENT', 'RUNNER_OS',
        'RUNNER_ARCH', 'ImageOS', 'RUNNER_TEMP', 'GITHUB_EVENT_PATH', 'GITHUB_OUTPUT')
    return {key: os.environ.get(key, '') for key in keys}


def token_free(environment):
    need(type(environment) is dict and all(type(k) is str and type(v) is str
        for k, v in environment.items()), 'claim_host_environment')
    need(not any(_credential_key(key) for key in environment), 'claim_host_credentials_present')


def early_environment(env, control):
    """No event-file read, token lookup, client or local output precedes this."""
    c.require_current_scope(control)
    fixed = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': REPOSITORY,
        'GITHUB_RUN_NUMBER': '1', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_EVENT_NAME': 'push',
        'GITHUB_JOB': JOB_KEY, 'GITHUB_REF': 'refs/heads/' + control['source_ref'],
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux',
        'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
    need(all(env.get(key) == value for key, value in fixed.items()), 'claim_ordinal_attempt_source_gate')
    need(type(env.get('GITHUB_RUN_ID')) is str and re.fullmatch('[1-9][0-9]{0,19}', env['GITHUB_RUN_ID'])
        and type(env.get('GITHUB_SHA')) is str and c.HEAD.fullmatch(env['GITHUB_SHA']), 'claim_actual_run_source')
    need(env.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/' + control['workflow'] + '@refs/heads/' + control['source_ref']
        and env.get('GITHUB_WORKFLOW_SHA') == env['GITHUB_SHA'], 'claim_executed_workflow')


def identity_for(run, preflight, release, control, freeze_sha, tree, timestamp, release_raw):
    return c.sourcebound_identity(run=run, numeric_job_id=preflight['numeric_job_id'], release=release,
        source_tree=tree, source_timestamp=timestamp, source_freeze_sha256=freeze_sha,
        release_sha256=c.digest(release_raw), control=control, workflow_id=preflight['workflow_id'])


def validate_current(value, run, control):
    need(type(value) is dict and _positive(value.get('id')) and value['id'] == run['run_id']
        and _positive(value.get('workflow_id')) and type(value.get('run_number')) is int and value['run_number'] == 1
        and type(value.get('run_attempt')) is int and value['run_attempt'] == 1
        and value.get('status') == 'in_progress' and value.get('conclusion') is None
        and value.get('event') == 'push' and value.get('head_branch') == control['source_ref']
        and value.get('head_sha') == run['executed_head_sha'] and value.get('path') == control['workflow'],
        'claim_actual_current_run')
    for field in ('repository', 'head_repository'):
        repository = value.get(field)
        need(type(repository) is dict and repository.get('full_name') == REPOSITORY
            and repository.get('private') is False, 'claim_public_repository')
    return value['workflow_id']


def validate_workflow(value, workflow_id, control):
    need(type(value) is dict and type(value.get('id')) is int and value['id'] == workflow_id
        and value.get('path') == control['workflow'] and value.get('state') == 'active', 'claim_actual_workflow_path')


def validate_jobs(page, run):
    need(type(page) is dict and type(page.get('total_count')) is int and page['total_count'] == 1
        and type(page.get('jobs')) is list and len(page['jobs']) == 1, 'claim_unique_numeric_job')
    job = page['jobs'][0]
    need(type(job) is dict and job.get('name') == JOB_KEY and _positive(job.get('id'))
        and type(job.get('run_id')) is int and job['run_id'] == run['run_id']
        and job.get('status') == 'in_progress' and job.get('conclusion') is None
        and job.get('head_sha') == run['executed_head_sha'] and job.get('labels') == ['ubuntu-24.04']
        and type(job.get('runner_name')) is str and job['runner_name'].startswith('GitHub Actions ')
        and job.get('runner_group_name') == 'GitHub Actions', 'claim_actual_hosted_numeric_job')
    return job['id']


def validate_source_ref(value, run, control):
    need(type(value) is dict and value.get('ref') == 'refs/heads/' + control['source_ref']
        and type(value.get('object')) is dict and value['object'].get('type') == 'commit'
        and value['object'].get('sha') == run['executed_head_sha'], 'claim_actual_source_ref')


def history_route(control, workflow_id):
    need(_positive(workflow_id), 'claim_workflow_id')
    return BASE + '/actions/workflows/' + str(workflow_id) + '/runs?branch=' + urllib.parse.quote(
        control['source_ref'], safe='') + '&per_page=100&page=1'


class Budget:
    """Each phase ticket is charged before dispatch; any failure closes all use."""
    def __init__(self):
        self.started = time.monotonic()
        self.deadline = self.started + MAX_SECONDS
        self.used = []
        self.gets = self.posts = self.upload = self.responses = 0
        self.failed = False

    def remaining(self):
        value = self.deadline - time.monotonic()
        need(not self.failed and value > 0, 'claim_deadline_or_closed')
        return value

    def reserve(self, ticket, raw):
        self.remaining()
        need(len(self.used) < len(TICKET_ORDER) and ticket == TICKET_ORDER[len(self.used)], 'claim_exact_ticket_once')
        need(raw is None or type(raw) is bytes, 'claim_request_bytes')
        if ticket in POST_TICKETS:
            need(raw is not None and 0 < len(raw) <= (16384 if ticket == 'create_commit' else 512), 'claim_request_cap')
        else:
            need(raw is None, 'claim_get_no_body')
        need(self.responses < MAX_RESPONSE_TOTAL, 'claim_response_total')
        self.used.append(ticket)
        self.gets += int(ticket in GET_TICKETS)
        self.posts += int(ticket in POST_TICKETS)
        self.upload += len(raw or b'')
        need(self.gets <= 11 and self.posts <= 2 and self.upload <= 16384 + 512, 'claim_fixed_budget')

    def observe(self, raw):
        need(type(raw) is bytes, 'claim_response_bytes')
        self.responses += len(raw)
        self.remaining()
        need(len(raw) <= MAX_RESPONSE and self.responses <= MAX_RESPONSE_TOTAL, 'claim_response_cap')

    def record(self):
        return {'gets': self.gets, 'posts': self.posts, 'request_body_bytes': self.upload,
            'response_body_bytes': self.responses, 'failed': self.failed,
            'maximum_seconds': MAX_SECONDS, 'retry_count': 0}


@contextmanager
def _request_alarm(seconds):
    def expired(_signum, _frame):
        raise TimeoutError('claim_fixed_request_timeout')
    previous = signal.signal(signal.SIGALRM, expired)
    entered = time.monotonic()
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0]:
            # Do not extend an existing outer deadline.
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, previous_timer[0] - (time.monotonic() - entered)), previous_timer[1])


class DirectHTTPS:
    """Only fixed TLS origin; no proxy, redirects, retry, logs, or disk token."""
    def request(self, method, route, raw, token, allowance, remaining, observe):
        connection = None
        try:
            with _request_alarm(min(15, remaining)):
                connection = http.client.HTTPSConnection('api.github.com', 443,
                    timeout=min(15, remaining), context=ssl.create_default_context())
                connection.request(method, route, body=raw, headers={
                    'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'k1-asr18-fixed-claim-issuer',
                    'Content-Type': 'application/json', 'Accept-Encoding': 'identity'})
                response = connection.getresponse()
                need(response.getheader('Content-Encoding', 'identity') == 'identity', 'claim_response_encoding')
                chunks, count = [], 0
                while True:
                    room = min(65536, allowance + 1 - count)
                    need(room > 0, 'claim_response_cap')
                    try:
                        chunk = response.read1(room)
                    except http.client.IncompleteRead as error:
                        if error.partial:
                            observe(error.partial)
                        raise
                    observe(chunk)  # Partial/failing responses are counted before parsing.
                    if not chunk:
                        break
                    count += len(chunk)
                    chunks.append(chunk)
                    need(count <= allowance, 'claim_response_cap')
                return response.status, b''.join(chunks)
        finally:
            if connection is not None:
                connection.close()


class GitHubClaimAPI:
    """Fixed source-bound metadata routes and two exact create-only payloads."""
    def __init__(self, token, run, control, *, transport=None):
        c.validate_claim_control(control, require_enabled=True)
        need(type(token) is str and token.startswith('ghs_') and 5 <= len(token) <= 4096
            and all(32 <= ord(char) < 127 for char in token), 'claim_short_lived_issuer_token')
        need(type(run) is dict and _positive(run.get('run_id')), 'claim_run_identity')
        self._token = token
        self.transport = DirectHTTPS() if transport is None else transport
        self.budget = Budget()
        self.run = c.decode_json(c.canonical(run))
        self.control = c.decode_json(c.canonical(control))
        self.workflow_id = self.owner_sha = self.owner_body = None
        self.receipts = {}
        self.response_hashes = {}

    def bind_workflow(self, workflow_id):
        need(self.workflow_id is None and _positive(workflow_id), 'claim_workflow_bound_once')
        self.workflow_id = workflow_id

    def bind_owner(self, owner):
        need(self.owner_sha is None and self.budget.used == list(GET_TICKETS[:9]), 'claim_owner_bound_once')
        self.owner_body, self.owner_sha = c.owner_commit(owner)

    def route(self, ticket):
        run = str(self.run['run_id'])
        paths = {'run': BASE + '/actions/runs/' + run,
            'jobs': BASE + '/actions/runs/' + run + '/attempts/1/jobs?per_page=100&page=1',
            'source_ref': BASE + '/git/ref/heads/' + self.control['source_ref'],
            'final404': BASE + '/git/ref/heads/' + self.control['evidence_ref'],
            'checkpoint404': BASE + '/git/ref/heads/' + self.control['evidence_ref'] + '-checkpoint',
            'claim404': BASE + '/git/ref/' + self.control['claim_ref'].removeprefix('refs/'),
            'sourcecommit': BASE + '/git/commits/' + self.run['executed_head_sha'],
            'create_commit': BASE + '/git/commits', 'create_ref': BASE + '/git/refs',
            'claimreadback': BASE + '/git/ref/' + self.control['claim_ref'].removeprefix('refs/')}
        if self.workflow_id is not None:
            paths.update(workflow=BASE + '/actions/workflows/' + str(self.workflow_id),
                history=history_route(self.control, self.workflow_id))
        if self.owner_sha is not None:
            paths['ownercommitreadback'] = BASE + '/git/commits/' + self.owner_sha
        need(ticket in paths, 'claim_ticket_route_not_bound')
        return paths[ticket]

    def request(self, ticket):
        try:
            route = self.route(ticket)
            method = 'POST' if ticket in POST_TICKETS else 'GET'
            body = None
            if ticket == 'create_commit':
                need(self.owner_body is not None, 'claim_missing_owner_body')
                body = self.owner_body
            elif ticket == 'create_ref':
                need(self.owner_sha is not None, 'claim_missing_owner_identity')
                body = {'ref': self.control['claim_ref'], 'sha': self.owner_sha}
            raw = None if body is None else c.canonical(body)
            self.budget.reserve(ticket, raw)
            observed = self.budget.responses
            status, response_raw = self.transport.request(method, route, raw, self._token,
                min(MAX_RESPONSE, MAX_RESPONSE_TOTAL - self.budget.responses), self.budget.remaining(), self.budget.observe)
            need(type(response_raw) is bytes and len(response_raw) <= MAX_RESPONSE
                and self.budget.responses - observed == len(response_raw), 'claim_exact_wire_accounting')
            self.response_hashes[ticket] = c.digest(response_raw)
            need(self._token.encode() not in response_raw, 'claim_response_secret_rejected')
            expected = 201 if method == 'POST' else 404 if ticket.endswith('404') else 200
            need(type(status) is int and status == expected, 'claim_http_status_no_retry')
            if expected == 404:
                return None
            value = c.decode_json(response_raw, require_canonical=False)
            need(type(value) is dict, 'claim_api_object')
            if ticket in ('create_ref', 'claimreadback', 'ownercommitreadback'):
                key = {'create_ref': 'ref_create', 'claimreadback': 'ref_readback',
                    'ownercommitreadback': 'owner_readback'}[ticket]
                self.receipts[key] = c.response_receipt(status, response_raw, route=route)
            return value
        except BaseException:
            self.budget.failed = True
            raise


def _directory(path, *, private=False):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'claim_absolute_private_root')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        need(info.st_uid == os.getuid() and not stat.S_IMODE(info.st_mode) & 0o022
            and (not private or stat.S_IMODE(info.st_mode) == 0o700), 'claim_private_directory_owner_mode')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _same_root(path, fd):
    other = _directory(path)
    try:
        a, b = os.fstat(fd), os.fstat(other)
        need((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), 'claim_root_substitution')
    finally:
        os.close(other)


def _same_directory(parent, name, fd):
    current, opened = os.stat(name, dir_fd=parent, follow_symlinks=False), os.fstat(fd)
    need(stat.S_ISDIR(current.st_mode) and current.st_uid == os.getuid()
        and stat.S_IMODE(current.st_mode) == 0o700
        and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino), 'claim_directory_substitution')


def _filesystem(parent, directory):
    root, child = os.fstat(parent), os.fstat(directory)
    return {'root_device': root.st_dev, 'root_inode': root.st_ino,
        'directory_device': child.st_dev, 'directory_inode': child.st_ino}


def directory_name(run):
    need(type(run) is dict and _positive(run.get('run_id')) and run.get('job_key') == JOB_KEY
        and type(run.get('run_attempt')) is int and run['run_attempt'] == 1, 'claim_fixed_private_directory')
    return 'k1-asr18-claim-' + str(run['run_id']) + '-' + JOB_KEY


def _write(directory, name, value, *, cap=MAX_FILE):
    raw = c.canonical(value)
    need(len(raw) <= cap, 'claim_private_file_cap')
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600, dir_fd=directory)
    try:
        need(os.write(fd, raw) == len(raw), 'claim_private_short_write')
        os.fsync(fd)
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600, 'claim_private_file_alias')
    finally:
        os.close(fd)
    os.fsync(directory)


def _read(directory, name, *, cap=MAX_FILE):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o600 and 0 < before.st_size <= cap, 'claim_private_file_alias')
        raw = os.read(fd, cap + 1)
        after = os.fstat(fd)
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_nlink,
            value.st_mtime_ns, value.st_ctime_ns, value.st_mode, value.st_uid)
        need(len(raw) == before.st_size and stamp(before) == stamp(after) == stamp(current), 'claim_private_file_changed')
        return c.decode_json(raw)
    finally:
        os.close(fd)


def issue(env, event, run, release, control_raw, freeze_sha, tree, timestamp, release_raw, *, transport=None):
    """Production path; test transport injection never creates another CLI mode."""
    control = c.require_claim_admission(release, control_raw)
    early_environment(env, control)
    need(c.exact_json(c.decode_json(release_raw, require_canonical=False), release), 'claim_exact_release_body')
    import hosted_run as host
    need(c.exact_json(host.event_identity(env, event, control=control), run), 'claim_event_identity')
    need(c.digest(c.canonical(host.bound_prior_history_identity(control))) == release['prior_public_claim_history_sha256'],
        'independently released prior public claim history')
    parent = directory = None
    try:
        parent = _directory(env['RUNNER_TEMP'])
        name = directory_name(run)
        os.mkdir(name, mode=0o700, dir_fd=parent)
        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        _same_directory(parent, name, directory)
        os.fsync(parent)
        filesystem = _filesystem(parent, directory)
        _write(directory, 'attempt.json', {'run': run, 'filesystem': filesystem})
        nonce = secrets.token_hex(32)
        # Credential access is isolated here, after every source-local gate and
        # the permanent attempt directory. No token is written or returned.
        token = env.get('GITHUB_TOKEN') if 'GITHUB_TOKEN' in env else os.environ.get('GITHUB_TOKEN')
        api = GitHubClaimAPI(token, run, control, transport=transport)
        current = api.request('run')
        workflow_id = validate_current(current, run, control)
        api.bind_workflow(workflow_id)
        validate_workflow(api.request('workflow'), workflow_id, control)
        page = api.request('history')
        job_id = validate_jobs(api.request('jobs'), run)
        preflight = {'numeric_job_id': job_id, 'workflow_id': workflow_id}
        identity = identity_for(run, preflight, release, control, freeze_sha, tree, timestamp, release_raw)
        history = c.api_bound_history(identity, page, 'acquisition', raw_sha256=api.response_hashes['history'])
        validate_source_ref(api.request('source_ref'), run, control)
        api.request('final404')
        api.request('checkpoint404')
        api.request('claim404')
        source = api.request('sourcecommit')
        need(type(source.get('tree')) is dict and source['tree'].get('sha') == tree
            and source.get('sha') == run['executed_head_sha'] and type(source.get('committer')) is dict
            and source['committer'].get('date') in (timestamp, timestamp[:-1] + '+00:00'), 'claim_actual_source_commit_tree_time')
        owner = c.owner_record(identity, history, c.digest(nonce.encode('ascii')))
        body, sha = c.owner_commit(owner)
        api.bind_owner(owner)
        c.verify_owner_commit(api.request('create_commit'), owner)
        c.validate_claim_ref(api.request('create_ref'), owner)
        c.validate_claim_ref(api.request('claimreadback'), owner)
        c.verify_owner_commit(api.request('ownercommitreadback'), owner)
        hashes = c.validate_acquisition_responses(api.receipts, owner)
        public = {'owner': owner, 'commit': {'sha': sha, 'body': body},
            'acquisition_history': history, 'acquisition_responses': api.receipts}
        raw = c.canonical(public)
        need(len(raw) <= MAX_PUBLIC and nonce.encode() not in raw
            and token.encode() not in raw, 'claim_public_proof_bound')
        _same_root(env['RUNNER_TEMP'], parent)
        _same_directory(parent, name, directory)
        _write(directory, 'acquisition.json', public, cap=MAX_PUBLIC)
        _write(directory, 'handoff.json', {'schema': 'k1-asr18-private-handoff-v2',
            'nonce': nonce, 'filesystem': filesystem, 'owner_commit': sha, 'response_hashes': hashes})
        _same_root(env['RUNNER_TEMP'], parent)
        _same_directory(parent, name, directory)
        _write(directory, 'ready.json', {'ready': True})
        return True
    finally:
        if directory is not None:
            os.close(directory)
        if parent is not None:
            os.close(parent)


class HostPermit:
    """A private in-process continuation, created only after durable consumption."""
    __slots__ = ('_seal', '_entered', '_run', '_proof')
    def __init__(self, seal, run, proof):
        need(seal is _PRIVATE_SEAL, 'claim_private_continuation')
        self._seal, self._entered = seal, False
        self._run = c.canonical(run)
        self._proof = c.canonical(proof)

    def enter(self, run):
        need(self._seal is _PRIVATE_SEAL and not self._entered and self._run == c.canonical(run), 'claim_execute_once')
        self._entered = True
        return c.decode_json(self._proof)


def consume_from_preflight(env, run, release, control_raw, freeze_sha, tree, timestamp, release_raw, preflight):
    """Permanent marker BEFORE the original six GETs. No additional owner GET.

    On every failure the marker remains, including all six callback failures,
    partial writes, fsync failures, process death and invalid saved proof.
    """
    control = c.require_claim_admission(release, control_raw)
    early_environment(env, control)
    token_free(host_environment())
    token_free(env)
    need(callable(preflight), 'claim_preflight_callable')
    parent = directory = None
    try:
        parent = _directory(env['RUNNER_TEMP'])
        name = directory_name(run)
        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        _same_directory(parent, name, directory)
        _same_root(env['RUNNER_TEMP'], parent)
        _write(directory, 'consumed.json', {'schema': 'k1-asr18-consumed-before-preflight-v2', 'run': run})
        need(c.exact_json(_read(directory, 'ready.json'), {'ready': True}), 'claim_readiness_missing')
        actual = preflight()
        identity = identity_for(run, actual, release, control, freeze_sha, tree, timestamp, release_raw)
        filesystem = _filesystem(parent, directory)
        need(c.exact_json(_read(directory, 'attempt.json'), {'run': run, 'filesystem': filesystem}), 'claim_attempt_binding')
        permit = _read(directory, 'handoff.json')
        need(type(permit) is dict and set(permit) == {'schema', 'nonce', 'filesystem', 'owner_commit', 'response_hashes'}
            and permit['schema'] == 'k1-asr18-private-handoff-v2' and type(permit['nonce']) is str
            and re.fullmatch('[0-9a-f]{64}', permit['nonce']) and c.exact_json(permit['filesystem'], filesystem), 'claim_private_handoff')
        public = _read(directory, 'acquisition.json', cap=MAX_PUBLIC)
        need(type(public) is dict and set(public) == {'owner', 'commit', 'acquisition_history', 'acquisition_responses'}, 'claim_acquisition_shape')
        owner = c.validate_owner(public['owner'], identity, public['acquisition_history'])
        body, sha = c.owner_commit(owner)
        need(c.exact_json(public['commit'], {'sha': sha, 'body': body}) and permit['owner_commit'] == sha
            and c.digest(permit['nonce'].encode('ascii')) == owner['nonce_sha256'], 'claim_nonce_owner_binding')
        need(c.exact_json(permit['response_hashes'], c.validate_acquisition_responses(public['acquisition_responses'], owner)), 'claim_own_response_binding')
        consumed = {'schema': 'k1-asr18-consumed-observation-v2', 'owner_commit': sha,
            'identity': identity, 'permanent_marker_durable': True, 'model_consumption': None}
        admission = {**public, 'host_history': actual['history'], 'consumed': consumed}
        c.validate_bound_admission(admission, identity)
        need(len(c.canonical(admission)) <= MAX_PUBLIC and permit['nonce'].encode() not in c.canonical(admission), 'claim_public_proof_bound')
        _same_root(env['RUNNER_TEMP'], parent)
        _same_directory(parent, name, directory)
        return actual, HostPermit(_PRIVATE_SEAL, run, admission)
    finally:
        if directory is not None:
            os.close(directory)
        if parent is not None:
            os.close(parent)



def _emit_ready(env):
    target = c.safe_path(env['GITHUB_OUTPUT'])
    parent = c._open_directory(target.parent)
    fd = None
    try:
        fd = os.open(target.name, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1,
            'claim_fixed_step_output')
        need(os.write(fd, b'ready=true\n') == 11, 'claim_step_output_short_write')
        os.fsync(fd)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)

def main(argv=None):
    import hosted_run as host
    # Reads only fixed checked-in controls. Current source is CLOSED; no token,
    # caller event path, private directory, transport or model is reached.
    release = host.release_first()
    need((sys.argv[1:] if argv is None else argv) == [], 'claim_fixed_entry_no_arguments')
    control_raw = c.read_bytes(HERE / 'claim-admission-control.json', cap=MAX_FILE)
    control = c.require_claim_admission(release, control_raw)
    with _request_alarm(MAX_SECONDS):
        env = issuer_environment()
        early_environment(env, control)
        event = c.read_json(Path(env['GITHUB_EVENT_PATH']), c.MIB)
        run = host.event_identity(env, event, control=control)
        freeze, freeze_sha, tree = host.source_authority(run, release, control=control)
        timestamp = host.source_timestamp(event, run['executed_head_sha'])
        release_raw = c.read_bytes(HERE / 'execution-release.json', cap=65536)
        issue(env, event, run, release, control_raw, freeze_sha, tree, timestamp, release_raw)
        _emit_ready(env)
        print('{"ready":true}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BaseException as error:
        if type(error) is SystemExit and error.code == 0:
            raise
        # Deliberately no exception formatting, response, identity, or path.
        print('{"ready":false}', flush=True)
        raise SystemExit(2)
