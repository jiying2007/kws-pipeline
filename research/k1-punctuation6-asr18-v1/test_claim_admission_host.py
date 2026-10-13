"""Local-only production issuer/host regression. No HTTP, models, remote Git.

The transport returns real Git API shapes serialized into original whitespace-
containing bytes. Separate real OS processes share only the private temp files.
"""
import copy
import http.client
import io
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import campaign_contract as c
import claim_admission as a
import hosted_run as h
from test_bound_claim_contract import fixture as contract_fixture



def fixture(**kwargs):
    if kwargs.get('control') is None and kwargs.get('control_raw') is None:
        scope = c.current_scope(); control = c.disabled_claim_control()
        control.update({key: True for key in c.CLAIM_APPROVALS})
        control.update(campaign=scope['campaign'], source_ref=scope['source_branch'],
            evidence_ref=scope['result_branch'], claim_ref=scope['claim_ref'], workflow=scope['workflow'])
        kwargs = {**kwargs, 'control': control}
    initial = contract_fixture(**kwargs)
    release = copy.deepcopy(initial['release'])
    release['prior_public_claim_history_sha256'] = c.digest(c.canonical(h.bound_prior_history_identity(initial['control'])))
    return contract_fixture(count=kwargs.get('count', 2), release=release, release_raw=raw(release),
        control=initial['control'], control_raw=initial['control_raw'], run=initial['run'],
        source_tree=initial['identity']['source_tree'], source_timestamp=initial['identity']['source_timestamp'],
        source_freeze_sha256=initial['identity']['freeze_sha256'], numeric_job_id=initial['identity']['job_id'],
        workflow_id=initial['identity']['workflow_id'])

def environment(f, directory, *, token=False):
    identity, control = f['identity'], f['control']
    env = {'PATH': '/usr/bin:/bin', 'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': a.REPOSITORY,
        'GITHUB_RUN_NUMBER': '1', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_EVENT_NAME': 'push',
        'GITHUB_JOB': 'asr18', 'GITHUB_REF': 'refs/heads/' + control['source_ref'],
        'GITHUB_RUN_ID': str(identity['run_id']), 'GITHUB_SHA': identity['source_sha'],
        'GITHUB_WORKFLOW_REF': a.REPOSITORY + '/' + control['workflow'] + '@refs/heads/' + control['source_ref'],
        'GITHUB_WORKFLOW_SHA': identity['source_sha'], 'RUNNER_TEMP': str(directory),
        'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64', 'ImageOS': 'ubuntu24'}
    if token:
        env['GITHUB_TOKEN'] = 'ghs_SYNTHETIC_ISSUER_TOKEN_NEVER_PUBLIC'
    return env


def event(f):
    return {'created': True, 'deleted': False, 'forced': False, 'before': '0' * 40,
        'after': f['identity']['source_sha'], 'ref': 'refs/heads/' + f['control']['source_ref'],
        'repository': {'full_name': a.REPOSITORY, 'private': False}}


def current(f):
    row = copy.deepcopy(f['page']['workflow_runs'][0])
    row.update(status='in_progress', conclusion=None)
    return row


def jobs(f, *, wrong=False):
    return {'total_count': 1, 'jobs': [{'id': f['identity']['job_id'] + int(wrong),
        'name': 'asr18', 'run_id': f['identity']['run_id'], 'head_sha': f['identity']['source_sha'],
        'status': 'in_progress', 'conclusion': None, 'labels': ['ubuntu-24.04'],
        'runner_name': 'GitHub Actions fixture', 'runner_group_name': 'GitHub Actions'}]}


def source_ref(f):
    return {'ref': 'refs/heads/' + f['control']['source_ref'],
        'object': {'type': 'commit', 'sha': f['identity']['source_sha']}}


def raw(value):
    return json.dumps(value, indent=2, ensure_ascii=True).encode() + b'\n'


class Wire:
    def __init__(self, f, *, fail=None, mutation=None, existing=False):
        self.f, self.fail, self.mutation, self.existing = f, fail, mutation, existing
        self.calls, self.commit, self.ref = [], None, None

    def request(self, method, route, body, token, allowance, remaining, observe):
        ticket = a.TICKET_ORDER[len(self.calls)]
        self.calls.append((ticket, method, route, len(body or b'')))
        f = self.f
        status = 201 if method == 'POST' else 404 if ticket.endswith('404') else 200
        if ticket == 'run':
            value = current(f)
        elif ticket == 'workflow':
            value = {'id': f['identity']['workflow_id'], 'path': f['control']['workflow'], 'state': 'active'}
        elif ticket == 'history':
            value = f['page']
        elif ticket == 'jobs':
            value = jobs(f)
        elif ticket == 'source_ref':
            value = source_ref(f)
        elif ticket.endswith('404'):
            value = {'message': 'Not Found'}
            if ticket == 'claim404' and self.existing:
                status, value = 200, self.ref or {'ref': f['control']['claim_ref'], 'object': {'type': 'commit', 'sha': 'd' * 40}}
        elif ticket == 'sourcecommit':
            value = {'sha': f['identity']['source_sha'], 'tree': {'sha': f['identity']['source_tree']},
                'committer': {'date': f['identity']['source_timestamp']}}
        elif ticket == 'create_commit':
            posted = json.loads(body)
            owner = json.loads(posted['message'])
            expected, sha = c.owner_commit(owner)
            if posted != expected:
                raise AssertionError('not the exact owner commit')
            self.commit = {'sha': sha, 'message': posted['message'], 'tree': {'sha': posted['tree']},
                'parents': [{'sha': posted['parents'][0]}], 'author': posted['author'], 'committer': posted['committer'],
                'verification': {'verified': False, 'reason': 'unsigned', 'signature': None, 'payload': None}}
            value = self.commit
        elif ticket == 'create_ref':
            posted = json.loads(body)
            self.ref = {'ref': posted['ref'], 'node_id': 'fixture-node',
                'object': {'type': 'commit', 'sha': posted['sha']}}
            value = self.ref
            self.existing = True
        elif ticket == 'claimreadback':
            value = self.ref
        elif ticket == 'ownercommitreadback':
            value = self.commit
        else:
            raise AssertionError(ticket)
        if self.mutation is not None:
            status, value = self.mutation(ticket, status, copy.deepcopy(value))
        encoded = raw(value)
        observe(encoded)
        if self.fail == ticket:
            raise ConnectionResetError('synthetic acknowledged body discarded')
        return status, encoded


class Public:
    def __init__(self, f, *, fail=None, wrong_job=False):
        self.f, self.fail, self.wrong_job = f, fail, wrong_job
        self.calls, self.response_hashes = [], {}
        self.bytes = 0

    def get(self, route, *, absent=False):
        self.calls.append(route)
        if self.fail == len(self.calls):
            raise OSError('synthetic fixed GET failure')
        values = [current(self.f), {'id': self.f['identity']['workflow_id'],
            'path': self.f['control']['workflow'], 'state': 'active'}, self.f['page'],
            jobs(self.f, wrong=self.wrong_job), source_ref(self.f), {'message': 'Not Found'}]
        value = values[len(self.calls) - 1]
        body = raw(value)
        self.bytes += len(body)
        self.response_hashes[route] = c.digest(body)
        return None if absent else value

    def accounting(self):
        return {'gets': len(self.calls), 'response_body_bytes': self.bytes, 'response_body_cap_bytes': c.MIB,
            'scope': 'public_api_metadata_separate_from_model_audio_bodies', 'hard_wire_byte_cap': False}


def issue(f, directory, wire=None, env=None):
    wire = Wire(f) if wire is None else wire
    env = environment(f, directory, token=True) if env is None else env
    result = a.issue(env, event(f), f['run'], f['release'], f['control_raw'], f['identity']['freeze_sha256'],
        f['identity']['source_tree'], f['identity']['source_timestamp'], f['release_raw'], transport=wire)
    return result, wire


def consume(f, directory, public=None, env=None):
    public = Public(f) if public is None else public
    env = environment(f, directory) if env is None else env
    def six():
        return h.api_preflight(public, f['run'], f['release'], control=f['control'],
            freeze_sha=f['identity']['freeze_sha256'], tree=f['identity']['source_tree'],
            timestamp=f['identity']['source_timestamp'], release_raw=f['release_raw'])
    preflight, permit = a.consume_from_preflight(env, f['run'], f['release'], f['control_raw'],
        f['identity']['freeze_sha256'], f['identity']['source_tree'], f['identity']['source_timestamp'], f['release_raw'], six)
    return preflight, permit, public


def process_issue(queue, f, directory, failure=None):
    env = environment(f, directory, token=True)
    os.environ.clear(); os.environ.update(env)
    wire = Wire(f, fail=failure)
    try:
        issue(f, directory, wire, env)
        queue.put({'ready': True, 'calls': len(wire.calls), 'posts': sum(row[1] == 'POST' for row in wire.calls)})
    except BaseException as error:
        queue.put({'ready': False, 'error': type(error).__name__, 'calls': len(wire.calls)})


def process_host(queue, f, directory, failure=None, wrong_job=False, barrier=None, crash=None):
    env = environment(f, directory)
    os.environ.clear(); os.environ.update(env)
    public = Public(f, fail=failure, wrong_job=wrong_job)
    if barrier is not None:
        barrier.wait(5)
    original_write = a._write
    def faulty_write(fd, name, value, **kwargs):
        result = original_write(fd, name, value, **kwargs)
        if name == 'consumed.json' and crash:
            os._exit(86)
        return result
    try:
        with mock.patch.object(a, '_write', side_effect=faulty_write):
            preflight, permit, _ = consume(f, directory, public, env)
        proof = permit.enter(f['run'])
        queue.put({'entered': True, 'calls': len(public.calls), 'history_count': preflight['history']['total_count'],
            'model_consumption': proof['consumed']['model_consumption']})
    except BaseException as error:
        queue.put({'entered': False, 'error': type(error).__name__, 'calls': len(public.calls)})


class ClaimIssuerHostTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture()
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.clean_env = mock.patch.dict(os.environ, environment(self.f, self.directory), clear=True)
        self.clean_env.start()

    def tearDown(self):
        self.clean_env.stop()
        self.tmp.cleanup()

    def test_foreign_coherent_campaign_rejects_before_live_issuer_host_or_api(self):
        foreign = contract_fixture()
        c.validate_source_claim(c.canonical(foreign['claim']), expected_sha256=c.digest(c.canonical(foreign['claim'])),
            release=foreign['release'], source_freeze_sha256=foreign['identity']['freeze_sha256'],
            run_id=str(foreign['run']['run_id']), executed_head_sha=foreign['run']['executed_head_sha'])
        wire = Wire(foreign)
        with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
            issue(foreign, self.directory, wire)
        self.assertEqual(wire.calls, [])
        public = Public(foreign)
        with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
            consume(foreign, self.directory, public)
        self.assertEqual(public.calls, [])
        with self.assertRaisesRegex(ValueError, 'sole compiled current scope'):
            h.bound_api_preflight(public, foreign['run'], foreign['release'], foreign['control'],
                foreign['identity']['freeze_sha256'], foreign['source_tree'],
                foreign['identity']['source_timestamp'], foreign['release_raw'])
        self.assertEqual(public.calls, [])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_real_shapes_raw_bytes_and_truthful_history2(self):
        ready, wire = issue(self.f, self.directory)
        self.assertTrue(ready)
        self.assertEqual([row[0] for row in wire.calls], list(a.TICKET_ORDER))
        self.assertEqual(sum(row[1] == 'GET' for row in wire.calls), 11)
        self.assertEqual(sum(row[1] == 'POST' for row in wire.calls), 2)
        preflight, permit, public = consume(self.f, self.directory)
        self.assertEqual(len(public.calls), 6)
        proof = permit.enter(self.f['run'])
        self.assertEqual(proof['acquisition_history']['total_count'], 2)
        self.assertEqual(proof['host_history']['total_count'], 2)
        self.assertIsNone(proof['consumed']['model_consumption'])
        receipt = proof['acquisition_responses']['ref_create']
        self.assertIn('\n  ', receipt['raw_body_utf8'])
        self.assertEqual(c.digest(receipt['raw_body_utf8'].encode()), receipt['body_sha256'])
        self.assertNotEqual(receipt['body_sha256'], c.digest(c.canonical(json.loads(receipt['raw_body_utf8']))))
        raw_proof = c.canonical(proof)
        private = json.loads((self.directory / a.directory_name(self.f['run']) / 'handoff.json').read_bytes())
        self.assertNotIn(private['nonce'].encode(), raw_proof)
        self.assertNotIn(str(self.directory).encode(), raw_proof)
        self.assertNotIn(b'ghs_SYNTHETIC', raw_proof)
        with self.assertRaises(ValueError): permit.enter(self.f['run'])
        again = Public(self.f)
        with self.assertRaises(FileExistsError): consume(self.f, self.directory, again)
        self.assertEqual(again.calls, [])

    def test_real_os_issuer_then_host_then_restart(self):
        ctx = multiprocessing.get_context('fork')
        queue = ctx.Queue()
        issuer = ctx.Process(target=process_issue, args=(queue, self.f, str(self.directory)))
        issuer.start(); issuer.join(10)
        self.assertEqual(issuer.exitcode, 0)
        self.assertEqual(queue.get(timeout=2), {'ready': True, 'calls': 13, 'posts': 2})
        for expected in (True, False):
            host = ctx.Process(target=process_host, args=(queue, self.f, str(self.directory)))
            host.start(); host.join(10)
            self.assertEqual(host.exitcode, 0)
            result = queue.get(timeout=2)
            self.assertEqual(result['entered'], expected)
            self.assertEqual(result['calls'], 6 if expected else 0)

    def test_real_os_two_competing_hosts_only_one_six_get_entry(self):
        issue(self.f, self.directory)
        ctx = multiprocessing.get_context('fork'); queue = ctx.Queue(); barrier = ctx.Barrier(2)
        processes = [ctx.Process(target=process_host, args=(queue, self.f, str(self.directory), None, False, barrier)) for _ in range(2)]
        for process in processes: process.start()
        for process in processes: process.join(10); self.assertEqual(process.exitcode, 0)
        results = [queue.get(timeout=2) for _ in processes]
        self.assertEqual(sorted(result['calls'] for result in results), [0, 6])
        self.assertEqual(sum(result['entered'] for result in results), 1)

    def test_each_of_original_six_get_failures_burns_restart_before_zero_get(self):
        ctx = multiprocessing.get_context('fork')
        for number in range(1, 7):
            with self.subTest(number=number), tempfile.TemporaryDirectory() as folder:
                issue(self.f, folder)
                queue = ctx.Queue()
                for failure in (number, None):
                    process = ctx.Process(target=process_host, args=(queue, self.f, folder, failure))
                    process.start(); process.join(10)
                    self.assertEqual(process.exitcode, 0)
                    result = queue.get(timeout=2)
                    self.assertFalse(result['entered'])
                    self.assertEqual(result['calls'], number if failure else 0)

    def test_real_os_kill_after_durable_marker_keeps_restart_closed(self):
        issue(self.f, self.directory)
        ctx = multiprocessing.get_context('fork'); queue = ctx.Queue()
        process = ctx.Process(target=process_host, args=(queue, self.f, str(self.directory), None, False, None, True))
        process.start(); process.join(10)
        self.assertEqual(process.exitcode, 86)
        again = Public(self.f)
        with self.assertRaises(FileExistsError): consume(self.f, self.directory, again)
        self.assertEqual(again.calls, [])

    def test_marker_write_and_fsync_faults_never_allow_second_get(self):
        for operation in ('write', 'fsync', 'close'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as folder:
                issue(self.f, folder)
                original = getattr(a.os, operation)
                failed = False
                def fault(*args, **kwargs):
                    nonlocal failed
                    # First write/fsync is the permanent marker. Directory-open
                    # closes happen first, so close faults target regular files.
                    fd = args[0]
                    is_file = __import__('stat').S_ISREG(os.fstat(fd).st_mode)
                    if not failed and is_file:
                        failed = True
                        if operation == 'close': original(*args, **kwargs)
                        raise OSError('synthetic marker operation failure')
                    return original(*args, **kwargs)
                first = Public(self.f)
                with mock.patch.object(a.os, operation, side_effect=fault), self.assertRaises(OSError):
                    consume(self.f, folder, first)
                self.assertTrue(failed)
                self.assertEqual(first.calls, [])
                again = Public(self.f)
                with self.assertRaises(FileExistsError): consume(self.f, folder, again)
                self.assertEqual(again.calls, [])

    def test_marker_short_write_and_directory_fsync_keep_restart_closed(self):
        for stage in ('short_write', 'directory_fsync'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                issue(self.f, folder)
                original_write, original_fsync = a.os.write, a.os.fsync
                def write(fd, body):
                    if stage == 'short_write':
                        return original_write(fd, body[:-1])
                    return original_write(fd, body)
                def fsync(fd):
                    if stage == 'directory_fsync' and __import__('stat').S_ISDIR(os.fstat(fd).st_mode):
                        raise OSError('synthetic permanent marker directory fsync failed')
                    return original_fsync(fd)
                first = Public(self.f)
                with mock.patch.object(a.os, 'write', side_effect=write), mock.patch.object(a.os, 'fsync', side_effect=fsync), \
                        self.assertRaises((ValueError, OSError)):
                    consume(self.f, folder, first)
                self.assertEqual(first.calls, [])
                again = Public(self.f)
                with self.assertRaises(FileExistsError): consume(self.f, folder, again)
                self.assertEqual(again.calls, [])

    def test_early_ordinal_gate_does_not_read_token_or_event_file(self):
        f = self.f
        class TokenTrap(dict):
            reads = 0
            def __getitem__(self, key):
                if key == 'GITHUB_TOKEN':
                    self.reads += 1
                    raise AssertionError('token read before ordinal gate')
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key == 'GITHUB_TOKEN':
                    self.reads += 1
                    raise AssertionError('token read before ordinal gate')
                return super().get(key, default)
        env = TokenTrap(environment(f, self.directory, token=True)); env['GITHUB_RUN_NUMBER'] = '2'
        original = c.read_bytes
        def read(path, **kwargs):
            self.assertEqual(Path(path), h.HERE/'claim-admission-control.json')
            return f['control_raw']
        with mock.patch.object(a.os, 'environ', env), mock.patch.object(h, 'release_first', return_value=f['release']), \
                mock.patch.object(c, 'read_bytes', side_effect=read), \
                mock.patch.object(c, 'read_json', side_effect=AssertionError('event read before ordinal gate')), \
                self.assertRaises(ValueError):
            a.main([])
        self.assertEqual(env.reads, 0)
        self.assertEqual(list(self.directory.iterdir()), [])
        with mock.patch.object(a.os, 'environ', env), self.assertRaises(ValueError): a.host_environment()
        self.assertEqual(env.reads, 0)

    def test_private_file_hardlink_or_symlink_never_enters(self):
        for alias in ('hardlink', 'symlink'):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as folder:
                issue(self.f, folder)
                path = Path(folder)/a.directory_name(self.f['run'])/'handoff.json'
                other = Path(folder)/'copied-handoff'
                if alias == 'hardlink':
                    os.link(path, other)
                else:
                    path.rename(other); path.symlink_to(other)
                with self.assertRaises((ValueError, OSError)): consume(self.f, folder)
                again = Public(self.f)
                with self.assertRaises(FileExistsError): consume(self.f, folder, again)
                self.assertEqual(again.calls, [])

    def test_numeric_job_independently_observed_mismatch_burns(self):
        issue(self.f, self.directory)
        public = Public(self.f, wrong_job=True)
        with self.assertRaises(ValueError): consume(self.f, self.directory, public)
        self.assertEqual(len(public.calls), 6)
        again = Public(self.f)
        with self.assertRaises(FileExistsError): consume(self.f, self.directory, again)
        self.assertEqual(again.calls, [])

    def test_own_201_loss_no_permit_no_takeover(self):
        wire = Wire(self.f, fail='create_ref')
        with self.assertRaises(ConnectionResetError): issue(self.f, self.directory, wire)
        self.assertTrue(wire.existing)
        private = self.directory / a.directory_name(self.f['run'])
        self.assertFalse((private / 'handoff.json').exists())
        self.assertFalse((private / 'ready.json').exists())
        before = len(wire.calls)
        with self.assertRaises(FileExistsError): issue(self.f, self.directory, wire)
        self.assertEqual(len(wire.calls), before)
        with tempfile.TemporaryDirectory() as another:
            wire2 = Wire(self.f, existing=True)
            with self.assertRaises(ValueError): issue(self.f, another, wire2)
            self.assertEqual(sum(row[1] == 'POST' for row in wire2.calls), 0)
        host = Public(self.f)
        with self.assertRaises(FileNotFoundError): consume(self.f, self.directory, host)
        self.assertEqual(host.calls, [])

    def test_all_unconfirmed_response_paths_fail_closed_no_permit(self):
        for ticket in a.TICKET_ORDER:
            with self.subTest(ticket=ticket), tempfile.TemporaryDirectory() as folder:
                wire = Wire(self.f, fail=ticket)
                with self.assertRaises(ConnectionResetError): issue(self.f, folder, wire)
                self.assertFalse((Path(folder) / a.directory_name(self.f['run']) / 'ready.json').exists())
                self.assertEqual(len(wire.calls), a.TICKET_ORDER.index(ticket) + 1)

    def test_later_ordinal_attempt_and_host_token_reject_before_client_or_marker(self):
        for key, value in (('GITHUB_RUN_NUMBER', '2'), ('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_NUMBER', '01')):
            env = environment(self.f, self.directory, token=True); env[key] = value
            wire = Wire(self.f)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): issue(self.f, self.directory, wire, env)
            self.assertEqual(wire.calls, [])
            self.assertEqual(list(self.directory.iterdir()), [])
        issue(self.f, self.directory)
        env = environment(self.f, self.directory, token=True)
        with self.assertRaises(ValueError): consume(self.f, self.directory, env=env)
        self.assertFalse((self.directory / a.directory_name(self.f['run']) / 'consumed.json').exists())

    def test_wrong_workflow_path_and_api_ordinal_reject_before_writes(self):
        for ticket, field, value in (('run', 'run_number', 2), ('run', 'workflow_id', True),
                ('workflow', 'path', '.github/workflows/foreign.yml'), ('workflow', 'id', 999)):
            def mutation(got, status, obj):
                if got == ticket: obj[field] = value
                return status, obj
            with tempfile.TemporaryDirectory() as folder:
                wire = Wire(self.f, mutation=mutation)
                with self.assertRaises(ValueError): issue(self.f, folder, wire)
                self.assertFalse(any(row[1] == 'POST' for row in wire.calls))

    def test_private_nonce_tamper_public_receipt_cannot_mint_permit(self):
        issue(self.f, self.directory)
        folder = self.directory / a.directory_name(self.f['run'])
        path = folder / 'handoff.json'
        value = json.loads(path.read_bytes()); value['nonce'] = 'f' * 64
        path.write_bytes(c.canonical(value))
        with self.assertRaises(ValueError): consume(self.f, self.directory)
        self.assertTrue((folder / 'consumed.json').exists())
        with self.assertRaises(ValueError): a.HostPermit(object(), self.f['run'], self.f['admission'])
        with self.assertRaises(ValueError): h.execute({}, self.f['run'], self.f['release'], {}, 'a'*64, 'b'*40, {})

    def test_actual_closed_cli_no_api_private_files_or_models(self):
        for module in (a, h):
            with self.subTest(module=module.__name__), mock.patch.object(a.DirectHTTPS, 'request', side_effect=AssertionError('HTTP forbidden')):
                with self.assertRaises(ValueError): module.main([])
            self.assertEqual(list(self.directory.iterdir()), [])
        for filename in ('claim_admission.py', 'hosted_run.py'):
            process = subprocess.run([sys.executable, '-I', '-B', str(Path(a.__file__).with_name(filename))],
                env=environment(self.f, self.directory), capture_output=True, timeout=5)
            self.assertEqual(process.returncode, 2)
            self.assertNotIn(b'Traceback', process.stdout + process.stderr)
            self.assertNotIn(b'ghs_', process.stdout + process.stderr)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_real_main_paths_wire_source_then_issuer_and_private_host_continuation(self):
        f = self.f
        event_path = self.directory / 'event.json'; event_path.write_bytes(c.canonical(event(f)))
        step_output = self.directory / 'step-output'; step_output.write_bytes(b'')
        issuer_env = environment(f, self.directory, token=True)
        issuer_env.update(GITHUB_EVENT_PATH=str(event_path), GITHUB_OUTPUT=str(step_output))
        original_read = c.read_bytes
        def read(path, **kwargs):
            if Path(path) == h.HERE/'claim-admission-control.json': return f['control_raw']
            if Path(path) == h.HERE/'execution-release.json': return f['release_raw']
            return original_read(path, **kwargs)
        source_args = []
        def source(run, release, *, control=None):
            source_args.append((run, release, control))
            return c.decode_json(f['freeze_raw']), f['identity']['freeze_sha256'], f['identity']['source_tree']
        wire = Wire(f)
        output = io.StringIO()
        with mock.patch.dict(os.environ, issuer_env, clear=True), mock.patch.object(h, 'release_first', return_value=f['release']), \
                mock.patch.object(c, 'read_bytes', side_effect=read), mock.patch.object(h, 'read_bytes', side_effect=read), mock.patch.object(h, 'source_authority', side_effect=source), \
                mock.patch.object(h, 'source_timestamp', return_value=f['identity']['source_timestamp']), \
                mock.patch.object(a.DirectHTTPS, 'request', side_effect=wire.request), contextlib.redirect_stdout(output):
            self.assertEqual(a.main([]), 0)
        self.assertEqual(output.getvalue(), '{"ready":true}\n')
        self.assertEqual(step_output.read_bytes(), b'ready=true\n')
        self.assertEqual(len(source_args), 1)
        self.assertEqual(source_args[0], (f['run'], f['release'], f['control']))
        public = Public(f)
        host_env = environment(f, self.directory); host_env['GITHUB_EVENT_PATH'] = str(event_path)
        entered = []
        def factory():
            self.assertTrue((self.directory/a.directory_name(f['run'])/'consumed.json').exists())
            return public
        def execute(env, run, release, freeze, freeze_sha, tree, preflight, observer=None, *, admission_permit=None, timestamp=None):
            self.assertEqual(type(admission_permit), a.HostPermit)
            proof = admission_permit.enter(run)
            c.validate_bound_admission(proof, f['identity'])
            self.assertEqual(timestamp, f['identity']['source_timestamp'])
            entered.append(proof)
            return 0
        with mock.patch.dict(os.environ, host_env, clear=True), mock.patch.object(h, 'release_first', return_value=f['release']), \
                mock.patch.object(c, 'read_bytes', side_effect=read), mock.patch.object(h, 'read_bytes', side_effect=read), mock.patch.object(h, 'source_authority', side_effect=source), \
                mock.patch.object(h, 'source_timestamp', return_value=f['identity']['source_timestamp']), \
                mock.patch.object(h, 'PublicAPI', side_effect=factory), mock.patch.object(h, 'execute', side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h.main([]), 0)
        self.assertEqual(len(source_args), 2)
        self.assertEqual(len(public.calls), 6)
        self.assertEqual(len(entered), 1)
        self.assertFalse(any('/git/commits/' in route or '/git/ref/heads/claims/' in route for route in public.calls))

    def test_execute_rebinds_private_continuation_before_model_imports(self):
        f = self.f
        issue(f, self.directory)
        preflight, permit, _ = consume(f, self.directory)
        original_read = h.read_bytes
        def read(path, **kwargs):
            if Path(path) == h.HERE/'claim-admission-control.json': return f['control_raw']
            if Path(path) == h.HERE/'execution-release.json': return f['release_raw']
            return original_read(path, **kwargs)
        imported = []
        import builtins
        original_import = builtins.__import__
        def no_model(name, *args, **kwargs):
            imported.append(name)
            if name in ('resource_host', 'runtime_session', 'input_binding', 'torch', 'transformers'):
                raise AssertionError('model/setup import before source binding')
            return original_import(name, *args, **kwargs)
        with mock.patch.object(h, 'read_bytes', side_effect=read), mock.patch.object(builtins, '__import__', side_effect=no_model), \
                self.assertRaises(ValueError):
            h.execute(environment(f,self.directory), f['run'], f['release'], {}, f['identity']['freeze_sha256'],
                'f'*40, preflight, admission_permit=permit, timestamp=f['identity']['source_timestamp'])
        self.assertNotIn('resource_host', imported)
        with self.assertRaises(ValueError): permit.enter(f['run'])

    def test_source_claim_uses_original_release_and_safe_diagnostic_consumption_unknown(self):
        f = self.f
        issue(f, self.directory)
        preflight, permit, _ = consume(f, self.directory)
        proof = permit.enter(f['run'])
        original_read = h.read_bytes
        def read(path, **kwargs):
            if Path(path) == h.HERE/'claim-admission-control.json': return f['control_raw']
            if Path(path) == h.HERE/'execution-release.json': return f['release_raw']
            return original_read(path, **kwargs)
        with mock.patch.object(h, 'read_bytes', side_effect=read):
            claim = h.source_claim(f['run'], f['release'], preflight, f['identity']['freeze_sha256'],
                admission=proof, tree=f['identity']['source_tree'], timestamp=f['identity']['source_timestamp'])
        self.assertEqual(claim['release_raw_utf8'].encode(), f['release_raw'])
        self.assertEqual(claim['source_branch_workflow_runs'], 2)
        import failure_observer
        failure_observer._safe_source_claim(c.canonical(claim), f['run'])
        root = self.directory/'observation'; (root/'authority').mkdir(parents=True)
        (root/'authority/source-claim.json').write_bytes(c.canonical(claim))
        observed = failure_observer.claim_snapshot(root, f['run'])
        self.assertEqual(observed['source_claim']['status'], 'observed_file')
        self.assertIsNone(observed['consumed_attempts'])
        self.assertIsNone(observed['reported_terminal_attempts'])

    def test_step_output_only_after_durable_ready(self):
        issue(self.f, self.directory)
        path = self.directory / 'step-output'; path.write_bytes(b'')
        a._emit_ready({'GITHUB_OUTPUT': str(path)})
        self.assertEqual(path.read_bytes(), b'ready=true\n')


class ClaimHTTPBudgetTests(unittest.TestCase):
    def test_ticket_counts_deadline_and_request_caps(self):
        budget = a.Budget()
        with self.assertRaises(ValueError): budget.reserve('workflow', None)
        budget.reserve('run', None)
        with self.assertRaises(ValueError): budget.reserve('run', None)
        budget.deadline = 0
        with self.assertRaises(ValueError): budget.reserve('workflow', None)
        budget = a.Budget()
        for ticket in a.GET_TICKETS[:9]: budget.reserve(ticket, None)
        with self.assertRaises(ValueError): budget.reserve('create_commit', b'x' * 16385)
        budget.reserve('create_commit', b'x' * 16384)
        with self.assertRaises(ValueError): budget.reserve('create_ref', b'x' * 513)
        budget.reserve('create_ref', b'x' * 512)
        for ticket in a.GET_TICKETS[9:]: budget.reserve(ticket, None)
        self.assertEqual((budget.gets, budget.posts, budget.upload), (11, 2, 16896))

    def test_direct_http_counts_partial_before_transport_failure(self):
        budget = a.Budget(); budget.reserve('run', None)
        response = mock.Mock(status=200)
        response.getheader.return_value = 'identity'
        response.read1.side_effect = [b'abc', http.client.IncompleteRead(b'de', 10)]
        connection = mock.Mock(); connection.getresponse.return_value = response
        with mock.patch.object(a.http.client, 'HTTPSConnection', return_value=connection), self.assertRaises(http.client.IncompleteRead):
            a.DirectHTTPS().request('GET', a.BASE + '/actions/runs/1', None, 'ghs_FAKE', 20, 10, budget.observe)
        self.assertEqual(budget.responses, 5)
        connection.close.assert_called_once()
        self.assertEqual(connection.request.call_args.args[:2], ('GET', a.BASE + '/actions/runs/1'))

    def test_direct_http_single_response_and_aggregate_caps_before_parse(self):
        for remaining in (10, a.MAX_RESPONSE):
            budget = a.Budget(); budget.responses = a.MAX_RESPONSE_TOTAL - remaining
            response = mock.Mock(status=200); response.getheader.return_value = 'identity'
            response.read1.return_value = b'x' * (remaining + 1)
            connection = mock.Mock(); connection.getresponse.return_value = response
            with mock.patch.object(a.http.client, 'HTTPSConnection', return_value=connection), self.assertRaises(ValueError):
                a.DirectHTTPS().request('GET', a.BASE + '/actions/runs/1', None, 'ghs_FAKE', remaining, 10, budget.observe)
            self.assertEqual(budget.responses, a.MAX_RESPONSE_TOTAL + 1)
            connection.close.assert_called_once()

    def test_wrong_status_malformed_json_and_response_token_never_retry(self):
        f = fixture()
        for kind in ('status', 'json', 'token'):
            class Bad:
                def __init__(self): self.calls = 0
                def request(self, method, route, body, token, allowance, remaining, observe):
                    self.calls += 1
                    body = b'{' if kind == 'json' else token.encode() if kind == 'token' else b'{}'
                    observe(body)
                    return (500 if kind == 'status' else 200), body
            wire = Bad(); api = a.GitHubClaimAPI('ghs_FAKE_TOKEN', f['run'], f['control'], transport=wire)
            with self.assertRaises(ValueError): api.request('run')
            with self.assertRaises(ValueError): api.request('run')
            self.assertEqual(wire.calls, 1)
            self.assertTrue(api.budget.failed)
            self.assertGreater(api.budget.responses, 0)
            self.assertEqual(api.receipts, {})


if __name__ == '__main__':
    unittest.main()
