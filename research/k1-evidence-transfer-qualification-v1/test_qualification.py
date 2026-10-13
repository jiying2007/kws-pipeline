"""Invented local hosted/API fixtures only. No HTTP, credentials or models."""
import base64
import ast
import builtins
import copy
import hashlib
import io
from pathlib import Path
import sys
import time
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualification as q


def fixture_payload():
    name = 'synthetic.bin'
    data = hashlib.shake_256(b'k1-transfer-qualification-v1:synthetic-only:not-ASR:2026-10-12').digest(5884246)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data)
    return stream.getvalue()


def fixture_files(payload):
    lock = q.fixture_lock()
    chunks = [{'path': r['path'], 'bytes': r['bytes'], 'sha256': r['sha256'], 'git_blob': r['sha']}
              for r in lock['files'][:-1]]
    outer = {'archive_bytes': q.ARCHIVE_BYTES, 'archive_sha256': q.ARCHIVE_SHA256,
        'chunk_bytes': q.CHUNK_CAP, 'chunk_count': q.CHUNKS, 'chunks': chunks,
        'generator': 'SHAKE-256(seed ASCII), single ZIP_STORED synthetic.bin, fixed 1980 timestamp',
        'member': {'bytes': 5884246, 'path': 'synthetic.bin',
            'sha256': '5f3bf48c6e06dd42571de34ec715f37f5b599da63732f9491bc0ae01fbee510d'},
        'purpose': 'synthetic transport qualification only; not ASR evidence',
        'repository': q.REPOSITORY, 'schema': 'k1-synthetic-transfer-qualification-v1',
        'seed': 'k1-transfer-qualification-v1:synthetic-only:not-ASR:2026-10-12',
        'source_commit': '276e56062b8d9eeac11ea93348a792b6bd008c8c',
        'source_tree': '3e9e1617e0210cce80854c9ddcfb926b7d35d9a2'}
    files = {row['path']: payload[i*q.CHUNK_CAP:(i+1)*q.CHUNK_CAP]
             for i, row in enumerate(lock['files'][:-1])}
    files[lock['files'][-1]['path']] = q.canonical(outer)
    return files


def who(run=123, head='a' * 40):
    return {'repository': q.REPOSITORY, 'run_id': run, 'run_attempt': 1,
            'job_key': q.JOB_KEY, 'executed_head_sha': head}


def source(rows):
    return q.transport.tree_identity(rows)[0]


def tree_reply(rows):
    sha, dirs = q.transport.tree_identity(rows)
    return {'sha': sha, 'truncated': False, 'tree': rows + [
        {'path': name, 'type': 'tree', 'mode': '040000', 'sha': value} for name, value in dirs.items()]}


def release(identity=None, rows=None):
    rows = rows or [{'path': 'README.md', 'mode': '100644', 'type': 'blob',
                     'sha': q.transport.git_sha(b'public source\n')}]
    return {'identity': identity or who(), 'source_tree': source(rows),
            'source_freeze_sha256': 'f' * 64, 'commit_timestamp': '2026-10-13T01:00:00Z'}


def control(phase='checkpoint', pin=None):
    return {'schema': 'k1-hosted-transfer-qualification-control-v1',
        'checkpoint_enabled': phase == 'checkpoint', 'recovery_enabled': phase == 'recovery',
        'root_transfer_release': True, 'independent_source_review': True,
        'producer_pin': pin, **{name: False for name in q.FALSE_FLAGS}}


def pin_for(released, result):
    return {'identity': released['identity'], 'numeric_job_id': 456,
        **{key: released[key] for key in ('source_tree', 'source_freeze_sha256', 'commit_timestamp')},
        **{key: result[key] for key in ('checkpoint_commit', 'checkpoint_tree', 'manifest_sha256')},
        'archive_bytes': q.ARCHIVE_BYTES, 'archive_sha256': q.ARCHIVE_SHA256,
        'checkpoint_readback_verified': True, 'producer_terminal_conclusion': 'failure', 'injected_stop_verified': True}


class FakeAPI:
    """In-memory remote service; production writer is not mocked."""
    def __init__(self, released, phase='checkpoint', producer=None, saved=None):
        self.current = released; self.phase = phase; self.producer = producer
        self.original = [{'path': 'README.md', 'mode': '100644', 'type': 'blob',
                          'sha': q.transport.git_sha(b'public source\n')}]
        self.current_rows = self.original if phase == 'checkpoint' else self.original + [
            {'path': 'recovery-control.json', 'mode': '100644', 'type': 'blob', 'sha': q.transport.git_sha(b'root static pin\n')}]
        self.files = {}; self.objects = {}; self.refs = {}; self.calls = []; self.raw_calls = []
        self.source_sha = released['identity']['executed_head_sha']
        self.source_tree = released['source_tree']; self.result_branch = q.FINAL; self.checkpoint_branch = q.CHECKPOINT
        if saved:
            self.files = copy.deepcopy(saved.files); self.objects = copy.deepcopy(saved.objects)
            self.refs = copy.deepcopy(saved.refs)
            self.producer_rows = saved.original
        self.expected_body = None

    def bind_result(self, sha, tree):
        self.result = (sha, tree)

    def run_reply(self, released, phase, terminal):
        return {'id': released['identity']['run_id'], 'run_attempt': 1,
            'head_sha': released['identity']['executed_head_sha'], 'head_branch': q.PHASES[phase],
            'event': 'push', 'path': q.WORKFLOW, 'status': 'completed' if terminal else 'in_progress',
            'conclusion': 'failure' if terminal else None,
            'repository': {'full_name': q.REPOSITORY, 'private': False}}

    def __call__(self, method, route, body=None, expected=200):
        self.calls.append((method, route, body, expected))
        if route == q.BASE:
            return {'full_name': q.REPOSITORY, 'private': False}
        for released, phase, terminal in ((self.current, self.phase, False), (self.producer, 'checkpoint', True)):
            if released is None:
                continue
            rr = q.run_routes(released['identity'], phase)
            result = self.run_reply(released, phase, terminal)
            if route == rr[0]:
                return result
            if route == rr[1]:
                return {'total_count': 1, 'workflow_runs': [result]}
            if route == rr[2]:
                return {'total_count': 1, 'jobs': [{'id': 456 if phase == 'checkpoint' else 789,
                    'name': q.JOB_KEY, 'run_id': released['identity']['run_id'],
                    'head_sha': result['head_sha'], 'head_branch': result['head_branch'],
                    'run_url': 'https://api.github.com' + rr[0], 'status': result['status'], 'conclusion': result['conclusion']}]}
        for released, rows in ((self.current, self.current_rows), (self.producer, self.original)):
            if released is None:
                continue
            if route == q.BASE + '/git/commits/' + released['identity']['executed_head_sha']:
                return {'sha': released['identity']['executed_head_sha'], 'tree': {'sha': released['source_tree']}}
            if route == q.BASE + '/git/trees/' + released['source_tree'] + '?recursive=1':
                return tree_reply(rows)
        if route.startswith(q.BASE + '/git/ref/heads/'):
            branch = route.split('/git/ref/heads/')[1]
            if expected == 404:
                if branch in self.refs:
                    raise ValueError('destination exists')
                return None
            return {'ref': 'refs/heads/' + branch, 'object': {'type': 'commit', 'sha': self.refs[branch]}}
        if method == 'POST' and route == q.BASE + '/git/blobs':
            raw = base64.b64decode(body['content'], validate=True)
            sha = q.transport.git_sha(raw); self.objects[sha] = raw
            return {'sha': sha}
        if method == 'POST' and route == q.BASE + '/git/trees':
            self.output_rows = body['tree']
            self.files = {row['path']: self.objects[row['sha']] for row in body['tree']}
            self.objects[self.result[1]] = tree_reply(self.original + self.output_rows)
            return {'sha': self.result[1]}
        if method == 'POST' and route == q.BASE + '/git/commits':
            commit = {'sha': self.result[0], 'message': body['message'], 'tree': {'sha': body['tree']},
                'parents': [{'sha': body['parents'][0]}], 'author': body['author'], 'committer': body['committer']}
            self.objects[self.result[0]] = commit
            return commit
        if method == 'POST' and route == q.BASE + '/git/refs':
            branch = body['ref'][len('refs/heads/'):]
            if branch in self.refs:
                raise ValueError('existing ref')
            self.refs[branch] = body['sha']
            return {'ref': body['ref'], 'object': {'type': 'commit', 'sha': body['sha']}}
        if route.startswith(q.BASE + '/git/commits/'):
            return self.objects[route.rsplit('/', 1)[-1]]
        if route.startswith(q.BASE + '/git/trees/'):
            return self.objects[route.rsplit('/', 1)[-1].split('?')[0]]
        raise AssertionError((method, route))

    def raw(self, sha, paths):
        files = self.input_files if sha == q.INPUT_COMMIT else self.files
        def read(path):
            self.raw_calls.append((sha, path))
            return files[path]
        return read


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = fixture_payload()
        cls.inputs = fixture_files(cls.payload)

    def producer(self):
        released = release(); api = FakeAPI(released); api.input_files = self.inputs
        with self.assertRaises(q.InjectedCheckpointStop) as raised:
            q.checkpoint_phase(api, api.raw, released)
        return released, api, raised.exception.receipt

    def recovery(self):
        old, producer, receipt = self.producer()
        pin = pin_for(old, receipt)
        current = release(who(124, 'b' * 40), producer.original + [
            {'path': 'recovery-control.json', 'mode': '100644', 'type': 'blob', 'sha': q.transport.git_sha(b'root static pin\n')}])
        api = FakeAPI(current, 'recovery', old, producer)
        return current, pin, api

    def test_checked_in_source_authority_is_checkpoint_only(self):
        value = q.read_json(HERE / 'qualification-control.json')
        q.control_gate(value, 'checkpoint')
        self.assertIsNone(value['producer_pin'])
        self.assertEqual(len(q.FALSE_FLAGS), 12)
        with self.assertRaises(q.QualificationDisabled):
            q.control_gate(value, 'recovery')

    def test_fixture_exact_bytes(self):
        self.assertEqual(len(self.payload), q.ARCHIVE_BYTES)
        self.assertEqual(q.digest(self.payload), q.ARCHIVE_SHA256)
        self.assertEqual(q.read_fixture(self.inputs.__getitem__), self.payload)

    def test_fixture_mutation_rejected(self):
        files = dict(self.inputs); name = next(iter(files)); files[name] = b'0' + files[name][1:]
        with self.assertRaises(ValueError):
            q.read_fixture(files.__getitem__)

    def test_synthetic_never_admitted_as_asr(self):
        for phase in ('checkpoint', 'recovery'):
            with self.assertRaises(ValueError):
                q.transport.policy(phase)
        with self.assertRaises(ValueError):
            q.transport.identity(who())
        scientific = dict(who(), job_key='asr18')
        with self.assertRaises(ValueError):
            q.transport.validate_archive(self.payload, scientific)

    def test_model_flags_cannot_be_enabled(self):
        for key in q.FALSE_FLAGS:
            value = control(); value[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.control_gate(value, 'checkpoint')

    def test_control_strict_fields_and_types(self):
        for value in (dict(control(), unexpected=True), dict(control(), checkpoint_enabled=1),
                      dict(control(), root_transfer_release=1), dict(control(), recovery_enabled=True),
                      dict(control(), producer_pin={})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                q.control_gate(value, 'checkpoint')

    def test_disabled_gate_precedes_environment_or_token(self):
        value = control(); value['checkpoint_enabled'] = False
        class NoEnv(dict):
            def get(self, *args):
                raise AssertionError('environment access while disabled')
            def __contains__(self, key):
                raise AssertionError('environment access while disabled')
        with mock.patch.object(q, 'read_json', return_value=value), mock.patch.object(q.os, 'environ', NoEnv()):
            with self.assertRaises(q.QualificationDisabled):
                q.main([])

    def test_recovery_pin_required_before_environment(self):
        with mock.patch.object(q, 'read_json', return_value=control('recovery')), \
             mock.patch.object(q.os, 'environ', {}):
            with self.assertRaises(ValueError):
                q.main([])

    def test_no_cli_overrides(self):
        with self.assertRaises(ValueError):
            q.main(['--checkpoint', 'foreign'])

    def test_context_requires_exact_created_public_first_attempt(self):
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': q.REPOSITORY, 'GITHUB_EVENT_NAME': 'push',
            'GITHUB_REF': 'refs/heads/' + q.PHASES['checkpoint'], 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1',
            'GITHUB_JOB': q.JOB_KEY, 'GITHUB_SHA': 'a' * 40, 'GITHUB_WORKFLOW_SHA': 'a' * 40,
            'GITHUB_WORKFLOW_REF': q.REPOSITORY + '/' + q.WORKFLOW + '@refs/heads/' + q.PHASES['checkpoint']}
        event = {'repository': {'full_name': q.REPOSITORY, 'private': False}, 'created': True, 'deleted': False,
            'forced': False, 'before': '0' * 40, 'after': 'a' * 40, 'ref': env['GITHUB_REF']}
        self.assertEqual(q.context(env, event, 'checkpoint'), who())
        for key, value in (('GITHUB_EVENT_NAME', 'workflow_dispatch'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_JOB', 'asr18'), ('GITHUB_REF', 'refs/heads/main'),
                           ('GITHUB_WORKFLOW_SHA', 'b' * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.context(dict(env, **{key: value}), event, 'checkpoint')
        for key, value in (('created', False), ('deleted', True), ('forced', True), ('before', 'a' * 40),
                           ('after', 'b' * 40), ('repository', {'full_name': q.REPOSITORY, 'private': True})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.context(env, dict(event, **{key: value}), 'checkpoint')

    def test_real_shared_writer_checkpoint_then_fixed_failure(self):
        released, api, receipt = self.producer()
        self.assertEqual(len(api.calls), 38)
        self.assertEqual(len(api.raw_calls), 48)
        self.assertEqual(sum(method == 'POST' for method, *_ in api.calls), 27)
        self.assertEqual(api.refs, {q.CHECKPOINT: receipt['checkpoint_commit']})
        self.assertFalse(receipt['final_ref_created'])
        self.assertEqual(receipt['identity'], released['identity'])
        self.assertTrue(all(receipt[key] is False for key in q.FALSE_FLAGS))

    def test_recovery_only_reads_new_checkpoint_and_preserves_identity(self):
        current, pin, api = self.recovery()
        with mock.patch.object(q, 'read_fixture', side_effect=AssertionError('old fixture read')), \
             mock.patch.object(q.Path, 'read_bytes', side_effect=AssertionError('producer filesystem read')):
            receipt = q.recovery_phase(api, api.raw, current, pin)
        self.assertEqual(len(api.calls), 17)
        self.assertEqual(len(api.raw_calls), 24)
        self.assertEqual({sha for sha, _ in api.raw_calls}, {pin['checkpoint_commit']})
        posts = [(route, body) for method, route, body, _ in api.calls if method == 'POST']
        self.assertEqual(posts, [(q.BASE + '/git/refs', {'ref': 'refs/heads/' + q.FINAL, 'sha': pin['checkpoint_commit']})])
        self.assertEqual(receipt['producer_identity'], pin['identity'])
        self.assertEqual(receipt['producer_numeric_job_id'], 456)
        self.assertEqual(receipt['recovery_numeric_job_id'], 789)
        self.assertEqual(receipt['final_commit'], pin['checkpoint_commit'])
        self.assertEqual(receipt['producer_terminal_conclusion'], 'failure')

    def test_corrupt_checkpoint_prevents_final_ref(self):
        current, pin, api = self.recovery()
        name = q.DIRECTORY + '/chunks/0000.bin'; api.files[name] = b'x' + api.files[name][1:]
        with self.assertRaises(ValueError):
            q.recovery_phase(api, api.raw, current, pin)
        self.assertNotIn(q.FINAL, api.refs)
        self.assertFalse(any(method == 'POST' for method, *_ in api.calls))

    def test_recovery_wrong_identity_manifest_or_commit_fails_closed(self):
        for field, value in (('numeric_job_id', 999), ('manifest_sha256', '0' * 64),
                             ('checkpoint_commit', 'c' * 40), ('source_freeze_sha256', '1' * 64)):
            current, pin, api = self.recovery(); pin[field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                q.recovery_phase(api, api.raw, current, pin)
            self.assertFalse(any(method == 'POST' for method, *_ in api.calls))

    def test_recovery_requires_explicit_failed_producer_acceptance(self):
        current, pin, _ = self.recovery()
        for key, value in (('producer_terminal_conclusion', 'success'), ('injected_stop_verified', False),
                           ('checkpoint_readback_verified', False), ('archive_sha256', '0' * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_pin(dict(pin, **{key: value}))

    def test_same_size_manifest_rebinding_rejected(self):
        released, api, receipt = self.producer()
        files = dict(api.files); files[q.MANIFEST] = files[q.MANIFEST].replace(b'"run_id":123', b'"run_id":124')
        with self.assertRaises(ValueError):
            q.verify_files(files, released, 456)

    def test_extra_missing_reordered_paths_rejected(self):
        released, api, _ = self.producer()
        for files in (dict(api.files, extra=b'not allowed'), {k: v for k, v in api.files.items() if k != q.MANIFEST}):
            with self.assertRaises(ValueError):
                q.verify_files(files, released, 456)
        files = dict(api.files); a = q.DIRECTORY + '/chunks/0000.bin'; b = q.DIRECTORY + '/chunks/0001.bin'
        files[a], files[b] = files[b], files[a]
        with self.assertRaises(ValueError):
            q.verify_files(files, released, 456)

    def test_wrong_existing_ref_stops_before_upload(self):
        for branch in (q.CHECKPOINT, q.FINAL):
            api = FakeAPI(release()); api.input_files = self.inputs; api.refs[branch] = '0' * 40
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                q.checkpoint_phase(api, api.raw, release())
            self.assertFalse(any(method == 'POST' for method, *_ in api.calls))

    def test_final_failure_never_removes_checkpoint(self):
        current, pin, api = self.recovery()
        api.refs[q.FINAL] = '0' * 40
        with self.assertRaises(ValueError):
            q.recovery_phase(api, api.raw, current, pin)
        self.assertEqual(api.refs[q.CHECKPOINT], pin['checkpoint_commit'])

    def test_raw_recovery_has_no_old_fixture_route(self):
        current, pin, _ = self.recovery()
        factory = q.bounded_raw_factory('recovery', time.monotonic() + 30, pin)
        with self.assertRaises(ValueError):
            factory(q.INPUT_COMMIT, {r['path']: r['bytes'] for r in q.fixture_lock()['files']})

    def test_raw_deadline_stops_before_network(self):
        factory = q.bounded_raw_factory('checkpoint', time.monotonic() - 1)
        with self.assertRaises(ValueError):
            factory(q.INPUT_COMMIT, {r['path']: r['bytes'] for r in q.fixture_lock()['files']})

    def test_low_level_recovery_client_cannot_create_objects_or_checkpoint(self):
        current, pin, _ = self.recovery()
        api = q.publisher.GitHubAPI.__new__(q.publisher.GitHubAPI)
        body, sha = q.commit_body(pin['checkpoint_tree'], {k: pin[k] for k in
            ('identity', 'source_tree', 'source_freeze_sha256', 'commit_timestamp')})
        api._initialize('ghs_fake', get_routes=q.api_routes(current, 'recovery', pin),
            source_sha=pin['identity']['executed_head_sha'], source_tree=pin['source_tree'],
            result_branch=q.FINAL, checkpoint_branch=q.CHECKPOINT, paths=q.output_paths(),
            expected_body=body, final_ref_only=True, max_requests=17, timeout=600, max_upload=9 * 1024 ** 2)
        api.bind_result(sha, pin['checkpoint_tree'])
        with mock.patch.object(q.publisher.http.client, 'HTTPSConnection', side_effect=AssertionError('no HTTP expected')):
            for noun in ('blobs', 'trees', 'commits'):
                with self.subTest(noun=noun), self.assertRaises(ValueError):
                    api('POST', q.BASE + '/git/' + noun, {}, expected=201)
            with self.assertRaises(ValueError):
                api('POST', q.BASE + '/git/refs', {'ref': 'refs/heads/' + q.CHECKPOINT, 'sha': sha}, expected=201)

    def test_source_closure_is_separate_and_fixed(self):
        self.assertNotIn(q.authority.REL + 'source-freeze.json', q.authority.ROSTER)
        self.assertEqual(len(q.authority.ROSTER), len(set(q.authority.ROSTER)))
        self.assertIn(q.WORKFLOW, q.authority.ROSTER)
        self.assertNotIn(q.authority.REL + 'test_qualification.py', q.authority.ROSTER)
        import prepare_freeze
        for paths in prepare_freeze.declared_rosters(q.ROOT).values():
            self.assertFalse(any(name.startswith(q.authority.REL) for name in paths))

    def test_all_eager_local_imports_are_frozen_transitively(self):
        # Audit complete eager imports, including imports within module-level
        # conditionals. Lazy ASR-only functions are separately forbidden below.
        pending = [HERE / 'qualification.py']; seen = set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            nodes = list(ast.parse(path.read_text()).body)
            while nodes:
                node = nodes.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.Import):
                    modules = [item.name.split('.')[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [(node.module or '').split('.')[0]]
                else:
                    modules = []
                    nodes.extend(ast.iter_child_nodes(node))
                for name in modules:
                    for root in (HERE, q.ASR):
                        target = root / (name + '.py')
                        if target.is_file():
                            self.assertIn(str(target.relative_to(q.ROOT)), q.authority.ROSTER)
                            pending.append(target)
        self.assertIn(q.ASR / 'asr18_metadata.py', seen)

    def test_transport_phases_cannot_import_lazy_scientific_entrypoints(self):
        original_import = builtins.__import__
        forbidden = {'prepare_prerequisite', 'prepare_freeze', 'hosted_run', 'worker_receipts',
                     'native_worker', 'torch', 'transformers', 'funasr', 'whisper'}
        def admit(name, *args, **kwargs):
            if name.split('.')[0] in forbidden:
                raise AssertionError('scientific import reached')
            return original_import(name, *args, **kwargs)
        current, pin, api = self.recovery()
        with mock.patch.object(builtins, '__import__', side_effect=admit):
            self.producer()
            q.recovery_phase(api, api.raw, current, pin)

    def test_fixed_source_freeze_detects_content_and_symlink_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in q.authority.ROSTER:
                file = root / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'public source\n')
            before = q.authority.build(root)
            (root / q.authority.ROSTER[0]).write_bytes(b'different source\n')
            self.assertNotEqual(before, q.authority.build(root))
            file = root / q.authority.ROSTER[-1]; file.unlink(); file.symlink_to(root / q.authority.ROSTER[0])
            with self.assertRaises(ValueError):
                q.authority.build(root)

    def test_source_authority_fails_before_token_on_dirty_source(self):
        env = {'GITHUB_SHA': 'a' * 40, 'KWS_TRANSFER_SOURCE_FREEZE_SHA256': '0' * 64}
        with mock.patch.object(q, 'context', return_value=who()), \
             mock.patch.object(q.publisher, '_source_git', return_value='dirty'), \
             mock.patch.object(q.publisher.GitHubAPI, '_initialize', side_effect=AssertionError('token admission')):
            with self.assertRaisesRegex(ValueError, 'clean source checkout'):
                q.source_release(env, {}, 'checkpoint', control())

    def test_raw_two_pass_counter_deadline_and_repeated_reader(self):
        released, api, receipt = self.producer()
        # The production factory calls reader(commit,paths,deadline); this
        # explicit local fixture never opens a real network connection.
        with mock.patch.object(q, 'raw_reader', side_effect=lambda commit, paths, deadline: api.raw(commit, paths)):
            factory = q.bounded_raw_factory('checkpoint', time.monotonic() + 30)
            q.read_fixture(factory(q.INPUT_COMMIT, {r['path']: r['bytes'] for r in q.fixture_lock()['files']}))
            read = factory(receipt['checkpoint_commit'], q.output_paths())
            for path in q.output_paths():
                read(path)
            self.assertEqual(factory.counts['requests'], 48)
            self.assertLess(factory.counts['reserved_bytes'], 16 * 1024 ** 2)
            with self.assertRaises(ValueError):
                read(q.MANIFEST)
            with self.assertRaises(ValueError):
                factory(receipt['checkpoint_commit'], q.output_paths())

    def test_hosted_workflow_exact_free_runner_and_two_one_use_sources(self):
        workflow = (q.ROOT / q.WORKFLOW).read_text()
        trigger = "on:\n  push:\n    branches:\n      - " + q.PHASES['checkpoint'] + "\n      - " + q.PHASES['recovery'] + "\npermissions:"
        self.assertIn(trigger, workflow)
        self.assertEqual(workflow.count('    runs-on: ubuntu-24.04\n'), 1)
        self.assertEqual(workflow.count('      - uses:'), 1)
        self.assertIn('  transfer_qualification:\n', workflow)
        for text in ('github.event.created == true', 'github.event.deleted == false',
                     'github.event.forced == false', 'github.event.repository.private == false',
                     'github.run_attempt == 1', 'persist-credentials: false',
                     'KWS_TRANSFER_SOURCE_FREEZE_SHA256', '    timeout-minutes: 18\n',
                     '        timeout-minutes: 11\n'):
            self.assertIn(text, workflow)
        for forbidden in ('  workflow_dispatch:', '  workflow_run:', '  schedule:',
                          '  pull_request:', 'upload-artifact@', 'actions/cache@', 'strategy:', 'continue-on-error:'):
            self.assertNotIn(forbidden, workflow)
        self.assertEqual(workflow.count('GITHUB_TOKEN: ${{ github.token }}'), 1)

    def test_api_budget_caps_are_strict_subsets(self):
        current, pin, _ = self.recovery()
        for phase, cap in (('checkpoint', 40), ('recovery', 17)):
            released = release() if phase == 'checkpoint' else current
            api = q.publisher.GitHubAPI.__new__(q.publisher.GitHubAPI)
            api._initialize('ghs_fake', get_routes=q.api_routes(released, phase, pin if phase == 'recovery' else None),
                source_sha=released['identity']['executed_head_sha'], source_tree=released['source_tree'],
                result_branch=q.FINAL, checkpoint_branch=q.CHECKPOINT, paths=q.output_paths(),
                final_ref_only=phase == 'recovery', max_requests=cap, timeout=600,
                max_upload=9 * 1024 ** 2, max_body=8 * 1024 ** 2)
            api.count = cap
            with mock.patch.object(q.publisher.http.client, 'HTTPSConnection', side_effect=AssertionError('no HTTP beyond cap')):
                with self.assertRaises(ValueError):
                    api('GET', q.BASE)

    def test_failure_envelope_projects_only_safe_fixed_metadata(self):
        secret = 'never-log-token-body-private-payload'
        error = q.publisher.PublicationError(503)
        error.args = (secret,)
        error.untrusted_body = secret
        def failed(argv, state):
            api = q.publisher.GitHubAPI.__new__(q.publisher.GitHubAPI)
            api.count = 9; api.request_bytes = 123; api.body_bytes = 456
            api.blob_api_attempt_count = 1; api.blob_recovery_get_count = 0; api.blob_retry_count = 0
            api._token = secret
            def raw_get():
                pass
            raw_get.counts = {'requests': 24, 'reserved_bytes': 5889922, 'body_bytes': 5889922, 'extra': secret}
            state.update(phase='checkpoint', identity=who(), api=api, raw_get=raw_get,
                progress={'stage': 'create_blobs', 'checkpoint_commit': 'c' * 40,
                    'checkpoint_tree': 'd' * 40, 'checkpoint_preserved': False, 'error_body': secret})
            raise error
        with mock.patch.object(q, '_run', side_effect=failed), mock.patch('builtins.print') as output:
            with self.assertRaises(q.publisher.PublicationError) as caught:
                q.main([])
        self.assertIs(caught.exception, error)
        raw = output.call_args.args[0]
        self.assertNotIn(secret, raw)
        record = q.transport.decode(raw.encode())
        self.assertEqual(record['stage'], 'create_blobs')
        self.assertEqual(record['http_status'], 503)
        self.assertEqual(record['api']['count'], 9)
        self.assertEqual(record['raw']['requests'], 24)
        self.assertEqual(record['checkpoint_commit'], 'c' * 40)
        self.assertFalse(record['checkpoint_preserved'])
        self.assertIsNone(record['checkpoint_created'])
        empty = q.failure_record({}, RuntimeError(secret))
        self.assertIsNone(empty['identity'])
        self.assertIsNone(empty['http_status'])
        self.assertIsNone(empty['checkpoint_preserved'])
        self.assertIsNone(empty['api']['count'])
        self.assertNotIn(secret, q.canonical(empty).decode())

    def test_failure_observation_cannot_replace_primary_exception(self):
        error = ValueError('private failure text')
        with mock.patch.object(q, '_run', side_effect=error), \
             mock.patch('builtins.print', side_effect=OSError('observer failed')):
            with self.assertRaises(ValueError) as caught:
                q.main([])
        self.assertIs(caught.exception, error)

    def test_original_raw_reader_counts_failed_http_and_overlimit_bodies(self):
        paths = {row['path']: row['bytes'] for row in q.fixture_lock()['files']}
        path = next(iter(paths))
        for status, returned in ((503, b'received-17-bytes!'), (200, b'x' * (paths[path] + 1))):
            calls = []
            class Response:
                def __init__(self):
                    self.status = status
                def read(self, cap):
                    calls.append(('read', cap))
                    return returned[:cap]
            class Connection:
                def __init__(self, *args, **kwargs):
                    calls.append(('connect', args[0]))
                def request(self, *args, **kwargs):
                    calls.append(('request', args[0]))
                def getresponse(self):
                    return Response()
                def close(self):
                    pass
            factory = q.bounded_raw_factory('checkpoint', time.monotonic() + 30)
            with self.subTest(status=status, size=len(returned)), \
                 mock.patch.object(q.publisher.http.client, 'HTTPSConnection', Connection):
                read = factory(q.INPUT_COMMIT, paths)
                with self.assertRaises(q.publisher.PublicationError) as caught:
                    read(path)
                record = q.failure_record({'phase': 'checkpoint', 'raw_get': factory}, caught.exception)
            self.assertEqual(factory.counts['requests'], 1)
            self.assertEqual(factory.counts['body_bytes'], len(returned))
            self.assertEqual(record['raw']['body_bytes'], len(returned))
            self.assertEqual(record['http_status'], status)
            self.assertEqual(factory.counts['reserved_bytes'], paths[path] + 1)
            self.assertEqual(sum(item[0] == 'request' for item in calls), 1)
            self.assertEqual([item[1] for item in calls if item[0] == 'read'], [paths[path] + 1])

    def test_original_raw_reader_success_is_counted_once_across_reads(self):
        paths = {row['path']: row['bytes'] for row in q.fixture_lock()['files']}
        selected = list(paths)[:2]
        calls = []
        class Response:
            status = 200
            def read(self, cap):
                return b'ok'
        class Connection:
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                calls.append(args)
            def getresponse(self):
                return Response()
            def close(self):
                pass
        factory = q.bounded_raw_factory('checkpoint', time.monotonic() + 30)
        with mock.patch.object(q.publisher.http.client, 'HTTPSConnection', Connection):
            read = factory(q.INPUT_COMMIT, paths)
            self.assertEqual(read(selected[0]), b'ok')
            self.assertEqual(factory.counts['body_bytes'], 2)
            self.assertEqual(read(selected[1]), b'ok')
            self.assertEqual(factory.counts['body_bytes'], 4)
        self.assertEqual(len(calls), 2)
        self.assertEqual(factory.counts['requests'], 2)



if __name__ == '__main__':
    unittest.main(verbosity=2)
