"""Offline mock-only Docker API/command tests. Never contact a Docker daemon."""
import ast
import hashlib
from contextlib import redirect_stdout, nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import prepare_launch_ci as ci
host = ci.host


def inspect_fixture(args):
    """Docker CLI --tmpfs representation, with only six binds in Mounts."""
    index = args.index(ci.IMAGE)
    flag = lambda key: args[args.index(key) + 1]
    mounts, tmpfs, env = [], {}, []
    for i, item in enumerate(args[:index]):
        if item == '--mount':
            fields = args[i + 1].split(',')
            values = dict(part.split('=', 1) for part in fields if '=' in part)
            mounts.append({'Type': 'bind', 'Source': values['src'], 'Destination': values['dst'],
                'RW': 'readonly' not in fields, 'Propagation': 'rprivate'})
        if item == '--tmpfs':
            destination, options = args[i + 1].split(':', 1)
            tmpfs[destination] = options
        if item == '--env':
            env.append(args[i + 1])
    labels = dict([flag('--label').split('=', 1)]) if '--label' in args else {}
    return {'Id': 'a' * 64, 'Name': '/' + flag('--name'),
        'State': {'Status': 'created', 'Running': False, 'Paused': False, 'Restarting': False,
            'Dead': False, 'Pid': 0, 'StartedAt': ci.EMPTY_TIME, 'FinishedAt': ci.EMPTY_TIME},
        'Config': {'Image': ci.IMAGE, 'Entrypoint': None, 'Cmd': list(args[index + 1:]),
            'Env': env, 'User': flag('--user'), 'Labels': labels},
        'Mounts': mounts, 'HostConfig': {'Memory': 12 * 1024 ** 3, 'MemorySwap': 12 * 1024 ** 3,
            'NanoCpus': 4_000_000_000, 'PidsLimit': 256, 'NetworkMode': 'none',
            'ReadonlyRootfs': True, 'Privileged': False, 'CgroupnsMode': 'private', 'PidMode': '',
            'IpcMode': 'private', 'CapAdd': [], 'CapDrop': ['ALL'],
            'SecurityOpt': ['no-new-privileges'],
            'LogConfig': {'Type': 'local', 'Config': {'max-size': '1m', 'max-file': '1', 'compress': 'false'}},
            'Tmpfs': tmpfs}}


class DockerMock:
    def __init__(self, mutate=None, create_error=None, no_cid=False, cleanup_error=False, listed=b''):
        self.calls = []
        self.record = None
        self.mutate = mutate
        self.create_error = create_error
        self.no_cid = no_cid
        self.cleanup_error = cleanup_error
        self.listed = listed

    def __call__(self, command, **kwargs):
        if command[:3] != ['/usr/bin/docker', '--host', 'unix:///var/run/docker.sock']:
            raise AssertionError('unexpected executable')
        if set(kwargs['env']) != {'PATH', 'HOME', 'DOCKER_CONFIG', 'LANG'}:
            raise AssertionError('inherited environment')
        if kwargs['stdin'] is not subprocess.DEVNULL or kwargs['check'] is not True:
            raise AssertionError('process boundary')
        args = tuple(command[3:])
        self.calls.append(args)
        if args == ('version', '--format', ci.VERSION_FORMAT):
            data = {'Version': '28.0.4', 'ApiVersion': '1.48', 'MinAPIVersion': '1.24',
                'GitCommit': 'abc1234', 'GoVersion': 'go1.23.7', 'Os': 'linux', 'Arch': 'amd64'}
            output = json.dumps({'Client': data, 'Server': data}).encode()
        elif args == ('pull', '--platform', 'linux/amd64', ci.IMAGE):
            output = b'fixed test image pulled'
        elif args[0] == 'create':
            self.record = inspect_fixture(args)
            if self.mutate:
                self.mutate(self.record)
            if not self.no_cid:
                Path(args[args.index('--cidfile') + 1]).write_text('a' * 64)
            if self.create_error:
                raise self.create_error
            output = b'a' * 64 + b'\n'
        elif args == ('container', 'inspect', 'a' * 64):
            output = json.dumps([self.record]).encode()
        elif args == ('container', 'rm', 'a' * 64):
            if self.cleanup_error:
                raise ValueError('ci_removed_exact_cid')
            output = b'a' * 64 + b'\n'
        elif args == ('container', 'ls', '--all', '--no-trunc', '--filter', 'id=' + 'a' * 64, '--format', '{{.ID}}'):
            output = self.listed
        else:
            raise AssertionError('forbidden Docker command reached subprocess')
        return types.SimpleNamespace(stdout=output, stderr=b'')


class LaunchCITests(unittest.TestCase):
    def fixture(self, directory):
        docker = ci.NoStartDocker(Path(directory))
        args = docker.arguments()
        return docker, args, inspect_fixture(args)

    def test_pure_builder_exact_frozen_production_body(self):
        source = ast.parse((HERE / 'prepare_host.py').read_text())
        helper = next(node for node in source.body if isinstance(node, ast.FunctionDef)
            and node.name == '_worker_arguments')
        # Frozen main54ee9ee prepare_host worker_arguments, only release_first
        # removed and a docstring added: every argv construction node is intact.
        body = ast.dump(ast.Module(body=helper.body[1:], type_ignores=[]), include_attributes=False).encode()
        self.assertEqual(hashlib.sha256(body).hexdigest(),
            '3a28bc48c9226cdba97f38a583e84b3c4ee5d970abeea03db662ab558cf97d2e')

    def test_shared_builder_production_release_first_and_no_start_ci_no_release(self):
        kw = dict(name='fixed-name', image=ci.IMAGE, code='/fixture/code', runtime='/fixture/runtime',
            output='/fixture/output', metadata='/fixture/metadata', authority='/fixture/authority',
            family='qwen06', run_identity={'executed_head_sha': 'a' * 40, 'run_id': 123},
            freeze_sha='b' * 64, claim_sha='c' * 64)
        # Real source-local gates run against explicit control files, never a
        # mocked release return. Positive coverage constructs argv only.
        release=host.c.decode(host.c.regular(HERE/'prepare-release.json'))
        with tempfile.TemporaryDirectory() as temp:
            fixed=Path(temp)
            for key in ('enabled','independent_source_review','source_closure_ready'):release[key]=False
            (fixed/'prepare-release.json').write_bytes(host.c.canonical(release))
            with patch.object(host.c,'HERE',fixed), \
                    patch.object(host,'_worker_arguments',side_effect=AssertionError('builder reached')):
                with self.assertRaisesRegex(ValueError,'prepare_execution_disabled'):
                    host.worker_arguments(**kw)
            for key in ('enabled','independent_source_review','source_closure_ready'):release[key]=True
            (fixed/'prepare-release.json').write_bytes(host.c.canonical(release))
            with patch.object(host.c,'HERE',fixed):
                production=host.worker_arguments(**kw)
        self.assertEqual(production, host._worker_arguments(**kw))
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(host.c, 'release_first', side_effect=AssertionError('release bypass attempted')), \
                patch.object(ci.subprocess, 'run', side_effect=AssertionError('Docker is forbidden offline')):
            docker, args, record = self.fixture(temp)
            self.assertTrue(host.validate_launch(record, args))
            self.assertEqual(len(record['Mounts']), 6)
            self.assertEqual({r['Type'] for r in record['Mounts']}, {'bind'})

    def test_actual_tmpfs_shape_full_exact_options_required(self):
        with tempfile.TemporaryDirectory() as temp:
            _, args, record = self.fixture(temp)
            self.assertTrue(host.validate_launch(record, args))
            for bad in (None, {}, {'/tmp': record['HostConfig']['Tmpfs']['/tmp']},
                    {**record['HostConfig']['Tmpfs'], '/extra': 'rw,size=1'},
                    {**record['HostConfig']['Tmpfs'], '/tmp/': record['HostConfig']['Tmpfs']['/tmp']}):
                altered = deepcopy(record); altered['HostConfig']['Tmpfs'] = bad
                with self.subTest(tmpfs=bad), self.assertRaisesRegex(ValueError, 'fixed_tmpfs_roster'):
                    host.validate_launch(altered, args)
            for option in ('rw', 'noexec', 'nosuid', 'nodev', 'size=134217728', 'uid=', 'gid=', 'mode=0700'):
                altered = deepcopy(record)
                altered['HostConfig']['Tmpfs']['/tmp'] = ','.join(part for part in
                    altered['HostConfig']['Tmpfs']['/tmp'].split(',') if not part.startswith(option))
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'fixed_tmpfs_options'):
                    host.validate_launch(altered, args)
            altered = deepcopy(record)
            altered['HostConfig']['Tmpfs']['/tmp'] += ',exec'
            with self.assertRaisesRegex(ValueError, 'fixed_tmpfs_options'):
                host.validate_launch(altered, args)
            altered = list(args)
            index = altered.index('--tmpfs')
            altered[index + 1] = altered[index + 1].replace('noexec,', '')
            record['HostConfig']['Tmpfs']['/scratch'] = record['HostConfig']['Tmpfs']['/scratch'].replace('noexec,', '')
            with self.assertRaisesRegex(ValueError, 'fixed_tmpfs_arguments'):
                host.validate_launch(record, altered)

    def test_fake_tmpfs_rows_extra_mounts_aliases_and_bind_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _, args, record = self.fixture(temp)
            for extra in ({'Type': 'tmpfs', 'Destination': '/tmp'}, {'Type': 'volume', 'Destination': '/extra'},
                          {'Type': 'bind', 'Destination': '/input', 'Source': '/private/audio', 'RW': False}):
                altered = deepcopy(record); altered['Mounts'].append(extra)
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    host.validate_launch(altered, args)
            for key, value in (('RW', True), ('Source', '/private/audio'), ('Destination', '/code/'),
                               ('Propagation', 'shared')):
                altered = deepcopy(record); altered['Mounts'][0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    host.validate_launch(altered, args)
            altered = list(args); index = altered.index(ci.IMAGE)
            altered[index:index] = ['--tmpfs', '/tmp:rw']
            with self.assertRaisesRegex(ValueError, 'unique_tmpfs_destination'):
                host.validate_launch(record, altered)

    def test_old_security_boundaries_are_still_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            _, args, record = self.fixture(temp)
            for key, value in (('NetworkMode', 'bridge'), ('Privileged', True), ('Memory', 0),
                    ('MemorySwap', 0), ('NanoCpus', 0), ('PidsLimit', 0), ('ReadonlyRootfs', False),
                    ('CapAdd', ['ALL']), ('CapDrop', []), ('SecurityOpt', []), ('CgroupnsMode', 'host'),
                    ('PidMode', 'host'), ('IpcMode', 'host'), ('LogConfig', {}), ('Binds', ['/private:/input']),
                    ('Devices', [{}]), ('DeviceRequests', [{}]), ('ExtraHosts', ['secret'])):
                altered = deepcopy(record); altered['HostConfig'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    host.validate_launch(altered, args)
            for key, value in (('User', '0:0'), ('Env', ['GITHUB_TOKEN=secret']), ('Cmd', ['decode']),
                               ('Entrypoint', ['sh'])):
                altered = deepcopy(record); altered['Config'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    host.validate_launch(altered, args)

    def test_forbidden_commands_denied_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            docker, args, _ = self.fixture(temp)
            docker.cid = 'a' * 64
            denied = [('start', docker.name), ('run', ci.IMAGE), ('exec', docker.cid, 'true'),
                ('container', 'start', docker.cid), ('container', 'run', ci.IMAGE),
                ('container', 'exec', docker.cid, 'true'), ('restart', docker.cid), ('rm', docker.name),
                ('container', 'rm', docker.name), ('container', 'rm', 'b' * 64),
                ('pull', 'some/model:latest'), ('system', 'prune', '-f')]
            with patch.object(ci.subprocess, 'run', side_effect=AssertionError('subprocess reached')):
                for command in denied:
                    with self.subTest(command=command), self.assertRaises(ValueError):
                        docker(*command)
                altered = list(args); altered[altered.index('--network') + 1] = 'bridge'
                with self.assertRaisesRegex(ValueError, 'ci_exact_single_create'):
                    docker(*altered)

    def exercise(self, mock):
        with tempfile.TemporaryDirectory() as temp, patch.object(ci.subprocess, 'run', side_effect=mock), \
                patch.object(host.c, 'release_first', side_effect=AssertionError('CI release used')):
            return ci.run(Path(temp))

    def test_success_mock_versions_inspect_cleanup_and_no_inherited_tokens(self):
        mock = DockerMock()
        with patch.dict(ci.os.environ, {'GITHUB_TOKEN': 'PRIVATE_SENTINEL', 'HF_TOKEN': 'PRIVATE_SENTINEL',
                'DOCKER_HOST': 'tcp://untrusted', 'DOCKER_AUTH_CONFIG': 'PRIVATE_SENTINEL'}):
            result = self.exercise(mock)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['launch_validator_passed'])
        self.assertEqual(result['real_docker_versions']['Server']['Version'], '28.0.4')
        self.assertEqual(result['cleanup']['status'], 'removed_and_absent')
        self.assertFalse(result['cleanup']['removed_by_name'])
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(result))
        self.assertEqual([r[:2] if r[0] == 'container' else r[:1] for r in mock.calls],
            [('version',), ('pull',), ('create',), ('container', 'inspect'),
             ('container', 'inspect'), ('container', 'rm'), ('container', 'ls')])
        self.assertEqual(result['container_start_calls'], 0)
        self.assertEqual(result['model_loads'], 0)

    def test_validation_failure_keeps_actual_predicate_and_still_removes_only_owned_cid(self):
        mock = DockerMock(mutate=lambda record: record['HostConfig']['Tmpfs'].pop('/tmp'))
        result = self.exercise(mock)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['primary_error']['predicate'], 'fixed_tmpfs_roster')
        self.assertNotIn('/tmp', result['inspect']['HostConfig']['Tmpfs'])
        self.assertEqual(result['cleanup']['status'], 'removed_and_absent')
        self.assertIsNone(result['cleanup_error'])

    def test_name_collision_never_removes_by_name_or_unowned_id(self):
        mock = DockerMock(create_error=ValueError('ci_exact_single_create'), no_cid=True)
        result = self.exercise(mock)
        self.assertEqual(result['primary_error']['predicate'], 'ci_exact_single_create')
        self.assertEqual(result['cleanup']['status'], 'create_outcome_unresolved_no_owned_cid')
        self.assertFalse(any(row[:2] == ('container', 'rm') for row in mock.calls))
        for field in ('Name', 'Id', 'Labels'):
            def mutate(record):
                if field == 'Labels': record['Config']['Labels'] = {ci.LABEL: 'someone_else'}
                else: record[field] = 'someone_else'
            mock = DockerMock(mutate=mutate)
            result = self.exercise(mock)
            self.assertEqual(result['primary_error']['predicate'], 'ci_container_ownership')
            self.assertFalse(any(row[:2] == ('container', 'rm') for row in mock.calls))

    def test_uncertain_create_recovers_only_owned_cid_and_preserves_primary(self):
        mock = DockerMock(create_error=ValueError('ci_create_output_cid_match'))
        result = self.exercise(mock)
        self.assertEqual(result['primary_error']['predicate'], 'ci_create_output_cid_match')
        self.assertEqual(result['cleanup']['status'], 'removed_and_absent')

    def test_cleanup_failure_never_overwrites_primary_and_absence_is_required(self):
        mock = DockerMock(mutate=lambda record: record['HostConfig']['Tmpfs'].clear(), cleanup_error=True)
        result = self.exercise(mock)
        self.assertEqual(result['primary_error']['predicate'], 'fixed_tmpfs_roster')
        self.assertEqual(result['cleanup_error']['predicate'], 'ci_removed_exact_cid')
        result = self.exercise(DockerMock(listed=b'a' * 64 + b'\n'))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['cleanup_error']['predicate'], 'ci_owned_cid_absence_proof')

    def test_started_container_is_not_claimed_or_deleted(self):
        mock = DockerMock(mutate=lambda record: record['State'].update(Status='running', Running=True, Pid=100))
        result = self.exercise(mock)
        self.assertEqual(result['primary_error']['predicate'], 'ci_container_never_started')
        self.assertFalse(any(row[:2] == ('container', 'rm') for row in mock.calls))

    def test_nonempty_or_aliased_fixture_rejected_before_create(self):
        with tempfile.TemporaryDirectory() as temp:
            docker, args, _ = self.fixture(temp)
            (docker.root / 'runtime' / 'model.bin').write_bytes(b'not a model')
            with patch.object(ci.subprocess, 'run', side_effect=AssertionError('subprocess reached')):
                with self.assertRaisesRegex(ValueError, 'ci_empty_metadata_only_fixture'):
                    docker(*args)

    def test_unrecognized_environment_is_redacted_and_error_text_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            docker, args, record = self.fixture(temp)
            record['Config']['Env'].append('GITHUB_TOKEN=PRIVATE_SENTINEL')
            output = ci.inspect_projection(record, args, docker.root)
            self.assertNotIn('PRIVATE_SENTINEL', json.dumps(output))
        self.assertEqual(ci.failure(ValueError('private /home/path TOKEN'))['predicate'], 'unclassified_error')

    def test_failure_fixed_allowlist_no_secret_shaped_text_or_subclass_hooks(self):
        self.assertEqual(ci.failure(ValueError('private_secret_snake_case_token')),
            {'type': 'ValueError', 'predicate': 'unclassified_error'})
        self.assertEqual(ci.failure(ValueError('fixed_tmpfs_options')),
            {'type': 'ValueError', 'predicate': 'fixed_tmpfs_options'})
        touched = []
        class Malicious(ValueError):
            @property
            def args(self):
                touched.append('args')
                raise AssertionError('exception property touched')
            def __str__(self):
                touched.append('str')
                raise AssertionError('exception rendered')
            def __repr__(self):
                touched.append('repr')
                raise AssertionError('exception rendered')
        self.assertEqual(ci.failure(Malicious('fixed_tmpfs_options')),
            {'type': 'Exception', 'predicate': 'unclassified_error'})
        self.assertEqual(touched, [])


class PrepareObservationTests(unittest.TestCase):
    def simulate(self, failed_family, *, broken_observer=False):
        import prepare_run as runner
        import prepare_observer as observation
        import time
        identity = {'repository': 'jiying2007/kws-pipeline', 'run_id': 12345, 'run_attempt': 1,
            'job_key': 'prepare_diagnostic', 'executed_head_sha': 'c' * 40}
        release = {**host.c.FIXED, **host.c.BINDING_PINS, 'enabled': True,
            'independent_source_review': True, 'source_closure_ready': True,
            'prior_prepare_history_sha256': host.c.digest(b'[]')}
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(out), \
                patch.object(ci.subprocess, 'run', side_effect=AssertionError('external process forbidden')):
            root = Path(temp)
            for name in ('release', 'code', 'metadata', 'authority', 'output', 'runtime'):
                (root / name).mkdir()
            (root / 'release' / 'prepare-release.json').write_bytes(host.c.canonical(release))
            packet_sha = 'd' * 64
            (root / 'authority' / 'prepare-claim.json').write_bytes(host.c.canonical({'metadata_packet_sha256': packet_sha}))
            for family in host.c.FAMILIES:
                (root / 'runtime' / family).mkdir()
            observer = observation.Observation('host')
            observer.attach(identity, temp)
            observer.stage('setup')
            receipt = runner.host_receipt(identity, 'a' * 64, 'e' * 40, release)
            receipt['stages']['setup'] = 'complete'
            class SyntheticDocker:
                def __init__(self):
                    self.root = root
                    self.deadline = time.monotonic() + 10
                    self.names = {}
                    self.supervisions = []
                    self.calls = []
                    self.created = []
                    self.started = []
                    self.removed = []
                    self.records = {}
                def gate(self, name):
                    folder = root / 'gates' / name
                    folder.mkdir(parents=True)
                    return folder
                def __call__(self, *args):
                    operation = args[0]
                    self.calls.append(operation)
                    if operation == 'create':
                        name = args[args.index('--name') + 1]
                        self.created.append(name)
                        self.names[name] = {'create_attempted': True}
                        self.records[name] = inspect_fixture(list(args))
                        if name.endswith('-' + failed_family):
                            self.records[name]['HostConfig']['Tmpfs'].pop('/tmp')
                        return 'a' * 64
                    if operation == 'inspect':
                        return json.dumps([self.records[args[1]]])
                    if operation == 'start':
                        name = args[1]
                        if name.endswith('-' + failed_family):
                            raise AssertionError('failed worker must never start')
                        self.started.append(name)
                        family = name.rsplit('-', 1)[1]
                        value = {'schema': 'k1-prepare-worker-result-v1', 'identity': identity,
                            'family': family, 'status': 'prepared_configuration_only',
                            'model_loads_started': 1, 'model_loads_completed': 1, 'maximum_new_asr_calls': 0,
                            'native_decode_attempts': 0, 'audio_forward_calls': 0, 'audio_mounted': False,
                            'recognition_result': False, 'legacy_execution_contract_grants_asr': False,
                            'metadata_packet_sha256': packet_sha, 'source_freeze_sha256': 'a' * 64,
                            'guard': {'schema': 'k1-prepare-forward-guard-v1', 'forbidden_attempts': 0,
                                'python_entry_guard': 'active_through_constructor_and_config',
                                'coverage': 'reviewed_fixed_python_native_and_module_forward_entrypoints',
                                'universal_native_extension_proof': False}, 'wall_seconds': 0.001}
                        for filename in ('prepare-receipt.json', 'worker-complete.json'):
                            (root / 'output' / family / filename).write_bytes(host.c.canonical(value))
                        return ''
                    raise AssertionError('unexpected synthetic operation')
                def completed(self, name):
                    return True
                def remove(self, name):
                    self.removed.append(name)
                def gate_result(self, name):
                    return None
            docker = SyntheticDocker()
            setup = {'image': ci.IMAGE, 'code': root / 'code',
                'runtimes': {family: root / 'runtime' / family for family in host.c.FAMILIES}}
            original = None
            with patch.object(host.c, 'HERE', root / 'release'), \
                    (patch.object(observer, 'stage', side_effect=OSError(5, 'PRIVATE_SENTINEL'))
                     if broken_observer else nullcontext()):
                try:
                    host.run_prepares(root, setup, docker, identity, 'a' * 64, 'b' * 64, 0.0, observer=observer)
                except BaseException as error:
                    original = error
                    runner.failure(observer, error)
            self.assertIs(type(original), ValueError)
            self.assertEqual(BaseException.args.__get__(original), ('fixed_tmpfs_roster',))
            self.assertIs(docker.supervision_failure[0], original)
            self.assertEqual(observer.identity, identity)
            self.assertEqual(docker.supervisions[-1]['primary_error']['error_type'], 'ValueError')
            self.assertNotIn('fixed_tmpfs_roster', json.dumps(docker.supervisions[-1]['primary_error']))
            self.assertIsNone(docker.supervisions[-1]['cleanup_error'])
            self.assertFalse(docker.supervisions[-1]['worker_exit_success'])
            self.assertIsNone(docker.supervisions[-1]['gate_result'])
            for family in host.c.FAMILIES:
                if any(name.endswith('-' + family) for name in docker.started):
                    receipt['stages'][family + '_prepare'] = 'complete'
                elif any(name.endswith('-' + family) for name in docker.created):
                    receipt['stages'][family + '_prepare'] = 'failed_no_retry'
            observer.finish(False)
            rows = [json.loads(line) for line in out.getvalue().splitlines()]
            envelope = next(row for row in rows if row.get('schema') == 'k1-prepare-independent-diagnostic-envelope-v1')
            self.assertEqual(envelope['record']['identity'], identity)
            self.assertIsNone(envelope['record']['consumed_asr_attempts'])
            self.assertFalse(envelope['record']['retry_authorized'])
            self.assertNotIn('PRIVATE_SENTINEL', out.getvalue())
            return receipt, docker, envelope, rows

    def test_qwen_launch_failure_after_setup_is_identity_bound_and_never_started(self):
        receipt, docker, envelope, rows = self.simulate('qwen06')
        self.assertEqual(receipt['stages'], {'setup': 'complete', 'qwen06_prepare': 'failed_no_retry',
            'sensevoice_prepare': 'not_run', 'whisper_prepare': 'not_run'})
        self.assertEqual(docker.calls, ['create', 'inspect'])
        self.assertEqual(docker.started, [])
        self.assertEqual(docker.removed, docker.created)
        error = envelope['record']['errors'][0]
        self.assertEqual(error['stage'], 'prepare_qwen06')
        self.assertEqual(error['detail']['validation_code'], 'prepare_launch_tmpfs_roster')
        self.assertEqual([row['stage'] for row in rows if row.get('schema') == 'k1-prepare-safe-checkpoint-v1'],
            ['setup', 'prepare_qwen06'])

    def test_later_family_launch_failures_keep_correct_stage_and_stop_sequence(self):
        for family in ('sensevoice', 'whisper'):
            with self.subTest(family=family):
                receipt, docker, envelope, rows = self.simulate(family)
                index = host.c.FAMILIES.index(family)
                self.assertEqual(len(docker.created), index + 1)
                self.assertEqual(len(docker.started), index)
                self.assertEqual(docker.removed, docker.created)
                self.assertEqual(envelope['record']['errors'][0]['stage'], 'prepare_' + family)
                self.assertEqual(envelope['record']['errors'][0]['detail']['validation_code'], 'prepare_launch_tmpfs_roster')
                for after in host.c.FAMILIES[index + 1:]:
                    self.assertEqual(receipt['stages'][after + '_prepare'], 'not_run')

    def test_broken_stage_observer_cannot_replace_launch_failure_or_cleanup(self):
        _, docker, envelope, _ = self.simulate('qwen06', broken_observer=True)
        self.assertEqual(docker.started, [])
        self.assertEqual(docker.removed, docker.created)
        self.assertEqual(envelope['record']['errors'][0]['detail']['validation_code'], 'prepare_launch_tmpfs_roster')

    def test_production_passes_same_observer_and_all_launch_predicates_are_fixed_codes(self):
        import failure_observer
        import diagnostic_transport
        tree = ast.parse((HERE / 'prepare_run.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and node.func.attr == 'run_prepares']
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(item.arg == 'observer' and isinstance(item.value, ast.Name) and
            item.value.id == 'observer' for item in calls[0].keywords))
        source = ast.parse((HERE / 'prepare_host.py').read_text())
        validator = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_launch')
        literals = {node.args[1].value for node in ast.walk(validator) if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and node.func.attr == 'require' and isinstance(node.args[1], ast.Constant)}
        mapped = {message for message, code in failure_observer.VALIDATION_CODES.items() if code.startswith('prepare_launch_')}
        self.assertEqual(mapped, literals)
        for message in mapped:
            result = failure_observer.error_metadata(ValueError(message))
            diagnostic_transport._public_failure(result)
        for message in ('private_secret_snake_case_token', 'fixed_tmpfs_roster PRIVATE_SENTINEL'):
            result = failure_observer.error_metadata(ValueError(message))
            self.assertNotIn('validation_code', result)
        with self.assertRaises(ValueError):
            diagnostic_transport._public_failure({'error_type': 'ValueError',
                'error_text': 'omitted_by_fixed_privacy_policy', 'validation_code': 'prepare_launch_private_secret'})


if __name__ == '__main__':
    unittest.main()
