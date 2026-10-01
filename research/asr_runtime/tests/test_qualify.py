#!/usr/bin/env python3
"""Synthetic only: no network, Docker, builds, installs or third-party imports."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qualify as q
import controller as c


class FakeResponse(io.BytesIO):
    status = 200
    def __init__(self, body, item):
        super().__init__(body)
        self.headers = {'Content-Length': str(item['bytes'])}
        self.url = item['url']
    def geturl(self):
        return self.url


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.admission = q.load_json(ROOT / 'locks/admission.json')
        self.limits = self.admission['limits']
        self.body = b'synthetic bytes'
        self.item = {'name': 'fixture', 'version': '1', 'filename': 'fixture-1-py3-none-any.whl',
                     'bytes': len(self.body), 'size_bytes': len(self.body), 'sha256': hashlib.sha256(self.body).hexdigest(),
                     'url': 'https://files.pythonhosted.org/packages/ab/fixture-1-py3-none-any.whl'}
    def tearDown(self):
        self.temp.cleanup()
    def test_candidate_is_explicitly_not_execution(self):
        self.assertIs(type(q.candidate_status()['execution_enabled']), bool)
        with mock.patch.object(q, 'load_json', return_value={'execution_enabled':False}):
            with self.assertRaises(q.GateError):
                c.validate_admission(ROOT)
    def test_inventory_exact(self):
        self.assertEqual(q.validate_inventory({'files': [self.item]}, 1, len(self.body)), [self.item])
        for key, value in [('url', self.item['url'].replace('files.pythonhosted.org', 'evil.invalid')),
                           ('sha256', '0' * 63), ('filename', '../evil.whl'), ('size_bytes', True)]:
            item = dict(self.item, **{key: value})
            with self.subTest(key=key), self.assertRaises(q.GateError):
                q.validate_inventory({'files': [item]}, 1, len(self.body))
        with self.assertRaises(q.GateError):
            q.validate_inventory({'files': [self.item, self.item]}, 2, len(self.body) * 2)
    def test_download_strict_success(self):
        opener = mock.Mock()
        opener.open.return_value = FakeResponse(self.body, self.item)
        p = self.path / self.item['filename']
        self.assertEqual(q.download_one(self.item, p, opener), len(self.body))
        self.assertEqual(p.read_bytes(), self.body)
    def test_download_oversize_and_wrong_hash_fail_closed(self):
        for i, body in enumerate((self.body + b'x', self.body[:-1], b'x' * len(self.body))):
            opener = mock.Mock()
            opener.open.return_value = FakeResponse(body, self.item)
            p = self.path / (str(i) + '.whl')
            with self.assertRaises(q.GateError):
                q.download_one(self.item, p, opener)
            self.assertFalse(p.exists())
    def test_redirect_rejected(self):
        with self.assertRaises(q.GateError):
            q.NoRedirect().redirect_request(None, None, 302, None, None, 'https://other.invalid')
    def make_zip(self, entries):
        p = self.path / 'sample.zip'
        with zipfile.ZipFile(p, 'w') as z:
            for n, v in entries:
                z.writestr(n, v)
        return p
    def test_archive_actual_expansion(self):
        p = self.make_zip([('fixture/a.py', b'abc'), ('fixture/b.txt', b'1234')])
        r = q.inspect_archive(p, 7, 2)
        self.assertEqual(r['expanded_bytes'], 7)
        self.assertEqual(r['members'], 2)
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 6, 2)
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 7, 1)
    def test_archive_rejects_traversal_and_duplicate(self):
        for entries in ([('../evil', b'a')], [('a', b'a'), ('a', b'b')]):
            p = self.make_zip(entries)
            with self.assertRaises(q.GateError):
                q.inspect_archive(p, 100, 10)
    def test_archive_rejects_symlink(self):
        p = self.path / 'link.zip'
        with zipfile.ZipFile(p, 'w') as z:
            entry = zipfile.ZipInfo('link')
            entry.create_system = 3
            entry.external_attr = 0o120777 << 16
            z.writestr(entry, '../elsewhere')
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 100, 10)
    def test_pth_requires_exact_review(self):
        body = b'import reviewed_test_only\n'
        p = self.make_zip([('x.pth', body)])
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 100, 10)
        q.inspect_archive(p, 100, 10, {'x.pth': hashlib.sha256(body).hexdigest()})
    def test_tar_links_rejected(self):
        p = self.path / 'source.tar.gz'
        with tarfile.open(p, 'w:gz') as tar:
            entry = tarfile.TarInfo('source/link')
            entry.type = tarfile.SYMTYPE
            entry.linkname = '/etc/passwd'
            tar.addfile(entry)
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 100, 10)
    def test_wheel_record_covers_payload(self):
        import base64
        p = self.path / 'fixture-1-py3-none-any.whl'
        body = b'hello'
        encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b'=').decode()
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('fixture.py', body)
            z.writestr('fixture-1.dist-info/RECORD', 'fixture.py,sha256=' + encoded + ',5\nfixture-1.dist-info/RECORD,,\n')
        q.inspect_archive(p, 1000, 10)
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr('fixture.py', body)
            z.writestr('fixture-1.dist-info/RECORD', 'fixture-1.dist-info/RECORD,,\n')
        with self.assertRaises(q.GateError):
            q.inspect_archive(p, 1000, 10)

    def test_create_flags_and_sanitized_environment(self):
        command = q.create_command('/usr/bin/docker', 'docker.io/library/python@sha256:' + 'a'*64,
                                   'kws-asr-fixture', '/inputs-fixture', '/output-fixture', '/work-fixture', self.limits, 'runtime')
        for flag in ('--pull=never', '--network=none', '--read-only', '--user=1000:1000', '--cap-drop=ALL',
                     '--security-opt=no-new-privileges', '--cgroupns=private', '--ipc=private'):
            self.assertIn(flag, command)
        self.assertNotIn('--privileged', command)
        self.assertFalse(any('docker.sock' in x and x.startswith('--mount=') for x in command))
        self.assertEqual(q.clean_env()['HOME'], '/nonexistent')
        self.assertNotIn('GITHUB_TOKEN', q.clean_env())
        for value in ('USER=kws-asr-runtime', 'LOGNAME=kws-asr-runtime', 'HOME=/work/home',
                      'XDG_CACHE_HOME=/work/cache', 'TORCHINDUCTOR_CACHE_DIR=/work/cache/torchinductor'):
            self.assertIn(value, command)

    def identity_environment(self, work):
        return {'USER': 'kws-asr-runtime', 'LOGNAME': 'kws-asr-runtime', 'HOME': str(work / 'home'),
                'XDG_CACHE_HOME': str(work / 'cache'),
                'TORCHINDUCTOR_CACHE_DIR': str(work / 'cache/torchinductor')}

    def test_container_identity_does_not_inherit_host_environment(self):
        with mock.patch.dict(os.environ, {'USER': 'host-user', 'LOGNAME': 'host-login',
                                         'HOME': '/host/home', 'TORCHINDUCTOR_CACHE_DIR': '/host/cache'}):
            command = q.create_command('/usr/bin/docker', 'docker.io/library/python@sha256:' + 'a'*64,
                                       'kws-asr-fixture', '/inputs-fixture', '/output-fixture', '/work-fixture', self.limits, 'runtime')
        assignments = command[command.index('-i') + 1:command.index('/usr/local/bin/python')]
        env = dict(item.split('=', 1) for item in assignments)
        self.assertEqual(len(env), len(assignments), 'duplicate environment assignment')
        for key, value in self.identity_environment(Path('/work')).items():
            self.assertEqual(env[key], value)
        self.assertFalse(any('/host/' in item or 'host-user' in item or 'host-login' in item for item in assignments))

    def test_runtime_identity_supports_numeric_uid_without_passwd_entry(self):
        import container_stage
        import pwd
        work = self.path / 'work'
        work.mkdir()
        with mock.patch.dict(os.environ, self.identity_environment(work), clear=True), \
                mock.patch.object(os, 'getuid', return_value=1000), \
                mock.patch.object(os, 'getgid', return_value=1000), \
                mock.patch.object(os, 'geteuid', return_value=1000), \
                mock.patch.object(os, 'getegid', return_value=1000), \
                mock.patch.object(pwd, 'getpwuid', side_effect=KeyError('uid not found: 1000')) as passwd:
            result = container_stage.runtime_identity(work)
            passwd.assert_not_called()
        self.assertEqual({key: result[key] for key in ('uid', 'gid', 'euid', 'egid')},
                         dict(uid=1000, gid=1000, euid=1000, egid=1000))
        self.assertEqual(result['process_username_source'], 'fixed_environment_not_passwd')
        for name in ('home', 'cache', 'cache/torchinductor'):
            self.assertTrue((work / name).is_dir())
            self.assertEqual(list((work / name).glob('.kws-write-probe')), [])

    def test_runtime_identity_rejects_wrong_effective_identity(self):
        import container_stage
        for identity in ('getuid', 'getgid', 'geteuid', 'getegid'):
            with self.subTest(identity=identity), \
                    mock.patch.object(os, 'getuid', return_value=1000), \
                    mock.patch.object(os, 'getgid', return_value=1000), \
                    mock.patch.object(os, 'geteuid', return_value=1000), \
                    mock.patch.object(os, 'getegid', return_value=1000), \
                    mock.patch.object(os, identity, return_value=0):
                with self.assertRaisesRegex(RuntimeError, 'real/effective UID/GID'):
                    container_stage.runtime_identity(self.path)

    def test_runtime_identity_rejects_missing_or_redirected_environment(self):
        import container_stage
        expected = self.identity_environment(self.path)
        for name in expected:
            for value in (None, '/outside' if 'HOME' in name or name.endswith('_DIR') else 'root'):
                env = dict(expected)
                env.pop(name) if value is None else env.update({name: value})
                with self.subTest(name=name, value=value), mock.patch.dict(os.environ, env, clear=True), \
                        mock.patch.object(os, 'getuid', return_value=1000), \
                        mock.patch.object(os, 'getgid', return_value=1000), \
                        mock.patch.object(os, 'geteuid', return_value=1000), \
                        mock.patch.object(os, 'getegid', return_value=1000):
                    with self.assertRaisesRegex(RuntimeError, 'environment mismatch'):
                        container_stage.runtime_identity(self.path)

    def test_runtime_identity_rejects_linked_home_or_cache(self):
        import container_stage
        for name in ('home', 'cache', 'cache/torchinductor'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / 'work'
                work.mkdir()
                target = work / name
                target.parent.mkdir(exist_ok=True)
                target.symlink_to(self.path)
                with mock.patch.dict(os.environ, self.identity_environment(work), clear=True), \
                        mock.patch.object(os, 'getuid', return_value=1000), \
                        mock.patch.object(os, 'getgid', return_value=1000), \
                        mock.patch.object(os, 'geteuid', return_value=1000), \
                        mock.patch.object(os, 'getegid', return_value=1000):
                    with self.assertRaisesRegex(RuntimeError, 'linked runtime'):
                        container_stage.runtime_identity(work)

    def test_exit137_does_not_imply_oom(self):
        r = q.summarize_state({'State': {'ExitCode': 137, 'OOMKilled': False, 'Running': False, 'Dead': False}})
        self.assertEqual(r['status'], 'failed')
        self.assertFalse(r['oom_killed'])
    def test_timeout_is_failure_even_exit_zero(self):
        r = q.summarize_state({'State': {'ExitCode': 0, 'OOMKilled': False, 'Running': False}}, 'wall_timeout')
        self.assertEqual(r['status'], 'failed')
    def test_bounded_command_retains_tail(self):
        r = q.bounded_command([sys.executable, '-I', '-S', '-c', "print('a'*10000); print('critical-error-tail')"], maximum=128)
        self.assertTrue(r['truncated'])
        self.assertIn('critical-error-tail', r['output'])
        self.assertEqual(r['returncode'], 0)
    def test_bounded_command_kills_descendant_pipe(self):
        import time
        t = time.monotonic()
        r = q.bounded_command([sys.executable, '-I', '-S', '-c',
                               "import os,time; p=os.fork(); time.sleep(30) if p==0 else None"], timeout=.15)
        self.assertLess(time.monotonic() - t, 3)
        self.assertTrue(r['timeout'])
    def test_staging_rejects_links_and_sockets(self):
        (self.path / 'link').symlink_to('/tmp')
        with self.assertRaises(q.GateError):
            q.regular_tree(self.path)
    def test_source_extract_uses_fixed_safe_directory(self):
        import container_stage
        archive = self.path / 'fixture-1.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            entry = tarfile.TarInfo('fixture-1/module.py')
            entry.size = 3
            tar.addfile(entry, io.BytesIO(b'abc'))
        audit = q.inspect_archive(archive, 1000, 10)
        project = {'source_root':'fixture-1','archive_members':1,'source_payload_manifest_sha256':audit['payload_sha256']}
        target = container_stage.extract_source(archive, project, self.path / 'sources')
        self.assertEqual(target, self.path / 'sources/fixture-1')
        self.assertEqual((target / 'module.py').read_bytes(), b'abc')
        self.assertEqual(int((target / 'module.py').stat().st_mtime), 1704067200)
        with tarfile.open(archive, 'w:gz') as tar:
            entry = tarfile.TarInfo('../outside')
            entry.size = 3
            tar.addfile(entry, io.BytesIO(b'abc'))
        with self.assertRaises(RuntimeError):
            container_stage.extract_source(archive, project, self.path / 'unsafe')
        self.assertFalse((self.path / 'outside').exists())

    def test_work_venv_links_are_counted_without_following(self):
        work = self.path / 'work'
        (work / 'bin').mkdir(parents=True)
        (work / 'lib').mkdir()
        (work / 'lib' / 'payload').write_bytes(b'abc')
        (work / 'bin' / 'python').symlink_to('/usr/local/bin/python')
        (work / 'lib64').symlink_to('lib')
        expected = 3 + len('/usr/local/bin/python') + len('lib')
        self.assertEqual(c.work_size(work), expected)
        with self.assertRaises(q.GateError):
            c.folder_size(work)

    def test_receipt_byte_cap(self):
        with self.assertRaises(q.GateError):
            q.write_receipt(self.path / 'receipt.json', {'too_large': 'x' * 65536})
        self.assertFalse((self.path / 'receipt.json').exists())
    def fake_stage(self, fault=None):
        inputs = self.path / 'inputs'
        inputs.mkdir()
        (inputs / 'fixture').write_text('immutable')
        (inputs / 'runtime-versions.json').write_text(json.dumps({'fixture': '1'}, sort_keys=True))
        output = self.path / 'runtime'
        limits = dict(self.limits)
        if fault == 'output_cap':
            limits['output_bytes'] = 1
        cid, image_id = 'b' * 64, 'sha256:' + 'c' * 64
        args = q.create_command('/usr/bin/docker', 'docker.io/library/python@sha256:' + 'a'*64,
                                'kws-asr-fixture', inputs, output, self.path / 'runtime-work', limits, 'runtime')
        cfg = {'Image': image_id, 'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False},
               'HostConfig': {'NetworkMode': 'none', 'ReadonlyRootfs': True, 'Privileged': False,
               'CapDrop': ['ALL'], 'CapAdd': None, 'SecurityOpt': ['no-new-privileges'],
               'Memory': limits['container_memory_bytes'], 'MemorySwap': limits['container_memory_bytes'],
               'NanoCpus': limits['container_cpus'] * 10**9, 'PidsLimit': limits['container_pids'],
               'CgroupnsMode': 'private', 'IpcMode': 'private', 'PidMode': '', 'Devices': [], 'DeviceRequests': [], 'Init': True,
               'Tmpfs': {'/tmp': next(a.split(':', 1)[1] for a in args if a.startswith('--tmpfs='))}},
               'Config': {'User': '1000:1000', 'Entrypoint': ['/usr/bin/env'],
                          'Cmd': args[args.index('docker.io/library/python@sha256:' + 'a'*64)+1:],
                          'Labels': {'com.kws.asr-runtime.owner': 'kws-asr-fixture'}},
               'Mounts': [{'Type':'bind','Destination': dst, 'Source': str(src), 'RW': rw, 'Propagation': 'rprivate'} for src, dst, rw in
                          [(inputs, '/inputs', False), (output, '/output', True), (self.path/'runtime-work', '/work', True)]]}
        if fault == 'wrong_owner':
            cfg['Config']['Labels']['com.kws.asr-runtime.owner'] = 'different-owner'
        calls = []
        inspect_count = 0
        def fake(command, **kwargs):
            nonlocal inspect_count
            calls.append(command)
            result = {'returncode': 0, 'timeout': False, 'truncated': False, 'bytes': 0, 'output': ''}
            operation = command[2]
            if operation == 'create':
                result['output'] = 'unverified' if fault == 'unknown_create' else cid
            if operation == 'inspect':
                inspect_count += 1
                if fault == 'inspect' or (fault == 'late_inspect' and inspect_count > 1):
                    result.update(returncode=1, output='injected inspect failure')
                else:
                    result['output'] = json.dumps([cfg])
            if operation == 'start':
                cpu = {'schema':'kws.asr-runtime-cpu-probe.v1', 'device':'cpu', 'float32_matmul':True,
                       'bf16_matmul_layer_norm_conv1d':True, 'target_speech_model_weights_loaded':0, 'user_audio_loaded':0, 'installed_versions':{'fixture':'1'}}
                (output / 'cpu-probe.json').write_text(json.dumps(cpu))
                inner = {'schema':'kws.asr-runtime-container-stage.v1', 'stage':'runtime', 'target_speech_model_weights_loaded':0, 'user_audio_loaded':0,
                         'input_contracts_sha256':{'runtime-versions.json':q.sha256_file(inputs / 'runtime-versions.json')},
                         'cpu_probe_completed':True, 'cpu_probe_sha256':q.sha256_file(output / 'cpu-probe.json')}
                if fault == 'partial_receipt':
                    inner = {'stage':'runtime'}
                (output / 'stage-receipt.json').write_text(json.dumps(inner))
                if fault == 'input_mutation':
                    (inputs / 'fixture').write_text('changed')
            if operation == 'rm' and fault == 'remove':
                result.update(returncode=1, output='injected remove failure')
            return result
        record = c.stage('/usr/bin/docker', 'docker.io/library/python@sha256:' + 'a'*64,
                         image_id, 'kws-asr-fixture', inputs, output, limits, 'runtime', run=fake)
        return record, calls

    def test_lifecycle_pass_writes_terminal_cleanup_receipt(self):
        record, calls = self.fake_stage()
        self.assertEqual(record['status'], 'pass')
        self.assertTrue(record['container_removed'])
        final = json.loads((self.path / 'runtime.final.json').read_text())
        self.assertTrue(final['container_removed'])
        self.assertTrue(any(x[2] == 'rm' for x in calls))

    def test_lifecycle_failures_are_durable_and_never_green(self):
        for fault in ('inspect', 'remove', 'input_mutation', 'output_cap', 'unknown_create', 'wrong_owner', 'partial_receipt', 'late_inspect'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as p:
                saved, self.path = self.path, Path(p)
                try:
                    record, calls = self.fake_stage(fault)
                    self.assertEqual(record['status'], 'failed')
                    final = json.loads((self.path / 'runtime.final.json').read_text())
                    self.assertEqual(final['status'], 'failed')
                    if fault == 'late_inspect':
                        self.assertTrue(all(any(x[2] == action for x in calls) for action in ('stop','kill','rm')))
                        self.assertTrue(final['terminal_state_unknown'])
                    if fault in ('input_mutation', 'output_cap', 'partial_receipt'):
                        self.assertTrue(any(x[2] == 'rm' for x in calls))
                    if fault == 'wrong_owner':
                        self.assertFalse(final['ownership_verified'])
                        self.assertFalse(any(x[2] in ('start', 'stop', 'kill', 'rm') for x in calls))
                    if fault == 'unknown_create':
                        self.assertTrue(final['create_outcome_unknown'])
                        self.assertFalse(final['retry_permitted'])
                        self.assertEqual(len(calls), 1)
                finally:
                    self.path = saved

    def test_one_time_event_binds_exact_source_and_budget(self):
        import admit_run
        a = copy.deepcopy(self.admission)
        a['execution_enabled'] = True
        head, source = 'a' * 40, 'b' * 64
        approval = {'armed':True, 'pr_number':999, 'base_sha':a['baseline_sha'], 'nonce':a['activation']['nonce'], 'head_sha':head, 'source_sha256':source, 'dependency_bytes':a['expected_input_bytes'],
                    'image_compressed_bytes':a['image']['compressed_bytes'],
                    'total_input_bytes':a['expected_input_bytes']+a['image']['compressed_bytes'],
                    'scope':'single_no_model_dependency_cpu_qualification'}
        event = {'action':'labeled','label':{'name':a['activation']['label']},'number':999,'repository':{'full_name':admit_run.REPOSITORY,'private':False},
                 'pull_request':{'number':999,'head':{'repo':{'full_name':admit_run.REPOSITORY},'ref':admit_run.BRANCH,'sha':head},
                                 'base':{'sha':a['baseline_sha']},'user':{'login':'jiying2007'},
                                 'body':admit_run.PREFIX+json.dumps(approval)}}
        env = {'GITHUB_EVENT_NAME':'pull_request','GITHUB_RUN_ATTEMPT':'1','GITHUB_ACTOR':'jiying2007'}
        args = [event,env,a,source,head,['research/asr_runtime/qualify.py'],a['baseline_sha']]
        self.assertEqual(admit_run.validate_event(*args), approval)
        for field, value in [('action','synchronize'),('action','reopened'),('action','opened')]:
            changed = copy.deepcopy(event);changed[field]=value
            with self.assertRaises(q.GateError):
                admit_run.validate_event(changed,*args[1:])
        bad_env = dict(env,GITHUB_RUN_ATTEMPT='2')
        with self.assertRaises(q.GateError):
            admit_run.validate_event(event,bad_env,*args[2:])
        with self.assertRaises(q.GateError):
            admit_run.validate_event(*args[:5],['src/kws.c'],args[-1])
        event['pull_request']['body'] = admit_run.PREFIX + json.dumps(dict(approval,total_input_bytes=1))
        with self.assertRaises(q.GateError):
            admit_run.validate_event(*args)

    def test_workflow_has_only_explicitly_gated_runtime_entry(self):
        workflow = (ROOT.parents[1] / '.github/workflows/research-asr-runtime.yml').read_text()
        active = '\n'.join(x for x in workflow.splitlines() if not x.lstrip().startswith('#'))
        for forbidden in ('workflow_dispatch:', 'schedule:', 'pull_request_target:', 'docker run', 'pip install',
                          'upload-artifact', 'actions/cache', '--docker-readonly'):
            self.assertNotIn(forbidden, active)
        self.assertIn('persist-credentials: false', active)
        self.assertIn("github.event.action == 'labeled'", active)
        self.assertIn("github.event.label.name == 'asr-runtime-qualify-v3'", active)
        self.assertIn('github.run_attempt == 1', active)
        self.assertIn('needs: candidate-contract', active)
        self.assertIn('research/asr_runtime/admit_run.py', active)
        self.assertIn("github.event.pull_request.head.repo.full_name == github.repository", active)
        self.assertIn('python3 -I -S research/asr_runtime/tests/test_qualify.py', active)
    def synthetic_built_wheel(self, with_native):
        import base64
        payloads = {'fixture/__init__.py': b'# fixture only\n', 'fixture/LICENSE': b'synthetic license',
                    'fixture-1.dist-info/METADATA': b'Metadata-Version: 2.1\nName: fixture\nVersion: 1\n',
                    'fixture-1.dist-info/WHEEL': ('Wheel-Version: 1.0\nRoot-Is-Purelib: ' + ('false' if with_native else 'true') + '\nTag: cp312-cp312-linux_x86_64\n').encode()}
        native = 'fixture/_native.cpython-312-x86_64-linux-gnu.so'
        if with_native:
            payloads[native] = b'\x7fELF\x02\x01' + b'\x00'*12 + b'\x3e\x00'
        record = 'fixture-1.dist-info/RECORD'
        rows = [name + ',sha256:' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode() + ',' + str(len(data)) for name,data in payloads.items()]
        rows = [row.replace(',sha256:', ',sha256=') for row in rows]
        payloads[record] = ('\n'.join(rows + [record + ',,']) + '\n').encode()
        path = self.path / 'fixture-1-cp312-cp312-linux_x86_64.whl'
        with zipfile.ZipFile(path, 'w') as z:
            for name, data in payloads.items():
                z.writestr(name, data)
        return {'sources': [{'name':'fixture', 'version':'1', 'wheel_filename_prefix':'fixture-1-',
                            'required_wheel_members':['fixture/LICENSE'], 'required_native_members':[native],
                            'expected_requires_dist':[], 'allowed_generated_members':['fixture/__init__.py','fixture-1.dist-info/METADATA','fixture-1.dist-info/WHEEL','fixture-1.dist-info/RECORD'], 'expected_static_payload_sha256':
                                {'fixture/LICENSE': hashlib.sha256(payloads['fixture/LICENSE']).hexdigest()}}]}

    def test_source_wheel_native_payload_gate(self):
        plan = self.synthetic_built_wheel(True)
        result = c.validate_built_wheels(self.path, plan, self.limits)
        self.assertEqual(len(result), 1)
        plan['sources'][0]['expected_static_payload_sha256']['fixture/LICENSE'] = '0'*64
        with self.assertRaises(q.GateError):
            c.validate_built_wheels(self.path, plan, self.limits)

    def test_pure_python_source_fallback_is_rejected(self):
        plan = self.synthetic_built_wheel(False)
        with self.assertRaisesRegex(q.GateError, 'native payload'):
            c.validate_built_wheels(self.path, plan, self.limits)

    def test_image_repo_digest_name_normalization_is_narrow(self):
        image = self.admission['image']
        manifest = q.load_json(ROOT / 'locks/image.manifest.json')
        observed = {'Os':'linux','Architecture':'amd64','Size':123,'Id':manifest['config']['digest']}
        for name in ('python','library/python','docker.io/library/python'):
            observed['RepoDigests'] = [name+'@'+image['manifest_digest']]
            self.assertEqual(c.validate_pulled_image(observed,image,manifest),observed['Id'])
        observed['RepoDigests'] = ['evil.invalid/python@'+image['manifest_digest']]
        with self.assertRaises(q.GateError):
            c.validate_pulled_image(observed,image,manifest)

    def test_real_metadata_lock_and_oci_identities(self):
        path = ROOT / 'locks'
        inventory = q.load_json(path / 'inventory.json')
        self.assertEqual(q.sha256_file(path / 'inventory.json'), self.admission['required_inventory_sha256'])
        files = q.validate_inventory(inventory, 143, 3060297109)
        c.validate_root_policy(files, self.admission['runtime_root_policy'])
        c.validate_recipe(inventory, (path / 'sensevoice-recipe-requirements.txt').read_bytes())
        result = q.validate_image_metadata(self.admission['image'], (path / 'image.manifest.json').read_bytes(), (path / 'image.config.json').read_bytes())
        self.assertEqual(result['compressed_bytes'], 380580320)

    def test_recipe_source_or_constraints_cannot_be_removed(self):
        inventory = q.load_json(ROOT / 'locks/inventory.json')
        source = (ROOT / 'locks/sensevoice-recipe-requirements.txt').read_bytes()
        for key in ('source', 'applied_numeric_constraints'):
            altered = copy.deepcopy(inventory)
            altered['model_recipe'].pop(key)
            with self.subTest(key=key), self.assertRaises(q.GateError):
                c.validate_recipe(altered, source)
        with self.assertRaises(q.GateError):
            c.validate_recipe(inventory, source.replace(b'1.26.4', b'2.00.0'))
        files = [{'name': n, 'version': v} for n, v in self.admission['runtime_root_policy']['exact_versions'].items()]
        next(x for x in files if x['name'] == 'torch')['version'] = '2.11.0'
        with self.assertRaises(q.GateError):
            c.validate_root_policy(files, self.admission['runtime_root_policy'])

    def test_recipe_policy_rejects_numpy2_despite_graph_resolution(self):
        policy = self.admission['runtime_root_policy']
        files = [{'name': n, 'version': v} for n, v in policy['exact_versions'].items()]
        c.validate_root_policy(files, policy)
        next(x for x in files if x['name'] == 'numpy')['version'] = '2.5.3'
        with self.assertRaises(q.GateError):
            c.validate_root_policy(files, policy)

    def test_cpu_probe_source_receipt_newline(self):
        import ast
        import container_stage
        tree = ast.parse(container_stage.PROBE)
        writes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == 'write_text']
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].args[0].right.value, '\n')
        compile(container_stage.PROBE, '<cpu-probe-source-only>', 'exec')

    def test_no_model_entry_points(self):
        import ast
        tree = ast.parse((ROOT / 'container_stage.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                self.assertNotIn(n.func.attr, ('from_pretrained', 'snapshot_download', 'hf_hub_download', 'generate'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
