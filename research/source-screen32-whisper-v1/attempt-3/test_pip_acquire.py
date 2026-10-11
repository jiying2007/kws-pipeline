"""Offline worker contracts: never execute pip, download, install or infer."""
import hashlib
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pip_acquire_under_test', HERE / 'pip_acquire.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class WorkerTests(unittest.TestCase):
    def test_result_schema_rejects_unknown_fields_and_reason(self):
        value = p.safe_result(2, 'acquisition', '25.2', reason='pip_acquisition_failed')
        self.assertEqual(set(value), p.RESULT_FIELDS)
        self.assertEqual(value['filename'], None)
        for changed in ({**value, 'raw_log': 'secret'}, {**value, 'schema': 'other'},
                        {**value, 'reason': 'HTTP 403 secret'}, {**value, 'stage': 'raw URL'},
                        {**value, 'asset_index': True}):
            with self.assertRaises(p.WorkerFailure):
                p.validate_result(changed)
        sanitized = p.safe_result(-1, 'https://secret', 'secret', reason='HTTP 403 secret')
        self.assertEqual(sanitized['reason'], 'worker_internal_failure')
        self.assertEqual(sanitized['stage'], 'scope')
        self.assertIsNone(sanitized['asset_index'])
        self.assertNotIn('secret', json.dumps(sanitized))

    def test_result_success_has_only_verified_identity(self):
        row = p.read_asset(HERE.parent / 'runtime-lock.json', 0)
        outcome = {key: row[key] for key in ('filename', 'size_bytes', 'sha256')}
        outcome.update(status='ok', asset_index=0, pip_version='25.2')
        result = p.safe_result(0, 'export', '25.2', outcome)
        self.assertEqual(result['stage'], 'complete')
        self.assertEqual(result['filename'], row['filename'])
        self.assertIsNone(result['reason'])
        with self.assertRaises(p.WorkerFailure):
            p.safe_result(0, 'export', '25.2', {**outcome, 'url': 'secret'})

    def test_failure_writes_metadata_once_without_raw_exception(self):
        with tempfile.TemporaryDirectory() as td:
            evidence = p.Evidence(td)
            def fail(*args):
                args[-1].mark('acquisition')
                raise p.WorkerFailure('HTTP 403 username:secret')
            argv = ['pip_acquire.py', '--asset-index', '0', '--manifest', '/lock',
                    '--output-root', '/tmpfs', '--broker-socket', '/sock', '--candidate-root', '/candidate']
            with patch.object(p, 'Evidence', return_value=evidence), \
                    patch.object(p, 'acquire', side_effect=fail), \
                    patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(p.main(), 1)
            result = json.loads((Path(td) / 'worker-result.json').read_bytes())
            self.assertEqual(result['stage'], 'acquisition')
            self.assertEqual(result['reason'], 'worker_internal_failure')
            self.assertNotIn('secret', stream.getvalue())
            with self.assertRaises(FileExistsError):
                evidence.write('worker-result.json', result)

    def test_success_writes_structured_result(self):
        row = p.read_asset(HERE.parent / 'runtime-lock.json', 0)
        outcome = {key: row[key] for key in ('filename', 'size_bytes', 'sha256')}
        outcome.update(status='ok', asset_index=0, pip_version='25.2')
        with tempfile.TemporaryDirectory() as td:
            evidence = p.Evidence(td)
            def success(*args):
                args[-1].mark('export'); args[-1].pip_version = '25.2'
                return outcome
            argv = ['pip_acquire.py', '--asset-index', '0', '--manifest', '/lock',
                    '--output-root', '/tmpfs', '--broker-socket', '/sock', '--candidate-root', '/candidate']
            with patch.object(p, 'Evidence', return_value=evidence), \
                    patch.object(p, 'acquire', side_effect=success), \
                    patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                self.assertEqual(p.main(), 0)
            result = json.loads((Path(td) / 'worker-result.json').read_bytes())
            p.validate_result(result)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['sha256'], row['sha256'])

    def test_kernel_scope_sanitizes_raw_status(self):
        status = dict(NoNewPrivs='1', Seccomp='2', NSpid='42', Secret='credential')
        status.update({key: '0000000000000000' for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')})
        snapshot = dict(role='worker', uid=1000, cgroup='0::/',
                        limits={'memory.max': '12616466432', 'memory.swap.max': '0',
                                'cpu.max': '300000 100000', 'pids.max': '240'},
                        interfaces=['lo'], readonly_root=True, private_readonly_cgroup=True,
                        status=status, raw='private arbitrary status')
        clean = p.sanitized_scope(snapshot)
        self.assertEqual(clean['schema'], 'bounded-kernel-scope-v1')
        self.assertNotIn('Secret', clean['status'])
        self.assertNotIn('raw', clean)
        self.assertNotIn('credential', json.dumps(clean))
        snapshot['uid'] = 0
        with self.assertRaises(p.WorkerFailure):
            p.sanitized_scope(snapshot)

    def test_failed_help_probe_retains_only_sanitized_version(self):
        with tempfile.TemporaryDirectory() as td:
            evidence = p.Evidence(td)
            def fake(argv, env, deadline):
                return (0, b'pip 25.2 from /private/path (python 3.12)', b'') if argv[-1] == '--version' else (1, b'secret', b'secret')
            with self.assertRaises(p.WorkerFailure):
                p.probe_capabilities({}, 50, fake, evidence)
            self.assertEqual(evidence.stage, 'pip_help_probe')
            self.assertEqual(evidence.pip_version, '25.2')

    def test_deadline_uses_original_host_cutoff(self):
        with tempfile.TemporaryDirectory() as td, patch.object(p.time, 'monotonic', return_value=100):
            path = Path(td) / 'deadline.json'
            path.write_text(json.dumps({'deadline_monotonic': 150}))
            self.assertEqual(p.worker_deadline(300, path), 150)
            path.write_text(json.dumps({'deadline_monotonic': 900}))
            self.assertEqual(p.worker_deadline(300, path), 400)
            for value in (100, 99, 1301, float('nan'), True):
                path.write_text(json.dumps({'deadline_monotonic': value}))
                with self.assertRaisesRegex(p.WorkerFailure, 'host_deadline'):
                    p.worker_deadline(300, path)

    def test_exact_lock_all_22_rows(self):
        manifest = HERE.parent / 'runtime-lock.json'
        rows = json.loads(manifest.read_bytes())['files']
        self.assertEqual(len(rows), 22)
        for index, row in enumerate(rows):
            self.assertEqual(p.read_asset(manifest, index), row)
            text = p.requirement(row)
            self.assertEqual(text.count('\n'), 1)
            self.assertIn(row['url'], text)
            self.assertIn('--hash=sha256:' + row['sha256'], text)
        torch = p.read_asset(manifest, 18)
        self.assertEqual(torch['size_bytes'], 192268841)
        self.assertEqual(torch['sha256'], 'ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df')
        self.assertEqual(torch['version'], '2.12.1+cpu')

    def test_manifest_tamper_and_bad_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'lock.json'
            path.write_bytes((HERE.parent / 'runtime-lock.json').read_bytes() + b' ')
            with self.assertRaisesRegex(p.WorkerFailure, 'lock_identity'):
                p.read_asset(path, 0)
        for index in (-1, 22, True):
            with self.assertRaises(p.WorkerFailure):
                p.read_asset(HERE.parent / 'runtime-lock.json', index)

    def test_argv_official_single_requirement_retry_controls(self):
        argv = p.pip_argv('/work', '/usr/local/bin/python')
        self.assertEqual(argv[:4], ['/usr/local/bin/python', '-I', '-m', 'pip'])
        for required in ('--isolated', '--no-index', '--no-deps', '--require-hashes',
                         '--only-binary=:all:', '--no-cache-dir', '--no-input',
                         '--disable-pip-version-check'):
            self.assertIn(required, argv)
        for flag, value in (('--retries', '0'), ('--resume-retries', '0'),
                            ('--keyring-provider', 'disabled'),
                            ('--proxy', 'http://127.0.0.1:18080'),
                            ('--requirement', '/work/one.txt')):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn('--trusted-host', argv)
        self.assertNotIn('install', argv)

    def test_env_drops_credentials_proxy_and_python_injection(self):
        with patch.dict(os.environ, {'HTTPS_PROXY': 'https://secret@evil',
                                    'NETRC': '/secret', 'PIP_INDEX_URL': 'secret',
                                    'PYTHONPATH': '/evil'}):
            env = p.clean_env('/tmp/wheel')
        self.assertEqual(env['PIP_CONFIG_FILE'], '/dev/null')
        self.assertEqual(env['HOME'], '/tmp/wheel/home')
        self.assertEqual(env['TMPDIR'], '/tmp/wheel/tmp')
        self.assertFalse(set(env) & {'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY',
                                     'NO_PROXY', 'NETRC', 'PIP_INDEX_URL',
                                     'PYTHONPATH', 'PIP_CERT', 'REQUESTS_CA_BUNDLE'})

    def test_existing_global_or_site_config_fails(self):
        with patch.object(p.os.path, 'lexists', side_effect=lambda x: x == '/prefix/pip.conf'):
            with self.assertRaisesRegex(p.WorkerFailure, 'pip_config_present'):
                p.ensure_no_configs(p.clean_env('/root'), '/prefix')

    def good_runner(self, argv, env, deadline):
        if argv[-1] == '--version':
            return 0, b'pip 25.2 from /trusted/site (python 3.12)\n', b''
        if argv[-1] == '--help':
            return 0, ' '.join(p.REQUIRED_OPTIONS).encode(), b''
        return 0, b'', b''

    def test_capabilities_are_actual_runtime_probes_mocked_here(self):
        self.assertEqual(p.probe_capabilities({}, 50, self.good_runner), '25.2')

    def test_missing_resume_flag_is_fatal(self):
        def fake(argv, env, deadline):
            result = self.good_runner(argv, env, deadline)
            return result[0], result[1].replace(b'--resume-retries', b''), result[2]
        with self.assertRaisesRegex(p.WorkerFailure, 'pip_capability_probe'):
            p.probe_capabilities({}, 50, fake)

    def test_effective_config_must_be_empty(self):
        def fake(argv, env, deadline):
            if argv[-1] == 'list':
                return 0, b'global.index-url=bad\n', b''
            return self.good_runner(argv, env, deadline)
        with self.assertRaisesRegex(p.WorkerFailure, 'pip_effective_config'):
            p.probe_capabilities({}, 50, fake)

    def test_probe_failure_not_skipped(self):
        with self.assertRaisesRegex(p.WorkerFailure, 'pip_version_probe'):
            p.probe_capabilities({}, 50, lambda *args: (1, b'', b'failure'))

    def test_cleanup_kills_group_after_leader_exited(self):
        process = Mock(pid=4321)
        process.poll.return_value = 0
        with patch.object(p.os, 'killpg') as kill:
            p.stop_process(process)
        kill.assert_called_once_with(4321, p.signal.SIGKILL)
        process.wait.assert_called_once_with()
        with patch.object(p.os, 'killpg', side_effect=ProcessLookupError):
            p.stop_process(process)

    def test_capped_subprocess_only_harmless_python(self):
        # Explicitly not pip. Exercises pipe caps/deadlines without assets.
        code, out, err = p.run_capped([sys.executable, '-c', 'print("hello")'],
                                      p.clean_env('/tmp'), time.monotonic() + 3)
        self.assertEqual((code, out, err), (0, b'hello\n', b''))

    def test_output_flood_stops_child(self):
        with self.assertRaisesRegex(p.WorkerFailure, 'pip_output_limit'):
            p.run_capped([sys.executable, '-c', 'import os; os.write(1,b"x"*300000)'],
                         p.clean_env('/tmp'), time.monotonic() + 3)

    def test_process_deadline_stops_child(self):
        with self.assertRaisesRegex(p.WorkerFailure, 'deadline'):
            p.run_capped([sys.executable, '-c', 'import time; time.sleep(10)'],
                         p.clean_env('/tmp'), time.monotonic() + .1)

    def test_rlimit_exact_in_child(self):
        code, out, _ = p.run_capped([sys.executable, '-c',
                                     'import resource; print(resource.getrlimit(resource.RLIMIT_FSIZE))'],
                                    p.clean_env('/tmp'), time.monotonic() + 3, 12345)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), b'(12345, 12345)')

    def test_tmpfs_and_hard_quota_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'work'; root.mkdir()
            mounts = Path(td) / 'mountinfo'
            mounts.write_text(f'1 2 0:1 / {root} rw - ext4 /dev/root rw\n')
            with self.assertRaisesRegex(p.WorkerFailure, 'tmpfs_required'):
                p.ensure_tmpfs(root, mounts)
            mounts.write_text(f'1 2 0:1 / {root} rw - tmpfs tmpfs rw,size=524288k\n')
            class FS:
                f_blocks = 512; f_frsize = 1024**2
            with patch.object(p.os, 'statvfs', return_value=FS()):
                p.ensure_tmpfs(root, mounts)
            FS.f_blocks = 513
            with patch.object(p.os, 'statvfs', return_value=FS()):
                with self.assertRaisesRegex(p.WorkerFailure, 'tmpfs_quota'):
                    p.ensure_tmpfs(root, mounts)
            (root / 'old').touch()
            with self.assertRaisesRegex(p.WorkerFailure, 'tmpfs_not_fresh'):
                p.ensure_tmpfs(root, mounts)

    def test_final_identity_then_bounded_candidate_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'output').mkdir(); candidate = root / 'candidate'; candidate.mkdir()
            data = b'fake wheel bytes, not installable'
            row = {'filename': 'x.whl', 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            output = root / 'output/x.whl'; output.write_bytes(data)
            p.verify_output(root, row)
            p.export_candidate(root, candidate, row)
            self.assertEqual((candidate / 'x.whl').read_bytes(), data)
            (candidate / 'x.whl').unlink()
            for wrong, reason in ((data[:-1], 'output_size'), (b'x'*len(data), 'output_hash')):
                output.write_bytes(wrong)
                with self.assertRaisesRegex(p.WorkerFailure, reason):
                    p.verify_output(root, row)
            output.unlink(); output.symlink_to('/dev/null')
            with self.assertRaisesRegex(p.WorkerFailure, 'output_size'):
                p.verify_output(root, row)

    def test_extra_filename_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'output').mkdir()
            (root / 'output/unexpected.whl').write_bytes(b'')
            with self.assertRaisesRegex(p.WorkerFailure, 'output_filename'):
                p.verify_output(root, {'filename': 'expected.whl'})

    def test_first_pip_failure_has_no_retry_or_status_invention(self):
        row = p.read_asset(HERE.parent / 'runtime-lock.json', 0)
        class Context:
            def __enter__(self): return p.PROXY
            def __exit__(self, *args): return False
        with tempfile.TemporaryDirectory() as td, \
                patch.object(p, 'verify_worker_scope'), patch.object(p, 'worker_deadline', return_value=time.monotonic()+300), patch.object(p, 'ensure_tmpfs'), patch.object(p, 'ensure_no_configs'), \
                patch.object(p, 'probe_capabilities', return_value='25.2'), \
                patch.object(p, 'shim', return_value=Context()), \
                patch.object(p, 'run_capped', return_value=(1, b'HTTP 403 secret', b'secret')) as runner, \
                patch.object(p, 'verify_output') as verify:
            with self.assertRaisesRegex(p.WorkerFailure, '^pip_acquisition_failed$'):
                p.acquire(0, HERE.parent / 'runtime-lock.json', td, '/fixed.sock', '/candidate')
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[3], row['size_bytes'])
            verify.assert_not_called()

    def test_shim_fake_sockets_short_send_eof_no_byte_loss(self):
        payload = bytes(range(256)) * 300
        class FakeSocket:
            def __init__(self, incoming):
                self.incoming = bytearray(incoming)
                self.sent = bytearray()
                self.closed = False
                self.shutdowns = []
            def setblocking(self, value): pass
            def settimeout(self, value): pass
            def connect(self, value): self.destination = value
            def recv(self, size):
                data = bytes(self.incoming[:size]); del self.incoming[:size]
                return data
            def send(self, data):
                count = min(37, len(data)); self.sent.extend(data[:count]); return count
            def shutdown(self, how): self.shutdowns.append(how)
            def close(self): self.closed = True
        client = FakeSocket(b'opaque request')
        upstream = FakeSocket(payload)
        class Listener(FakeSocket):
            accepted = False
            def accept(self):
                self.accepted = True
                return client, ('127.0.0.1', 1234)
        listener = Listener(b'')
        class Selector:
            def __init__(self): self.keys = {}
            def register(self, item, events): self.keys[item] = SimpleNamespace(fileobj=item, events=events)
            def unregister(self, item): return self.keys.pop(item)
            def modify(self, item, events): self.register(item, events)
            def get_key(self, item): return self.keys[item]
            def select(self, timeout):
                return [(key, key.events) for key in list(self.keys.values())
                        if key.fileobj is not listener or not listener.accepted]
            def close(self): pass
        def now(): return 2 if client.closed and upstream.closed else 0
        with patch.object(p.selectors, 'DefaultSelector', Selector), \
                patch.object(p.socket, 'socket', return_value=upstream), \
                patch.object(p.time, 'monotonic', now):
            p.serve_shim(listener, '/fixed/broker.sock', 1)
        self.assertEqual(client.sent, payload)
        self.assertEqual(upstream.sent, b'opaque request')
        self.assertIn(socket.SHUT_WR, client.shutdowns)
        self.assertIn(socket.SHUT_WR, upstream.shutdowns)
        self.assertEqual(upstream.destination, '/fixed/broker.sock')

    def test_opaque_shim_drains_eof_under_backpressure(self):
        payload = bytes(range(256)) * 512
        request = b'arbitrary opaque bytes' * 1000
        errors = []
        with tempfile.TemporaryDirectory() as td:
            try:
                broker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            except PermissionError:
                if os.environ.get('GITHUB_ACTIONS') == 'true':
                    raise  # Required hosted fixture: never skip a permission failure.
                self.skipTest('NOT_RUN: local socket creation denied; required on GitHub CI')
            broker.bind(td + '/broker.sock'); broker.listen(1); broker.settimeout(5)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(('127.0.0.1', 0)); listener.listen(4)
            address = listener.getsockname()
            def fake_broker():
                try:
                    conn, _ = broker.accept()
                    with conn:
                        conn.settimeout(5)
                        got = bytearray()
                        while True:
                            block = conn.recv(1024)
                            if not block: break
                            got.extend(block)
                        self.assertEqual(bytes(got), request)
                        conn.sendall(payload)
                        conn.shutdown(socket.SHUT_WR)
                except BaseException as error:
                    errors.append(error)
            def relay():
                try:
                    p.serve_shim(listener, td + '/broker.sock', time.monotonic() + 3)
                except BaseException as error:
                    errors.append(error)
            bt = threading.Thread(target=fake_broker); st = threading.Thread(target=relay)
            bt.start(); st.start()
            with socket.create_connection(address, timeout=4) as client:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                client.sendall(request); client.shutdown(socket.SHUT_WR)
                time.sleep(.03)
                result = bytearray()
                while True:
                    block = client.recv(777)
                    if not block: break
                    result.extend(block)
            bt.join(5); st.join(5); broker.close()
            self.assertFalse(bt.is_alive() or st.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(bytes(result), payload)


if __name__ == '__main__':
    unittest.main()
