"""Fictional kernel/container/PCM fixtures only; no Docker or ML calls."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_scope as scope
import hosted_run as host
import tts_worker as tts


def kernel():
    return {'/proc/self/cgroup': '0::/\n',
            '/proc/self/mountinfo': '1 0 0:1 / / ro - overlay overlay ro\n2 1 0:2 / /sys/fs/cgroup ro - cgroup2 cgroup ro\n',
            '/proc/self/status': 'NoNewPrivs:\t1\nSeccomp:\t2\nNSpid:\t1\n' +
              ''.join(name + ':\t0000000000000000\n' for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')),
            **{'/sys/fs/cgroup/' + name: value for name, value in
               {'memory.max': str(scope.MEMORY), 'memory.swap.max': '0', 'cpu.max': '400000 100000', 'pids.max': '256'}.items()}}


class ScopeTests(unittest.TestCase):
    def run_scope(self, values, interfaces=('lo',), uid=1001, network=True):
        path = mock.Mock(); path.iterdir.return_value = [types.SimpleNamespace(name=name) for name in interfaces]
        with mock.patch.object(scope, 'read', side_effect=lambda path: values[path]), \
                mock.patch.object(scope.os, 'geteuid', return_value=uid), mock.patch.object(scope, 'Path', return_value=path):
            return scope.verify_runtime_scope(network_required=network)

    def test_actual_limit_readback(self):
        result = self.run_scope(kernel())
        self.assertTrue(result['network_disabled']); self.assertEqual(result['limits']['memory.swap.max'], '0')

    def test_unknown_unlimited_or_wrong_limits_fail(self):
        for key, values in [('memory.max', ('max', str(scope.MEMORY + 1))), ('memory.swap.max', ('max', '1')),
                            ('cpu.max', ('max 100000', '800000 100000')), ('pids.max', ('max', '512'))]:
            for value in values:
                data = kernel(); data['/sys/fs/cgroup/' + key] = value
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError): self.run_scope(data)

    def test_namespace_privilege_and_writable_root_fail(self):
        for key, value in [('/proc/self/cgroup', '0::/user.slice/other'),
                           ('/proc/self/mountinfo', kernel()['/proc/self/mountinfo'].replace('/ / ro', '/ / rw')),
                           ('/proc/self/status', kernel()['/proc/self/status'].replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0')),
                           ('/proc/self/status', kernel()['/proc/self/status'].replace('0000000000000000', '0000000000000001'))]:
            data = kernel(); data[key] = value
            with self.assertRaises(RuntimeError): self.run_scope(data)
        with self.assertRaises(RuntimeError): self.run_scope(kernel(), uid=0)

    def test_network_exception_is_setup_only(self):
        with self.assertRaises(RuntimeError): self.run_scope(kernel(), ('eth0', 'lo'))
        self.assertFalse(self.run_scope(kernel(), ('eth0', 'lo'), network=False)['network_disabled'])


class HostTests(unittest.TestCase):
    def command(self, profile='tts', stage='tts'):
        with mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=1001):
            return host.container_args('fixed', 'python@sha256:' + 'a' * 64, Path('/stage/code'),
                Path('/stage/runtime'), Path('/stage/output'), profile, stage,
                Path('/stage/input') if profile == 'asr' else None)

    def test_fixed_container_command_has_real_limits_and_offline_readonly_inputs(self):
        command = self.command()
        for option, value in [('--memory', str(scope.MEMORY)), ('--memory-swap', str(scope.MEMORY)),
                               ('--cpus', '4'), ('--pids-limit', '256'), ('--network', 'none'),
                               ('--cgroupns', 'private'), ('--cap-drop', 'ALL')]:
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn('--read-only', command)
        self.assertIn('type=bind,src=/stage/runtime,dst=/runtime,readonly', command)
        self.assertFalse(any(value in command for value in ('--privileged', '--pid=host', '--network=host')))
        self.assertNotIn('/input', ' '.join(command))

    def log_record(self):
        return {'HostConfig': {'LogConfig': {'Type': 'local', 'Config': {
            'max-size': '1m', 'max-file': '1', 'compress': 'false'}},
            'Memory': scope.MEMORY, 'MemorySwap': scope.MEMORY, 'NanoCpus': 4_000_000_000, 'PidsLimit': scope.PIDS,
            'ReadonlyRootfs': True, 'Privileged': False, 'CgroupnsMode': 'private', 'PidMode': '', 'IpcMode': 'private',
            'CapAdd': None, 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'], 'NetworkMode': 'bridge'},
            'Mounts': [{'Type': 'bind', 'Destination': name, 'RW': writable}
                       for name, writable in (('/code', False), ('/runtime', True), ('/output', True))]}

    def test_single_file_local_logger_explicitly_disables_compression(self):
        command = self.command(stage='setup')
        options = [command[index + 1] for index, item in enumerate(command) if item == '--log-opt']
        self.assertEqual(options, ['max-size=1m', 'max-file=1', 'compress=false'])
        self.assertEqual(command[command.index('--log-driver') + 1], 'local')
        host.validate_container_inspect(self.log_record(), 'setup')
        for mutation in ({'max-size': '1m', 'max-file': '1'},
                         {'max-size': '1m', 'max-file': '1', 'compress': 'true'},
                         {'max-size': '2m', 'max-file': '1', 'compress': 'false'},
                         {'max-size': '1m', 'max-file': '2', 'compress': 'false'}):
            record = self.log_record(); record['HostConfig']['LogConfig']['Config'] = mutation
            with self.assertRaisesRegex(ValueError, 'local log configuration'): host.validate_container_inspect(record, 'setup')
        record = self.log_record(); record['HostConfig']['LogConfig']['Type'] = 'json-file'
        with self.assertRaises(ValueError): host.validate_container_inspect(record, 'setup')

    def test_probe1_derived_logging_error_is_byte_exact_and_not_raw_capture(self):
        root = HERE / 'evidence/probe-1-38016468182'
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(provenance['archive']['sha256'], 'd13958e194de647d42df073932c0d341482b3b4ca915198ede2dd14a95a5a207')
        for name, row in provenance['exact_members'].items():
            self.assertEqual(host.file_hash(root / name), row['sha256'])
            self.assertEqual((root / name).stat().st_size, row['bytes'])
        for name, digest in json.loads((root / 'artifact-freeze.json').read_text())['files'].items():
            self.assertEqual(host.file_hash(root / name), digest)
        raw = (root / 'derived-stderr-reconstruction.txt').read_bytes()
        recorded = json.loads((root / 'container-setup.json').read_text())['primary_failure']['stderr']
        self.assertEqual(len(raw), recorded['observed_bytes']); self.assertEqual(len(raw), 229)
        self.assertEqual(host.hashlib.sha256(raw).hexdigest(), recorded['sha256'])
        self.assertEqual(recorded['sha256'], 'c88004f52342b8848ee6f308b1976580f0f79d062a316435dc7c0ee6308dd9c2')
        self.assertEqual(provenance['derived_stderr_proof']['classification'], 'DERIVED_RECONSTRUCTION_NOT_RAW_CAPTURE')
        self.assertEqual(provenance['observed_result']['docker_daemon_version'], 'NOT_RETAINED')
        failure = host.public_failure(host.subprocess.CalledProcessError(1, ['docker', 'start'], stderr=raw), 'start')
        self.assertIn('compression cannot be enabled when max file count is 1', failure['stderr']['safe_fragments'])
        self.assertIn('failed to initialize logging driver', failure['stderr']['safe_fragments'])
        self.assertNotIn('qwen16-container-probe-1-38016468182', json.dumps(failure))
        self.assertEqual(provenance['probe_slots_used'], 1); self.assertEqual(provenance['probe_slots_remaining'], 1)

    def test_setup_network_and_asr_input_are_explicit(self):
        self.assertEqual(self.command(stage='setup')[self.command(stage='setup').index('--network') + 1], 'bridge')
        command = self.command('asr', 'qwen06')
        self.assertIn('type=bind,src=/stage/input,dst=/input,readonly', command)
        self.assertIn('/code/research/source-screen32-v1/asr_worker.py', command)
        self.assertNotIn('tts_worker.py', ' '.join(command))

    def test_no_arbitrary_stage_or_command(self):
        with self.assertRaises(ValueError): self.command(stage='bash')

    def test_frozen_profiles_bind_sources_and_keep_asr_blind(self):
        freeze = json.loads((HERE / 'execution-freeze.json').read_text())
        for name, digest in freeze['files'].items():
            self.assertEqual(host.file_hash(host.ROOT / name), digest, name)
        reused = set(json.loads((HERE / 'reuse-pins.json').read_text()))
        self.assertTrue(reused <= set(freeze['profiles']['tts']))
        for name in freeze['profiles']['asr']:
            self.assertNotIn(name, ('research/source-screen32-v1/contract.py', 'research/source-screen32-v1/plan.json'))
            raw = (host.ROOT / name).read_bytes()
            for text in ('你好小窝', '小窝小窝', '你好小屋', '小屋小屋', '成年女性', '成年男性', 'screen32-001'):
                self.assertNotIn(text.encode(), raw, name)
        with tempfile.TemporaryDirectory() as temporary:
            staged = Path(temporary)
            host.stage_code(freeze, 'tts', staged)
            result = host.subprocess.run([sys.executable, '-I', '-B', str(staged / 'research/source-screen32-v1/contract.py')],
                                         capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['local_reuse_pins_verified'], len(reused))

    def test_release_false_blocks_before_docker(self):
        release = {'schema': 'screen32-execution-release-v1', 'experiment': host.EXPERIMENT,
                   'approved': False, 'source_freeze_sha256': 'a' * 64}
        with mock.patch.object(host, 'bounded_json', return_value=release), \
                mock.patch.object(host, 'file_hash', return_value='a' * 64), \
                mock.patch.object(host, 'docker') as docker, self.assertRaises(ValueError):
            host.verify_admission({})
        docker.assert_not_called()

    def admission_fixture(self, root):
        here = root / 'research/source-screen32-v1'; here.mkdir(parents=True)
        (here / 'workflow.yml').write_text('fictional-v3-workflow')
        (here / 'execution-freeze.json').write_text(json.dumps({'schema': 'screen32-execution-source-freeze-v1', 'files': {}, 'profiles': {}}))
        release = {'schema': 'screen32-execution-release-v1', 'experiment': 'qwen16-voicedesign-once-v3',
                   'approved': True, 'source_freeze_sha256': host.file_hash(here / 'execution-freeze.json'), 'successful_probe': None}
        (here / 'execution-release.json').write_text(json.dumps(release))
        workflows = root / '.github/workflows'; workflows.mkdir(parents=True)
        (workflows / 'source-screen32-run-v3.yml').write_text('fictional-v3-workflow')
        event = {'created': True, 'deleted': False, 'after': 'a' * 40,
                 'repository': {'private': False, 'full_name': 'jiying2007/kws-pipeline'}}
        (root / 'event.json').write_text(json.dumps(event))
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/research/qwen16-voicedesign-once-v3',
               'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
               'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '12345', 'GITHUB_EVENT_PATH': str(root / 'event.json')}
        return here, release, event, env

    def test_v3_admission_rejects_consumed_v1_release_ref_or_workflow(self):
        self.assertEqual(host.EXPERIMENT, 'qwen16-voicedesign-once-v3')
        self.assertEqual(host.BRANCH, 'research/qwen16-voicedesign-once-v3')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); here, release, event, env = self.admission_fixture(root)
            with mock.patch.object(host, 'HERE', here), mock.patch.object(host, 'ROOT', root), mock.patch.object(host, 'verify_successful_probe'):
                host.verify_admission(env)
                old = dict(env, GITHUB_REF='refs/heads/research/qwen16-voicedesign-once-v1')
                with self.assertRaises(ValueError): host.verify_admission(old)
                release['experiment'] = 'qwen16-voicedesign-once-v1'
                (here / 'execution-release.json').write_text(json.dumps(release))
                with self.assertRaises(ValueError): host.verify_admission(env)
                release['experiment'] = host.EXPERIMENT
                (here / 'execution-release.json').write_text(json.dumps(release))
                (root / '.github/workflows/source-screen32-run-v3.yml').rename(root / '.github/workflows/source-screen32-run.yml')
                with self.assertRaisesRegex(ValueError, 'activated workflow drift'): host.verify_admission(env)

    def test_v3_still_requires_first_created_run_and_attempt_and_exact_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); here, release, event, env = self.admission_fixture(root)
            with mock.patch.object(host, 'HERE', here), mock.patch.object(host, 'ROOT', root), mock.patch.object(host, 'verify_successful_probe'):
                for field in ('GITHUB_RUN_NUMBER', 'GITHUB_RUN_ATTEMPT'):
                    with self.assertRaises(ValueError): host.verify_admission(dict(env, **{field: '2'}))
                event['created'] = False; (root / 'event.json').write_text(json.dumps(event))
                with self.assertRaises(ValueError): host.verify_admission(env)
                event['created'] = True; (root / 'event.json').write_text(json.dumps(event))
                (root / '.github/workflows/source-screen32-run-v3.yml').write_text('modified-workflow')
                with self.assertRaisesRegex(ValueError, 'activated workflow drift'): host.verify_admission(env)
        workflow = (HERE / 'workflow.yml').read_text()
        for expected in ('name: Qwen16 VoiceDesign source screen once v3',
                         'branches: [research/qwen16-voicedesign-once-v3]', 'group: qwen16-voicedesign-once-v3',
                         'github.run_number == 1', 'github.run_attempt == 1', 'github.event.created == true'):
            self.assertIn(expected, workflow)
        self.assertNotIn('qwen16-voicedesign-once-v1', workflow)

    def network(self, received=100, virtual=200, interface='eth0', ifindex=2):
        return {'default_interface': interface, 'interfaces': {
            interface: {'ifindex': ifindex, 'rx_bytes': received},
            'veth0': {'ifindex': 3, 'rx_bytes': virtual}}}

    def lock(self):
        return json.loads((HERE / 'container-lock.json').read_text())

    def test_default_interface_observer_uses_readonly_route_and_counter_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {'/proc/net/route': 'Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n'
                'eth0 00000000 0100000A 0003 0 0 100 00000000 0 0 0\n',
                '/proc/net/ipv6_route': '0' * 32 + ' 00 ' + '0' * 32 + ' 00 ' + '0' * 32 + ' 00000000 00000000 00000000 00200200 lo\n',
                '/sys/class/net/eth0/statistics/rx_bytes': '100', '/sys/class/net/eth0/ifindex': '2',
                '/sys/class/net/veth0/statistics/rx_bytes': '200', '/sys/class/net/veth0/ifindex': '3',
                '/sys/class/net/lo/statistics/rx_bytes': '999999'}
            for name, text in fixtures.items():
                path = root / name.lstrip('/'); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
            with mock.patch.object(host, 'Path', side_effect=lambda name: root / str(name).lstrip('/')):
                self.assertEqual(host.network_received(), self.network())
                (root / 'proc/net/route').write_text(fixtures['/proc/net/route'] +
                    'eth1 00000000 0200000A 0003 0 0 100 00000000 0 0 0\n')
                with self.assertRaisesRegex(ValueError, 'single non-loopback'): host.network_received()
                (root / 'proc/net/route').write_text(fixtures['/proc/net/route'].splitlines()[0] + '\n')
                with self.assertRaisesRegex(ValueError, 'single non-loopback'): host.network_received()
                (root / 'proc/net/route').write_text(fixtures['/proc/net/route'])
                (root / 'proc/net/ipv6_route').write_text('0' * 32 + ' 00 ' + '0' * 32 + ' 00 ' + '0' * 32 +
                    ' 00000000 00000000 00000000 00000001 eth1\n')
                with self.assertRaisesRegex(ValueError, 'single non-loopback'): host.network_received()

    def test_stacked_interface_aggregate_is_diagnostic_never_image_gate(self):
        observation = {}
        before = self.network(100, 200)
        after = self.network(100 + 70 * 1024 ** 2, 200 + 70 * 1024 ** 2)
        host.record_network_delta(before, after, observation)
        self.assertEqual(observation['default_interface_received_delta_bytes'], 70 * 1024 ** 2)
        self.assertEqual(observation['all_interface_delta_diagnostic_only_bytes'], 140 * 1024 ** 2)
        self.assertEqual(observation['network_after'], after)
        self.assertGreater(observation['all_interface_delta_diagnostic_only_bytes'], host.IMAGE_TRANSFER_RESERVE)

    def test_selected_interface_change_reset_or_overshoot_fails_closed(self):
        for after in (self.network(interface='eth1'), self.network(ifindex=99), self.network(received=99),
                      self.network(received=100 + host.IMAGE_TRANSFER_RESERVE + 1)):
            observation = {}
            with self.assertRaises(ValueError): host.record_network_delta(self.network(), after, observation)
            self.assertEqual(observation['network_after'], after)

    def test_virtual_interface_churn_does_not_hide_selected_interface_observation(self):
        after = self.network(received=110); del after['interfaces']['veth0']
        observation = {}; host.record_network_delta(self.network(), after, observation)
        self.assertEqual(observation['default_interface_received_delta_bytes'], 10)
        self.assertIsNone(observation['all_interface_delta_diagnostic_only_bytes'])

    def test_successful_pull_retains_background_inclusive_observation_not_exact_bytes(self):
        process = mock.Mock(); process.poll.return_value = 0; process.returncode = 0
        lock = self.lock()
        image = [{'Id': lock['manifest']['config']['digest'], 'Architecture': 'amd64', 'Os': 'linux'}]
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen', return_value=process) as popen, \
                mock.patch.object(host, 'network_received', side_effect=[self.network(), self.network(received=100000000)]), \
                mock.patch.object(host, 'docker', return_value=json.dumps(image)):
            target = Path(temporary) / 'image-transfer.json'
            result = host.pull_image(lock, target)
            self.assertEqual(result, json.loads(target.read_text()))
            self.assertTrue(result['daemon_completion_verified'])
            self.assertFalse(result['cumulative_image_transfer_bytes_verified'])
            self.assertEqual(result['declared_config_and_compressed_layer_bytes'], 45449534)
            self.assertEqual(result['default_interface_received_delta_bytes'], 99999900)
            self.assertIn('unrelated_host_traffic', result['byte_scope'])
            self.assertEqual(result['pull_client_attempts'], 1); popen.assert_called_once()
            process.terminate.assert_not_called()

    def test_failed_image_pull_does_not_claim_daemon_transfer_cutoff(self):
        process = mock.Mock(); process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen', return_value=process), \
                mock.patch.object(host, 'network_received', side_effect=[self.network(), self.network(received=101 + host.IMAGE_TRANSFER_RESERVE)]), \
                mock.patch.object(host, 'docker') as docker:
            target = Path(temporary) / 'image-transfer.json'
            with self.assertRaises(ValueError): host.pull_image(self.lock(), target)
            record = json.loads(target.read_text())
            self.assertEqual(record['status'], 'stopped_before_setup_daemon_completion_unverified')
            self.assertTrue(record['daemon_fetch_may_continue'])
            self.assertFalse(record['daemon_completion_verified'])
            self.assertFalse(record['cumulative_image_transfer_bytes_verified'])
            self.assertEqual(record['default_interface_received_delta_bytes'], host.IMAGE_TRANSFER_RESERVE + 1)
            self.assertEqual(record['network_before'], self.network())
            process.terminate.assert_called_once(); docker.assert_not_called()

    def test_immediate_failed_client_retains_terminal_counter_and_exit_code(self):
        process = mock.Mock(); process.poll.return_value = 1; process.returncode = 1
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen', return_value=process) as popen, \
                mock.patch.object(host, 'network_received', side_effect=[self.network(), self.network(received=500)]), \
                mock.patch.object(host, 'docker') as docker:
            target = Path(temporary) / 'image-transfer.json'
            with self.assertRaisesRegex(ValueError, 'single image pull failed'): host.pull_image(self.lock(), target)
            record = json.loads(target.read_text())
            self.assertEqual(record['pull_client_exit_code'], 1)
            self.assertEqual(record['default_interface_received_delta_bytes'], 400)
            self.assertTrue(record['after_snapshot_available']); self.assertTrue(record['daemon_fetch_may_continue'])
            popen.assert_called_once(); docker.assert_not_called()

    def test_unavailable_terminal_counter_is_explicit_and_never_retries(self):
        process = mock.Mock(); process.poll.return_value = 1; process.returncode = 1
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen', return_value=process) as popen, \
                mock.patch.object(host, 'network_received', side_effect=[self.network(), OSError('fictional missing counter')]):
            target = Path(temporary) / 'image-transfer.json'
            with self.assertRaises(OSError): host.pull_image(self.lock(), target)
            record = json.loads(target.read_text())
            self.assertEqual(record['pull_client_exit_code'], 1)
            self.assertFalse(record['after_snapshot_available']); self.assertNotIn('network_after', record)
            self.assertFalse(record['cumulative_image_transfer_bytes_verified']); popen.assert_called_once()

    def test_route_failure_records_zero_pull_attempts(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(host.subprocess, 'Popen') as popen, \
                mock.patch.object(host, 'network_received', side_effect=ValueError('fictional ambiguous default')):
            target = Path(temporary) / 'image-transfer.json'
            with self.assertRaises(ValueError): host.pull_image(self.lock(), target)
            record = json.loads(target.read_text())
            self.assertEqual(record['pull_client_attempts'], 0)
            self.assertFalse(record['daemon_fetch_may_continue']); popen.assert_not_called()

    def test_image_allowance_stays_inside_unchanged_tts_transfer_budget(self):
        import setup_adapter
        self.assertEqual(host.IMAGE_TRANSFER_RESERVE, setup_adapter.IMAGE_TRANSFER_RESERVE)
        self.assertEqual(host.IMAGE_TRANSFER_RESERVE, 128 * 1024 ** 2)
        self.assertEqual(setup_adapter.TTS_INPUT_CAP + host.IMAGE_TRANSFER_RESERVE, 6 * host.GIB)
        self.assertLess(4956729799, setup_adapter.TTS_INPUT_CAP)

    def test_retained_failed_run_exact_bytes_and_zero_model_attempts(self):
        root = HERE / 'evidence/run-38012657985'
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(provenance['archive']['sha256'], '6caaff3718e9e44ae66c345121b49edc0cb6d74979b3a6d1d26c09e883eea138')
        for name, row in provenance['exact_members'].items():
            self.assertEqual(host.file_hash(root / name), row['sha256'])
            self.assertEqual((root / name).stat().st_size, row['bytes'])
        freeze = json.loads((root / 'artifact-freeze.json').read_text())
        for name, digest in freeze['files'].items(): self.assertEqual(host.file_hash(root / name), digest)
        receipt = json.loads((root / 'tts/tts-receipt.json').read_text())
        self.assertEqual(len(receipt['ledger']), 32)
        self.assertTrue(all(row['attempts'] == 0 and row['status'] == 'NOT_RUN' for row in receipt['ledger']))
        transfer = json.loads((root / 'image-transfer.json').read_text())
        self.assertEqual(transfer['host_received_delta_bytes'], 93615466)
        self.assertFalse(transfer['cumulative_image_transfer_bytes_verified'])
        self.assertFalse(provenance['automatic_retry_authorized'])

    def test_retained_v2_failure_exact_bytes_and_unknown_docker_operation(self):
        root = HERE / 'evidence/run-38014209435'
        provenance = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(provenance['archive']['sha256'], '0f1a229464c0e6d1928116d2a0ff42cf8ef2cb9bf33fdb44e00c7d7b2ec96738')
        for name, row in provenance['exact_members'].items():
            self.assertEqual(host.file_hash(root / name), row['sha256'])
            self.assertEqual((root / name).stat().st_size, row['bytes'])
        for name, digest in json.loads((root / 'artifact-freeze.json').read_text())['files'].items():
            self.assertEqual(host.file_hash(root / name), digest)
        self.assertEqual(provenance['observed_result']['failed_docker_operation'], 'UNKNOWN')
        transfer = json.loads((root / 'image-transfer.json').read_text())
        self.assertEqual(transfer['default_interface_received_delta_bytes'], 45832909)
        self.assertTrue(transfer['daemon_completion_verified'])
        self.assertFalse(transfer['cumulative_image_transfer_bytes_verified'])
        receipt = json.loads((root / 'tts/tts-receipt.json').read_text())
        self.assertTrue(all(row['attempts'] == 0 and row['status'] == 'NOT_RUN' for row in receipt['ledger']))

    def test_terminal_tts_load_failure_retains_all32(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = host.terminal_tts(Path(temporary))
            self.assertEqual(len(receipt['ledger']), 32)
            self.assertTrue(all(row['attempts'] == 0 and row['status'] == 'NOT_RUN' for row in receipt['ledger']))

    def test_tts_claim_precedes_receipt_even_on_interrupt(self):
        for body in (b"screen32-001\n", b""):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tts.atomic(root / 'tts-receipt.json', {'schema': 'screen32-tts-receipt-v1', 'status': 'started',
                    'ledger': tts.initial_ledger(), 'outcomes': [], 'human_gold': False, 'training_admitted': False})
                (root / 'screen32-001.claim').write_bytes(body)
                receipt = host.terminal_tts(root)
                self.assertEqual(receipt['ledger'][0]['attempts'], 1)
                self.assertEqual(receipt['ledger'][0]['status'], 'FAILED_NO_RETRY')
                self.assertTrue(all(row['attempts'] == 0 for row in receipt['ledger'][1:]))
                self.assertEqual(receipt['outcomes'][0]['status'], 'failed_no_retry')

    def test_host_kills_container_on_watchdog_failure(self):
        calls = []
        def docker(*args, **kwargs):
            calls.append(args)
            if args == ('inspect', 'test'): return '[{}]'
            return ''
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host, 'docker', side_effect=docker), \
                mock.patch.object(host, 'validate_container_inspect'), \
                mock.patch.object(host.time, 'monotonic', side_effect=[0, 2000, 2001]), self.assertRaises(ValueError):
            host.supervise_container(['create'], 'test', Path(temporary) / 'setup', 'setup', 7200)
        self.assertEqual(calls[-1], ('rm', '--force', 'test'))

    def test_docker_stderr_is_positive_allowlisted_bounded_and_digest_bound(self):
        raw = (b"Error: invalid ulimit type fsize=4294967296 --ulimit; max-file=1 max-size=1m "
               b"tmpfs size=268435456 mode=0700 noexec uid=1001 gid=118 "
               b"/home/private/path https://private.example/?token=SECRET123 Authorization: Bearer SECRET456 " + b'x' * 20000)
        error = host.subprocess.CalledProcessError(125, ['docker', 'create', 'private'], stderr=raw)
        with mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=118):
            record = host.public_failure(error, 'create')
        encoded = json.dumps(record)
        for secret in ('SECRET123', 'SECRET456', 'private.example', '/home/private', 'Authorization', 'Bearer'):
            self.assertNotIn(secret, encoded)
        self.assertLess(len(encoded), 4096)
        self.assertEqual(record['returncode'], 125)
        self.assertEqual(record['stderr']['sha256'], host.hashlib.sha256(raw).hexdigest())
        self.assertEqual(record['stderr']['observed_bytes'], len(raw))
        self.assertTrue(record['stderr']['prefix_truncated']); self.assertFalse(record['stderr']['unclassified'])
        for value in ('fsize=4294967296', 'max-file=1', 'max-size=1m', 'size=268435456', 'mode=0700', 'uid=1001', 'gid=118'):
            self.assertIn(value, record['stderr']['fixed_argument_values'])
        unknown = host.public_failure(host.subprocess.CalledProcessError(1, ['docker'], stderr='arbitrary private unknown message'), 'start')
        self.assertTrue(unknown['stderr']['unclassified'])
        self.assertNotIn('arbitrary private unknown message', json.dumps(unknown))

    def test_docker_wrapper_preserves_original_exception_and_attaches_safe_summary(self):
        error = host.subprocess.CalledProcessError(125, ['docker', 'create'], stderr='unknown flag --tmpfs secret-value')
        with mock.patch.object(host.subprocess, 'run', side_effect=error):
            with self.assertRaises(host.subprocess.CalledProcessError) as caught: host.docker('create', 'fixed')
        self.assertIs(caught.exception, error)
        self.assertEqual(error.screen32_diagnostic['operation'], 'create')
        self.assertIn('unknown flag', error.screen32_diagnostic['stderr']['safe_fragments'])
        self.assertNotIn('secret-value', json.dumps(error.screen32_diagnostic))

    def test_docker_captures_binary_stderr_without_newline_normalization(self):
        raw = b'unknown flag --tmpfs\r\nprivate\rdata\xff'
        error = host.subprocess.CalledProcessError(125, ['docker', 'create'], stderr=raw)
        with mock.patch.object(host.subprocess, 'run', side_effect=error) as run:
            with self.assertRaises(host.subprocess.CalledProcessError) as caught: host.docker('create', 'fixed')
        self.assertNotIn('text', run.call_args.kwargs)
        self.assertIs(caught.exception.stderr, raw)
        record = error.screen32_diagnostic['stderr']
        self.assertEqual(record['representation'], 'captured_bytes')
        self.assertEqual(record['observed_bytes'], len(raw))
        self.assertEqual(record['sha256'], host.hashlib.sha256(raw).hexdigest())
        with mock.patch.object(host.subprocess, 'run', return_value=types.SimpleNamespace(stdout=b'{"fixed":true}')):
            self.assertEqual(host.docker('inspect', 'fixed'), '{"fixed":true}')

    def supervise_fixture(self, root, failure=None, cleanup_failure=None, failed_state=None, invalid_config=False):
        output = root / 'setup'; output.mkdir()
        calls = []
        error = host.subprocess.CalledProcessError(125, ['docker', failure or 'none'], stderr='permission denied /private SECRET')
        cleanup = host.subprocess.CalledProcessError(1, ['docker', 'rm'], stderr='no such container private-name')
        def docker(*args, **kwargs):
            calls.append(args)
            record = json.loads((root / 'container-setup.json').read_text())
            if args[0] == 'rm':
                self.assertEqual(record['operations']['cleanup']['status'], 'started')
                if failure is not None or failed_state is not None or invalid_config:
                    self.assertIsNotNone(record['primary_failure'])
                if cleanup_failure: raise cleanup
                return ''
            operation = ('inspect_configuration' if args == ('inspect', 'fixed') else
                         'watch' if args[0] == 'inspect' else args[0])
            self.assertEqual(record['active_operation'], operation)
            self.assertEqual(record['operations'][operation]['status'], 'started')
            if operation == failure: raise error
            if operation == 'inspect_configuration': return '[{}]'
            if operation == 'watch': return json.dumps(failed_state or {'Running': False, 'ExitCode': 0, 'OOMKilled': False})
            return ''
        config = mock.Mock(side_effect=ValueError('fictional invalid configuration')) if invalid_config else mock.Mock()
        with mock.patch.object(host, 'docker', side_effect=docker), mock.patch.object(host, 'validate_container_inspect', config):
            try: result = host.supervise_container(['create'], 'fixed', output, 'setup', host.time.monotonic() + 100)
            except BaseException as caught: result = caught
        return result, json.loads((root / 'container-setup.json').read_text()), calls, error, cleanup

    def test_create_failure_retains_operation_and_does_not_remove_unconfirmed_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, calls, error, cleanup = self.supervise_fixture(Path(temporary), failure='create')
            self.assertIs(result, error); self.assertFalse(record['created_confirmed'])
            self.assertEqual(record['failure_operation'], 'create')
            self.assertEqual(record['operations']['cleanup']['status'], 'not_run')
            self.assertEqual(calls, [('create',)])

    def test_primary_inspect_start_watch_failure_survives_cleanup_failure(self):
        for phase in ('inspect_configuration', 'start', 'watch'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                result, record, calls, error, cleanup = self.supervise_fixture(Path(temporary), failure=phase, cleanup_failure=True)
                self.assertIs(result, error)
                self.assertEqual(record['failure_operation'], phase)
                self.assertEqual(record['primary_failure']['returncode'], 125)
                self.assertEqual(record['cleanup_failure']['returncode'], 1)
                self.assertEqual(calls[-1], ('rm', '--force', 'fixed'))
                self.assertNotIn('SECRET', json.dumps(record)); self.assertNotIn('/private', json.dumps(record))

    def test_validation_or_container_exit_failure_is_retained_before_cleanup(self):
        for options in ({'invalid_config': True}, {'failed_state': {'Running': False, 'ExitCode': 137, 'OOMKilled': True}}):
            with tempfile.TemporaryDirectory() as temporary:
                result, record, calls, error, cleanup = self.supervise_fixture(Path(temporary), cleanup_failure=True, **options)
                self.assertIsInstance(result, ValueError)
                self.assertEqual(record['primary_failure']['error_type'], 'ValueError')
                self.assertEqual(record['cleanup_failure']['returncode'], 1)
                if 'failed_state' in options: self.assertEqual(record['container_state'], options['failed_state'])

    def test_diagnostic_write_failure_does_not_replace_primary_exception(self):
        real_atomic = host.atomic
        def atomic(path, value):
            if value.get('primary_failure') is not None: raise OSError('fictional receipt disk full')
            return real_atomic(path, value)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host, 'atomic', side_effect=atomic):
            result, record, calls, error, cleanup = self.supervise_fixture(Path(temporary), failure='create')
            self.assertIs(result, error)
            self.assertEqual(calls, [('create',)])

    def test_cleanup_only_failure_stops_successful_stage_and_is_not_masked(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, calls, error, cleanup = self.supervise_fixture(Path(temporary), cleanup_failure=True)
            self.assertIs(result, cleanup)
            self.assertIsNone(record['primary_failure']); self.assertEqual(record['status'], 'failed_no_retry')
            self.assertEqual(record['operations']['watch']['status'], 'complete')
            self.assertEqual(record['operations']['cleanup']['status'], 'failed_no_retry')

    def test_successful_stage_retains_complete_operations_and_public_allowlisted_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, record, calls, error, cleanup = self.supervise_fixture(root)
            self.assertEqual(result['status'], 'complete'); self.assertEqual(record['status'], 'complete')
            self.assertTrue(all(row['status'] == 'complete' for row in record['operations'].values()))
            (root / 'host-receipt.json').write_text('{}'); (root / 'tts').mkdir()
            artifact = host.public_evidence(root, 'tts')
            self.assertEqual((artifact / 'container-setup.json').read_bytes(), (root / 'container-setup.json').read_bytes())
            self.assertNotIn('setup/', json.dumps(json.loads((artifact / 'artifact-freeze.json').read_text())['files']))

    def test_primary_projection_preserves_unknown_and_never_creates_human_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = {'schema': 'blind-source-screen32-job-v1', 'clips': [{'audio_id': 'clip-000001',
                   'audio_path': 'audio/' + 'a' * 64 + '.wav', 'wav_sha256': 'a' * 64}]}
            for model in ('qwen06', 'sensevoice'):
                (root / model).mkdir()
                tts.atomic(root / model / 'terminal-outcomes.json', [{'opaque_id': 'clip-000001',
                    'status': 'success', 'raw_text': 'fictional words', 'completeness': 'unknown', 'quality_flags': []}])
            host.freeze_primary(root, job)
            dispute = json.loads((root / 'frozen-disputes.json').read_text())
            self.assertEqual(dispute['unresolved_audio_ids'], ['clip-000001'])
            self.assertEqual(dispute['whisper_audio_ids'], [])
            self.assertFalse(dispute['human_gold']); self.assertFalse(dispute['training_admitted'])
            freeze = json.loads((root / 'primary-raw-freeze.json').read_text())
            self.assertEqual(freeze['primary_raw_sha256'], host.file_hash(root / 'primary-raw.json'))

    def test_truncated_setup_receipt_still_publishes_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'tts').mkdir(); (root / 'runtime').mkdir()
            (root / 'host-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json').write_text('{}')
            (root / 'runtime/setup-receipt.json').write_bytes(b'{"started":')
            artifact = host.public_evidence(root, 'tts')
            summary = json.loads((artifact / 'setup-public.json').read_text())
            self.assertEqual(summary['verification_evidence'], 'UNAVAILABLE_OR_TRUNCATED')
            self.assertTrue((artifact / 'tts/tts-receipt.json').exists())

    def test_known_interrupted_temporaries_do_not_hide_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'tts').mkdir()
            (root / 'host-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json').write_text('{}')
            (root / 'tts/tts-receipt.json.tmp').write_bytes(b'{')
            artifact = host.public_evidence(root, 'tts')
            freeze = json.loads((artifact / 'artifact-freeze.json').read_text())
            self.assertIn('tts/tts-receipt.json', freeze['files'])
            self.assertEqual(freeze['incomplete_temporary_files_excluded'], ['tts/tts-receipt.json.tmp'])
            self.assertFalse((artifact / 'tts/tts-receipt.json.tmp').exists())

    def test_blind_handoff_only_selected_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'out'; output.mkdir()
            pcm = struct.pack('<hh', 100, -100)
            body = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 40, b'WAVE', b'fmt ', 16,
                               1, 1, 16000, 32000, 2, 16, b'data', 4) + pcm
            audio = output / 'screen32-001.wav'; audio.write_bytes(body)
            ledger = tts.initial_ledger(); ledger[0].update(status='GENERATED', attempts=1, audio={
                'wav_sha256': host.file_hash(audio), 'pcm_sha256': host.hashlib.sha256(pcm).hexdigest(),
                'frames': 2, 'sample_rate_hz': 16000, 'channels': 1, 'sample_width_bytes': 2})
            (output / 'private-plan.json').write_text('fictional source cue')
            receipt = host.pack_blind(output, root / 'blind', ledger)
            self.assertEqual(receipt['clips'], 1)
            self.assertEqual(len(receipt['files']), 2)
            self.assertNotIn('screen32-001', (root / 'blind/job.json').read_text())
            self.assertEqual(len(host.validate_blind_input(root / 'blind')['clips']), 1)
            (root / 'blind/leak.json').write_text('{}')
            with self.assertRaises(ValueError): host.validate_blind_input(root / 'blind')


class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.adapter = types.SimpleNamespace(generate_cell=self.generate)
    def generate(self, cell):
        self.calls.append(cell)
        return [0.0], {'native': {'quality_flags': [], 'native_float_values_sha256': 'a' * 64}}
    def writer(self, array, native, cell):
        index = int(cell[-3:]); digest = format(index, '064x')
        return {'wav_sha256': digest, 'pcm_sha256': digest, 'frames': 1, 'sample_rate_hz': 16000,
                'channels': 1, 'sample_width_bytes': 2}, {}
    def run_cells(self, path, writer=None):
        with mock.patch.object(tts, 'verify_runtime_scope'), mock.patch.object(tts.signal, 'alarm'):
            return tts.run_cells(self.adapter, {}, path, writer or self.writer)
    def test_sixteen_once_durable_and_original32_denominator(self):
        with tempfile.TemporaryDirectory() as path:
            receipt = self.run_cells(Path(path))
            self.assertEqual(len(self.calls), 16)
            self.assertEqual(sum(row['attempts'] for row in receipt['ledger']), 16)
            self.assertTrue(all(row['status'] == 'NOT_RUN' for row in receipt['ledger'][16:]))
            with self.assertRaises(ValueError): self.run_cells(Path(path))
            self.assertEqual(len(self.calls), 16)
    def test_writer_failure_consumes_cell_stops_and_retains_partial(self):
        def writer(array, native, cell): raise OSError('fictional output failure')
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(OSError): self.run_cells(Path(path), writer)
            receipt = json.loads((Path(path) / 'tts-receipt.json').read_text())
            self.assertEqual(self.calls, ['screen32-001'])
            self.assertEqual(receipt['ledger'][0]['status'], 'FAILED_NO_RETRY')
            self.assertEqual(receipt['ledger'][0]['attempts'], 1)
            self.assertTrue(all(row['attempts'] == 0 for row in receipt['ledger'][1:]))
            self.assertTrue((Path(path) / 'screen32-001.claim').exists())


if __name__ == '__main__': unittest.main()
