"""Fictional no-model probe fixtures. Never launch Docker or contact a registry."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hosted_run as host


class ProbeHostTests(unittest.TestCase):
    def lock(self):
        return json.loads((HERE / 'container-lock.json').read_text())

    def freeze(self):
        return json.loads((HERE / 'execution-freeze.json').read_text())

    def kernel(self, freeze):
        return {'schema': 'screen32-probe-kernel-v1', 'status': 'complete', 'python_version': '3.12.14',
            'model_calls': 0, 'runtime_installs': 0, 'human_gold': False, 'training_admitted': False,
            'worker_sha256': freeze['files']['research/source-screen32-v1/probe_worker.py'],
            'runtime_scope_sha256': freeze['files']['research/source-screen32-v1/runtime_scope.py'],
            'kernel': {'schema': 'screen32-kernel-scope-v1', 'limits': {'memory.max': str(host.MEMORY),
                'memory.swap.max': '0', 'cpu.max': '400000 100000', 'pids.max': '256'}, 'uid': 1001,
                'private_cgroup_root': True, 'read_only_root': True, 'capabilities': 'none',
                'no_new_privileges': True, 'seccomp_filter': True, 'network_disabled': False,
                'network_interfaces': ['eth0', 'lo']}}

    def image(self):
        lock = self.lock()
        return {'status': 'completed_observed_within_reservation', 'daemon_completion_verified': True,
                'image': lock['image'], 'image_config_digest': lock['manifest']['config']['digest']}

    def diagnostic(self):
        return {'schema': 'screen32-container-diagnostic-v1', 'stage': 'setup', 'status': 'complete',
            'created_confirmed': True, 'primary_failure': None, 'cleanup_failure': None,
            'container_state': {'Running': False, 'ExitCode': 0, 'OOMKilled': False},
            'operations': {name: {'status': 'complete'} for name in ('create', 'inspect_configuration', 'start', 'watch', 'cleanup')}}

    def reference(self, root, mutate=None):
        directory = root / 'research/source-screen32-v1/evidence/probe-1-12345'
        directory.mkdir(parents=True)
        freeze = self.freeze()
        values = {'host-receipt.json': {'schema': 'screen32-probe-host-v1', 'status': 'complete',
            'probe_attempt': 1, 'run_id': '12345', 'run_number': 1, 'run_attempt': 1, 'head_sha': 'a' * 40,
            'source_freeze_sha256': 'b' * 64, 'container_start_attempts': 1, 'model_calls': 0, 'runtime_installs': 0,
            'stages': {'setup': {'status': 'complete'}},
            'container_contract_sha256': host.setup_contract_sha256(self.lock()['image'])},
            'image-transfer.json': self.image(), 'container-setup.json': self.diagnostic(),
            'setup/probe-kernel.json': self.kernel(freeze)}
        if mutate: mutate(values)
        files = {}
        for name, value in values.items():
            path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); host.atomic(path, value)
            files[name] = host.file_hash(path)
        host.atomic(directory / 'artifact-freeze.json', {'schema': 'screen32-evidence-freeze-v1', 'files': files,
            'human_gold': False, 'training_admitted': False, 'incomplete_temporary_files_excluded': []})
        return {'directory': str(directory.relative_to(root)), 'artifact_freeze_sha256': host.file_hash(directory / 'artifact-freeze.json'), 'source_head_sha': 'a' * 40}, freeze

    def test_probe_replaces_only_fixed_setup_tail_and_keeps_prefix(self):
        with mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=118):
            args = ('fixed', self.lock()['image'], Path('/fixture/code'), Path('/fixture/runtime'), Path('/fixture/output'))
            original = host.container_args(*args, 'tts', 'setup')
            probe = host.setup_probe_args(*args)
            self.assertEqual(original[:-10], probe[:-4])
            self.assertEqual(probe[-4:], ['python', '-I', '-B', '/code/research/source-screen32-v1/probe_worker.py'])
            self.assertNotIn('setup_adapter.py', ' '.join(probe))
            digest = host.setup_contract_sha256(args[1])
        with mock.patch.object(host.os, 'getuid', return_value=2002), mock.patch.object(host.os, 'getgid', return_value=222):
            self.assertEqual(host.setup_contract_sha256(args[1]), digest)
        with mock.patch.object(host, 'container_args', return_value=['create', 'arbitrary-command']):
            with self.assertRaises(ValueError): host.setup_probe_args(*args)

    def test_probe_admission_allows_only_two_explicit_identities(self):
        for attempt in (1, 2):
            release = {'schema': 'screen32-probe-release-v1', 'approved': True, 'attempt': attempt, 'source_freeze_sha256': 'a' * 64}
            with mock.patch.object(host, 'bounded_json', return_value=release), mock.patch.object(host, 'verify_first_created_source', return_value={'fixture': True}) as verify:
                freeze, observed = host.verify_probe_admission({'fictional': 'environment'})
                self.assertEqual(observed, attempt)
                self.assertEqual(verify.call_args.args[1:4], ('research/qwen16-container-probe-' + str(attempt), attempt, 'a' * 64))
                self.assertEqual(verify.call_args.args[4:], ('source-screen32-probe.yml', 'probe-workflow.yml'))
        for attempt, approved in ((0, True), (3, True), (True, True), (1, False), (None, False)):
            release = {'schema': 'screen32-probe-release-v1', 'approved': approved, 'attempt': attempt, 'source_freeze_sha256': 'a' * 64}
            with mock.patch.object(host, 'bounded_json', return_value=release), mock.patch.object(host, 'docker') as docker:
                with self.assertRaises(ValueError): host.verify_probe_admission({})
                docker.assert_not_called()

    def test_positive_proof_requires_all_bound_raw_files_and_same_freeze(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=118):
            root = Path(temporary); reference, freeze = self.reference(root)
            with mock.patch.object(host, 'ROOT', root):
                result = host.verify_successful_probe(reference, freeze, 'b' * 64)
                self.assertEqual(result['probe_attempt'], 1)
                with self.assertRaises(ValueError): host.verify_successful_probe(reference, freeze, 'c' * 64)
                with self.assertRaises(ValueError): host.verify_successful_probe(True, freeze, 'b' * 64)
                with self.assertRaises(ValueError): host.verify_successful_probe(None, freeze, 'b' * 64)
                (root / reference['directory'] / 'private-extra.json').write_text('{}')
                with self.assertRaises(ValueError): host.verify_successful_probe(reference, freeze, 'b' * 64)

    def test_failure_wrong_contract_or_kernel_cannot_be_admitted(self):
        mutations = [
            lambda v: v['host-receipt.json'].update(status='stopped_no_retry'),
            lambda v: v['host-receipt.json'].update(container_start_attempts=2),
            lambda v: v['host-receipt.json'].update(container_contract_sha256='c' * 64),
            lambda v: v['host-receipt.json'].update(model_calls=1),
            lambda v: v['container-setup.json']['operations']['cleanup'].update(status='failed_no_retry'),
            lambda v: v['container-setup.json'].update(cleanup_failure={'error_type': 'CalledProcessError'}),
            lambda v: v['image-transfer.json'].update(daemon_completion_verified=False),
            lambda v: v['setup/probe-kernel.json']['kernel']['limits'].update({'memory.swap.max': 'max'}),
            lambda v: v['setup/probe-kernel.json']['kernel'].update(network_disabled=True, network_interfaces=['lo']),
            lambda v: v['setup/probe-kernel.json'].update(worker_sha256='d' * 64),
        ]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=118):
                root = Path(temporary); reference, freeze = self.reference(root, mutate)
                with mock.patch.object(host, 'ROOT', root), self.assertRaises(ValueError):
                    host.verify_successful_probe(reference, freeze, 'b' * 64)

    def test_one_mocked_probe_start_has_no_runtime_install_and_fixed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(host.os, 'getuid', return_value=1001), mock.patch.object(host.os, 'getgid', return_value=118):
            root = Path(temporary); freeze = self.freeze()
            env = {'RUNNER_TEMP': str(root), 'GITHUB_RUN_ID': '12345', 'GITHUB_RUN_NUMBER': '1', 'GITHUB_SHA': 'a' * 40}
            def pull(lock, path): host.atomic(path, self.image()); return self.image()
            def supervise(args, name, output, stage, deadline):
                self.assertEqual(args[-4:], ['python', '-I', '-B', '/code/research/source-screen32-v1/probe_worker.py'])
                self.assertEqual(stage, 'setup')
                host.atomic(output / 'probe-kernel.json', self.kernel(freeze))
                host.atomic(output.parent / 'container-setup.json', self.diagnostic())
                return {'status': 'complete', 'wall_seconds': 0.1, 'exit_code': 0}
            with mock.patch.object(host, 'verify_probe_admission', return_value=(freeze, 1)), \
                    mock.patch.object(host.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 20 * host.GIB})()), \
                    mock.patch.object(host, 'host_capacity', return_value={'fixture': True}), \
                    mock.patch.object(host, 'pull_image', side_effect=pull) as pull_mock, \
                    mock.patch.object(host, 'supervise_container', side_effect=supervise) as start:
                host.run_probe(env); start.assert_called_once(); pull_mock.assert_called_once()
                with self.assertRaises(FileExistsError): host.run_probe(env)
                start.assert_called_once(); pull_mock.assert_called_once()
            result = root / 'qwen16-container-probe-1-12345'
            receipt = json.loads((result / 'host-receipt.json').read_text())
            self.assertEqual(receipt['container_start_attempts'], 1)
            self.assertEqual(receipt['runtime_installs'], 0); self.assertEqual(receipt['model_calls'], 0)
            self.assertEqual(list((result / 'runtime').iterdir()), [])
            self.assertEqual(set(json.loads((result / 'artifact/artifact-freeze.json').read_text())['files']),
                {'host-receipt.json', 'image-transfer.json', 'container-setup.json', 'setup/probe-kernel.json'})

    def test_probe_workflow_is_inert_and_does_not_offer_model_or_retry_entry(self):
        workflow = (HERE / 'probe-workflow.yml').read_text()
        self.assertIn('--phase probe', workflow)
        for unwanted in ('workflow_dispatch', '--phase tts', '--phase asr', 'pip install', 'docker run', 'secrets.'):
            self.assertNotIn(unwanted, workflow)
        for guard in ('github.run_attempt == 1', 'github.run_number == 1', 'github.run_number == 2', 'github.event.created == true'):
            self.assertIn(guard, workflow)
        self.assertEqual(self.freeze()['profiles']['probe'], ['research/source-screen32-v1/probe_worker.py', 'research/source-screen32-v1/runtime_scope.py'])


if __name__ == '__main__': unittest.main()
