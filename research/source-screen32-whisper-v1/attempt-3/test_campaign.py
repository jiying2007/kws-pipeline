"""Offline stdlib contract tests. No Docker, package execution, or model calls."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continue_once as gate
import campaign_host as runner
prepare_freeze = gate.freeze_builder
from common import canonical, file_hash, BOUNDARY

# Source-only future-bundle fixture; never rewrite the historical third gate.
CURRENT_SETUP_FIXTURE = sorted(set(gate.SETUP_CODE) | {
    'research/source-screen32-whisper-v1/attempt-3/pip_compatibility.py'})


class CampaignTests(unittest.TestCase):
    def test_prior_archives_are_exact_and_zero_attempt(self):
        campaign = gate.verify_prior()
        self.assertEqual([r['run_id'] for r in campaign['prior_runs']], [38105471896, 38107706348])
        self.assertTrue(all(r['audio_attempts'] == 0 for r in campaign['prior_runs']))
        self.assertFalse(campaign['automatic_retry'])

    def test_prior_modified_provenance_fails(self):
        with patch.object(gate, 'file_hash', return_value='0' * 64):
            self.assertRaises(ValueError, gate.verify_prior)

    def test_source_closure_and_exact_inference_membership(self):
        paths = {p.relative_to(gate.ROOT).as_posix() for p in prepare_freeze.source_paths()}
        old = gate.read_json(gate.BASE / 'attempt-2/execution-freeze.json')
        self.assertTrue(set(old['files']) <= paths)
        self.assertEqual(gate.INFERENCE_CODE, gate.read_json(gate.BASE / 'execution-freeze.json')['blind_code'])
        self.assertTrue(set(gate.SETUP_CODE) <= paths)
        self.assertEqual(len(gate.INFERENCE_CODE), 8)
        self.assertFalse(any('/attempt-' in p or '/evidence/' in p for p in gate.INFERENCE_CODE))
        self.assertNotIn(str(HERE.relative_to(gate.ROOT)) + '/execution-freeze.json', paths)
        self.assertNotIn(str(HERE.relative_to(gate.ROOT)) + '/execution-release.json', paths)

    def fixture(self, root):
        base = root / 'research/source-screen32-whisper-v1'; here = base / 'attempt-3'; here.mkdir(parents=True)
        (base / 'attempt-2').mkdir()
        for directory in (base, base / 'attempt-2'):
            (directory / 'execution-release.json').write_bytes(canonical({'approved': False}))
        template = here / 'workflow.yml'; template.write_text('fixed workflow')
        workflow = root / gate.WORKFLOW; workflow.parent.mkdir(parents=True); workflow.write_bytes(template.read_bytes())
        freeze = {'schema': 'whisper-screen-third-source-freeze-v1',
                  'files': {p: 'd' * 64 for p in gate.SETUP_CODE},
                  'inference_code': gate.INFERENCE_CODE, 'setup_code': gate.SETUP_CODE}
        for p in gate.SETUP_CODE:
            path = root / p; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source')
            freeze['files'][p] = file_hash(path)
        (here / 'execution-freeze.json').write_bytes(canonical(freeze))
        release = {'schema': 'whisper-screen-third-execution-release-v1', 'experiment': gate.EXPERIMENT,
            'campaign_attempt': 3, 'prior_run_ids': [38105471896, 38107706348], 'approved': True,
            'source_freeze_sha256': file_hash(here / 'execution-freeze.json')}
        (here / 'execution-release.json').write_bytes(canonical(release))
        event = {'created': True, 'deleted': False, 'before': '0' * 40, 'after': 'a' * 40,
                 'ref': 'refs/heads/' + gate.BRANCH, 'head_commit': {'id': 'a' * 40},
                 'repository': {'private': False, 'full_name': 'jiying2007/kws-pipeline'}}
        event_path = root / 'event.json'; event_path.write_bytes(canonical(event))
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + gate.BRANCH,
            'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
            'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux', 'GITHUB_SHA': 'a' * 40,
            'GITHUB_RUN_ID': '99999999', 'GITHUB_EVENT_PATH': str(event_path)}
        return base, here, env, event, release

    def test_admission_and_negative_identity_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, event, release = self.fixture(root)
            with patch.multiple(gate, ROOT=root, BASE=base, HERE=here), \
                 patch.object(prepare_freeze, 'source_paths', return_value=[root / p for p in gate.SETUP_CODE]), \
                 patch.object(gate, 'verify_prior', return_value={'verified': True}), \
                 patch.object(gate, 'checkout_head', return_value='a' * 40):
                self.assertEqual(gate.admission(env)[1], {'verified': True})
                for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_NUMBER', '2'),
                    ('GITHUB_RUN_ID', '38107706348'), ('GITHUB_SHA', 'b' * 40), ('RUNNER_ENVIRONMENT', 'self-hosted')]:
                    bad = dict(env); bad[key] = value
                    self.assertRaises(ValueError, gate.admission, bad)
                for key, value in [('created', False), ('deleted', True), ('before', 'b' * 40), ('ref', 'refs/heads/main')]:
                    bad = dict(event); bad[key] = value
                    Path(env['GITHUB_EVENT_PATH']).write_bytes(canonical(bad))
                    self.assertRaises(ValueError, gate.admission, env)
                Path(env['GITHUB_EVENT_PATH']).write_bytes(canonical(event))
                (root / gate.INFERENCE_CODE[0]).write_text('drift')
                self.assertRaises(ValueError, gate.admission, env)
                (root / gate.INFERENCE_CODE[0]).write_text('source')
                (root / gate.WORKFLOW).write_text('workflow drift')
                self.assertRaises(ValueError, gate.admission, env)
                (root / gate.WORKFLOW).write_text('fixed workflow')
                release['approved'] = False
                (here / 'execution-release.json').write_bytes(canonical(release))
                self.assertRaises(ValueError, gate.admission, env)

    def test_failed_setup_never_calls_inference_and_preserves_sixteen(self):
        host = Mock(); host.host_capacity.return_value = {'cpu_count': 4}
        fake = types.ModuleType('sense_continuation')
        fake.load_lock = lambda h: {}
        fake.verify_retained = lambda *a: None
        fake.prepare_input = lambda inputs, *a: inputs.mkdir()
        rows = [{'status': 'not_run', 'attempts': 0, 'validation_errors': []} for _ in range(16)]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'sense_continuation': fake}), \
             patch.object(runner.original, 'historical_host', return_value=host), \
             patch.object(runner.original, 'terminal_rows', return_value=rows), \
             patch.object(runner.bounded_host, 'acquire_runtime', create=True, side_effect=ValueError('fixed failure')) as acquire, \
             patch.object(runner.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=20 * 1024**3)), \
             patch.object(runner, 'collect_public'), patch.object(runner.signal, 'signal'):
            env = {'RUNNER_TEMP': tmp, 'GITHUB_RUN_ID': '99999999', 'GITHUB_SHA': 'a' * 40}
            rc = runner.execute(env, {'inference_code': gate.INFERENCE_CODE, 'setup_code': CURRENT_SETUP_FIXTURE},
                                runner.time.monotonic(), gate.EXPERIMENT, 'f' * 64, gate.verify_prior())
            self.assertEqual(rc, 1); acquire.assert_called_once(); host.supervise_container.assert_not_called()
            root = Path(tmp) / (gate.EXPERIMENT + '-99999999')
            receipt = gate.read_json(root / 'host-receipt.json')
            self.assertEqual(receipt['attempts'], 0); self.assertEqual(receipt['completed'], 0)
            self.assertEqual(receipt['stages'], {'setup': 'failed_no_retry', 'whisper': 'not_run'})
            self.assertEqual(len(gate.read_json(root / 'terminal-outcomes.json')), 16)
            self.assertEqual({p.relative_to(root / 'code').as_posix() for p in (root / 'code').rglob('*') if p.is_file()}, set(gate.INFERENCE_CODE))
            self.assertEqual({p.relative_to(root / 'setup-code').as_posix() for p in (root / 'setup-code').rglob('*') if p.is_file()}, set(CURRENT_SETUP_FIXTURE))

    def test_success_uses_original_offline_constructor_after_setup(self):
        historical = runner.original.historical_host()
        host = Mock(); host.container_args.side_effect = historical.container_args
        host.host_capacity.return_value = {'cpu_count': 4}
        fake = types.ModuleType('sense_continuation')
        fake.load_lock = lambda h: {}; fake.verify_retained = lambda *a: None
        fake.prepare_input = lambda inputs, *a: inputs.mkdir()
        rows = [{'status': 'completed_native_eot', 'attempts': 1, 'validation_errors': []} for _ in range(16)]
        order = []
        host.supervise_container.side_effect = lambda *a: order.append('inference')
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'sense_continuation': fake}), \
             patch.object(runner.original, 'historical_host', return_value=host), \
             patch.object(runner.original, 'terminal_rows', return_value=rows), \
             patch.object(runner.bounded_host, 'acquire_runtime', create=True, side_effect=lambda **k: order.append('setup')), \
             patch.object(runner.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=20 * 1024**3)), \
             patch.object(runner, 'collect_public'), patch.object(runner.signal, 'signal'):
            env = {'RUNNER_TEMP': tmp, 'GITHUB_RUN_ID': '99999999', 'GITHUB_SHA': 'a' * 40}
            rc = runner.execute(env, {'inference_code': gate.INFERENCE_CODE, 'setup_code': CURRENT_SETUP_FIXTURE},
                                runner.time.monotonic(), gate.EXPERIMENT, 'f' * 64, gate.verify_prior())
            self.assertEqual(rc, 0); self.assertEqual(order, ['setup', 'inference'])
            args = host.supervise_container.call_args.args[0]
            self.assertEqual(args[args.index('--network') + 1], 'none')
            self.assertEqual(args[-4:], ['/runtime/venv/bin/python', '-I', '-B',
                                        '/code/research/source-screen32-whisper-v1/worker.py'])
            self.assertFalse(any('/setup-code' in arg for arg in args))
            self.assertTrue(any('dst=/runtime,readonly' in arg for arg in args))
            self.assertTrue(any('dst=/input,readonly' in arg for arg in args))
            root = Path(tmp) / (gate.EXPERIMENT + '-99999999')
            receipt = gate.read_json(root / 'host-receipt.json')
            self.assertEqual(receipt['attempts'], 16); self.assertEqual(receipt['completed'], 16)
            self.assertFalse(receipt['training_admitted']); self.assertFalse(receipt['shipping_approved'])

    def test_missing_current_helper_fails_before_image_or_docker(self):
        host = Mock()
        fake = types.ModuleType('sense_continuation')
        fake.load_lock = lambda h: {}; fake.verify_retained = lambda *a: None
        fake.prepare_input = lambda inputs, *a: inputs.mkdir()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'sense_continuation': fake}), \
             patch.object(runner.original, 'historical_host', return_value=host), \
             patch.object(runner.bounded_host, 'acquire_runtime') as acquire:
            env = {'RUNNER_TEMP': tmp, 'GITHUB_RUN_ID': '99999999', 'GITHUB_SHA': 'a' * 40}
            with self.assertRaises(FileNotFoundError):
                runner.execute(env, {'inference_code': gate.INFERENCE_CODE,
                                     'setup_code': gate.SETUP_CODE},
                    runner.time.monotonic(), gate.EXPERIMENT, 'f' * 64)
            host.host_capacity.assert_not_called()
            host.pull_image.assert_not_called()
            host.docker.assert_not_called()
            host.supervise_container.assert_not_called()
            acquire.assert_not_called()
            root = Path(tmp) / (gate.EXPERIMENT + '-99999999')
            self.assertEqual({p.relative_to(root / 'code').as_posix()
                              for p in (root / 'code').rglob('*') if p.is_file()},
                             set(gate.INFERENCE_CODE))

    def test_broker_seal_adapter_with_real_historical_allowlist(self):
        historical = runner.original.historical_host()
        adapter = runner.CampaignDocker(historical, '99999999')
        broker = 'kws-bound-99999999-wheel-01-broker'
        result = types.SimpleNamespace(stdout=b'ok', stderr=b'', returncode=0)
        with patch.object(runner.subprocess, 'run', return_value=result) as process:
            self.assertRaises(ValueError, historical.docker, 'kill', '--signal=USR1', broker)
            self.assertRaises(ValueError, adapter, 'kill', '--signal=USR1', broker)
            process.assert_not_called()
            adapter('create', '--name', broker)
            for command in [('kill', '--signal=TERM', broker), ('kill', '--signal=USR1', 'unknown'),
                            ('kill', '--signal=USR1', 'kws-bound-99999999-wheel-02-broker'), ('exec', broker)]:
                before = process.call_count
                self.assertRaises(ValueError, adapter, *command)
                self.assertEqual(process.call_count, before)
            self.assertEqual(adapter('kill', '--signal=USR1', broker, timeout=5), 'ok')
            self.assertEqual(process.call_args.args[0], ['docker', 'kill', '--signal=USR1', broker])
            self.assertEqual(process.call_args.kwargs['timeout'], 5)
            self.assertRaises(ValueError, adapter, 'kill', '--signal=USR1', broker)
            adapter('rm', '--force', broker)
            self.assertIsNone(adapter.current)
            self.assertRaises(ValueError, adapter, 'kill', '--signal=USR1', broker)
            adapter('create', '--name', 'kws-bound-99999999-wheel-02-broker')
            self.assertRaises(ValueError, adapter, 'create', '--name', broker)

    def test_uncertain_broker_create_requires_cleanup_not_signal(self):
        historical = runner.original.historical_host()
        adapter = runner.CampaignDocker(historical, '99999999')
        broker = 'kws-bound-99999999-wheel-01-broker'
        with patch.object(runner.subprocess, 'run', side_effect=OSError('private error')):
            self.assertRaises(OSError, adapter, 'create', '--name', broker)
        self.assertEqual(adapter.current, broker)
        self.assertFalse(adapter.confirmed)
        self.assertRaises(ValueError, adapter, 'kill', '--signal=USR1', broker)
        with patch.object(runner.subprocess, 'run', return_value=types.SimpleNamespace(stdout=b'', stderr=b'')):
            adapter('rm', '--force', broker)
        self.assertIsNone(adapter.current)

    def test_public_collection_omits_packages_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'input').mkdir(); (root / 'setup').mkdir()
            (root / 'input/job.json').write_bytes(canonical({'clips': []}))
            (root / 'host-receipt.json').write_bytes(canonical({'status': 'failed_no_retry'}))
            for name in ('small.pt', 'package.whl', 'secret.log', 'unexpected.json'):
                (root / 'setup' / name).write_text('private')
            runner.collect_public(root)
            files = gate.read_json(root / 'artifact/artifact-freeze.json')['files']
            self.assertEqual(set(files), {'host-receipt.json'})


if __name__ == '__main__':
    unittest.main()
