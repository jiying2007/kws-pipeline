"""Fourth-gate source-only tests: no pip execution, Docker or model calls."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('whisper_fourth_gate', HERE / 'continue_once.py')
gate = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gate)
from common import canonical, file_hash


class FourthCampaignTests(unittest.TestCase):
    def test_all_three_prior_campaigns_are_exact_and_zero_audio(self):
        record = gate.verify_prior()
        self.assertEqual(record['campaign_attempt'], 4)
        self.assertEqual([row['run_id'] for row in record['prior_runs']],
                         [38105471896, 38107706348, 38112558973])
        self.assertTrue(all(row['audio_attempts'] == 0 for row in record['prior_runs']))
        self.assertIs(record['automatic_retry'], False)
        self.assertIs(record['same_model_and_input'], True)

    def test_prior_provenance_drift_is_rejected(self):
        real = gate.file_hash
        for run in (38105471896, 38107706348, 38112558973):
            with self.subTest(run=run):
                def digest(path):
                    return '0' * 64 if path.parent.name == 'run-' + str(run) else real(path)
                with patch.object(gate, 'file_hash', side_effect=digest):
                    self.assertRaisesRegex(ValueError, 'immutable prior provenance', gate.verify_prior)

    def test_prior_evidence_drift_is_rejected(self):
        real = gate.identity
        for run in (38105471896, 38107706348, 38112558973):
            with self.subTest(run=run):
                def identity(path):
                    if 'run-' + str(run) in path.parts:
                        return {'sha256': '0' * 64, 'bytes': 0}
                    return real(path)
                with patch.object(gate, 'identity', side_effect=identity):
                    self.assertRaisesRegex(ValueError, 'immutable prior bytes', gate.verify_prior)

    def test_current_freeze_is_exact_complete_and_reproducible(self):
        freeze = gate.read_json(HERE / 'execution-freeze.json')
        paths = gate.freeze_builder.source_paths()
        self.assertEqual(freeze['files'], {p.relative_to(gate.ROOT).as_posix(): file_hash(p) for p in paths})
        self.assertEqual(freeze['inference_code'], gate.INFERENCE_CODE)
        self.assertEqual(freeze['setup_code'], gate.SETUP_CODE)
        third = gate.read_json(gate.BASE / 'attempt-3/execution-freeze.json')
        self.assertTrue(set(third['files']) < set(freeze['files']))
        self.assertEqual(len([p for p in freeze['files'] if '/evidence/run-38112558973/' in p]), 16)
        for name in ('execution-freeze.json', 'execution-release.json'):
            self.assertNotIn(str(HERE.relative_to(gate.ROOT) / name), freeze['files'])

    def test_compatibility_policy_manifest_and_tests_are_frozen(self):
        files = gate.read_json(HERE / 'execution-freeze.json')['files']
        for name in ('attempt-3/pip_compatibility.py', 'attempt-3/pip_acquire.py',
                     'attempt-3/test_pip_compatibility.py', 'attempt-3/test_pip_acquire.py',
                     'attempt-3/campaign_host.py', 'attempt-3/test_campaign.py',
                     'pip-compatibility/upstream-source-manifest.json', 'pip-compatibility/README.md'):
            self.assertEqual(files['research/source-screen32-whisper-v1/' + name], file_hash(gate.BASE / name))

    def test_exact_setup_and_blind_inference_bundles(self):
        original = gate.read_json(gate.BASE / 'execution-freeze.json')['blind_code']
        self.assertEqual(gate.INFERENCE_CODE, original)
        self.assertEqual(len(gate.INFERENCE_CODE), 8)
        self.assertEqual(len(gate.SETUP_CODE), 13)
        self.assertTrue(set(gate.INFERENCE_CODE) < set(gate.SETUP_CODE))
        self.assertIn('research/source-screen32-whisper-v1/attempt-3/pip_compatibility.py', gate.SETUP_CODE)
        self.assertFalse(any('/attempt-' in p or '/evidence/' in p or '/pip-compatibility/' in p
                             for p in gate.INFERENCE_CODE))
        self.assertTrue(all('/attempt-3/' in p for p in set(gate.SETUP_CODE) - set(gate.INFERENCE_CODE)))
        with tempfile.TemporaryDirectory() as tmp:
            setup = Path(tmp)
            for name in gate.SETUP_CODE:
                path = setup / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((gate.ROOT / name).read_bytes())
            gate.campaign_host.bounded_host.validate_setup_closure(setup)

    def test_fixed_runtime_source_model_and_audio_identity(self):
        self.assertEqual(file_hash(gate.BASE / 'runtime-lock.json'),
                         '881443c6e66acb99223137bd8342788468f814d0a1c3e73ba33a27da3ad1fe6d')
        self.assertEqual(file_hash(gate.BASE / 'source-lock.json'),
                         '0635fb8759fb56439f59db6956612f78cc8903c239f81f3eb7f9b0c47eb99649')
        self.assertEqual(len(gate.read_json(gate.BASE / 'runtime-lock.json')['files']), 22)
        self.assertEqual(len(gate.read_json(gate.BASE / 'source-lock.json')['files']), 22)
        self.assertEqual(gate.LIMITS['clips'], 16)
        self.assertEqual(gate.LIMITS['new_whisper_calls'], 16)
        self.assertEqual(gate.LIMITS['job_seconds'], 2400)
        self.assertEqual(gate.LIMITS['setup_seconds'], 1200)
        self.assertEqual(gate.LIMITS['download_bytes'], 1536 * 1024**2)

    def test_historical_third_freeze_never_regenerated(self):
        self.assertEqual(file_hash(gate.BASE / 'attempt-3/execution-freeze.json'),
                         '16901d0d47d10486a8754568cb03020a3a155a9034ccc4e522fb796929a48ad4')
        self.assertEqual(file_hash(gate.BASE / 'attempt-3/execution-release.json'),
                         '72db5d50b56a771658842f58ab55fde94d82a81f5f24742aa14fd0e2d54c85cf')
        with patch.object(gate.freeze_builder, 'file_hash', return_value='0' * 64):
            self.assertRaisesRegex(ValueError, 'immutable third freeze', gate.freeze_builder.source_paths)

    def test_public_source_release_and_all_workflows_are_disarmed(self):
        release = gate.read_json(HERE / 'execution-release.json')
        self.assertIs(release['approved'], False)
        self.assertEqual(release['source_freeze_sha256'], file_hash(HERE / 'execution-freeze.json'))
        self.assertFalse((gate.ROOT / gate.WORKFLOW).exists())
        for suffix, directory in (('', gate.BASE), ('-attempt-2', gate.BASE / 'attempt-2'),
                                  ('-attempt-3', gate.BASE / 'attempt-3')):
            self.assertIs(gate.read_json(directory / 'execution-release.json')['approved'], False)
            self.assertFalse((gate.ROOT / ('.github/workflows/source-screen32-whisper-third' + suffix + '-v1.yml')).exists())

    def test_disarmed_main_stops_before_any_runtime_call(self):
        with patch.object(gate.campaign_host, 'execute') as execute, patch.object(gate, 'checkout_head') as head:
            self.assertRaisesRegex(ValueError, 'reviewed fourth activation required', gate.main, {})
            execute.assert_not_called(); head.assert_not_called()

    def test_workflow_is_distinct_one_shot_and_budgeted(self):
        text = (HERE / 'workflow.yml').read_text()
        for value in ('branches: [' + gate.BRANCH + ']', 'timeout-minutes: 45', 'timeout-minutes: 42',
                      'timeout-minutes: 2', 'runs-on: ubuntu-24.04', 'github.event.created == true',
                      'github.run_number == 1', 'github.run_attempt == 1', 'persist-credentials: false',
                      'python3 -I -B research/source-screen32-whisper-v1/attempt-4/continue_once.py',
                      'qwen16-whisper-small-third-attempt-4-terminal-v1'):
            self.assertIn(value, text)
        for value in ('workflow_dispatch:', 'schedule:', 'attempt-3/continue_once.py', 'continue-on-error:', 'secrets.'):
            self.assertNotIn(value, text)

    def fixture(self, root):
        base = root / 'research/source-screen32-whisper-v1'; here = base / 'attempt-4'
        here.mkdir(parents=True)
        for directory in (base, base / 'attempt-2', base / 'attempt-3'):
            directory.mkdir(exist_ok=True)
            (directory / 'execution-release.json').write_bytes(canonical({'approved': False}))
        template = here / 'workflow.yml'; template.write_text('fixed fourth workflow')
        workflow = root / gate.WORKFLOW; workflow.parent.mkdir(parents=True); workflow.write_bytes(template.read_bytes())
        freeze = {'schema': 'whisper-screen-fourth-source-freeze-v1', 'files': {},
                  'inference_code': gate.INFERENCE_CODE, 'setup_code': gate.SETUP_CODE}
        for name in gate.SETUP_CODE:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source')
            freeze['files'][name] = file_hash(path)
        (here / 'execution-freeze.json').write_bytes(canonical(freeze))
        release = {'schema': 'whisper-screen-fourth-execution-release-v1', 'experiment': gate.EXPERIMENT,
                   'campaign_attempt': 4, 'prior_run_ids': [row[0] for row in gate.PRIOR], 'approved': True,
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
        return base, here, env, event, release, freeze

    def admitted(self, root, base, here):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.multiple(gate, ROOT=root, BASE=base, HERE=here))
        stack.enter_context(patch.object(gate.freeze_builder, 'source_paths', return_value=[root / p for p in gate.SETUP_CODE]))
        stack.enter_context(patch.object(gate, 'verify_prior', return_value={'verified': True}))
        stack.enter_context(patch.object(gate, 'checkout_head', return_value='a' * 40))
        return stack

    def test_admission_accepts_only_new_exact_event_and_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, event, _, _ = self.fixture(root)
            with self.admitted(root, base, here):
                self.assertEqual(gate.admission(env)[1], {'verified': True})
                bad_fields = [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_NUMBER', '2'),
                              ('GITHUB_SHA', 'b' * 40), ('GITHUB_SHA', ''), ('GITHUB_RUN_ID', '0'),
                              ('GITHUB_EVENT_NAME', 'workflow_dispatch'), ('GITHUB_REF', 'refs/heads/main'),
                              ('GITHUB_REPOSITORY', 'other/repository'), ('RUNNER_ENVIRONMENT', 'self-hosted'),
                              ('RUNNER_OS', 'Windows')]
                bad_fields.extend(('GITHUB_RUN_ID', str(row[0])) for row in gate.PRIOR)
                for key, value in bad_fields:
                    with self.subTest(key=key, value=value):
                        bad = dict(env); bad[key] = value
                        self.assertRaises(ValueError, gate.admission, bad)
                for key, value in [('created', False), ('created', 1), ('deleted', True), ('before', 'b' * 40),
                                   ('ref', 'refs/heads/main'), ('after', 'b' * 40), ('head_commit', {'id': 'b' * 40}),
                                   ('repository', {'private': True, 'full_name': 'jiying2007/kws-pipeline'}),
                                   ('repository', {'private': False, 'full_name': 'other/repository'})]:
                    with self.subTest(event=key):
                        bad = copy.deepcopy(event); bad[key] = value
                        Path(env['GITHUB_EVENT_PATH']).write_bytes(canonical(bad))
                        self.assertRaises(ValueError, gate.admission, env)

    def test_release_requires_new_exact_schema_identity_and_real_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, _, release, _ = self.fixture(root)
            with self.admitted(root, base, here):
                for key, value in [('schema', 'whisper-screen-third-execution-release-v1'), ('experiment', 'old'),
                                   ('campaign_attempt', 3), ('prior_run_ids', [38105471896, 38107706348]),
                                   ('approved', False), ('approved', 1), ('approved', 'true'),
                                   ('source_freeze_sha256', '0' * 64), ('unexpected', True)]:
                    with self.subTest(key=key, value=value):
                        bad = dict(release); bad[key] = value
                        (here / 'execution-release.json').write_bytes(canonical(bad))
                        self.assertRaises(ValueError, gate.admission, env)

    def test_freeze_requires_exact_schema_membership_and_blind_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, _, release, freeze = self.fixture(root)
            bad_freezes = []
            bad = copy.deepcopy(freeze); bad['schema'] = 'whisper-screen-third-source-freeze-v1'; bad_freezes.append(bad)
            bad = copy.deepcopy(freeze); del bad['files'][gate.SETUP_CODE[-1]]; bad_freezes.append(bad)
            bad = copy.deepcopy(freeze); bad['files']['../outside'] = '0' * 64; bad_freezes.append(bad)
            bad = copy.deepcopy(freeze); bad['inference_code'] = gate.SETUP_CODE; bad_freezes.append(bad)
            bad = copy.deepcopy(freeze); bad['setup_code'] = gate.INFERENCE_CODE; bad_freezes.append(bad)
            bad = copy.deepcopy(freeze); bad['unexpected'] = True; bad_freezes.append(bad)
            with self.admitted(root, base, here):
                for index, bad in enumerate(bad_freezes):
                    with self.subTest(index=index):
                        (here / 'execution-freeze.json').write_bytes(canonical(bad))
                        release['source_freeze_sha256'] = file_hash(here / 'execution-freeze.json')
                        (here / 'execution-release.json').write_bytes(canonical(release))
                        self.assertRaises(ValueError, gate.admission, env)

    def test_source_and_workflow_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, _, _, _ = self.fixture(root)
            with self.admitted(root, base, here):
                for name in gate.SETUP_CODE:
                    path = root / name; original = path.read_bytes(); path.write_text('drift')
                    self.assertRaisesRegex(ValueError, 'source drift', gate.admission, env)
                    path.write_bytes(original)
                (root / gate.WORKFLOW).write_text('drift')
                self.assertRaisesRegex(ValueError, 'workflow drift', gate.admission, env)

    def test_old_active_campaign_or_approved_release_blocks_new_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, here, env, _, _, _ = self.fixture(root)
            with self.admitted(root, base, here):
                for directory, suffix in ((base, ''), (base / 'attempt-2', '-attempt-2'),
                                          (base / 'attempt-3', '-attempt-3')):
                    path = directory / 'execution-release.json'
                    path.write_bytes(canonical({'approved': True}))
                    self.assertRaisesRegex(ValueError, 'prior campaigns disarmed', gate.admission, env)
                    path.write_bytes(canonical({'approved': False}))
                    workflow = root / ('.github/workflows/source-screen32-whisper-third' + suffix + '-v1.yml')
                    workflow.write_text('active old workflow')
                    self.assertRaisesRegex(ValueError, 'prior campaigns disarmed', gate.admission, env)
                    workflow.unlink()

    def test_main_delegates_once_only_after_admission(self):
        frozen = {'frozen': True}; campaign = {'campaign_attempt': 4}
        with patch.object(gate, 'admission', return_value=(frozen, campaign)) as admission, \
             patch.object(gate.campaign_host, 'execute', return_value=1) as execute:
            self.assertEqual(gate.main({'source': 'fixture'}), 1)
            admission.assert_called_once_with({'source': 'fixture'})
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0:2], ({'source': 'fixture'}, frozen))
            self.assertEqual(execute.call_args.args[3:],
                             (gate.EXPERIMENT, file_hash(HERE / 'execution-freeze.json'), campaign))


if __name__ == '__main__':
    unittest.main()
