"""Retained real public bytes plus adversarial stdlib-only fixtures. No model/Docker calls."""
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hosted_run as host
import sense_continuation as continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.lock = continuation.load_lock(host)
        for name in ('docker', 'pull_image', 'supervise_container'):
            patch = mock.patch.object(host, name, side_effect=AssertionError('No runtime calls in these tests'))
            patch.start(); self.addCleanup(patch.stop)

    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); here = root / 'research/source-screen32-v1'
            here.mkdir(parents=True)
            for name in ('sense_continuation.py', 'continuation-lock.json', 'hosted_run.py',
                         'runtime_scope.py', 'probe_worker.py', 'setup_adapter.py', 'container-lock.json'):
                shutil.copyfile(HERE / name, here / name)
            for name in ('evidence/run-38018029787', 'evidence/probe-2-38017741858',
                         'history/executed-v3-621ba90'):
                shutil.copytree(HERE / name, here / name)
            workflow = 'name: Inert fixture\non: push\n'
            (here / 'continuation-workflow.yml').write_text(workflow)
            active = root / '.github/workflows/source-screen32-sense-continuation-v1.yml'
            active.parent.mkdir(parents=True); active.write_text(workflow)
            freeze = {'schema': 'screen32-execution-source-freeze-v1', 'files': {}, 'profiles': {'asr': []}}
            for path in here.iterdir():
                if path.is_file(): freeze['files'][path.relative_to(root).as_posix()] = host.file_hash(path)
            (here / 'execution-freeze.json').write_text(json.dumps(freeze))
            release = {'schema': 'screen32-sense-continuation-release-v1', 'experiment': continuation.EXPERIMENT,
                       'approved': True, 'source_freeze_sha256': host.file_hash(here / 'execution-freeze.json')}
            (here / 'continuation-release.json').write_text(json.dumps(release))
            env = {'GITHUB_REPOSITORY': 'jiying2007/kws-pipeline', 'GITHUB_REF': 'refs/heads/' + continuation.BRANCH,
                   'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_NUMBER': '1',
                   'GITHUB_RUN_ID': '12345', 'GITHUB_SHA': 'a' * 40, 'GITHUB_EVENT_PATH': str(root / 'event.json')}
            event = {'created': True, 'deleted': False, 'after': env['GITHUB_SHA'],
                     'repository': {'private': False, 'full_name': env['GITHUB_REPOSITORY']}}
            (root / 'event.json').write_text(json.dumps(event))
            with mock.patch.object(host, 'ROOT', root), mock.patch.object(host, 'HERE', here), \
                    mock.patch.object(host.os, 'getuid', return_value=1001), \
                    mock.patch.object(host.os, 'getgid', return_value=1001):
                yield root, here, env

    def test_genuine_complete_retained_sources_and_literal_job(self):
        job, provenance = continuation.verify_retained(self.lock, host)
        self.assertEqual(len(job['clips']), 16)
        self.assertEqual(len(provenance['archives']['generation']['exact_members']), 58)
        self.assertEqual(len(provenance['archives']['asr']['exact_members']), 108)
        self.assertEqual(self.lock['reference_sha256']['blind/job.json'],
                         '425a705515339ff7054e4f1148971a342e5f882a1e9b84ac2be144a3d7aeecc5')
        self.assertEqual([c['prior_sensevoice_attempts'] for c in self.lock['clips']], [1] + [0] * 15)

    def test_genuine_historical_probe_uses_original_freeze_and_limited_claim(self):
        original_verify = host.verify_successful_probe
        with mock.patch.object(host, 'verify_successful_probe', wraps=original_verify) as verify:
            compatibility = continuation.verify_compatibility(self.lock, host)
        arguments = verify.call_args.args
        self.assertEqual(len(arguments[1]['files']), 66)
        self.assertEqual(arguments[2], 'abfc1121ed82367eb8d18fb79ad28911a8453b867be49298ac3cfd93e5ebf2f2')
        self.assertEqual(arguments[0]['artifact_freeze_sha256'],
                         '16367bda2b3873c8c14bb11ff36f4a4f78570e51c87844b6712108559b996f4f')
        self.assertEqual(compatibility['asr_setup_input_bytes'], 3_340_966_447)
        self.assertFalse(compatibility['new_whole_source_freeze_probed'])
        self.assertFalse(compatibility['new_probe_executed'])

    def test_literal_lock_rejects_input_mapping_attempt_and_probe_drift(self):
        mutations = [lambda x: x.update(reference_directory='elsewhere'),
                     lambda x: x['clips'][0].update(producer_id='screen32-001'),
                     lambda x: x['clips'][0].update(wav_sha256='0' * 64),
                     lambda x: x['clips'][0].update(prior_sensevoice_attempts=0),
                     lambda x: x['limits'].update(new_sensevoice_calls=17),
                     lambda x: x['probe'].update(original_freeze_sha256='0' * 64)]
        for mutate in mutations:
            changed = copy.deepcopy(self.lock); mutate(changed)
            with self.assertRaisesRegex(ValueError, 'literal continuation lock identity'):
                continuation.verify_retained(changed, host)
        with self.fixture() as (_, here, _):
            with (here / 'continuation-lock.json').open('ab') as stream: stream.write(b' ')
            with self.assertRaisesRegex(ValueError, 'literal continuation lock bytes'):
                continuation.load_lock(host)

    def test_original_byte_drift_and_extra_files_rejected(self):
        with self.fixture() as (root, _, _):
            base = root / continuation.REFERENCE
            for name in ('generation/tts/screen32-010.wav', 'asr/qwen06/qwen06/clip-000001.decoder.json',
                         'asr/qwen06/terminal-outcomes.json', 'asr/sensevoice/terminal-outcomes.json', 'blind/job.json'):
                path = base / name; original = path.read_bytes(); path.write_bytes(original + b' ')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    continuation.verify_retained(self.lock, host)
                path.write_bytes(original)
            for name in ('generation/extra.json', 'asr/qwen06/extra.json', 'blind/extra.json'):
                path = base / name; path.write_text('{}')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'exact retained member set'):
                    continuation.verify_retained(self.lock, host)
                path.unlink()
            (base / 'blind/empty-extra').mkdir()
            with self.assertRaisesRegex(ValueError, 'exact retained member set'):
                continuation.verify_retained(self.lock, host)

    def test_symlink_and_hardlink_aliases_rejected(self):
        with self.fixture() as (root, _, _):
            path = root / continuation.REFERENCE / self.lock['clips'][0]['producer_path']
            outside = root / 'original.wav'; path.rename(outside); path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'aliases'):
                continuation.verify_retained(self.lock, host)
            path.unlink(); path.hardlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'unaliased'):
                continuation.verify_retained(self.lock, host)

    def test_repaired_source_is_not_claimed_as_historical_whole_probe(self):
        with self.fixture() as (_, here, _):
            path = here / 'setup_adapter.py'; path.write_text(path.read_text() + '\n# drift\n')
            with self.assertRaisesRegex(ValueError, 'unchanged setup/kernel'):
                continuation.verify_compatibility(self.lock, host)
        with self.fixture() as (_, here, _):
            path = here / 'hosted_run.py'
            path.write_text(path.read_text().replace("'--cpus', '4'", "'--cpus', '8'"))
            with self.assertRaisesRegex(ValueError, 'host safety AST identity: container_args'):
                continuation.verify_compatibility(self.lock, host)
        with self.fixture() as (_, here, _):
            path = here / 'evidence/probe-2-38017741858/setup/probe-kernel.json'
            path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'probe raw evidence hash'):
                continuation.verify_compatibility(self.lock, host)

    def test_default_release_false_stops_before_admission_or_acquisition(self):
        with self.fixture() as (_, here, env):
            release = host.bounded_json(here / 'continuation-release.json')
            release['approved'] = False
            (here / 'continuation-release.json').write_text(json.dumps(release))
            with mock.patch.object(host, 'verify_first_created_source') as first:
                with self.assertRaisesRegex(ValueError, 'exact reviewed continuation admission'):
                    continuation.verify_admission(env, host)
            first.assert_not_called()

    def test_unique_first_branch_identity_and_host_only_freeze(self):
        with self.fixture() as (root, here, env):
            freeze, lock, compatibility = continuation.verify_admission(env, host)
            self.assertEqual(lock, self.lock)
            self.assertEqual(compatibility['status'], 'limited_setup_kernel_compatibility_verified')
            for key, value in (('GITHUB_REF', 'refs/heads/research/qwen16-voicedesign-once-v3'),
                               ('GITHUB_RUN_NUMBER', '2'), ('GITHUB_RUN_ATTEMPT', '2'),
                               ('GITHUB_EVENT_NAME', 'workflow_dispatch'), ('GITHUB_REPOSITORY', 'someone/fork')):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'one-shot workflow identity'):
                    continuation.verify_admission({**env, key: value}, host)
            event = host.bounded_json(root / 'event.json'); event['created'] = False
            (root / 'event.json').write_text(json.dumps(event))
            with self.assertRaisesRegex(ValueError, 'first public branch creation'):
                continuation.verify_admission(env, host)
            event['created'] = True; (root / 'event.json').write_text(json.dumps(event))
            freeze['profiles']['asr'].append('research/source-screen32-v1/continuation-lock.json')
            (here / 'execution-freeze.json').write_text(json.dumps(freeze))
            release = host.bounded_json(here / 'continuation-release.json')
            release['source_freeze_sha256'] = host.file_hash(here / 'execution-freeze.json')
            (here / 'continuation-release.json').write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, 'host-only'):
                continuation.verify_admission(env, host)

    def test_prepare_copies_exact_original_job_and_only_opaque_wavs(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'input'
            job = continuation.prepare_input(destination, self.lock, host)
            self.assertEqual((destination / 'job.json').read_bytes(),
                             (host.ROOT / continuation.REFERENCE / 'blind/job.json').read_bytes())
            self.assertEqual({p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()},
                             {'job.json'} | {r['audio_path'] for r in job['clips']})
            self.assertEqual(host.validate_blind_input(destination), job)
            with self.assertRaisesRegex(ValueError, 'fresh blind destination'):
                continuation.prepare_input(destination, self.lock, host)
            (destination / 'producer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'only audio and blind job'):
                host.validate_blind_input(destination)

    def test_qwen_reuse_is_byte_exact_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = continuation.retain_prior_qwen(root, self.lock, host)
            original = host.ROOT / continuation.REFERENCE / 'asr/qwen06'
            for path in original.rglob('*'):
                if path.is_file(): self.assertEqual(path.read_bytes(), (root / 'qwen06' / path.relative_to(original)).read_bytes())
            self.assertTrue(value['reused_qwen']['byte_exact'])
            self.assertEqual(value['new_qwen_calls'], 0)
            self.assertEqual(host.bounded_json(root / 'continuation-provenance.json'), value)
            with self.assertRaisesRegex(ValueError, 'fresh Qwen reuse'):
                continuation.retain_prior_qwen(root, self.lock, host)

    def make_terminal(self, root, job, statuses):
        (root / 'sensevoice').mkdir(exist_ok=True)
        rows = []
        for clip, status in zip(job['clips'], statuses):
            rows.append({'opaque_id': clip['audio_id'], 'wav_sha256': clip['wav_sha256'], 'status': status,
                         'raw_text': 'unaltered raw output' if status == 'success' else None,
                         'started_marker_present': status != 'not_run',
                         'execution_receipt_sha256': 'a' * 64 if status != 'not_run' else None,
                         'validation': 'validated_receipt' if status == 'success' else 'untouched',
                         'completeness': 'unknown', 'quality_flags': [], 'human_gold': False, 'training_admitted': False})
        (root / 'sensevoice/terminal-outcomes.json').write_text(json.dumps(rows))
        return rows

    def test_full_denominator_and_zero_partial_or_sixteen_new_calls(self):
        cases = [(['not_run'] * 16, 0), (['success', 'failed_no_retry'] + ['not_run'] * 14, 2),
                 (['success'] * 16, 16)]
        for statuses, count in cases:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); job, _ = continuation.verify_retained(self.lock, host)
                continuation.retain_prior_qwen(root, self.lock, host)
                self.make_terminal(root, job, statuses)
                result = continuation.finalize_cross_run(root, job, self.lock, host)
                self.assertEqual(result['denominator'], 16)
                self.assertEqual(len(result['rows']), 16)
                self.assertEqual(result['new_sensevoice_calls'], count)
                self.assertEqual(result['cumulative_sensevoice_calls'], count + 1)
                self.assertEqual([r['prior_attempts'] for r in result['rows']], [1] + [0] * 15)
                self.assertEqual([r['cumulative_attempt_cap'] for r in result['rows']], [2] + [1] * 15)
                self.assertFalse(result['human_gold']); self.assertFalse(result['training_admitted'])
                self.assertEqual(result['acoustic_completeness'], 'UNKNOWN')
                self.assertEqual([r['new_status'] for r in result['rows']], statuses)
                self.assertEqual(host.bounded_json(root / 'continuation-ledger.json'), result)
                with self.assertRaises(FileExistsError):
                    continuation.finalize_cross_run(root, job, self.lock, host)

    def test_finalize_rejects_new_attempt_after_failure_or_untouched_gap(self):
        cases = [['failed_no_retry', 'success'], ['not_run', 'success'],
                 ['failed_no_retry', 'failed_no_retry'], ['not_run', 'failed_no_retry']]
        for prefix in cases:
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); job, _ = continuation.verify_retained(self.lock, host)
                continuation.retain_prior_qwen(root, self.lock, host)
                self.make_terminal(root, job, prefix + ['not_run'] * 14)
                with self.assertRaisesRegex(ValueError, 'stops at first failure or untouched clip'):
                    continuation.finalize_cross_run(root, job, self.lock, host)
                self.assertFalse((root / 'continuation-ledger.json').exists())

    def test_finalize_rejects_missing_rows_drift_or_label_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _ = continuation.verify_retained(self.lock, host)
            continuation.retain_prior_qwen(root, self.lock, host)
            for mutation in (lambda rows: rows.pop(), lambda rows: rows[0].update(wav_sha256='0' * 64),
                             lambda rows: rows[0].update(human_gold=True),
                             lambda rows: rows[0].update(status='success'),
                             lambda rows: rows[0].update(started_marker_present=True)):
                rows = self.make_terminal(root, job, ['not_run'] * 16); mutation(rows)
                (root / 'sensevoice/terminal-outcomes.json').write_text(json.dumps(rows))
                with self.assertRaises(ValueError): continuation.finalize_cross_run(root, job, self.lock, host)
            self.make_terminal(root, job, ['not_run'] * 16)
            bad_job = copy.deepcopy(job); bad_job['clips'].reverse()
            with self.assertRaisesRegex(ValueError, 'fixed continuation job'):
                continuation.finalize_cross_run(root, bad_job, self.lock, host)
            path = root / 'qwen06/terminal-outcomes.json'; path.write_text(path.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'retained member bytes/hash drift'):
                continuation.finalize_cross_run(root, job, self.lock, host)

    def test_both_historical_freezes_reconstruct_all_original_bytes(self):
        for name, count, digest in (
            ('executed-v3-621ba90', 66, 'abfc1121ed82367eb8d18fb79ad28911a8453b867be49298ac3cfd93e5ebf2f2'),
            ('asr-repair-fd3049f', 69, '8a400de6dac99cc4cb0834e50492d2bfdb1307069257f2ac6d9f1fd878ac9ce7')):
            with self.subTest(snapshot=name):
                base = HERE / 'history' / name
                self.assertEqual(host.file_hash(base / 'execution-freeze.json'), digest)
                freeze = host.bounded_json(base / 'execution-freeze.json')
                snapshot = host.bounded_json(base / 'snapshot.json')
                self.assertEqual(len(freeze['files']), count)
                self.assertEqual(snapshot['source_freeze_sha256'], digest)
                self.assertEqual(set(snapshot['overrides']) | set(snapshot['unchanged_source_paths']), set(freeze['files']))
                self.assertFalse(set(snapshot['overrides']) & set(snapshot['unchanged_source_paths']))
                for path, sha in freeze['files'].items():
                    source = host.ROOT / snapshot['overrides'][path]['snapshot_path'] if path in snapshot['overrides'] else host.ROOT / path
                    self.assertEqual(host.file_hash(source), sha, path)

    def test_fixed_template_has_one_sense_only_command_and_no_runtime_selector(self):
        raw = (HERE / 'continuation-workflow.yml').read_text()
        self.assertIn('branches: [research/qwen16-sensevoice-continuation-v1]', raw)
        self.assertEqual([line for line in raw.splitlines() if line.startswith('        run: ')],
            ['        run: python3 -I -B research/source-screen32-v1/hosted_run.py --phase sense-continuation'])
        self.assertEqual([line for line in raw.splitlines() if line.startswith('  ') and not line.startswith('   ') and line.endswith(':')],
                         ['  push:', '  sensevoice:'])
        for required in ('github.event.created == true', 'github.run_number == 1', 'github.run_attempt == 1',
                         'persist-credentials: false', 'runs-on: ubuntu-24.04', 'permissions:', '  contents: read'):
            self.assertIn(required, raw)
        for forbidden in ('workflow_dispatch', 'schedule:', 'download-artifact', '--input', '--model', 'secrets.'):
            self.assertNotIn(forbidden, raw)

    def test_module_has_no_execution_or_acquisition_entrypoint(self):
        tree = ast.parse((HERE / 'sense_continuation.py').read_text())
        imports = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {'subprocess', 'requests', 'urllib', 'torch', 'numpy', 'funasr'})
        self.assertNotIn('main', {node.name for node in tree.body if isinstance(node, ast.FunctionDef)})


if __name__ == '__main__':
    unittest.main()
