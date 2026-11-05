#!/usr/bin/env python3
"""Synthetic local Git tests; no GitHub calls, real branch deletes or models."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('research_ci_changes', ROOT / 'tools/research_ci_changes.py')
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)


class ChangeSelectionTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='research-changes-')
        self.addCleanup(self.tmp.cleanup)
        old = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, old)
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Synthetic Test')
        self.git('config', 'user.email', 'test@example.invalid')
        Path('README.md').write_text('base')
        self.git('add', '.')
        self.git('commit', '-m', 'base')
        self.base = self.git('rev-parse', 'HEAD')

    def git(self, *args):
        return subprocess.check_output(['git', *args], stderr=subprocess.PIPE, text=True).strip()

    def commit(self, path, content):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
        self.git('add', '.')
        self.git('commit', '-m', path)

    def test_maintained_speech_queue_selects_research_dependencies(self):
        for path in ('tools/speech_data_queue.py', 'tools/import_saved_speech_result.py',
                     'training/weak_speech_dataset.py', 'tests/test_speech_data_queue.py',
                     'docs/speech-data-queue.md'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_runtime_changes_select_research_dependencies(self):
        for path in ('configs/runtime/research-runtime.json', 'tools/research_runtime.py',
                     'tools/research_python_entry.py', 'tests/test_research_python_entry.py',
                     'training/setup_core_runtime.sh', '.github/actions/setup-research-python/action.yml'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_active_asr_changes_select_research_dependencies(self):
        for path in ('tools/setup_asr_runtime.py', 'tools/active_asr_batch.py',
                     'tools/active_asr_worker.py', 'tools/summarize_active_asr_batch.py',
                     'tests/test_active_asr_batch.py', 'tests/test_active_asr_supervision.py',
                     'tools/active_asr_selective.py', 'tests/test_active_asr_selective.py',
                     'configs/research/selective-asr-v2.json', 'docs/ACTIVE_ASR_SELECTIVE_V2.md',
                     'docs/ACTIVE_ASR_BATCH.md'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_conditioned_queue_selects_all_incremental_paths(self):
        for path in ('docs/language-conditioned-weak-import.md', 'tests/test_language_conditioned_queue.py', 'tools/import_language_conditioned_result.py', 'tools/language_conditioned_vote.py', 'tools/speech_data_queue.py', 'tools/speech_history_evidence.py', 'training/weak_speech_dataset.py'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_native_zipformer_paths_select_existing_research_checks(self):
        for path in ('tools/research_job_lock.py', 'tests/test_research_job_lock.py',
                     'tools/run_zipformer_system.py', 'tests/test_zipformer_system.py',
                     'tools/run_d1_zipformer_system.py', 'tests/test_d1_zipformer_system.py',
                     'docs/research-d1-zipformer-native.md',
                     'tools/run_d1_zipformer_extension.py', 'tests/test_d1_zipformer_extension.py',
                     'tools/run_d1_zipformer_continuation.py',
                     'tools/research_d1_native_coverage.py', 'tests/test_d1_native_coverage.py',
                     'tools/research_d1_previous_native.py', 'docs/research-d1-zipformer-extension-v2.md',
                     'docs/research-d1-zipformer-accounted-v3.md', 'docs/research-d1-partial-observation-v3.md',
                     'configs/research/zipformer-system-20261105.lock.json', 'docs/ZIPFORMER_SYSTEM.md'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_active_execution_changes_select_research_dependencies(self):
        for path in ('tools/research_queue.py', 'tests/test_research_queue.py',
                     'tools/research_no_network.py', 'tests/test_research_no_network.py',
                     'tools/setup_tts_runtime.py', 'tests/test_active_tts_runtime.py',
                     'configs/runtime/qwen-base-tts-model-lock.json',
                     'tools/active_tts_smoke.py', 'tests/test_active_tts_smoke.py',
                     'tools/active_tts_pair.py', 'tests/test_active_tts_pair.py',
                     'tools/active_tts_expand.py', 'tests/test_active_tts_expand.py', 'docs/research-tts-continuation.md',
                     'tools/active_tts_continuation.py', 'tools/research_tts_partial.py', 'tools/research_tts_handoff.py',
                     'tests/test_active_tts_continuation.py', 'docs/tts-accounted-checkpoint-v2.md',
                     'training/research_weak_recipe.py', 'training/research_weak_train.py',
                     'training/research_fsmn_weak.py', 'configs/runtime/fsmn-fulltext-sources.json',
                     'tests/test_research_weak_training.py', 'tests/test_research_fsmn_weak.py',
                     'docs/research-weak-training.md', 'docs/LOCAL_RESEARCH_QUEUE.md',
                     'docs/active-tts-smoke.md', 'configs/research/active-plan-20261105.json',
                     'docs/KWS_ACTIVE_EXECUTION_20261105.md',
                     'training/research_d1_vocab.py', 'training/research_d1_inputs.py',
                     'training/research_d1_recipe.py', 'training/research_d1_models.py',
                     'training/research_d1_train.py',
                     'training/research_d1_evaluate.py', 'tests/test_research_d1_evaluate.py',
                     'tests/test_research_d1.py', 'tests/fixtures/d1-fulltext/fsmn-official-tokens.txt',
                     'tests/fixtures/d1-fulltext/research_weak_sampling_v1.py',
                     'tests/fixtures/d1-fulltext/README.md', 'docs/research-d1-fulltext-v2.md',
                     'docs/research-d1-development-v2.md'):
            with self.subTest(path=path):
                self.assertTrue(changes.relevant([path]))

    def test_conditioned_queue_ci_runs_three_modes_without_gate(self):
        spec = importlib.util.spec_from_file_location('test_inventory', ROOT / 'tools/test_inventory.py')
        inventory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inventory)
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        expected = {'python3 -S -B ' + mode + 'tests/test_language_conditioned_queue.py'
                    for mode in ('', '-O ', '-OO ')}
        seen = set()
        for job, block, gated in inventory.run_blocks(workflow):
            commands = set(line.strip() for line in block.splitlines())
            matches = expected & commands
            if matches:
                self.assertFalse(gated)
                seen.update(matches)
        self.assertEqual(seen, expected)

    def test_full_pr_delta_against_first_merge_parent(self):
        self.git('checkout', '-b', 'topic')
        self.commit('research/synthetic.txt', 'a')
        self.commit('README.md', 'second head commit is unrelated')
        self.git('checkout', 'main')
        self.commit('base-only.txt', 'unrelated base changes')
        self.git('merge', '--no-ff', 'topic', '-m', 'synthetic PR merge')
        paths = changes.changed_paths('pull_request', {}, 'refs/pull/1/merge')
        self.assertIn('research/synthetic.txt', paths)
        self.assertNotIn('base-only.txt', paths)
        self.assertTrue(changes.relevant(paths))

    def test_unrelated_pr_needs_no_expensive_checks(self):
        self.git('checkout', '-b', 'topic')
        self.commit('README.md', 'documentation only')
        self.git('checkout', 'main')
        self.git('merge', '--no-ff', 'topic', '-m', 'synthetic PR merge')
        self.assertFalse(changes.relevant(changes.changed_paths('pull_request', {}, 'refs/pull/2/merge')))

    def test_multi_commit_push_and_deleted_research_path(self):
        self.commit('research/synthetic.txt', 'a')
        before = self.git('rev-parse', 'HEAD')
        Path('research/synthetic.txt').unlink()
        self.git('add', '-A')
        self.git('commit', '-m', 'delete research')
        self.commit('README.md', 'later unrelated commit')
        self.assertTrue(changes.relevant(changes.changed_paths('push', {'before': before}, 'refs/heads/main')))

    def test_shallow_multi_commit_push_fetches_exact_before_locally(self):
        self.commit('research/synthetic.txt', 'early push change')
        self.commit('README.md', 'middle')
        self.commit('README.md', 'last')
        # file:// forces real shallow-clone semantics while remaining local.
        remote = Path(self.tmp.name) / 'fixture.git'
        shallow = Path(self.tmp.name) / 'shallow'
        self.git('clone', '--bare', '.', str(remote))
        self.git('clone', '--depth=2', remote.as_uri(), str(shallow))
        old = Path.cwd()
        os.chdir(shallow)
        try:
            self.assertNotEqual(subprocess.run(
                ['git', 'cat-file', '-e', self.base + '^{commit}'],
                capture_output=True).returncode, 0)
            self.assertTrue(changes.relevant(changes.changed_paths(
                'push', {'before': self.base}, 'refs/heads/main')))
        finally:
            os.chdir(old)

    def test_first_push_uses_all_tracked_paths(self):
        self.commit('research/synthetic.txt', 'a')
        self.assertTrue(changes.relevant(changes.changed_paths('push', {'before': '0' * 40}, 'refs/heads/main')))

    def test_invalid_merge_and_invalid_before_fail_closed(self):
        with self.assertRaises(ValueError):
            changes.changed_paths('pull_request', {}, 'refs/pull/1/merge')
        with self.assertRaises(ValueError):
            changes.changed_paths('push', {'before': '--untrusted'}, 'refs/heads/main')

    def test_missing_before_fetches_once_and_failure_is_not_a_skip(self):
        # Mock only subprocess.run: no network-capable fetch is executed.
        with patch.object(changes.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 1), subprocess.CalledProcessError(1, 'fetch')]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                changes.changed_paths('push', {'before': 'a' * 40}, 'refs/heads/main')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0],
                             ['git', 'fetch', '--no-tags', '--depth=1', 'origin', 'a' * 40])

    def test_host_verifier_source_exact_prefix_triggers_shared_gate(self):
        for path in ('host-tts-verifier-source/synthetic.py',
                     'host-tts-verifier-source/nested/test_synthetic.py'):
            self.assertTrue(changes.relevant([path]), path)
        for path in ('host-tts-verifier-source', 'host-tts-verifier-source-extra/file.py',
                     'host-tts-verifier-sources/file.py', 'host-tts-verifier-source2/file.py',
                     'other/host-tts-verifier-source/file.py', 'src/other.py'):
            self.assertFalse(changes.relevant([path]), path)

    def test_deleted_host_verifier_source_triggers_shared_gate(self):
        self.commit('host-tts-verifier-source/synthetic.py', '# fixture')
        before = self.git('rev-parse', 'HEAD')
        Path('host-tts-verifier-source/synthetic.py').unlink()
        self.git('add', '-A')
        self.git('commit', '-m', 'delete host verifier fixture')
        self.commit('README.md', 'later unrelated commit')
        self.assertTrue(changes.relevant(changes.changed_paths(
            'push', {'before': before}, 'refs/heads/main')))

    def test_original_event_scope_and_all_workflow_paths(self):
        self.assertFalse(changes.changed_paths('push', {}, 'refs/heads/feature/example'))
        for path in ('research/any/file', '.github/workflows/new.yml',
                     'tools/verify_research_sources.py', 'tools/verify_research_publication.py',
                     'tools/run_research_source_checks.py', 'tools/check_workflow_path_filters.py',
                     'tools/research_ci_changes.py', 'tests/test_research_ci_changes.py'):
            self.assertTrue(changes.relevant([path]), path)
        self.assertFalse(changes.relevant(['README.md', 'src/kws.c', 'research-lookalike/file']))


if __name__ == '__main__':
    unittest.main()
