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
