#!/usr/bin/env python3
"""Offline contracts for release commands and atomic bootstrap cleanup."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / '.github/workflows'


def job(source: str, name: str) -> str:
    tail = source.split(f'\n  {name}:\n', 1)[1]
    return re.split(r'(?m)^  [a-z][a-z0-9-]*:\s*$', tail, maxsplit=1)[0]


def step(source: str, name: str) -> str:
    tail = source.split(f'      - name: {name}\n', 1)[1]
    return re.split(r'(?m)^      - ', tail, maxsplit=1)[0]


def command(source: str, name: str) -> str:
    value = step(source, name).split('        run: ', 1)[1]
    if value.startswith('|\n'):
        return textwrap.dedent(value[2:]).rstrip()
    return value.splitlines()[0]


class ReleaseCommands(unittest.TestCase):
    def setUp(self):
        self.release = (WORKFLOWS / 'release.yml').read_text()
        self.ci = (WORKFLOWS / 'ci.yml').read_text()

    def test_shared_inventory_executes_and_catches_missing_test(self):
        inventory = command(self.release, 'Python test inventory')
        self.assertEqual(inventory, command(self.ci, 'Python test inventory'))
        self.assertEqual(inventory, 'python3 tools/test_inventory.py --official-workflows')
        # Run the real release command, not just a filename-presence assertion.
        subprocess.run(['bash', '-e', '-c', inventory], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'test_deliberately_unwired_release_test.py').write_text('')
            result = subprocess.run([sys.executable, 'tools/test_inventory.py',
                                     '--official-workflows', '--root', tmp], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('test_deliberately_unwired_release_test.py', result.stderr)

    def test_fixture_preparation_precedes_release_evaluator_and_fails_closed(self):
        hosted = job(self.release, 'hosted')
        prepare = command(hosted, 'Prepare pinned public evaluation context fixtures')
        product = command(hosted, 'Python/product tests')
        self.assertEqual(prepare, 'python3 tools/prepare_eval_context_fixtures.py')
        self.assertLess(hosted.index(prepare), hosted.index('python3 tests/test_eval.py'))
        self.assertLess(self.ci.index(prepare), self.ci.index('python3 tests/test_eval.py'))
        self.assertIn('python3 tests/test_release_workflow_contract.py', hosted)
        self.assertIn('python3 tests/test_release_workflow_contract.py', job(self.ci, 'python-contracts'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / 'python3'
            stub.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''
                import os, pathlib, sys
                root = pathlib.Path(os.environ['FIXTURE_TEST_ROOT'])
                target = sys.argv[1]
                with (root / 'commands').open('a') as log:
                    log.write(target + '\\n')
                if target == 'tools/prepare_eval_context_fixtures.py':
                    if os.environ.get('FAIL_PREPARE') == '1':
                        sys.exit(7)
                    (root / 'prepared').touch()
                elif target == 'tests/test_eval.py':
                    sys.exit(0 if (root / 'prepared').exists() else 9)
            '''))
            stub.chmod(0o755)
            env = {'PATH': f'{root}:/usr/bin:/bin', 'FIXTURE_TEST_ROOT': str(root)}
            script = prepare + '\n' + product
            result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', script], env=env)
            self.assertEqual(result.returncode, 0)
            calls = (root / 'commands').read_text().splitlines()
            self.assertLess(calls.index('tools/prepare_eval_context_fixtures.py'),
                            calls.index('tests/test_eval.py'))
            (root / 'commands').unlink()
            (root / 'prepared').unlink()
            result = subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', script],
                                    env=dict(env, FAIL_PREPARE='1'))
            self.assertEqual(result.returncode, 7)
            self.assertEqual((root / 'commands').read_text().splitlines(),
                             ['tools/prepare_eval_context_fixtures.py'])

    def test_decoder_replay_contract_uses_built_runner_in_ci_and_release(self):
        replay = 'python3 tests/test_decoder_policy_replay.py --path-runner ./build/kws_decoder_path_replay'
        self.assertEqual(command(job(self.ci, 'hosted'), 'Decoder path shadow/runtime replay contract'), replay)
        self.assertNotIn('test_decoder_policy_replay.py', job(self.ci, 'python-contracts'))
        self.assertIn(replay, command(job(self.release, 'hosted'), 'Python/product tests'))

    def test_cleanup_requires_all_success_and_original_ref(self):
        for filename, mode, ref in (
            ('release.yml', 'sdk', 'refs/heads/release/v1.2.3'),
            ('deployment-release.yml', 'deployment', 'refs/heads/deployment/commercial-candidate'),
        ):
            cleanup = job((WORKFLOWS / filename).read_text(), 'cleanup-bootstrap-branch')
            condition = re.search(r'^    if: \$\{\{ (.*?) \}\}$', cleanup, re.M).group(1)
            expected_ref = ("startsWith(github.ref, 'refs/heads/release/v')" if mode == 'sdk'
                            else "github.ref == 'refs/heads/deployment/commercial-candidate'")
            self.assertEqual(condition, "success() && needs.publish.result == 'success' && " + expected_ref)
            self.assertNotIn('always()', cleanup)
            self.assertNotIn('--method DELETE', cleanup)
            self.assertIn('ref: ${{ github.sha }}', cleanup)
            self.assertIn('persist-credentials: false', cleanup)
            self.assertIn(f'run: bash tools/cleanup_bootstrap_branch.sh {mode}', cleanup)
            if mode == 'sdk':
                self.assertIn('needs: [hosted, coverage, sanitizers, fuzz, armv7-cross, publish]', cleanup)
            else:
                self.assertIn('needs: publish', cleanup)
            # Exercise this exact bounded expression for failure, cancellation,
            # skipped dependencies, and non-bootstrap refs. No Actions dispatch.
            for all_success in (True, False):
                for publish in ('success', 'failure', 'cancelled', 'skipped', ''):
                    for current_ref in (ref, 'refs/tags/v1.2.3', 'refs/heads/main'):
                        expr = condition.replace('success()', str(all_success))
                        expr = expr.replace('needs.publish.result', repr(publish))
                        expr = expr.replace("startsWith(github.ref, 'refs/heads/release/v')",
                                            str(current_ref.startswith('refs/heads/release/v')))
                        expr = expr.replace('github.ref', repr(current_ref)).replace('&&', 'and')
                        allowed = eval(expr, {'__builtins__': {}}, {})
                        self.assertEqual(bool(allowed), all_success and publish == 'success' and current_ref == ref)


class BootstrapLease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bootstrap-lease-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.remote = self.root / 'remote.git'
        self.writer = self.root / 'writer'
        self.real_git = shutil.which('git')
        self.env = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root),
                    'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                    'GIT_TERMINAL_PROMPT': '0'}
        self.git('init', '--bare', self.remote)
        self.git('init', self.writer)
        self.git('-C', self.writer, 'config', 'user.name', 'Synthetic Test')
        self.git('-C', self.writer, 'config', 'user.email', 'test@example.invalid')
        self.git('-C', self.writer, 'commit', '--allow-empty', '-m', 'A')
        self.a = self.git('-C', self.writer, 'rev-parse', 'HEAD')
        self.git('-C', self.writer, 'commit', '--allow-empty', '-m', 'B')
        self.b = self.git('-C', self.writer, 'rev-parse', 'HEAD')
        stubs = self.root / 'bin'
        stubs.mkdir()
        (stubs / 'gh').write_text('#!/bin/sh\n[ "$*" = "auth setup-git" ] || exit 90\n')
        # Interpose only the fixed production remote URL. Every actual Git push
        # in this test reaches a disposable local bare repository.
        (stubs / 'git').write_text(f'#!{sys.executable}\n' + textwrap.dedent(f'''
            import os, sys
            args = sys.argv[1:]
            args = [{str(self.remote)!r} if arg == 'https://github.com/test/synthetic.git' else arg for arg in args]
            if any('https://' in arg or 'ssh://' in arg for arg in args):
                sys.exit(91)
            os.execv({self.real_git!r}, [{self.real_git!r}, *args])
        '''))
        for path in stubs.iterdir():
            path.chmod(0o755)
        self.env.update(PATH=f'{stubs}:/usr/bin:/bin', RUNNER_TEMP=str(self.root),
                        GITHUB_REPOSITORY='test/synthetic')

    def git(self, *args):
        return subprocess.check_output([self.real_git, *map(str, args)], env=self.env,
                                       stderr=subprocess.PIPE, text=True).strip()

    def cleanup(self, mode, branch, sha, **extra):
        env = dict(self.env, GITHUB_REF_NAME=branch, GITHUB_REF=f'refs/heads/{branch}', GITHUB_SHA=sha)
        env.update(extra)
        return subprocess.run(['bash', str(ROOT / 'tools/cleanup_bootstrap_branch.sh'), mode],
                              env=env, text=True, capture_output=True)

    def test_stale_a_preserves_new_b_and_matching_sha_deletes_local_branch(self):
        for mode, branch in (('sdk', 'release/v1.2.3'), ('deployment', 'deployment/commercial-candidate')):
            self.git('-C', self.writer, 'push', self.remote, f'{self.a}:refs/heads/{branch}')
            self.git('-C', self.writer, 'push', self.remote, f'{self.b}:refs/heads/{branch}')
            stale = self.cleanup(mode, branch, self.a)
            self.assertNotEqual(stale.returncode, 0, stale.stdout)
            self.assertEqual(self.git('--git-dir', self.remote, 'rev-parse', f'refs/heads/{branch}'), self.b)
            matched = self.cleanup(mode, branch, self.b)
            self.assertEqual(matched.returncode, 0, matched.stderr)
            self.assertEqual(self.git('ls-remote', '--heads', self.remote, f'refs/heads/{branch}'), '')

    def test_rejects_wrong_branch_tag_malformed_sha_and_mode(self):
        branch = 'release/v1.2.3'
        self.git('-C', self.writer, 'push', self.remote, f'{self.a}:refs/heads/{branch}')
        for mode, name, sha, extra in (
            ('sdk', 'main', self.a, {}), ('sdk', branch, '', {}),
            ('sdk', branch, 'z' * 40, {}), ('sdk', branch, self.a[:7], {}),
            ('sdk', branch, self.a, {'GITHUB_REF': 'refs/tags/v1.2.3'}),
            ('deployment', branch, self.a, {}), ('wrong-mode', branch, self.a, {}),
        ):
            self.assertNotEqual(self.cleanup(mode, name, sha, **extra).returncode, 0)
        self.assertEqual(self.git('--git-dir', self.remote, 'rev-parse', f'refs/heads/{branch}'), self.a)


if __name__ == '__main__':
    unittest.main()
