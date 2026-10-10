#!/usr/bin/env python3
"""Offline contracts for release commands and atomic bootstrap cleanup."""
from __future__ import annotations

import json
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
    return re.split(r'(?m)^      - |^  [a-z][a-z0-9-]*:\s*$', tail, maxsplit=1)[0]


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

    def test_canonical_identity_and_event_checks_survive_optimized_python(self):
        canonical = job(self.ci, 'python-contracts')
        corpus = command(canonical, 'Corpus byte identity')
        for name in ('test_corpus_identity.py', 'test_training_admission.py'):
            self.assertIn(f'python3 tests/{name}', corpus.splitlines())
            for mode in ('-O', '-OO'):
                self.assertIn(f'python3 -S -B {mode} tests/{name}', corpus.splitlines())
        qualification = command(job(self.ci, 'hosted'), 'Release qualification manifest and gate')
        self.assertEqual(qualification.splitlines(), [
            'python3 tests/test_release_qualification.py',
            'python3 -S -B -O tests/test_release_qualification.py --event-metadata-only',
            'python3 -S -B -OO tests/test_release_qualification.py --event-metadata-only',
        ])
        # Optimized regressions remain metadata-only. The existing normal full
        # qualification test is retained without adding another execution lane.
        self.assertEqual(qualification.count('--event-metadata-only'), 2)

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


class ReleaseTagBinding(unittest.TestCase):
    """Exercise exact publication shell commands against a strictly offline gh."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='release-tag-binding-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'bin').mkdir()
        (self.root / 'tools').mkdir()
        shutil.copyfile(ROOT / 'tools/verify_release_tag.py', self.root / 'tools/verify_release_tag.py')
        gh = self.root / 'bin/gh'
        gh.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''
            import json, os, pathlib, sys
            root = pathlib.Path(os.environ['FAKE_GH_ROOT'])
            args = sys.argv[1:]
            with (root / 'commands.jsonl').open('a') as out:
                out.write(json.dumps(args) + '\\n')
            if args[:1] == ['api']:
                if args[1:3] != ['--method', 'GET']:
                    sys.exit(91)
                responses = json.loads((root / 'responses.json').read_text())
                endpoint = args[-1]
                if endpoint not in responses:
                    sys.exit(92)
                row = responses[endpoint]
                if row.get('exit'):
                    sys.exit(row['exit'])
                print(row.get('raw', json.dumps(row.get('value'))))
            elif args[:2] == ['release', 'create']:
                (root / 'publication.json').write_text(json.dumps(args))
            else:
                sys.exit(93)
        '''))
        gh.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.root / "bin"}:/usr/bin:/bin',
                        FAKE_GH_ROOT=str(self.root), GITHUB_REPOSITORY='test/synthetic',
                        GITHUB_SHA='b' * 40, EXPECTED_HEAD_SHA='b' * 40,
                        RELEASE_TAG='v1.2.3', MODEL_RELEASE_TAG='model-test',
                        TRAINING_RUN_ID='12345', MODEL_SHA256='c' * 64, VERSION='1.2.3')
        self.endpoint = 'repos/test/synthetic/git/matching-refs/tags/v1.2.3'

    def run_publish(self, refs, *, annotations=None, bootstrap=True, error=None, model=False):
        endpoint = self.endpoint.replace('v1.2.3', 'model-test') if model else self.endpoint
        responses = {endpoint: error or {'value': refs}}
        for sha, value in (annotations or {}).items():
            responses[f'repos/test/synthetic/git/tags/{sha}'] = {'value': value}
        (self.root / 'responses.json').write_text(json.dumps(responses))
        published = self.root / 'publication.json'
        published.unlink(missing_ok=True)
        if model:
            source = (WORKFLOWS / 'model-promotion.yml').read_text()
            script = command(source, 'Publish trained model release')
        else:
            source = (WORKFLOWS / 'release.yml').read_text()
            script = command(source, 'Publish bootstrap release and tag' if bootstrap else 'Publish existing-tag release')
        result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script],
                                cwd=self.root, env=self.env, text=True, capture_output=True)
        return result, published.exists()

    @staticmethod
    def ref(sha, kind='commit', tag='v1.2.3'):
        return {'ref': 'refs/tags/' + tag, 'object': {'type': kind, 'sha': sha}}

    def test_absent_tag_is_allowed_only_for_bootstrap(self):
        for refs in ([], [self.ref('a' * 40, tag='v1.2.30')]):
            result, published = self.run_publish(refs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(published)
            result, published = self.run_publish(refs, bootstrap=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(published)

    def test_existing_lightweight_tag_must_match_artifact_source(self):
        for bootstrap in (True, False):
            for sha in ('a' * 40, 'b' * 40):
                result, published = self.run_publish([self.ref(sha)], bootstrap=bootstrap)
                self.assertEqual(result.returncode == 0, sha == 'b' * 40, result.stderr)
                self.assertEqual(published, sha == 'b' * 40)
                if not published:
                    self.assertIn('release tag commit mismatch', result.stderr)

    def test_annotated_and_nested_tags_are_peeled_before_publish(self):
        for target in ('a' * 40, 'b' * 40):
            for nested in (False, True):
                annotations = {'c' * 40: {'sha': 'c' * 40, 'object': {
                    'type': 'tag' if nested else 'commit', 'sha': 'd' * 40 if nested else target}}}
                if nested:
                    annotations['d' * 40] = {'sha': 'd' * 40, 'object': {'type': 'commit', 'sha': target}}
                result, published = self.run_publish([self.ref('c' * 40, 'tag')], annotations=annotations)
                self.assertEqual(result.returncode == 0, target == 'b' * 40, result.stderr)
                self.assertEqual(published, target == 'b' * 40)

    def test_api_errors_malformed_objects_and_cycles_fail_closed(self):
        cases = [
            ([], {}, {'exit': 1}), ([], {}, {'raw': 'not JSON'}),
            ([], {}, {'value': {'message': 'Not Found'}}),
            ([self.ref('b' * 40), self.ref('b' * 40)], {}, None),
            ([self.ref('b' * 40, 'tree')], {}, None),
            ([self.ref('b' * 39)], {}, None),
            ([self.ref('c' * 40, 'tag')], {'c' * 40: {'sha': 'a' * 40, 'object': {'type': 'commit', 'sha': 'b' * 40}}}, None),
            ([self.ref('c' * 40, 'tag')], {'c' * 40: {'sha': 'c' * 40, 'object': {'type': 'tag', 'sha': 'c' * 40}}}, None),
        ]
        for refs, annotations, error in cases:
            with self.subTest(refs=refs, error=error):
                result, published = self.run_publish(refs, annotations=annotations, error=error)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(published)

    def test_model_publication_uses_same_commit_binding(self):
        for sha in ('a' * 40, 'b' * 40):
            result, published = self.run_publish([self.ref(sha, tag='model-test')], model=True)
            self.assertEqual(result.returncode == 0, sha == 'b' * 40, result.stderr)
            self.assertEqual(published, sha == 'b' * 40)
        source = (WORKFLOWS / 'model-promotion.yml').read_text()
        for name in ('Resolve immutable model release tag', 'Verify immutable model release'):
            self.assertIn('tools/verify_release_tag.py', command(source, name))
        self.assertNotIn('--allow-absent', command(source, 'Verify immutable model release'))


class PromotionControlPlane(unittest.TestCase):
    def test_current_protected_main_precedes_all_promotion_work(self):
        source = (WORKFLOWS / 'model-promotion.yml').read_text()
        guard_name = 'Require exact current protected main control plane'
        guard = command(source, guard_name)
        self.assertEqual(guard, 'bash governance/require_current_main.sh')
        for name in ('Verify requested training run', 'Download exact trained-model artifact',
                     'Freeze and verify deployable model evidence', 'Attest promoted model assets',
                     'Publish trained model release', 'Mirror promoted model into Git registry'):
            self.assertLess(source.index(guard_name), source.index(name))
        self.assertIn("run.get('head_sha') != expected", source)
        with tempfile.TemporaryDirectory(prefix='promotion-control-plane-') as tmp:
            root = Path(tmp)
            (root / 'bin').mkdir()
            (root / 'governance').mkdir()
            for name in ('require_current_main.sh', 'main-ruleset-target.json', 'verify_live_main_ruleset.py'):
                shutil.copyfile(ROOT / 'governance' / name, root / 'governance' / name)
            gh = root / 'bin/gh'
            gh.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''
                import json, os, pathlib, sys
                args = sys.argv[1:]
                if len(args) != 2 or args[0] != 'api':
                    sys.exit(91)
                responses = json.loads(pathlib.Path('responses.json').read_text())
                if args[1] not in responses:
                    sys.exit(92)
                print(json.dumps(responses[args[1]]))
            '''))
            gh.chmod(0o755)
            target = json.loads((root / 'governance/main-ruleset-target.json').read_text())
            for ref, sha, protected, ruleset_valid, allowed in (
                ('refs/heads/main', 'b' * 40, True, True, True),
                ('refs/heads/main', 'a' * 40, True, True, False),
                ('refs/heads/feature/test', 'b' * 40, True, True, False),
                ('refs/tags/v1.2.3', 'b' * 40, True, True, False),
                ('refs/heads/main', 'b' * 40, False, True, False),
                ('refs/heads/main', 'b' * 40, True, False, False),
            ):
                live = dict(target, id=1, bypass_actors=[] if ruleset_valid else [{'actor_id': 7}])
                responses = {
                    'repos/test/synthetic/branches/main': {'commit': {'sha': 'b' * 40}, 'protected': protected},
                    'repos/test/synthetic/rulesets': [dict(target, id=1)],
                    'repos/test/synthetic/rulesets/1': live,
                }
                (root / 'responses.json').write_text(json.dumps(responses))
                env = dict(os.environ, PATH=f'{root / "bin"}:/usr/bin:/bin', GH_TOKEN='offline-fixture',
                           GITHUB_REPOSITORY='test/synthetic', GITHUB_REF=ref, GITHUB_SHA=sha,
                           # Historical training identity remains separate from current control-plane identity.
                           EXPECTED_HEAD_SHA='c' * 40)
                marker = root / 'promotion-may-start'
                marker.unlink(missing_ok=True)
                result = subprocess.run(['bash', '-euo', 'pipefail', '-c', guard + '\ntouch promotion-may-start'],
                                        cwd=root, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                self.assertEqual(marker.exists(), allowed)


class WorkflowProvenance(unittest.TestCase):
    def test_full_model_and_dataset_pr_bodies_are_literal_and_complete(self):
        cases = (
            ('model-promotion.yml', 'Mirror promoted model into Git registry', 'model-registry-pr.md'),
            ('dataset-driven-iteration.yml', 'Record scorecard through a protected-main PR', 'dataset-iteration-pr.md'),
        )
        for hostile in (False, True):
            suffix = "`touch should-not-exist` $(touch should-not-exist) \"' 中文" if hostile else ''
            values = {'MODEL_RELEASE_TAG': 'model-aaaaaaaaaaaa' + suffix, 'TRAINING_RUN_ID': '12345',
                      'EXPECTED_HEAD_SHA': 'a' * 40, 'dataset': 'dataset-abc' + suffix,
                      'run_id': 'run-abc' + suffix, 'CONTROLLED_VARIABLE': 'dataset',
                      'MODEL_LABEL': 'git-registry:model-aaaaaaaaaaaa' + suffix}
            for filename, name, output in cases:
                with self.subTest(filename=filename, hostile=hostile), tempfile.TemporaryDirectory() as tmp:
                    script = command((WORKFLOWS / filename).read_text(), name)
                    body = re.search(r"(?m)^[^\n]*python3 - <<'PYBODY'\n.*?^PYBODY$", script, re.S).group()
                    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', body], cwd=tmp,
                                            env=dict(os.environ, RUNNER_TEMP=tmp, **values),
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, '')
                    self.assertFalse((Path(tmp) / 'should-not-exist').exists())
                    actual = (Path(tmp) / output).read_text()
                    if filename == 'model-promotion.yml':
                        expected = (
                            f"Byte-for-byte Git mirror of immutable model release `{values['MODEL_RELEASE_TAG']}`.\n\n"
                            f"Generated by `model-promotion` from training run `{values['TRAINING_RUN_ID']}`\n"
                            f"at source `{values['EXPECTED_HEAD_SHA']}`.\n\n"
                            'Shipping approval remains governed independently by\n'
                            '`configs/shipping.xiaowo.json` and protected qualification evidence.\n')
                    else:
                        expected = (
                            'Dataset-driven KWS measurement.\n\n'
                            f"- dataset id: `{values['dataset']}`\n"
                            f"- run id: `{values['run_id']}`\n"
                            f"- controlled variable: `{values['CONTROLLED_VARIABLE']}`\n"
                            f"- model: `{values['MODEL_LABEL']}`\n\n"
                            'The scorecard is evidence only; it does not freeze, promote or\n'
                            'shipping-approve a model.\n')
                    self.assertEqual(actual, expected)
                    self.assertIn('--body-file "$RUNNER_TEMP/' + output + '"', script)

    def test_dataset_shell_reads_dispatch_values_from_environment(self):
        source = (WORKFLOWS / 'dataset-driven-iteration.yml').read_text()
        for name in ('Run the dataset iteration', 'Compare against the declared baseline',
                     'Record scorecard through a protected-main PR'):
            script = command(source, name)
            self.assertNotIn('${{', script)
            subprocess.run(['bash', '-n'], input=script, text=True, check=True)
        self.assertIn('bash governance/require_current_main.sh', source)

    def test_tuple_output_rejects_multiline_values(self):
        source = (WORKFLOWS / 'dataset-driven-iteration.yml').read_text()
        script = command(source, 'Resolve deployable model tuple')
        with tempfile.TemporaryDirectory(prefix='dataset-tuple-') as tmp:
            root = Path(tmp)
            for name in ('model.kwm', 'keywords.kwk', 'model\\nforged=value.kwm'):
                (root / name.replace('\\n', '\n')).touch()
            for model, allowed in (('model.kwm', True), ('model\nforged=value.kwm', False)):
                output = root / 'output'
                output.unlink(missing_ok=True)
                result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=root,
                                        env=dict(os.environ, INPUT_MODEL_PATH=model,
                                                 INPUT_KEYWORDS_PATH='keywords.kwk', GITHUB_OUTPUT=str(output)),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, allowed, result.stderr)
                self.assertEqual(output.exists(), allowed)
                if allowed:
                    self.assertEqual(output.read_text(),
                                     'model_path=model.kwm\nkeywords_path=keywords.kwk\nmodel_label=custom:model.kwm\n')

    def test_governance_preserves_failed_bootstrap_branches(self):
        source = (ROOT / 'docs/REPOSITORY_GOVERNANCE.md').read_text()
        self.assertIn('Failed, cancelled or skipped bootstrap runs preserve the temporary branch', source)
        self.assertIn('atomic lease', source)
        self.assertNotIn('Failed bootstrap runs delete', source)


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
