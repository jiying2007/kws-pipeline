#!/usr/bin/env python3
"""Exercise the real publication shell with local Git and an offline gh stub."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/dataset-driven-iteration.yml'


def workflow_script(name):
    lines = WORKFLOW.read_text().splitlines()
    start = lines.index('      - name: ' + name)
    start = lines.index('        run: |', start) + 1
    script = []
    for line in lines[start:]:
        if line and not line.startswith('          '):
            break
        script.append(line[10:] if line else '')
    return '\n'.join(script) + '\n'


def publication_script():
    return workflow_script('Record scorecard through a protected-main PR')


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / 'remote.git'
        self.repo = self.root / 'checkout'
        self.repo.mkdir()
        self.git('init', '--bare', str(self.remote), cwd=self.root)
        self.git('init', '--initial-branch=main')
        self.git('config', 'user.name', 'offline test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.repo / '.gitignore').write_text('build/\n')
        self.git('add', '.gitignore')
        self.git('commit', '-m', 'fixture')
        self.git('remote', 'add', 'origin', str(self.remote))
        self.git('push', '-u', 'origin', 'main')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.state = self.root / 'prs.json'
        self.state.write_text('[]')
        self.calls = self.root / 'calls.jsonl'
        gh = self.bin / 'gh'
        gh.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['FAKE_GH_CALLS'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
state = pathlib.Path(os.environ['FAKE_GH_STATE'])
if args[:2] == ['auth', 'setup-git']:
    raise SystemExit(0)
if args[:1] == ['api']:
    print(pathlib.Path(os.environ['FAKE_GH_RUN']).read_text())
    raise SystemExit(0)
if args[:2] == ['run', 'download']:
    import shutil
    shutil.copytree(os.environ['FAKE_GH_ARTIFACT'], args[args.index('--dir') + 1], dirs_exist_ok=True)
    raise SystemExit(0)
if args[:2] == ['pr', 'list']:
    print(state.read_text())
    raise SystemExit(0)
if args[:2] == ['pr', 'create']:
    once = state.with_suffix('.failure')
    if os.environ.get('FAKE_GH_FAIL_ONCE') == '1' and not once.exists():
        once.write_text('injected transient PR failure')
        raise SystemExit(17)
    body = pathlib.Path(args[args.index('--body-file') + 1]).read_text()
    if 'research evidence only' not in body:
        raise SystemExit('missing evidence scope')
    url = 'https://example.invalid/pr/1'
    state.write_text(json.dumps([{'state': 'OPEN', 'url': url}]))
    print(url)
    raise SystemExit(0)
raise SystemExit('unexpected gh command: ' + repr(args))
''')
        gh.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        RUNNER_TEMP=str(self.root), GITHUB_REPOSITORY='offline/fixture',
                        GITHUB_STEP_SUMMARY=str(self.root / 'summary'),
                        FAKE_GH_STATE=str(self.state), FAKE_GH_CALLS=str(self.calls))
        self.dataset = 'a' * 64
        self.run_id = 'b' * 16
        self.branch = 'dataset-iteration/' + self.run_id
        self.card = self.repo / 'build/dataset-iteration/scorecard.json'
        self.card.parent.mkdir(parents=True)
        self.card.write_text(json.dumps({'run_id': self.run_id,
            'identity': {'dataset_id': self.dataset, 'variable': 'model'},
            'labels': {'model': 'offline fixture'}}) + '\n')
        self.script = self.root / 'publish.sh'
        self.script.write_text(publication_script())

    def git(self, *args, cwd=None):
        return subprocess.check_output(['git', *args], cwd=cwd or self.repo,
                                       stderr=subprocess.STDOUT, text=True).strip()

    def publish(self, expected=0):
        result = subprocess.run(['bash', str(self.script)], cwd=self.repo, env=self.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=20)
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_failed_pr_creation_resumes_same_branch_and_is_idempotent(self):
        self.env['FAKE_GH_FAIL_ONCE'] = '1'
        self.publish(expected=17)
        pushed = self.git('rev-parse', self.branch)
        self.git('switch', 'main')
        self.publish()
        self.publish()
        self.assertEqual(self.git('rev-parse', 'refs/remotes/origin/' + self.branch), pushed)
        calls = [json.loads(x) for x in self.calls.read_text().splitlines()]
        self.assertEqual(sum(c[:2] == ['pr', 'create'] for c in calls), 2)

    def test_same_identity_different_scorecard_never_overwrites_remote(self):
        self.publish()
        pushed = self.git('rev-parse', self.branch)
        self.git('switch', 'main')
        self.card.write_text(self.card.read_text().replace('offline fixture', 'changed evidence'))
        self.assertIn('different scorecard bytes', self.publish(expected=2))
        self.assertEqual(self.git('rev-parse', 'refs/remotes/origin/' + self.branch), pushed)

    def test_comparison_must_match_on_resume(self):
        comparison = self.card.with_name('comparison.json')
        comparison.write_text('{"result": "held"}\n')
        self.publish()
        self.git('switch', 'main')
        comparison.unlink()
        self.assertIn('comparison identity collision', self.publish(expected=2))

    def test_closed_unmerged_pr_is_not_silently_reopened(self):
        self.publish()
        self.git('switch', 'main')
        self.state.write_text('[{"state":"CLOSED","url":"https://example.invalid/pr/1"}]')
        self.assertIn('closed without merging', self.publish(expected=2))

    def test_evidence_upload_precedes_publication_even_after_measurement_failure(self):
        text = WORKFLOW.read_text()
        upload = text.index('- name: Upload the full evidence before publication')
        publish = text.index('- name: Record scorecard through a protected-main PR')
        self.assertLess(upload, publish)
        self.assertIn('if: always()', text[upload:publish])
        self.assertIn('resume_artifact_run_id', text)
        self.assertIn('gh run download', text)

    def test_recovery_requires_completed_main_dispatch_before_downloading(self):
        source = self.root / 'source-run.json'
        artifact = self.root / 'saved-artifact'
        artifact.mkdir()
        (artifact / 'scorecard.json').write_bytes(self.card.read_bytes())
        self.env.update(FAKE_GH_RUN=str(source), FAKE_GH_ARTIFACT=str(artifact), RESUME_RUN_ID='123')
        valid = {'path': '.github/workflows/dataset-driven-iteration.yml',
                 'event': 'workflow_dispatch', 'head_branch': 'main', 'status': 'completed'}
        script = workflow_script('Restore saved evidence for publication recovery')
        for field, value in [('status', 'in_progress'), ('head_branch', 'feature/unreviewed'),
                             ('event', 'pull_request'), ('path', '.github/workflows/other.yml')]:
            with self.subTest(field=field):
                source.write_text(json.dumps(dict(valid, **{field: value})))
                self.calls.write_text('')
                result = subprocess.run(['bash', '-c', script], cwd=self.repo, env=self.env,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        timeout=20)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                calls = [json.loads(x) for x in self.calls.read_text().splitlines()]
                self.assertFalse(any(c[:2] == ['run', 'download'] for c in calls))
        source.write_text(json.dumps(valid))
        self.card.unlink()
        result = subprocess.run(['bash', '-c', script], cwd=self.repo, env=self.env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.card.read_bytes(), (artifact / 'scorecard.json').read_bytes())

    def test_baseline_artifact_stays_separate_and_matches_declared_card(self):
        source = self.root / 'baseline-source-run.json'
        source.write_text(json.dumps({'path': '.github/workflows/dataset-driven-iteration.yml',
            'event': 'workflow_dispatch', 'head_branch': 'main', 'status': 'completed'}))
        artifact = self.root / 'baseline-artifact'
        artifact.mkdir()
        raw = json.dumps({'run_id': 'c' * 16, 'baseline': True}) + '\n'
        (artifact / 'scorecard.json').write_text(raw)
        bundle = artifact / 'evidence' / ('c' * 16)
        bundle.mkdir(parents=True)
        (bundle / 'needed-input').write_text('retained input')
        declared = self.repo / 'baseline.json'
        declared.write_text(raw)
        candidate_before = self.card.read_bytes()
        self.env.update(FAKE_GH_RUN=str(source), FAKE_GH_ARTIFACT=str(artifact),
                        BASELINE_ARTIFACT_RUN_ID='456', BASELINE_CARD=str(declared))
        script = workflow_script('Restore the declared baseline with its evidence')
        def restore():
            return subprocess.run(['bash', '-c', script], cwd=self.repo, env=self.env,
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  timeout=20)
        result = restore()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.card.read_bytes(), candidate_before)
        self.assertEqual((self.root / 'baseline-iteration/evidence' / ('c' * 16) / 'needed-input')
                         .read_text(), 'retained input')
        declared.write_text(raw + ' ')
        result = restore()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differs from the declared baseline_card', result.stdout)
        self.assertEqual(self.card.read_bytes(), candidate_before)

    def test_builtin_system_runner_rejects_an_unexecuted_block_size(self):
        context = self.root / 'system-context.json'
        context.write_text('{"feed_chunk_samples": 320}')
        self.env.update(CONTROLLED_VARIABLE='system', SYSTEM_CONTEXT_PATH=str(context),
                        NATIVE_FRONTEND='native test frontend')
        result = subprocess.run(['bash', '-c', workflow_script('Run the dataset iteration')],
                                cwd=self.repo, env=self.env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('feed_chunk_samples must be 160', result.stdout)


if __name__ == '__main__':
    unittest.main()
