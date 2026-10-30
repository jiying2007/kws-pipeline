"""Stdlib-only tests: no model imports, network, Docker, TTS or training."""
from pathlib import Path
import ast
import copy
import json
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract as c


class FreshBatchTests(unittest.TestCase):
    def environment(self):
        env = dict(GITHUB_REPOSITORY=c.REPOSITORY, GITHUB_REF='refs/heads/' + c.BRANCH,
                   GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1',
                   RUNNER_ENVIRONMENT='github-hosted', RUNNER_OS='Linux',
                   GITHUB_SHA='a' * 40, GITHUB_RUN_ID='123')
        event = dict(created=True, deleted=False, before='0' * 40, after='a' * 40,
                     repository=dict(full_name=c.REPOSITORY, private=False))
        return env, event

    def test_fixed_plan(self):
        rows = c.plan()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r['cell_id'] for r in rows}), 16)
        self.assertEqual(len({r['seed'] for r in rows}), 16)
        self.assertEqual({r['speaker'] for r in rows}, set(c.VOICES))
        self.assertEqual({r['intended_text'] for r in rows}, set(c.TEXTS))
        for voice in c.VOICES:
            self.assertEqual(sum(r['speaker'] == voice for r in rows), 4)

    def test_preserved_voice_roles(self):
        rows = c.plan()
        self.assertEqual(sum(r['source_role'] == 'train' for r in rows), 12)
        for row in rows:
            self.assertEqual(row['intended_split'], 'development-a' if row['speaker'] == 'Serena' else 'train')
            self.assertEqual(row['voice_group'], row['source_family_id'])
            self.assertTrue(row['synthetic'])
            self.assertIsNone(row['actual_text'])
            for name in ('ctc_training_allowed', 'evaluation_gold_allowed', 'qualification_allowed', 'human_gold'):
                self.assertIs(row[name], False)

    def test_first_public_execution(self):
        c.verify_environment(*self.environment())

    def test_reject_wrong_environment(self):
        for key, value in dict(GITHUB_REPOSITORY='other/repo', GITHUB_REF='refs/heads/main',
                GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_RUN_ATTEMPT='2',
                RUNNER_ENVIRONMENT='self-hosted', RUNNER_OS='Windows',
                GITHUB_SHA='not-a-sha', GITHUB_RUN_ID='0').items():
            with self.subTest(key=key):
                env, event = self.environment()
                env[key] = value
                with self.assertRaises(ValueError):
                    c.verify_environment(env, event)

    def test_reject_event_mutations(self):
        for key, value in dict(created=False, deleted=True, before='b' * 40, after='b' * 40).items():
            env, event = self.environment()
            event[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.verify_environment(env, event)
        for value in (True, 'false', None, 0):
            env, event = self.environment()
            event['repository']['private'] = value
            with self.subTest(private=value), self.assertRaises(ValueError):
                c.verify_environment(env, event)

    def test_audio_flags_never_admit(self):
        for samples, eos, valid in (([0.1, -0.1], True, True), ([0.0], True, False),
                                    ([1.0], True, False), ([0.1], False, False)):
            result = c.signal_metrics(samples, 24000, eos)
            self.assertIs(result['technically_valid'], valid)
            self.assertIs(result['ctc_training_allowed'], False)
            self.assertIs(result['qualification_allowed'], False)

    def test_invalid_audio_geometry(self):
        for samples, rate, eos in (([], 24000, True), ([float('nan')], 24000, True),
                ([float('inf')], 24000, True), ([0.1], 16000, True),
                ([0.1], 24000, 'true'), ([0.1] * 288001, 24000, True)):
            with self.assertRaises(ValueError):
                c.signal_metrics(samples, rate, eos)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'receipt.json'
            c.save(target, {'text': '你好小窝', 'allowed': False})
            self.assertEqual(json.loads(target.read_bytes()), {'text': '你好小窝', 'allowed': False})
            self.assertFalse(target.with_name('receipt.json.tmp').exists())
            self.assertEqual(len(c.file_hash(target)), 64)

    def test_all_new_source_compiles_without_execution(self):
        for name in ('contract.py', 'worker.py', 'run.py', 'test_fresh.py'):
            ast.parse((HERE / name).read_bytes(), filename=name)

    def test_workflow_has_no_pr_inference(self):
        root = HERE.parents[1]
        text = (root / '.github/workflows/fresh-qwen16-20261030.yml').read_text()
        self.assertIn("github.ref == 'refs/heads/" + c.BRANCH + "'", text)
        self.assertIn('github.event.created == true', text)
        self.assertIn('github.run_attempt == 1', text)
        self.assertIn('github.event.repository.private == false', text)
        self.assertIn('needs: fresh-data-contracts', text)
        self.assertIn('ref: ' + c.RETAINED_COMMIT, text)
        self.assertNotIn('workflow_dispatch:', text)
        self.assertNotIn('secrets.', text)
        self.assertNotIn('continue-on-error:', text)
        self.assertIn('persist-credentials: false', text)
        self.assertIn('retention-days: 14', text)


if __name__ == '__main__':
    unittest.main()
