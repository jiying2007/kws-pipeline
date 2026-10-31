#!/usr/bin/env python3
"""Strict isolated regressions for CI failure-report retention; metadata only."""
from pathlib import Path
import json
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_checked_unittest as original
checked = original.checked
fixture = original.fixture


class FailureRetention(unittest.TestCase):
    # Reuse reviewed fixture helpers, not its TestCase inheritance/discovery.
    setUp = original.CheckedExecution.setUp
    execute = original.CheckedExecution.execute

    def test_failed_report_retains_actual_ids_without_becoming_passing_evidence(self):
        self.fail("CI_RETENTION_ACCEPTANCE_ONLY_PYTHON_CONTRACTS_DO_NOT_MERGE")
        rc, report = self.execute(fixture('fail'))
        self.assertEqual(rc, 1)
        self.assertFalse(report['successful'])
        self.assertEqual(report['tests_run'], 1)
        self.assertEqual(report['discovered_ids'], report['started_ids'])
        self.assertEqual(report['started_ids'], report['finished_ids'])
        self.assertEqual(report['failures'], report['finished_ids'])
        self.assertEqual(report['successful_ids'], [])
        self.assertEqual(list(self.root.glob('*.tmp')), [])
        with self.assertRaises(ValueError):
            checked.validate_report(report, self.spec, 'fixture', self.mode)
        # Retaining failure JSON must not make the three-mode verifier accept it.
        for mode in checked.MODES:
            (self.root / ('fixture-' + mode + '.json')).write_text(
                json.dumps(dict(report, mode=mode)))
        with self.assertRaises(ValueError):
            checked.verify_reports(self.root, ['fixture'])


    def test_ci_retains_checked_reports_after_failure_but_not_cancellation(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        jobs = dict(re.findall(r'^  ([a-zA-Z0-9_-]+):\n(.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)',
                               workflow, re.M | re.S))
        for job, title, artifact in (
                ('python-contracts', 'compiler-independent', 'python-contracts'),
                ('asr-offline-posterior', 'posterior', 'posterior')):
            with self.subTest(job=job):
                name = 'Retain actual ' + title + ' execution evidence'
                steps = re.findall(r'^      - .*?(?=^      - |\Z)',
                                   jobs[job], re.M | re.S)
                uploads = [step for step in steps
                           if step.splitlines()[0] == '      - name: ' + name]
                self.assertEqual(len(uploads), 1)
                # A status function overrides implicit success(), but !cancelled()
                # deliberately does not promise retention for canceled jobs.
                self.assertEqual(uploads[0].rstrip().splitlines(), [
                    '      - name: ' + name,
                    '        if: ${{ !cancelled() }}',
                    '        uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7.0.2',
                    '        with:',
                    '          name: checked-unittest-' + artifact,
                    '          path: ${{ env.KWS_TEST_REPORT_DIR }}/*.json',
                    '          if-no-files-found: error',
                ])



if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'tools'))
    import checked_unittest as runner
    runner.CONTRACTS = ROOT / 'tools/ci_failure_retention_contracts.json'
    raise SystemExit(runner.main(['--suite', 'ci-failure-retention', *sys.argv[1:]]))
