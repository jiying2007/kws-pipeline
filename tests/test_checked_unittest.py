#!/usr/bin/env python3
"""Synthetic, stdlib-only counterexamples for CI execution evidence."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checked_unittest', ROOT / 'tools/checked_unittest.py')
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)


def fixture(kind='success'):
    class Fixture(unittest.TestCase):
        def test_one(self):
            if kind == 'skip': self.skipTest('unexpected fixture skip')
            if kind == 'fail': self.fail('fixture failure')
            if kind == 'error': raise ValueError('fixture error')
            if kind == 'subtest':
                with self.subTest(part=1): self.fail('fixture subtest failure')
            if kind == 'xfail': self.fail('expected failure remains non-passing')
    if kind in ('xfail', 'xpass'):
        Fixture.test_one = unittest.expectedFailure(Fixture.test_one)
    return unittest.defaultTestLoader.loadTestsFromTestCase(Fixture)


class CheckedExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        ids = list(checked.case_ids(fixture()))
        self.spec = dict(count=len(ids), ids_sha256=checked.ids_digest(ids),
                         modes=list(checked.MODES), allowed_skips={})
        self.contract = self.root / 'contracts.json'
        self.contract.write_text(json.dumps(dict(schema=1, suites={'fixture': self.spec})))
        self.patch = patch.object(checked, 'CONTRACTS', self.contract)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.mode = checked.MODES[sys.flags.optimize]

    def execute(self, suite=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = checked.run_checked_suite(fixture() if suite is None else suite,
                                           'fixture', report_dir=self.root)
        return rc, json.loads((self.root / ('fixture-' + self.mode + '.json')).read_text())

    def test_records_exact_actual_ids_counts_modes_and_empty_skips(self):
        rc, report = self.execute()
        self.assertEqual(rc, 0)
        self.assertEqual(report['tests_run'], 1)
        self.assertEqual(report['mode'], self.mode)
        self.assertEqual(report['discovered_ids'], report['started_ids'])
        self.assertEqual(report['started_ids'], report['finished_ids'])
        self.assertEqual(report['finished_ids'], report['successful_ids'])
        self.assertEqual(report['skipped'], [])
        checked.validate_report(report, self.spec, 'fixture', self.mode)

    def test_empty_discovery_is_rejected_before_execution(self):
        rc, report = self.execute(unittest.TestSuite())
        self.assertEqual(rc, 1)
        self.assertEqual(report['tests_run'], 0)
        self.assertTrue(report['violations'])

    def test_missing_or_duplicate_discovery_ids_are_rejected(self):
        for suite in (unittest.TestSuite([fixture(), fixture()]),
                      unittest.defaultTestLoader.loadTestsFromName('missing_fixture_module')):
            with self.subTest(suite=suite):
                self.assertEqual(self.execute(suite)[0], 1)

    def test_changed_id_with_same_count_is_rejected(self):
        self.spec['ids_sha256'] = '0' * 64
        self.contract.write_text(json.dumps(dict(schema=1, suites={'fixture': self.spec})))
        self.assertEqual(self.execute()[0], 1)

    def test_skip_failure_error_subtest_expected_failure_and_unexpected_success_fail(self):
        for kind in ('skip', 'fail', 'error', 'subtest', 'xfail', 'xpass'):
            with self.subTest(kind=kind):
                rc, report = self.execute(fixture(kind))
                self.assertEqual(rc, 1)
                self.assertFalse(report['successful'])
                if kind == 'skip':
                    self.assertEqual(report['skipped'][0]['reason'], 'unexpected fixture skip')

    def test_allowed_skips_are_exact_id_and_reason_and_cannot_go_stale(self):
        test_id = list(checked.case_ids(fixture()))[0]
        self.spec['allowed_skips'] = {test_id: 'unexpected fixture skip'}
        self.contract.write_text(json.dumps(dict(schema=1, suites={'fixture': self.spec})))
        self.assertEqual(self.execute(fixture('skip'))[0], 0)
        self.assertEqual(self.execute(fixture())[0], 1)

    def test_partial_execution_is_rejected(self):
        class Partial(unittest.TestSuite):
            def run(self, result, debug=False):
                return result
        rc, report = self.execute(Partial([fixture()]))
        self.assertEqual(rc, 1)
        self.assertEqual(report['discovered_count'], 1)
        self.assertEqual(report['tests_run'], 0)

    def test_report_mutations_fail_closed(self):
        _, good = self.execute()
        mutations = [('mode', 'wrong'), ('suite', 'wrong'), ('schema', 2),
                     ('contract_sha256', '0' * 64), ('run_identity', {}),
                     ('tests_run', 0), ('tests_run', True), ('discovered_count', True),
                     ('schema', True), ('discovered_count', 0), ('started_ids', []),
                     ('finished_ids', []), ('successful_ids', []), ('successful', False),
                     ('violations', ['not successful']), ('errors', ['fixture']),
                     ('expected_failures', ['fixture'])]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(good); bad[key] = value
                with self.assertRaises(ValueError):
                    checked.validate_report(bad, self.spec, 'fixture', self.mode)

    def test_all_three_modes_required_and_missing_report_fails(self):
        _, good = self.execute()
        with self.assertRaises(FileNotFoundError):
            checked.verify_reports(self.root, ['fixture'])
        for mode in checked.MODES:
            report = dict(good, mode=mode)
            (self.root / ('fixture-' + mode + '.json')).write_text(json.dumps(report))
        checked.verify_reports(self.root, ['fixture'])
        (self.root / 'fixture-OO.json').write_text(json.dumps(dict(good, mode='O')))
        with self.assertRaises(ValueError):
            checked.verify_reports(self.root, ['fixture'])

    def test_stale_run_attempt_is_rejected(self):
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='1'):
            _, good = self.execute()
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='2'):
            with self.assertRaises(ValueError):
                checked.validate_report(good, self.spec, 'fixture', self.mode)

    def test_source_contracts_are_nonzero_unique_and_three_mode(self):
        source = json.loads((ROOT / 'tools/test_execution_contracts.json').read_text())
        for name, entry in source['suites'].items():
            self.assertGreater(entry['count'], 0, name)
            self.assertEqual(entry['modes'], list(checked.MODES), name)
            self.assertEqual(len(entry['modules']), len(set(entry['modules'])), name)
            self.assertEqual(entry['allowed_skips'], {}, name)
            self.assertEqual(len(entry['ids_sha256']), 64, name)


if __name__ == '__main__':
    unittest.main()
