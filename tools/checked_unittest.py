#!/usr/bin/env python3
"""Fail-closed unittest execution evidence for explicitly inventoried CI suites.

This is case execution accounting, not line/branch or whole-repository coverage.
The checked-in count and ID digest must be reviewed when tests change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / 'tools/test_execution_contracts.json'
MODES = ('normal', 'O', 'OO')


def ids_digest(ids):
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()


def case_ids(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from case_ids(case)
        else:
            yield case.id()


def contracts():
    value = json.loads(CONTRACTS.read_text())
    if value['schema'] != 1:
        raise ValueError('unsupported execution contract schema')
    return value['suites']


def validate_ids(ids, spec):
    if (type(ids) is not list or any(type(name) is not str or not name for name in ids) or
            not ids or len(ids) != len(set(ids)) or
            type(spec['count']) is not int or spec['count'] <= 0 or
            len(ids) != spec['count'] or ids_digest(ids) != spec['ids_sha256']):
        raise ValueError('discovery count/IDs differ from the reviewed execution contract')
    if any('_FailedTest.' in name for name in ids):
        raise ValueError('unittest import/discovery failed')


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_ids = []
        self.finished_ids = []
        self.successful_ids = []

    def startTest(self, test):
        self.started_ids.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.finished_ids.append(test.id())
        super().stopTest(test)

    def addSuccess(self, test):
        self.successful_ids.append(test.id())
        super().addSuccess(test)


def run_identity():
    return {name: os.environ.get(name, '') for name in
            ('GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_JOB')}


def validate_report(report, spec, suite, mode):
    if type(report['schema']) is not int or report['schema'] != 1 or report['suite'] != suite or report['mode'] != mode:
        raise ValueError('report schema/suite/mode mismatch')
    if report['contract_sha256'] != hashlib.sha256(CONTRACTS.read_bytes()).hexdigest():
        raise ValueError('report belongs to a different execution contract')
    if report['run_identity'] != run_identity():
        raise ValueError('report belongs to another source/run/attempt/job')
    ids = report['discovered_ids']
    validate_ids(ids, spec)
    if (type(report['discovered_count']) is not int or type(report['tests_run']) is not int or
            report['discovered_count'] != len(ids) or report['tests_run'] != len(ids)):
        raise ValueError('zero/partial test execution')
    for key in ('started_ids', 'finished_ids'):
        if Counter(report[key]) != Counter(ids):
            raise ValueError('missing, duplicate or unexpected actual test IDs: ' + key)
    skipped = report['skipped']
    if type(skipped) is not list or any(type(item) is not dict or set(item) != {'id', 'reason'} or
            type(item['id']) is not str or type(item['reason']) is not str for item in skipped):
        raise ValueError('malformed skip evidence')
    if {item['id']: item['reason'] for item in skipped} != spec['allowed_skips']:
        raise ValueError('unexpected or stale allowed skip')
    if len(skipped) != len(spec['allowed_skips']):
        raise ValueError('duplicate skip')
    if Counter(report['successful_ids'] + [item['id'] for item in skipped]) != Counter(ids):
        raise ValueError('not every discovered case succeeded or had an exact allowed skip')
    for key in ('failures', 'errors', 'expected_failures', 'unexpected_successes'):
        if report[key]:
            raise ValueError('non-passing unittest outcome: ' + key)
    if report['violations'] or report['successful'] is not True:
        raise ValueError('unsuccessful checked execution')


def run_checked_suite(suite, name, *, report_dir=None):
    spec = contracts()[name]
    ids = list(case_ids(suite))
    mode = MODES[sys.flags.optimize]
    report = dict(schema=1, suite=name, mode=mode,
                  contract_sha256=hashlib.sha256(CONTRACTS.read_bytes()).hexdigest(),
                  run_identity=run_identity(), python=sys.version,
                  discovered_ids=ids, discovered_count=len(ids), tests_run=0,
                  started_ids=[], finished_ids=[], successful_ids=[], skipped=[],
                  failures=[], errors=[], expected_failures=[], unexpected_successes=[],
                  violations=[], successful=False)
    try:
        validate_ids(ids, spec)
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
        report.update(tests_run=result.testsRun, started_ids=result.started_ids,
                      finished_ids=result.finished_ids, successful_ids=result.successful_ids,
                      skipped=[dict(id=test.id(), reason=reason) for test, reason in result.skipped],
                      failures=[test.id() for test, _ in result.failures],
                      errors=[test.id() for test, _ in result.errors],
                      expected_failures=[test.id() for test, _ in result.expectedFailures],
                      unexpected_successes=[test.id() for test in result.unexpectedSuccesses],
                      successful=result.wasSuccessful())
        validate_report(report, spec, name, mode)
    except Exception as exc:
        report['violations'].append(str(exc))
        report['successful'] = False
    destination = report_dir or os.environ.get('KWS_TEST_REPORT_DIR')
    if destination:
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{name}-{mode}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)
    print('CHECKED_UNITTEST ' + json.dumps({key: report[key] for key in
          ('suite', 'mode', 'discovered_count', 'tests_run', 'skipped', 'violations', 'successful')}, sort_keys=True))
    return 0 if report['successful'] else 1


def verify_reports(directory, names):
    specs = contracts()
    for name in names:
        spec = specs[name]
        if spec['modes'] != list(MODES):
            raise ValueError('all three Python modes are required')
        for mode in MODES:
            path = directory / f'{name}-{mode}.json'
            validate_report(json.loads(path.read_text()), spec, name, mode)
    print(f'verified actual test IDs/counts/outcomes in {len(names) * len(MODES)} suite-mode reports')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='append', required=True)
    parser.add_argument('--verify-reports', type=Path)
    args = parser.parse_args(argv)
    if args.verify_reports is not None:
        verify_reports(args.verify_reports, args.suite)
        return 0
    if len(args.suite) != 1:
        parser.error('execute one suite at a time to avoid module-name collisions')
    name = args.suite[0]
    spec = contracts()[name]
    sys.path.insert(0, str(ROOT / spec['root']))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(spec['modules'])
    return run_checked_suite(suite, name)


if __name__ == '__main__':
    raise SystemExit(main())
