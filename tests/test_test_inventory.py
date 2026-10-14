#!/usr/bin/env python3
"""Declaration inventory regressions; no assertions disabled by Python -O."""
from __future__ import annotations
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_inventory', ROOT / 'tools/test_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class Inventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.tests = self.root / 'tests'; self.tests.mkdir()
        (self.tests / 'test_ok.py').write_text('')
        self.workflow = self.root / 'ci.yml'

    def check(self, source):
        self.workflow.write_text(source)
        return inventory.inventory(self.tests, [self.workflow])

    def test_actual_command_is_declared_but_never_execution_proof(self):
        report = self.check('run: python3 tests/test_ok.py\n')
        self.assertEqual(report['missing'], [])
        self.assertFalse(report['execution_coverage_proven'])
        (self.tests / 'test_missing.py').write_text('')
        self.assertEqual(self.check('run: python3 tests/test_ok.py')['missing'], ['test_missing.py'])

    def test_comments_filters_echo_and_heredoc_are_not_test_execution_declarations(self):
        for source in ('# python3 tests/test_ok.py\n', 'paths: [tests/test_ok.py]\n',
                       'run: echo python3 tests/test_ok.py\n',
                       'run: python3 tools/check.py tests/test_ok.py\n',
                       "run: |\n  python3 - <<'PY'\n  python3 tests/test_ok.py\n  PY\n",
                       'run: |\n  # python3 tests/test_ok.py\n'):
            with self.subTest(source=source):
                self.assertEqual(self.check(source)['missing'], ['test_ok.py'])

    def test_discovery_pattern_and_folded_python_command(self):
        for source in ('run: python3 -m unittest discover -s tests -p test_*.py\n',
                       'run: >-\n  python3\n  tests/test_ok.py\n'):
            self.assertEqual(self.check(source)['missing'], [])

    def test_manual_path_and_hardware_scopes_are_distinct(self):
        self.workflow = self.root / 'optional.yml'
        for prefix, scope in [('on:\n  workflow_dispatch:\n', 'manual-or-scheduled'),
                              ('on:\n  pull_request:\n    paths: [tests/**]\n', 'path-filtered'),
                              ('runs-on: self-hosted\n', 'hardware-or-self-hosted')]:
            report = self.check(prefix + 'run: python3 tests/test_ok.py\n')
            self.assertEqual(report['files']['test_ok.py'][0]['scope'], scope)

    def test_selected_class_does_not_claim_whole_file_execution(self):
        report = self.check('run: python3 tests/test_ok.py SomeClass\n')
        self.assertEqual(report['files']['test_ok.py'][0]['kind'], 'script-selected')

    def test_conditional_job_or_step_does_not_claim_required_mainline(self):
        for guard in ('    if: false\n    steps:\n      - name: test\n',
                      '    steps:\n      - name: test\n        if: false\n'):
            report = self.check('jobs:\n  optional:\n' + guard +
                                '        run: python3 tests/test_ok.py\n')
            self.assertEqual(report['files']['test_ok.py'][0]['scope'], 'conditional-mainline')

    def test_unknown_exclusions_fail(self):
        self.workflow.write_text('')
        with self.assertRaises(ValueError):
            inventory.inventory(self.tests, [self.workflow], ['test_missing.py'])

    def test_strict_runner_maps_only_execution_to_the_declared_source(self):
        report = inventory.inventory(ROOT / 'tests', [ROOT / '.github/workflows/ci.yml'])
        for name in ('test_model_promotion_admission.py', 'test_model_promotion_source.py'):
            self.assertEqual({r['kind'] for r in report['files'][name]}, {'checked-unittest'})
        self.assertEqual(inventory.declared_paths('python3 tools/checked_unittest.py --verify-reports /tmp --suite promotion-source'), [])


if __name__ == '__main__':
    unittest.main()
