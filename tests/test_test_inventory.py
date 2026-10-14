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

    def test_positive_timeout_declares_the_real_python_command(self):
        for duration in ('120s', '1', '0.5s', '.5m', '1.', '2h', '1d'):
            with self.subTest(duration=duration):
                command = (f'timeout {duration} .venv-objective/bin/python -B '
                           'tests/test_ok.py --repo . --receipt "$RUNNER_TEMP/numeric.json"')
                report = self.check('run: ' + command + '\n')
                self.assertEqual(report['missing'], [])
                self.assertFalse(report['execution_coverage_proven'])
                self.assertEqual(report['files']['test_ok.py'][0]['kind'], 'script-selected')
                self.assertEqual(list(inventory.python_commands(command))[0][0],
                                 '.venv-objective/bin/python')

    def test_timeout_continuation_and_trailing_comment(self):
        report = self.check('run: |\n  timeout 120s .venv-objective/bin/python -B ' +
                            '\\\n    tests/test_ok.py # source declaration; not proof\n')
        self.assertEqual(report['missing'], [])

    def test_timeout_zero_negative_or_malformed_duration_is_not_declared(self):
        for duration in ('0', '0s', '0.0m', '000.000', '-1', '-0.5s', '+1s',
                         '1e3', 'inf', 'nan', '1ms', '1ss', '1.2.3s', '.', 's',
                         '$SECONDS', '${SECONDS}'):
            with self.subTest(duration=duration):
                command = f'timeout {duration} python3 tests/test_ok.py'
                self.assertEqual(self.check('run: ' + command)['missing'], ['test_ok.py'])
                self.assertEqual(list(inventory.python_commands(command)), [])
        for command in ('timeout', 'timeout 120s', 'timeout python3 tests/test_ok.py',
                        'timeout 120s "python3 tests/test_ok.py',
                        'timeout --signal=TERM 120s python3 tests/test_ok.py',
                        'timeout -- 120s python3 tests/test_ok.py'):
            with self.subTest(command=command):
                self.assertEqual(self.check('run: ' + command)['missing'], ['test_ok.py'])

    def test_timeout_mentions_shell_and_nested_wrappers_are_not_declared(self):
        for command in ('echo timeout 120s python3 tests/test_ok.py',
                        '# timeout 120s python3 tests/test_ok.py',
                        'sh -c "timeout 120s python3 tests/test_ok.py"',
                        'timeout 120s sh -c "python3 tests/test_ok.py"',
                        'env timeout 120s python3 tests/test_ok.py',
                        'timeout 120s timeout 10s python3 tests/test_ok.py',
                        'timeout 120s python3 tests/test_ok.py; echo done',
                        'timeout 120s python3 tests/test_ok.py && echo done',
                        'timeout 120s python3 tests/test_ok.py || true',
                        'timeout 120s python3 tests/test_ok.py | cat',
                        'timeout 120s python3 tests/test_ok.py > result.txt',
                        'timeout 120s python3 tests/test_ok.py $(echo injected)',
                        'timeout 120s python3 tests/test_ok.py `echo injected`',
                        'timeout 120s python3 tests/test_ok.py &'):
            with self.subTest(command=command):
                self.assertEqual(self.check('run: ' + command)['missing'], ['test_ok.py'])
                self.assertEqual(list(inventory.python_commands(command)), [])
        self.assertEqual(self.check("run: |\n  timeout 120s python3 - <<'PY'\n"
                                    '  timeout 120s python3 tests/test_ok.py\n  PY\n')
                         ['missing'], ['test_ok.py'])

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
