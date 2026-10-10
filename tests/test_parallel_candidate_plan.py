#!/usr/bin/env python3
"""Pure stdlib metadata fixtures; no audio, model, backend, probe or benchmark."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_parallel_candidate_plan import DEFAULT_PLAN, MAX_PLAN_BYTES, load_plan, validate


def members(value, path=()):
    if type(value) is dict:
        for key, item in value.items():
            yield path + (key,), item
            yield from members(item, path + (key,))
    elif type(value) is list:
        for key, item in enumerate(value):
            yield path + (key,), item
            yield from members(item, path + (key,))


def at(value, path):
    for key in path:
        value = value[key]
    return value


class ParallelPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_plan(DEFAULT_PLAN)

    def test_repository_plan_is_shape_only_and_always_disabled(self):
        self.assertEqual(validate(self.plan), dict(metadata_shape_valid=True, evidence_status='NOT_RUN',
            execution_ready=False, input_authenticity_verified=False, numerical_admission=False,
            product_qualified=False, shipping_approved=False))
        for path in self.plan['authority_paths'] + [self.plan['retained_plan_path']]:
            self.assertTrue((ROOT / path).is_file(), path)
        old = json.loads((ROOT / self.plan['retained_plan_path']).read_text())
        self.assertIn('model-architecture', old['controlled_variables_fixed_between_arms'])
        self.assertIsNone(self.plan['supersedes'])

    def test_every_field_required_and_no_extra_fields(self):
        for path, value in members(self.plan):
            if type(path[-1]) is str:
                bad = copy.deepcopy(self.plan)
                del at(bad, path[:-1])[path[-1]]
                with self.subTest(missing=path), self.assertRaises(ValueError):
                    validate(bad)
        for path, value in [((), self.plan), *members(self.plan)]:
            if type(value) is dict:
                bad = copy.deepcopy(self.plan)
                at(bad, path)['extra_execution_override'] = True
                with self.subTest(extra=path), self.assertRaises(ValueError):
                    validate(bad)

    def test_numbers_reject_boolean_float_and_string_coercion(self):
        for path, value in members(self.plan):
            if type(value) is int:
                for replacement in (bool(value), float(value), str(value), None):
                    bad = copy.deepcopy(self.plan)
                    at(bad, path[:-1])[path[-1]] = replacement
                    with self.subTest(path=path, replacement=replacement), self.assertRaises(ValueError):
                        validate(bad)

    def test_all_boolean_controls_are_strict_and_noninvertible(self):
        for path, value in members(self.plan):
            if type(value) is bool:
                for replacement in (int(value), str(value).lower(), None, not value):
                    bad = copy.deepcopy(self.plan)
                    at(bad, path[:-1])[path[-1]] = replacement
                    with self.subTest(path=path, replacement=replacement), self.assertRaises(ValueError):
                        validate(bad)

    def test_unknown_duplicate_extra_and_missing_candidates_rejected(self):
        changes = [lambda p: p['candidates'].append(copy.deepcopy(p['candidates'][0])),
                   lambda p: p['candidates'].pop(),
                   lambda p: p['candidates'][1].update(id='rnn64_control'),
                   lambda p: p['candidates'][1].update(id='unbounded-fourth-arm'),
                   lambda p: p['candidates'][1].update(id=True),
                   lambda p: p['candidates'][1].update(id=[]),
                   lambda p: p['candidates'][0].update(role='frozen-anchor'),
                   lambda p: p['candidates'][1].update(architecture='arbitrary-transformer')]
        for change in changes:
            bad = copy.deepcopy(self.plan)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(bad)
        self.plan['candidates'].reverse()
        self.assertFalse(validate(self.plan)['execution_ready'])

    def test_unobserved_fail_unknown_and_inconclusive_never_become_pass(self):
        for status in ('PENDING', 'NOT_RUN', 'BLOCKED', 'UNKNOWN', 'FAIL'):
            self.plan['candidates'][1]['gates']['candidate_c_export_parity'] = status
            self.assertFalse(validate(self.plan)['numerical_admission'])
        for status in ('PASS', 'passed', 'INCONCLUSIVE', '', 'future-status', True, 1, None, []):
            bad = copy.deepcopy(self.plan)
            bad['candidates'][1]['gates']['candidate_c_export_parity'] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                validate(bad)
        for status in ('PASS', 'UNKNOWN', 'INCONCLUSIVE', True):
            bad = copy.deepcopy(self.plan)
            bad['selection']['verdict'] = status
            with self.subTest(selection=status), self.assertRaises(ValueError):
                validate(bad)

    def test_source_paths_reject_traversal_and_noncanonical_forms(self):
        for path in ('../private', '/tmp/private', 'a/../b', './file', 'a//b', 'a\\b', '.', '', 'a\0b'):
            bad = copy.deepcopy(self.plan)
            bad['candidates'][1]['source']['paths'][0] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(bad)
        bad = copy.deepcopy(self.plan)
        bad['candidates'][1]['source']['paths'].append('LICENSE')
        with self.assertRaises(ValueError):
            validate(bad)

    def test_source_identity_license_and_candidate_binding_are_required(self):
        for key, value in (('commit', 'a' * 40), ('commit', 'a' * 64), ('commit', 1),
                           ('repository', 'https://example.invalid/source'),
                           ('code_license', 'unspecified'), ('weights_license', 'same-as-code'),
                           ('training_data_license', 'same-as-code')):
            bad = copy.deepcopy(self.plan)
            bad['candidates'][1]['source'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(bad)
        for key in ('source_sha', 'source_tree'):
            bad = copy.deepcopy(self.plan)
            bad[key] = '0' * 40
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(bad)
        bad = copy.deepcopy(self.plan)
        bad['candidates'][1]['source'] = copy.deepcopy(bad['candidates'][2]['source'])
        with self.assertRaises(ValueError):
            validate(bad)

    def test_pending_identities_seeds_budgets_cannot_silently_enable_execution(self):
        for path, value in members(self.plan):
            if value is None:
                bad = copy.deepcopy(self.plan)
                at(bad, path[:-1])[path[-1]] = 'a' * 64
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate(bad)
        for key in ('max_screen_arms', 'max_screen_fits', 'max_confirmation_fits', 'max_total_fits',
                    'max_optimizer_updates_per_fit', 'max_concurrent_fits'):
            bad = copy.deepcopy(self.plan)
            bad['budget'][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(bad)

    def test_comparison_admission_calibration_and_support_cannot_be_relaxed(self):
        cases = [('comparison_contract', 'fixed_between_arms', ['architecture']),
                 ('comparison_contract', 'architecture_specific_initialization', 'identical-weights'),
                 ('comparison_contract', 'calibration_policy', 'tune-on-final-heldout'),
                 ('comparison_contract', 'required_candidate_specification', []),
                 ('data_contract', 'required_lineage', ['intended-text-only']),
                 ('data_contract', 'oov_nearphone_role', 'train-oov-as-blank'),
                 ('selection', 'required_metrics', ['clip-accuracy']),
                 ('selection', 'unknown_evidence_verdict', 'PASS'),
                 ('selection', 'insufficient_support_verdict', 'PASS'),
                 ('selection', 'tie_or_tradeoff_action', 'automatically-pick-smallest'),
                 ('selection', 'selected_candidate_id', 'causal_ds_tcn_ctc'),
                 ('historical_anchor', 'd20_scope', 'all-new-candidates-blocked'),
                 ('historical_anchor', 'missing_original_state_blocks', 'all-parallel-comparisons')]
        for section, key, value in cases:
            bad = copy.deepcopy(self.plan)
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate(bad)

    def test_strict_bounded_json_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invented-plan.json'
            for raw in ('{"execution_enabled": false, "execution_enabled": true}',
                        '{"nested": {"a": 1, "a": 2}}', '{"a": NaN}', '{"a": Infinity}',
                        '{"a": -Infinity}', ' ' * (MAX_PLAN_BYTES + 1)):
                path.write_text(raw)
                with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                    load_plan(path)

    def test_cli_has_no_execution_mode_or_numerical_import(self):
        tree = ast.parse((ROOT / 'tools/validate_parallel_candidate_plan.py').read_text())
        imports = [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(set(imports), {'__future__', 'argparse', 'json', 'pathlib', 're'})
        for flags in ([], ['--execute'], ['--train'], ['--probe']):
            process = subprocess.run([sys.executable, '-S', '-B', str(ROOT / 'tools/validate_parallel_candidate_plan.py'),
                                      *flags], capture_output=True, text=True)
            with self.subTest(flags=flags):
                self.assertEqual(process.returncode == 0, not flags)
                if not flags:
                    self.assertFalse(json.loads(process.stdout)['execution_ready'])

    def test_only_existing_required_ci_metadata_lane_is_used(self):
        ci = (ROOT / '.github/workflows/ci.yml').read_text()
        shared = ci.split('  python-contracts:\n', 1)[1].split('\n  research-sources:', 1)[0]
        for mode in ('', ' -O', ' -OO'):
            command = 'python3 -S -B' + mode + ' tests/test_parallel_candidate_plan.py'
            self.assertEqual(ci.count(command + '\n'), 1)
            self.assertIn(command, shared)
        self.assertIn('"$RUNNER_TEMP/check-tracked-worktree.sh" "Disabled parallel candidate plan metadata"', shared)


if __name__ == '__main__':
    unittest.main()
