"""Execute full invented promotion bundles without Torch, models or network."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'training')]
import promotion_fixtures as fixtures
import verify_model_promotion_bundle as promotion


def leaves(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, prefix + (index,))
    else:
        yield prefix, value


def replace(value, path, replacement):
    for part in path[:-1]:
        value = value[part]
    value[path[-1]] = replacement


class PromotionAdmissionTests(unittest.TestCase):
    def check(self, documents):
        with tempfile.TemporaryDirectory(prefix='promotion-admission-') as temporary:
            root = Path(temporary)
            fixtures.materialize(root / 'artifact', documents)
            args = argparse.Namespace(artifact_root=root / 'artifact', dist=root / 'dist',
                repository='jiying2007/kws-pipeline', training_run_id=123,
                expected_head_sha=fixtures.HEAD, artifact_id=456,
                artifact_digest='sha256:' + 'c' * 64, artifact_size=42)
            # Only the external Git API is substituted. The complete production
            # consumer reads, hashes and validates all real fixture files.
            with patch.object(promotion, 'git_tree', return_value=fixtures.TREE):
                return promotion.verify(args)

    def test_complete_valid_bundle_and_stdlib_cli(self):
        result = self.check(fixtures.documents())
        self.assertEqual(result['schema_version'], 4)
        self.assertEqual(result['source']['training_run_id'], 123)
        self.assertEqual(result['acceptance']['qualification_false_accepts'], 0)
        product = self.check(fixtures.documents(product=True))
        self.assertEqual(product['source']['training_event_name'], 'workflow_dispatch')
        self.check(fixtures.documents(product=True, label_prior=True))
        for mode in ([], ['-O'], ['-OO']):
            subprocess.run([sys.executable, '-S', '-B', *mode,
                str(ROOT / 'tools/verify_model_promotion_bundle.py'), '--help'],
                check=True, capture_output=True, text=True)
        self.assertFalse({'torch', 'numpy', 'whisper'} & sys.modules.keys())

    def test_full_cli_uses_only_offline_git_api_stub(self):
        with tempfile.TemporaryDirectory(prefix='promotion-cli-') as temporary:
            root = Path(temporary)
            fixtures.materialize(root / 'artifact', fixtures.documents(product=True))
            (root / 'bin').mkdir()
            gh = root / 'bin/gh'
            gh.write_text(f'#!{sys.executable}\nimport json, sys\n'
                f'if sys.argv[1:] != ["api", "repos/jiying2007/kws-pipeline/git/commits/{fixtures.HEAD}"]: raise SystemExit(91)\n'
                f'print(json.dumps({{"tree": {{"sha": "{fixtures.TREE}"}}}}))\n')
            gh.chmod(0o755)
            mode = ['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else []
            result = subprocess.run([sys.executable, '-S', '-B', *mode,
                str(ROOT / 'tools/verify_model_promotion_bundle.py'),
                '--artifact-root', str(root / 'artifact'), '--dist', str(root / 'dist'),
                '--repository', 'jiying2007/kws-pipeline', '--training-run-id', '123',
                '--expected-head-sha', fixtures.HEAD, '--artifact-id', '456',
                '--artifact-digest', 'sha256:' + 'c' * 64, '--artifact-size', '42'],
                env=dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ.get('PATH', '')),
                check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['source']['head_tree_sha'], fixtures.TREE)
            self.assertTrue((root / 'dist/MODEL_SHA256SUMS').is_file())

    def test_every_recorded_boolean_rejects_coercions(self):
        original = fixtures.documents(product=True)
        paths = [(path, value) for path, value in leaves(original) if type(value) is bool]
        self.assertGreater(len(paths), 40)
        for path, expected in paths:
            for value in ('false', int(expected), None, [], {}, not expected):
                with self.subTest(path=path, value=value):
                    changed = copy.deepcopy(original)
                    replace(changed, path, value)
                    with self.assertRaises((ValueError, TypeError)):
                        self.check(changed)

    def test_every_count_rejects_noncanonical_types_and_negative(self):
        original = fixtures.documents(product=True)
        paths = [(path, value) for path, value in leaves(original) if type(value) is int]
        self.assertGreater(len(paths), 60)
        for path, expected in paths:
            for value in (True, float(expected), str(expected), -1, None):
                with self.subTest(path=path, value=value):
                    changed = copy.deepcopy(original)
                    replace(changed, path, value)
                    with self.assertRaises((ValueError, TypeError)):
                        self.check(changed)

    def test_fractional_error_counts_cannot_be_rounded_to_zero(self):
        for field in ('false_accepts', 'false_rejects'):
            for value in (0.9, -0.9):
                changed = fixtures.documents()
                changed['best/qualification/summary.json'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(changed)

    def test_recorded_numbers_are_finite(self):
        original = fixtures.documents(product=True)
        paths = [(path, value) for path, value in leaves(original) if type(value) is float]
        for path, expected in paths:
            for value in (float('nan'), float('inf'), -float('inf'), str(expected), True):
                with self.subTest(path=path, value=value):
                    changed = copy.deepcopy(original)
                    replace(changed, path, value)
                    with self.assertRaises((ValueError, TypeError)):
                        self.check(changed)

    def test_duplicate_keys_and_overflow_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.json'
            for raw in ('{"qualified":false,"qualified":true}', '{"score":1e999}',
                        '{"score":NaN}', '{"score":Infinity}'):
                path.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    promotion.load_json(path, 'invented fixture')

    def test_label_prior_promotion_rejects_coercion_before_historical_normalizer(self):
        for key, values in [('schema_version', [True, '1', 1.0]),
                            ('frame_count', [True, '123', 123.9, -1]),
                            ('alpha', ['0.3', True]),
                            ('values', [['0.8', '0.1', '0.1'], [True, 0.1, 0.1]])]:
            for value in values:
                changed = fixtures.documents(product=True, label_prior=True)
                changed['best/model-provenance.json']['training']['ctc_label_prior'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.check(changed)

    def test_zero_error_helper_requires_exact_nonnegative_integers(self):
        valid = dict(expected=256, matched=256, false_rejects=0, false_accepts=0)
        promotion.require_zero_error(valid, expected=256, label='fixture')
        for key in valid:
            for value in (True, False, str(valid[key]), valid[key] + 0.9, -1, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    promotion.require_zero_error(dict(valid, **{key: value}), expected=256, label='fixture')


if __name__ == '__main__':
    unittest.main()
