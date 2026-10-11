"""Static integer/format fixtures only; no model, device probe or measurements."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import model_family_resource_estimate as estimate


class ResourceIntegerTests(unittest.TestCase):
    def setUp(self):
        self.contract_path = ROOT / 'configs/training/model-family-resource-contract-v1.json'
        self.contract = estimate.load_contract(self.contract_path)

    def test_frozen_valid_report_is_unchanged_and_not_measurement(self):
        report = estimate.build_report(self.contract)
        estimate.verify_contract(self.contract, report)
        digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, 'd36cf5a82be9e4c512f5a04d8dbd25ecbc5d9915f32cdd40c253420fb5922bba')
        self.assertEqual(report['evidence_class'], 'static-model-family-resource-estimate')
        self.assertFalse(report['hosted_timing_is_shipping_evidence'])
        self.assertTrue(report['requires_target_board_measurement_for_shipping'])

    def test_exact_positive_integers_only(self):
        for good in (1, 32, 64, 420):
            self.assertEqual(estimate._positive_int(good, 'size'), good)
        for bad in (True, False, 0, -1, 1.0, 1.9, '1', None, [], {}, float('inf'), float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimate._positive_int(bad, 'size')

    def test_fractional_dimensions_never_silently_truncate(self):
        for family in ('rnn', 'gru'):
            for function in (estimate.estimate_model_bytes, estimate.estimate_dense_macs_per_step):
                for index in range(3):
                    dimensions = [32, 64, 5]
                    dimensions[index] += 0.9
                    with self.subTest(family=family, function=function.__name__, index=index), self.assertRaises(ValueError):
                        function(family, *dimensions)
            with self.assertRaises(ValueError):
                estimate.estimate_non_linear_units_per_step(family, 32.9)

    def test_direct_report_dimension_validation_is_not_bypassed(self):
        for bad in (True, 32.0, 32.9, '32', 0, -1):
            contract = copy.deepcopy(self.contract); contract['feature_dim'] = bad
            with self.subTest(field='feature_dim', value=bad), self.assertRaises(ValueError):
                estimate.build_report(contract)
        for group, field in (('candidates', 'hidden_dim'), ('vocabularies', 'size')):
            for bad in (True, 5.0, 5.9, '5', 0, -1):
                contract = copy.deepcopy(self.contract); contract[group][0][field] = bad
                with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                    estimate.build_report(contract)

    def test_existing_feature_and_runtime_bounds_remain_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'contract.json'
            for bad in (True, 32.0, 32.9, '32', 0, -1, 41):
                contract = copy.deepcopy(self.contract); contract['feature_dim'] = bad
                path.write_text(json.dumps(contract))
                with self.subTest(feature_dim=bad), self.assertRaises(ValueError):
                    estimate.load_contract(path)
        contract = copy.deepcopy(self.contract); contract['candidates'][0]['hidden_dim'] = 65
        with self.assertRaisesRegex(ValueError, 'runtime/export bound'):
            estimate.build_report(contract)


if __name__ == '__main__':
    unittest.main()
