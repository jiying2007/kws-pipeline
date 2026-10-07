"""Saved-data tests for the new context/prefix boundary only."""
from pathlib import Path
import argparse
import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_context as verifier
ARGS = None

class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = verifier.staged(ARGS.core_source, ARGS.core_data)
        cls.binding, cls.scorer, cls.comparator, cls.work, _ = cls.context.__enter__()
        cls.records = [json.loads(line) for line in (ARGS.data_dir / 'original_A20.raw.jsonl').read_bytes().splitlines()]
        cls.manifest = json.loads((cls.work / 'metadata/manifest.json').read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def compare(self, records, manifest=None):
        return self.comparator.compare_prefix(records, self.manifest if manifest is None else manifest)

    def callback_at(self, rows, center):
        return next(r for r in rows if r.get('kind') == 'callback' and
                    r['recording'] == 'M1' and center in r['centers'])

    def test_complete_exact_report_and_early_activation_accounting(self):
        result = verifier.verify(ARGS.data_dir, ARGS.core_source, ARGS.core_data)
        self.assertEqual(result['observations_bytes_reproduced'], 30514)
        self.assertEqual((result['model_rows'], result['decoded_rows']), (70, 69))
        self.assertEqual(result['prefix_float32_scalars_equal'], 300)
        self.assertEqual(result['modified_events'], {'M1': 1, 'M2': 0})

    def test_last_supported_row_difference_detected(self):
        rows = copy.deepcopy(self.records)
        callback = self.callback_at(rows, 69)
        callback['logits'][callback['centers'].index(69)][0] += 1.0
        result = self.compare(rows)
        self.assertEqual(result['status'], 'PREFIX_LOGIT_DIFFERENCE_OBSERVED')
        self.assertEqual(result['streams'][0]['differing_centers'], [69])

    def test_new_context_difference_is_outside_original_prefix(self):
        rows = copy.deepcopy(self.records)
        callback = self.callback_at(rows, 72)
        callback['logits'][callback['centers'].index(72)][0] += 1.0
        result = self.compare(rows)
        self.assertEqual(result['status'], 'PREFIX_LOGITS_EQUAL')
        self.assertIn(72, result['streams'][0]['new_context_centers'])

    def test_missing_supported_center_rejected(self):
        rows = copy.deepcopy(self.records)
        callback = self.callback_at(rows, 69)
        index = callback['centers'].index(69)
        callback['centers'].pop(index)
        callback['logits'].pop(index)
        with self.assertRaisesRegex(ValueError, 'support-contained centers'):
            self.compare(rows)

    def test_duplicate_center_rejected(self):
        rows = copy.deepcopy(self.records)
        callback = self.callback_at(rows, 69)
        callback['centers'][-1] = callback['centers'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate center'):
            self.compare(rows)

    def test_callback_group_change_and_EOF_availability(self):
        result = self.compare(self.records)['streams'][0]
        row = next(c for c in result['comparisons'] if c['center'] == 69)
        self.assertEqual((row['original_callback_phase'], row['modified_callback_phase']), ('finish', 'feed'))
        self.assertEqual((row['original_callback_available_samples'], row['modified_callback_available_samples']), (12075, 14400))
        self.assertTrue(row['fp32_logits_equal'])
        event = result['modified_events'][0]
        self.assertEqual(event['availability_minus_original_file_end_samples'], 2325)
        self.assertEqual(event['availability_minus_original_file_end_seconds'], 0.1453125)
        self.assertIsNone(event['word_end_latency'])

    def test_different_tail_condition_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['rows'][0]['appended_samples'] = 1600
        with self.assertRaisesRegex(ValueError, 'fixed appended-context condition'):
            self.compare(self.records, manifest)

    def test_source_delta_refuses_changed_core(self):
        delta = self.binding['source_deltas']['src/score_endpoint.py']
        original = (ARGS.core_source / 'src/score_endpoint.py').read_bytes()
        with self.assertRaisesRegex(verifier.VerificationError, 'source delta base mismatch'):
            verifier.apply_delta(original + b'\n', delta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--core-source', type=Path, default=ROOT.parent / 'melo5-a20-diagnostic-v1')
    parser.add_argument('--core-data', type=Path)
    ARGS, extra = parser.parse_known_args()
    if ARGS.core_data is None:
        ARGS.core_data = ARGS.data_dir.parent / '2026-10-06-melo5-a20-diagnostic'
    unittest.main(argv=[sys.argv[0], *extra])
