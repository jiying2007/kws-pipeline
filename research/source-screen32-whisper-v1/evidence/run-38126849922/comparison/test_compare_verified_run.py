"""Synthetic fixtures test comparison semantics only, never model execution."""
import copy
import json
import pathlib
import unittest

from compare_verified_run import detailed_compare, normalized

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / 'comparison-input.json').read_text())


def fixtures():
    return [dict(audio_id=row['audio_id'], wav_sha256=row['wav_sha256'], raw_text=row['sensevoice']['raw_text'], status='completed_native_eot', attempts=1, stop_reason='eot', validation_errors=[], raw_output_sha256='a' * 64) for row in DATA['rows']]


class VerifiedComparisonTests(unittest.TestCase):
    def test_zero_coverage_rates_are_null(self):
        for terminal in [None, []]:
            result = detailed_compare(DATA, terminal)
            self.assertEqual(result['observed_valid_triple_denominator'], 0)
            self.assertTrue(all(value is None for value in result['agreement_rates_observed_valid_triples'].values()))
            self.assertEqual(result['counts']['fixed_denominator'], 16)
            self.assertEqual(len(result['categories']['incomplete_three_way']), 16)

    def test_full_counts_and_categories(self):
        result = detailed_compare(DATA, fixtures())
        c = result['counts']
        self.assertEqual(c['literal_three_way_agreement'], 0)
        self.assertEqual(c['normalized_three_way_agreement'], 13)
        self.assertEqual(c['normalized_three_way_agreement_matches_target'], 8)
        self.assertEqual(c['normalized_three_way_agreement_differs_from_target'], 5)
        self.assertEqual(c['normalized_three_way_disagreement'], 3)
        self.assertEqual(result['agreement_rates_observed_valid_triples']['normalized_three_way_agreement'], 13 / 16)
        self.assertEqual(sum(map(len, result['categories'].values())), 16)
        self.assertEqual(result['categories']['three_way_disagreement'], ['clip-000004', 'clip-000007', 'clip-000016'])

    def test_partial_coverage_uses_observed_denominator(self):
        result = detailed_compare(DATA, fixtures()[:1])
        self.assertEqual(result['counts']['all_three_valid_outputs'], 1)
        self.assertEqual(result['counts']['incomplete_three_way'], 15)
        self.assertEqual(result['agreement_rates_observed_valid_triples']['normalized_three_way_agreement'], 1)
        self.assertEqual(result['rates_fixed_16']['normalized_three_way_agreement'], 1 / 16)

    def test_whitespace_normalization_is_explicit_and_no_homophone_merge(self):
        terminal = fixtures()
        terminal[0]['raw_text'] = '小窝 小窝'
        result = detailed_compare(DATA, terminal)
        self.assertEqual(result['counts']['normalized_three_way_agreement'], 12)
        self.assertEqual(result['punctuation_whitespace_comparison']['counts']['normalized_three_way_agreement'], 13)
        self.assertEqual(result['punctuation_whitespace_comparison']['additional_agreement_ids_vs_punctuation_only'], ['clip-000001'])
        terminal[0]['raw_text'] = '小屋 小屋'
        result = detailed_compare(DATA, terminal)
        self.assertFalse(result['rows'][0]['punctuation_whitespace_three_way_agreement'])

    def test_whitespace_rates_null_without_valid_triples(self):
        result = detailed_compare(DATA, [])
        self.assertTrue(all(rate is None for rate in result['punctuation_whitespace_comparison']['agreement_rates_observed_valid_triples'].values()))

    def test_failure_never_votes_or_backfills_target(self):
        terminal = fixtures()
        for row in terminal:
            row['status'] = 'failed_no_retry'
        result = detailed_compare(DATA, terminal)
        self.assertEqual(result['counts']['third_valid_outputs'], 0)
        self.assertIsNone(result['agreement_rates_observed_valid_triples']['normalized_three_way_agreement'])
        self.assertTrue(all('whisper' not in row['valid_literal_votes'] for row in result['rows']))
        self.assertIsNotNone(result['rows'][0]['raw_texts']['whisper'])

    def test_empty_native_eot_coverage_not_lexical_vote(self):
        terminal = fixtures()
        terminal[0]['raw_text'] = ' 。 \n'
        result = detailed_compare(DATA, terminal)
        self.assertEqual(result['counts']['third_native_eot_outputs'], 16)
        self.assertEqual(result['counts']['third_native_eot_nonlexical_outputs'], 1)
        self.assertEqual(result['counts']['third_valid_outputs'], 15)
        self.assertIsNone(result['rows'][0]['normalized_three_way_agreement'])

    def test_validation_error_or_non_eot_not_a_vote(self):
        for key, value in [('validation_errors', ['error']), ('stop_reason', 'length'), ('attempts', 2)]:
            terminal = fixtures()
            terminal[0][key] = value
            result = detailed_compare(DATA, terminal)
            self.assertEqual(result['counts']['third_valid_outputs'], 15)

    def test_invalid_hash_fails_closed(self):
        terminal = fixtures()
        terminal[0]['raw_output_sha256'] = 'z' * 64
        with self.assertRaises(ValueError):
            detailed_compare(DATA, terminal)

    def test_changed_or_unknown_audio_fails(self):
        for key, value in [('audio_id', 'clip-999999'), ('wav_sha256', '0' * 64)]:
            terminal = fixtures()
            terminal[0][key] = value
            with self.assertRaises(ValueError):
                detailed_compare(DATA, terminal)
        with self.assertRaises(ValueError):
            detailed_compare(DATA, fixtures() + fixtures()[:1])

    def test_input_identity_and_gates_cannot_change(self):
        for key, value in [('wav_sha256', '0' * 64), ('role', 'train'), ('human_gold', True), ('training_admitted', True)]:
            data = copy.deepcopy(DATA)
            data['rows'][0][key] = value
            with self.assertRaises(ValueError):
                detailed_compare(data, fixtures())

    def test_targets_do_not_determine_unanimity(self):
        changed = copy.deepcopy(DATA)
        for row in changed['rows']:
            row['target_text'] = 'unrelated target'
        result = detailed_compare(changed, fixtures())
        self.assertEqual(result['counts']['normalized_three_way_agreement'], 13)
        self.assertEqual(result['counts']['normalized_three_way_agreement_matches_target'], 0)
        self.assertEqual(result['counts']['normalized_three_way_agreement_differs_from_target'], 13)

    def test_conservative_normalization_and_oov(self):
        self.assertEqual(normalized('你好，小屋。'), '你好小屋')
        for left, right in [('小窝', '小屋'), ('小吳', '小吴'), ('A B', 'AB'), ('A', 'a'), ('OOV+', 'OOV')]:
            self.assertNotEqual(normalized(left), normalized(right))

    def test_source_flags_and_original_failures_stay_intact(self):
        before = copy.deepcopy(DATA)
        result = detailed_compare(DATA, fixtures())
        self.assertEqual(DATA, before)
        self.assertEqual(sum(row['original_sensevoice_status'] == 'failed_no_retry' for row in result['rows']), 1)
        self.assertEqual(sum(row['original_sensevoice_status'] == 'not_run' for row in result['rows']), 15)
        self.assertTrue(all(row['role'] == 'source_screen_only' and not row['human_gold'] and not row['training_admitted'] for row in result['rows']))


if __name__ == '__main__':
    unittest.main()
