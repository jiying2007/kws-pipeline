#!/usr/bin/env python3
"""No-model paired-design and three-route-content-boundary contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import active_tts_pair as pair


def fixture():
    refs = []
    for index in range(3):
        refs.append({'reference_id': 'ref' + str(index), 'source_family_id': 'family' + str(index),
                     'speaker_group_id': 'speaker' + str(index), 'role': 'train-candidate',
                     'wav_sha256': str(index) * 64, 'pcm_sha256': str(index + 1) * 64,
                     'reference_text': '示例内容', 'reference_text_status': 'dataset-provided-content-checked',
                     'reference_text_provenance': {'content_observations': [
                         {'engine': engine, 'actual_text': '示例内容', 'input_mode': 'audio-only',
                          'decoder_completed': True} for engine in ('qwen06', 'sensevoice', 'whisper')]}})
    return {'references': list(reversed(refs)), 'model_revision': 'a' * 40}, {
        'calls': 4, 'seed': 20261105, 'prompt': '你好小窝', 'modes': ['ICL', 'x-vector-only'],
        'reference_ids': ['ref0', 'ref1']}


class PairDesign(unittest.TestCase):
    def test_same_prompt_seed_options_and_distinct_mode_identity(self):
        plan, declaration = fixture()
        cases = pair.cases_for(plan, declaration, lambda text: text)
        self.assertEqual(len(cases), 4)
        self.assertEqual(len({x['case_id'] for x in cases}), 4)
        for left, right in (cases[:2], cases[2:]):
            for key in ('reference_id', 'prompt', 'seed', 'reference_wav_sha256', 'model_revision'):
                self.assertEqual(left[key], right[key])
            self.assertEqual([left['reference_mode'], right['reference_mode']], ['ICL', 'x-vector-only'])

    def test_no_substitution_after_disagreement_or_missing_third_route(self):
        for change in ('disagreement', 'missing'):
            plan, declaration = fixture()
            observations = plan['references'][2]['reference_text_provenance']['content_observations']
            if change == 'disagreement':
                observations[0]['actual_text'] = '不同内容'
            else:
                observations.pop()
            with self.assertRaisesRegex(ValueError, 'PREDECLARED_TWO_THREE_ROUTE_MATCHES_REQUIRED'):
                pair.cases_for(plan, declaration, lambda text: text)

    def test_no_adaptive_seed_or_reference_selection(self):
        plan, declaration = fixture()
        declaration['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'FIXED_PAIR_DESIGN'):
            pair.cases_for(plan, declaration, lambda text: text)
        plan, declaration = fixture()
        declaration['reference_ids'] = ['ref1', 'ref2']
        with self.assertRaisesRegex(ValueError, 'PREDECLARED_TWO_THREE_ROUTE_MATCHES_REQUIRED'):
            pair.cases_for(plan, declaration, lambda text: text)


if __name__ == '__main__':
    unittest.main()
