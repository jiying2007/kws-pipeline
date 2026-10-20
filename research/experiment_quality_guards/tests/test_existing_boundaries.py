"""Invented text/metadata mocks only. No ASR, audio, private labels or downloads."""
import sys, pathlib, unittest, copy
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'research/experiment_quality_guards')]
import speech_asr_consensus as consensus
import asr_native_contracts as native
import quality_gates as research

class ExistingBoundaries(unittest.TestCase):

    def test_punctuation_compare_does_not_rewrite_raw(self):
        raw = {'text': '你好，小窝。'}
        before = copy.deepcopy(raw)
        self.assertEqual(consensus.normalized_transcript(raw['text']), '你好小窝')
        self.assertEqual(raw, before)

    def test_confusables_remain_distinct(self):
        values = ['小窝', '小屋', '小哦', '小欧']
        self.assertEqual(len({consensus.normalized_transcript(x) for x in values}), 4)

    def test_two_engines_cannot_be_three_family_policy(self):
        with self.assertRaisesRegex(ValueError, 'exactly three'):
            consensus.validate_recognizer_policy({'schema_version': 1, 'evidence_class': consensus.POLICY_CLASS, 'normalization': consensus.NORMALIZATION, 'frozen_before_recognition': True, 'target_conditioning_allowed': False, 'recognizers': [{}, {}]})

    def row(self, a, b):
        return {'intended_text': '你好小窝', 'actual_text': None, 'review': {}, 'asr_results': [{'status': 'complete', 'raw_text': a}, {'status': 'complete', 'raw_text': b}]}

    def test_agreement_remains_weak_no_admission(self):
        r = research.label_preparation(self.row('你好，小窝。', '你好小窝'))
        self.assertEqual(r['machine_state'], 'machinesAgreeWeak')
        self.assertFalse(r['training_admitted'])
        self.assertFalse(r['original_complete_label_eligible'])
        self.assertEqual(r['acoustic_completeness'], 'UNKNOWN')

    def test_disagreement_is_preserved(self):
        r = research.label_preparation(self.row('你好小窝', '你好小屋'))
        self.assertEqual(r['machine_state'], 'disagree')
        self.assertFalse(r['automatic_relabel'])
        self.assertIsNone(r['ctc_target'])

    def test_agree_other_string_does_not_support_plan(self):
        r = research.label_preparation(self.row('你好小屋', '你好，小屋。'))
        self.assertEqual(r['planned_lexical_support'], 'REJECTED')

    def test_missing_result_unknown_not_blank(self):
        r = research.label_preparation(self.row('', None))
        self.assertEqual(r['machine_state'], 'unresolved')
        self.assertFalse(r['silence_or_blank_target_inferred'])

    def test_contract_versions_cannot_be_relabelled(self):
        self.assertEqual(native.QWEN_REQUEST['max_new_tokens'], 256)
        self.assertNotEqual(native.QWEN_REQUEST['max_new_tokens'], 512)
        self.assertEqual(native.EXECUTION['dtype'], 'float32')

    def test_product_keywords_remain_wo_not_wu(self):
        self.assertEqual(research.KEYWORDS, {'K1': '你好小窝', 'K2': '小窝小窝'})
if __name__ == '__main__':
    unittest.main()
