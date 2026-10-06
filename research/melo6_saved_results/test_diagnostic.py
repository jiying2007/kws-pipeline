import unittest

from diagnostic import classify


class TextBoundaryTests(unittest.TestCase):
    def test_tone_is_not_erased(self):
        self.assertEqual(classify("窝", "沃"), "segmental_agreement_tone_conflict")
        self.assertEqual(classify("窝", "我"), "segmental_agreement_tone_conflict")

    def test_wo_wu_is_not_equated(self):
        self.assertEqual(classify("窝", "屋"), "segmental_mismatch")

    def test_two_asrs_agreeing_on_wrong_word_does_not_match_intent(self):
        for _ in range(2):
            self.assertEqual(classify("今天天气很好", "明天天气很好"), "segmental_mismatch")

    def test_full_repeated_wake_is_not_reduced_to_wo(self):
        self.assertEqual(classify("小窝小窝", "叫我小我"), "segmental_mismatch")

    def test_missing_ambiguous_unaligned_and_incomplete_inputs_stay_unknown(self):
        for text in (None, "", "你好小", "你好小喔", "你好，小窝。"):
            self.assertEqual(classify("你好小窝", text), "unknown")
        self.assertEqual(classify("你好小窝", "你好小窝", status="failed"), "unknown")
        self.assertEqual(classify("你好小窝", "你好小窝", completeness="unknown"), "unknown")

    def test_normalized_lexical_equality_is_only_text_equality(self):
        self.assertEqual(classify("你好小屋", "你好小屋"), "lexical_match")
        self.assertEqual(classify("你好你好", "你好你好"), "lexical_match")


if __name__ == "__main__":
    unittest.main()
