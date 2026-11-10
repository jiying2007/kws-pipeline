import unittest
from research.cnative_landing_v1.metrics import collapse_ctc, contains_keyword, score_rows


def row(seq, gold=True, **kw):
    return dict(source_id=kw.pop('source_id','s1'), greedy_local_ids=seq, gold_present=gold, **kw)


class MetricsTests(unittest.TestCase):
    def test_empty(self): self.assertFalse(contains_keyword([], 'K1'))
    def test_partial_1(self): self.assertFalse(contains_keyword([1], 'K1'))
    def test_partial_3(self): self.assertFalse(contains_keyword([1,2,3], 'K1'))
    def test_k1(self): self.assertTrue(contains_keyword([1,2,3,4], 'K1'))
    def test_k2(self): self.assertTrue(contains_keyword([3,4,3,4], 'K2'))
    def test_k2_half(self): self.assertFalse(contains_keyword([3,4], 'K2'))
    def test_wrong_order(self): self.assertFalse(contains_keyword([4,3,2,1], 'K1'))
    def test_interrupted(self): self.assertFalse(contains_keyword([1,2,4,3,4], 'K1'))
    def test_context(self): self.assertTrue(contains_keyword([3,4,1,2,3,4], 'K1'))
    def test_repeated_wake(self): self.assertTrue(contains_keyword([3,4,3,4,3,4,3,4], 'K2'))
    def test_head_identity(self): self.assertFalse(contains_keyword([1,2,3,4], 'K2'))
    def test_ctc_repeat(self): self.assertEqual(collapse_ctc([3,3,0,3,4,4]), (3,3,4))
    def test_ctc_blank_first_is_wrong(self): self.assertEqual(collapse_ctc([3,0,3]), (3,3))
    def test_ctc_all_blank(self): self.assertEqual(collapse_ctc([0,0,0]), ())
    def test_boolean_id_rejected(self):
        with self.assertRaises(ValueError): contains_keyword([True,2,3,4], 'K1')
    def test_float_id_rejected(self):
        with self.assertRaises(ValueError): contains_keyword([1.0,2,3,4], 'K1')
    def test_blank_collapsed_rejected(self):
        with self.assertRaises(ValueError): contains_keyword([1,0,2,3,4], 'K1')
    def test_invalid_token(self):
        with self.assertRaises(ValueError): contains_keyword([1,2,3,99], 'K1')
    def test_unknown_keyword(self):
        with self.assertRaises(ValueError): contains_keyword([], 'K3')
    def test_nonempty_is_not_wake(self):
        r=score_rows([row([1])],'K1')
        self.assertEqual(r['nonempty_diagnostic_only']['recall'],1)
        self.assertEqual(r['exact_sequence']['recall'],0)
        self.assertEqual(r['complete_keyword_contains']['recall'],0)
    def test_contains_is_not_exact(self):
        r=score_rows([row([3,4,1,2,3,4])],'K1')
        self.assertEqual(r['exact_sequence']['recall'],0)
        self.assertEqual(r['complete_keyword_contains']['recall'],1)
    def test_unknown_mask(self):
        r=score_rows([row([1,2,3,4],None,known=False)],'K1')
        self.assertEqual(r['unknown_rows'],1)
        self.assertIsNone(r['exact_sequence']['recall'])
        self.assertIsNone(r['exact_sequence']['row_fpr'])
    def test_unknown_cannot_be_negative(self):
        with self.assertRaises(ValueError):score_rows([row([],False,known=False)],'K1')
    def test_duplicate(self):
        with self.assertRaises(ValueError):score_rows([row([]),row([])],'K1')
    def test_no_positive_denominator(self):
        self.assertIsNone(score_rows([row([],False)],'K1')['exact_sequence']['recall'])
    def test_no_negative_denominator(self):
        self.assertIsNone(score_rows([row([])],'K1')['exact_sequence']['row_fpr'])
    def test_far_is_never_row_fpr(self):
        self.assertIsNone(score_rows([row([1,2,3,4],False)],'K1')['exact_sequence']['far_per_hour'])
    def test_confusion_counts(self):
        rs=[row([1,2,3,4],True,source_id='a'),row([],True,source_id='b'),
            row([1,2,3,4],False,source_id='c'),row([],False,source_id='d')]
        r=score_rows(rs,'K1')['complete_keyword_contains']
        self.assertEqual([r[k] for k in ('tp','fn','fp','tn')],[1,1,1,1])


if __name__=='__main__': unittest.main()
