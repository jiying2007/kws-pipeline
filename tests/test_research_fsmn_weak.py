#!/usr/bin/env python3
"""Offline FSMN full-text weight contracts; no official weight download in CI."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'training')]
import research_fsmn_weak as trainer


def fixture():
    vocab = {'<blk>': 0, '你': 300, '好': 500, '小': 700, '窝': 900, '我': 1100}
    groups = [([1, 0], 0.5)] * 6 + [([0, 1], 1.0)] * 5 + [([0, 1], 0.5)]
    groups += [([0, 0], 1.0)] * 21 + [([0, 0], 0.5)] * 15
    rows = []
    for index, (target, weight) in enumerate(groups):
        text = '你好小窝' if target[0] else '小窝小窝' if target[1] else ('我好小窝' if index >= 40 else '好小窝')
        rows.append({'consumer': 'fsmn-fulltext', 'purpose': 'technical-research-only', 'split': 'train',
                     'human_gold': False, 'independent_test': False, 'full_text': text, 'target_ids': [vocab[c] for c in text],
                     'ctc_mask': True, 'ctc_weight': weight, 'keyword_targets': target, 'keyword_mask': [True, True],
                     'intended_kind': 'positive-intent' if any(target) else 'near-miss-intent',
                     'item_id': f'fixture-{index}', 'source_id': f'fixture-source-{index}',
                     'source_family_id': f'fixture-family-{index % 4}', 'speaker_id': f'fixture-speaker-{index % 4}'})
    return rows, vocab


class RecipeContracts(unittest.TestCase):
    def test_official_ids_and_all48_fixed_mass(self):
        rows, vocab = fixture()
        recipe = trainer.make_recipe(rows, vocab)
        self.assertEqual([value['effective_mass']['value'] for value in recipe['classes'].values()], [3, 3, 2 / 3])
        self.assertEqual(recipe['classes']['nearphone-negative']['multiplier']['numerator'], 4)
        self.assertEqual(recipe['classes']['nearphone-negative']['multiplier']['denominator'], 171)
        self.assertAlmostEqual(recipe['fixed_ctc_normalizer'], 5 / 36)
        self.assertEqual(rows[-1]['target_ids'], [1100, 500, 700, 900])
        for order in recipe['stage_row_orders'].values():
            self.assertEqual(sorted(order), list(range(48)))
        self.assertEqual(trainer.make_recipe(copy.deepcopy(rows), vocab), recipe)
        self.assertFalse(recipe['historical_b_or_a20_weights_used'])
        self.assertEqual(recipe['frontend']['audio_padding_samples'], 0)

    def test_oov_blank_and_target_aliases_rejected(self):
        for field, value in (('target_ids', [0]), ('full_text', '外'), ('keyword_targets', [0, 1]),
                             ('ctc_mask', False), ('human_gold', True)):
            rows, vocab = fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer.make_recipe(rows, vocab)

    def test_official_raw_id_minus_one_and_alias_blank(self):
        raw = 'sil 1\n<blk> 1\n<eps> 0\n' + ''.join(f'token{number} {number}\n' for number in range(2, 2600))
        vocab = trainer.vocabulary(raw.encode())
        self.assertEqual(vocab['<blk>'], 0)
        self.assertEqual(vocab['token2599'], 2598)
        self.assertNotIn('<eps>', vocab)
        with self.assertRaises(ValueError):
            trainer.vocabulary(raw.replace('token2599 2599\n', '').encode())

    def test_source_lock_is_exact_official_pair(self):
        lock = trainer.source_lock()
        files = {value['path']: value for value in lock['files']}
        self.assertEqual(lock['model'], 'thuduj12/kws_wenwen_fsmn_ctc')
        self.assertEqual(files['avg_30.pt']['sha256'], 'd3d89b5dde08c2926d46a97eab6f7179a87b1946ccebd16c02e77042ea8389fe')
        self.assertEqual(files['tokens.txt']['sha256'], '5914b5e048e14dd8e1cce6914333cbbf8b382df7679c26041e35bd1c86db43aa')
        self.assertEqual(len(files), 12)


class NumericContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, _ = trainer.torch_runtime()

    def test_full_vocabulary_targets_and_weighted_gradient(self):
        torch = self.torch
        gradients, losses = [], []
        for weight in (1.0, 0.5):
            logits = torch.zeros(7, 1, 2599, requires_grad=True)
            loss, _ = trainer.fulltext_ctc(logits, [300, 1100, 900], weight, normalizer=5 / 36)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            gradients.append(logits.grad)
            losses.append(loss.detach())
        torch.testing.assert_close(gradients[1], gradients[0] / 2)
        torch.testing.assert_close(losses[1], losses[0] / 2)
        self.assertNotEqual(gradients[0][:, :, 1100].sum().item(), 0)
        with self.assertRaises(ValueError):
            trainer.fulltext_ctc(torch.zeros(7, 1, 2599), [0], 1, normalizer=5 / 36)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numeric', action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecipeContracts)
    if args.numeric:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NumericContracts))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
