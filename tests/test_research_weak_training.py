#!/usr/bin/env python3
"""Synthetic weak-consumption contracts; no private audio or model downloads."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
import research_weak_recipe as recipe_module


def rows_fixture():
    rows = []
    groups = [([1, 0], 0.5)] * 6 + [([0, 1], 1.0)] * 5 + [([0, 1], 0.5)]
    groups += [([0, 0], 1.0)] * 19 + [([0, 0], 0.5)] * 9
    for index, (target, weight) in enumerate(groups):
        text = "你好小窝" if target[0] else "小窝小窝" if target[1] else "好小窝"
        rows.append({"consumer": "rnn-four-token", "purpose": "technical-research-only",
                     "split": "train", "human_gold": False, "independent_test": False,
                     "full_text": text, "target_ids": [recipe_module.CHAR_IDS[c] for c in text],
                     "ctc_mask": True, "ctc_weight": weight, "keyword_targets": target,
                     "keyword_mask": [True, True], "keyword_weight": [weight, weight],
                     "intended_kind": "positive-intent" if any(target) else "near-miss-intent",
                     "source_id": f"synthetic-source-{index:02d}", "item_id": f"synthetic-item-{index:02d}",
                     "speaker_id": f"synthetic-speaker-{index % 4}", "source_family_id": f"synthetic-family-{index % 4}"})
    return rows


class RecipeContracts(unittest.TestCase):
    def test_balance_cap_and_every_source_exposed(self):
        rows = rows_fixture()
        recipe = recipe_module.make_recipe(rows)
        self.assertEqual([value["items"] for value in recipe["classes"].values()], [6, 6, 28])
        self.assertEqual([value["effective_mass"]["value"] for value in recipe["classes"].values()], [3.0, 3.0, 2 / 3])
        self.assertEqual(recipe["nearphone_fraction_of_effective_ctc_mass"], {"numerator": 1, "denominator": 10, "value": 0.1})
        self.assertAlmostEqual(recipe["fixed_ctc_normalizer"], 1 / 6)
        self.assertAlmostEqual(recipe["fixed_keyword_normalizer"], 1 / 6)
        for epoch in recipe["epoch_row_order"]:
            self.assertEqual(sorted(epoch), list(range(40)))
        self.assertEqual(set(recipe["main_exposure_count_per_source"].values()), {3})
        self.assertEqual(recipe_module.make_recipe(copy.deepcopy(rows)), recipe)
        self.assertEqual(recipe["ordinary_negative_items"], 0)
        self.assertFalse(recipe["nearphone_exposure_replacement_experiment"])

    def test_rejects_oov_blank_or_false_scope(self):
        for field, value in (("full_text", "你好外"), ("target_ids", [0, 1, 2, 3]),
                             ("human_gold", True), ("split", "dev"), ("ctc_mask", False),
                             ("ctc_weight", float("nan"))):
            rows = rows_fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                recipe_module.make_recipe(rows)

    def test_rejects_duplicate_or_invented_negative_pool(self):
        rows = rows_fixture()
        rows[1]["source_id"] = rows[0]["source_id"]
        with self.assertRaises(ValueError):
            recipe_module.make_recipe(rows)

    def test_rejects_keyword_target_detached_from_full_text(self):
        rows = rows_fixture()
        rows[6]["full_text"] = "小窝你好"
        rows[6]["target_ids"] = [recipe_module.CHAR_IDS[c] for c in rows[6]["full_text"]]
        with self.assertRaises(ValueError):
            recipe_module.make_recipe(rows)
        rows = rows_fixture()
        rows[-1]["intended_kind"] = "ordinary-negative-intent"
        with self.assertRaises(ValueError):
            recipe_module.make_recipe(rows)

    def test_unknown_keyword_slot_has_no_recipe_weight(self):
        rows = rows_fixture()
        rows[0]["keyword_mask"][1] = False
        with self.assertRaises(ValueError):
            recipe_module.make_recipe(rows)
        rows[0]["keyword_weight"][1] = 0.0
        recipe = recipe_module.make_recipe(rows)
        self.assertEqual(recipe["items"][0]["effective_keyword_weight"][1], 0.0)


class NumericContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import research_weak_train as trainer
        cls.trainer = trainer
        cls.torch, cls.reference = trainer.torch_runtime()

    def test_ctc_weight_changes_loss_and_gradient(self):
        torch = self.torch
        baseline = torch.linspace(-0.7, 0.8, 35).reshape(7, 1, 5)
        losses, gradients = [], []
        for weight in (1.0, 0.5):
            logits = baseline.clone().requires_grad_()
            loss, raw = self.trainer.weighted_ctc(logits, [1, 2, 3], weight, True, normalizer=0.25)
            loss.backward()
            self.assertTrue(math.isfinite(raw.item()))
            losses.append(loss.detach())
            gradients.append(logits.grad)
        torch.testing.assert_close(losses[1], losses[0] * 0.5)
        torch.testing.assert_close(gradients[1], gradients[0] * 0.5)
        self.assertGreater(gradients[0].abs().sum().item(), 0.0)

    def test_masked_ctc_has_zero_gradient(self):
        torch = self.torch
        logits = torch.zeros(5, 1, 5, requires_grad=True)
        loss, _ = self.trainer.weighted_ctc(logits, [], 0.0, False, normalizer=0.25)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_keyword_masks_and_weights_change_gradients(self):
        torch = self.torch
        logits = torch.tensor([[0.2, -0.3], [0.4, 0.1]], requires_grad=True)
        targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        weights = torch.tensor([[1.0, 1.0], [0.5, 1.0]])
        masks = torch.tensor([[True, False], [True, True]])
        loss, _ = self.trainer.masked_keyword_loss(logits, targets, weights, masks, normalizer=0.5)
        loss.backward()
        expected = (logits.detach().sigmoid() - torch.nan_to_num(targets)) * weights * masks / 2
        torch.testing.assert_close(logits.grad, expected)
        self.assertEqual(logits.grad[0, 1].item(), 0.0)
        self.assertTrue(math.isfinite(loss.item()))

    def test_unpadded_architectures_and_checkpoint_readback(self):
        torch = self.torch
        from training_state import atomic_save_checkpoint, load_checkpoint_snapshot, state_identity
        recipe = recipe_module.make_recipe(rows_fixture())
        for kind in ("rnn64", "c1-ctc"):
            with self.subTest(kind=kind):
                torch.manual_seed(recipe["seed"])
                model, _ = self.trainer.build_main(kind, recipe, self.reference)
                loss = 0.0
                for frames in (7, 9):
                    logits = model(torch.linspace(-0.1, 0.1, frames * 32).reshape(1, frames, 32))
                    self.assertEqual(tuple(logits.shape), (frames, 1, 5))
                    term, _ = self.trainer.weighted_ctc(logits, [1, 2, 3], 0.5, True, normalizer=0.25)
                    loss = loss + term / 2
                loss.backward()
                self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()))
                if kind == "c1-ctc":
                    captured = []
                    handle = model.out_proj.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
                    with torch.no_grad():
                        model(torch.zeros(1, 7, 32))
                    handle.remove()
                    self.assertEqual([tuple(value.shape) for value in captured], [(1, 7, 64)])
                identity = state_identity(model.state_dict())
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "epoch-000.pt"
                    atomic_save_checkpoint({"state_dict": model.state_dict(), "float_state_identity": identity,
                                            "python_rng_state": (3, (1, 2), None), "torch_rng_state": torch.get_rng_state()}, path)
                    saved, sha = load_checkpoint_snapshot(path)
                    self.assertEqual(saved["float_state_identity"], identity)
                    self.assertEqual(sha, self.trainer.sha_file(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true", help="also run pinned CPU Torch gradient/architecture contracts")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecipeContracts)
    if args.numeric:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NumericContracts))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
