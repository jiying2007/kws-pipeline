#!/usr/bin/env python3
"""Real vocabulary and sampler contracts; optional small synthetic numerics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
import research_d1_inputs as inputs_module
import research_d1_models as models
import research_d1_recipe as recipes
import research_d1_vocab as vocab_module
from research.candidate_encoders.contract import EncoderSpec
from research.candidate_encoders.fulltext_contract import c1_fulltext_spec, rnn64_fulltext_spec

FIXTURES = ROOT / "tests/fixtures/d1-fulltext"


def sampler():
    path = FIXTURES / "research_weak_sampling_v1.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != inputs_module.RETAINED_SAMPLER_SHA256:
        raise ValueError("saved real sampler fixture changed")
    spec = importlib.util.spec_from_file_location("d1_real_sampler_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vocabulary():
    return vocab_module.make_vocabulary((FIXTURES / "fsmn-official-tokens.txt").read_bytes())


def raw_rows():
    prompts = [("你好小窝", "full-positive"), ("今天晚上吃什么", "ordinary-negative"),
               ("小窝小窝", "full-positive"), ("小窝", "partial-negative"),
               ("你好小窝", "full-positive"), ("请把灯打开", "ordinary-negative"),
               ("小窝小窝", "full-positive"), ("你好小屋", "near-negative"),
               ("你好小窝", "full-positive"), ("我想听一首音乐", "ordinary-negative"),
               ("小窝小窝", "full-positive"), ("小窝", "partial-negative"),
               ("你好小窝", "full-positive"), ("现在外面下雨了", "ordinary-negative"),
               ("小窝小窝", "full-positive"), ("小沃小沃", "near-negative")] * 2
    result = []
    for family in range(8):
        for number, (text, category) in enumerate(prompts):
            index = family * 32 + number
            source = "bulk-" + format(index, "024x")
            weight = 0.5 if number % 3 == 0 else 1.0
            result.append({
                "consumer": "d1-verified-full-transcript-v2", "purpose": "technical-research-only",
                "split": "train" if family < 6 else "development", "human_gold": False,
                "independent_test": False, "source_role": "train-candidate" if family < 6 else "development-only",
                "source_plan_sha256": "a" * 64, "global_source_key": ["a" * 64, source],
                "source_id": source, "item_id": "clip-" + format(index % 64, "03d"),
                "source_family_id": "synthetic-family-" + str(family), "speaker_id": "synthetic-speaker-" + str(family),
                "full_text": text, "target_ids": None, "ctc_mask": True, "ctc_weight": weight,
                "keyword_targets": [int(word in text) for word in recipes.KEYWORDS],
                "keyword_mask": [True, True], "keyword_weight": [weight, weight],
                "generation_category": category, "label_status": "machine_candidate", "label_reason": "fixture_agreement",
                "intended_kind": "positive-intent" if category == "full-positive" else "speech-negative-intent",
                "samples": 16000, "audio": "/synthetic/unused.wav", "audio_path": "/synthetic/unused.wav",
                "file_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "pcm_sha256": hashlib.sha256((str(index) + "pcm").encode()).hexdigest(),
            })
    return result


class VocabularyContracts(unittest.TestCase):
    def test_actual_official_bytes_have_missing_chars_and_decoder_symbols(self):
        raw = (FIXTURES / "fsmn-official-tokens.txt").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), vocab_module.BASE_TOKENS_SHA256)
        self.assertIn(b"#0 2600\n", raw)
        self.assertNotIn("灯 ".encode(), raw)
        self.assertNotIn("沃 ".encode(), raw)
        vocab = vocabulary()
        self.assertEqual(vocab["appended"], [{"text": "沃", "output_id": 2599}, {"text": "灯", "output_id": 2600}])
        self.assertEqual(set(vocab["token_to_id"].values()), set(range(2601)))
        self.assertNotIn("#0", vocab["token_to_id"])
        self.assertNotIn("<eps>", vocab["token_to_id"])
        for line in raw.decode().splitlines():
            token, number = line.rsplit(maxsplit=1)
            if 1 <= int(number) <= 2599:
                self.assertEqual(vocab["token_to_id"][token], int(number) - 1)

    def test_full_text_retained_and_future_oov_is_explicit(self):
        vocab = vocabulary()
        for text in vocab_module.TASK_TEXTS:
            projection = vocab_module.fulltext_targets(text, vocab)
            self.assertTrue(projection["supported"])
            self.assertEqual(len(projection["target_ids"]), len(text))
            self.assertNotIn(0, projection["target_ids"])
        self.assertIn(2600, vocab_module.fulltext_targets("请把灯打开", vocab)["target_ids"])
        self.assertEqual(vocab_module.fulltext_targets("小沃小沃", vocab)["target_ids"][1::2], [2599, 2599])
        unsupported = vocab_module.fulltext_targets("请把𠮷灯打开", vocab)
        self.assertFalse(unsupported["supported"])
        self.assertIsNone(unsupported["target_ids"])
        self.assertEqual(unsupported["oov_characters"], ["𠮷"])
        altered = copy.deepcopy(vocab)
        altered["token_to_id"]["灯"] = 0
        with self.assertRaises(ValueError):
            vocab_module.validate_vocabulary(altered, (FIXTURES / "fsmn-official-tokens.txt").read_bytes())

    def test_v1_stays_five_and_v2_binds_new_identity(self):
        with self.assertRaises(ValueError):
            EncoderSpec("rnn64_control", 32, 64, 2601)
        self.assertEqual(EncoderSpec("rnn64_control", 32, 64, 5).to_dict()["schema"], "candidate-encoder-source-spec-v1")
        for spec, count in ((rnn64_fulltext_spec(vocab_module.digest(vocabulary())), 175273),
                            (c1_fulltext_spec(vocab_module.digest(vocabulary())), 184425)):
            self.assertEqual(spec.vocab_size, 2601)
            self.assertEqual(spec.parameter_elements, count)
            self.assertEqual(spec.to_dict()["schema"], "candidate-encoder-fulltext-spec-v2")
            self.assertIn("ctc2601-v2", spec.to_dict()["candidate_id"])


class RecipeContracts(unittest.TestCase):
    def test_all_declared_full_texts_reach_common_ctc_and_dev_stays_out(self):
        rows = recipes.project_rows(raw_rows(), vocabulary())
        self.assertEqual(len(rows), 256)
        self.assertTrue(all(row["ctc_mask"] for row in rows))
        cohort = recipes.cohort_rows(rows)
        self.assertEqual(len(cohort), 192)
        self.assertEqual(len({row["source_id"] for row in cohort}), 192)
        self.assertLess(len({row["item_id"] for row in cohort}), len(cohort))
        self.assertEqual({row["split"] for row in cohort}, {"train"})
        self.assertEqual(len(recipes.cohort_rows(rows, objective="keyword")), 192)

    def test_exact_draws_fixed_positive_sequence_and_real_weight_mass(self):
        rows = recipes.project_rows(raw_rows(), vocabulary())
        plans = recipes.make_comparison_recipes(rows, vocabulary(), sampler())
        sequences = []
        for value in (0, 5, 10):
            recipe = plans[str(value)]
            self.assertEqual(recipe["exposure_plan"]["counts"], {
                "full-positive": 300, "ordinary-or-partial": 300 - 6 * value, "near-negative": 6 * value})
            self.assertEqual(recipe["positive_exposures_by_word"], {"你好小窝": 150, "小窝小窝": 150})
            self.assertEqual(set(recipe["positive_exposures_by_family_and_word"].values()), {25})
            self.assertEqual(recipe["fixed_ctc_normalizer"], 1.0)
            self.assertEqual(recipe["optimizer_steps"], 150)
            self.assertEqual(sum(recipe["exposures_by_family"].values()), 600)
            self.assertEqual(len(recipe["exposures_by_family"]), 6)
            sequences.append([draw for draw in recipe["exposure_plan"]["draws"] if draw["bucket"] == "full-positive"])
            self.assertFalse(recipe["draw_fraction_equals_effective_weight_fraction"])
            self.assertFalse(recipe["actual_training_executed"])
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[1], sequences[2])

    def test_zero_arm_never_requires_near_pool_or_every_row_exposure(self):
        rows = recipes.project_rows([row for row in raw_rows() if row["generation_category"] != "near-negative"], vocabulary())
        result, _ = recipes.make_recipe(rows, vocabulary(), sampler(), near_percent=0)
        self.assertNotIn("near-negative", result["exposures_by_generation_category"])
        with self.assertRaisesRegex(ValueError, "MISSING_SAMPLING_POOL:near-negative"):
            recipes.make_recipe(rows, vocabulary(), sampler(), near_percent=5)
        no_ordinary = [row for row in rows if row["generation_category"] == "full-positive"]
        with self.assertRaisesRegex(ValueError, "MISSING_SAMPLING_POOL:ordinary-or-partial"):
            recipes.make_recipe(no_ordinary, vocabulary(), sampler(), near_percent=0)

    def test_oov_and_uncertain_keep_different_objective_masks(self):
        rows = raw_rows()
        row = rows[1]
        row["full_text"] = "请把𠮷灯打开"
        projected = recipes.project_rows(rows, vocabulary())
        self.assertFalse(projected[1]["ctc_mask"])
        self.assertIsNone(projected[1]["target_ids"])
        self.assertEqual(projected[1]["ctc_weight"], 0)
        self.assertTrue(projected[1]["keyword_eligible"])
        self.assertEqual(projected[1]["keyword_targets"], [0, 0])
        self.assertEqual(projected[1]["full_text"], row["full_text"])
        row.update(label_status="uncertain", ctc_mask=False, ctc_weight=0,
                   keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0])
        unknown = recipes.project_rows(rows, vocabulary())[1]
        self.assertFalse(unknown["ctc_mask"] or unknown["keyword_eligible"])
        self.assertEqual(unknown["keyword_targets"], [None, None])
        row["keyword_targets"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "D1_UNKNOWN_KEYWORD_IS_NOT_NEGATIVE"):
            recipes.project_rows(rows, vocabulary())

    def test_global_cell_identity_not_local_clip_id_and_geometry_not_padded(self):
        rows = raw_rows()
        rows[1]["global_source_key"] = rows[0]["global_source_key"]
        with self.assertRaisesRegex(ValueError, "D1_DISTINCT_GLOBAL_ROWS"):
            recipes.project_rows(rows, vocabulary())
        rows = raw_rows()
        rows[0]["samples"] = 400
        projected = recipes.project_rows(rows, vocabulary())
        self.assertFalse(projected[0]["ctc_mask"])
        self.assertEqual(projected[0]["ctc_reason"], "common_unpadded_ctc_geometry_infeasible")
        self.assertEqual(projected[0]["expected_output_frames"], {"rnn64": 1, "c1": 1, "fsmn": 0})


class AggregateContracts(unittest.TestCase):
    """Exercise cross-package orchestration with an explicit replay test double.

    This does not claim to test native/ASR authenticity: that is the retained
    data loader's responsibility. The double intentionally yields *no* legacy
    CTC rows, so a consumer that forgets to exhaust it is observably wrong.
    """
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = raw_rows()
        self.plan = {"cells": [{"cell_id": row["source_id"]} for row in self.rows]}
        self.plan_sha = vocab_module.digest(self.plan)
        self.replayed, self.role_checks = [], []

        def replay(package, consumer):
            self.replayed.append(str(package))
            if getattr(self, "mutate_during_replay", False):
                (package / "report.json").write_text("{}")
            if False:
                yield None

        def add_role(roles, source, clip, measured):
            self.role_checks.append(source["source_id"])
            for kind, value in (("family", source["source_family_id"]),
                                ("parent", source["parent_source_sha256"])):
                if value is not None:
                    key = kind, value
                    if key in roles and roles[key] != source["role"]:
                        raise ValueError("SOURCE_IDENTITY_SPLIT_LEAK:" + kind)
                    roles[key] = source["role"]

        self.loader = SimpleNamespace(read_bytes=lambda path, *args: Path(path).read_bytes(), loads=json.loads,
            canonical=vocab_module.canonical, load_training_rows=replay, SourceRoles=dict,
            add_source_role=add_role, relative=Path, training_row=lambda raw, consumer: dict(raw["training"]))

    def package(self, name, index, *, source_override=None):
        row = copy.deepcopy(self.rows[index])
        row.update(source_plan_sha256=self.plan_sha, global_source_key=[self.plan_sha, row["source_id"]], item_id="clip-000")
        source = {**row, "source_kind": "synthetic-reference-clone", "role": row["source_role"],
                  "parent_source_sha256": "parent-" + row["source_family_id"],
                  "reference_wav_sha256": "reference-wav-" + row["source_family_id"],
                  "reference_pcm_sha256": "reference-pcm-" + row["source_family_id"]}
        source.update(source_override or {})
        clip = {"id": "clip-000", "original_pcm": {"sha256": "original-pcm-" + str(index)}}
        raw = {"item_id": "clip-000", "source": source, "input": clip,
               "audio_derivation": {"pcm_sha256": row["pcm_sha256"]}, "audio": "audio/clip-000.wav",
               "decision": {"status": row["label_status"], "reason": row["label_reason"]},
               "asr": [{"fixture": True}], "training": row}
        report = {"source_adapter": {"schema": "kws-active-d1-frozen-plan-source-v1", "plan_sha256": self.plan_sha},
                  "items": [raw], "test_package": name}
        files = {"report.json": report, "evidence/inputs.json": {"clips": [clip]},
                 "evidence/source/d1/plan.json": self.plan}
        root = self.root / name
        manifest = {}
        for filename, value in files.items():
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(vocab_module.canonical(value))
            manifest[filename] = hashlib.sha256(target.read_bytes()).hexdigest()
        (root / "MANIFEST.json").write_bytes(vocab_module.canonical(manifest))
        return {"path": str(root), "manifest_sha256": hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()}

    def test_exhausts_replay_even_when_legacy_projection_empty_and_local_ids_repeat(self):
        packages = [self.package("first", 5), self.package("second", 47)]
        rows, receipt = inputs_module.aggregate_batches(packages, self.loader, expected_plan_sha256=self.plan_sha)
        self.assertEqual(len(self.replayed), 2)
        self.assertEqual(len(self.role_checks), 2)
        self.assertEqual([row["item_id"] for row in rows], ["clip-000", "clip-000"])
        self.assertEqual(len({tuple(row["global_source_key"]) for row in rows}), 2)
        self.assertEqual(receipt["observed_items"], 2)
        self.assertFalse(receipt["complete_plan_coverage"])
        self.assertTrue(all(row["asr_transcripts"] == [{"fixture": True}] for row in rows))

    def test_duplicate_global_cell_and_cross_package_ancestry_leak_fail(self):
        packages = [self.package("first", 5), self.package("duplicate", 5)]
        with self.assertRaisesRegex(ValueError, "D1_DUPLICATE_GLOBAL_CELL"):
            inputs_module.aggregate_batches(packages, self.loader, expected_plan_sha256=self.plan_sha)
        packages = [packages[0], self.package("development", 224, source_override={
            "parent_source_sha256": "parent-synthetic-family-0"})]
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:parent"):
            inputs_module.aggregate_batches(packages, self.loader, expected_plan_sha256=self.plan_sha)

    def test_changed_report_during_complete_replay_is_rejected(self):
        package = self.package("changing", 5)
        self.mutate_during_replay = True
        with self.assertRaisesRegex(ValueError, "D1_PACKAGE_CHANGED_DURING_REPLAY"):
            inputs_module.aggregate_batches([package], self.loader, expected_plan_sha256=self.plan_sha)


class NumericContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from research_weak_train import torch_runtime
        cls.torch, cls.reference = torch_runtime()

    def test_fresh_models_use_same_2601_full_text_head(self):
        torch = self.torch
        for kind in ("rnn64-ctc2601", "c1-ctc2601"):
            torch.manual_seed(recipes.SEED)
            model, spec = models.build_fresh(kind, vocabulary(), self.reference)
            feature = torch.linspace(-0.1, 0.2, 8 * 32).reshape(1, 8, 32)
            logits = model(feature)
            self.assertEqual(tuple(logits.shape), (8, 1, 2601))
            target = vocab_module.fulltext_targets("小沃小沃", vocabulary())["target_ids"]
            loss, _ = models.fulltext_ctc(logits, target, 0.5, True)
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            self.assertEqual(spec["vocabulary_sha256"], vocab_module.digest(vocabulary()))

    def test_extension_copies_all_old_rows_and_tensors_bitwise(self):
        torch = self.torch
        old = {models.FSMN_WEIGHT: torch.arange(2599 * 140, dtype=torch.float32).reshape(2599, 140) / 1024,
               models.FSMN_BIAS: torch.arange(2599, dtype=torch.float32) / 1024,
               "backbone.fixture": torch.tensor([-0.0, 1.0, -2.0])}
        saved = {name: value.clone() for name, value in old.items()}
        state, record = models.extend_fsmn_state(old)
        again, _ = models.extend_fsmn_state(old)
        from research_weak_train import tensor_sha
        for name, value in old.items():
            self.assertEqual(tensor_sha(value), tensor_sha(saved[name]))
            restored = state[name][:2599] if name in (models.FSMN_WEIGHT, models.FSMN_BIAS) else state[name]
            self.assertEqual(tensor_sha(restored), tensor_sha(saved[name]))
            self.assertEqual(tensor_sha(state[name]), tensor_sha(again[name]))
        self.assertEqual(tuple(state[models.FSMN_WEIGHT].shape), (2601, 140))
        self.assertFalse(record["original_2599_checkpoint_identity"])
        self.assertFalse(record["old_softmax_probabilities_preserved"])
        self.assertGreater(state[models.FSMN_WEIGHT][2599:].abs().sum().item(), 0)

    def test_actual_confidence_scaling_and_unknown_zero_gradient(self):
        torch = self.torch
        target = vocab_module.fulltext_targets("请把灯打开", vocabulary())["target_ids"]
        gradients, losses = [], []
        for weight in (1.0, 0.5):
            logits = torch.zeros(8, 1, 2601, requires_grad=True)
            loss, _ = models.fulltext_ctc(logits, target, weight, True)
            loss.backward()
            gradients.append(logits.grad)
            losses.append(loss.detach())
        torch.testing.assert_close(gradients[1], gradients[0] / 2)
        torch.testing.assert_close(losses[1], losses[0] / 2)
        self.assertNotEqual(gradients[0][:, :, 2600].sum().item(), 0)
        logits = torch.zeros(8, 1, 2601, requires_grad=True)
        loss, _ = models.fulltext_ctc(logits, None, 0.0, False)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    args = parser.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (VocabularyContracts, RecipeContracts, AggregateContracts))
    if args.numeric:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NumericContracts))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
