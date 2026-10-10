#!/usr/bin/env python3
"""Bookkeeping/AST contracts only: no Torch import, model, audio or numeric run."""
from __future__ import annotations

import ast
import contextlib
import gc
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import types
import unittest
from unittest import mock
import weakref

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import feature_cached_trainer as cache  # noqa: E402


class Storage:
    def __init__(self, size):
        self.size = size

    def nbytes(self):
        return self.size

    def data_ptr(self):
        return id(self)


class Tensor:
    def __init__(self, size=8, *, storage=None, device="cpu", requires_grad=False):
        self.storage = storage if storage is not None else Storage(size)
        self.device = types.SimpleNamespace(type=device)
        self.requires_grad = requires_grad

    def untyped_storage(self):
        return self.storage

    def __bool__(self):
        raise AssertionError("cache must never test tensor truthiness")


class Manifest:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return self.values[index]


def pair(size):
    return (Tensor(size), Tensor(0))


def promotion_cache_check():
    # Execute just the metadata validator AST, never the Torch-dependent module.
    source = ast.parse((ROOT / "tools/verify_model_promotion_bundle.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "require_current_feature_cache")
    scope = {"normalize_feature_cache": cache.normalize_feature_cache,
             "CACHE_POLICY": cache.CACHE_POLICY,
             "feature_cache_max_bytes": cache.feature_cache_max_bytes,
             "feature_cache_max_items": cache.feature_cache_max_items}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<cache metadata>", "exec"), scope)
    return scope[function.name]


class CacheContractTests(unittest.TestCase):
    def test_lru_hits_preserve_exact_feature_target_and_vad_objects(self):
        rows = [(Tensor(8), Tensor(4), Tensor(2)), pair(14), pair(14)]
        dataset = cache.cached_manifest_class(Manifest, 2, 28)(rows)
        self.assertIs(dataset[0], rows[0])
        self.assertIs(dataset[1], rows[1])
        self.assertIs(dataset[0], rows[0])
        self.assertIs(dataset[2], rows[2])
        self.assertEqual(dataset.calls, [0, 1, 2])
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 28)
        dataset[1]  # index 1 was the least recently used item, not index 0.
        self.assertEqual(dataset.calls, [0, 1, 2, 1])
        stats = dataset.feature_cache_stats()
        self.assertEqual((stats["hits"], stats["misses"], stats["evictions"]), (1, 4, 2))

    def test_byte_limit_evicts_before_item_limit(self):
        dataset = cache.cached_manifest_class(Manifest, 100, 16)([pair(8) for _ in range(3)])
        dataset[0], dataset[1], dataset[2], dataset[0]
        stats = dataset.feature_cache_stats()
        self.assertEqual(dataset.calls, [0, 1, 2, 0])
        self.assertEqual(stats["resident_items"], 2)
        self.assertEqual(stats["resident_bytes"], 16)
        self.assertEqual(stats["peak_resident_bytes"], 16)

    def test_one_budget_shared_across_manifests_without_index_aliasing(self):
        cached = cache.cached_manifest_class(Manifest, 2, 12)
        a, b = cached([pair(6), pair(6)]), cached([pair(6)])
        self.assertIs(a[0], a.values[0])
        self.assertIs(b[0], b.values[0])
        self.assertIs(a[0], a.values[0])
        a[1]
        self.assertEqual(a.feature_cache_stats()["resident_bytes"], 12)
        b[0]
        self.assertEqual(b.calls, [0, 0])
        self.assertEqual(a.feature_cache_stats(), b.feature_cache_stats())

    def test_views_and_shared_storage_count_backing_bytes_once(self):
        backing = Storage(100)
        rows = [(Tensor(storage=backing), Tensor(storage=backing)),
                (Tensor(storage=backing), Tensor(4)), pair(10), pair(110)]
        dataset = cache.cached_manifest_class(Manifest, 2, 110)(rows)
        dataset[0], dataset[1]
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 104)
        dataset[2]  # Evict both shared-storage owners to fit this entry.
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 10)
        self.assertEqual(dataset.feature_cache_stats()["evictions"], 2)
        dataset[3]
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 110)
        self.assertEqual(dataset.feature_cache_stats()["resident_items"], 1)

    def test_oversize_skips_admission_without_evicting_good_entry(self):
        rows = [pair(8), pair(17)]
        dataset = cache.cached_manifest_class(Manifest, 3, 16)(rows)
        dataset[0]
        self.assertIs(dataset[1], rows[1])
        dataset[1], dataset[0]
        stats = dataset.feature_cache_stats()
        self.assertEqual((stats["resident_bytes"], stats["resident_items"]), (8, 1))
        self.assertEqual((stats["oversized"], stats["evictions"], stats["hits"]), (2, 0, 1))

    def test_zero_in_either_limit_retains_nothing(self):
        class EphemeralManifest:
            def __getitem__(self, index):
                return pair(8)

        for items, budget in ((0, 16), (2, 0), (0, 0)):
            with self.subTest(items=items, budget=budget):
                dataset = cache.cached_manifest_class(EphemeralManifest, items, budget)()
                value = dataset[0]
                reference = weakref.ref(value[0])
                del value
                gc.collect()
                self.assertIsNone(reference())
                dataset[0]
                stats = dataset.feature_cache_stats()
                self.assertEqual((stats["resident_bytes"], stats["resident_items"]), (0, 0))
                self.assertEqual((stats["hits"], stats["misses"], stats["bypasses"]), (0, 2, 2))

    def test_accelerator_gradient_and_non_tensor_results_bypass(self):
        for value in ((Tensor(device="cuda"),), (Tensor(requires_grad=True),), (), None, (3,)):
            dataset = cache.cached_manifest_class(Manifest, 2, 16)([value])
            self.assertIs(dataset[0], value)
            self.assertIs(dataset[0], value)
            self.assertEqual(dataset.calls, [0, 0])
            self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 0)

    def test_zero_byte_tensors_are_cacheable(self):
        row = (Tensor(0),)
        dataset = cache.cached_manifest_class(Manifest, 2, 1)([row])
        self.assertIs(dataset[0], row)
        self.assertIs(dataset[0], row)
        self.assertEqual(dataset.calls, [0])
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 0)
        self.assertEqual(dataset.feature_cache_stats()["resident_items"], 1)

    def test_stats_are_observations_not_mutable_cache_state(self):
        dataset = cache.cached_manifest_class(Manifest, 1, 16)([pair(8)])
        dataset[0]
        view = dataset.feature_cache_stats()
        view["resident_bytes"] = -100
        self.assertEqual(dataset.feature_cache_stats()["resident_bytes"], 8)

    def test_explicit_validated_budget_and_command_passthrough(self):
        for value in (-1, True, 1.5, "12", cache.MAX_FEATURE_CACHE_BYTES + 1):
            with self.assertRaises(ValueError):
                cache.feature_cache_max_bytes({"feature_cache_max_bytes": value})
        for value in (-1, True, 1.5, "12", cache.MAX_FEATURE_CACHE_ITEMS + 1):
            with self.assertRaises(ValueError):
                cache.feature_cache_max_items({"feature_cache_max_items": value})
        with self.assertRaises(ValueError):
            cache.feature_cache_max_bytes({"feature_cache_max_items": 1})
        command = [sys.executable, str(cache.TRAINING / "train_ctc.py"), "--epochs", "8"]
        for items, budget in ((0, 16), (2, 0)):
            self.assertEqual(cache.rewrite_training_command(command, items, budget), command)
        rewritten = cache.rewrite_training_command(command, 2, 16)
        self.assertEqual(rewritten[2:8], ["--trainer", "rnn", "--feature-cache-max-items", "2",
                                          "--feature-cache-max-bytes", "16"])
        self.assertEqual(rewritten[-3:], ["--", "--epochs", "8"])
        unrelated = [sys.executable, "unrelated.py", "x"]
        self.assertEqual(cache.rewrite_training_command(unrelated, 2, 16), unrelated)

    def test_retired_trainer_fails_before_torch_import(self):
        with self.assertRaisesRegex(ValueError, "retired"):
            cache.rewrite_training_command([sys.executable, str(cache.TRAINING / "train_gru_ctc.py")], 0, 0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cache._run_trainer("gru", 1, 16, [])
        result = subprocess.run([sys.executable, str(cache.__file__), "--trainer", "gru",
                                 "--feature-cache-max-items", "1", "--feature-cache-max-bytes", "16", "--"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_historical_v1_is_readable_but_never_current_promotion_evidence(self):
        historical = {"policy": "deterministic-feature-cache-v1", "max_items": 8192,
                      "training_math_changed": False}
        self.assertEqual(cache.normalize_feature_cache(historical), historical)
        current = cache.cache_contract(8192, 256)
        self.assertEqual(cache.normalize_feature_cache(current), current)
        check = promotion_cache_check()
        policy = {"feature_cache_max_items": 8192, "feature_cache_max_bytes": 256}
        self.assertEqual(check(current, policy), current)
        for invalid in (historical, {**current, "max_bytes": 128},
                        {**current, "max_bytes": 0}, {**current, "max_items": 1}):
            with self.assertRaises(ValueError):
                check(invalid, policy)
        with self.assertRaises(ValueError):
            check(current, {**policy, "feature_cache_max_items": 1})
        for key in current:
            invalid = current.copy()
            del invalid[key]
            with self.assertRaises(ValueError, msg=key):
                cache.normalize_feature_cache(invalid)
        for key, value in (("max_bytes", True), ("max_items", "8192"),
                           ("cache_scope", "per-manifest"), ("byte_accounting", "tensor-numel"),
                           ("training_math_changed", True), ("eviction_policy", "random")):
            with self.assertRaises(ValueError, msg=key):
                cache.normalize_feature_cache({**current, key: value})

    def test_trainer_reports_stats_and_restores_bindings_even_on_failure(self):
        original_manifest = Manifest
        original_environment = lambda: {"training_code_sha256": {"z": "0" * 64}}
        module = types.ModuleType("train_ctc")
        module.Manifest = original_manifest
        module.training_environment = original_environment
        module.sha256_file = lambda path: "1" * 64

        def main():
            metadata = module.training_environment()
            self.assertEqual(metadata["feature_cache"], cache.cache_contract(2, 16))
            self.assertIn("training/feature_cached_trainer.py", metadata["training_code_sha256"])
            dataset = module.Manifest([pair(8)])
            dataset[0], dataset[0]
            raise ValueError("fixture trainer failure")

        module.main = main
        output, original_argv = io.StringIO(), sys.argv
        with mock.patch.dict(sys.modules, {"train_ctc": module}), contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, "fixture trainer failure"):
                cache._run_trainer("rnn", 2, 16, ["--fixture"])
        self.assertIs(module.Manifest, original_manifest)
        self.assertIs(module.training_environment, original_environment)
        self.assertIs(sys.argv, original_argv)
        stats = json.loads(output.getvalue().split("feature_cache_stats=", 1)[1])
        self.assertEqual((stats["hits"], stats["misses"], stats["resident_bytes"]), (1, 1, 8))

    def test_frontend_has_no_hidden_cache_and_unchanged_numerical_ast(self):
        source = (ROOT / "training/frontend.py").read_text()
        tree = ast.parse(source)
        self.assertNotIn("_FEATURE_CACHE", source)
        self.assertNotIn("_cache_key", source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "features")
        # Baseline features() AST after removing only cache lookup/insertion.
        digest = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, "fa29ff12db7dc6f18a3e88990dcccfe96cb02fe76e8ac7975cc031c78b599dab")

    def test_effect_chain_is_a_join_after_independent_offline_reviews(self):
        plan = json.loads((ROOT / "configs/research/effect-chain-iteration-2026-10-10.json").read_text())
        phases = {row["id"]: row for row in plan["sequence"]}
        self.assertEqual(phases["original-failure-mechanism"]["requires"], ["actual-label-and-pcm-lineage"])
        self.assertEqual(phases["bounded-source-screen"]["requires"], ["actual-label-and-pcm-lineage"])
        self.assertEqual(set(phases["paired-two-seed-screen"]["requires"]),
                         {"bounded-source-screen", "original-failure-mechanism"})
        for key in ("execution_enabled", "automatic_phase_advancement", "ci_pass_is_acoustic_or_shipping_approval"):
            self.assertIs(plan[key], False)
        self.assertEqual(phases["original-failure-mechanism"]["missing_state_verdict"], "UNKNOWN")
        self.assertIs(phases["bounded-source-screen"]["training_admitted"], False)


if __name__ == "__main__":
    unittest.main()
