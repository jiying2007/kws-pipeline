from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import textwrap
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import torch
import train_ctc
import export_model
import adversarial_refinement
import iterate_domain
from objective_config import (
    AUXILIARY_LOSS_WEIGHT_NAMES, auxiliary_loss_weights,
    optional_objective_cli_args, verify_auxiliary_loss_readback,
)
from product_development_experiment import verify_spec
from reproducibility_smoke import fixture
from training_state import state_identity
from verify_training_readback import verify_candidate


class ObjectiveContractTests(unittest.TestCase):
    def test_absent_controls_preserve_default_argv(self):
        self.assertEqual(optional_objective_cli_args({}), [])

    def test_zero_and_fractional_weights_are_forwarded(self):
        for weight in (0.0, 0.125):
            for name in AUXILIARY_LOSS_WEIGHT_NAMES:
                with self.subTest(name=name, weight=weight):
                    self.assertEqual(optional_objective_cli_args({name: weight}),
                                     ["--" + name.replace("_", "-"), str(weight)])

    def test_invalid_weights_fail_closed(self):
        for name in AUXILIARY_LOSS_WEIGHT_NAMES:
            for value in (True, False, None, "0", -0.1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    auxiliary_loss_weights({name: value})

    def test_marker_accepts_all_four_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "spec.json"
            spec = {
                "schema_version": 1, "experiment_id": "ctc-only-contract-v1",
                "development_only": True, "source_policy": "exact-pr-head",
                "protected_evidence_used": False, "reason": "unit fixture",
                "config_overrides": {"train." + name: 0.0 for name in AUXILIARY_LOSS_WEIGHT_NAMES},
            }
            path.write_text(json.dumps(spec), encoding="utf-8")
            self.assertEqual(verify_spec(path)["config_overrides"], spec["config_overrides"])
            for bad in (True, "0", -1.0, 1.1):
                spec["config_overrides"]["train.ordered_token_loss_weight"] = bad
                path.write_text(json.dumps(spec), encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    verify_spec(path)

    def test_readback_rejects_missing_or_wrong_weights(self):
        zero = dict.fromkeys(AUXILIARY_LOSS_WEIGHT_NAMES, 0.0)
        self.assertEqual(verify_auxiliary_loss_readback(zero, zero), zero)
        with self.assertRaises(ValueError):
            verify_auxiliary_loss_readback(zero, {})
        with self.assertRaises(ValueError):
            verify_auxiliary_loss_readback(zero, {**zero, "ordered_token_loss_weight": 0.35})

    def test_base_and_refinement_use_shared_forwarding(self):
        zero = dict.fromkeys(AUXILIARY_LOSS_WEIGHT_NAMES, 0.0)
        expected = optional_objective_cli_args(zero)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            commands = []
            with mock.patch.object(iterate_domain, "run", side_effect=commands.append):
                iterate_domain.build_torch(
                    cfg={"train": zero}, frontend="logmel", tokens=root/"tokens",
                    keywords=root/"keywords", manifest=root/"manifest", output=root,
                    previous=None, hard_negative_manifest=None, wake_balance=None,
                    warm_start_strategy="full", round_index=0,
                )
            for i in range(0, len(expected), 2):
                self.assertEqual(commands[0][commands[0].index(expected[i])+1], expected[i+1])
            commands.clear()
            balance = {"positive_example_weight": 2.0, "default_wake_example_weight": 1.0,
                       "wake_keyword_weights": {}}
            with mock.patch.object(adversarial_refinement, "run", side_effect=commands.append), \
                 mock.patch.object(adversarial_refinement, "derive_refinement_wake_balance", return_value=balance):
                adversarial_refinement._train_refinement(
                    cfg={"train": zero}, frontend="logmel", tokens=root/"tokens",
                    keywords=root/"keywords", dataset_manifest=root/"data",
                    static_manifest=root/"static", adversarial_manifest=root/"adversarial",
                    failure_manifest=None, focus_rows_by_manifest={}, warm_start=root/"warm.pt",
                    output=root, epochs=1, lr_scale=0.5, refinement_round=2,
                )
            for i in range(0, len(expected), 2):
                self.assertEqual(commands[0][commands[0].index(expected[i])+1], expected[i+1])


class StateIdentityTests(unittest.TestCase):
    def test_order_does_not_change_identity(self):
        a, b = torch.tensor([1.0]), torch.tensor([2.0])
        self.assertEqual(state_identity({"a": a, "b": b}), state_identity({"b": b, "a": a}))

    def test_same_int8_is_not_same_float_state(self):
        a = torch.tensor([1.0, 0.5000, 0.1])
        b = torch.tensor([1.0, 0.5001, 0.1])
        qa, sa, _ = export_model.q8(a, "a")
        qb, sb, _ = export_model.q8(b, "b")
        self.assertEqual((qa, sa), (qb, sb))
        self.assertNotEqual(state_identity({"w": a})["sha256"], state_identity({"w": b})["sha256"])

    def test_name_shape_dtype_are_bound(self):
        t = torch.arange(4, dtype=torch.float32)
        hashes = {state_identity(state)["sha256"] for state in (
            {"a": t}, {"b": t}, {"a": t.reshape(2, 2)}, {"a": t.double()})}
        self.assertEqual(len(hashes), 4)

    def test_noncontiguous_and_scalar_supported(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        self.assertEqual(state_identity({"w": t}), state_identity({"w": t.contiguous()}))
        self.assertEqual(state_identity({"s": torch.tensor(1.0)})["tensors"][0]["bytes"], 4)

    def test_empty_or_nonfinite_state_rejected(self):
        for value in ({}, {"w": torch.tensor([float("nan")])}, {1: torch.ones(1)}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state_identity(value)


class RealTrainerReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="kws-aux-readback-")
        cls.root = pathlib.Path(cls.tmp.name)
        cls.tokens, cls.keywords, manifest = fixture(cls.root)
        # No exact wake rows: disabled ordered loss must not demand an exact-wake mean.
        rows = manifest.read_text(encoding="utf-8").splitlines()
        manifest.write_text("\n".join(rows[2:6]) + "\n", encoding="utf-8")
        cls.checkpoint = cls.root / "model.pt"
        cls.model = cls.root / "model.kwm"
        cls.zero = dict.fromkeys(AUXILIARY_LOSS_WEIGHT_NAMES, 0.0)
        argv = ["train_ctc.py", "--manifest", str(manifest), "--tokens", str(cls.tokens),
                "--keywords", str(cls.keywords), "--output", str(cls.checkpoint),
                "--feature-dim", "8", "--hidden-dim", "8", "--epochs", "1",
                "--batch-size", "4", "--seed", "7", "--ordered-token-scope", "exact-configured-wake-targets-v1",
                *optional_objective_cli_args(cls.zero)]
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for name in ("ordered_token_loss", "keyword_sequence_margin_loss",
                         "strict_prefix_completion_loss", "recurrent_release_loss"):
                stack.enter_context(mock.patch.object(train_ctc, name, side_effect=AssertionError("disabled loss evaluated")))
            train_ctc.main()
        with mock.patch.object(sys, "argv", ["export_model.py", "--checkpoint", str(cls.checkpoint),
                                           "--tokens", str(cls.tokens), "--output", str(cls.model)]), \
             contextlib.redirect_stdout(io.StringIO()):
            export_model.main()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ctc_only_really_bypasses_all_auxiliaries(self):
        payload = torch.load(self.checkpoint, weights_only=True, map_location="cpu")
        epoch = payload["epoch_history"][0]
        for key in ("ordered", "margin", "completion", "release"):
            self.assertEqual(epoch[key], 0.0)
        self.assertEqual(epoch["loss"], epoch["ctc"])
        self.assertGreater(epoch["ctc"], 0.0)
        self.assertEqual(payload["float_state_identity"], state_identity(payload["state_dict"]))

    def test_config_checkpoint_export_roundtrip(self):
        receipt = verify_candidate(self.zero, self.checkpoint)
        self.assertEqual(receipt["auxiliary_loss_weights"], self.zero)
        self.assertEqual(receipt["resume_authority"], "weights-only-not-optimizer-continuous")

    def test_dropped_override_is_detected(self):
        with self.assertRaisesRegex(ValueError, "readback mismatch"):
            verify_candidate({"ordered_token_loss_weight": 0.35}, self.checkpoint)

    def test_runtime_metadata_is_retained(self):
        payload = torch.load(self.checkpoint, weights_only=True, map_location="cpu")
        provenance = json.loads(pathlib.Path(str(self.model) + ".provenance.json").read_text())
        for key in ("cpu_runtime", "torch_runtime"):
            self.assertEqual(payload["training_environment"][key], provenance["training"]["environment"][key])

    def test_export_rejects_stale_float_hash(self):
        payload = torch.load(self.checkpoint, weights_only=True, map_location="cpu")
        first = next(iter(payload["state_dict"]))
        payload["state_dict"][first].reshape(-1)[0] += 0.001
        bad = self.root / "tampered.pt"
        torch.save(payload, bad)
        with mock.patch.object(sys, "argv", ["export_model.py", "--checkpoint", str(bad),
                                           "--tokens", str(self.tokens), "--output", str(self.root/"bad.kwm")]), \
             self.assertRaisesRegex(ValueError, "float state identity mismatch"):
            export_model.main()
        self.assertFalse((self.root / "bad.kwm").exists())


class WorkflowTests(unittest.TestCase):
    def test_repro_gate_keeps_same_class_strict_and_cross_vendor_diagnostic(self):
        source = (ROOT/".github/workflows/product-training-data-contract.yml").read_text()
        block = source.split("      - name: Enforce numerical-class reproducibility policy", 1)[1]
        script = textwrap.dedent(block.split("python3 - <<'PY'\n", 1)[1].rsplit("          PY", 1)[0])
        row = {"pythonhashseed": "0", "torch_num_threads": 1, "torch_num_interop_threads": 1,
               "torch_version": "fixture", "fixture_manifest_sha256": "f"*64,
               "training_code_sha256": {"fixture": "c"*64}, "model_sha256": "a"*64,
               "float_state_sha256": "b"*64}
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            root = work/".repro"
            root.mkdir()
            (root/"training-reproducibility-a.json").write_text(json.dumps(row))
            b = root/"training-reproducibility-b.json"
            b.write_text(json.dumps({**row, "float_state_sha256": "d"*64}))
            (work/"build").mkdir()
            report = {"infrastructure_complete": True, "release_authority": False,
                      "cross_vendor_pair_observed": False, "exact_match": True,
                      "first_observed_divergence": None, "cpu_vendors": ["AMD", "AMD"]}
            (work/"build/numeric-divergence.json").write_text(json.dumps(report))
            result = subprocess.run([sys.executable, "-c", script], cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("training float state mismatch", result.stderr)

            b.write_text(json.dumps(row))
            result = subprocess.run([sys.executable, "-c", script], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

            # Cross-vendor divergence is retained as diagnostic evidence rather
            # than runner-shopping until a random same-vendor pair appears.
            b.write_text(json.dumps({**row, "float_state_sha256": "d"*64, "model_sha256": "e"*64}))
            report.update(cross_vendor_pair_observed=True, exact_match=False,
                          first_observed_divergence={"phase": "optimizer-step", "step": 3},
                          cpu_vendors=["AMD", "Intel"])
            (work/"build/numeric-divergence.json").write_text(json.dumps(report))
            result = subprocess.run([sys.executable, "-c", script], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cross-vendor numerical boundary retained", result.stdout)

    def test_base_upload_precedes_expensive_work(self):
        source = (ROOT/".github/workflows/product-development-experiment.yml").read_text()
        early = source.index("Retain completed base before expensive refinement")
        self.assertLess(source.index("Verify base objective and float-state readback"), early)
        self.assertLess(early, source.index("Diagnose base acoustic/runtime alignment"))
        self.assertLess(early, source.index("Run bounded refinement experiment"))
        self.assertIn("github.run_attempt", source)
        self.assertIn("candidates/*/model.pt", source)
        self.assertIn("candidates/*/model.kwm.provenance.json", source)
        self.assertEqual(source.count("include-hidden-files: true"), 2)


if __name__ == "__main__":
    unittest.main()
