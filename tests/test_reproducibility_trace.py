#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from compare_reproducibility_trace import compare, load_trace, observer_parity
from reproducibility_trace import Recorder


class NumericTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="kws-numeric-trace-")
        cls.root = pathlib.Path(cls.tmp.name)
        for name, observed in (("plain", False), ("observed-a", True), ("observed-b", True)):
            cmd = [sys.executable, str(ROOT / "training/reproducibility_smoke.py"),
                   "--work-dir", str(cls.root / name), "--output", str(cls.root / (name + ".json"))]
            if observed:
                cmd.append("--numeric-trace")
            with (cls.root / (name + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": "0"},
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        cls.identity, cls.trace, _ = load_trace(cls.root / "observed-a.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def mutated(self, change):
        sandbox = tempfile.TemporaryDirectory(prefix="kws-trace-mutation-")
        self.addCleanup(sandbox.cleanup)
        root = pathlib.Path(sandbox.name)
        shutil.copytree(self.root / "observed-a", root / "case")
        identity = copy.deepcopy(self.identity)
        trace = copy.deepcopy(self.trace)
        change(identity, trace, root / "case/numeric-trace")
        path = root / "case/numeric-trace/trace.json"
        path.write_text(json.dumps(trace))
        identity["numeric_trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "case.json").write_text(json.dumps(identity))
        return root / "case.json"

    def test_observer_does_not_change_real_training(self):
        observer_parity(self.root / "plain", self.root / "observed-a")

    def test_independent_processes_match_every_recorded_stage(self):
        report = compare(self.root / "observed-a.json", self.root / "observed-b.json")
        self.assertTrue(report["exact_match"])
        self.assertIsNone(report["first_observed_divergence"])
        self.assertFalse(report["historical_failure_resolved"])
        self.assertFalse(report["release_authority"])

    def test_all_actual_steps_and_gradient_states_are_retained(self):
        self.assertEqual(self.trace["steps"], 9)
        for phase in ("initial-state", "features", "first-recurrent-step", "log-probabilities",
                      "raw-ctc", "gradient-before-clip", "gradient-after-clip", "optimizer-step"):
            self.assertTrue(any(e["phase"] == phase for e in self.trace["events"]), phase)
        event = next(e for e in self.trace["events"] if e["phase"] == "optimizer-step")
        self.assertTrue(any("exp_avg" in t["name"] for t in event["tensors"]))
        self.assertLess(self.trace["tensor_bytes"], 2 * 1024 * 1024)

    def test_valid_tensor_perturbation_reports_first_observed_stage(self):
        def perturb(identity, trace, root):
            event = next(e for e in trace["events"] if e["phase"] == "forward-logits")
            row = event["tensors"][0]
            path = root / row["file"]
            data = bytearray(path.read_bytes())
            old = struct.unpack_from("<f", data)[0]
            struct.pack_into("<f", data, 0, old + 0.125)
            path.write_bytes(data)
            row["sha256"] = hashlib.sha256(data).hexdigest()
        other = self.mutated(perturb)
        report = compare(self.root / "observed-a.json", other)
        self.assertFalse(report["exact_match"])
        first = report["first_observed_divergence"]
        self.assertEqual((first["phase"], first["step"]), ("forward-logits", 1))
        self.assertEqual(first["different_elements"], 1)
        self.assertGreater(first["max_abs_delta"], 0)

    def test_tensor_tampering_fails_before_numeric_interpretation(self):
        def tamper(identity, trace, root):
            row = next(e for e in trace["events"] if e["tensors"])["tensors"][0]
            path = root / row["file"]
            data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_trace(self.mutated(tamper))

    def test_missing_tensor_is_not_an_equal_hash(self):
        def remove(identity, trace, root):
            row = next(e for e in trace["events"] if e["tensors"])["tensors"][0]
            (root / row["file"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_trace(self.mutated(remove))

    def test_tensor_paths_are_confined(self):
        def change(identity, trace, root):
            next(e for e in trace["events"] if e["tensors"])["tensors"][0]["file"] = "../outside.bin"
        with self.assertRaisesRegex(ValueError, "confined"):
            load_trace(self.mutated(change))

    def test_partial_and_string_verdicts_cannot_pass(self):
        for bad in (False, "true", 1, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "incomplete"):
                load_trace(self.mutated(lambda i, t, r: t.update(completed=bad)))

    def test_missing_phase_and_bad_order_are_rejected(self):
        def remove(identity, trace, root):
            next(e for e in trace["events"] if e["phase"] == "raw-ctc")["phase"] = "not-ctc"
        with self.assertRaisesRegex(ValueError, "missing ordered phase"):
            load_trace(self.mutated(remove))
        with self.assertRaisesRegex(ValueError, "event order"):
            load_trace(self.mutated(lambda i, t, r: t["events"][2].update(ordinal=400)))

    def test_final_state_is_bound_to_training_identity(self):
        with self.assertRaisesRegex(ValueError, "final-state identity"):
            load_trace(self.mutated(lambda i, t, r: i.update(float_state_sha256="0" * 64)))

    def test_mismatched_training_inputs_do_not_become_numeric_failure(self):
        other = self.mutated(lambda i, t, r: i.update(fixture_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            compare(self.root / "observed-a.json", other)

    def test_same_vendor_pass_does_not_erase_historical_failure(self):
        other = self.mutated(lambda i, t, r: i.update(cpu_runtime=self.identity["cpu_runtime"]))
        result = compare(self.root / "observed-a.json", other)
        self.assertTrue(result["exact_match"])
        self.assertFalse(result["cross_vendor_pair_observed"])
        self.assertFalse(result["historical_failure_resolved"])

    def test_cross_vendor_pair_is_machine_classified_without_release_authority(self):
        # Both identities are controlled fixtures. The hosted runner may itself
        # be Intel or AMD, so changing only the right side to Intel cannot prove
        # that a cross-vendor pair was supplied.
        amd = copy.deepcopy(self.identity["cpu_runtime"])
        amd["model"] = "AMD EPYC TEST CPU"
        intel = copy.deepcopy(self.identity["cpu_runtime"])
        intel["model"] = "INTEL(R) XEON(R) TEST CPU"
        left = self.mutated(lambda i, t, r: i.update(cpu_runtime=amd))
        right = self.mutated(lambda i, t, r: i.update(cpu_runtime=intel))
        result = compare(left, right)
        self.assertTrue(result["cross_vendor_pair_observed"])
        self.assertEqual(sorted(result["cpu_vendors"]), ["AMD", "Intel"])
        self.assertFalse(result["release_authority"])

    def test_failing_comparison_still_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "report.json"
            cmd = [sys.executable, str(ROOT / "training/compare_reproducibility_trace.py"),
                   "--left", str(pathlib.Path(tmp) / "missing.json"), "--right", str(self.root / "observed-a.json"),
                   "--output", str(output)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(output.read_text())["infrastructure_complete"])

    def test_recorder_refuses_overwriting_existing_evidence(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            Recorder(self.root / "observed-a/numeric-trace")

    def test_workflow_retains_failure_and_preserves_float_kwm_checks(self):
        workflow = (ROOT / ".github/workflows/product-training-data-contract.yml").read_text()
        for expected in ("--numeric-trace", "--observer-parity", "training-numeric-divergence",
                         "continue-on-error: true", "cross_vendor_pair_observed",
                         "non-cross-vendor training reproducibility mismatch",
                         "same-class training reproducibility: PASS"):
            self.assertIn(expected, workflow)
        self.assertNotIn("cross-runner float training state mismatch", workflow)
        self.assertIn("name: Retain reproducibility identity\n        if: always()", workflow)


if __name__ == "__main__":
    unittest.main()
