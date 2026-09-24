#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from development_signal import POLICY, metric_signal, record_rank, record_signal, selection_enabled, selection_report
from iterate_domain import calibration_behavior_key, calibrate, keyword_rows, objective, select_strict_candidate
from product_development_experiment import materialize
from adversarial_refinement import select_refinement_source

IDS = ("1", "2")
GATES = {"max_frr": 0.0, "max_far_per_hour": 0.0, "max_p95_latency_ms": 800.0, "max_far_frr": 0.0}


def metrics(matches=(2, 2), far=600.0):
    per_keyword = {str(i + 1): {"expected": 4, "matched": n, "false_rejects": 4 - n, "frr": (4 - n) / 4} for i, n in enumerate(matches)}
    frr = (8 - sum(matches)) / 8
    base = {"expected": 8, "matched": sum(matches), "false_rejects": 8 - sum(matches),
            "frr": frr, "far_per_hour": far, "p95_post_end_latency_ms": 0.0,
            "per_keyword": per_keyword}
    domains = {"domains": {"distance:far": {"frr": frr}}, "worst_domain_score": 1000 * frr + far}
    return base, domains


def record(matches=(2, 2), far=600.0, round_index=0):
    base, domains = metrics(matches, far)
    strict = matches == (4, 4) and far == 0
    return {"round": round_index, "frontend": "logmel", "model_sha256": str(round_index) * 64,
            "calibration": base, "test": copy.deepcopy(base), "calibration_domains": domains,
            "test_domains": copy.deepcopy(domains), "score": 2 * objective(base, domains, GATES),
            "calibration_gate": strict, "test_gate": strict}


class SignalTests(unittest.TestCase):
    def test_guard_is_explicit_and_boolean(self):
        self.assertFalse(selection_enabled({}))
        self.assertTrue(selection_enabled({"domain_iteration": {"nondegenerate_selection_enabled": True}}))
        for bad in ("true", 1, None, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selection_enabled({"domain_iteration": {"nondegenerate_selection_enabled": bad}})

    def test_one_match_is_only_observed_signal_not_release_authority(self):
        row = record((1, 1))
        report = selection_report([row], row, IDS)
        self.assertEqual(report["nondegenerate_candidate_count"], 1)
        self.assertFalse(report["release_authority"])
        self.assertFalse(row["calibration_gate"])

    def test_each_keyword_and_each_split_is_required(self):
        row = record()
        row["test"], _ = metrics((0, 2))
        signal = record_signal(row, IDS)
        self.assertFalse(signal["nondegenerate"])
        self.assertEqual(signal["collapsed_keyword_splits"], ["test:1"])

    def test_missing_keyword_is_not_inferred_away(self):
        base, _ = metrics()
        del base["per_keyword"]["2"]
        with self.assertRaisesRegex(ValueError, "missing keyword 2"):
            metric_signal(base, IDS)

    def test_missing_and_invalid_counts_fail_closed(self):
        for name in ("expected", "matched", "false_rejects"):
            for bad in (None, True, 1.5, "2", -1):
                base, _ = metrics()
                base["per_keyword"]["1"][name] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    metric_signal(base, IDS)

    def test_inconsistent_or_zero_support_fails_closed(self):
        for values in ({"expected": 0, "matched": 0, "false_rejects": 0},
                       {"expected": 4, "matched": 3, "false_rejects": 3}):
            base, _ = metrics()
            base["per_keyword"]["1"].update(values)
            with self.assertRaises(ValueError):
                metric_signal(base, IDS)

    def test_required_ids_must_be_explicit_and_unique(self):
        base, _ = metrics()
        for ids in ((), ("1", "1"), ("",), (1,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                metric_signal(base, ids)

    def test_all_reject_scalar_trap_does_not_win_guarded_rank(self):
        useful, mute = record(), record((0, 0), 0.0, 1)
        self.assertLess(mute["score"], useful["score"])
        self.assertLess(record_rank(useful, IDS), record_rank(mute, IDS))
        self.assertLess(record_rank(mute), record_rank(useful))  # unopted canonical behavior retained

    def test_one_keyword_collapse_is_not_hidden_by_aggregate_recall(self):
        balanced, partial = record((1, 1)), record((4, 0), 0.0, 1)
        self.assertLess(partial["calibration"]["frr"], balanced["calibration"]["frr"])
        self.assertLess(record_rank(balanced, IDS), record_rank(partial, IDS))

    def test_nonfinite_score_rejected(self):
        for bad in (float("nan"), float("inf"), True, "1"):
            row = record(); row["score"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                record_rank(row, IDS)

    def test_empty_viable_set_has_no_candidate_but_retains_diagnostic(self):
        mute = record((0, 0), 0.0)
        report = selection_report([mute], mute, IDS)
        self.assertIsNone(report["selected_candidate"])
        self.assertEqual(report["nondegenerate_candidate_count"], 0)
        self.assertEqual(report["diagnostic_source"]["model_sha256"], mute["model_sha256"])
        self.assertEqual(report["best_artifact_role"], "diagnostic-only-no-nondegenerate-candidate")

    def test_guard_report_rejects_wrong_selection(self):
        useful, mute = record(), record((0, 0), 0.0, 1)
        with self.assertRaisesRegex(ValueError, "displaced"):
            selection_report([useful, mute], mute, IDS)

    def test_actual_strict_selection_still_requires_both_gates(self):
        useful, strict = record(), record((4, 4), 0.0, 1)
        self.assertIs(select_strict_candidate([useful, strict], IDS), strict)
        strict["test_gate"] = False
        self.assertIsNone(select_strict_candidate([useful, strict], IDS))

    def test_permissive_gate_cannot_turn_silence_into_guarded_candidate(self):
        mute = record((0, 0), 0.0)
        mute["calibration_gate"] = mute["test_gate"] = True
        self.assertIsNone(select_strict_candidate([mute], IDS))

    def test_calibration_key_keeps_strict_plateau_and_blocks_mute(self):
        good, gd = metrics((4, 4), 0.0)
        useful, ud = metrics()
        mute, md = metrics((0, 0), 0.0)
        self.assertEqual(calibration_behavior_key(good, gd, GATES, IDS), (0.0,) * 9)
        self.assertLess(calibration_behavior_key(useful, ud, GATES, IDS),
                        calibration_behavior_key(mute, md, GATES, IDS))


class IntegrationTests(unittest.TestCase):
    def test_refinement_prefers_viable_source_despite_worse_far_slice(self):
        useful, partial = record((1, 1)), record((4, 0), 0.0, 1)
        for row in (useful, partial):
            row["checkpoint"] = f"round-{row['round']}.pt"
        useful["calibration_domains"]["domains"]["distance:far"]["frr"] = 1.0
        manifest = {"development_qualified": False, "records": [useful, partial],
                    "candidate_selection": {"objective_fallback_used": True, "qualification_used_for_selection": False,
                        "nondegeneracy": {"policy": POLICY, "required_keyword_ids": list(IDS)}}}
        selected, _ = select_refinement_source(manifest)
        self.assertIs(selected, useful)

    def test_refinement_can_still_inspect_collapsed_source_when_no_viable_one(self):
        mute = record((0, 0), 0.0)
        mute["checkpoint"] = "diagnostic.pt"
        manifest = {"development_qualified": False, "records": [mute],
                    "candidate_selection": {"objective_fallback_used": True, "qualification_used_for_selection": False,
                        "nondegeneracy": {"policy": POLICY, "required_keyword_ids": list(IDS)}}}
        selected, _ = select_refinement_source(manifest)
        self.assertIs(selected, mute)
        self.assertIsNone(selection_report([mute], mute, IDS)["selected_candidate"])

    def test_refinement_rejects_unknown_guard_policy(self):
        row = record(); row["checkpoint"] = "fixture.pt"
        manifest = {"development_qualified": False, "records": [row],
                    "candidate_selection": {"objective_fallback_used": True, "qualification_used_for_selection": False,
                        "nondegeneracy": {"policy": "unknown", "required_keyword_ids": list(IDS)}}}
        with self.assertRaisesRegex(ValueError, "unsupported refinement source"):
            select_refinement_source(manifest)

    def test_real_calibrator_uses_and_records_guarded_order(self):
        # Execute the real coordinate calibrator; only expensive corpus execution is a fixture.
        import iterate_domain
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            keywords = root / "keywords.tsv"
            keywords.write_text("1\twake-one\t0.55\ta b\n2\twake-two\t0.55\tb a\n")
            packs = {}
            def compile_fixture(tokens, tsv, pack):
                packs[pack] = keyword_rows(tsv)
            def evaluate_fixture(**kwargs):
                matches = tuple(2 if row["threshold"] < 0.8 else 0 for row in packs[kwargs["pack"]])
                return metrics(matches, 300.0 * sum(n > 0 for n in matches))
            with mock.patch.object(iterate_domain, "compile_pack", side_effect=compile_fixture), \
                 mock.patch.object(iterate_domain, "evaluate", side_effect=evaluate_fixture):
                for enabled, expected in ((False, 0.85), (True, 0.55)):
                    output = root / str(enabled)
                    _, _, base, _ = calibrate(runner=root/"runner", model=root/"model", tokens=root/"tokens",
                        source_keywords=keywords, references=root/"refs", output=output,
                        thresholds=[0.55, 0.85], rounds=2, gates=GATES, require_keyword_signal=enabled)
                    self.assertEqual(base["calibrated_thresholds"], {"1": expected, "2": expected})
                    curve = json.loads((output/"calibration-operating-curve.json").read_text())
                    if enabled:
                        self.assertEqual(curve["nondegeneracy_policy"], POLICY)
                        self.assertTrue(base["calibration_nondegeneracy"]["nondegenerate"])
                    else:
                        self.assertNotIn("nondegeneracy_policy", curve)

    def test_product_materializer_activates_guard_without_changing_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            spec = {"schema_version": 1, "experiment_id": "signal-guard-contract-v1",
                "development_only": True, "source_policy": "exact-pr-head", "protected_evidence_used": False,
                "reason": "contract fixture", "config_overrides": {"train.ordered_token_loss_weight": 0.0}}
            cfg = {"product_candidate_data": {"policy": "external-speech-like-product-base-v1",
                "tone_fallback_allowed": False, "protected_evidence_used": False},
                "domain_iteration": {"adversarial_lexicon": {"enabled": True}}, "domain_gates": GATES}
            sp, cp, op = root/"spec.json", root/"config.json", root/"output.json"
            sp.write_text(json.dumps(spec)); cp.write_text(json.dumps(cfg))
            materialize(spec_path=sp, effective_config_path=cp, output_path=op,
                receipt_path=root/"receipt.json", base_sha="a"*40, head_sha="b"*40)
            result = json.loads(op.read_text())
            self.assertTrue(selection_enabled(result))
            self.assertEqual(result["domain_gates"], GATES)
            self.assertEqual(json.loads(cp.read_text()), cfg)


if __name__ == "__main__":
    unittest.main()
