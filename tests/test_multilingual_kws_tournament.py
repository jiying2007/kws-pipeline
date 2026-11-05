"""No-model confidence/Pareto candidate selector contracts for frozen development."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import select_multilingual_kws_models as rank  # noqa: E402
from compare_dataset_iterations import APPROX_FAR_POLICY as FAR_POLICY, DATASET_IDENTITY_POLICY  # noqa: E402


def scorecard(model: str, *, failures: int, false_accepts: int,
              latency: float = 210, dataset: str = "d", hours: float = 10.0) -> dict:
    n = 100
    if not false_accepts:
        far_bound = 2.9957 / hours
    else:
        far_bound = (false_accepts + 1.959964 * false_accepts ** 0.5) / hours
    return {
        "schema_version": 4, "run_id": "dev-" + model[:8],
        "identity": {
            "protocol": "dataset-iteration-v4",
            "far_policy": FAR_POLICY,
            "dataset_identity_policy": DATASET_IDENTITY_POLICY,
            "dataset_id": dataset * 64, "model_id": model * 64,
            "code_sha": "a" * 40, "seed": "dev-1", "variable": "model"},
        "metrics": {
            "expected": n, "matched": n - failures, "false_rejects": failures,
            "frr": failures / n, "false_accepts": float(false_accepts),
            "far_per_hour": false_accepts / hours,
            "p95_post_end_latency_ms": latency},
        "far_rate": {
            "observed_false_accepts": false_accepts,
            "point_estimate_per_hour": false_accepts / hours,
            "upper_bound_95_per_hour": far_bound},
        "exposure": {"exposure_hours": hours, "negative_recordings": 4},
        "artifacts": {"references_sha256": "e" * 64, "detections_sha256": "f" * 64},
        "domains": {},
    }


def sample_entry(root: Path, name: str, card: dict,
                 *, cpu: float = 100, rss: float = 1500,
                 p99: float = 3.0) -> dict:
    path = root / (name + ".json")
    path.write_text(json.dumps(card, sort_keys=True, allow_nan=False), encoding="utf-8")
    receipt = root / (name + ".resource.json")
    receipt.write_text(json.dumps({
        "schema_version": 1, "policy": "kws-host-development-resource-v1",
        "model_id": card["identity"]["model_id"],
        "code_sha": card["identity"]["code_sha"],
        "measurement_class": "host-development-only",
        "simulated": False,
        "metric_values": {"cpu_ms_per_s": cpu, "rss_kib": rss,
                          "p99_block_ms": p99},
        "training_admitted": False, "qualification_allowed": False,
    }, sort_keys=True), encoding="utf-8")
    return {
        "id": name, "scorecard_path": str(path.resolve()),
        "scorecard_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resource": {
            "measurement_class": "host-development-only",
            "measurement_receipt_path": str(receipt.resolve()),
            "measurement_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "cpu_ms_per_s": cpu, "rss_kib": rss, "p99_block_ms": p99,
        }
    }


def language_binding(root: Path, card: dict, *, locale: str = "zh-CN") -> dict:
    identity = card["identity"]
    payload = {
        "schema_version": 1,
        "policy": "kws-development-language-cohort-binding-v1",
        "locale": locale, "dataset_id": identity["dataset_id"],
        "references_sha256": card["artifacts"]["references_sha256"],
        "code_sha": identity["code_sha"], "protocol": identity["protocol"],
        "seed": identity["seed"], "source_role": "development-only",
        "independent_language_verified": False,
        "sealed_or_formal_data_used": False, "training_admitted": False,
        "qualification_allowed": False,
    }
    path = root / ("cohort-" + locale + ".json")
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class DevelopmentTournamentTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.baseline = sample_entry(self.root, "control",
                                     scorecard("1", failures=10, false_accepts=2),
                                     cpu=130, rss=1800, p99=5)
        self.fast = sample_entry(self.root, "better-recall",
                                 scorecard("2", failures=5, false_accepts=0),
                                 cpu=90, rss=1400, p99=3)
        self.low_far = sample_entry(self.root, "lower-far",
                                    scorecard("3", failures=8, false_accepts=1),
                                    cpu=80, rss=1300, p99=2)
        self.plan = {
            "schema_version": 1, "policy": rank.POLICY,
            "locale": "zh-CN", "role": "development-only",
            "language_binding": language_binding(
                self.root, scorecard("1", failures=10, false_accepts=2)),
            "sealed_or_formal_data_used": False, "model_calls_authorized": False,
            "baseline": self.baseline, "candidates": [self.fast, self.low_far],
            "budgets": {
                "min_expected_wakes": 100, "min_negative_hours": 8.0,
                "max_frr_upper_95": 0.20,
                "max_far_upper_95_per_hour": 0.80,
                "max_latency_ms": 500.0,
                "max_cpu_ms_per_s": 200.0, "max_rss_kib": 3000.0,
                "max_p99_block_ms": 15.0, "frr_point_tolerance": 0.0}
        }

    def test_two_objective_frontier_and_safe_research_leader(self) -> None:
        result = rank.select(self.plan)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["research_leader_id"], "better-recall")
        self.assertEqual(set(result["pareto_frontier_ids"]),
                         {"better-recall", "lower-far"})
        self.assertFalse(result["production_model_selected"])
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["production_promotion_allowed"])
        self.assertFalse(result["shipping_approved"])
        self.assertEqual(result["new_model_or_training_calls_authorized"], 0)
        self.assertFalse(result["source_role_authentication_verified"])
        self.assertEqual(result, rank.select(self.plan))

    def test_language_is_research_label_not_an_english_shipping_license(self) -> None:
        p = copy.deepcopy(self.plan)
        p["locale"] = "en-US"
        p["language_binding"] = language_binding(
            self.root, scorecard("1", failures=10, false_accepts=2),
            locale="en-US")
        r = rank.select(p)
        self.assertEqual(r["locale"], "en-US")
        self.assertTrue(r["research_leader_only"])
        self.assertFalse(r["independent_real_human_verified"])
        self.assertFalse(r["physical_ssc305_verified"])

    def test_one_variable_same_dev_corpus_required(self) -> None:
        bad = copy.deepcopy(self.plan)
        card = scorecard("2", failures=5, false_accepts=0, dataset="c")
        bad["candidates"][0] = sample_entry(self.root, "moved-dataset", card)
        with self.assertRaisesRegex(ValueError, "both dataset and model changed"):
            rank.select(bad)
        bad = copy.deepcopy(self.plan)
        card = scorecard("2", failures=5, false_accepts=0)
        card["identity"]["code_sha"] = "b" * 40
        bad["candidates"][0] = sample_entry(self.root, "moved-code", card)
        with self.assertRaisesRegex(ValueError, "code_sha differs"):
            rank.select(bad)
        bad = copy.deepcopy(self.plan)
        card = scorecard("2", failures=5, false_accepts=0)
        card["artifacts"]["references_sha256"] = "9" * 64
        bad["candidates"][0] = sample_entry(self.root, "moved-refs", card)
        with self.assertRaisesRegex(ValueError, "NONCOMPARABLE_DEVELOPMENT_COHORT"):
            rank.select(bad)

    def test_single_heldout_or_disguised_formal_role_rejected(self) -> None:
        for key in ("role", "sealed_or_formal_data_used", "model_calls_authorized"):
            p = copy.deepcopy(self.plan)
            p[key] = "sealed-evaluation" if key == "role" else True
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "DEVELOPMENT_ONLY_PRE_REGISTRATION_REQUIRED"):
                rank.select(p)

    def test_false_reject_point_must_match_actual_counts(self) -> None:
        altered = scorecard("2", failures=5, false_accepts=0)
        altered["metrics"]["frr"] = 0
        p = copy.deepcopy(self.plan)
        p["candidates"][0] = sample_entry(self.root, "wrong-frr", altered)
        with self.assertRaisesRegex(ValueError, "FRR_DISAGREES_WITH_EVENT_COUNTS"):
            rank.select(p)
        altered = scorecard("2", failures=5, false_accepts=0)
        altered["metrics"]["expected"] = True
        p["candidates"][0] = sample_entry(self.root, "bool-expected", altered)
        with self.assertRaisesRegex(ValueError, "INVALID_POSITIVE_EVENT_COUNTS"):
            rank.select(p)

    def test_upper_far_bound_cannot_be_forged(self) -> None:
        altered = scorecard("2", failures=5, false_accepts=0)
        altered["far_rate"]["upper_bound_95_per_hour"] = 0
        p = copy.deepcopy(self.plan)
        p["candidates"][0] = sample_entry(self.root, "forged-upper-bound", altered)
        with self.assertRaisesRegex(ValueError, "inconsistent negative FAR bound"):
            rank.select(p)

    def test_resource_and_evidence_hash_bound(self) -> None:
        p = copy.deepcopy(self.plan)
        p["candidates"][0]["resource"]["cpu_ms_per_s"] = True
        with self.assertRaisesRegex(ValueError, "INVALID_NUMERIC_TYPE"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["candidates"][0]["scorecard_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SCORECARD_BYTES_CHANGED"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["candidates"][0]["resource"]["measurement_class"] = "ssc305-certified"
        with self.assertRaisesRegex(ValueError, "RESOURCE_EVIDENCE_UNQUALIFIED"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["candidates"][0]["resource"]["cpu_ms_per_s"] = 2
        with self.assertRaisesRegex(ValueError, "RESOURCE_RECEIPT_VALUES_DRIFT"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["candidates"][0]["resource"]["measurement_receipt_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "RESOURCE_RECEIPT_BYTES_CHANGED"):
            rank.select(p)

    def test_regressed_resource_or_frr_candidates_are_not_winners(self) -> None:
        p = copy.deepcopy(self.plan)
        p["candidates"][0] = sample_entry(
            self.root, "rss-over", scorecard("2", failures=5, false_accepts=0),
            rss=10000)
        r = rank.select(p)
        self.assertNotIn("rss-over", r["pareto_frontier_ids"])
        bad = scorecard("4", failures=15, false_accepts=1)
        p["candidates"][0] = sample_entry(self.root, "bad-recall", bad)
        r = rank.select(p)
        entry = next(x for x in r["candidate_results"] if x["id"] == "bad-recall")
        self.assertFalse(entry["eligible_for_development_rank"])
        self.assertIn("positive-recall-regressed", entry["blockers"])

    def test_no_qualifying_candidate_returns_none_without_promotion(self) -> None:
        p = copy.deepcopy(self.plan)
        p["budgets"]["max_far_upper_95_per_hour"] = .001
        r = rank.select(p)
        self.assertIsNone(r["research_leader_id"])
        self.assertEqual(r["pareto_frontier_ids"], [])
        self.assertFalse(r["production_model_selected"])

    def test_duplicate_candidate_and_model_identity_rejected(self) -> None:
        p = copy.deepcopy(self.plan)
        p["candidates"][1]["id"] = p["candidates"][0]["id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CANDIDATE"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["candidates"][1] = sample_entry(
            self.root, "duplicate-model", scorecard("2", failures=8, false_accepts=1))
        with self.assertRaisesRegex(ValueError, "REPEATED_CANDIDATE_MODEL"):
            rank.select(p)

    def test_exact_zero_one_event_monotonic_preserves_legacy(self) -> None:
        entries = [sample_entry(self.root, "count-" + str(k),
                   scorecard(str(k + 2), failures=5, false_accepts=k)) for k in (0, 1)]
        points = [rank.load_entry(e, expected_role="development-only")[1] for e in entries]
        self.assertGreater(points[0]["far_upper_95_per_hour"], points[1]["far_upper_95_per_hour"])
        self.assertLess(points[0]["selection_far_upper_95_per_hour"], points[1]["selection_far_upper_95_per_hour"])
        self.assertAlmostEqual(points[0]["selection_far_upper_95_per_hour"], .2995732273553991)
        self.assertAlmostEqual(points[1]["selection_far_upper_95_per_hour"], .4743864518390577)

    def test_two_events_eight_hours_approx_green_exact_rejects(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "base-eight", scorecard("1", failures=10, false_accepts=2, hours=8))
        p["candidates"] = [sample_entry(self.root, "cand-eight", scorecard("2", failures=5, false_accepts=2, hours=8))]
        p["budgets"]["max_far_upper_95_per_hour"] = .65
        result = rank.select(p)
        row = result["candidate_results"][0]
        self.assertEqual(row["paired_verdict"], "improved")
        self.assertLess(row["development_scores"]["far_upper_95_per_hour"], .65)
        self.assertGreater(row["development_scores"]["selection_far_upper_95_per_hour"], .78)
        self.assertIsNone(result["research_leader_id"])
        self.assertIn("far-confidence-over-development-budget", row["blockers"])
        self.assertFalse(result["control_eligible_for_development_rank"])

    def test_thirty_hours_zero_events_success_without_keyword_claim(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "base-thirty", scorecard("1", failures=10, false_accepts=0, hours=30))
        p["candidates"] = [sample_entry(self.root, "cand-thirty", scorecard("2", failures=5, false_accepts=0, hours=30))]
        p["budgets"].update(min_negative_hours=30, max_far_upper_95_per_hour=.13)
        result = rank.select(p)
        self.assertEqual(result["research_leader_id"], "cand-thirty")
        self.assertEqual(result["selection_statistical_policy"], rank.SELECTION_STATISTICAL_POLICY)
        self.assertEqual(result["per_keyword_support_status"], "NOT_VERIFIED")
        self.assertEqual(result["per_keyword_non_regression_status"], "NOT_VERIFIED")
        self.assertFalse(result["shipping_approved"])

    def test_equal_metrics_keep_champion(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "equal-base", scorecard("1", failures=5, false_accepts=0))
        p["candidates"] = [sample_entry(self.root, "equal-candidate", scorecard("2", failures=5, false_accepts=0))]
        result = rank.select(p)
        self.assertIsNone(result["research_leader_id"])
        self.assertEqual(result["candidate_results"][0]["paired_verdict"], "neutral")
        self.assertIn("no-strict-baseline-dominance", result["candidate_results"][0]["blockers"])

    def test_invalid_baseline_budget_does_not_rank_good_candidate(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "ineligible-base",
            scorecard("1", failures=10, false_accepts=2), rss=9000)
        result = rank.select(p)
        self.assertFalse(result["control_eligible_for_development_rank"])
        self.assertIsNone(result["research_leader_id"])
        self.assertTrue(all("baseline-ineligible" in c["blockers"] for c in result["candidate_results"]))

    def test_legacy_neutral_does_not_block_exact_improvement(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "one-base", scorecard("1", failures=5, false_accepts=1))
        p["candidates"] = [sample_entry(self.root, "zero-candidate", scorecard("2", failures=5, false_accepts=0))]
        result = rank.select(p)
        self.assertEqual(result["candidate_results"][0]["paired_verdict"], "neutral")
        self.assertEqual(result["research_leader_id"], "zero-candidate")

    def test_exact_regression_not_hidden_by_legacy_reverse_order(self) -> None:
        p = copy.deepcopy(self.plan)
        p["baseline"] = sample_entry(self.root, "zero-base", scorecard("1", failures=10, false_accepts=0))
        p["candidates"] = [sample_entry(self.root, "one-candidate", scorecard("2", failures=5, false_accepts=1))]
        result = rank.select(p)
        self.assertEqual(result["candidate_results"][0]["paired_verdict"], "improved")
        self.assertIsNone(result["research_leader_id"])
        self.assertIn("false-accept-confidence-regressed", result["candidate_results"][0]["blockers"])

    def test_huge_counts_are_rejected_before_statistical_math(self) -> None:
        for field in ("expected", "false_rejects", "observed_false_accepts"):
            p = copy.deepcopy(self.plan)
            card = scorecard("2", failures=5, false_accepts=0)
            target = card["far_rate"] if field == "observed_false_accepts" else card["metrics"]
            target[field] = 10 ** 1000
            p["candidates"] = [sample_entry(self.root, "huge-" + field, card)]
            with self.subTest(field=field), self.assertRaises(ValueError):
                rank.select(p)

    def test_bad_exposure_is_rejected(self) -> None:
        for hours in (0, -1, True, None, "10", 10 ** 1000):
            p = copy.deepcopy(self.plan)
            card = scorecard("2", failures=5, false_accepts=0)
            card["exposure"]["exposure_hours"] = hours
            p["candidates"] = [sample_entry(self.root, "bad-hours", card)]
            with self.subTest(hours=str(hours)[:20]), self.assertRaises(ValueError):
                rank.select(p)
        for hours in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                rank.number(hours, "exposure_hours", strictly_positive=True)

    def test_language_binding_cannot_be_report_time_spoofed(self) -> None:
        p = copy.deepcopy(self.plan)
        p["locale"] = "en-US"
        with self.assertRaisesRegex(ValueError, "LANGUAGE_COHORT_BINDING_DRIFT"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        p["language_binding"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "LANGUAGE_BINDING_BYTES_CHANGED"):
            rank.select(p)
        p = copy.deepcopy(self.plan)
        note = json.loads(Path(p["language_binding"]["path"]).read_text())
        note["training_admitted"] = True
        source = self.root / "forged-locale.json"
        source.write_text(json.dumps(note, sort_keys=True), encoding="utf-8")
        p["language_binding"] = {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        with self.assertRaisesRegex(ValueError, "LANGUAGE_COHORT_BINDING_DRIFT"):
            rank.select(p)

    def upgrade_entry(self, entry):
        card_path = Path(entry["scorecard_path"])
        card = json.loads(card_path.read_text())
        card["schema_version"] = 5
        card["identity"]["far_policy"] = rank.SELECTION_STATISTICAL_POLICY
        card["identity"]["comparison_mode"] = "ablation"
        card["far_rate"]["confidence_level"] = .95
        card["far_rate"]["upper_bound_95_per_hour"] = rank.poisson_rate_upper(
            card["far_rate"]["observed_false_accepts"], card["exposure"]["exposure_hours"], .95)
        card_path.write_text(json.dumps(card))
        entry["scorecard_sha256"] = hashlib.sha256(card_path.read_bytes()).hexdigest()

    def test_schema5_exact_cards_are_accepted_without_legacy_relabelling(self) -> None:
        plan = copy.deepcopy(self.plan)
        for entry in [plan["baseline"], *plan["candidates"]]:
            self.upgrade_entry(entry)
        result = rank.select(plan)
        self.assertEqual(result["input_far_policy"], rank.SELECTION_STATISTICAL_POLICY)
        for candidate in result["candidate_results"]:
            self.assertIsNone(candidate["legacy_v4_comparison"])
            self.assertFalse(candidate["scorecard_comparison"]["historical_statistics"])

    def test_same_protocol_label_cannot_mix_v4_and_v5(self) -> None:
        plan = copy.deepcopy(self.plan)
        self.upgrade_entry(plan["candidates"][0])
        with self.assertRaisesRegex(ValueError, "FAR policy mismatch"):
            rank.select(plan)

    def test_source_only_private_create_once_file(self) -> None:
        plan = self.root / "plan.json"
        plan.write_text(json.dumps(self.plan, sort_keys=True), encoding="utf-8")
        output = self.root / "ranked.json"
        self.assertEqual(rank.main(["--plan", str(plan), "--output", str(output)]), 0)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        stored = json.loads(output.read_text())
        self.assertFalse(stored["shipping_approved"])
        with self.assertRaises(FileExistsError):
            rank.main(["--plan", str(plan), "--output", str(output)])
        with self.assertRaisesRegex(ValueError, "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED"):
            rank.main(["--plan", str(plan), "--output",
                       str(ROOT / "must-not-publish-model-ranking.json")])


if __name__ == "__main__":
    unittest.main()
