"""Saved JSON and invented objects only: no acoustic or process execution."""
import copy
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import quality_gates as gates
import review_saved

FIXTURE = json.loads((HERE / "fixtures/public_regressions.json").read_text())


def frozen(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def row(identity, text="你好小窝", **changes):
    return dict({"id": identity, "actual_text": text, "source_group": "voice-A", "split": "dev",
                 "review": {"status": "clean", "independent_human": True, "complete": True},
                 "pcm_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                 "exposure": "FRESH", "lineage_status": "unknown"}, **changes)


class CoverageTests(unittest.TestCase):
    def run_gate(self, rows, declarations=None, policy=None):
        policy = policy or FIXTURE["coverage_policy"]
        declarations = declarations or [{"source_group": "voice-A", "split": "dev", "role": "balanced"}]
        return gates.coverage_admission(rows, declarations, policy, frozen(policy))

    def complete_group(self):
        return [row(str(i), text) for i, text in enumerate(("你好小窝", "小窝小窝", "你好小屋", "小屋小屋", "你好你好", "小窝", "小挖"))]

    def test_illustrative_missing_dev_and_train_wakes_reject_balanced_claim(self):
        result = self.run_gate(FIXTURE["coverage_rows"], FIXTURE["declarations"])
        by_voice = {r["source_group"]: r for r in result["groups"]}
        self.assertEqual(by_voice["illustrative:dev"]["counts"]["K1"], 0)
        self.assertEqual(by_voice["illustrative:dev"]["counts"]["K2"], 0)
        self.assertEqual(by_voice["illustrative:train-a"]["counts"]["K2"], 0)
        self.assertEqual(by_voice["illustrative:train-b"]["counts"]["K1"], 0)
        self.assertFalse(result["balanced_admission"])
        self.assertIn("illustrative-dev-incomplete", by_voice["illustrative:dev"]["excluded_unclean_or_unbound"])

    def test_configured_absolute_coverage_can_pass_but_is_not_training_authority(self):
        result = self.run_gate(self.complete_group())
        self.assertTrue(result["balanced_admission"])
        self.assertFalse(result["training_admitted"])
        policy = copy.deepcopy(FIXTURE["coverage_policy"])
        policy["minima"]["K1"] = 2
        self.assertFalse(self.run_gate(self.complete_group(), policy=policy)["balanced_admission"])

    def test_policy_mutation_after_freeze_fails(self):
        policy = copy.deepcopy(FIXTURE["coverage_policy"])
        policy["minima"]["K1"] = 20
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            gates.coverage_admission([], [], policy, FIXTURE["coverage_policy_sha256"])

    def test_zero_wake_minimum_cannot_remove_required_positive(self):
        policy = copy.deepcopy(FIXTURE["coverage_policy"])
        policy["minima"]["K2"] = 0
        with self.assertRaisesRegex(ValueError, "positive wake minima"):
            self.run_gate([], policy=policy)

    def test_intent_asr_or_partial_labels_cannot_fill_missing_positives(self):
        for status in ("missing", "inaudible", "incomplete", "ambiguous"):
            rows = self.complete_group()
            rows[0].update(intended_text="你好小窝", asr_text="你好小窝", asr_agreement=1.0)
            rows[0]["review"]["status"] = status
            self.assertFalse(self.run_gate(rows)["balanced_admission"])
        for field in ("independent_human", "complete"):
            rows = self.complete_group()
            rows[0]["review"][field] = False
            self.assertFalse(self.run_gate(rows)["balanced_admission"])

    def test_actual_text_wins_over_intent_and_nonwake_category(self):
        rows = self.complete_group()
        rows[0].update(actual_text="你好", intended_text="你好小窝")
        rows[2].update(actual_text="小屋", nonwake_case="你好小屋")
        result = self.run_gate(rows)["groups"][0]
        self.assertEqual(result["counts"]["K1"], 0)
        self.assertEqual(result["counts"]["nonwake:你好小屋"], 0)

    def test_background_is_separate_and_cannot_replace_missing_dev(self):
        rows = [row("bg", "你好小屋", source_group="background", split="background")]
        declarations = [{"source_group": "voice-A", "split": "dev", "role": "balanced"},
                        {"source_group": "background", "split": "background", "role": "negative_only"}]
        result = self.run_gate(rows, declarations)
        self.assertFalse(result["balanced_admission"])
        self.assertEqual(result["groups"][1]["status"], "SEPARATE_NEGATIVE_ONLY")
        rows[0]["actual_text"] = "你好小窝"
        self.assertEqual(self.run_gate(rows, declarations)["groups"][1]["status"], "INVALID_NEGATIVE_ONLY")

    def test_renamed_duplicate_pcm_does_not_meet_two_example_minimum(self):
        policy = copy.deepcopy(FIXTURE["coverage_policy"])
        policy["minima"]["K1"] = 2
        rows = self.complete_group()
        rows.append(dict(rows[0], id="renamed"))
        result = self.run_gate(rows, policy=policy)
        self.assertEqual(result["groups"][0]["counts"]["K1"], 1)
        self.assertFalse(result["balanced_admission"])

    def test_undeclared_source_cannot_disappear_from_report(self):
        with self.assertRaisesRegex(ValueError, "undeclared group"):
            self.run_gate([row("bad", source_group="other")])

    def test_same_wav_with_changed_pcm_is_corruption(self):
        rows = [row("a", wav_sha256="same"), row("b", wav_sha256="same")]
        with self.assertRaisesRegex(ValueError, "conflicting PCM"):
            self.run_gate(rows)

    def test_wav_only_alias_deduplicates_against_pcm_bound_original(self):
        rows = self.complete_group()
        rows[0]["wav_sha256"] = "same-wave"
        alias = dict(rows[0], id="wav-only-alias")
        alias.pop("pcm_sha256")
        rows.append(alias)
        self.assertEqual(self.run_gate(rows)["groups"][0]["counts"]["K1"], 1)

    def test_empty_transcript_is_not_silence_or_nonwake_truth(self):
        self.assertIsNone(gates.human_truth(row("empty", "")))


class IdentityTests(unittest.TestCase):
    def test_saved_dylan_new_waveforms_are_not_unseen_voice(self):
        report = gates.identity_audit(FIXTURE["old6_rows"], FIXTURE["identity_history"])
        self.assertEqual(report["status"], "CHECKED_DECLARED_LINEAGE")
        self.assertTrue(all(r["fresh_recording"] for r in report["rows"]))
        self.assertTrue(all(r["effective_exposure"] == "EXPOSED" and not r["unseen_voice"] and not r["strong_independence"] for r in report["rows"]))

    def test_cross_generator_reference_hash_defeats_renamed_voice(self):
        rows = copy.deepcopy(FIXTURE["old6_rows"])
        rows[0].update(id="new-generator-new-name", voice_identity="renamed", exposure="FRESH", split="heldout")
        report = gates.identity_audit(rows, FIXTURE["identity_history"])
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("SHARED_IDENTITY_ACROSS_SPLITS", {c["kind"] for c in report["conflicts"]})
        self.assertIn("EXPOSURE_DOWNGRADE", {c["kind"] for c in report["conflicts"]})

    def test_renamed_audio_or_pcm_cannot_become_fresh(self):
        old = row("old", exposure="EXPOSED", use="regression", split="regression", wav_sha256="W")
        for field in ("pcm_sha256", "wav_sha256"):
            new = row("new", wav_sha256="new-wave", split="heldout")
            new[field] = old[field]
            report = gates.identity_audit([new], [old])
            self.assertEqual(report["rows"][0]["effective_exposure"], "EXPOSED")
            self.assertFalse(report["rows"][0]["fresh_recording"])
            self.assertEqual(report["status"], "FAIL")

    def test_explicit_lineage_transitively_propagates_exposure(self):
        old = row("a", exposure="EXPOSED", split="regression", lineage_ids=["root"])
        middle = row("b", exposure="EXPOSED", split="regression", lineage_ids=["root", "next"])
        ledger = gates.identity_audit([middle], [old])["ledger"]
        new = row("c", split="heldout", lineage_ids=["next"])
        self.assertEqual(gates.identity_audit([new], ledger)["rows"][0]["effective_exposure"], "EXPOSED")

    def test_unknown_lineage_never_earns_strong_independence(self):
        report = gates.identity_audit([row("new", split="heldout")])
        self.assertFalse(report["rows"][0]["strong_independence"])
        self.assertFalse(report["rows"][0]["unseen_voice"])

    def test_exposed_old6_is_accepted_for_code_regression_but_not_training_or_heldout(self):
        old = copy.deepcopy(FIXTURE["old6_rows"][0])
        self.assertEqual(gates.identity_audit([old])["status"], "CHECKED_DECLARED_LINEAGE")
        for update in ({"use": "training", "split": "train"}, {"split": "heldout"}):
            self.assertEqual(gates.identity_audit([dict(old, **update)])["status"], "FAIL")

    def test_restriction_cannot_be_dropped_by_renaming(self):
        old = FIXTURE["old6_rows"][0]
        new = dict(old, id="alias", use="development", split="dev")
        new.pop("restriction")
        report = gates.identity_audit([new], [old])
        self.assertIn("REGRESSION_LINEAGE_REPURPOSED", {c["kind"] for c in report["conflicts"]})

    def test_repeated_current_content_is_reported(self):
        first = row("one")
        second = dict(first, id="two")
        report = gates.identity_audit([first, second])
        self.assertIn(["one", "two"], report["repeated_content"])
        self.assertTrue(all(not r["fresh_recording"] for r in report["rows"]))

    def test_restriction_follows_direct_derivative_but_not_whole_stock_voice(self):
        old = FIXTURE["old6_rows"][0]
        sibling = row("new-voice-recording", exposure="EXPOSED", voice_identity=old["voice_identity"],
                      use="development", split="dev", reference_sha256=old["reference_sha256"])
        self.assertEqual(gates.identity_audit([sibling], [old])["status"], "CHECKED_DECLARED_LINEAGE")
        derived = dict(sibling, reference_sha256=old["wav_sha256"])
        self.assertEqual(gates.identity_audit([derived], [old])["status"], "FAIL")

    def test_restriction_survives_multiple_alias_ledgers(self):
        old = FIXTURE["old6_rows"][0]
        alias = dict(old, id="alias")
        alias.pop("restriction")
        ledger = gates.identity_audit([alias], [old])["ledger"]
        self.assertEqual(ledger[-1]["restriction"], "regression_only")
        renamed = dict(alias, id="third-name", use="model_selection")
        self.assertEqual(gates.identity_audit([renamed], ledger)["status"], "FAIL")


class EventAndTimingTests(unittest.TestCase):
    def test_saved_equal_total_events_conceal_new_miss_and_false(self):
        result = gates.event_comparison(FIXTURE["old6_rows"], FIXTURE["old6_original"], FIXTURE["old6_candidate"], {"greedy_exact": "49/49", "ctc": 0.0001})
        self.assertEqual(result["counts"]["original"]["events"], 2)
        self.assertEqual(result["counts"]["candidate"]["events"], 2)
        self.assertEqual(result["event_status"], "EVENT_REGRESSION")
        rows = {r["id"]: r for r in result["rows"]}
        self.assertEqual(rows["Z5"]["new_misses"], ["K1"])
        self.assertEqual(rows["Z6"]["new_false_keywords"], {"K1": 1})
        self.assertEqual(rows["Z2"]["new_false_keywords"], {})
        self.assertEqual(rows["Z3"]["expected"], [])
        self.assertEqual(result["counts"]["candidate"]["repeats"], 0)
        self.assertFalse(result["scientific_qualification"])

    def test_training_greedy_does_not_supply_missing_event_evidence(self):
        result = gates.event_comparison([row("a")], {}, {}, {"greedy_exact": "49/49", "ctc_improved": True})
        self.assertEqual(result["event_status"], "UNKNOWN")
        self.assertFalse(result["training_fit_implies_event_pass"])

    def test_wrong_keyword_and_repeat_are_distinct(self):
        result = gates.event_comparison([row("a")], {"a": [{"keyword": 1}]}, {"a": [{"keyword": 2}, {"keyword": 2}]})
        self.assertEqual(result["rows"][0]["new_wrong_keywords"], {"K2": 2})
        self.assertEqual(result["rows"][0]["new_repeats"], {"K2": 1})
        self.assertEqual(result["rows"][0]["new_misses"], ["K1"])

    def test_unknown_labels_and_multiple_occurrences_never_enter_clean_denominator(self):
        for r in (row("a", "你好小窝你好小窝"), row("a", review={"status": "ambiguous"})):
            result = gates.event_comparison([r], {"a": []}, {"a": []})
            self.assertEqual(result["event_status"], "UNKNOWN")
            self.assertEqual(result["unscored"], ["a"])

    def test_illustrative_28_pairs_keep_published_aggregate_delays_visible(self):
        result = gates.availability_report(FIXTURE["timing_illustration_original"], FIXTURE["timing_illustration_candidate"], FIXTURE["timing_illustration_sample_rates"])
        self.assertEqual(result["matched_events"], 28)
        self.assertEqual(result["counts"], {"later": 3, "unchanged": 25, "earlier": 0})
        self.assertEqual(sorted(r["availability_delta_ms"] for r in result["matches"] if r["delta_samples"]), [120, 180, 300])
        self.assertIsNone(result["word_end_latency_ms"])

    def test_event_protection_pass_still_reports_delay(self):
        original = {"a": [{"keyword": 1, "available_samples": 100}]}
        candidate = {"a": [{"keyword": 1, "available_samples": 200}]}
        self.assertEqual(gates.event_comparison([row("a")], original, candidate)["event_status"], "EVENT_PROTECTION_PASS")
        timing = gates.availability_report(original, candidate, {"a": 1000})
        self.assertEqual(timing["matches"][0]["availability_delta_ms"], 100)
        self.assertEqual(timing["word_end_latency_status"], "NOT_MEASURED")

    def test_duplicate_event_matches_are_ambiguous_not_nearest_paired(self):
        events = {"a": [{"keyword": 1, "available_samples": 100}, {"keyword": 1, "available_samples": 200}]}
        result = gates.availability_report(events, events, {"a": 1000})
        self.assertEqual(result["matched_events"], 0)
        self.assertEqual(len(result["ambiguous"]), 1)

    def test_missing_rate_or_coordinate_never_becomes_zero_delay(self):
        old = {"a": [{"keyword": 1, "available_samples": 100}]}
        for new, rate in ((old, {}), ({"a": [{"keyword": 1}]}, {"a": 1000})):
            result = gates.availability_report(old, new, rate)
            self.assertEqual(result["matched_events"], 0)
            self.assertEqual(len(result["unknown"]), 1)


class SupervisionRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = HERE / "tests/vendor/supervision_observation.py"
        spec = importlib.util.spec_from_file_location("existing_supervisor_pure_tests", path)
        cls.supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.supervisor)

    def observe(self, status="VmRSS: 100 kB\nThreads: 1\nVmHWM: 100 kB\nVmSize: 200 kB\n", child_error=None):
        class Process:
            pid = 42
            def poll(self):
                return None
        def reader(path):
            if str(path).endswith("/status"):
                return status
            if str(path).endswith("/children"):
                raise child_error or FileNotFoundError(errno.ENOENT, "missing", str(path))
            return "\n".join(k + ": 0" for k in self.supervisor.IO_COUNTERS)
        return self.supervisor.observe_process(Process(), reader)

    def test_reuses_existing_children_enoent_fix_not_zero(self):
        state = self.observe()
        self.assertEqual(state["children_observation"], "NOT_AVAILABLE")
        self.assertIsNone(state["children_count"])
        self.assertIsNone(state["children"])
        self.assertEqual(state["lifetime_child_absence"], "NOT_PROVEN")
        self.assertIsNone(self.supervisor.guard_reason(state, 0, 0))

    def test_reuses_existing_missing_critical_rss_and_threads_stop(self):
        for status in ("Threads: 1\n", "VmRSS: 100 kB\n", "VmRSS: nope kB\nThreads: 1\n"):
            state = self.observe(status)
            self.assertEqual(self.supervisor.guard_reason(state, 0, 0), "CRITICAL_MONITORING_UNAVAILABLE")

    def test_children_permission_or_wrong_path_enoent_not_relaxed(self):
        for error in (PermissionError(errno.EACCES, "denied", "/proc/42/task/42/children"), FileNotFoundError(errno.ENOENT, "missing", "/wrong/children")):
            self.assertEqual(self.observe(child_error=error)["children_observation"], "FAILED_UNAVAILABLE")

    def test_illustrative_supervision_integrity_is_not_scientific_pass(self):
        for resource in FIXTURE["supervision_illustration"].values():
            result = gates.supervision_interpretation(resource)
            self.assertEqual(result["integrity_status"], "PASS_SAVED_LIMITED_OBSERVATIONS")
            self.assertFalse(result["continuous_lifetime_compliance_proven"])
            self.assertEqual(result["scientific_outcome"], "NOT_ASSESSED")

    def test_missing_critical_or_rewritten_children_invalidates_saved_report(self):
        resource = copy.deepcopy(FIXTURE["supervision_illustration"]["original_A20"])
        for index, value in ((2, None), (5, None), (6, 0)):
            changed = copy.deepcopy(resource)
            changed["proc_samples_compact"][0][index] = value
            self.assertEqual(gates.supervision_interpretation(changed)["integrity_status"], "FAIL")

    def test_lifetime_rss_difference_does_not_become_model_memory_regression(self):
        report = gates.rss_comparison(13848, 58892, 4079616, 4079616)
        self.assertEqual(report["lifetime_native"]["delta"], 45044)
        self.assertEqual(report["sampled"]["delta"], 0)
        self.assertFalse(report["scopes_interchangeable"])
        self.assertEqual(report["model_memory_regression"], "NOT_ESTABLISHED")

    def test_missing_prego_or_failed_terminal_observation_fails(self):
        original = FIXTURE["supervision_illustration"]["original_A20"]
        for mutation in ("pre_go", "terminal", "failed_status", "unknown_status"):
            resource = copy.deepcopy(original)
            if mutation == "pre_go":
                resource.pop("pre_go_observation")
            elif mutation == "terminal":
                resource["proc_samples_compact"][-1][8] = "FAILED_UNAVAILABLE"
            else:
                resource["status"] = "FAILED_NO_RETRY" if mutation == "failed_status" else "UNKNOWN"
            result = gates.supervision_interpretation(resource)
            self.assertEqual(result["integrity_status"], "FAIL")
            self.assertEqual(result["original_execution_status"], resource["status"])

    def test_metadata_receipt_is_not_byte_recovery(self):
        data = b"saved raw JSON\n"
        expected = [{"path": "raw.jsonl", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "required_raw": True}]
        for supplied in ({}, {"raw.jsonl": expected[0]}, {"raw.jsonl": b"corrupt"}):
            result = gates.recovery_interpretation("FAILED_NO_RETRY", expected, supplied)
            self.assertEqual(result["byte_recovery"], "INCOMPLETE")
            self.assertTrue(result["missing_raw"])
            self.assertEqual(result["execution_status"], "FAILED_NO_RETRY")

    def test_actual_bytes_do_not_rewrite_failed_unknown_or_missing_raw(self):
        data = b"saved raw JSON\n"
        expected = [{"path": "raw.jsonl", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "required_raw": True}]
        for status in ("FAILED_NO_RETRY", "UNKNOWN"):
            result = gates.recovery_interpretation(status, expected, {"raw.jsonl": data})
            self.assertEqual(result["byte_recovery"], "BYTE_VERIFIED")
            self.assertEqual(result["execution_status"], status)
            self.assertFalse(result["recovery_implies_execution_pass"])
        self.assertEqual(gates.recovery_interpretation("UNKNOWN", [], {})["byte_recovery"], "INCOMPLETE")


class IntegrationTests(unittest.TestCase):
    def test_cli_report_keeps_evidence_controls_and_unknowns_separate(self):
        result = review_saved.report(FIXTURE)
        self.assertEqual(result["new_model_asr_training_calls"], 0)
        self.assertTrue(result["implemented_controls"])
        self.assertTrue(result["hypotheses"])
        self.assertIn("real ASR accuracy", result["remaining_unknowns"])
        self.assertFalse(result["candidate_adopted"])

    def test_actual_label_policy_agrees_including_oov_nonwake(self):
        sibling = HERE / "tests/vendor"
        sys.path.insert(0, str(sibling))
        import actual_label_policy
        for item in FIXTURE["old6_rows"] + FIXTURE["coverage_rows"]:
            expected = actual_label_policy.expected_keywords(item)
            ours = gates.human_truth(item)
            self.assertEqual(None if ours is None else sorted(ours), expected)
        assessment = actual_label_policy.assess_actual_label(next(r for r in FIXTURE["old6_rows"] if r["id"] == "Z3"))
        self.assertEqual(assessment["expected_keywords"], [])
        self.assertFalse(assessment["ctc_label_eligible"])
        self.assertIsNone(assessment["ctc_target"])


if __name__ == "__main__":
    unittest.main()
