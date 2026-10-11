#!/usr/bin/env python3
"""Fabricated contract fixtures only; no acoustic model or decoder is executed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import contracts as c
import compare_saved as compare


def fixture_manifest():
    """Two disjoint real-like fixture splits, not actual human data."""
    manifest = c.load_json(HERE / "data_manifest.json")
    manifest["purpose"] = "qualified_event_evaluation_fixture"
    manifest["recordings"] = []
    for split in ("dev", "test"):
        for kind in ("positive", "negative"):
            row = {"recording": f"{split}-{kind}",
                   "trace": {"path": f"{split}-{kind}.kwtr", "sha256": hashlib.sha256(f"{split}-{kind}".encode()).hexdigest()},
                   "source_recording_sha256": hashlib.sha256(f"source-{split}-{kind}".encode()).hexdigest(),
                   "source_family": f"family-{split}", "speaker": f"speaker-{split}", "session": f"session-{split}",
                   "initial_state": {"decoder": "reset_at_trace_start", "acoustic": "documented_zero_state",
                                     "original_nightly_equivalence": "not_applicable"},
                   "overlap_group": f"source-{split}-{kind}", "source_interval_s": [0, 10 if kind == "positive" else 36000],
                   "annotation_role": f"gold_{split}", "annotation_status": "complete", "split": split,
                   "expected": [{"keyword_id": 1, "start_s": 1, "end_s": 2}] if kind == "positive" else [],
                   "truth_source": {"sha256": "c" * 64}, "origin": "real_human", "independent_exposure": True,
                   "ordinary_negative": kind == "negative", "duration_s": 10 if kind == "positive" else 36000}
            manifest["recordings"].append(row)
    return manifest


def future_report(manifest, arm, threshold, split="dev", hit=True, negative_events=0):
    rows = [row for row in manifest["recordings"] if row["split"] == split]
    controls = c.load_json(HERE / "plan.json")["shared_controls"]
    del controls["threshold"]
    replay = {"contract": {"kind": "full_decoder_replay", "split": split, "arm": arm, "threshold": threshold,
              "model_sha256": c.MODEL_SHA, "keyword_pack_sha256": c.PACK_SHA,
              "manifest_canonical_sha256": c.canonical_sha(manifest),
              "decoder_executable_sha256": "a" * 64, "shared_controls_sha256": c.canonical_sha(controls),
              "trace_sha256_by_recording": {row["recording"]: row["trace"]["sha256"] for row in rows}},
              "receipt_sha256": hashlib.sha256(f"fixture-{arm}-{threshold}-{split}".encode()).hexdigest()}
    detections = ([{"recording": f"{split}-positive", "keyword_id": 1, "time_s": 2.1, "confidence": .9}]
                  if hit else [])
    detections += [{"recording": f"{split}-negative", "keyword_id": 2, "time_s": 2 + i, "confidence": .9}
                   for i in range(negative_events)]
    return compare.score_qualified(manifest, detections, split=split, arm=arm, threshold=threshold, replay=replay)


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.plan = c.load_json(HERE / "plan.json")
        self.manifest = c.load_json(HERE / "data_manifest.json")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def receipts(self):
        result = {}
        for arm in c.ARMS:
            replays = []
            for index, row in enumerate(self.manifest["recordings"]):
                path = self.root / f"{arm}-{index}.jsonl"
                found = []
                if arm == "single_fuzzy_wo" and index == 0:
                    found = [{"recording": row["recording"], "keyword_id": 2, "time_s": 1.005, "confidence": .6}]
                path.write_text("".join(json.dumps(event) + "\n" for event in found))
                replays.append({"recording": row["recording"], "trace_sha256": row["trace"]["sha256"],
                     "initial_state": row["initial_state"], "returncode": 0, "complete_trace_replay": True,
                     "argv": ["fixture-replay", "fixture-model", "fixture-pack", "fixture-trace",
                              row["recording"], "--child-admission", arm],
                     "invocation_input_hashes": {"executable": "a" * 64, "model": c.MODEL_SHA,
                                                 "keywords": c.PACK_SHA, "trace": row["trace"]["sha256"]},
                     "observed_invocation_inputs": [
                         {"path": "fixture-replay", "sha256": "a" * 64, "bytes": 100},
                         {"path": "fixture-model", "sha256": c.MODEL_SHA, "bytes": 6812},
                         {"path": "fixture-pack", "sha256": c.PACK_SHA, "bytes": 100},
                         {"path": "fixture-trace", "sha256": row["trace"]["sha256"], "bytes": row["trace"]["bytes"]}],
                     "stdout": {"path": path.name, "sha256": c.sha256_file(path), "bytes": path.stat().st_size}})
            result[arm] = {"schema": "kws-saved-paired-arm-v1", "arm": arm, "threshold": .55,
                           "input_binding": c.input_binding(self.plan, self.manifest),
                           "decoder_executable_sha256": "a" * 64, "replay_kind": "full_decoder_replay",
                           "acoustic_forward_calls": 0, "replays": replays}
        return result

    def test_committed_contracts_valid(self):
        c.validate_plan(self.plan)
        c.validate_manifest(self.manifest)
        self.assertEqual(len(self.manifest["recordings"]), 13)
        self.assertTrue(all(row["expected"] is None for row in self.manifest["recordings"]))
        self.assertEqual(self.manifest["coverage"]["ordinary_real_negative_hours"], 0)

    def test_model_no_wu(self):
        self.plan["model"]["tokens"].append("wu1")
        with self.assertRaisesRegex(c.ContractError, "vocabulary"):
            c.validate_plan(self.plan)

    def test_current_threshold_sweep_rejected(self):
        self.plan["thresholds"] = [.5, .55]
        with self.assertRaisesRegex(c.ContractError, "fixed threshold"):
            c.validate_plan(self.plan)

    def test_shared_controls_cannot_drift(self):
        self.plan["shared_controls"]["root_nonblank_margin"] = 0
        with self.assertRaisesRegex(c.ContractError, "controls"):
            c.validate_plan(self.plan)

    def test_missing_expected_is_not_negative(self):
        del self.manifest["recordings"][0]["expected"]
        with self.assertRaisesRegex(c.ContractError, "missing expected"):
            c.validate_manifest(self.manifest)

    def test_unknown_truth_not_complete(self):
        self.manifest["recordings"][0]["annotation_status"] = "complete"
        with self.assertRaisesRegex(c.ContractError, "remain unknown|stay unknown"):
            c.validate_manifest(self.manifest)

    def test_protected_historical_misrole(self):
        for change in ({"split": "dev", "annotation_role": "gold_dev"}, {"origin": "real_human"},
                       {"independent_exposure": True}, {"ordinary_negative": True}):
            with self.subTest(change=change):
                manifest = deepcopy(self.manifest)
                manifest["recordings"][0].update(change)
                with self.assertRaisesRegex(c.ContractError, "protected historical"):
                    c.validate_manifest(manifest)

    def test_protected_hash_cannot_be_laundered_by_removing_artifact_id(self):
        row = self.manifest["recordings"][0]
        del row["source_artifact_id"]
        row["annotation_role"] = "gold_test"
        with self.assertRaisesRegex(c.ContractError, "protected historical"):
            c.validate_manifest(self.manifest)

    def test_train_role_rejected(self):
        self.manifest["recordings"][0]["annotation_role"] = "train"
        with self.assertRaises(c.ContractError):
            c.validate_manifest(self.manifest)

    def test_missing_cache_stops_without_creation(self):
        with self.assertRaisesRegex(c.ContractError, "STOP, no fallback"):
            c.validate_saved_inputs(self.plan, self.manifest, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(c.ContractError, "escapes"):
            c.saved_path(self.root, "../not-authorized")

    def test_hash_mismatch_stops(self):
        (self.root / "trace").write_bytes(b"saved")
        with self.assertRaisesRegex(c.ContractError, "hash mismatch"):
            c.checked_file(self.root, {"path": "trace", "sha256": "0" * 64})

    def test_kwtr_geometry_frames_and_grid_checked(self):
        path = self.root / "fixture.kwtr"
        header = c.HEADER.pack(b"KWTRACE1", 1, 5, 0, 16000, 400, 320, 0,
                               c.FINGERPRINT, 2, c.MODEL_SHA.encode(), b"\0" * 8)
        payload = c.FRAME.pack(400, 1, 0, 0, 1, 2, 3, 4) + c.FRAME.pack(720, 1, 0, 4, 3, 2, 1, 0)
        path.write_bytes(header + payload)
        decl = {"frames": 2, "first_end_sample": 400, "last_end_sample": 720}
        c.inspect_trace(path, decl)
        for corrupt in (header + payload + b"x", header + payload[:-1],
                        header + c.FRAME.pack(400, 2, 0, 0, 1, 2, 3, 4) + payload[36:]):
            path.write_bytes(corrupt)
            with self.assertRaises(c.ContractError):
                c.inspect_trace(path, decl)

    def test_saved_comparison_counts_no_aggregate_or_qualification(self):
        result = compare.compare_saved(self.plan, self.manifest, self.receipts(), self.root)
        self.assertEqual(result["recordings"][0]["counts"]["single_fuzzy_wo"]["total"], 1)
        self.assertEqual(result["recordings"][0]["single_fuzzy_wo_minus_strict_events"], 1)
        self.assertIsNone(result["aggregate_event_count"])
        self.assertIsNone(result["qualification"]["frr"])
        self.assertIsNone(result["qualification"]["negative_far_upper95_per_hour"])
        self.assertEqual(result["qualification"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_two_arm_inputs_differ_rejected(self):
        receipts = self.receipts()
        receipts[c.ARMS[1]]["input_binding"]["data_manifest_canonical_sha256"] = "d" * 64
        with self.assertRaisesRegex(c.ContractError, "input/denominator"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_two_arm_builds_differ_rejected(self):
        receipts = self.receipts()
        receipts[c.ARMS[1]]["decoder_executable_sha256"] = "b" * 64
        for row in receipts[c.ARMS[1]]["replays"]:
            row["invocation_input_hashes"]["executable"] = "b" * 64
            row["observed_invocation_inputs"][0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(c.ContractError, "same research executable"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_partial_replay_and_wrong_initial_state_rejected(self):
        for key, value in (("returncode", 1), ("complete_trace_replay", False), ("initial_state", {})):
            receipts = self.receipts()
            receipts[c.ARMS[0]]["replays"][0][key] = value
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_wrong_threshold_or_filtered_detections_rejected(self):
        for key, value in (("threshold", .6), ("replay_kind", "filtered_detections"), ("acoustic_forward_calls", 1)):
            receipts = self.receipts()
            receipts[c.ARMS[0]][key] = value
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_missing_replay_and_extra_cli_override_rejected(self):
        receipts = self.receipts()
        receipts[c.ARMS[0]]["replays"].pop()
        with self.assertRaisesRegex(c.ContractError, "every manifest trace"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)
        receipts = self.receipts()
        receipts[c.ARMS[0]]["replays"][0]["argv"].extend(["--refractory-ms", "0"])
        with self.assertRaisesRegex(c.ContractError, "invocation"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_optional_production_reference_is_separate(self):
        receipts = self.receipts()
        reference = deepcopy(receipts[c.ARMS[0]])
        reference.update(schema="kws-saved-production-reference-v1", arm="production_reference",
                         reference_only=True, source_kind="unmodified_production", decoder_executable_sha256="f" * 64)
        for replay in reference["replays"]:
            replay["argv"] = ["production-replay", "model", "pack", "trace", replay["recording"]]
            replay["invocation_input_hashes"]["executable"] = "f" * 64
            for observed, path in zip(replay["observed_invocation_inputs"], replay["argv"][:4]):
                observed["path"] = path
            replay["observed_invocation_inputs"][0]["sha256"] = "f" * 64
        report = compare.compare_saved(self.plan, self.manifest, receipts, self.root, reference_receipt=reference)
        self.assertEqual(report["production_reference"]["status"], "SAVED_CONTROLLED_REFERENCE")
        self.assertFalse(report["production_reference"]["selector_candidate"])
        self.assertEqual(report["recordings"][0]["production_reference"]["total"], 0)
        self.assertEqual(set(report["recordings"][0]["counts"]), set(c.ARMS))
        reference["reference_only"] = False
        with self.assertRaisesRegex(c.ContractError, "reference-only"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root, reference_receipt=reference)

    def test_output_hash_and_missing_saved_stdout_rejected(self):
        receipts = self.receipts()
        output = receipts[c.ARMS[0]]["replays"][0]["stdout"]
        (self.root / output["path"]).write_text("mutated")
        with self.assertRaisesRegex(c.ContractError, "hash mismatch"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)
        (self.root / output["path"]).unlink()
        with self.assertRaisesRegex(c.ContractError, "STOP, no fallback"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_result_event_wrong_recording_or_keyword_rejected(self):
        for key, value in (("recording", "unknown"), ("keyword_id", 3), ("confidence", float("nan")), ("time_s", -1)):
            receipts = self.receipts()
            output = receipts[c.ARMS[1]]["replays"][0]["stdout"]
            path = self.root / output["path"]
            event = json.loads(path.read_text())
            event[key] = value
            path.write_text(json.dumps(event) + "\n")
            output.update(sha256=c.sha256_file(path), bytes=path.stat().st_size)
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_current_manifest_cannot_omit_protected_trace(self):
        self.manifest["recordings"].pop()
        with self.assertRaisesRegex(c.ContractError, "exact 13 protected"):
            c.validate_manifest(self.manifest)

    def test_top_level_objects_and_huge_numbers_fail_closed(self):
        for bad in (None, [], 1, "wrong"):
            for function in (c.validate_plan, c.validate_manifest):
                with self.subTest(bad=bad, function=function.__name__), self.assertRaises(c.ContractError):
                    function(bad)
            with self.assertRaises(c.ContractError):
                c.validate_arm_receipt(bad, c.ARMS[0], self.plan, self.manifest, self.root)
        with self.assertRaises(c.ContractError):
            c.number(10 ** 400, "huge")

    def test_duplicate_jsonl_keys_are_rejected(self):
        path = self.root / "duplicate.jsonl"
        path.write_text('{"recording":"first","recording":"second"}\n')
        with self.assertRaisesRegex(c.ContractError, "duplicate JSON key"):
            c.load_detections(path)

    def test_historical_unknown_truth_cannot_be_empty_complete(self):
        self.manifest["recordings"][0].update(expected=[], annotation_status="complete",
            truth_source={"sha256": "c" * 64}, duration_s=7200)
        with self.assertRaisesRegex(c.ContractError, "protected historical unknown"):
            c.validate_manifest(self.manifest)

    def test_trace_path_and_historical_metadata_bindings(self):
        for mutate in (lambda m: m["recordings"][0]["trace"].update(path=m["recordings"][1]["trace"]["path"]),
                       lambda m: m["source_archive"].update(run_id=1),
                       lambda m: m.update(source_member_identity_sha256="a" * 64),
                       lambda m: m["source_cases"][0].update(expected_source_token_ids=[1, 2, 3, 4])):
            manifest = deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(c.ContractError):
                c.validate_manifest(manifest)

    def test_observed_invocation_path_hash_size_binding(self):
        receipts = self.receipts()
        receipts[c.ARMS[0]]["replays"][0]["argv"][1] = "a-different-model-file"
        with self.assertRaisesRegex(c.ContractError, "argv path/identity"):
            compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_trace_grid_and_score_serialization_tolerances(self):
        cases = [("time_s", 1.000, False), ("time_s", 1.00500049, True),
                 ("time_s", 1.00500051, False), ("confidence", .01, False),
                 ("confidence", .54999951, True), ("confidence", .54999949, False)]
        for key, value, accepted in cases:
            receipts = self.receipts()
            output = receipts[c.ARMS[1]]["replays"][0]["stdout"]
            path = self.root / output["path"]
            event = json.loads(path.read_text())
            event[key] = value
            path.write_text(json.dumps(event) + "\n")
            output.update(sha256=c.sha256_file(path), bytes=path.stat().st_size)
            with self.subTest(key=key, value=value):
                if accepted:
                    compare.compare_saved(self.plan, self.manifest, receipts, self.root)
                else:
                    with self.assertRaises(c.ContractError):
                        compare.compare_saved(self.plan, self.manifest, receipts, self.root)

    def test_duplicate_json_and_nan_rejected(self):
        path = self.root / "bad.json"
        for raw in ('{"x": 1, "x": 2}', '{"x": NaN}'):
            path.write_text(raw)
            with self.assertRaises(c.ContractError):
                c.load_json(path)


class FutureSelectorFixtureTest(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()

    def candidates(self):
        return [future_report(self.manifest, arm, threshold)
                for arm in c.ARMS for threshold in (.55, .65)]

    def test_fixture_measurement_never_claims_verified_evidence(self):
        report = future_report(self.manifest, c.ARMS[0], .55)
        self.assertEqual(report["status"], "FIXTURE_ONLY_MEASUREMENT")
        self.assertIs(report["fixture_only"], True)
        self.assertIs(report["actual_evidence_verified"], False)
        self.assertIs(report["qualification_claimed"], False)
        compare.validate_measurement_report(report)

    def test_real_purpose_rejects_unverified_receipts_and_inflated_exposure(self):
        for duration in (36000, 36000000):
            manifest = deepcopy(self.manifest)
            manifest["purpose"] = "qualified_event_evaluation"
            for row in manifest["recordings"]:
                if not row["expected"]:
                    row["duration_s"] = duration
                    row["source_interval_s"] = [0, duration]
            # Recomputed hashes and internally consistent declarations cannot
            # authenticate even the original denominator, let alone inflation.
            with self.subTest(duration=duration), self.assertRaisesRegex(
                    c.ContractError, "UNVERIFIED_QUALIFIED_EVIDENCE"):
                future_report(manifest, c.ARMS[0], .55)

    def test_real_selector_and_test_reject_fixtures_and_status_laundering(self):
        reports = self.candidates()
        selection = compare.select_fixture_dev_thresholds(
            reports, max_negative_far_upper95_per_hour=.4)
        test = [future_report(self.manifest, arm, .55, split="test") for arm in c.ARMS]
        for forged in (False, True):
            if forged:
                for report in reports + test:
                    report.update(status="QUALIFIED_RESEARCH_MEASUREMENT",
                                  fixture_only=False, actual_evidence_verified=True)
                selection.update(status="DEVELOPMENT_THRESHOLDS_FROZEN",
                                 fixture_only=False, actual_evidence_verified=True)
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    c.ContractError, "UNVERIFIED_QUALIFIED_EVIDENCE"):
                compare.select_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    c.ContractError, "UNVERIFIED_QUALIFIED_EVIDENCE"):
                compare.evaluate_frozen_test(selection, test)

    def test_fixture_status_and_evidence_flags_cannot_be_laundered(self):
        for change in ({"status": "QUALIFIED_RESEARCH_MEASUREMENT"},
                       {"fixture_only": False}, {"actual_evidence_verified": True}):
            report = future_report(self.manifest, c.ARMS[0], .55)
            report.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(
                    c.ContractError, "recomputed scoring inputs"):
                compare.validate_measurement_report(report)

    def test_known_machine_or_forbidden_gold_labels_rejected_at_every_depth(self):
        markers = [{"label_origin": "machine-consensus"},
                   {"label_source": "machine-three-asr-consensus-v1"},
                   {"evidence_class": "speech-three-asr-consensus-v1"},
                   {"evidence_class": "speech-three-asr-consensus-v2"},
                   {"evidence_class": "speech-three-asr-training-admission-v1"},
                   {"consensus_record": {}}, {"evaluation_gold_allowed": False},
                   {"qualification_allowed": False}, {"human_gold": False},
                   {"source_screen_only": True}, {"evaluation_gold_allowed": None}]
        for marker in markers:
            for placement in ("top", "truth_source", "nested"):
                manifest = deepcopy(self.manifest)
                row = manifest["recordings"][0]
                if placement == "top":
                    row.update(marker)
                elif placement == "truth_source":
                    row["truth_source"].update(marker)
                else:
                    row["truth_source"]["label_provenance"] = {"observations": [marker]}
                with self.subTest(marker=marker, placement=placement), self.assertRaisesRegex(
                        c.ContractError, "independent reference labels|screen-only"):
                    c.validate_manifest(manifest)

    def test_screen_weak_label_annotations_are_not_globally_forbidden(self):
        manifest = deepcopy(self.manifest)
        manifest["recordings"] = [manifest["recordings"][0]]
        manifest["recordings"][0].update(split="screen", annotation_role="screen_only",
            expected=None, annotation_status="unknown", independent_exposure=False,
            label_origin="machine-consensus", evaluation_gold_allowed=False)
        c.validate_manifest(manifest)

    def test_future_threshold_confidence_and_rounding_contract(self):
        for threshold, accepted in ((.55, True), (.9, True), (.90000049, True),
                                    (.90000051, False), (.99, False)):
            with self.subTest(threshold=threshold):
                if accepted:
                    future_report(self.manifest, c.ARMS[0], threshold)
                else:
                    with self.assertRaisesRegex(c.ContractError, "below declared replay threshold"):
                        future_report(self.manifest, c.ARMS[0], threshold)

    def test_qualified_fixture_is_valid(self):
        c.validate_manifest(self.manifest)

    def test_missing_truth_rejected(self):
        self.manifest["recordings"][0]["expected"] = None
        self.manifest["recordings"][0]["annotation_status"] = "unknown"
        with self.assertRaisesRegex(c.ContractError, "complete gold truth"):
            c.validate_manifest(self.manifest)

    def test_group_leakage_each_dimension(self):
        for key in ("source_family", "speaker", "session", "overlap_group"):
            manifest = deepcopy(self.manifest)
            manifest["recordings"][2][key] = manifest["recordings"][0][key]
            with self.subTest(key=key), self.assertRaisesRegex(c.ContractError, "group leakage"):
                c.validate_manifest(manifest)

    def test_overlap_rejected(self):
        row = deepcopy(self.manifest["recordings"][1])
        row["recording"] = "duplicate-negative-exposure"
        row["trace"]["sha256"] = "f" * 64
        self.manifest["recordings"].append(row)
        with self.assertRaisesRegex(c.ContractError, "overlapping qualified"):
            c.validate_manifest(self.manifest)

    def test_unknown_speaker_or_acoustic_state_rejected(self):
        self.manifest["recordings"][0]["speaker"] = None
        with self.assertRaisesRegex(c.ContractError, "speaker unknown"):
            c.validate_manifest(self.manifest)
        self.manifest = fixture_manifest()
        self.manifest["recordings"][0]["initial_state"]["acoustic"] = "unknown"
        with self.assertRaisesRegex(c.ContractError, "state unknown"):
            c.validate_manifest(self.manifest)

    def test_zero_negative_hours_not_zero_far(self):
        self.manifest["recordings"] = [row for row in self.manifest["recordings"] if row["expected"]]
        report = future_report(self.manifest, c.ARMS[0], .55)
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(report["summary"]["negative_recording_far_per_hour"])
        self.assertIsNone(report["summary"]["negative_recording_far_upper_95_per_hour"])
        with self.assertRaisesRegex(c.ContractError, "INSUFFICIENT_EVIDENCE"):
            compare.select_fixture_dev_thresholds([report], max_negative_far_upper95_per_hour=.4)

    def test_positive_false_accept_not_in_negative_only_far(self):
        rows = [row for row in self.manifest["recordings"] if row["split"] == "dev"]
        refs = c.score_events.validate_recordings(rows)
        detections = c.score_events.validate_detections([
            {"recording": "dev-positive", "keyword_id": 2, "time_s": 5, "confidence": .9}], refs)
        summary, _, _ = c.score_events.score(refs, detections, .15, .5)
        self.assertEqual(summary["false_accepts"], 1)
        self.assertEqual(summary["negative_recording_false_accepts"], 0)
        self.assertAlmostEqual(summary["negative_recording_far_upper_95_per_hour"], .299573227355399, places=12)

    def test_same_explicit_budget_and_frozen_test(self):
        result = compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=.4)
        self.assertEqual(result["selected"], {arm: .55 for arm in c.ARMS})
        test = [future_report(self.manifest, arm, .55, split="test") for arm in c.ARMS]
        self.assertEqual(compare.evaluate_fixture_frozen_test(result, test)["status"], "FIXTURE_FROZEN_TEST_REPORTED")
        test[0]["threshold"] = .65
        with self.assertRaisesRegex(c.ContractError, "development-frozen"):
            compare.evaluate_fixture_frozen_test(result, test)

    def test_test_selection_rejected(self):
        reports = self.candidates()
        reports[0]["split"] = "test"
        with self.assertRaisesRegex(c.ContractError, "test-set threshold selection"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)

    def test_missing_budget_rejected(self):
        with self.assertRaisesRegex(c.ContractError, "explicit caller FAR budget"):
            compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=None)

    def test_no_threshold_passes_budget(self):
        result = compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=.1)
        self.assertEqual(result["status"], "FIXTURE_NO_DEVELOPMENT_THRESHOLD_MEETS_BUDGET")
        self.assertIsNone(result["selected"])

    def test_same_threshold_grid_and_denominators_required(self):
        reports = self.candidates()
        reports.pop()
        with self.assertRaisesRegex(c.ContractError, "same unique"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)
        reports = self.candidates()
        reports[0]["denominator_sha256"] = "a" * 64
        with self.assertRaisesRegex(c.ContractError, "recomputed scoring inputs|inputs/denominators"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)

    def test_zero_positive_truth_not_zero_frr(self):
        self.manifest["recordings"] = [row for row in self.manifest["recordings"] if not row["expected"]]
        report = future_report(self.manifest, c.ARMS[0], .55, hit=False)
        self.assertEqual(report["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(report["summary"]["frr"])

    def test_changed_threshold_cannot_reuse_replay_contract(self):
        reports = self.candidates()
        reports[0]["threshold"] = .6
        with self.assertRaisesRegex(c.ContractError, "differs from its full replay"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)

    def test_frr_selection_only_among_budget_feasible_thresholds(self):
        reports = [future_report(self.manifest, arm, .55, negative_events=4) for arm in c.ARMS]
        reports += [future_report(self.manifest, arm, .65, hit=False) for arm in c.ARMS]
        selection = compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)
        self.assertEqual(selection["selected"], {arm: .65 for arm in c.ARMS})

    def test_exact_trace_content_cannot_leak_with_renamed_groups(self):
        self.manifest["recordings"][2]["trace"]["sha256"] = self.manifest["recordings"][0]["trace"]["sha256"]
        with self.assertRaisesRegex(c.ContractError, "duplicate qualified trace"):
            c.validate_manifest(self.manifest)

    def test_source_content_overlap_cannot_hide_behind_renamed_group(self):
        row = deepcopy(self.manifest["recordings"][1])
        row.update(recording="derived-same-source", overlap_group="new-group")
        row["trace"]["sha256"] = "f" * 64
        self.manifest["recordings"].append(row)
        with self.assertRaisesRegex(c.ContractError, "overlapping qualified"):
            c.validate_manifest(self.manifest)

    def test_truth_numbers_must_not_be_bools_or_strings(self):
        for key, value in (("start_s", True), ("end_s", "2.0"), ("keyword_id", 1.0)):
            manifest = deepcopy(self.manifest)
            manifest["recordings"][0]["expected"][0][key] = value
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                c.validate_manifest(manifest)

    def test_detection_numbers_must_not_be_bools_or_strings(self):
        report = future_report(self.manifest, c.ARMS[0], .55)
        for key, value in (("time_s", "2.1"), ("confidence", True), ("keyword_id", 1.0)):
            detections = deepcopy(report["scoring_inputs"]["detections"])
            detections[0][key] = value
            with self.subTest(key=key), self.assertRaises(c.ContractError):
                compare.score_qualified(self.manifest, detections, split="dev", arm=c.ARMS[0], threshold=.55,
                    replay={"contract": report["replay_contract"], "receipt_sha256": report["replay_receipt_sha256"]})

    def test_future_reports_recompute_metrics_and_identity(self):
        mutations = [lambda r: r["summary"].update(frr=.99),
                     lambda r: r["replay_contract"].update(model_sha256="b" * 64),
                     lambda r: r["replay_contract"].update(keyword_pack_sha256="b" * 64),
                     lambda r: r["replay_contract"].update(shared_controls_sha256="b" * 64),
                     lambda r: r["replay_contract"]["trace_sha256_by_recording"].update({"dev-positive": "b" * 64})]
        for mutate in mutations:
            reports = self.candidates()
            mutate(reports[0])
            with self.assertRaises(c.ContractError):
                compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)

    def test_future_decoder_identity_shared_between_arms(self):
        reports = self.candidates()
        reports[0]["replay_contract"]["decoder_executable_sha256"] = "b" * 64
        with self.assertRaisesRegex(c.ContractError, "paired decoder executable"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)

    def test_frozen_test_requires_complete_measurement_and_development_decoder(self):
        selection = compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=.4)
        reports = [future_report(self.manifest, arm, .55, split="test") for arm in c.ARMS]
        reports[0]["status"] = "INSUFFICIENT_EVIDENCE"
        with self.assertRaisesRegex(c.ContractError, "INSUFFICIENT_EVIDENCE"):
            compare.evaluate_fixture_frozen_test(selection, reports)
        reports[0] = future_report(self.manifest, c.ARMS[0], .55, split="test")
        reports[0]["replay_contract"]["decoder_executable_sha256"] = "b" * 64
        with self.assertRaisesRegex(c.ContractError, "differs from development freeze"):
            compare.evaluate_fixture_frozen_test(selection, reports)

    def test_frozen_selection_recomputed_from_development_evidence(self):
        selection = compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=.4)
        selection["selected"] = {arm: .65 for arm in c.ARMS}
        reports = [future_report(self.manifest, arm, .65, split="test") for arm in c.ARMS]
        with self.assertRaisesRegex(c.ContractError, "frozen development selection differs"):
            compare.evaluate_fixture_frozen_test(selection, reports)

    def test_replay_receipt_not_reused_across_distinct_contracts(self):
        reports = self.candidates()
        reports[1]["replay_receipt_sha256"] = reports[0]["replay_receipt_sha256"]
        with self.assertRaisesRegex(c.ContractError, "cannot reuse one replay receipt"):
            compare.select_fixture_dev_thresholds(reports, max_negative_far_upper95_per_hour=.4)
        selection = compare.select_fixture_dev_thresholds(self.candidates(), max_negative_far_upper95_per_hour=.4)
        test = [future_report(self.manifest, arm, .55, split="test") for arm in c.ARMS]
        test[0]["replay_receipt_sha256"] = selection["development_reports"][0]["replay_receipt_sha256"]
        with self.assertRaisesRegex(c.ContractError, "cannot reuse one replay receipt"):
            compare.evaluate_fixture_frozen_test(selection, test)

    def test_each_threshold_requires_own_full_replay(self):
        row = future_report(self.manifest, c.ARMS[0], .55)
        row["full_decoder_replay"] = False
        with self.assertRaisesRegex(c.ContractError, "full decoder replay"):
            compare.select_fixture_dev_thresholds([row], max_negative_far_upper95_per_hour=.4)


if __name__ == "__main__":
    unittest.main()
