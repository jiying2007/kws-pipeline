#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_training_diagnostics import build, file_entry, load_json, refinement_outcome  # noqa: E402


class OptionalJsonEncodingTests(unittest.TestCase):
    def invoke(self, root: pathlib.Path, payload: bytes | None, *,
               config_bytes: bytes | None = None, sentinel: bytes | None = None):
        work = root / "work"
        progress = work / "adversarial-refinement/progress.json"
        progress.parent.mkdir(parents=True)
        progress.write_text('{"current_phase": "calibration"}\n', encoding="utf-8")
        config = root / "config.json"
        config.write_bytes(config_bytes if config_bytes is not None else
                           b'{"qualification_holdout_seed":42,"retired_qualification_holdout_seeds":[]}\n')
        paths = [config, progress]
        if payload is not None:
            accessory = work / "robustness-summary.json"
            accessory.write_bytes(payload)
            paths.append(accessory)
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        output = root / "compact.json"
        if sentinel is not None:
            output.write_bytes(sentinel)
        done = subprocess.run(
            [sys.executable, "-S", "-B",
             *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
             str(ROOT / "tools/build_training_diagnostics.py"),
             "--config", str(config), "--work-dir", str(work), "--output", str(output)],
            capture_output=True, text=True, timeout=5, check=False,
        )
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, before)
        return done, output

    def check_unrelated_evidence(self, report: dict) -> None:
        self.assertEqual(report["adversarial_refinement_progress"], {"current_phase": "calibration"})
        self.assertEqual(report["adversarial_refinement_outcome"], {
            "classification": "no-summary", "last_active_phase": "calibration", "cause_verified": False,
        })
        self.assertIsNone(report["development"]["qualified"])
        self.assertIsNone(report["development"]["qualification_qualified"])
        self.assertFalse(report["formal_seed"]["consumed_by_this_run"])

    def test_invalid_optional_artifacts_keep_other_diagnostics(self) -> None:
        cases = ((b'{"text":', "invalid:JSONDecodeError"),
                 (b'\xff', "invalid:UnicodeDecodeError"),
                 (b'{"text":"\xe4\xb8', "invalid:UnicodeDecodeError"),
                 (b'[]\n', "not-object"), (b'', "missing"))
        for payload, error in cases:
            with self.subTest(payload=payload, error=error), tempfile.TemporaryDirectory() as td:
                done, output = self.invoke(pathlib.Path(td), payload)
                self.assertEqual(done.returncode, 0, done.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.check_unrelated_evidence(report)
                self.assertIsNone(report["robustness"])
                self.assertEqual(report["diagnostic_errors"], {} if error == "missing" else {"robustness": error})
                entry = next(row for row in report["files"] if row["path"] == "robustness-summary.json")
                self.assertIs(entry["present"], False)
                self.assertEqual(entry["error"], error)
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())

    def test_valid_non_ascii_and_absent_optional_artifacts(self) -> None:
        value = {"text": "中"}
        encoded = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        for payload in (encoded, None):
            with self.subTest(present=payload is not None), tempfile.TemporaryDirectory() as td:
                done, output = self.invoke(pathlib.Path(td), payload)
                self.assertEqual(done.returncode, 0, done.stderr)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.check_unrelated_evidence(report)
                self.assertEqual(report["robustness"], value if payload is not None else None)
                self.assertEqual(report["diagnostic_errors"], {})
                entry = next(row for row in report["files"] if row["path"] == "robustness-summary.json")
                self.assertIs(entry["present"], payload is not None)
                if payload is not None:
                    self.assertNotIn("error", entry)
                    self.assertEqual(entry["bytes"], len(payload))
                    self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
                else:
                    self.assertEqual(entry["error"], "missing")
                    self.assertNotIn("bytes", entry)
                    self.assertNotIn("sha256", entry)

    def test_reader_and_file_entry_preserve_invalid_bytes(self) -> None:
        for payload in (b'\xff', b'{"text":"\xe4\xb8'):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                path = root / "optional.json"
                path.write_bytes(payload)
                self.assertEqual(load_json(path), (None, "invalid:UnicodeDecodeError"))
                self.assertEqual(file_entry(root, "optional.json"), {
                    "path": "optional.json", "present": False, "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(), "error": "invalid:UnicodeDecodeError",
                })
                self.assertEqual(path.read_bytes(), payload)

    def test_invalid_required_config_remains_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sentinel = b"prior compact report\n"
            done, output = self.invoke(pathlib.Path(td), b'{}\n', config_bytes=b'\xff', sentinel=sentinel)
            self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
            self.assertIn("UnicodeDecodeError", done.stderr)
            self.assertEqual(done.stdout, "")
            self.assertEqual(output.read_bytes(), sentinel)


def main() -> int:
    strict = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(OptionalJsonEncodingTests)
    )
    if strict.testsRun != 4 or strict.skipped or not strict.wasSuccessful():
        return 1

    assert refinement_outcome(None, {"current_phase": "train-export"}) == {
        "classification": "no-summary",
        "last_active_phase": "train-export",
        "cause_verified": False,
    }
    assert refinement_outcome(
        {"qualified": False, "record": {"calibration_gate": False, "test_gate": True}},
        {"current_phase": None},
    )["classification"] == "development-strict-gate-failed"
    assert refinement_outcome(
        {"qualified": False, "record": {"calibration_gate": True, "test_gate": True}},
        None,
    )["classification"] == "qualification-gate-failed"
    assert refinement_outcome({"qualified": False}, None)["cause_verified"] is False
    with tempfile.TemporaryDirectory(prefix="training-diagnostics-test-") as tmp:
        root = pathlib.Path(tmp)
        work = root / "work"
        work.mkdir()
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "qualification_holdout_seed": 42,
                    "retired_qualification_holdout_seeds": [40, 41],
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "expected": 8,
            "matched": 2,
            "false_rejects": 6,
            "false_accepts": 3,
            "frr": 0.75,
            "far_per_hour": 12.5,
            "p95_post_end_latency_ms": 180.0,
            "per_keyword": {
                "1": {
                    "expected": 4,
                    "matched": 0,
                    "false_rejects": 4,
                    "false_accepts": 1,
                    "frr": 1.0,
                },
                "2": {
                    "expected": 4,
                    "matched": 2,
                    "false_rejects": 2,
                    "false_accepts": 2,
                    "frr": 0.5,
                },
            },
            "calibrated_thresholds": {"1": 0.6, "2": 0.55},
            "calibration_threshold_grid": [0.5, 0.55, 0.6],
            "calibration_coordinate_rounds": 2,
            "calibration_coordinate_rounds_executed": 2,
            "calibration_parallel_trials": 2,
            "calibration_operating_curve_path": "build/calibration-operating-curve.json",
            "calibration_operating_curve_sha256": "9" * 64,
            "calibration_operating_curve_summary": {
                "policy": "coordinate-threshold-trials-v1",
                "trial_count": 28,
                "coordinates_executed": 2,
                "grid_saturated": True,
            },
        }
        domains = {
            "worst_domain": "distance:far",
            "worst_domain_score": 9.5,
            "keyword_confusion": {
                "assignment": "global-monotonic-one-to-one-v1",
                "expected_events": 8,
                "correct_keyword": 2,
                "wrong_keyword": 1,
                "missed": 5,
                "matrix": {"1": {"2": 1}},
            },
        }
        manifest = {
            "development_qualified": False,
            "qualification_qualified": None,
            "candidate_selection": {
                "objective_fallback_used": True,
                "objective_best_round": 0,
                "objective_best_frontend": "logmel",
            },
            "records": [
                {
                    "round": 0,
                    "frontend": "logmel",
                    "score": 123.0,
                    "model_sha256": "a" * 64,
                    "calibration_gate": False,
                    "test_gate": False,
                    "calibration": metrics,
                    "test": metrics,
                    "calibration_domains": domains,
                    "test_domains": domains,
                }
            ],
        }
        (work / "domain-loop-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        progress = {
            "schema_version": 1,
            "policy": "refinement-phase-progress-v1",
            "current_phase": "calibration",
            "phase_seconds": {"train-export": 10.0},
        }
        (work / "adversarial-refinement").mkdir()
        (work / "adversarial-refinement/progress.json").write_text(
            json.dumps(progress), encoding="utf-8"
        )
        eligibility = {
            "schema_version": 1,
            "policy": "selected-refinement-source-signal-v1",
            "eligible": True,
            "source_round": 0,
            "collapsed_keyword_ids": [],
        }
        (work / "base-refinement-eligibility.json").write_text(
            json.dumps(eligibility), encoding="utf-8"
        )
        acoustic = {
            "schema_version": 1,
            "evidence_class": "kws-acoustic-alignment-diagnostic-v3",
            "development_only": True,
            "source_round": 0,
            "source_frontend": "logmel",
            "source_selection_policy": "development-recall-first-refinement-source-v1",
            "source_was_strict": False,
            "model_sha256": "b" * 64,
            "tokens_sha256": "c" * 64,
            "keywords_sha256": "d" * 64,
            "domain_index_sha256": "e" * 64,
            "feature_dump_sha256": "f" * 64,
            "runner_sha256": "3" * 64,
            "keyword_pack_sha256": "4" * 64,
            "parameter_contract_sha256": "1" * 64,
            "root_start_logit_margin": 0.5,
            "event_match_pre_tolerance_ms": 150.0,
            "event_match_post_tolerance_ms": 500.0,
            "max_recordings_per_keyword_split": 8,
            "model": {"feature_dim": 32, "hidden_dim": 64, "vocab_size": 5},
            "aggregates": {
                "calibration": {
                    "1": {
                        "recordings": 8,
                        "greedy_subsequence_recordings": 3,
                        "surrogate_above_threshold_recordings": 3,
                        "surrogate_above_threshold_runtime_miss_recordings": 2,
                        "greedy_subsequence_runtime_miss_recordings": 2,
                        "greedy_strict_dominance_path_recordings": 1,
                        "greedy_strict_dominance_path_runtime_match_recordings": 1,
                        "greedy_strict_dominance_path_runtime_miss_recordings": 0,
                        "surrogate_above_threshold_greedy_path_incompatible_recordings": 2,
                        "runtime_decoder_hit_recordings": 2,
                        "runtime_decoder_hit_without_expected_match_recordings": 1,
                        "surrogate_above_threshold_without_decoder_hit_recordings": 1,
                        "runtime_pending_prefix_recordings": 1,
                        "runtime_refractory_suppressed_recordings": 0,
                        "runtime_zero_speech_recordings": 0,
                        "decoder_root_admissible_recordings": 1,
                    }
                },
                "test": {
                    "1": {
                        "recordings": 8,
                        "greedy_subsequence_recordings": 2,
                        "surrogate_above_threshold_recordings": 2,
                        "surrogate_above_threshold_runtime_miss_recordings": 1,
                        "greedy_subsequence_runtime_miss_recordings": 1,
                        "greedy_strict_dominance_path_recordings": 1,
                        "greedy_strict_dominance_path_runtime_match_recordings": 0,
                        "greedy_strict_dominance_path_runtime_miss_recordings": 1,
                        "surrogate_above_threshold_greedy_path_incompatible_recordings": 1,
                        "runtime_decoder_hit_recordings": 1,
                        "runtime_decoder_hit_without_expected_match_recordings": 1,
                        "surrogate_above_threshold_without_decoder_hit_recordings": 0,
                        "runtime_pending_prefix_recordings": 0,
                        "runtime_refractory_suppressed_recordings": 0,
                        "runtime_zero_speech_recordings": 1,
                        "decoder_root_admissible_recordings": 2,
                    }
                },
            },
            "records": [{"wav_sha256": "2" * 64}],
        }
        (work / "acoustic-alignment.json").write_text(
            json.dumps(acoustic), encoding="utf-8"
        )

        result = build(config, work)
        assert result["adversarial_refinement_progress"] == progress
        assert result["adversarial_refinement_outcome"] == {
            "classification": "no-summary",
            "last_active_phase": "calibration",
            "cause_verified": False,
        }
        assert result["development"]["refinement_eligibility"]["eligible"] is True
        compact_acoustic = result["development"]["acoustic_alignment"]
        assert compact_acoustic["source_round"] == 0
        assert compact_acoustic["aggregates"]["calibration"]["1"]["recordings"] == 8
        assert compact_acoustic["runner_sha256"] == "3" * 64
        assert compact_acoustic["keyword_pack_sha256"] == "4" * 64
        assert compact_acoustic["event_match_pre_tolerance_ms"] == 150.0
        assert compact_acoustic["event_match_post_tolerance_ms"] == 500.0
        assert "records" not in compact_acoustic
        row = result["development"]["rounds"][0]
        operating = row["calibration_operating_point"]
        assert operating["grid_saturated"] is True
        assert operating["edge_keywords"] == {"1": "max"}
        assert operating["selected"] == {"1": 0.6, "2": 0.55}
        assert operating["coordinate_rounds_executed"] == 2
        assert operating["operating_curve_summary"]["trial_count"] == 28
        assert operating["operating_curve_summary"]["grid_saturated"] is True
        assert operating["operating_curve_sha256"] == "9" * 64
        confusion = row["test_domain_summary"]["keyword_confusion"]
        assert confusion["wrong_keyword"] == 1
        assert confusion["missed"] == 5
        assert confusion["matrix"] == {"1": {"2": 1}}
        file_rows = {item["path"]: item for item in result["files"]}
        assert file_rows["adversarial-refinement/progress.json"]["present"] is True
        assert file_rows["base-refinement-eligibility.json"]["present"] is True
        assert file_rows["acoustic-alignment.json"]["present"] is True
        signals = result["evidence_signals"]
        assert signals["cross_split_keyword_collapse"]["observed"] is False
        assert signals["threshold_grid_saturation"]["observed"] is True
        assert signals["threshold_grid_saturation"]["occurrences"] == [
            {
                "round": 0,
                "edge_keywords": {"1": "max"},
                "grid_min": 0.5,
                "grid_max": 0.6,
            }
        ]
        assert signals["keyword_confusion"]["observed"] is True
        assert len(signals["keyword_confusion"]["occurrences"]) == 2
        assert (
            signals["acoustic_sequence_observed_but_runtime_missed"]["observed"]
            is True
        )
        acoustic_gaps = signals[
            "acoustic_sequence_observed_but_runtime_missed"
        ]["occurrences"]
        assert {(row["split"], row["keyword_id"]) for row in acoustic_gaps} == {
            ("calibration", "1"),
            ("test", "1"),
        }
        assert {row["same_sample_runtime_misses"] for row in acoustic_gaps} == {1, 2}
        assert all(
            row["evidence"] == "same-sample-exact-runtime-v1"
            for row in acoustic_gaps
        )
        surrogate_gaps = signals[
            "surrogate_above_threshold_but_runtime_missed"
        ]
        assert surrogate_gaps["observed"] is True
        assert {
            (row["split"], row["same_sample_runtime_misses"])
            for row in surrogate_gaps["occurrences"]
        } == {("calibration", 2), ("test", 1)}
        greedy_path_gaps = signals[
            "surrogate_above_threshold_but_greedy_path_incompatible"
        ]
        assert greedy_path_gaps["observed"] is True
        assert {
            (row["split"], row["same_sample_greedy_path_incompatible"])
            for row in greedy_path_gaps["occurrences"]
        } == {("calibration", 2), ("test", 1)}
        assert all(
            row["evidence"] == "same-sample-greedy-dominance-v1"
            for row in greedy_path_gaps["occurrences"]
        )

        path_gaps = signals["surrogate_above_threshold_without_decoder_hit"]
        assert path_gaps["observed"] is True
        assert path_gaps["occurrences"] == [
            {
                "round": 0,
                "split": "calibration",
                "keyword_id": "1",
                "sampled_recordings": 8,
                "same_sample_no_decoder_hit": 1,
                "evidence": "same-sample-engine-stats-v1",
            }
        ]
        event_gaps = signals["decoder_hit_without_expected_event_match"]
        assert event_gaps["observed"] is True
        assert {
            (row["split"], row["same_sample_decoder_hit_event_miss"])
            for row in event_gaps["occurrences"]
        } == {("calibration", 1), ("test", 1)}
        zero_speech = signals["runtime_zero_speech"]
        assert zero_speech["observed"] is True
        assert zero_speech["occurrences"][0]["split"] == "test"
        pending = signals["runtime_pending_prefix"]
        assert pending["observed"] is True
        assert pending["occurrences"][0]["split"] == "calibration"
        assert signals["sampled_acoustic_sequence_absent"]["observed"] is False
        assert result["diagnostic_errors"] == {}

    print("compact training diagnostics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
