"""Exact-statistics and system-mode regressions; no model is executed.

The numeric oracle uses high-precision Decimal arithmetic and Newton's method,
independent of the production log-domain CDF and bisection implementation.
System fixtures run the real Python scoring/corpus pipeline on invented PCM and
JSON labels, with only the native runner replaced by a mock.
"""
from __future__ import annotations

import contextlib
import copy
from decimal import Decimal, localcontext
import io
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
import wave
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "eval")]
import compare_dataset_iterations as comparison
import dataset_iteration_policy as policy
import dataset_iteration_inputs as evidence_inputs
import domain_metrics
import run_corpus
import run_dataset_iteration as iteration
import score_events


def decimal_poisson_upper(count: int, hours: str) -> float:
    with localcontext() as context:
        context.prec = 70
        alpha = Decimal("0.05")
        if count == 0:
            return float(-alpha.ln() / Decimal(hours))
        mean = Decimal(count + 1)
        for _ in range(100):
            term = (-mean).exp()
            cdf = term
            for index in range(1, count + 1):
                term *= mean / Decimal(index)
                cdf += term
            # d P(X <= count) / d mean = -P(X == count).
            step = (cdf - alpha) / term
            mean += step
            if abs(step) < Decimal("1e-55"):
                return float(mean / Decimal(hours))
        raise AssertionError("independent Decimal oracle failed to converge")


def exact_card(*, count=0, hours=8.0, misses=20, model="a", code="c"):
    _, rate = iteration.negative_far_rate({
        "negative_recording_false_accepts": count,
        "negative_recording_audio_hours": hours,
        "negative_recording_far_per_hour": count / hours,
    })
    return {
        "schema_version": 5, "protocol": policy.PROTOCOL, "run_id": model,
        "identity": {
            "protocol": policy.PROTOCOL, "far_policy": policy.FAR_POLICY,
            "dataset_identity_policy": policy.DATASET_IDENTITY_POLICY,
            "dataset_id": "d" * 64, "model_id": model * 64, "code_sha": code * 40,
            "seed": "1", "variable": "model", "comparison_mode": "ablation",
        },
        "exposure": {"negative_recordings": 1, "exposure_hours": hours},
        "metrics": {"expected": 100, "matched": 100 - misses, "false_rejects": misses,
                    "false_accepts": count, "frr": misses / 100,
                    "far_per_hour": count / hours, "p95_post_end_latency_ms": 100.0},
        "far_rate": rate,
    }


class ExactFarTests(unittest.TestCase):
    def test_independent_decimal_oracle(self):
        for count in (0, 1, 2, 5, 10, 50, 100):
            for hours in ("0.25", "8", "24"):
                with self.subTest(count=count, hours=hours):
                    actual = iteration.far_upper_bound_95(count, float(hours))
                    expected = decimal_poisson_upper(count, hours)
                    self.assertTrue(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-12),
                                    (count, hours, actual, expected))

    def test_zero_to_one_counterexample_and_monotonicity(self):
        self.assertLess(1 + 1.959964, 2.9957)  # The retained historical defect.
        bounds = [exact_card(count=count)["far_rate"]["upper_bound_95_per_hour"]
                  for count in range(21)]
        self.assertTrue(all(right > left for left, right in zip(bounds, bounds[1:])))
        for count in (0, 1, 10):
            self.assertEqual(iteration.far_upper_bound_95(count, 16),
                             iteration.far_upper_bound_95(count, 8) / 2)

    def test_recall_gain_cannot_hide_zero_to_one_false_accept(self):
        before, after = exact_card(), exact_card(count=1, misses=10, model="b")
        result = comparison.compare(before, after, .5, 0, 800)
        self.assertEqual(result["verdict"], "regressed")
        self.assertGreater(result["deltas"]["far_upper_bound_95"], 0)
        self.assertEqual(comparison.compare(before, after, 1, 0, 800)["verdict"], "neutral")
        improved = comparison.compare(exact_card(count=1), exact_card(misses=10, model="b"), 1, 0, 800)
        self.assertEqual(improved["verdict"], "improved")

    def test_legacy_policy_is_not_silently_upgraded_or_mixed(self):
        exact = exact_card()
        old = copy.deepcopy(exact)
        old["schema_version"] = 4
        old["identity"].pop("comparison_mode")
        old["identity"]["far_policy"] = policy.APPROX_FAR_POLICY
        old["far_rate"]["upper_bound_95_per_hour"] = 2.9957 / 8
        self.assertEqual(comparison.far_policy(old, "legacy"), policy.APPROX_FAR_POLICY)
        for left, right in ((old, exact), (exact, old)):
            with self.assertRaisesRegex(ValueError, "FAR policy mismatch"):
                comparison.framing(left, right)
        changed = copy.deepcopy(old)
        changed["identity"]["model_id"] = "b" * 64
        result = comparison.compare(old, changed, 1, 0, 800)
        self.assertTrue(result["historical_statistics"])
        self.assertEqual(old["far_rate"]["upper_bound_95_per_hour"], 2.9957 / 8)

    def test_exact_policy_confidence_bounds_and_types_are_required(self):
        before = exact_card()
        changes = [
            ("identity", "far_policy", policy.APPROX_FAR_POLICY),
            ("far_rate", "confidence_level", .9),
            ("far_rate", "confidence_level", True),
            ("far_rate", "upper_bound_95_per_hour", 2.9957 / 8),
            ("far_rate", "upper_bound_95_per_hour", float("nan")),
            ("far_rate", "observed_false_accepts", 0.0),
            ("far_rate", "observed_false_accepts", True),
            ("exposure", "exposure_hours", 0),
        ]
        for section, field, value in changes:
            with self.subTest(section=section, field=field, value=value):
                after = exact_card(model="b")
                after[section][field] = value
                with self.assertRaises(ValueError):
                    comparison.compare(before, after, 1, 0, 800)
        for count in (True, -1, 1.0, "1", policy.MAX_FA_EVENTS + 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                iteration.far_upper_bound_95(count, 1)

    def test_nonfinite_budgets_cannot_widen_selection(self):
        for invalid in (float("nan"), float("inf"), True):
            for index in range(3):
                args = [1, 0, 800]
                args[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    comparison.compare(exact_card(), exact_card(model="b"), *args)

    def test_invalid_metrics_and_oversized_counts_fail_before_arithmetic(self):
        for field, values in (("frr", (True, False, -0.1, 1.1)),
                              ("far_per_hour", (True, -1)),
                              ("p95_post_end_latency_ms", (False, -1))):
            for value in values:
                after = exact_card(model="b")
                after["metrics"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    comparison.compare(exact_card(), after, 1, 0, 800)
        after = exact_card(model="b")
        after["far_rate"]["observed_false_accepts"] = 10 ** 400
        with self.assertRaisesRegex(ValueError, "exceed"):
            comparison.compare(exact_card(), after, 1, 0, 800)
        for exposure in (True, None, float("nan"), float("inf"), 0, -1, 10 ** 400):
            with self.subTest(exposure=exposure), self.assertRaises(ValueError):
                iteration.far_upper_bound_95(0, exposure)


class SystemComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename in ("positive.wav", "negative.wav"):
            with wave.open(str(self.root / filename), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000)
        self.references = self.root / "references.jsonl"
        refs = [{"recording": "p" + str(keyword), "path": "positive.wav", "duration_s": 1.0,
                 "expected": [{"keyword_id": keyword, "start_s": .2, "end_s": .4}]}
                for keyword in (1, 2)]
        refs.append({"recording": "n", "path": "negative.wav", "duration_s": 1.0, "expected": []})
        self.references.write_text("".join(json.dumps(row) + "\n" for row in refs))
        self.context = {
            "sample_rate_hz": 16000, "feed_chunk_samples": 320, "reset": "per-recording",
            "appended_context_samples": 0, "eof": "no-extra-audio",
            "event_time": "input-wav-sample-seconds",
            "keyword_map": {"1": "你好小窝", "2": "小窝小窝"},
        }
        self.context_path = self.root / "context.json"
        self.context_path.write_text(json.dumps(self.context))
        self.runner_calls = []

    def produce(self, name, *, mode="system", code="a", frontend="native-a", miss_second=False,
                context=None):
        folder = self.root / name
        folder.mkdir()
        for filename in ("runner", "model", "keywords"):
            (folder / filename).write_text("INERT " + filename + " " + name)
        context_path = self.context_path
        if context is not None:
            context_path = folder / "context.json"
            context_path.write_text(json.dumps(context))
        argv = ["run_dataset_iteration", "--runner", str(folder / "runner"),
                "--model", str(folder / "model"), "--keywords", str(folder / "keywords"),
                "--references", str(self.references), "--audio-root", str(self.root),
                "--work-dir", str(folder / "work"), "--output", str(folder / "card.json"),
                "--code-sha", code * 40, "--variable", "system" if mode == "system" else "model",
                "--mode", mode]
        if mode == "system":
            argv += ["--system-context", str(context_path), "--native-frontend", frontend]

        def native_runner(command, **kwargs):
            self.assertEqual(command[0], str(folder / "runner"))
            self.runner_calls.append(command)
            recording = command[4]
            detected = recording == "p1" or (recording == "p2" and not miss_second)
            output = (json.dumps({"recording": recording, "keyword_id": int(recording[1:]),
                                  "time_s": .5, "confidence": .9}) + "\n") if detected else ""
            return subprocess.CompletedProcess(command, 0, stdout=output)

        def pipeline(command):
            module = {"run_corpus.py": run_corpus, "score_events.py": score_events,
                      "domain_metrics.py": domain_metrics}[Path(command[1]).name]
            with mock.patch.object(sys, "argv", command[1:]):
                if module is run_corpus:
                    with mock.patch.object(run_corpus.subprocess, "run", native_runner):
                        self.assertEqual(module.main(), 0)
                else:
                    self.assertEqual(module.main(), 0)

        with mock.patch.object(sys, "argv", argv), mock.patch.object(iteration, "run", pipeline), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(iteration.main(), 0)
        return json.loads((folder / "card.json").read_text())

    def compare(self, before, after):
        return comparison.compare(before, after, 1e9, 0, 800, mode="system")

    def test_real_scoring_pipeline_allows_different_systems(self):
        before = self.produce("a", miss_second=True)
        after = self.produce("b", code="b", frontend="native-80-bin-transducer")
        result = self.compare(before, after)
        self.assertEqual(len(self.runner_calls), 6)
        self.assertEqual(result["verdict"], "improved")
        self.assertEqual(result["comparison_mode"], "system")
        self.assertIn("code_sha", result["moved"])
        self.assertIn("native_frontend", result["moved"])
        self.assertEqual(result["verified_evaluation"]["expected"], 2)
        self.assertEqual(result["comparison_scope"], "system-quality-without-single-change-attribution")
        self.assertFalse(result["historical_statistics"])

    def test_modes_are_explicit_and_never_auto_downgraded(self):
        before, after = self.produce("a"), self.produce("b", code="b")
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            comparison.compare(before, after, 1e9, 0, 800)
        for card in (before, after):
            card["identity"].update(comparison_mode="ablation", variable="model")
        with self.assertRaisesRegex(ValueError, "code_sha differs"):
            comparison.compare(before, after, 1e9, 0, 800)
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            self.compare(before, after)
        after["identity"]["code_sha"] = before["identity"]["code_sha"]
        after["identity"]["dataset_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "both dataset and model changed"):
            comparison.compare(before, after, 1e9, 0, 800)

    def test_matching_caller_ids_without_input_evidence_are_rejected(self):
        before, after = exact_card(), exact_card(model="b", code="b")
        for card in (before, after):
            card["identity"].update(comparison_mode="system", variable="system", native_frontend="named")
        with self.assertRaisesRegex(ValueError, "input evidence is missing"):
            self.compare(before, after)

    def test_different_shared_runner_context_or_keyword_meaning_is_rejected(self):
        before = self.produce("a")
        for index, changed in enumerate((dict(self.context, feed_chunk_samples=160),
                                        dict(self.context, keyword_map={"1": "other", "2": "小窝小窝"}))):
            after = self.produce("context" + str(index), context=changed)
            with self.assertRaisesRegex(ValueError, "identical verified inputs and evaluation protocol"):
                self.compare(before, after)

    def test_changed_audio_or_missing_input_cannot_reuse_dataset_id(self):
        before, after = self.produce("a"), self.produce("b")
        audio = self.root / "negative.wav"
        original = audio.read_bytes()
        audio.write_bytes(original[:-2] + b"\x01\x00")
        with self.assertRaisesRegex(ValueError, "measured input identity differs"):
            self.compare(before, after)
        audio.unlink()
        with self.assertRaisesRegex(ValueError, "audio file is missing"):
            self.compare(before, after)

    def test_changed_reference_events_provenance_and_scores_are_rejected(self):
        before, after = self.produce("a"), self.produce("b")
        for name in ("references", "source_references", "detections", "provenance"):
            path = Path(after["system_evaluation"]["files"][name]["path"])
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "bytes changed"):
                    self.compare(before, after)
            finally:
                path.write_bytes(original)
        forged = copy.deepcopy(after)
        forged["metrics"]["frr"] = .123
        with self.assertRaisesRegex(ValueError, "score differs from saved events"):
            self.compare(before, forged)
        forged = copy.deepcopy(after)
        forged["identity"]["model_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "model tuple differs"):
            self.compare(before, forged)

    def test_cli_system_mode_and_failure_preserve_existing_output(self):
        before = self.produce("a", miss_second=True)
        after = self.produce("b", code="b")
        paths = []
        for name, card in (("before", before), ("after", after)):
            path = self.root / (name + ".json")
            path.write_text(json.dumps(card))
            paths.append(path)
        output = self.root / "comparison.json"
        argv = ["compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]),
                "--mode", "system", "--far-budget", "1e9", "--output", str(output)]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(comparison.main(), 0)
        self.assertEqual(json.loads(output.read_text())["comparison_mode"], "system")
        sentinel = b"prior comparison retained\n"
        output.write_bytes(sentinel)
        Path(after["system_evaluation"]["files"]["provenance"]["path"]).unlink()
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "unavailable"):
            comparison.main()
        self.assertEqual(output.read_bytes(), sentinel)

    def test_producer_system_options_fail_before_launch(self):
        for extra in (("--variable", "system"),
                      ("--variable", "system", "--mode", "system"),
                      ("--variable", "model", "--native-frontend", "declared")):
            argv = ["run", "--runner", "missing", "--model", "missing", "--keywords", "missing",
                    "--references", "missing", "--work-dir", str(self.root / "never"),
                    "--output", str(self.root / "never.json"), *extra]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(iteration, "run") as run:
                with self.assertRaisesRegex(ValueError, "system"):
                    iteration.main()
                run.assert_not_called()
            self.assertFalse((self.root / "never").exists())

    def test_reference_exposure_is_rechecked_against_actual_pcm(self):
        card = self.produce("duration")
        files = card["system_evaluation"]["files"]
        # Re-hash every changed declaration as a faulty receipt producer could.
        # The actual WAV and recorded PCM remain one second long.
        for name in ("references", "source_references"):
            path = Path(files[name]["path"])
            rows = evidence_inputs.load_jsonl(path)
            next(row for row in rows if row["recording"] == "n")["duration_s"] = 60.0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            files[name]["sha256"] = iteration.sha256_file(path)
        path = Path(files["provenance"]["path"])
        provenance = json.loads(path.read_text())
        provenance["references_sha256"] = files["references"]["sha256"]
        path.write_text(json.dumps(provenance))
        files["provenance"]["sha256"] = iteration.sha256_file(path)
        card["artifacts"].update(references_sha256=files["source_references"]["sha256"],
                                  provenance_sha256=files["provenance"]["sha256"])
        card["exposure"]["exposure_hours"], card["far_rate"] = iteration.negative_far_rate({
            "negative_recording_false_accepts": 0, "negative_recording_audio_hours": 60 / 3600,
            "negative_recording_far_per_hour": 0.0,
        })
        with self.assertRaisesRegex(ValueError, "duration_s does not match WAV duration"):
            iteration.verify_system_evidence(card, "duration")

    def test_input_bundles_restore_absolute_audio_and_afe_without_changing_cards(self):
        metadata = self.root / "afe.tsv"
        metadata.write_text("kws-afe-timeline-v1\n0 16000 0 0 0 0 raw 0 config\n")
        rows = evidence_inputs.load_jsonl(self.references)
        for row in rows:
            row["path"] = str(self.root / row["path"])
            row["metadata_path"] = str(metadata)
            row["metadata_sha256"] = iteration.sha256_file(metadata)
        self.references.write_text("".join(json.dumps(row) + "\n" for row in rows))
        before, after = self.produce("a", miss_second=True), self.produce("b", code="b")
        restored = tempfile.TemporaryDirectory()
        self.addCleanup(restored.cleanup)
        paths = []
        originals = []
        for name in ("a", "b"):
            source = self.root / name
            destination = Path(restored.name) / name
            destination.mkdir()
            originals.append((source / "card.json").read_bytes())
            shutil.copyfile(source / "card.json", destination / "card.json")
            shutil.copytree(source / "evidence", destination / "evidence")
            paths.append(destination / "card.json")
            self.assertFalse(any((destination / "evidence").rglob("model")))
            shutil.rmtree(source)
        for source in (metadata, self.references, self.root / "positive.wav", self.root / "negative.wav"):
            source.unlink()
        argv = ["compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]),
                "--mode", "system", "--far-budget", "1e9"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(comparison.main(), 0)
        self.assertEqual(json.loads(out.getvalue())["verdict"], "improved")
        roots = [evidence_inputs.adjacent_evidence_root(path, card)
                 for path, card in zip(paths, (before, after))]
        result = comparison.compare(before, after, 1e9, 0, 800, mode="system",
                                    baseline_evidence_root=roots[0], candidate_evidence_root=roots[1])
        self.assertEqual(result["verdict"], "improved")
        self.assertEqual([path.read_bytes() for path in paths], originals)
        audio = next(roots[1].rglob("negative.wav"))
        audio.write_bytes(audio.read_bytes()[:-2] + b"\1\0")
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "measured input identity differs"):
            comparison.main()

    def test_comparison_outputs_cannot_overwrite_cards_or_saved_evidence(self):
        before, after = self.produce("a"), self.produce("b")
        paths = [self.root / name / "card.json" for name in ("a", "b")]
        link = self.root / "card-link.json"
        link.symlink_to(paths[0])
        hardlink = self.root / "card-hardlink.json"
        os.link(paths[1], hardlink)
        saved_root = evidence_inputs.adjacent_evidence_root(paths[1], after)
        saved_audio = next(saved_root.rglob("negative.wav"))
        for output in (paths[0], paths[1], link, hardlink, saved_audio):
            sentinel = output.read_bytes()
            argv = ["compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]),
                    "--mode", "system", "--far-budget", "1e9", "--output", str(output)]
            with self.subTest(output=output), mock.patch.object(sys, "argv", argv), \
                    self.assertRaisesRegex(ValueError, "aliases an input"):
                comparison.main()
            self.assertEqual(output.read_bytes(), sentinel)

    def test_producer_aliases_fail_before_outputs_or_runner_calls(self):
        for name in ("runner", "model", "keywords"):
            (self.root / name).write_text("inert")
        symlink = self.root / "audio-link"
        symlink.symlink_to(self.root / "negative.wav")
        hardlink = self.root / "model-link"
        os.link(self.root / "model", hardlink)
        work = self.root / "not-created"
        for output in (self.references, symlink, hardlink, work / "summary.json"):
            sentinel = output.read_bytes() if output.exists() else None
            argv = ["run", "--runner", str(self.root / "runner"), "--model", str(self.root / "model"),
                    "--keywords", str(self.root / "keywords"), "--references", str(self.references),
                    "--audio-root", str(self.root), "--work-dir", str(work), "--output", str(output),
                    "--variable", "model"]
            with self.subTest(output=output), mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(iteration, "run") as run, self.assertRaisesRegex(ValueError, "aliases"):
                iteration.main()
            run.assert_not_called()
            self.assertFalse(work.exists())
            if sentinel is not None:
                self.assertEqual(output.read_bytes(), sentinel)

    def test_existing_bundle_is_reverified_and_not_overwritten(self):
        card = self.produce("a")
        bundle = evidence_inputs.adjacent_evidence_root(self.root / "a" / "card.json", card)
        evidence_inputs.bundle_system_evidence(card, bundle)
        saved_audio = next(bundle.rglob("negative.wav"))
        sentinel = saved_audio.read_bytes()[:-2] + b"\1\0"
        saved_audio.write_bytes(sentinel)
        with self.assertRaisesRegex(ValueError, "measured input identity differs"):
            evidence_inputs.bundle_system_evidence(card, bundle)
        self.assertEqual(saved_audio.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
