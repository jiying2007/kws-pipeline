from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import math
import itertools
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools" / "compare_dataset_iterations.py"
RUN = ROOT / "tools" / "run_dataset_iteration.py"
sys.path.insert(0, str(ROOT / "tools"))
from run_dataset_iteration import dataset_identity  # noqa: E402
import run_dataset_iteration as iteration  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
import score_events  # noqa: E402

PROTOCOL = "dataset-iteration-v1"


def scorecard(*, run_id="r1", protocol=PROTOCOL, dataset="d1", model="m1",
              code="c1", seed="s1", variable="model", frr=0.1, far=0.0,
              latency=100.0, exposure_hours=0.25, bound=11.98, domains=None):
    return {
        "schema_version": 1,
        "protocol": protocol,
        "run_id": run_id,
        "identity": {
            "protocol": protocol,
            "dataset_id": dataset,
            "model_id": model,
            "code_sha": code,
            "seed": seed,
            "variable": variable,
        },
        "exposure": {"negative_recordings": 3, "exposure_hours": exposure_hours},
        "metrics": {
            "expected": 100, "matched": int(round(100 * (1 - frr))),
            "false_rejects": int(round(100 * frr)), "false_accepts": 0,
            "frr": frr, "far_per_hour": far,
            "p95_post_end_latency_ms": latency,
        },
        "far_rate": {
            "point_estimate_per_hour": far,
            "upper_bound_95_per_hour": bound,
            "basis": "zero observed: chi-square(2) 95% / (2T)",
        },
        "domains": domains or {},
        "artifacts": {"detections_sha256": "0" * 64, "references_sha256": "0" * 64},
    }


def write(root: pathlib.Path, name: str, payload: dict) -> pathlib.Path:
    path = root / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def run(root: pathlib.Path, baseline: dict, candidate: dict, *extra: str) -> subprocess.CompletedProcess:
    before = write(root, "before.json", baseline)
    after = write(root, "after.json", candidate)
    # 0.25 h of negative exposure implies an upper bound near 12/h, so the
    # default budget has to be realistic for the fixture. A later --far-budget
    # in `extra` overrides this one, which is how the exposure test tightens it.
    return subprocess.run(
        [sys.executable, "-S", "-B",
         *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
         str(COMPARE), "--baseline", str(before),
         "--candidate", str(after), "--far-budget", "20.0", *extra],
        check=False, capture_output=True, text=True,
    )


class RequiredMetricTests(unittest.TestCase):
    FIELDS = ("frr", "far_per_hour", "p95_post_end_latency_ms")

    def compare(self, root: pathlib.Path, baseline: dict, candidate: dict,
                *extra: str) -> subprocess.CompletedProcess:
        expected = {
            name: hashlib.sha256((json.dumps(payload, ensure_ascii=False, indent=2,
                                             sort_keys=True) + "\n").encode("utf-8")).hexdigest()
            for name, payload in (("before.json", baseline), ("after.json", candidate))
        }
        done = run(root, baseline, candidate, *extra)
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        return done

    def test_missing_required_metrics_on_either_scorecard(self) -> None:
        for side in ("baseline", "candidate"):
            for field in self.FIELDS:
                with self.subTest(side=side, field=field), tempfile.TemporaryDirectory() as td:
                    root = pathlib.Path(td)
                    before = scorecard(run_id="base", frr=0.20)
                    after = scorecard(run_id="cand", model="m2", frr=0.10)
                    del (before if side == "baseline" else after)["metrics"][field]
                    output = root / "comparison.json"
                    done = self.compare(root, before, after, "--output", str(output))
                    self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                    self.assertIn(f"metrics is missing required '{field}'", done.stderr)
                    self.assertEqual(done.stdout, "")
                    self.assertFalse(output.exists())

    def test_explicit_zero_metrics_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            before = scorecard(run_id="base", frr=0.0, far=0.0, latency=0.0)
            after = scorecard(run_id="cand", model="m2", frr=0.0, far=0.0, latency=0.0)
            done = self.compare(root, before, after)
            self.assertEqual(done.returncode, 0, done.stderr)
            report = json.loads(done.stdout)
            self.assertEqual(report["verdict"], "neutral")
            for field in self.FIELDS:
                self.assertEqual(report["deltas"][field], 0.0)

    def test_complete_scorecards_keep_all_three_verdicts(self) -> None:
        for frr, exit_code, verdict in ((0.10, 0, "improved"),
                                       (0.20, 0, "neutral"),
                                       (0.30, 1, "regressed")):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as td:
                root = pathlib.Path(td)
                output = root / "comparison.json"
                done = self.compare(root, scorecard(run_id="base", frr=0.20),
                                    scorecard(run_id="cand", model="m2", frr=frr),
                                    "--output", str(output))
                self.assertEqual(done.returncode, exit_code, done.stdout + done.stderr)
                report = json.loads(done.stdout)
                self.assertEqual(report["verdict"], verdict)
                self.assertEqual(json.loads(output.read_text()), report)
                self.assertEqual(report["deltas"]["frr"], frr - 0.20)
                self.assertEqual(report["budgets"], {
                    "far_upper_bound_95_per_hour": 20.0,
                    "frr_tolerance": 0.0,
                    "p95_latency_ms": 800.0,
                })

    def test_missing_metric_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            output = root / "comparison.json"
            sentinel = b"existing report sentinel\n"
            output.write_bytes(sentinel)
            after = scorecard(run_id="cand", model="m2", frr=0.30)
            del after["metrics"]["frr"]
            done = self.compare(root, scorecard(run_id="base", frr=0.20), after,
                                "--output", str(output))
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertEqual(done.stdout, "")
            self.assertIn("metrics is missing required 'frr'", done.stderr)
            self.assertEqual(output.read_bytes(), sentinel)


class DatasetIdentityAliasTests(unittest.TestCase):
    """Real metadata/WAV helper checks, not execution of the model/eval CLI."""

    FIELDS = ("audio", "audio_path", "path")
    # Preserve the pre-domain-policy golden as historical evidence, not a current ID.
    LEGACY_PATH_ONLY_ID = "47be6a9d292ea745056a302e773906a725c5b2eb366535c33fb07df35d2534d1"

    # Preserve schema-3 domain-only identity; schema 4 also binds AFE absence.
    DOMAIN_ONLY_PATH_ID = "9a440389f273044a474f2873a3e77ddd8c0398ff468b6cd30be2b72a48d6c030"
    PATH_ONLY_ID = "ca9680a375cbd57fa87956e9d38762b20efbedcd1ec59881762d23b9ab74daee"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.audio = self.root / "sample.wav"
        self.write_audio((0, 1, -1, 2, -2, 3, -3, 0))
        self.reference = {"recording": "alias-fixture", "duration_s": 8 / 16000,
                          "expected": []}

    def write_audio(self, samples):
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<" + "h" * len(samples), *samples))

    def identity(self, fields, *, expected=None, audio_root=None):
        row = {**self.reference, **fields}
        if expected is not None:
            row["expected"] = expected
        refs = self.root / "references.jsonl"
        refs.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        before = {path: path.read_bytes() for path in (refs, self.audio)}
        try:
            return dataset_identity(refs, self.root if audio_root is None else audio_root)
        finally:
            for path, raw in before.items():
                self.assertEqual(path.read_bytes(), raw)

    def test_all_agreeing_aliases_and_whitespace_keep_path_identity(self):
        for count in range(1, 4):
            for fields in itertools.combinations(self.FIELDS, count):
                for whitespace in (False, True):
                    with self.subTest(fields=fields, whitespace=whitespace):
                        value = " \tsample.wav \n" if whitespace else "sample.wav"
                        actual = self.identity({field: value for field in fields})
                        self.assertEqual(actual, (self.PATH_ONLY_ID, {"recordings": 1}))
        # Distinct surrounding whitespace must not create a conflict.
        actual = self.identity({"audio": " sample.wav", "audio_path": "sample.wav\t",
                                "path": "sample.wav"})
        self.assertEqual(actual[0], self.PATH_ONLY_ID)

    def test_absolute_aliases_and_relative_audio_root(self):
        for field in self.FIELDS:
            with self.subTest(field=field):
                self.assertEqual(self.identity({field: str(self.audio)},
                                               audio_root=self.root / "unused-root")[0],
                                 self.PATH_ONLY_ID)
        nested = self.root / "nested"
        nested.mkdir()
        self.assertEqual(self.identity({"path": "../sample.wav"}, audio_root=nested)[0],
                         self.PATH_ONLY_ID)

    def test_conflicting_and_invalid_present_aliases_stay_invalid(self):
        for left, right in itertools.combinations(self.FIELDS, 2):
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, "aliases disagree"):
                    self.identity({left: "sample.wav", right: "other.wav"})
        for field in self.FIELDS:
            other = next(name for name in self.FIELDS if name != field)
            for value in (None, "", "  ", 1, True, []):
                with self.subTest(field=field, invalid=value):
                    with self.assertRaisesRegex(ValueError, "must be non-empty text"):
                        self.identity({other: "sample.wav", field: value})
        with self.assertRaisesRegex(ValueError, "expected non-empty audio/audio_path/path"):
            self.identity({})

    def test_stale_execution_path_is_not_an_identity_alias(self):
        for field in self.FIELDS:
            with self.subTest(field=field):
                actual = self.identity({field: "sample.wav", "_execution_path": "obsolete.wav"})
                self.assertEqual(actual[0], self.PATH_ONLY_ID)

    def test_original_path_golden_and_content_sensitivity(self):
        baseline = self.identity({"path": "sample.wav"})
        self.assertEqual(baseline, (self.PATH_ONLY_ID, {"recordings": 1}))
        changed_expected = self.identity(
            {"path": "sample.wav"},
            expected=[{"keyword_id": 1, "start_s": 0, "end_s": 4 / 16000}],
        )
        self.assertNotEqual(changed_expected[0], baseline[0])
        self.write_audio((0, 1, -1, 2, -2, 3, -3, 1))
        self.assertNotEqual(self.identity({"path": "sample.wav"})[0], baseline[0])




def far_summary(*, negative_hours=1.0, positive_hours=1.0, negative_count=0,
                positive_unmatched=2, include_positive=True):
    """Artificial metadata only: these durations are not measured exposure."""
    rows = [{"recording": "n", "duration_s": negative_hours * 3600, "expected": []}]
    dets = [{"recording": "n", "keyword_id": 1, "time_s": 1.0 + i,
             "confidence": 0.9} for i in range(negative_count)]
    if include_positive:
        rows.append({"recording": "p", "duration_s": positive_hours * 3600,
                     "expected": [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]})
        dets.append({"recording": "p", "keyword_id": 1, "time_s": 2.0, "confidence": 0.9})
        dets.extend({"recording": "p", "keyword_id": 2, "time_s": 4.0 + i,
                     "confidence": 0.9} for i in range(positive_unmatched))
    recordings = score_events.validate_recordings(rows)
    detections = score_events.validate_detections(dets, recordings)
    return score_events.score(recordings, detections, 0.15, 0.5)[0]


def iteration_assembly_fixture(protocol=None, missing=False, invalid_bound=False):
    """Normal-interpreter main with three backend commands mocked, never a model."""
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        audio = root / "tiny.wav"
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 800)
        rows = [{"recording": "n", "path": "tiny.wav", "duration_s": 0.05, "expected": []},
                {"recording": "p", "path": "tiny.wav", "duration_s": 0.05,
                 "expected": [{"keyword_id": 1, "start_s": 0.01, "end_s": 0.02}]}]
        dets = [{"recording": "p", "keyword_id": k, "time_s": t, "confidence": 0.9}
                for k, t in ((1, 0.02), (2, 0.03), (3, 0.04))]
        refs = root / "refs.jsonl"
        refs.write_text("".join(json.dumps(r) + "\n" for r in rows))
        inputs = [refs, audio]
        for name in ("runner", "model", "keywords"):
            path = root / name
            path.write_bytes(b"inert software fixture, never executed\n")
            inputs.append(path)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        target = root / "card.json"
        sentinel = b"existing scorecard sentinel\n"
        if missing or invalid_bound:
            target.write_bytes(sentinel)
        calls = []
        summaries = []
        def fake_run(argv):
            name = pathlib.Path(argv[1]).name
            calls.append(name)
            option = lambda key: pathlib.Path(argv[argv.index(key) + 1])
            if name == "run_corpus.py":
                option("--detections").write_text("".join(json.dumps(d) + "\n" for d in dets))
                option("--provenance").write_text("{}\n")
            elif name == "score_events.py":
                recordings = score_events.validate_recordings(score_events.load_jsonl(option("--references")))
                detections = score_events.validate_detections(score_events.load_jsonl(option("--detections")), recordings)
                summary = score_events.score(recordings, detections, 0.15, 0.5)[0]
                summaries.append(copy.deepcopy(summary))
                if missing:
                    del summary["negative_recording_false_accepts"]
                if invalid_bound:
                    summary["negative_recording_audio_hours"] = 1e-309
                option("--summary").write_text(json.dumps(summary))
                option("--false-positives").write_text("")
                option("--false-rejects").write_text("")
            elif name == "domain_metrics.py":
                option("--output").write_text("{}\n")
            else:
                raise AssertionError("unexpected backend command: " + name)
        argv = [str(RUN), "--runner", str(root / "runner"), "--model", str(root / "model"),
                "--keywords", str(root / "keywords"), "--references", str(refs),
                "--audio-root", str(root), "--work-dir", str(root / "work"),
                "--output", str(target), "--variable", "model", "--code-sha", "fixture-code"]
        if protocol is not None:
            argv += ["--protocol", protocol]
        error = None
        with mock.patch.object(sys, "argv", argv), mock.patch.object(iteration, "run", fake_run), contextlib.redirect_stdout(io.StringIO()):
            try:
                exit_code = iteration.main()
            except ValueError as exc:
                error, exit_code = str(exc), 2
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        return {"exit": exit_code, "error": error, "calls": calls, "input_unchanged": before == after,
                "card": json.loads(target.read_text()) if error is None else None,
                "summary": summaries[0], "sentinel_preserved": (missing or invalid_bound) and target.read_bytes() == sentinel,
                "earlier_work_remains": (root / "work/detections.jsonl").is_file()}


class FarPopulationPolicyTests(unittest.TestCase):
    POLICY = "negative-only-recordings-iteration-approx95-v1"

    def new_card(self, **kwargs):
        card = scorecard(**kwargs)
        card["schema_version"] = 2
        card["identity"]["far_policy"] = self.POLICY
        card["far_rate"]["observed_false_accepts"] = 0
        card["far_rate"]["upper_bound_95_per_hour"] = 2.9957 / card["exposure"]["exposure_hours"]
        return card

    def test_mixed_global_diagnostics_and_positive_invariance(self):
        summary = far_summary()
        hours, rate = iteration.negative_far_rate(summary)
        self.assertEqual(summary["false_accepts"], 2)
        self.assertEqual(summary["far_per_hour"], 1.0)
        self.assertEqual((hours, rate["observed_false_accepts"], rate["point_estimate_per_hour"]), (1.0, 0, 0.0))
        self.assertEqual(rate["upper_bound_95_per_hour"], 2.9957)
        for kwargs in ({"positive_unmatched": 0}, {"positive_unmatched": 4}, {"positive_hours": 2.0}):
            with self.subTest(kwargs=kwargs):
                changed = far_summary(**kwargs)
                self.assertEqual(iteration.negative_far_rate(changed), (hours, rate))

    def test_negative_sensitivity_and_pure_negative_compatibility(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                summary = far_summary(negative_count=count, include_positive=False)
                hours, rate = iteration.negative_far_rate(summary)
                self.assertEqual(rate["point_estimate_per_hour"], summary["far_per_hour"])
                self.assertEqual(rate["upper_bound_95_per_hour"], iteration.far_upper_bound_95(summary["false_accepts"], hours))
                longer_hours, longer = iteration.negative_far_rate(far_summary(negative_count=count, negative_hours=2.0, include_positive=False))
                self.assertEqual(longer_hours, 2.0)
                self.assertEqual(longer["point_estimate_per_hour"], rate["point_estimate_per_hour"] / 2)
                self.assertEqual(longer["upper_bound_95_per_hour"], rate["upper_bound_95_per_hour"] / 2)

    def test_missing_and_invalid_producer_tuple(self):
        fields = ("negative_recording_false_accepts", "negative_recording_audio_hours", "negative_recording_far_per_hour")
        good = far_summary()
        for field in fields:
            with self.subTest(missing=field):
                value = dict(good)
                del value[field]
                with self.assertRaisesRegex(ValueError, "missing required"):
                    iteration.negative_far_rate(value)
        recordings = score_events.validate_recordings([{"recording": "only-positive", "duration_s": 10,
                     "expected": [{"keyword_id": 1, "start_s": 1, "end_s": 2}]}])
        no_negative = score_events.score(recordings, {}, 0.15, 0.5)[0]
        self.assertIsNone(no_negative[fields[2]])
        with self.assertRaises(ValueError):
            iteration.negative_far_rate(no_negative)
        invalid = [(fields[0], x) for x in (True, -1, 0.0)] + [(fields[1], x) for x in (0, -1, float("inf"), True)] + [(fields[2], x) for x in (None, float("nan"), 1.0, True)]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                iteration.negative_far_rate({**good, field: value})

        # A finite positive exposure can still overflow the computed bound.
        for count, hours in ((0, 1e-309), (1, 1e-308)):
            with self.subTest(count=count, hours=hours), self.assertRaisesRegex(ValueError, "bound must be finite"):
                iteration.negative_far_rate({fields[0]: count, fields[1]: hours,
                                             fields[2]: count / hours})

    def test_policy_schema_compatibility_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for protocol in ("dataset-iteration-v1", "custom-label"):
                before = scorecard(protocol=protocol)
                after = scorecard(protocol=protocol, model="m2")
                self.assertEqual(run(root, before, after).returncode, 0)
                output = root / "comparison.json"
                output.write_bytes(b"sentinel\n")
                done = run(root, before, self.new_card(protocol=protocol, model="m2"), "--output", str(output))
                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                self.assertIn("FAR policy mismatch", done.stderr)
                self.assertEqual(output.read_bytes(), b"sentinel\n")
            for kind in ("missing", "unknown", "legacy-with-policy", "unknown-schema"):
                before = self.new_card(protocol="custom-label")
                after = self.new_card(protocol="custom-label", model="m2")
                if kind == "missing":
                    del after["identity"]["far_policy"]
                elif kind == "unknown":
                    after["identity"]["far_policy"] = "other"
                elif kind == "legacy-with-policy":
                    after["schema_version"] = 1
                else:
                    after["schema_version"] = 4
                done = run(root, before, after)
                self.assertEqual(done.returncode, 2, kind + done.stdout + done.stderr)

    def test_new_comparator_rejects_incomplete_tuple(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for side in ("before", "after"):
                for group, key in (("far_rate", "observed_false_accepts"), ("far_rate", "point_estimate_per_hour"), ("far_rate", "upper_bound_95_per_hour"), ("exposure", "exposure_hours")):
                    before, after = self.new_card(), self.new_card(model="m2")
                    del (before if side == "before" else after)[group][key]
                    done = run(root, before, after)
                    self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            for group in ("far_rate", "exposure"):
                after = self.new_card(model="m2")
                after[group] = []
                done = run(root, self.new_card(), after)
                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                self.assertIn("must be objects", done.stderr)
            for group, key, value in (("far_rate", "observed_false_accepts", True), ("exposure", "exposure_hours", 0), ("far_rate", "point_estimate_per_hour", 1.0), ("far_rate", "upper_bound_95_per_hour", -1)):
                after = self.new_card(model="m2")
                after[group][key] = value
                self.assertEqual(run(root, self.new_card(), after).returncode, 2)

            output = root / "comparison.json"
            for count in (0, 1, 2):
                for hours in (0.25, 2.0):
                    before, after = self.new_card(exposure_hours=hours), self.new_card(model="m2", exposure_hours=hours)
                    expected = iteration.far_upper_bound_95(count, hours)
                    for card in (before, after):
                        card["far_rate"].update(observed_false_accepts=count,
                                                point_estimate_per_hour=count / hours,
                                                upper_bound_95_per_hour=expected)
                    done = run(root, before, after, "--far-budget", "100")
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    for side in ("before", "after"):
                        for bound in (0.0, expected / 2, expected * 2):
                            with self.subTest(count=count, hours=hours, side=side, bound=bound):
                                left, right = copy.deepcopy(before), copy.deepcopy(after)
                                (left if side == "before" else right)["far_rate"]["upper_bound_95_per_hour"] = bound
                                output.write_bytes(b"sentinel bound validation\n")
                                done = run(root, left, right, "--output", str(output))
                                self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
                                self.assertIn("inconsistent negative FAR bound", done.stderr)
                                self.assertEqual(output.read_bytes(), b"sentinel bound validation\n")
            # Legacy cards retain their historical permissive bound behavior.
            done = run(root, scorecard(bound=0), scorecard(model="m2", bound=0))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_new_cards_keep_verdicts(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for frr, code, verdict in ((0.05, 0, "improved"), (0.1, 0, "neutral"), (0.2, 1, "regressed")):
                done = run(root, self.new_card(), self.new_card(model="m2", frr=frr))
                self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                self.assertEqual(json.loads(done.stdout)["verdict"], verdict)

    def test_actual_main_assembly_and_run_identity(self):
        # The normal-interpreter child isolates the known RUN --OO docstring
        # limitation. It mocks backend commands, not scorer arithmetic; helper
        # and comparator tests above really use the parent optimization mode.
        for protocol, missing, invalid_bound in ((None, False, False), ("custom-label", False, False), ("custom-label", True, False), ("custom-label", False, True)):
            command = [sys.executable, "-S", "-B", "-c",
                       "import sys,json;sys.path.insert(0," + repr(str(ROOT / "tests")) + ");import test_dataset_iteration as t;print(json.dumps(t.iteration_assembly_fixture(" + repr(protocol) + "," + repr(missing) + "," + repr(invalid_bound) + ")))"]
            done = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            result = json.loads(done.stdout)
            self.assertTrue(result["input_unchanged"])
            self.assertEqual(result["calls"], ["run_corpus.py", "score_events.py", "domain_metrics.py"])
            if missing or invalid_bound:
                self.assertEqual(result["exit"], 2)
                self.assertTrue(result["sentinel_preserved"])
                self.assertTrue(result["earlier_work_remains"])
                continue
            card = result["card"]
            self.assertEqual(card["schema_version"], 4)
            self.assertEqual(card["identity"]["dataset_identity_policy"], "effective-domain-keys-verified-afe-v2")
            self.assertEqual(card["identity"]["far_policy"], self.POLICY)
            self.assertEqual(card["protocol"], "dataset-iteration-v4" if protocol is None else protocol)
            self.assertEqual(card["metrics"]["false_accepts"], result["summary"]["false_accepts"])
            self.assertEqual(card["metrics"]["far_per_hour"], result["summary"]["far_per_hour"])
            self.assertEqual(card["far_rate"]["point_estimate_per_hour"], 0.0)
            self.assertEqual(card["far_rate"]["upper_bound_95_per_hour"], iteration.far_upper_bound_95(0, card["exposure"]["exposure_hours"]))
            self.assertEqual(card["run_id"], iteration.canonical_hash(card["identity"])[:16])
            legacy_identity = dict(card["identity"])
            del legacy_identity["far_policy"]
            self.assertNotEqual(card["run_id"], iteration.canonical_hash(legacy_identity)[:16])
            domain_untagged = dict(card["identity"])
            del domain_untagged["dataset_identity_policy"]
            self.assertNotEqual(card["run_id"], iteration.canonical_hash(domain_untagged)[:16])


class EffectiveDomainIdentityTests(unittest.TestCase):
    """Literal metadata oracle plus tiny inert WAVs; no backend/model runs."""

    setUp = DatasetIdentityAliasTests.setUp
    write_audio = DatasetIdentityAliasTests.write_audio

    def identity(self, fields, *, rows=None):
        row = {**self.reference, "path": "sample.wav", **fields}
        refs = self.root / "domain-references.jsonl"
        # Preserve category-container insertion order: domain_keys uses str().
        refs.write_text("".join(json.dumps(r) + "\n" for r in
                                ([row] if rows is None else rows)), encoding="utf-8")
        before = {p: p.read_bytes() for p in (refs, self.audio)}
        result = dataset_identity(refs, self.root)
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        return result[0]

    def check_domain(self, domain, expected):
        fields = {"domain": domain}
        self.assertEqual(sorted(iteration.domain_keys(fields)), sorted(expected))
        actual = self.identity(fields)
        # Independent literal envelope oracle, not the producer's hashing helper.
        entries = [{
            "recording": "alias-fixture",
            "wav_sha256": hashlib.sha256(self.audio.read_bytes()).hexdigest(),
            "duration_s": 0.0005, "expected": [], "domains": sorted(expected),
            "afe_metadata": {"present": False},
        }]
        encoded = json.dumps({"dataset_identity_policy": "effective-domain-keys-verified-afe-v2",
                              "entries": entries}, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertEqual(actual, hashlib.sha256(encoded).hexdigest())
        return actual

    def test_nested_collision_and_ignored_top_level(self):
        near = self.check_domain({"distance_m": 0.75}, ["all", "distance_bin:0.5m"])
        far = self.check_domain({"distance_m": 0.76}, ["all", "distance_bin:1m"])
        self.assertNotEqual(near, far)
        baseline = self.identity({})
        for fields in ({"distance_m": 0.75}, {"distance_m": 4.1},
                       {"distance_band": "near", "domain_id": "x",
                        "scene": {"distance_m": 2}}, {"domain": {"unknown": 9}}):
            self.assertEqual(self.identity(fields), baseline)
        self.assertEqual(self.identity({"domain": {"distance_m": 0.75},
                                        "distance_m": 99}), near)

    def test_axes_and_sparse_precedence(self):
        baseline = self.check_domain({}, ["all"])
        self.assertEqual(self.identity({}), baseline)
        for domain in (None, [], "ignored", 0):
            self.assertEqual(self.check_domain(domain, ["all"]), baseline)
        cases = [
            ({"distance_band": "near"}, ["distance:near"]),
            ({"distance_m": 0}, ["distance_bin:0.5m"]),
            ({"azimuth_deg": 0}, ["azimuth:front", "azimuth_deg:0"]),
            ({"snr_db": 0}, ["snr:critical"]),
            ({"rt60_s": 0}, ["rt60:dry"]),
            ({"noise_profile": "fan"}, ["noise:fan"]),
            ({"playback_sir_db": None}, ["playback:no-playback"]),
            ({"playback_sir_db": 0}, ["playback:playback"]),
            ({"distance_band": "far", "distance_m": 0.5},
             ["distance:far", "distance_bin:0.5m"]),
        ]
        for domain, keys in cases:
            with self.subTest(domain=domain):
                self.assertNotEqual(self.check_domain(domain, ["all", *keys]), baseline)
        for axis, a, b in (("distance_band", "near", "far"),
                           ("noise_profile", "fan", "street"),
                           ("playback_sir_db", None, 0)):
            self.assertNotEqual(self.identity({"domain": {axis: a}}),
                                self.identity({"domain": {axis: b}}))

    def test_pairwise_triple_and_composite_creation(self):
        cases = [
            ({"distance_m": 1, "azimuth_deg": 0},
             ["all", "distance_bin:1m", "azimuth:front", "azimuth_deg:0",
              "distance_azimuth:distance_bin=1m|azimuth=front"]),
            ({"distance_m": 1, "snr_db": 6},
             ["all", "distance_bin:1m", "snr:critical",
              "distance_snr:distance_bin=1m|snr=critical"]),
            ({"azimuth_deg": 0, "snr_db": 6},
             ["all", "azimuth:front", "azimuth_deg:0", "snr:critical",
              "azimuth_snr:azimuth=front|snr=critical"]),
            ({"distance_m": 1, "azimuth_deg": 0, "snr_db": 6},
             ["all", "distance_bin:1m", "azimuth:front", "azimuth_deg:0",
              "snr:critical", "distance_azimuth:distance_bin=1m|azimuth=front",
              "distance_snr:distance_bin=1m|snr=critical",
              "azimuth_snr:azimuth=front|snr=critical",
              "distance_azimuth_snr:distance_bin=1m|azimuth=front|snr=critical"]),
            ({"distance_band": "near", "azimuth_deg": 0, "rt60_s": 0.1,
              "noise_profile": "fan", "playback_sir_db": None},
             ["all", "distance:near", "azimuth:front", "azimuth_deg:0",
              "rt60:dry", "noise:fan", "playback:no-playback",
              "composite:distance=near|az=front|rt60=dry|noise=fan|no-playback"]),
        ]
        for domain, expected in cases:
            with self.subTest(domain=domain):
                full = self.check_domain(domain, expected)
                for final in domain:
                    sparse = {k: v for k, v in domain.items() if k != final}
                    self.assertNotEqual(self.identity({"domain": sparse}), full)
                    self.assertFalse(any(k.startswith("composite:")
                                         for k in iteration.domain_keys({"domain": sparse})))

    def test_canonical_equivalences(self):
        pairs = [
            ({"distance_m": 1}, {"distance_m": 1.0}),
            ({"distance_m": 0.8}, {"distance_m": 1.5}),
            ({"azimuth_deg": 20}, {"azimuth_deg": 380}),
            ({"azimuth_deg": -180}, {"azimuth_deg": 180}),
            ({"azimuth_deg": 0}, {"azimuth_deg": -0.0}),
            ({"snr_db": 7}, {"snr_db": 12}),
            ({"rt60_s": 0.3}, {"rt60_s": 0.54}),
            ({"playback_sir_db": 0}, {"playback_sir_db": 90}),
            ({"noise_profile": 1}, {"noise_profile": "1"}),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(sorted(iteration.domain_keys({"domain": left})),
                                 sorted(iteration.domain_keys({"domain": right})))
                self.assertEqual(self.identity({"domain": left}),
                                 self.identity({"domain": right}))
        domain = {"distance_m": 1, "azimuth_deg": 0, "snr_db": 6}
        self.assertEqual(self.identity({"domain": domain}),
                         self.identity({"domain": dict(reversed(list(domain.items())))}))

    def test_numeric_cutoffs(self):
        for axis, prefix, cases in (
            ("distance_m", "distance_bin", [(0.75, "0.5m", "1m"),
                (1.5, "1m", "2m"), (2.5, "2m", "3m"), (4.0, "3m", "5m")]),
            ("snr_db", "snr", [(6.0, "critical", "low"),
                (12.0, "low", "mid"), (20.0, "mid", "high")]),
        ):
            for edge, lower, upper in cases:
                with self.subTest(axis=axis, edge=edge):
                    a = self.check_domain({axis: edge}, ["all", prefix + ":" + lower])
                    self.assertEqual(self.check_domain({axis: math.nextafter(edge, -math.inf)},
                                                       ["all", prefix + ":" + lower]), a)
                    b = self.check_domain({axis: math.nextafter(edge, math.inf)},
                                          ["all", prefix + ":" + upper])
                    self.assertNotEqual(a, b)
        for edge, lower, upper in ((0.30, "dry", "medium"), (0.55, "medium", "reverb")):
            a = self.check_domain({"rt60_s": math.nextafter(edge, -math.inf)},
                                  ["all", "rt60:" + lower])
            b = self.check_domain({"rt60_s": edge}, ["all", "rt60:" + upper])
            self.assertNotEqual(a, b)
            self.assertEqual(self.check_domain({"rt60_s": math.nextafter(edge, math.inf)},
                                               ["all", "rt60:" + upper]), b)

    def test_azimuth_boundaries_and_ties(self):
        # Values comfortably straddle modulo rounding as well as 30/90 bands.
        cases = [(15, "front", "0"), (15.000001, "front", "30"),
                 (14.999999, "front", "0"), (45, "side", "60"),
                 (44.999999, "side", "30"), (-15, "front", "0"),
                 (-15.000001, "front", "-30"), (30, "front", "30"),
                 (30.000001, "side", "30"), (90, "side", "90"),
                 (90.000001, "rear", "90"), (-90, "side", "-90"),
                 (-90.000001, "rear", "-90"), (180, "rear", "180"),
                 (-180, "rear", "180")]
        for value, band, point in cases:
            with self.subTest(value=value):
                self.check_domain({"azimuth_deg": value},
                                  ["all", "azimuth:" + band, "azimuth_deg:" + point])
        for a, b in ((15, 15.000001), (45, 44.999999), (30, 30.000001),
                     (90, 90.000001)):
            self.assertNotEqual(self.identity({"domain": {"azimuth_deg": a}}),
                                self.identity({"domain": {"azimuth_deg": b}}))

    def test_validation_and_categorical_playback_semantics(self):
        for axis in ("distance_m", "azimuth_deg", "rt60_s", "snr_db"):
            for value in (True, False, "1", None, float("nan"), float("inf"), -float("inf")):
                with self.subTest(axis=axis, value=value):
                    domain = {axis: value}
                    with self.assertRaises(ValueError) as helper:
                        iteration.domain_keys({"domain": domain})
                    with self.assertRaises(ValueError) as producer:
                        self.identity({"domain": domain})
                    self.assertEqual(str(producer.exception), str(helper.exception))
        self.check_domain({"distance_band": None, "noise_profile": None},
                          ["all", "distance:None", "noise:None"])
        self.check_domain({"distance_band": ["near"], "noise_profile": True},
                          ["all", "distance:['near']", "noise:True"])
        a = self.check_domain({"noise_profile": {"a": 1, "b": 2}},
                              ["all", "noise:{'a': 1, 'b': 2}"])
        b = self.check_domain({"noise_profile": {"b": 2, "a": 1}},
                              ["all", "noise:{'b': 2, 'a': 1}"])
        self.assertNotEqual(a, b)  # Do not invent categorical normalization.
        for value in ("arbitrary", [], {}, False):
            self.check_domain({"playback_sir_db": value}, ["all", "playback:playback"])

    def test_row_order_and_policy_tagged_envelope(self):
        rows = [{**self.reference, "recording": name, "path": "sample.wav",
                 "domain": {"distance_m": distance}}
                for name, distance in (("b", 2), ("a", 1))]
        self.assertEqual(self.identity({}, rows=rows),
                         self.identity({}, rows=list(reversed(rows))))
        baseline = self.identity({})
        self.assertEqual(baseline, DatasetIdentityAliasTests.PATH_ONLY_ID)
        self.assertNotEqual(baseline, DatasetIdentityAliasTests.LEGACY_PATH_ONLY_ID)
        self.assertNotEqual(baseline, DatasetIdentityAliasTests.DOMAIN_ONLY_PATH_ID)
        # A policy tag change must change the envelope even for all-only data.
        with mock.patch.object(iteration, "DATASET_IDENTITY_POLICY", "test-other-policy"):
            self.assertNotEqual(self.identity({}), baseline)


class DomainPolicyCompatibilityTests(unittest.TestCase):
    POLICY = FarPopulationPolicyTests.POLICY

    def new_card(self, **kwargs):
        card = FarPopulationPolicyTests.new_card(self, **kwargs)
        card["schema_version"] = 3
        card["identity"]["dataset_identity_policy"] = "effective-domain-keys-v1"
        return card

    def reject(self, root, before, after, message):
        target = root / "policy-output.json"
        target.write_bytes(b"existing report sentinel\n")
        done = run(root, before, after, "--output", str(target))
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertNotIn("Traceback", done.stderr)
        self.assertIn(message, done.stderr)
        self.assertEqual(target.read_bytes(), b"existing report sentinel\n")

    def test_legacy_and_new_controls_and_mixed_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for protocol in ("dataset-iteration-v3", "custom-label"):
                factories = [scorecard,
                             lambda **kw: FarPopulationPolicyTests.new_card(self, **kw),
                             self.new_card]
                for factory in factories:
                    before, after = factory(protocol=protocol), factory(protocol=protocol, model="m2")
                    done = run(root, before, after)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                for left, right in itertools.permutations(range(3), 2):
                    before = factories[left](protocol=protocol)
                    after = factories[right](protocol=protocol, model="m2")
                    self.reject(root, before, after,
                                "dataset identity policy mismatch" if 2 in (left, right)
                                else "FAR policy mismatch")

    def test_unknown_missing_forged_policies_and_schemas(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            invalid = []
            for policy in (None, "", "unknown", True, [], {}):
                card = self.new_card(model="m2", protocol="custom-label")
                card["identity"]["dataset_identity_policy"] = policy
                invalid.append(card)
            card = self.new_card(model="m2", protocol="custom-label")
            del card["identity"]["dataset_identity_policy"]
            invalid.append(card)
            for schema in (None, True, False, 0, 4, "3", 3.0):
                card = self.new_card(model="m2", protocol="custom-label")
                card["schema_version"] = schema
                invalid.append(card)
            card = self.new_card(model="m2", protocol="custom-label")
            del card["schema_version"]
            invalid.append(card)
            for schema in (1, 2):
                for policy in (None, "effective-domain-keys-v1", "legacy-top-level-distance-v1"):
                    card = scorecard(model="m2", protocol="custom-label")
                    card["schema_version"] = schema
                    if schema == 2:
                        card = FarPopulationPolicyTests.new_card(self, model="m2", protocol="custom-label")
                    card["identity"]["dataset_identity_policy"] = policy
                    invalid.append(card)
            for card in invalid:
                for reverse in (False, True):
                    good = self.new_card(protocol="custom-label")
                    before, after = (card, good) if reverse else (good, card)
                    self.reject(root, before, after, "policy")
            for policy in (None, "unknown"):
                card = self.new_card(model="m2")
                if policy is None:
                    del card["identity"]["far_policy"]
                else:
                    card["identity"]["far_policy"] = policy
                for before, after in ((self.new_card(), card), (card, self.new_card())):
                    self.reject(root, before, after, "requires the known FAR policy")

    def test_schema3_keeps_all_far_tuple_validation(self):
        # Run the unchanged schema-2 strict matrix against schema 3 too.
        FarPopulationPolicyTests.test_new_comparator_rejects_incomplete_tuple(self)
        FarPopulationPolicyTests.test_new_cards_keep_verdicts(self)

    def test_framing_uses_effective_domains(self):
        fixture = EffectiveDomainIdentityTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        base = fixture.identity({"domain": {"distance_m": 1}})
        changed = fixture.identity({"domain": {"distance_m": 2}})
        equivalent = fixture.identity({"domain": {"distance_m": 1.4}})
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            before = self.new_card(dataset=base, variable="dataset")
            after = self.new_card(dataset=changed, variable="dataset")
            done = run(root, before, after)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)["variable"], "dataset")
            after = self.new_card(dataset=equivalent, model="m2")
            done = run(root, self.new_card(dataset=base), after)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)["variable"], "model")
            self.reject(root, before, self.new_card(dataset=changed, model="m2"),
                        "both dataset and model changed")
            self.reject(root, self.new_card(dataset=base),
                        self.new_card(dataset=equivalent), "nothing moved")
            self.reject(root, before, self.new_card(dataset=changed), "dataset_id is what actually changed")
            self.reject(root, before, self.new_card(dataset=changed, variable="dataset", code="c2"),
                        "code_sha differs")
            self.reject(root, before, self.new_card(dataset=changed, variable="dataset", protocol="other"),
                        "protocol mismatch")



def afe_pipeline_fixture():
    """Normal-interpreter only: actual Python stages, solely external runner mocked.

    Literal PCM/TSV and nonexecutable placeholders are not model evidence. The
    parent tests exercise helpers/comparator under their actual optimization mode.
    """
    import run_corpus
    import domain_metrics
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        with wave.open(str(root / "sample.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(struct.pack("<8h", 0, 1, -1, 2, -2, 3, -3, 0))
        low = "kws-afe-timeline-v1\n0 8 0 0 16 0 0.1 0 " + "0" * 64 + "\n"
        high = low.replace("0.1", "0.9")
        for name, raw in (("low.tsv", low), ("same.tsv", low), ("high.tsv", high)):
            (root / name).write_text(raw)
        for name in ("runner", "model-a", "model-b", "pack"):
            (root / name).write_text("NONEXECUTABLE ARTIFICIAL PLACEHOLDER " + name)
        base = {"recording": "literal-eight", "path": "sample.wav",
                "duration_s": 8 / 16000, "expected": []}
        def binding(name):
            return dict(base, metadata_path=name,
                        metadata_sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())
        cases = [("low", binding("low.tsv"), "model-a"),
                 ("high", binding("high.tsv"), "model-a"),
                 ("high-model", binding("high.tsv"), "model-b"),
                 ("same", binding("same.tsv"), "model-a"),
                 ("same-model", binding("same.tsv"), "model-b"),
                 ("equivalent-binding", binding("low.tsv"), "model-a"),
                 ("absent", dict(base), "model-a")]
        for name, fields in (("wrong-sha", {"metadata_sha256": "0" * 64}),
                             ("missing-file", {"metadata_path": "missing.tsv"}),
                             ("empty-path", {"metadata_path": ""})):
            cases.append((name, {**binding("low.tsv"), **fields}, "model-a"))
        cases.append(("missing-sha", dict(base, metadata_path="low.tsv"), "model-a"))
        cases.append(("only-sha", dict(base, metadata_sha256=binding("low.tsv")["metadata_sha256"]), "model-a"))
        for name in ("during-runner", "after-corpus", "effective-binding", "original-binding"):
            cases.append((name, binding("low.tsv"), "model-a"))
        results = {}
        for name, row, model in cases:
            (root / "low.tsv").write_text(low)
            refs = root / (name + ".jsonl")
            refs.write_text(json.dumps(row) + "\n")
            output = root / (name + ".card.json")
            sentinel = b"existing scorecard sentinel\n"
            output.write_bytes(sentinel)
            calls, stages = [], []
            inputs = [refs, *[root / item for item in
                      ("sample.wav", "low.tsv", "high.tsv", "same.tsv", "runner", "model-a", "model-b", "pack")]]
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            def runner(command, **kwargs):
                if command[0] != str(root / "runner"):
                    raise RuntimeError("unexpected subprocess: " + repr(command))
                calls.append(list(map(str, command)))
                if name == "during-runner":
                    (root / "low.tsv").write_text(high)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            def stage(command):
                script = pathlib.Path(command[1]).name
                stages.append(script)
                with mock.patch.object(sys, "argv", command[1:]):
                    if script == "run_corpus.py":
                        with mock.patch.object(run_corpus.subprocess, "run", runner):
                            code = run_corpus.main()
                    elif script == "score_events.py":
                        code = score_events.main()
                    elif script == "domain_metrics.py":
                        code = domain_metrics.main()
                    else:
                        raise RuntimeError("unexpected stage: " + script)
                if code != 0:
                    raise RuntimeError("Python stage failed: " + script)
                if script == "run_corpus.py" and name == "after-corpus":
                    (root / "low.tsv").write_text(high)
                if script == "domain_metrics.py" and name in ("effective-binding", "original-binding", "equivalent-binding"):
                    target = (root / (name + ".work") / "references.effective.jsonl"
                              if name != "original-binding" else refs)
                    target.write_text(json.dumps(binding("same.tsv" if name == "equivalent-binding" else "high.tsv")) + "\n")
            argv = [str(RUN), "--runner", str(root / "runner"), "--model", str(root / model),
                    "--keywords", str(root / "pack"), "--references", str(refs),
                    "--audio-root", str(root), "--work-dir", str(root / (name + ".work")),
                    "--output", str(output), "--variable", "model", "--seed", "fixture",
                    "--code-sha", "fixture-code", "--protocol", "custom-label"]
            error = None
            with mock.patch.object(sys, "argv", argv), mock.patch.object(iteration, "run", stage), contextlib.redirect_stdout(io.StringIO()):
                try:
                    code = iteration.main()
                except (OSError, ValueError) as exc:
                    code, error = 2, str(exc)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            unchanged = before == after
            results[name] = {"code": code, "error": error, "calls": calls, "stages": stages,
                             "unchanged": unchanged, "input_before": before, "input_after": after,
                             "sentinel_preserved": output.read_bytes() == sentinel}
            if code == 0:
                results[name]["card"] = json.loads(output.read_text())
        return results


class VerifiedAfeIdentityTests(unittest.TestCase):
    POLICY = FarPopulationPolicyTests.POLICY
    setUp = DatasetIdentityAliasTests.setUp
    write_audio = DatasetIdentityAliasTests.write_audio
    identity = EffectiveDomainIdentityTests.identity
    reject = DomainPolicyCompatibilityTests.reject

    def new_card(self, **kwargs):
        card = FarPopulationPolicyTests.new_card(self, **kwargs)
        card["schema_version"] = 4
        card["identity"]["dataset_identity_policy"] = "effective-domain-keys-verified-afe-v2"
        return card

    def sidecar(self, name="sidecar.tsv", value="0.1"):
        path = self.root / name
        path.write_text("kws-afe-timeline-v1\n0 8 0 0 16 0 " + value + " 0 " + "0" * 64 + "\n")
        return {"metadata_path": name, "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_presence_content_and_path_equivalence(self):
        absent = self.identity({})
        fields = self.sidecar()
        present = self.identity(fields)
        self.assertNotEqual(absent, present)
        self.assertEqual(present, self.identity(self.sidecar("relocated.tsv")))
        self.assertEqual(present, self.identity({**fields, "metadata_path": str(self.root / fields["metadata_path"])}))
        self.assertNotEqual(present, self.identity(self.sidecar("changed.tsv", "0.9")))
        # The literal envelope oracle does not call either implementation hash helper.
        entries = [{"recording": "alias-fixture", "wav_sha256": hashlib.sha256(self.audio.read_bytes()).hexdigest(),
                    "duration_s": 0.0005, "expected": [], "domains": ["all"],
                    "afe_metadata": {"present": True, "sha256": fields["metadata_sha256"]}}]
        encoded = json.dumps({"dataset_identity_policy": "effective-domain-keys-verified-afe-v2", "entries": entries},
                             ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(present, hashlib.sha256(encoded).hexdigest())

    def test_invalid_declarations_cannot_be_absence_or_trusted_digest(self):
        good = self.sidecar()
        invalid = [{"metadata_path": value, "metadata_sha256": good["metadata_sha256"]}
                   for value in (None, "", 12, [], {})]
        invalid += [{"metadata_path": "missing.tsv", "metadata_sha256": good["metadata_sha256"]},
                    {"metadata_path": good["metadata_path"]}, {"metadata_sha256": good["metadata_sha256"]}]
        invalid += [{**good, "metadata_sha256": value} for value in (None, "", "0" * 64, True, [], {}, good["metadata_sha256"].upper())]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises((OSError, ValueError)):
                self.identity(fields)
        self.assertNotEqual(self.identity(good), self.identity({}))
        (self.root / good["metadata_path"]).write_text("changed bytes")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.identity(good)


    def test_shared_corpus_preflight_timing_and_cache_contracts(self):
        import run_corpus
        for name in ("runner", "model", "pack", "dump", "replay"):
            (self.root / name).write_text("NONEXECUTABLE PLACEHOLDER " + name)
        good = self.sidecar()
        rows = [(good, None), ({}, None),
                ({**good, "metadata_path": str(self.root / good["metadata_path"])}, None),
                ({**good, "timing_contract": "afe-output-sample-v1", "afe_latency_samples": 0}, None),
                ({**good, "metadata_sha256": "0" * 64}, "SHA-256 mismatch"),
                ({"metadata_path": good["metadata_path"]}, "SHA-256 mismatch"),
                ({"metadata_sha256": good["metadata_sha256"]}, "metadata_path"),
                ({**good, "metadata_path": ""}, "metadata_path"),
                ({**good, "metadata_path": "missing.tsv"}, "missing.tsv"),
                ({**good, "duration_s": 10}, "duration"),
                ({**good, "timing_contract": "afe-output-sample-v1", "afe_latency_samples": 1}, "AFE")]
        calls = []
        def runner(command, **kwargs):
            self.assertEqual(command[0], str(self.root / "runner"))
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        for fields, error in rows:
            with self.subTest(fields=fields):
                refs = self.root / "corpus.jsonl"
                refs.write_text(json.dumps({**self.reference, "path": "sample.wav", **fields}) + "\n")
                out = self.root / "corpus-detections.jsonl"
                out.write_bytes(b"sentinel\n")
                argv = ["run_corpus.py", "--runner", str(self.root / "runner"),
                        "--model", str(self.root / "model"), "--keywords", str(self.root / "pack"),
                        "--references", str(refs), "--audio-root", str(self.root), "--detections", str(out)]
                before = len(calls)
                with mock.patch.object(sys, "argv", argv), mock.patch.object(run_corpus.subprocess, "run", runner), contextlib.redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex((OSError, ValueError), error):
                            run_corpus.main()
                        self.assertEqual(len(calls), before)
                        self.assertEqual(out.read_bytes(), b"sentinel\n")
                    else:
                        self.assertEqual(run_corpus.main(), 0)
                        self.assertEqual(len(calls), before + 1)
                        self.assertEqual("--metadata-tsv" in calls[-1], "metadata_path" in fields)
        refs.write_text(json.dumps({**self.reference, "path": "sample.wav", **good}) + "\n")
        argv += ["--posterior-dump", str(self.root / "dump"), "--decoder-replay", str(self.root / "replay"),
                 "--posterior-cache", str(self.root / "cache")]
        before = len(calls)
        with mock.patch.object(sys, "argv", argv), mock.patch.object(run_corpus.subprocess, "run", runner):
            with self.assertRaisesRegex(ValueError, "AFE metadata replay is not supported by the posterior cache"):
                run_corpus.main()
        self.assertEqual(len(calls), before)

    def test_four_way_framing_and_declared_variable(self):
        low = self.identity(self.sidecar())
        high = self.identity(self.sidecar("high.tsv", "0.9"))
        same = self.identity(self.sidecar("same.tsv"))
        before = self.new_card(dataset=low)
        for after, expected in ((self.new_card(dataset=high, variable="dataset"), "dataset"),
                                (self.new_card(dataset=same, model="m2"), "model")):
            done = run(self.root, before, after)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(json.loads(done.stdout)["variable"], expected)
        self.reject(self.root, before, self.new_card(dataset=high, model="m2"), "both dataset and model changed")
        self.reject(self.root, before, self.new_card(dataset=same), "nothing moved")
        self.reject(self.root, before, self.new_card(dataset=high), "dataset_id is what actually changed")

    def test_historical_policy_matrix_cannot_use_custom_protocol_to_bypass(self):
        factories = [scorecard, lambda **kw: FarPopulationPolicyTests.new_card(self, **kw),
                     lambda **kw: DomainPolicyCompatibilityTests.new_card(self, **kw), self.new_card]
        for protocol in ("custom-label", "dataset-iteration-v4"):
            for factory in factories:
                done = run(self.root, factory(protocol=protocol), factory(protocol=protocol, model="m2"))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            for older in factories[:-1]:
                old = older(protocol=protocol)
                new = self.new_card(protocol=protocol, model="m2")
                for before, after in ((old, new), (new, old)):
                    self.reject(self.root, before, after, "dataset identity policy mismatch")

    def test_schema4_rejects_missing_unknown_and_relabelled_policies(self):
        bad = []
        for policy in (None, "", "unknown", True, [], {}, "effective-domain-keys-v1"):
            card = self.new_card(model="m2", protocol="custom-label")
            card["identity"]["dataset_identity_policy"] = policy
            bad.append(card)
        card = self.new_card(model="m2", protocol="custom-label")
        del card["identity"]["dataset_identity_policy"]
        bad.append(card)
        for schema in (None, True, False, 0, 1, 2, 3, 5, "4", 4.0):
            card = self.new_card(model="m2", protocol="custom-label")
            card["schema_version"] = schema
            bad.append(card)
        for card in bad:
            good = self.new_card(protocol="custom-label")
            for before, after in ((good, card), (card, good)):
                self.reject(self.root, before, after, "policy")

    def test_schema4_preserves_far_tuple_and_verdict_matrix(self):
        FarPopulationPolicyTests.test_new_comparator_rejects_incomplete_tuple(self)
        FarPopulationPolicyTests.test_new_cards_keep_verdicts(self)

    def test_actual_python_pipeline_preflight_mutations_and_identity(self):
        # Normal child by design: no claim that producer/corpus CLI supports -OO.
        command = [sys.executable, "-S", "-B", "-c",
                   "import sys,json;sys.path.insert(0," + repr(str(ROOT / "tests")) +
                   ");import test_dataset_iteration as t;print(json.dumps(t.afe_pipeline_fixture()))"]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        result = json.loads(done.stdout)
        for name in ("low", "high", "high-model", "same", "same-model", "equivalent-binding", "absent"):
            item = result[name]
            self.assertEqual(item["code"], 0, item)
            self.assertEqual(len(item["calls"]), 1)
            self.assertEqual(item["stages"], ["run_corpus.py", "score_events.py", "domain_metrics.py"])
            self.assertTrue(item["unchanged"])
            card = item["card"]
            self.assertEqual(card["schema_version"], 4)
            self.assertEqual(card["identity"]["dataset_identity_policy"], "effective-domain-keys-verified-afe-v2")
            self.assertEqual(card["identity"]["protocol"], "custom-label")
            self.assertEqual(card["run_id"], iteration.canonical_hash(card["identity"])[:16])
        cards = {key: result[key]["card"] for key in ("low", "high", "high-model", "same", "same-model", "equivalent-binding", "absent")}
        self.assertEqual(cards["low"]["run_id"], cards["same"]["run_id"])
        self.assertEqual(cards["low"]["run_id"], cards["equivalent-binding"]["run_id"])
        self.assertNotEqual(cards["low"]["run_id"], cards["high"]["run_id"])
        self.assertNotEqual(cards["low"]["run_id"], cards["absent"]["run_id"])
        self.assertIn("--metadata-tsv", result["low"]["calls"][0])
        self.assertNotIn("--metadata-tsv", result["absent"]["calls"][0])
        for name in ("wrong-sha", "missing-file", "empty-path", "missing-sha", "only-sha"):
            item = result[name]
            self.assertEqual(item["code"], 2, item)
            self.assertEqual(item["stages"], [])
            self.assertEqual(item["calls"], [])
            self.assertTrue(item["sentinel_preserved"])
            self.assertTrue(item["unchanged"])
        for name in ("during-runner", "after-corpus", "effective-binding", "original-binding"):
            item = result[name]
            self.assertEqual(item["code"], 2, item)
            self.assertEqual(len(item["calls"]), 1)
            self.assertTrue(item["sentinel_preserved"])
            self.assertIn("changed" if name != "after-corpus" else "SHA-256 mismatch", item["error"])
        self.assertEqual(result["during-runner"]["stages"], ["run_corpus.py"])
        self.assertEqual(len(result["after-corpus"]["stages"]), 3)
        # Compare real assembled cards with actual scorer/domain results, not fake metrics.
        cards["high"]["identity"]["variable"] = "dataset"
        good = run(self.root, cards["low"], cards["high"], "--far-budget", "1e9")
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertEqual(json.loads(good.stdout)["variable"], "dataset")
        model_only = run(self.root, cards["low"], cards["same-model"], "--far-budget", "1e9")
        self.assertEqual(model_only.returncode, 0, model_only.stdout + model_only.stderr)
        self.assertEqual(json.loads(model_only.stdout)["variable"], "model")
        self.reject(self.root, cards["low"], cards["high-model"], "both dataset and model changed")
        self.reject(self.root, cards["low"], cards["same"], "nothing moved")


def main() -> int:
    strict = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromTestCase(RequiredMetricTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(DatasetIdentityAliasTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(FarPopulationPolicyTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(EffectiveDomainIdentityTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(DomainPolicyCompatibilityTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(VerifiedAfeIdentityTests),
        ])
    )
    if strict.testsRun != 36 or strict.skipped or not strict.wasSuccessful():
        return 1

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # Same dataset, better model: recall improves and the false-accept bound
        # does not grow. This is the case the lane exists to detect.
        done = run(root,
                   scorecard(run_id="base", frr=0.20, bound=11.98),
                   scorecard(run_id="cand", model="m2", frr=0.10, bound=11.98))
        assert done.returncode == 0, done.stderr
        report = json.loads(done.stdout)
        assert report["verdict"] == "improved", report
        assert report["variable"] == "model", report
        assert report["deltas"]["frr"] < 0, report

        # Worse recall at the same operating point is a regression, and the tool
        # must say so rather than stay quiet.
        done = run(root,
                   scorecard(run_id="base", frr=0.10),
                   scorecard(run_id="cand", model="m2", frr=0.30))
        assert done.returncode == 1, done.stdout
        assert "frr regressed" in json.loads(done.stdout)["regressions"][0]

        # The whole point of the upper bound: a run whose point estimate is a
        # clean 0.0 can still exceed the budget once its exposure is accounted
        # for. The governed gate would call this a pass.
        done = run(root,
                   scorecard(run_id="base", frr=0.10, exposure_hours=2.0, bound=1.5),
                   scorecard(run_id="cand", model="m2", frr=0.05,
                             exposure_hours=0.05, bound=59.9),
                   "--far-budget", "5.0")
        assert done.returncode == 1, done.stdout
        payload = json.loads(done.stdout)
        assert any("exceeds budget" in item for item in payload["regressions"]), payload
        # Recall genuinely improved, so the verdict must be a regression, not a
        # silent pass: a shorter exposure is weaker evidence, not better.
        assert payload["deltas"]["frr"] < 0

        # Declaring the dataset as the variable and then changing the model is a
        # lie about the experiment; refuse it.
        done = run(root,
                   scorecard(run_id="base", variable="dataset"),
                   scorecard(run_id="cand", model="m2", variable="dataset"))
        assert done.returncode == 2, done.stdout
        assert "model_id is what actually changed" in done.stderr, done.stderr

        # Moving both variables leaves the delta uninterpretable.
        done = run(root,
                   scorecard(run_id="base"),
                   scorecard(run_id="cand", dataset="d2", model="m2"))
        assert done.returncode == 2, done.stdout
        assert "both dataset and model changed" in done.stderr, done.stderr

        # Different protocols are different measurements.
        done = run(root,
                   scorecard(run_id="base"),
                   scorecard(run_id="cand", model="m2", protocol="other-v1"))
        assert done.returncode == 2, done.stdout
        assert "protocol mismatch" in done.stderr, done.stderr

        # A code change confounds the result unless re-baselined.
        done = run(root,
                   scorecard(run_id="base", code="c1"),
                   scorecard(run_id="cand", model="m2", code="c2"))
        assert done.returncode == 2, done.stdout
        assert "code_sha differs" in done.stderr, done.stderr

        # Nothing moved means there is nothing to compare.
        done = run(root, scorecard(run_id="base"), scorecard(run_id="cand"))
        assert done.returncode == 2, done.stdout
        assert "nothing moved" in done.stderr, done.stderr

        # A broken scorecard is an error, never a pass.
        bad = write(root, "bad.json", {"schema_version": 1})
        good = write(root, "good.json", scorecard(run_id="cand"))
        done = subprocess.run(
            [sys.executable, str(COMPARE), "--baseline", str(bad),
             "--candidate", str(good)], check=False, capture_output=True, text=True)
        assert done.returncode == 2, done.stdout

        # Missing file is an error, never a skip.
        done = subprocess.run(
            [sys.executable, str(COMPARE), "--baseline", str(root / "absent.json"),
             "--candidate", str(good)], check=False, capture_output=True, text=True)
        assert done.returncode == 2, done.stdout

        # Nonsense budgets are rejected rather than silently widening the gate.
        done = run(root,
                   scorecard(run_id="base"), scorecard(run_id="cand", model="m2"),
                   "--latency-budget", "0")
        assert done.returncode == 2, done.stdout

    # The run step must at least be importable and refuse a missing input; the
    # full pipeline needs a trained model and a corpus and is exercised by the
    # workflow, not here.
    done = subprocess.run(
        [sys.executable, str(RUN), "--runner", str(ROOT / "absent-runner"),
         "--model", str(ROOT / "absent.kwm"), "--keywords", str(ROOT / "absent.kwk"),
         "--references", str(ROOT / "absent.jsonl"), "--work-dir", str(ROOT / "build"),
         "--output", str(ROOT / "build" / "out.json"), "--variable", "model"],
        check=False, capture_output=True, text=True)
    assert done.returncode == 2, done.stdout
    assert "not found" in done.stderr, done.stderr


    # The workflow must be executable from protected main without bypassing the
    # repository ruleset, and its default model must be the Git-registry tuple.
    workflow = (ROOT / ".github/workflows/dataset-driven-iteration.yml").read_text(
        encoding="utf-8"
    )
    assert "bash governance/require_current_main.sh" in workflow
    assert "pull-requests: write" in workflow
    assert "persist-credentials: false" in workflow
    assert "models/registry/$tag/xiaowo-model.kwm" in workflow
    assert "models/registry/$tag/xiaowo-keywords.kwk" in workflow
    assert "tools/verify_model_registry.py" in workflow
    assert 'branch="dataset-iteration/$run_id"' in workflow
    assert "gh pr create" in workflow
    assert "\n          git push\n" not in workflow

    print("test_dataset_iteration: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
