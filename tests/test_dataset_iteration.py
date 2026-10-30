from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools" / "compare_dataset_iterations.py"
RUN = ROOT / "tools" / "run_dataset_iteration.py"
sys.path.insert(0, str(ROOT / "tools"))
from run_dataset_iteration import dataset_identity  # noqa: E402

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
    # Frozen output from the original path-only implementation and eight literal
    # PCM samples. Alias normalization must not silently change existing IDs.
    PATH_ONLY_ID = "47be6a9d292ea745056a302e773906a725c5b2eb366535c33fb07df35d2534d1"

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


def main() -> int:
    strict = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromTestCase(RequiredMetricTests),
            unittest.defaultTestLoader.loadTestsFromTestCase(DatasetIdentityAliasTests),
        ])
    )
    if strict.testsRun != 9 or strict.skipped or not strict.wasSuccessful():
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
