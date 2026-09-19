from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools" / "compare_dataset_iterations.py"
RUN = ROOT / "tools" / "run_dataset_iteration.py"

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
        [sys.executable, str(COMPARE), "--baseline", str(before),
         "--candidate", str(after), "--far-budget", "20.0", *extra],
        check=False, capture_output=True, text=True,
    )


def main() -> int:
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

    print("test_dataset_iteration: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
