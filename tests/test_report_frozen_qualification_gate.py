from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "report_frozen_qualification_gate.py"

# The limits the real frozen qualification path runs with
# (configs/training/xiaowo.torch-domain.json).
GATES = {"max_frr": 0.0, "max_far_per_hour": 0.0, "max_p95_latency_ms": 800.0, "max_far_frr": 0.0}


def split_row(qualified: bool, frr: float, far: float, latency: float, far_frr: float) -> dict:
    return {
        "qualified": qualified,
        "metrics": {"frr": frr, "far_per_hour": far, "p95_post_end_latency_ms": latency},
        "domains": {"domains": {"distance:far": {"frr": far_frr}}},
    }


def summary(qualified: bool, rows: dict[str, dict]) -> dict:
    return {"schema_version": 1, "qualified": qualified, "splits": rows}


def write(root: pathlib.Path, name: str, payload: dict) -> pathlib.Path:
    path = root / f"{name}"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run(root: pathlib.Path, stages: list[str], *extra: str) -> subprocess.CompletedProcess:
    gates = root / "gates.json"
    gates.write_text(json.dumps({"domain_gates": GATES}) + "\n", encoding="utf-8")
    args = [sys.executable, str(TOOL), "--gates", str(gates)]
    for stage in stages:
        args += ["--stage", stage]
    args += list(extra)
    return subprocess.run(args, check=False, capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        clean = write(root, "clean.json", summary(True, {
            "calibration": split_row(True, 0.0, 0.0, 120.0, 0.0),
            "test": split_row(True, 0.0, 0.0, 130.0, 0.0),
            "qualification": split_row(True, 0.0, 0.0, 140.0, 0.0),
        }))
        # The failure mode the repository has actually hit: the model misses
        # wakes at distance, so one split's far-domain FRR is non-zero.
        far_miss = write(root, "far.json", summary(False, {
            "calibration": split_row(True, 0.0, 0.0, 120.0, 0.0),
            "test": split_row(False, 0.0, 0.0, 130.0, 0.25),
        }))
        # The degenerate all-reject point: FAR is zero but every wake is missed.
        all_reject = write(root, "reject.json", summary(False, {
            "calibration": split_row(False, 1.0, 0.0, 0.0, 1.0),
        }))

        spec = lambda path, code: f"{{name}}:{path}:{code}"  # noqa: E731

        # All three stages clean -> pass.
        done = run(root, [
            spec(clean, "0").format(name="fresh"),
            spec(clean, "0").format(name="shadow"),
            spec(clean, "0").format(name="formal"),
        ])
        assert done.returncode == 0, done.stderr

        # A failing stage must name the split, the metric, the value and the limit.
        done = run(root, [
            spec(far_miss, "1").format(name="fresh"),
            spec(clean, "").format(name="shadow"),
            spec(clean, "").format(name="formal"),
        ])
        assert done.returncode == 1, "a failing fresh stage must fail the gate"
        assert "distance:far.frr=0.25 exceeds max_far_frr=0" in done.stderr, done.stderr
        assert "test:" in done.stderr, "the diagnosis must name the split"
        # Downstream stages must be reported as not run, not as failed.
        assert "shadow: did-not-run" in done.stderr, done.stderr

        # The degenerate point must be legible as a recall failure, not a FAR failure.
        done = run(root, [
            spec(all_reject, "1").format(name="fresh"),
            spec(clean, "").format(name="shadow"),
        ])
        assert done.returncode == 1
        assert "frr=1 exceeds max_frr=0" in done.stderr, done.stderr
        assert "far_per_hour" not in done.stderr.split("frr=1")[1].split("\n")[0]

        # A summary that claims a pass while its metrics violate the gate must
        # fail, and say why: trusting the flag would make the gate decorative.
        lying = write(root, "lying.json", summary(True, {
            "test": split_row(True, 0.5, 0.0, 10.0, 0.0),
        }))
        done = run(root, [spec(lying, "0").format(name="fresh")])
        assert done.returncode == 1, "a self-contradicting summary must not pass"
        assert "claims a pass but its own metrics violate" in done.stderr, done.stderr

        # A stage that succeeded but left no summary is an error, not a pass.
        done = run(root, ["fresh::0"])
        assert done.returncode == 1, "a stage with no summary must fail closed"
        done = run(root, [f"fresh:{root / 'absent.json'}:0"])
        assert done.returncode == 1, "a missing summary file must fail closed"

        # A non-zero exit with a missing summary still fails, and says which stage.
        done = run(root, [f"fresh:{root / 'absent.json'}:1"])
        assert done.returncode == 1
        assert "fresh: failed" in done.stderr, done.stderr

        # Malformed input is an error, never a silent pass.
        bad = root / "bad.json"
        bad.write_text("{not json\n", encoding="utf-8")
        done = run(root, [f"fresh:{bad}:0"])
        assert done.returncode == 1, "an unreadable summary must fail closed"

        # Without --gates the flag alone still decides, so the tool degrades
        # instead of breaking when the config cannot be located.
        plain = subprocess.run(
            [sys.executable, str(TOOL), "--stage", f"fresh:{clean}:0",
             "--stage", f"shadow:{far_miss}:0"],
            check=False, capture_output=True, text=True,
        )
        assert plain.returncode == 1, plain.stdout
        assert "qualified=false" in plain.stderr, plain.stderr

        # Duplicate stage names would let one stage masquerade as another.
        done = run(root, [spec(clean, "0").format(name="fresh")] * 2)
        assert done.returncode == 2, done.stderr

        # A malformed stage spec is an error, not a stage that silently passes.
        done = run(root, ["fresh"])
        assert done.returncode == 2, done.stderr

        # The markdown rendering is what lands in the job summary; it must carry
        # the findings rather than only the status.
        out = root / "step-summary.md"
        done = run(root, [spec(far_miss, "1").format(name="fresh"),
                          spec(clean, "").format(name="shadow")],
                   "--step-summary", str(out))
        assert done.returncode == 1
        text = out.read_text(encoding="utf-8")
        assert "max_far_frr" in text, text
        assert "did-not-run" in text, text

    print("test_report_frozen_qualification_gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
