from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_bench_signal.py"

# Real `kws_bench` output from a hosted Release build, pinned so the fixture is
# production-shaped rather than invented.
BASELINE = """\
case=baseline frontend=logmel keywords=1 trie_nodes=5 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033764 rtf=0.003376 us_per_audio_s=3376.4
case=baseline-gated frontend=logmel keywords=1 trie_nodes=5 geometry=32x48x420 gate=enabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033765 rtf=0.003377 us_per_audio_s=3376.5
case=product frontend=logmel keywords=4 trie_nodes=8 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033800 rtf=0.003380 us_per_audio_s=3380.0
case=pcen-product frontend=pcen-lite keywords=4 trie_nodes=8 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033859 rtf=0.003386 us_per_audio_s=3385.9
case=worst-case frontend=pcen-lite keywords=16 trie_nodes=23 geometry=32x48x420 gate=disabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033982 rtf=0.003398 us_per_audio_s=3398.2
case=worst-case-gated frontend=pcen-lite keywords=16 trie_nodes=23 geometry=32x48x420 gate=enabled model_bytes=25944 engine_bytes=22056 estimated_macs_per_frame=24000 audio_s=10 cpu_s=0.033911 rtf=0.003391 us_per_audio_s=3391.1
"""


def require(condition: bool, message: str = "contract expectation failed") -> None:
    # Unlike assert, these checks (and their subprocess calls) survive -O/-OO.
    if not condition:
        raise AssertionError(message)


def run_result(root: pathlib.Path, text: str, *extra: str) -> subprocess.CompletedProcess[str]:
    payload = root / "bench.txt"
    payload.write_text(text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
         str(CHECKER), "--input", str(payload), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def run(root: pathlib.Path, text: str, *extra: str) -> int:
    return run_result(root, text, *extra).returncode


def rewrite_last_field(text: str, replacement: str) -> str:
    """Return the fixture with its final line's rtf replaced."""
    lines = text.strip().splitlines()
    parts = lines[-1].split()
    parts = [replacement if item.startswith("rtf=") else item for item in parts]
    lines[-1] = " ".join(parts)
    return "\n".join(lines) + "\n"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        # The measured production output is the passing case.
        require(run(root, BASELINE) == 0, "the shipped bench output must satisfy the contract")

        # A legal model update moves every case together and must still pass,
        # because structural fields are checked by agreement rather than value.
        moved = BASELINE.replace("geometry=32x48x420", "geometry=40x64x512").replace(
            "model_bytes=25944", "model_bytes=31000"
        )
        require(run(root, moved) == 0, "a uniform model update must not fail the contract")

        # The six producer names are a set contract, not a row-count heuristic.
        lines = BASELINE.strip().splitlines()
        names = [line.split()[0].removeprefix("case=") for line in lines]
        for index, name in enumerate(names):
            dropped = "\n".join(lines[:index] + lines[index + 1:]) + "\n"
            result = run_result(root, dropped)
            require(result.returncode == 1, f"missing {name} must fail")
            require(result.stdout == "", f"missing {name} must not print success")
            require(result.stderr == f"missing required case names: {name}\n")

        first_two = "\n".join(lines[:2]) + "\n"
        result = run_result(root, first_two)
        require(result.returncode == 1)
        require(result.stderr == "missing required case names: pcen-product, product, "
                "worst-case, worst-case-gated\n", "missing names must be sorted")
        renamed = BASELINE.replace("case=worst-case ", "case=other ")
        require(run(root, renamed) == 1, "six rows cannot hide a renamed required case")
        extra_row = lines[0].replace("case=baseline ", "case=extra ") + "\n"
        replaced = "\n".join(lines[:-1]) + "\n" + extra_row
        require(run(root, replaced) == 1, "an unrelated extra cannot replace a required case")
        require(run(root, "\n".join(reversed(lines)) + "\n") == 0,
                "case order must not matter")
        require(run(root, BASELINE + extra_row) == 0, "valid unique extra cases are allowed")
        require(run(root, BASELINE + extra_row, "--min-cases", "7") == 0)
        require(run(root, first_two, "--min-cases", "1") == 1,
                "a lower count floor must not waive required names")
        require(run(root, BASELINE, "--min-cases", "7") == 1,
                "the additional count floor still applies")

        # Timing must be measured, not zeroed: a bench that stops measuring
        # would otherwise stay green.
        require(run(root, rewrite_last_field(BASELINE, "rtf=0.000000")) == 1)

        # An order-of-magnitude regression must not hide behind runner noise.
        require(run(root, rewrite_last_field(BASELINE, "rtf=1.000000")) == 1)

        # A non-numeric measurement is a broken bench, not a slow one.
        require(run(root, rewrite_last_field(BASELINE, "rtf=abc")) == 1)

        # A missing field means the contract can no longer be evaluated.
        stripped = "\n".join(
            " ".join(item for item in line.split() if not item.startswith("cpu_s="))
            for line in BASELINE.strip().splitlines()
        ) + "\n"
        require(run(root, stripped) == 1)

        # One case drifting structurally must fail even though its timing is
        # fine -- this is what catches a silently changed model or geometry.
        drifted = BASELINE.replace(
            "case=worst-case frontend=pcen-lite keywords=16",
            "case=worst-case frontend=pcen-lite keywords=16",
        )
        lines = drifted.strip().splitlines()
        lines[-1] = lines[-1].replace("geometry=32x48x420", "geometry=32x48x999")
        require(run(root, "\n".join(lines) + "\n") == 1)

        # A single case leaves nothing to compare against.
        require(run(root, BASELINE.strip().splitlines()[0] + "\n") == 1)

        # Duplicate case names would let one case masquerade as two.
        require(run(root, BASELINE + BASELINE.strip().splitlines()[0] + "\n") == 1)

        # Trailing junk is not silently ignored.
        require(run(root, BASELINE + "garbage\n") == 1)

        # An empty capture is not a pass: a vacuous result must be an error.
        require(run(root, "") == 2, "empty bench output must not pass")

        # Invalid budgets must fail closed rather than disable the timing gate.
        # Equals form also ensures negative infinity reaches float validation.
        for ceiling in ("nan", "inf", "+inf", "-inf", "1e309", "0", "-1"):
            result = run_result(root, BASELINE, f"--max-rtf={ceiling}")
            require(result.returncode == 2, f"invalid ceiling {ceiling} must be an error")
            require(result.stdout == "error: --max-rtf must be positive and finite\n",
                    f"invalid ceiling {ceiling} must report the budget error, not success")
            require(result.stderr == "", f"invalid ceiling {ceiling} must use the error path")

        # Positive finite ceilings retain default, inclusive-boundary, and
        # restrictive behavior; a rejected measurement is not a CLI error.
        for options, ceiling in (((), "0.1"), (("--max-rtf=0.004",), "0.004"),
                                 (("--max-rtf=0.003398",), "0.003398")):
            result = run_result(root, BASELINE, *options)
            require(result.returncode == 0, f"valid ceiling {ceiling} must pass")
            require(result.stdout == "check_bench_signal: ok cases=6 "
                    f"max_rtf=0.003398 ceiling={ceiling}\n")
            require(result.stderr == "")
        result = run_result(root, BASELINE, "--max-rtf=0.003")
        require(result.returncode == 1, "a finite restrictive ceiling must still reject")
        require(result.stdout == "", "a timing failure must not print success")
        expected_errors = []
        for line in BASELINE.strip().splitlines():
            row = dict(item.split("=", 1) for item in line.split())
            expected_errors.append(
                f"{row['case']}: rtf {float(row['rtf'])} exceeds --max-rtf 0.003\n"
            )
        require(result.stderr == "".join(expected_errors))

        for floor in ("0", "-1"):
            result = run_result(root, BASELINE, f"--min-cases={floor}")
            require(result.returncode == 2, "an invalid count floor must remain an error")
            require(result.stdout == "error: --min-cases must be at least 1\n")
            require(result.stderr == "")

    print("test_bench_signal: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
