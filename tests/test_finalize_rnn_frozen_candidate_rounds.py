from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import finalize_rnn_frozen_candidate as finalize  # noqa: E402

# finalize_rnn_frozen_candidate is called by three workflows and had no test at
# all. This one pins round_gate_evidence, which turns development records into
# the round_gates carried by the frozen candidate.
#
# The GRU twin rejects a boolean or a float round. This side compared with a
# bare !=, accepted both, and then wrote the loop index rather than the input
# value into the result, so a malformed round left no trace in the output --
# the record was silently corrected instead of rejected.


def row(round_value, calibration=True, test=True) -> dict:
    return {"round": round_value, "calibration_gate": calibration, "test_gate": test}


def expect(needle: str, records) -> None:
    try:
        finalize.round_gate_evidence(records)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def main() -> int:
    # Happy path: contiguous rounds survive with their gate values intact.
    result = finalize.round_gate_evidence([row(0), row(1), row(2, False, True)])
    assert result == [
        {"round": 0, "calibration_gate": True, "test_gate": True},
        {"round": 1, "calibration_gate": True, "test_gate": True},
        {"round": 2, "calibration_gate": False, "test_gate": True},
    ], result

    # Anti-vacuity: empty and non-list inputs are rejected, not passed through.
    expect("must be a non-empty list", [])
    expect("must be a non-empty list", "012")

    # Three shape problems, each named separately rather than folded into one
    # round comparison.
    expect("must be an object", [row(0), "not-a-row"])
    expect("must be an integer", [row(0), row(True)])
    expect("must be an integer", [row(0), row(1.0)])
    expect("must be an integer", [row(0), {"calibration_gate": True, "test_gate": True}])

    # A gap is a different failure from a bad type, and keeps the message that
    # predates the split.
    expect("round evidence is malformed", [row(1)])

    print("test_finalize_rnn_frozen_candidate_rounds: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
