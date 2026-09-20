from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import finalize_gru_frozen_candidate as gru  # noqa: E402
import finalize_rnn_frozen_candidate as rnn  # noqa: E402

# Both finalizers used to read development gates with bool(row.get(...)).
# bool("false"), bool(0) and bool("no") are all True, so a round that recorded
# a failure either counted towards the strict-pass streak, got picked as the
# stable candidate, or was written into the frozen manifest as a pass -- three
# places where recorded evidence was silently upgraded.
#
# The producers write real booleans, so both sides now require one. Each case
# below names this side's own wording, so neither module can be swapped for the
# other without turning this file red.

TWINS = (("RNN ", rnn), ("", gru))


def expect(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def passing(round_number: int) -> dict:
    return {
        "round": round_number,
        "calibration_gate": True,
        "test_gate": True,
        "score": 0.9 - round_number * 0.01,
        "frontend": "logmel",
    }


def main() -> int:
    for prefix, mod in TWINS:
        # The streak counts real passes and resets on a real failure.
        assert mod.terminal_strict_streak([passing(0), passing(1), passing(2)]) == 3
        # A real failure resets the streak; the trailing pass starts a new one.
        reset = [passing(0), dict(passing(1), calibration_gate=False), passing(2)]
        assert mod.terminal_strict_streak(reset) == 1, prefix
        trailing = [passing(0), passing(1), dict(passing(2), test_gate=False)]
        assert mod.terminal_strict_streak(trailing) == 0, prefix

        # ...but a failure recorded as a string is not a pass, and must not be
        # counted as one. Every one of these is bool()-true.
        for bad in ("false", "0", 0, 1, "no", None, [], {}):
            records = [passing(0), dict(passing(1), calibration_gate=bad)]
            expect(
                f"{prefix}development record calibration_gate must be a boolean",
                lambda m=mod, r=records: m.terminal_strict_streak(r),
            )
        for bad in ("false", "0", 0, 1):
            records = [passing(0), dict(passing(1), test_gate=bad)]
            expect(
                f"{prefix}development record test_gate must be a boolean",
                lambda m=mod, r=records: m.terminal_strict_streak(r),
            )

        # Selection applies the same rule: a record whose gates did not pass
        # must not become the frozen candidate.
        manifest = {
            "selection_policy": mod.SELECTION_POLICY,
            "selected_round": 0,
            "selected_score": 0.9,
            "records": [passing(0), passing(1)],
        }
        assert mod.select_record(dict(manifest, records=[passing(0)]))["round"] == 0, prefix

        for key in ("calibration_gate", "test_gate"):
            records = [passing(0), dict(passing(1), **{key: "false"})]
            expect(
                f"{prefix}development record {key} must be a boolean",
                lambda m=mod, r=records: m.select_record(
                    {
                        "selection_policy": m.SELECTION_POLICY,
                        "selected_round": 0,
                        "selected_score": 0.9,
                        "records": r,
                    }
                ),
            )

        # The manifest writer reads through the same helper, so a frozen
        # manifest can no longer claim a gate passed when it did not.
        assert mod.gate_bool({"calibration_gate": False}, "calibration_gate") is False
        assert mod.gate_bool({"calibration_gate": True}, "calibration_gate") is True
        expect(
            f"{prefix}development record calibration_gate must be a boolean",
            lambda m=mod: m.gate_bool({"calibration_gate": "false"}, "calibration_gate"),
        )

    print("test_finalize_gate_booleans: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
