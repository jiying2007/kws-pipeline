from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gru_development_gate as gru  # noqa: E402
import rnn_development_gate as rnn  # noqa: E402

# terminal_strict_streak decides whether a development candidate is stable
# enough to qualify. It used bool(record.get("calibration_gate")), and
# bool("false"), bool(0) and bool("no") are all True -- so a round that
# recorded a failure counted towards the streak.
#
# The finalizers had the same function over the same records and were fixed
# first; these two are the copies that remained.
#
# An absent gate is handled differently on purpose: it means "did not pass",
# which is fail-closed, so it is not an error. Only a present value that is
# not a boolean is rejected.

TWINS = (("RNN ", rnn), ("", gru))


def expect(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def passing(round_number: int) -> dict:
    return {"round": round_number, "calibration_gate": True, "test_gate": True}


def main() -> int:
    for prefix, mod in TWINS:
        assert mod.terminal_strict_streak([passing(0), passing(1), passing(2)]) == 3
        trailing = [passing(0), passing(1), dict(passing(2), test_gate=False)]
        assert mod.terminal_strict_streak(trailing) == 0, prefix
        reset = [passing(0), dict(passing(1), calibration_gate=False), passing(2)]
        assert mod.terminal_strict_streak(reset) == 1, prefix

        # A round with no recorded gate did not pass: not an error, not a pass.
        assert mod.terminal_strict_streak([passing(0), {"round": 1}]) == 0, prefix
        assert mod.gate_bool({}, "calibration_gate") is False, prefix

        # Every one of these is bool()-true, and none of them is a pass.
        for bad in ("false", "0", 0, 1, "no", [], {}):
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

        # The helper is the only way in, so the same rule holds for it.
        assert mod.gate_bool({"calibration_gate": False}, "calibration_gate") is False
        assert mod.gate_bool({"calibration_gate": True}, "calibration_gate") is True
        expect(
            f"{prefix}development record calibration_gate must be a boolean",
            lambda m=mod: m.gate_bool({"calibration_gate": "false"}, "calibration_gate"),
        )

    print("test_development_gate_booleans: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
