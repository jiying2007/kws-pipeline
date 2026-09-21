#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gru_development_gate as gru  # noqa: E402
import rnn_development_gate as rnn  # noqa: E402


def expect_value_error(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def passing(round_number: int) -> dict:
    return {
        "round": round_number,
        "calibration_gate": True,
        "test_gate": True,
    }


def test_development_gate_types() -> None:
    for prefix, module in (("RNN ", rnn), ("", gru)):
        assert module.terminal_strict_streak([passing(0), passing(1)]) == 2
        assert module.terminal_strict_streak(
            [passing(0), dict(passing(1), calibration_gate=False)]
        ) == 0
        assert module.terminal_strict_streak([passing(0), {"round": 1}]) == 0
        for key in ("calibration_gate", "test_gate"):
            for bad in ("false", "0", "no", 0, 1, [], {}):
                row = dict(passing(1))
                row[key] = bad
                expect_value_error(
                    f"{prefix}development record {key} must be a boolean",
                    lambda m=module, r=[passing(0), row]: m.terminal_strict_streak(r),
                )


def test_recorded_verdict_sources() -> None:
    contracts = {
        "training/adversarial_refinement.py": {
            "forbidden": (
                'bool(row.get("calibration_gate"))',
                'bool(row.get("test_gate"))',
                'bool(manifest.get("development_qualified"))',
                'bool(source.get("calibration_gate"))',
                'bool(source.get("test_gate"))',
                'bool(adversarial.get("formal_qualification_used", True))',
                'bool(failure.get("formal_qualification_used", True))',
                'bool(failure.get("development_source_wav_bytes_copied", True))',
            ),
            "required": (
                'row.get("calibration_gate") is True',
                'row.get("test_gate") is True',
                'manifest.get("development_qualified") is True',
            ),
        },
        "training/finalize_domain_candidate.py": {
            "forbidden": (
                'bool(row.get("calibration_gate"))',
                'bool(row.get("test_gate"))',
            ),
            "required": (
                'row.get("calibration_gate") is True',
                'row.get("test_gate") is True',
            ),
        },
        "training/render_qualification_holdout.py": {
            "forbidden": (
                'bool(row.get("calibration_gate"))',
                'bool(row.get("test_gate"))',
                'bool(manifest.get("development_qualified"))',
            ),
            "required": (
                'manifest.get("development_qualified") is not True',
            ),
        },
        "training/render_qualification_guarded.py": {
            "forbidden": (
                'bool(row.get("calibration_gate"))',
                'bool(row.get("test_gate"))',
                'bool(selection.get("adversarial_refinement_used"))',
                'bool(refinement.get("qualified"))',
                'bool(shadow.get("qualified"))',
                'bool(row.get("qualified"))',
                'bool(row.get("runtime_qualified"))',
                'bool(row.get("surrogate_separation_qualified"))',
            ),
            "required": (
                'selection.get("adversarial_refinement_used") is not True',
                'refinement.get("qualified") is not True',
                'shadow.get("qualified") is not True',
            ),
        },
        "training/shadow_qualification.py": {
            "forbidden": (
                'bool(row.get("calibration_gate"))',
                'bool(row.get("test_gate"))',
                'bool(row["runtime_qualified"])',
                'bool(row["surrogate_separation_qualified"])',
                'not bool(row["qualified"])',
            ),
            "required": (
                'row.get("calibration_gate") is True',
                'row.get("test_gate") is True',
            ),
        },
        "training/iterate_domain.py": {
            "forbidden": (
                'bool(record.get("calibration_gate"))',
                'bool(record.get("test_gate"))',
            ),
            "required": (
                'record.get("calibration_gate") is True',
                'record.get("test_gate") is True',
            ),
        },
        "training/development_resume.py": {
            "forbidden": ('bool(value.get("complete"))',),
            "required": ('value.get("complete") is True',),
        },
        "training/qualification_failure_replay.py": {
            "forbidden": (
                'bool(previous.get("formal_qualification_used", True))',
                'bool(previous.get("development_source_wav_bytes_copied", True))',
            ),
            "required": (
                'previous.get("formal_qualification_used", True) is not False',
                'previous.get("development_source_wav_bytes_copied", True) is not False',
            ),
        },
        "training/iterate_rnn_development.py": {
            "forbidden": (
                'bool(record.get("calibration_gate"))',
                'bool(record.get("test_gate"))',
                'bool(manifest.get("development_qualified"))',
            ),
            "required": (
                'record.get("calibration_gate") is True',
                'record.get("test_gate") is True',
                'manifest.get("development_qualified") is True',
            ),
        },
        "training/iterate_gru_development.py": {
            "forbidden": (
                'bool(record.get("calibration_gate"))',
                'bool(record.get("test_gate"))',
                'bool(manifest.get("development_qualified"))',
            ),
            "required": (
                'record.get("calibration_gate") is True',
                'record.get("test_gate") is True',
                'manifest.get("development_qualified") is True',
            ),
        },
        "tools/gru_development_gate.py": {
            "forbidden": (
                'bool(robustness.get("qualified"))',
                'bool(record.get("calibration_gate"))',
                'bool(record.get("test_gate"))',
            ),
            "required": (
                'if not isinstance(robustness_qualified, bool):',
                'gate_bool(record, "calibration_gate")',
                'gate_bool(record, "test_gate")',
            ),
        },
        "tools/rnn_development_gate.py": {
            "forbidden": (
                'bool(robustness.get("qualified"))',
                'bool(record.get("calibration_gate"))',
                'bool(record.get("test_gate"))',
            ),
            "required": (
                'if not isinstance(robustness_qualified, bool):',
                'gate_bool(record, "calibration_gate")',
                'gate_bool(record, "test_gate")',
            ),
        },
    }

    for relative, contract in contracts.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in contract["forbidden"]:
            assert needle not in text, f"{relative}: recorded verdict coercion remains: {needle}"
        for needle in contract["required"]:
            assert needle in text, f"{relative}: strict boolean contract missing: {needle}"


def main() -> int:
    test_development_gate_types()
    test_recorded_verdict_sources()
    print("test_recorded_verdict_booleans: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
