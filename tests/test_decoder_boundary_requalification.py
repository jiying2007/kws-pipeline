from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from requalify_decoder_boundary_grid import (  # noqa: E402
    compose_point,
    validate_source_grid,
)


def expect_failure(call, needle: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected failure containing {needle!r}")


def source_grid(pack_sha: str) -> dict:
    return {
        "schema_version": 1,
        "evidence_class": "decoder-policy-posterior-replay-grid-development-v1",
        "development_only": True,
        "protected_evidence_used": False,
        "selection_feedback_allowed": False,
        "model_sha256": "m" * 64,
        "operating_grid": [
            {
                "blank_retention": 0.70,
                "fuzzy_child_cost_log": -8.25,
                "strict": True,
                "calibrated_pack_sha256": pack_sha,
                "calibrated_thresholds": {"1": 0.6, "2": 0.6},
            },
            {
                "blank_retention": 0.85,
                "fuzzy_child_cost_log": -4.0,
                "strict": False,
                "calibrated_pack_sha256": pack_sha,
                "calibrated_thresholds": {"1": 0.6, "2": 0.6},
            },
        ],
    }


def positive_metrics(*, false_rejects: int = 0) -> dict:
    return {
        "recordings": 2,
        "expected": 2,
        "false_rejects": false_rejects,
        "false_accepts": 0,
        "negative_recording_audio_hours": 0.0,
        "negative_recording_false_accepts": 0,
        "negative_recording_far_per_hour": 0.0,
    }


def negative_metrics(*, false_accepts: int = 0) -> dict:
    return {
        "recordings": 2,
        "expected": 0,
        "false_rejects": 0,
        "false_accepts": false_accepts,
        "negative_recording_audio_hours": 0.01,
        "negative_recording_false_accepts": false_accepts,
        "negative_recording_far_per_hour": float(false_accepts) * 100.0,
    }


def main() -> int:
    pack_sha = "p" * 64
    grid = source_grid(pack_sha)
    rows = validate_source_grid(
        grid,
        model_sha256="m" * 64,
        pack_sha256=pack_sha,
    )
    assert len(rows) == 2

    protected = source_grid(pack_sha)
    protected["protected_evidence_used"] = True
    expect_failure(
        lambda: validate_source_grid(
            protected,
            model_sha256="m" * 64,
            pack_sha256=pack_sha,
        ),
        "must not use protected evidence",
    )

    feedback = source_grid(pack_sha)
    feedback["selection_feedback_allowed"] = True
    expect_failure(
        lambda: validate_source_grid(
            feedback,
            model_sha256="m" * 64,
            pack_sha256=pack_sha,
        ),
        "forbid selection feedback",
    )

    mixed = source_grid(pack_sha)
    mixed["operating_grid"][1]["calibrated_pack_sha256"] = "q" * 64
    expect_failure(
        lambda: validate_source_grid(
            mixed,
            model_sha256="m" * 64,
            pack_sha256=pack_sha,
        ),
        "must exactly match every source grid point",
    )

    qualified = compose_point(
        rows[0],
        within_word_metrics=positive_metrics(),
        cross_boundary_metrics=negative_metrics(),
    )
    assert qualified["source_strict"] is True
    assert qualified["boundary"]["acceptance"]["qualified"] is True
    assert qualified["joint_strict"] is True

    source_failed = compose_point(
        rows[1],
        within_word_metrics=positive_metrics(),
        cross_boundary_metrics=negative_metrics(),
    )
    assert source_failed["source_strict"] is False
    assert source_failed["boundary"]["acceptance"]["qualified"] is True
    assert source_failed["joint_strict"] is False

    boundary_failed = compose_point(
        rows[0],
        within_word_metrics=positive_metrics(false_rejects=1),
        cross_boundary_metrics=negative_metrics(),
    )
    assert boundary_failed["source_strict"] is True
    assert boundary_failed["boundary"]["acceptance"]["qualified"] is False
    assert boundary_failed["joint_strict"] is False

    print("test_decoder_boundary_requalification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
