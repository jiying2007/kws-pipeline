#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from adversarial_refinement import (  # noqa: E402
    PRESSURE_ASSIGNMENT_POLICY,
    REFINEMENT_SOURCE_POLICY,
    WAKE_BALANCE_POLICY,
    build_refinement_focus_rows,
    derive_refinement_wake_balance,
    select_refinement_source,
)

from evaluate_refinement_eligibility import (  # noqa: E402
    POLICY as REFINEMENT_ELIGIBILITY_POLICY,
    STAGE_COLLAPSE,
    STAGE_POLICY,
    STAGE_SIGNAL,
    STAGE_STRICT,
    evaluate_refinement_eligibility,
)


def keyword_metrics(frr: float) -> dict:
    expected = 32
    false_rejects = int(round(frr * expected))
    matched = expected - false_rejects
    return {
        "expected": expected,
        "matched": matched,
        "false_rejects": false_rejects,
        "frr": frr,
    }


def row(
    round_index: int,
    *,
    cal_frr: float,
    test_frr: float,
    cal_far_frr: float,
    test_far_frr: float,
    cal_far: float,
    test_far: float,
    cal_kw1_frr: float,
    cal_kw2_frr: float,
    test_kw1_frr: float,
    test_kw2_frr: float,
    score: float,
    strict: bool = False,
) -> dict:
    return {
        "round": round_index,
        "frontend": "logmel",
        "score": score,
        "checkpoint": f"/tmp/round-{round_index}.pt",
        "calibration": {
            "frr": cal_frr,
            "far_per_hour": cal_far,
            "per_keyword": {
                "1": keyword_metrics(cal_kw1_frr),
                "2": keyword_metrics(cal_kw2_frr),
            },
        },
        "test": {
            "frr": test_frr,
            "far_per_hour": test_far,
            "per_keyword": {
                "1": keyword_metrics(test_kw1_frr),
                "2": keyword_metrics(test_kw2_frr),
            },
        },
        "calibration_domains": {
            "domains": {"distance:far": {"frr": cal_far_frr}}
        },
        "test_domains": {
            "domains": {"distance:far": {"frr": test_far_frr}}
        },
        "calibration_gate": strict,
        "test_gate": strict,
    }


def main() -> int:
    # Exact rounded operating-point shape from governed run 35447750283.
    # Round 2 wins the old zero-gate objective mainly by rejecting almost all
    # wakes. Refinement should instead start from the development checkpoint
    # that preserves the best worst-case recall, including every shipping
    # keyword, with FAR used only after base/far-field/per-keyword recall.
    records = [
        row(
            0,
            cal_frr=0.84375,
            test_frr=0.84375,
            cal_far_frr=0.9230769231,
            test_far_frr=0.6153846154,
            cal_far=928.5468,
            test_far=1066.4509,
            cal_kw1_frr=0.90625,
            cal_kw2_frr=0.78125,
            test_kw1_frr=0.875,
            test_kw2_frr=0.8125,
            score=235595.94,
        ),
        row(
            1,
            cal_frr=0.71875,
            test_frr=0.90625,
            cal_far_frr=0.6153846154,
            test_far_frr=0.8461538462,
            cal_far=790.6438,
            test_far=750.4654,
            cal_kw1_frr=0.8125,
            cal_kw2_frr=0.625,
            test_kw1_frr=0.84375,
            test_kw2_frr=0.96875,
            score=188598.69,
        ),
        row(
            2,
            cal_frr=0.96875,
            test_frr=0.984375,
            cal_far_frr=0.9230769231,
            test_far_frr=1.0,
            cal_far=64.3547,
            test_far=98.7455,
            cal_kw1_frr=1.0,
            cal_kw2_frr=0.9375,
            test_kw1_frr=1.0,
            test_kw2_frr=0.96875,
            score=59537.30,
        ),
        row(
            3,
            cal_frr=0.875,
            test_frr=0.90625,
            cal_far_frr=0.8461538462,
            test_far_frr=0.8461538462,
            cal_far=524.0314,
            test_far=533.2254,
            cal_kw1_frr=0.875,
            cal_kw2_frr=0.875,
            test_kw1_frr=0.84375,
            test_kw2_frr=0.96875,
            score=144543.13,
        ),
    ]
    manifest = {
        "development_qualified": False,
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [],
            "qualification_used_for_selection": False,
            "selected_round": None,
            "selected_frontend": None,
            "selected_score": None,
            "objective_fallback_used": True,
            "objective_best_round": 2,
            "objective_best_frontend": "logmel",
            "objective_best_score": 59537.30,
        },
        "records": records,
    }
    selected, policy = select_refinement_source(manifest)
    assert selected["round"] == 0
    assert policy == REFINEMENT_SOURCE_POLICY

    eligibility = evaluate_refinement_eligibility(
        manifest,
        expected_keyword_ids=("1", "2"),
    )
    assert eligibility["policy"] == REFINEMENT_ELIGIBILITY_POLICY
    assert eligibility["development_stage_policy"] == STAGE_POLICY
    assert eligibility["development_stage"] == STAGE_SIGNAL
    assert eligibility["eligible"] is True
    assert eligibility["source_round"] == 0
    assert eligibility["collapsed_keyword_ids"] == []

    # One development split may completely miss a keyword while the other still
    # carries a real recognition signal. This is weak, but it is not a total
    # collapse and refinement is explicitly allowed to repair it.
    weak = row(
        0,
        cal_frr=0.9375,
        test_frr=0.921875,
        cal_far_frr=0.9,
        test_far_frr=0.8,
        cal_far=100.0,
        test_far=120.0,
        cal_kw1_frr=1.0,
        cal_kw2_frr=0.875,
        test_kw1_frr=0.96875,
        test_kw2_frr=0.875,
        score=10.0,
    )
    weak_manifest = {
        "development_qualified": False,
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [],
            "qualification_used_for_selection": False,
            "selected_round": None,
            "selected_frontend": None,
            "selected_score": None,
            "objective_fallback_used": True,
            "objective_best_round": 0,
            "objective_best_frontend": "logmel",
            "objective_best_score": 10.0,
        },
        "records": [weak],
    }
    weak_eligibility = evaluate_refinement_eligibility(
        weak_manifest,
        expected_keyword_ids=("1", "2"),
    )
    assert weak_eligibility["development_stage"] == STAGE_SIGNAL
    assert weak_eligibility["eligible"] is True
    assert weak_eligibility["keyword_signal"]["1"]["calibration"]["matched"] == 0
    assert weak_eligibility["keyword_signal"]["1"]["test"]["matched"] == 1
    assert weak_eligibility["keyword_signal"]["1"]["combined_matched"] == 1

    collapsed = row(
        0,
        cal_frr=0.9375,
        test_frr=0.9375,
        cal_far_frr=0.9,
        test_far_frr=0.9,
        cal_far=80.0,
        test_far=90.0,
        cal_kw1_frr=1.0,
        cal_kw2_frr=0.875,
        test_kw1_frr=1.0,
        test_kw2_frr=0.875,
        score=11.0,
    )
    collapsed_manifest = dict(weak_manifest)
    collapsed_manifest["records"] = [collapsed]
    collapsed_eligibility = evaluate_refinement_eligibility(
        collapsed_manifest,
        expected_keyword_ids=("1", "2"),
    )
    assert collapsed_eligibility["development_stage"] == STAGE_COLLAPSE
    assert collapsed_eligibility["eligible"] is False
    assert collapsed_eligibility["collapsed_keyword_ids"] == ["1"]

    strict = row(
        4,
        cal_frr=0.0,
        test_frr=0.0,
        cal_far_frr=0.0,
        test_far_frr=0.0,
        cal_far=0.0,
        test_far=0.0,
        cal_kw1_frr=0.0,
        cal_kw2_frr=0.0,
        test_kw1_frr=0.0,
        test_kw2_frr=0.0,
        score=0.0,
        strict=True,
    )
    strict_manifest = {
        "development_qualified": True,
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": [4],
            "qualification_used_for_selection": False,
            "selected_round": 4,
            "selected_frontend": "logmel",
            "selected_score": 0.0,
            "objective_fallback_used": False,
        },
        "records": records + [strict],
    }
    selected, policy = select_refinement_source(strict_manifest)
    assert selected["round"] == 4
    assert policy == "strict-development-candidate"
    strict_eligibility = evaluate_refinement_eligibility(
        strict_manifest,
        expected_keyword_ids=("1", "2"),
    )
    assert strict_eligibility["development_stage"] == STAGE_STRICT
    assert strict_eligibility["eligible"] is True
    assert strict_eligibility["source_was_strict"] is True

    with tempfile.TemporaryDirectory(prefix="refinement-wake-balance-") as tmp:
        root = pathlib.Path(tmp)
        tokens = root / "tokens.txt"
        keywords = root / "keywords.tsv"
        tokens.write_text("<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n", encoding="utf-8")
        keywords.write_text(
            "1\t你好小窝\t0.55\tni3 hao3 xiao3 wo1\n"
            "2\t小窝小窝\t0.55\txiao3 wo1 xiao3 wo1\n",
            encoding="utf-8",
        )

        manifests = []
        specs = [
            [
                ("a.wav", "1 2 3 4"),
                ("b.wav", "3 4 3 4"),
                ("c.wav", "1 2 3"),
                ("d.wav", ""),
            ],
            [
                ("e.wav", "1 2 3 4"),
                ("f.wav", "1 2"),
                ("g.wav", "3 4"),
                ("h.wav", "2 3 4"),
            ],
            [
                ("i.wav", "1 3 4"),
                ("j.wav", "4 3 4"),
                ("k.wav", "1 2 4"),
                ("l.wav", "3 1 2 4"),
            ],
        ]
        for index, rows in enumerate(specs):
            path = root / f"manifest-{index}.tsv"
            path.write_text(
                "".join(f"{audio}\t{targets}\n" for audio, targets in rows),
                encoding="utf-8",
            )
            manifests.append(path)

        balance = derive_refinement_wake_balance(
            manifests=manifests,
            tokens=tokens,
            keywords=keywords,
            positive_example_weight=2.0,
        )
        assert balance["policy"] == WAKE_BALANCE_POLICY
        assert balance["schema_version"] == 3
        assert balance["pressure_assignment"] == PRESSURE_ASSIGNMENT_POLICY
        assert balance["explicit_focus_nonwake_rows"] == 0
        assert balance["fallback_edit_distance_nonwake_rows"] == 9
        assert balance["wake_rows"] == 3
        assert balance["wake_rows_by_keyword"] == {"1": 2, "2": 1}
        assert balance["tokenized_nonwake_rows"] == 8
        assert balance["empty_nonwake_rows"] == 1
        assert balance["wake_base_mass"] == 6.0
        assert balance["nonwake_mass"] == 17.0
        assert abs(balance["keyword_balance"]["1"]["assigned_nonwake_mass"] - 12.5) < 1.0e-12
        assert abs(balance["keyword_balance"]["2"]["assigned_nonwake_mass"] - 4.5) < 1.0e-12
        assert abs(balance["wake_keyword_weights"]["1"] - 3.125) < 1.0e-12
        assert abs(balance["wake_keyword_weights"]["2"] - 2.25) < 1.0e-12
        assert abs(balance["effective_wake_mass"] - 17.0) < 1.0e-12
        assert balance["bounded"] is False

        extreme = root / "extreme.tsv"
        extreme.write_text(
            "wake1.wav\t1 2 3 4\n"
            "wake2.wav\t3 4 3 4\n"
            + "".join(f"n{index}.wav\t1 2 3\n" for index in range(100)),
            encoding="utf-8",
        )
        capped = derive_refinement_wake_balance(
            manifests=[extreme],
            tokens=tokens,
            keywords=keywords,
            positive_example_weight=2.0,
        )
        assert capped["wake_keyword_weights"]["1"] == 12.0
        assert capped["wake_keyword_weights"]["2"] == 1.0
        assert capped["keyword_balance"]["1"]["bounded"] is True
        assert capped["keyword_balance"]["2"]["bounded"] is True
        assert capped["bounded"] is True

        # Explicit replay focus is authoritative even when token edit distance
        # points at the other wake word. "hao wo xiao wo ni" is closer to
        # keyword 2 by edit distance but is deliberately a keyword-1 negative
        # in the product replay policy.
        focused = root / "focused.tsv"
        focused.write_text(
            "wake1.wav\t1 2 3 4\n"
            "wake2.wav\t3 4 3 4\n"
            "negative.wav\t2 4 3 4 1\n",
            encoding="utf-8",
        )
        focused_balance = derive_refinement_wake_balance(
            manifests=[focused],
            tokens=tokens,
            keywords=keywords,
            positive_example_weight=2.0,
            focus_rows_by_manifest={
                focused: [(), (), (1,)],
            },
        )
        assert focused_balance["explicit_focus_nonwake_rows"] == 1
        assert focused_balance["fallback_edit_distance_nonwake_rows"] == 0
        assert focused_balance["keyword_balance"]["1"]["assigned_nonwake_mass"] == 2.0
        assert focused_balance["keyword_balance"]["2"]["assigned_nonwake_mass"] == 0.0
        assert focused_balance["manifests"][0]["explicit_focus_nonwake_rows"] == 1

        # Sidecar evidence expansion must preserve renderer row order and repeat
        # counts for all three replay sources.
        static_manifest = root / "static.tsv"
        adversarial_manifest = root / "adversarial.tsv"
        failure_manifest = root / "failure.tsv"
        focus_map = build_refinement_focus_rows(
            static={
                "manifest": str(static_manifest),
                "examples": 3,
                "sequences": [
                    {"examples": 2, "focus_keyword_id": 1},
                ],
                "positive_stress": [
                    {"examples": 1, "keyword_id": 2},
                ],
            },
            adversarial={
                "manifest": str(adversarial_manifest),
                "replay_examples": 2,
                "replay_examples_per_sequence": 2,
                "selected": [
                    {"focus_keyword_id": 2},
                ],
            },
            failure={
                "manifest": str(failure_manifest),
                "examples": 2,
                "examples_per_failure": 2,
                "selected": [
                    {
                        "focus_keyword_ids": [1],
                        "source_keyword_id": None,
                        "source_splits": ["test"],
                    }
                ],
            },
        )
        assert focus_map[static_manifest.resolve()] == [(1,), (1,), (2,)]
        assert focus_map[adversarial_manifest.resolve()] == [(2,), (2,)]
        assert focus_map[failure_manifest.resolve()] == [(1,), (1,)]

    leaked = dict(manifest)
    leaked["candidate_selection"] = dict(manifest["candidate_selection"])
    leaked["candidate_selection"]["qualification_used_for_selection"] = True
    try:
        select_refinement_source(leaked)
    except ValueError as exc:
        assert "must not use qualification" in str(exc)
    else:
        raise AssertionError("qualification-backed refinement source was accepted")

    refinement_source = (ROOT / "training/adversarial_refinement.py").read_text(
        encoding="utf-8"
    )
    assert "optional_objective_cli_args(train)" in refinement_source

    print("adversarial refinement source/provenance-first wake pressure balance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
