#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from hard_negative_replay import (  # noqa: E402
    adaptive_focus,
    hard_negative_stress_focus,
    normalize_hard_negative_replay,
    normalize_positive_stress_replay,
)
from iterate_domain import (  # noqa: E402
    parse_warm_start_strategy,
    select_calibration_threshold,
    select_strict_candidate,
    strict_gate_candidate,
    warm_start_args,
)


def validate_torch_iteration_policy() -> None:
    assert parse_warm_start_strategy({}) == "full"
    assert parse_warm_start_strategy({"warm_start_strategy": "full"}) == "full"
    assert parse_warm_start_strategy({"warm_start_strategy": "head-only"}) == "head-only"
    previous = pathlib.Path("previous.pt")
    assert warm_start_args(None, "full") == []
    assert warm_start_args(previous, "full") == ["--warm-start", "previous.pt"]
    assert warm_start_args(previous, "head-only") == [
        "--warm-start",
        "previous.pt",
        "--head-only",
    ]
    try:
        parse_warm_start_strategy({"warm_start_strategy": "frozen"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid warm-start strategy was accepted")

    thresholds = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.15,
        0.25,
        0.40,
        0.55,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.93,
        0.95,
    ]
    zero_error_key = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert select_calibration_threshold([(value, zero_error_key) for value in thresholds]) == 0.55
    worse_key = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    plateau = [
        (0.10, worse_key),
        (0.25, zero_error_key),
        (0.40, zero_error_key),
        (0.55, zero_error_key),
        (0.70, zero_error_key),
        (0.85, worse_key),
    ]
    assert select_calibration_threshold(plateau) == 0.40
    try:
        select_calibration_threshold([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty calibration candidate set was accepted")

    # Regression for model-training #132: calibration-only and test-only passes
    # from different rounds must never be promoted into a synthetic qualification.
    split_pass = [
        {
            "round": 0,
            "frontend": "logmel",
            "score": 1.0,
            "calibration_gate": False,
            "test_gate": True,
        },
        {
            "round": 2,
            "frontend": "logmel",
            "score": 0.5,
            "calibration_gate": True,
            "test_gate": False,
        },
    ]
    assert all(not strict_gate_candidate(record) for record in split_pass)
    assert select_strict_candidate(split_pass) is None

    strict_candidates = split_pass + [
        {
            "round": 1,
            "frontend": "pcen-lite",
            "score": 0.1,
            "calibration_gate": True,
            "test_gate": True,
        },
        {
            "round": 3,
            "frontend": "pcen-lite",
            "score": 0.3,
            "calibration_gate": True,
            "test_gate": True,
        },
        {
            "round": 3,
            "frontend": "logmel",
            "score": 0.2,
            "calibration_gate": True,
            "test_gate": True,
        },
    ]
    selected_strict = select_strict_candidate(strict_candidates)
    assert selected_strict is not None
    assert int(selected_strict["round"]) == 3
    assert selected_strict["frontend"] == "logmel"
    assert float(selected_strict["score"]) == 0.2

    formal = json.loads(
        (ROOT / "configs" / "training" / "xiaowo.torch-domain.json").read_text(encoding="utf-8")
    )
    assert float(formal["domain_gates"]["max_far_per_hour"]) == 0.0
    assert float(formal["domain_gates"]["max_frr"]) == 0.0
    assert float(formal["domain_gates"]["max_far_frr"]) == 0.0
    assert 0.55 in formal["calibration"]["thresholds"]
    assert len(formal["calibration"]["thresholds"]) >= 5
    assert int(formal["calibration"]["coordinate_rounds"]) >= 2
    positive_stress = formal["domain_iteration"]["positive_stress_replay"]
    assert {int(item["keyword_id"]) for item in positive_stress} == {1, 2}
    assert all(item["focus"] == "adaptive" for item in positive_stress)
    assert all(item["fallback"]["distance_bin"] == "5m" for item in positive_stress)
    assert all(item["fallback"]["azimuth"] == "rear" for item in positive_stress)
    assert all(item["fallback"]["snr"] == "critical" for item in positive_stress)

    formal_hard_negative = formal["domain_iteration"]["hard_negative_replay"]
    assert formal_hard_negative
    assert min(int(item["examples"]) for item in formal_hard_negative) >= 24

    coverage_domains = {
        "azimuth_deg": [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180],
        "snr_db": [3.0, 30.0],
        "playback_probability": 0.35,
    }

    def azimuth_band(value: float) -> str:
        if abs(value) <= 30.0:
            return "front"
        if abs(value) <= 90.0:
            return "side"
        return "rear"

    cube = [
        hard_negative_stress_focus(
            coverage_domains,
            round_index=0,
            item_index=3,
            example_index=index,
        )
        for index in range(24)
    ]
    observed = {
        (azimuth_band(float(item["azimuth"])), str(item["snr"]), bool(item["playback"]))
        for item in cube
    }
    expected = {
        (azimuth, snr, playback)
        for playback in (False, True)
        for snr in ("critical", "low", "mid", "high")
        for azimuth in ("front", "side", "rear")
    }
    assert len(cube) == 24
    assert observed == expected

    side_angles = {
        float(item["azimuth"])
        for round_index in range(4)
        for example_index in range(24)
        for item in [
            hard_negative_stress_focus(
                coverage_domains,
                round_index=round_index,
                item_index=3,
                example_index=example_index,
            )
        ]
        if azimuth_band(float(item["azimuth"])) == "side"
    }
    assert side_angles == {-90.0, -60.0, 60.0, 90.0}

    measured_domains = {**coverage_domains, "rir_manifest": {"entries": []}}
    measured_focus = hard_negative_stress_focus(
        measured_domains,
        round_index=0,
        item_index=0,
        example_index=0,
    )
    assert "azimuth" not in measured_focus
    assert set(measured_focus) == {"snr", "playback"}

    active = ["ni3", "hao3", "xiao3", "wo1"]
    token_map = {"<blank>": 0, "ni3": 1, "hao3": 2, "xiao3": 3, "wo1": 4}
    forbidden = [
        ["ni3", "hao3", "xiao3", "wo1"],
        ["xiao3", "wo1", "xiao3", "wo1"],
    ]
    replay = normalize_hard_negative_replay(
        [
            {
                "tokens": ["hao3", "ni3", "xiao3", "wo1"],
                "examples": 24,
                "focus_keyword_id": 1,
            },
            {"tokens": ["hao3", "hao3", "xiao3", "wo1"], "examples": 16},
            {
                "tokens": ["hao3", "wo1", "xiao3", "wo1", "ni3"],
                "examples": 16,
                "focus_keyword_id": 1,
            },
        ],
        active_tokens=active,
        forbidden=forbidden,
        token_map=token_map,
        keyword_ids={1, 2},
    )
    assert [item["examples"] for item in replay] == [24, 16, 16]
    assert replay[0]["target_ids"] == [2, 1, 3, 4]
    assert replay[0]["focus_keyword_id"] == 1
    assert replay[2]["target_ids"] == [2, 4, 3, 4, 1]
    try:
        normalize_hard_negative_replay(
            [{"tokens": ["ni3", "hao3", "xiao3", "wo1"], "examples": 1}],
            active_tokens=active,
            forbidden=forbidden,
            token_map=token_map,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("wake path was accepted as a hard negative")

    keywords = [
        {
            "id": 1,
            "text": "你好小窝",
            "tokens": ["ni3", "hao3", "xiao3", "wo1"],
            "token_ids": [1, 2, 3, 4],
        },
        {
            "id": 2,
            "text": "小窝小窝",
            "tokens": ["xiao3", "wo1", "xiao3", "wo1"],
            "token_ids": [3, 4, 3, 4],
        },
    ]
    normalized_positive = normalize_positive_stress_replay(
        [
            {
                "keyword_id": 1,
                "examples": 8,
                "focus": "adaptive",
                "fallback": {
                    "distance_bin": "5m",
                    "azimuth": "rear",
                    "snr": "critical",
                },
            },
            {
                "keyword_id": 2,
                "examples": 8,
                "focus": "adaptive",
                "fallback": {
                    "distance_bin": "5m",
                    "azimuth": "rear",
                    "snr": "critical",
                },
            },
        ],
        keywords=keywords,
    )
    assert normalized_positive[0]["target_ids"] == [1, 2, 3, 4]
    assert normalized_positive[1]["target_ids"] == [3, 4, 3, 4]
    curriculum = {
        "keyword_worst_domains": {
            "2": [
                {
                    "domain": "distance_azimuth_snr:distance_bin=3m|azimuth=side|snr=low",
                    "hardness": 2.0,
                },
                {
                    "domain": "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical",
                    "hardness": 4.0,
                },
            ]
        }
    }
    focus = adaptive_focus(
        curriculum,
        2,
        {"distance_bin": "5m", "azimuth": "rear", "snr": "critical"},
    )
    assert focus == {"distance_bin": "5m", "azimuth": "rear", "snr": "critical"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()
    validate_torch_iteration_policy()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        config = json.loads(
            (ROOT / "configs" / "training" / "xiaowo.domain.json").read_text(encoding="utf-8")
        )
        for split in ("train", "calibration", "test", "qualification"):
            config["dataset"][split] = {
                "positive_families_per_keyword": 1,
                "confusable_families_per_keyword": 1,
                "random_negative_families": 1,
                "background_seconds_per_profile": 0.1,
                "variants_per_family": 1,
            }
            config["domains"]["scenes_per_example"][split] = 1
        config["domains"]["distance_bands"] = {
            "near": {"distance_m": [0.4, 0.8], "weight": 1.0},
            "mid": {"distance_m": [1.2, 1.8], "weight": 1.0},
            "far": {"distance_m": [3.0, 3.6], "weight": 2.0},
        }
        config["domains"]["azimuth_deg"] = [-60, 0, 60]
        config["domains"]["rt60_s"] = [0.12, 0.28]
        config["domains"]["snr_db"] = [22.0, 34.0]
        config["domains"]["playback"]["probability"] = 0.1
        config["model"]["frontends"] = ["logmel", "pcen-lite"]
        config["model"]["domain_variants_per_token"] = 16
        config["model"]["prototype_candidates"] = [
            {"input_scale": 0.010, "output_scale": 0.050, "blank_bias": 1.8, "token_bias": -1.2}
        ]
        config["calibration"] = {"thresholds": [0.25, 0.55], "coordinate_rounds": 1}
        config["domain_iteration"] = {
            "backend": "prototype",
            "max_rounds": 1,
            "min_rounds": 1,
            "patience": 0,
            "curriculum_strength": 2.0,
            "max_domain_weight": 4.0,
            "stop_on_gate": True,
        }
        config["domain_gates"] = {
            "max_frr": 1.0,
            "max_far_per_hour": 1000000.0,
            "max_p95_latency_ms": 1000000.0,
            "max_far_frr": 1.0,
        }
        path = root / "domain-smoke.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        work = root / "work"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "iterate_domain.py"),
                "--config",
                str(path),
                "--runner",
                str(args.runner.resolve()),
                "--work-dir",
                str(work),
            ],
            check=False,
        )
        assert completed.returncode == 0, completed.returncode
        manifest = json.loads((work / "domain-loop-manifest.json").read_text(encoding="utf-8"))
        assert manifest["qualified"] is True
        assert manifest["development_qualified"] is True
        assert manifest["qualification_qualified"] is True
        assert manifest["evidence_class"] == "synthetic-domain-qualified"
        assert manifest["candidate_selection"]["policy"] == "latest-strict-gate-passing-round"
        assert manifest["candidate_selection"]["qualification_used_for_selection"] is False
        assert manifest["candidate_selection"]["objective_fallback_used"] is False
        assert manifest["best_frontend"] in {"logmel", "pcen-lite"}
        assert {row["frontend"] for row in manifest["records"]} == {"logmel", "pcen-lite"}
        far = manifest["qualification_domains"]["domains"]["distance:far"]
        assert int(far["expected"]) >= 1
        assert pathlib.Path(work / "best" / "model.kwm").is_file()
        assert pathlib.Path(work / "best" / "keywords.kwk").is_file()

    print("test_domain_loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
