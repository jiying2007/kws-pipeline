#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import development_resume as development_resume  # noqa: E402
import hard_negative_replay as hard_negative_replay_module  # noqa: E402
from adversarial_lexicon import enumerate_safe_sequences  # noqa: E402
from hard_negative_replay import (  # noqa: E402
    _render_command_tts_cached,
    adaptive_focus,
    hard_negative_stress_focus,
    normalize_hard_negative_replay,
    normalize_positive_stress_replay,
    positive_stress_focus,
)
from synthetic_audio import write_wav  # noqa: E402
from iterate_domain import (  # noqa: E402
    calibration_behavior_key,
    calibration_operating_curve_summary,
    calibration_threshold_vector,
    calibration_trial_evidence,
    parse_warm_start_strategy,
    posterior_replay_cli_args,
    resolve_posterior_replay,
    select_calibration_threshold,
    select_strict_candidate,
    strict_gate_candidate,
    torch_round_training_values,
    train_acoustic_seed_offset,
    warm_start_args,
)


def validate_clean_tts_round_cache() -> None:
    calls: list[tuple[str, tuple[str, ...], str, str]] = []
    original = hard_negative_replay_module.render_command_tts

    def fake_render_command_tts(
        text: str,
        token_names: list[str],
        kind: str,
        output: pathlib.Path,
        cfg: dict,
    ) -> list[int]:
        del cfg
        calls.append((text, tuple(token_names), kind, output.name))
        samples = [1000, -1000, 500, -500]
        output.parent.mkdir(parents=True, exist_ok=True)
        write_wav(output, samples)
        return samples

    hard_negative_replay_module.render_command_tts = fake_render_command_tts
    try:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            cache = root / "cache"
            tts = {
                "backend": "command",
                "command": ["fake", "--output={output}", "{text}"],
                "speaker_profiles": [{"speaker_id": 7, "length_scale": 1.0}],
                "provider_identity_sha256": "a" * 64,
                "reuse_clean_across_rounds": True,
            }
            round0 = root / "round-00" / "clean" / "h00-e000.wav"
            round1 = root / "round-01" / "clean" / "h00-e000.wav"
            first = _render_command_tts_cached(
                "hao3 ni3",
                ["hao3", "ni3"],
                "hard-negative",
                round0,
                tts,
                cache_root=cache,
            )
            second = _render_command_tts_cached(
                "hao3 ni3",
                ["hao3", "ni3"],
                "hard-negative",
                round1,
                tts,
                cache_root=cache,
            )
            assert first == second
            assert round0.read_bytes() == round1.read_bytes()
            assert len(calls) == 1

            changed = root / "round-02" / "clean" / "h00-e000.wav"
            _render_command_tts_cached(
                "hao3 xiao3",
                ["hao3", "xiao3"],
                "hard-negative",
                changed,
                tts,
                cache_root=cache,
            )
            assert len(calls) == 2

            uncached = dict(tts)
            uncached["reuse_clean_across_rounds"] = False
            _render_command_tts_cached(
                "hao3 ni3",
                ["hao3", "ni3"],
                "hard-negative",
                root / "round-03" / "clean" / "h00-e000.wav",
                uncached,
                cache_root=cache,
            )
            assert len(calls) == 3

            invalid = dict(tts)
            invalid["reuse_clean_across_rounds"] = "true"
            try:
                _render_command_tts_cached(
                    "hao3 ni3",
                    ["hao3", "ni3"],
                    "hard-negative",
                    root / "round-04" / "clean" / "h00-e000.wav",
                    invalid,
                    cache_root=cache,
                )
            except ValueError as exc:
                assert "must be boolean" in str(exc)
            else:
                raise AssertionError("non-boolean clean-TTS reuse flag was accepted")
    finally:
        hard_negative_replay_module.render_command_tts = original


def validate_torch_iteration_policy() -> None:
    validate_clean_tts_round_cache()
    development_resume.self_test()
    for iterator in (
        ROOT / "training" / "iterate_gru_development.py",
        ROOT / "training" / "iterate_rnn_development.py",
    ):
        source = iterator.read_text(encoding="utf-8")
        assert '"--resume-state"' in source
        assert '"--round-budget"' in source
        assert "development_resume.write_state(" in source
        assert "SEGMENT_CONTINUE_EXIT_CODE" in source
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

    round_policy = {
        "seed": 1337,
        "train": {
            "epochs": 36,
            "warm_start_epochs": 12,
            "lr": 0.001,
        },
        "domain_iteration": {
            "lr_decay_per_round": 0.85,
            "training_seed_stride": 1009,
            "training_acoustic_seed_stride": 104729,
        },
    }
    assert torch_round_training_values(
        round_policy, round_index=0, warm_started=False
    ) == (36, 0.001, 1337)
    warm_round_1 = torch_round_training_values(
        round_policy, round_index=1, warm_started=True
    )
    assert warm_round_1[0] == 12
    assert abs(warm_round_1[1] - 0.00085) < 1.0e-15
    assert warm_round_1[2] == 2346
    warm_round_3 = torch_round_training_values(
        round_policy, round_index=3, warm_started=True
    )
    assert warm_round_3[0] == 12
    assert abs(warm_round_3[1] - 0.001 * (0.85 ** 3)) < 1.0e-15
    assert warm_round_3[2] == 4364
    assert train_acoustic_seed_offset(round_policy["domain_iteration"], 0) == 0
    assert train_acoustic_seed_offset(round_policy["domain_iteration"], 1) == 104729
    assert train_acoustic_seed_offset(round_policy["domain_iteration"], 3) == 314187

    vector_a = calibration_threshold_vector(
        [
            {"id": 2, "threshold": 0.60},
            {"id": 1, "threshold": 0.58},
        ]
    )
    vector_b = calibration_threshold_vector(
        [
            {"id": 1, "threshold": 0.58},
            {"id": 2, "threshold": 0.60},
        ]
    )
    assert vector_a == ((1, 0.58), (2, 0.60))
    assert vector_b == vector_a
    try:
        calibration_threshold_vector(
            [
                {"id": 1, "threshold": 0.58},
                {"id": 1, "threshold": 0.60},
            ]
        )
    except ValueError as exc:
        assert "duplicate keyword ids" in str(exc)
    else:
        raise AssertionError("duplicate calibration keyword id was accepted")
    try:
        calibration_threshold_vector([{"id": 1, "threshold": 1.0}])
    except ValueError as exc:
        assert "invalid threshold" in str(exc)
    else:
        raise AssertionError("invalid calibration threshold was accepted")

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

    strict_gates = {
        "max_frr": 0.0,
        "max_far_per_hour": 0.0,
        "max_p95_latency_ms": 800.0,
        "max_far_frr": 0.0,
    }

    def calibration_metrics(frr: float, far_per_hour: float) -> tuple[dict, dict]:
        base = {
            "frr": frr,
            "far_per_hour": far_per_hour,
            "p95_post_end_latency_ms": 100.0,
        }
        domains = {
            "domains": {"distance:far": {"frr": frr}},
            "worst_domain_score": max(frr, far_per_hour / 1000.0),
        }
        return base, domains

    reject_base, reject_domains = calibration_metrics(1.0, 0.0)
    useful_base, useful_domains = calibration_metrics(0.10, 10.0)
    reject_key = calibration_behavior_key(reject_base, reject_domains, strict_gates)
    useful_key = calibration_behavior_key(useful_base, useful_domains, strict_gates)
    assert useful_key < reject_key, (useful_key, reject_key)
    assert select_calibration_threshold(
        [(0.60, reject_key), (0.52, useful_key)]
    ) == 0.52

    strict_base, strict_domains = calibration_metrics(0.0, 0.0)
    strict_key = calibration_behavior_key(strict_base, strict_domains, strict_gates)
    assert strict_key == zero_error_key
    assert select_calibration_threshold(
        [(0.50, strict_key), (0.52, strict_key), (0.54, strict_key)]
    ) == 0.52

    trial = calibration_trial_evidence(
        coordinate=0,
        keyword_id=1,
        threshold=0.60,
        trial_keywords=[
            {"id": 1, "threshold": 0.60},
            {"id": 2, "threshold": 0.55},
        ],
        base={
            "frr": 0.75,
            "far_per_hour": 12.0,
            "p95_post_end_latency_ms": 120.0,
            "per_keyword": {
                "1": {"frr": 1.0},
                "2": {"frr": 0.5},
            },
        },
        domains={
            "domains": {"distance:far": {"frr": 0.8}},
            "worst_domain_score": 12.0,
        },
        gates=strict_gates,
    )
    assert trial["keyword_thresholds"] == {"1": 0.60, "2": 0.55}
    assert trial["strict"] is False
    assert trial["metrics"]["per_keyword_frr"] == {"1": 1.0, "2": 0.5}

    trial2 = dict(trial)
    trial2["coordinate"] = 1
    trial2["threshold"] = 0.55
    trial2["keyword_thresholds"] = {"1": 0.55, "2": 0.55}
    trial2["metrics"] = dict(trial["metrics"])
    trial2["metrics"]["frr"] = 0.5
    trial2["metrics"]["far_per_hour"] = 20.0
    trial_kw2 = dict(trial)
    trial_kw2["keyword_id"] = 2
    trial_kw2["threshold"] = 0.50
    trial_kw2["keyword_thresholds"] = {"1": 0.55, "2": 0.50}
    curve = calibration_operating_curve_summary(
        [trial, trial2, trial_kw2],
        selected_thresholds={"1": 0.60, "2": 0.50},
        threshold_grid=[0.50, 0.55, 0.60],
        coordinates_executed=2,
    )
    assert curve["trial_count"] == 3
    assert curve["coordinates_executed"] == 2
    assert curve["grid_saturated"] is True
    assert curve["per_keyword"]["1"]["selected_on_grid_max"] is True
    assert curve["per_keyword"]["2"]["selected_on_grid_min"] is True
    assert curve["per_keyword"]["1"]["min_frr"] == 0.5
    assert curve["per_keyword"]["1"]["min_far_per_hour"] == 12.0

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
    assert int(formal["calibration"]["max_parallel_trials"]) == 2
    assert int(formal["train"]["epochs"]) == 36
    assert int(formal["train"]["warm_start_epochs"]) == 12
    assert abs(float(formal["domain_iteration"]["lr_decay_per_round"]) - 0.85) < 1.0e-12
    assert int(formal["domain_iteration"]["training_seed_stride"]) == 1009
    assert int(formal["domain_iteration"]["training_acoustic_seed_stride"]) == 104729
    assert int(formal["domain_iteration"]["min_rounds"]) == 2
    assert int(formal["domain_iteration"]["max_rounds"]) == 4
    assert int(formal["domain_iteration"]["patience"]) == 2
    assert formal["domain_iteration"]["stop_on_gate"] is True

    product_iterator = (ROOT / "training" / "iterate_domain.py").read_text(
        encoding="utf-8"
    )
    product_experiment_workflow = (
        ROOT / ".github" / "workflows" / "product-development-experiment.yml"
    ).read_text(encoding="utf-8")
    recalibrated_retention = (
        ROOT / "tools" / "diagnose_decoder_retention_recalibrated_curve.py"
    ).read_text(encoding="utf-8")
    threshold_diagnostic = (
        ROOT / "tools" / "diagnose_kws_threshold_operating_curve.py"
    ).read_text(encoding="utf-8")
    assert "merge_domain_metrics(" in product_iterator
    assert '"development_split_roles": {' in product_iterator
    assert '"train": "development-training"' in product_iterator
    assert '"calibration": "development-calibration"' in product_iterator
    assert '"test": "development-feedback"' in product_iterator
    compact_diagnostics = (
        ROOT / "tools" / "build_training_diagnostics.py"
    ).read_text(encoding="utf-8")
    assert '"split_roles": (' in compact_diagnostics
    assert '"test": "development-feedback"' in compact_diagnostics
    assert 'round_best["calibration_domains"]' in product_iterator
    assert 'round_best["test_domains"]' in product_iterator
    assert product_iterator.count("suppress_stdout=True") == 2
    assert "optional_objective_cli_args(train)" in product_iterator
    assert "exact-keyword-threshold-vector-v1" in product_iterator
    assert 'base["calibration_trial_cache_hits"]' in product_iterator
    assert 'base["calibration_unique_trial_vectors"]' in product_iterator
    assert "decoder_state_retention=decoder_state_retention" in product_iterator
    assert "decoder_refractory_ms=decoder_refractory_ms" in product_iterator
    assert "thresholds_recalibrated_per_retention" in recalibrated_retention
    assert "selection_feedback_allowed" in recalibrated_retention
    assert "decoder_state_retention=retention" in recalibrated_retention
    assert "calibrate(" in recalibrated_retention
    assert "evaluate(" in recalibrated_retention
    assert "resolve_posterior_replay" in threshold_diagnostic
    assert threshold_diagnostic.count("posterior_replay=posterior_replay") == 2
    assert 'parser.add_argument("--posterior-dump"' in threshold_diagnostic
    assert 'parser.add_argument("--decoder-replay"' in threshold_diagnostic
    assert 'parser.add_argument("--posterior-cache"' in threshold_diagnostic
    assert (
        'parser.add_argument("--development-authority-receipt"'
        in threshold_diagnostic
    )
    assert "exact-pr-head-development-receipt-v1" in threshold_diagnostic
    assert "product-development-pr-head-experiment-v1" in threshold_diagnostic
    assert "development authority receipt config does not match manifest" in threshold_diagnostic
    assert '"calibration_references_sha256": str(calibration.get("references_sha256", ""))' in threshold_diagnostic
    assert '"test_references_sha256": str(test.get("references_sha256", ""))' in threshold_diagnostic
    assert "calibration references do not match development manifest" in threshold_diagnostic
    assert "test references do not match development manifest" in threshold_diagnostic
    assert (
        '"posterior_replay_enabled": posterior_replay is not None'
        in threshold_diagnostic
    )
    assert "'--development-authority-receipt'" in product_experiment_workflow
    assert "KWS_EXPERIMENT_RECEIPT" in product_experiment_workflow
    assert "'--development-domain-summary'" not in product_experiment_workflow
    score_block = product_iterator.split('str(EVAL / "score_events.py")', 1)[1]
    assert "suppress_stdout=True" in score_block.split('str(EVAL / "domain_metrics.py")', 1)[0]
    domain_block = product_iterator.split('str(EVAL / "domain_metrics.py")', 1)[1]
    assert "suppress_stdout=True" in domain_block.split("base = json.loads", 1)[0]

    shadow = formal["shadow_qualification"]
    assert shadow["enabled"] is True
    assert len(shadow["seeds"]) == 8
    assert len(set(shadow["seeds"])) == 8
    assert int(shadow["expected_wakes_per_seed"]) == 256
    assert float(shadow["min_surrogate_separation"]) == 0.06
    assert int(formal["qualification_holdout_seed"]) not in set(shadow["seeds"])
    assert not set(formal["retired_qualification_holdout_seeds"]) & set(shadow["seeds"])

    adversarial = formal["domain_iteration"]["adversarial_lexicon"]
    assert set(adversarial) == {
        "enabled",
        "max_length",
        "top_k",
        "probes_per_sequence",
        "replay_examples_per_sequence",
        "refinement_epochs",
        "refinement_lr_scale",
    }
    assert adversarial["enabled"] is True
    assert int(adversarial["max_length"]) == 5
    assert int(adversarial["top_k"]) == 24
    assert int(adversarial["probes_per_sequence"]) == 1
    assert int(adversarial["replay_examples_per_sequence"]) == 4
    assert int(adversarial["refinement_epochs"]) == 12
    assert float(adversarial["refinement_lr_scale"]) == 0.5

    positive_stress = formal["domain_iteration"]["positive_stress_replay"]
    assert {int(item["keyword_id"]) for item in positive_stress} == {1, 2}
    assert all(item["focus"] == "adaptive" for item in positive_stress)
    assert all(int(item["examples"]) == 32 for item in positive_stress)
    assert all(item["fallback"]["distance_bin"] == "5m" for item in positive_stress)
    assert all(item["fallback"]["azimuth"] == "rear" for item in positive_stress)
    assert all(item["fallback"]["snr"] == "critical" for item in positive_stress)

    formal_hard_negative = formal["domain_iteration"]["hard_negative_replay"]
    assert formal_hard_negative
    assert min(int(item["examples"]) for item in formal_hard_negative) >= 24
    prefix = next(
        item
        for item in formal_hard_negative
        if item["tokens"] == ["ni3", "hao3", "xiao3"]
    )
    assert int(prefix["examples"]) >= 24
    assert int(prefix["focus_keyword_id"]) == 1

    formal_domains = formal["domains"]
    coverage_domains = {
        "distance_bands": formal_domains["distance_bands"],
        "azimuth_deg": formal_domains["azimuth_deg"],
        "rt60_s": formal_domains["rt60_s"],
        "snr_db": formal_domains["snr_db"],
        "noise_profiles": formal_domains["noise_profiles"],
        "playback_probability": formal_domains["playback"]["probability"],
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

    factor_levels = {
        "azimuth": ("front", "side", "rear"),
        "snr": ("critical", "low", "mid", "high"),
        "playback": (False, True),
        "noise": tuple(formal_domains["noise_profiles"]),
        "distance_bin": ("0.5m", "1m", "2m", "3m", "5m"),
        "rt60": ("dry", "medium", "reverb"),
    }

    def factor_value(item: dict, factor: str):
        if factor == "azimuth":
            return azimuth_band(float(item["azimuth"]))
        return item[factor]

    assert all(set(item) == set(factor_levels) for item in cube)
    for left, right in itertools.combinations(factor_levels, 2):
        observed_pairs = {
            (factor_value(item, left), factor_value(item, right)) for item in cube
        }
        expected_pairs = set(itertools.product(factor_levels[left], factor_levels[right]))
        assert observed_pairs == expected_pairs, (left, right, expected_pairs - observed_pairs)

    adaptive = {"distance_bin": "5m", "azimuth": "rear", "snr": "critical"}
    positive_cover = [
        positive_stress_focus(
            coverage_domains,
            round_index=0,
            keyword_id=1,
            example_index=index,
            total_examples=32,
            adaptive=adaptive,
        )
        for index in range(24)
    ]
    assert len(positive_cover) == 24
    for left, right in itertools.combinations(factor_levels, 2):
        observed_pairs = {
            (factor_value(item, left), factor_value(item, right))
            for item in positive_cover
        }
        expected_pairs = set(itertools.product(factor_levels[left], factor_levels[right]))
        assert observed_pairs == expected_pairs, ("positive", left, right)
    for index in range(24, 32):
        assert positive_stress_focus(
            coverage_domains,
            round_index=0,
            keyword_id=1,
            example_index=index,
            total_examples=32,
            adaptive=adaptive,
        ) == adaptive

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

    lexical = enumerate_safe_sequences(active, forbidden, max_length=5)
    assert len(lexical) == 1330
    assert lexical == enumerate_safe_sequences(active, forbidden, max_length=5)
    assert ("ni3", "hao3", "xiao3") in lexical
    assert ("ni3", "hao3", "xiao3", "wo1") not in lexical
    assert ("xiao3", "wo1", "xiao3", "wo1") not in lexical

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
    assert posterior_replay_cli_args(None) == []
    try:
        posterior_replay_cli_args(None, decoder_state_retention=0.94)
    except ValueError as exc:
        assert "require posterior replay" in str(exc)
    else:
        raise AssertionError("decoder override without posterior replay was accepted")
    replay_args = posterior_replay_cli_args(
        (
            pathlib.Path("/tmp/posterior-dump"),
            pathlib.Path("/tmp/decoder-replay"),
            pathlib.Path("/tmp/posterior-cache"),
        ),
        decoder_state_retention=0.94,
        decoder_refractory_ms=1200,
    )
    assert replay_args[-4:] == [
        "--decoder-state-retention",
        "0.94",
        "--decoder-refractory-ms",
        "1200",
    ]
    with tempfile.TemporaryDirectory(prefix="posterior-replay-contract-") as td:
        root = pathlib.Path(td)
        dump = root / "dump"
        replay = root / "replay"
        cache = root / "cache"
        dump.write_text("", encoding="utf-8")
        replay.write_text("", encoding="utf-8")
        resolved = resolve_posterior_replay(dump, replay, cache)
        assert resolved is not None
        assert resolved[0] == dump.resolve()
        assert resolved[1] == replay.resolve()
        assert resolved[2] == cache.resolve()
        assert cache.is_dir()
        assert posterior_replay_cli_args(resolved) == [
            "--posterior-dump",
            str(dump.resolve()),
            "--decoder-replay",
            str(replay.resolve()),
            "--posterior-cache",
            str(cache.resolve()),
        ]
        try:
            resolve_posterior_replay(dump, None, cache)
        except ValueError as exc:
            assert "requires --posterior-dump" in str(exc)
        else:
            raise AssertionError("partial posterior replay configuration was accepted")

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
        assert manifest["records"][0]["training_acoustic_seed_policy"] == "train-only-scene-seed-offset-v1"
        assert int(manifest["records"][0]["training_acoustic_seed_offset"]) == 0
        assert manifest["best_frontend"] in {"logmel", "pcen-lite"}
        assert {row["frontend"] for row in manifest["records"]} == {"logmel", "pcen-lite"}
        far = manifest["qualification_domains"]["domains"]["distance:far"]
        assert int(far["expected"]) >= 1
        assert pathlib.Path(work / "best" / "model.kwm").is_file()
        assert pathlib.Path(work / "best" / "keywords.kwk").is_file()

        deferred_work = root / "deferred-work"
        deferred = subprocess.run(
            [
                sys.executable,
                str(ROOT / "training" / "iterate_domain.py"),
                "--config",
                str(path),
                "--runner",
                str(args.runner.resolve()),
                "--work-dir",
                str(deferred_work),
                "--defer-qualification",
            ],
            check=False,
        )
        assert deferred.returncode == 0, deferred.returncode
        deferred_manifest = json.loads(
            (deferred_work / "domain-loop-manifest.json").read_text(encoding="utf-8")
        )
        assert deferred_manifest["development_qualified"] is True
        assert deferred_manifest["qualification_deferred"] is True
        assert deferred_manifest["qualification_qualified"] is None
        assert deferred_manifest["qualified"] is False
        assert deferred_manifest["qualification"] == {}
        assert deferred_manifest["qualification_domains"] == {}
        assert deferred_manifest["evidence_class"] == "synthetic-domain-development-only"
        assert not (deferred_work / "best" / "qualification").exists()

    print("test_domain_loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
