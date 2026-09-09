from __future__ import annotations

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "training" / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "training" / "requirements.lock").read_text(encoding="utf-8")
    model_source = (ROOT / "training" / "model.py").read_text(encoding="utf-8")
    site_source = (ROOT / "training" / "sitecustomize.py").read_text(encoding="utf-8")
    train_source = (ROOT / "training" / "train_ctc.py").read_text(encoding="utf-8")
    iterate_source = (ROOT / "training" / "iterate_domain.py").read_text(encoding="utf-8")
    margin_source = (ROOT / "training" / "sequence_margin.py").read_text(encoding="utf-8")
    margin_test_source = (ROOT / "training" / "test_sequence_margin.py").read_text(
        encoding="utf-8"
    )
    qualification_source = (ROOT / "training" / "render_qualification_holdout.py").read_text(
        encoding="utf-8"
    )
    model_training_workflow = (ROOT / ".github" / "workflows" / "model-training.yml").read_text(
        encoding="utf-8"
    )
    curriculum_source = (ROOT / "training" / "domain_curriculum.py").read_text(
        encoding="utf-8"
    )
    formal = json.loads(
        (ROOT / "configs" / "training" / "xiaowo.torch-domain.json").read_text(
            encoding="utf-8"
        )
    )
    assert "ARG KWS_TRAINING_BASE" in dockerfile
    assert "FROM ${KWS_TRAINING_BASE}" in dockerfile
    assert "pip install" not in dockerfile
    assert "pip install --upgrade" not in dockerfile
    assert "python==3.12.11" in lock
    assert "torch==2.13.0" in lock

    for source in (site_source, model_source):
        assert '"OMP_NUM_THREADS": "2"' in source
        assert '"OMP_DYNAMIC": "FALSE"' in source
        assert '"MKL_NUM_THREADS": "2"' in source
        assert '"MKL_CBWR": "AVX2"' in source
        assert '"OPENBLAS_NUM_THREADS": "2"' in source
        assert '"NUMEXPR_NUM_THREADS": "2"' in source
        assert '"ATEN_CPU_CAPABILITY": "avx2"' in source
    assert "TRAINING_TORCH_NUM_THREADS = 2" in model_source
    assert "TRAINING_TORCH_NUM_INTEROP_THREADS = 1" in model_source
    assert "torch.set_num_threads(TRAINING_TORCH_NUM_THREADS)" in model_source
    assert (
        "torch.set_num_interop_threads(TRAINING_TORCH_NUM_INTEROP_THREADS)"
        in model_source
    )
    assert "torch.backends.mkldnn.enabled = False" in model_source
    assert '"torch_num_threads": int(torch.get_num_threads())' in train_source
    assert (
        '"torch_num_interop_threads": int(torch.get_num_interop_threads())'
        in train_source
    )
    assert "torch.use_deterministic_algorithms(True)" in train_source

    # The decoder-confidence auxiliary objective is runtime-aligned and now owns
    # an explicit operating point for every shipping keyword. This keeps the two
    # wake words independently tunable while preserving wake-on-any-keyword FAR
    # semantics and provenance of the exact profile used for training.
    assert "KEYWORD_SEQUENCE_MARGIN = 0.05" in train_source
    assert "KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT = 0.10" in train_source
    assert "keyword_sequence_margin_loss" in train_source
    assert "load_keyword_operating_points" in train_source
    assert '"keyword_operating_points"' in train_source
    assert '"keyword_margin_profile_sha256"' in train_source
    assert '"keywords_sha256"' in train_source
    assert '"keyword_sequence_margin"' in train_source
    assert '"keyword_sequence_margin_loss_weight"' in train_source
    assert "keywords: pathlib.Path" in iterate_source
    assert '"--keywords"' in iterate_source
    assert "keywords=keywords" in iterate_source
    assert "DECODER_CONFIDENCE_THRESHOLD = 0.55" in margin_source
    assert "_decoder_sequence_log_confidence" in margin_source
    assert "keyword_operating_points" in margin_source
    assert "positive_floors" in margin_source
    assert "negative_ceilings" in margin_source
    assert "wake-on-any-keyword" in margin_source
    assert "test_sequence_margin: ok" in margin_test_source
    assert "Per-keyword thresholds are independent" in margin_test_source
    assert "stricter negative margin on keyword 2" in margin_test_source
    assert "invalid keyword operating point must fail closed" in margin_test_source

    keyword_path = ROOT / str(formal["keywords"])
    keyword_rows = [
        raw.split("\t")
        for raw in keyword_path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]
    assert [(int(row[0]), row[1]) for row in keyword_rows] == [
        (1, "你好小窝"),
        (2, "小窝小窝"),
    ]
    assert {float(row[2]) for row in keyword_rows} == {0.55}
    margin_profile_path = keyword_path.with_suffix(keyword_path.suffix + ".margin.json")
    margin_profile = json.loads(margin_profile_path.read_text(encoding="utf-8"))
    assert int(margin_profile["schema_version"]) == 1
    assert set(margin_profile["keywords"]) == {"1", "2"}
    assert margin_profile["keywords"]["1"]["text"] == "你好小窝"
    assert margin_profile["keywords"]["2"]["text"] == "小窝小窝"
    for keyword_id in ("1", "2"):
        item = margin_profile["keywords"][keyword_id]
        assert float(item["positive_margin"]) >= 0.05
        assert float(item["negative_margin"]) >= 0.05

    # Interaction curriculum is adaptive rather than a hard far/rear floor. Keep
    # playback anti-starvation while explicitly allowing pairwise/triple domain
    # weights and per-keyword interaction feedback.
    assert 'dimension == "playback"' in curriculum_source
    assert 'weights["playback"] = max' in curriculum_source
    assert '"distance_azimuth_snr"' in curriculum_source
    assert '"keyword_dimension_weights"' in curriculum_source
    assert '"keyword_worst_domains"' in curriculum_source
    assert 'weights["far"] = max(weights["far"], weights["mid"])' not in curriculum_source
    assert 'weights["rear"] = max(weights["rear"], *peer_weights)' not in curriculum_source

    qualification_seed = int(formal["qualification_holdout_seed"])
    retired_qualification_seeds = [
        int(value) for value in formal["retired_qualification_holdout_seeds"]
    ]
    assert retired_qualification_seeds
    assert retired_qualification_seeds == sorted(set(retired_qualification_seeds))
    assert retired_qualification_seeds == list(
        range(retired_qualification_seeds[0], qualification_seed)
    )
    assert qualification_seed == retired_qualification_seeds[-1] + 1
    assert qualification_seed not in retired_qualification_seeds
    assert int(formal["far_holdout_round_namespace"]) == 3000000
    assert [int(value) for value in formal["retired_far_holdout_round_namespaces"]] == [
        1000000,
        2000000,
    ]
    replay = {
        tuple(str(token) for token in item["tokens"]): int(item["examples"])
        for item in formal["domain_iteration"]["hard_negative_replay"]
    }
    assert replay[("ni3", "hao3", "xiao3")] >= 24
    assert replay[("xiao3", "wo1", "xiao3")] >= 24
    assert replay[("xiao3", "wo1")] >= 24
    positive_stress = formal["domain_iteration"]["positive_stress_replay"]
    assert {int(item["keyword_id"]) for item in positive_stress} == {1, 2}
    assert all(item["focus"] == "adaptive" for item in positive_stress)
    assert all(item["fallback"]["distance_bin"] == "5m" for item in positive_stress)
    assert all(item["fallback"]["azimuth"] == "rear" for item in positive_stress)
    assert all(item["fallback"]["snr"] == "critical" for item in positive_stress)
    assert "normalize_retired_qualification_seeds" in qualification_source
    assert "retired_exposed_qualification_seeds" in qualification_source
    assert "overlapping_exposed_active_wav_sha256" in qualification_source
    assert "active qualification overlaps" in qualification_source
    assert "retired-qualification-scratch" in qualification_source

    assert "--hard-negative-rate-per-minute 0.0" in model_training_workflow
    assert "actual_injection_rate_per_minute" in model_training_workflow
    assert "minimum_payload_gap_seconds" in model_training_workflow
    assert "observed_min_payload_gap_seconds" in model_training_workflow
    assert "semantic-negative-boundary-v1" in model_training_workflow

    module_path = ROOT / "training" / "build_container.py"
    spec = importlib.util.spec_from_file_location("build_container", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = "registry.example/kws-training@sha256:" + "a" * 64
    assert module.validate_base_image(valid) == valid
    for invalid in ("python:3.12", "image@sha256:abc", "image:latest"):
        try:
            module.validate_base_image(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mutable/invalid base accepted: {invalid}")
    print("test_training_supply_chain: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
