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

    # Reproducible CPU training requires more than a fixed RNG seed. Pin the
    # pre-import BLAS/ATen dispatch environment to the common AVX2 baseline,
    # preserve the qualified two-thread intra-op path, pin one inter-op thread,
    # and disable MKLDNN before model/feature work. Keep the same literals in
    # model.py so exported training-code provenance attests the effective contract.
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

    # #121 proved that relative CTC sequence separation is directionally useful
    # but still mismatched the shipping decoder's actual confidence. The formal
    # auxiliary objective now operates in the same acoustic units used by
    # candidate_confidence = exp(acoustic_score / depth): a real wake must clear
    # threshold+margin, every non-wake/competing path must stay below
    # threshold-margin, and the worst path is never diluted by keyword count.
    assert "KEYWORD_SEQUENCE_MARGIN = 0.05" in train_source
    assert "KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT = 0.10" in train_source
    assert "keyword_sequence_margin_loss" in train_source
    assert '"keywords_sha256"' in train_source
    assert '"keyword_sequence_margin"' in train_source
    assert '"keyword_sequence_margin_loss_weight"' in train_source
    assert "keywords: pathlib.Path" in iterate_source
    assert '"--keywords"' in iterate_source
    assert "keywords=keywords" in iterate_source
    assert "DECODER_CONFIDENCE_THRESHOLD = 0.55" in margin_source
    assert "_decoder_sequence_log_confidence" in margin_source
    assert "torch.cummax" in margin_source
    assert "positive_floor" in margin_source
    assert "negative_ceiling" in margin_source
    assert "wake-on-any-keyword" in margin_source
    assert "test_sequence_margin: ok" in margin_test_source
    assert "Adding unrelated wake words must not dilute" in margin_test_source
    assert "Recall-side contract" in margin_test_source
    assert "impossible wake sequence" in margin_test_source

    # Until train_ctc carries per-keyword thresholds explicitly, keep the formal
    # decoder-confidence surrogate coupled to the exact threshold used by every
    # shipping keyword. A keyword TSV threshold change must fail this contract
    # instead of silently training against the wrong operating point.
    keyword_thresholds = {
        float(raw.split("\t")[2])
        for raw in (ROOT / str(formal["keywords"])).read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    }
    assert keyword_thresholds == {0.55}

    # #120 proved that forcing far/rear adaptive multiplier floors destabilizes
    # the FR/FA trade-off. Keep the earlier playback anti-starvation rule, but do
    # not silently reintroduce the rejected distance/azimuth hard floors.
    assert 'dimension == "playback"' in curriculum_source
    assert 'weights["playback"] = max' in curriculum_source
    assert 'weights["far"] = max(weights["far"], weights["mid"])' not in curriculum_source
    assert 'weights["rear"] = max(weights["rear"], *peer_weights)' not in curriculum_source

    # Once a qualification seed has been inspected it becomes development
    # evidence. Keep all exposed cohorts explicit and prove zero active overlap.
    assert int(formal["qualification_holdout_seed"]) == 271834
    assert [int(value) for value in formal["retired_qualification_holdout_seeds"]] == [
        271828,
        271829,
        271830,
        271831,
        271832,
        271833,
    ]
    assert int(formal["far_holdout_round_namespace"]) == 3000000
    assert [int(value) for value in formal["retired_far_holdout_round_namespaces"]] == [
        1000000,
        2000000,
    ]
    replay = {
        tuple(str(token) for token in item["tokens"]): int(item["examples"])
        for item in formal["domain_iteration"]["hard_negative_replay"]
    }
    assert replay[("ni3", "hao3", "xiao3")] == 24
    assert replay[("xiao3", "wo1", "xiao3")] == 24
    assert "normalize_retired_qualification_seeds" in qualification_source
    assert "retired_exposed_qualification_seeds" in qualification_source
    assert "overlapping_exposed_active_wav_sha256" in qualification_source
    assert "active qualification overlaps" in qualification_source
    assert "retired-qualification-scratch" in qualification_source

    # Continuous FAR must not synthesize a real wake phrase by placing two
    # individually-negative payloads inside the decoder retention window. The
    # formal gate uses only deterministic forced coverage, still exceeds 8
    # payloads/minute, and proves a >=2 s payload gap (>1.6 s retention test).
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
