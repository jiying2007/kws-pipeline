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

    # Once a qualification seed has been inspected it becomes development
    # evidence. Keep all exposed cohorts explicit and prove zero active overlap.
    assert int(formal["qualification_holdout_seed"]) == 271833
    assert [int(value) for value in formal["retired_qualification_holdout_seeds"]] == [
        271828,
        271829,
        271830,
        271831,
        271832,
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

    # Adaptive curriculum must preserve product-risk floors rather than letting
    # a transient development slice reverse the configured far/rear priority.
    assert 'dimension == "distance"' in curriculum_source
    assert 'weights["far"] = max(weights["far"], weights["mid"])' in curriculum_source
    assert 'dimension == "azimuth"' in curriculum_source
    assert 'weights["rear"] = max(weights["rear"], *peer_weights)' in curriculum_source

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
