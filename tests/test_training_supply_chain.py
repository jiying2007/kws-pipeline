from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "training" / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "training" / "requirements.lock").read_text(encoding="utf-8")
    model_source = (ROOT / "training" / "model.py").read_text(encoding="utf-8")
    train_source = (ROOT / "training" / "train_ctc.py").read_text(encoding="utf-8")
    assert "ARG KWS_TRAINING_BASE" in dockerfile
    assert "FROM ${KWS_TRAINING_BASE}" in dockerfile
    assert "pip install" not in dockerfile
    assert "pip install --upgrade" not in dockerfile
    assert "python==3.12.11" in lock
    assert "torch==2.13.0" in lock

    # Deterministic RNG/algorithm settings are insufficient when hosted runners
    # choose different Torch scheduler topologies.  Pin the training-only model
    # module before any model work and retain the effective topology in exported
    # provenance so model SHA comparisons are auditable across workflow runs.
    assert "TRAINING_TORCH_NUM_THREADS = 2" in model_source
    assert "TRAINING_TORCH_NUM_INTEROP_THREADS = 1" in model_source
    assert "torch.set_num_threads(TRAINING_TORCH_NUM_THREADS)" in model_source
    assert (
        "torch.set_num_interop_threads(TRAINING_TORCH_NUM_INTEROP_THREADS)"
        in model_source
    )
    assert '"torch_num_threads": int(torch.get_num_threads())' in train_source
    assert (
        '"torch_num_interop_threads": int(torch.get_num_interop_threads())'
        in train_source
    )
    assert "torch.use_deterministic_algorithms(True)" in train_source

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
