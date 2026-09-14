#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "run_gru_development.py"
POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-loop.json"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module(SCRIPT, "gru_train_acoustic_rotation_contract")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))

    assert policy["training_acoustic_seed_namespace"] == 193000019
    assert policy["training_acoustic_seed_stride"] == 1009
    first, stride = module._rotation(policy, 0)
    second, _ = module._rotation(policy, 1)
    third, _ = module._rotation(policy, 2)
    assert stride == 1009
    assert [first, second, third] == [193000019, 193001028, 193002037]
    assert len({first, second, third}) == 3
    assert first != int(policy["training_seed_namespace"])
    assert first != int(policy["candidate_freeze"]["fresh_validation_seed_namespace"])

    canonical = [
        {"split": "train", "wav_sha256": "a", "scene": {"distance_band": "near"}},
        {"split": "calibration", "wav_sha256": "b", "scene": {"distance_band": "mid"}},
        {"split": "test", "wav_sha256": "c", "scene": {"distance_band": "far"}},
        {"split": "train", "wav_sha256": "d", "scene": {"distance_band": "far"}},
        {"split": "qualification", "wav_sha256": "e", "scene": {"distance_band": "near"}},
    ]
    rotated = [
        {"split": "train", "wav_sha256": "x", "scene": {"distance_band": "mid"}},
        {"split": "calibration", "wav_sha256": "y", "scene": {"distance_band": "far"}},
        {"split": "train", "wav_sha256": "z", "scene": {"distance_band": "near"}},
    ]
    merged = module._merge_domain_rows(canonical, rotated)
    assert [row["wav_sha256"] for row in merged] == ["x", "b", "c", "z", "e"]
    for index in (1, 2, 4):
        assert merged[index] is canonical[index]

    bad = dict(policy)
    bad["training_acoustic_seed_namespace"] = policy["candidate_freeze"][
        "fresh_validation_seed_namespace"
    ]
    try:
        module._rotation(bad, 0)
    except ValueError as exc:
        assert "overlaps model/fresh namespace" in str(exc)
    else:
        raise AssertionError("fresh namespace overlap was accepted")

    source = SCRIPT.read_text(encoding="utf-8")
    assert "renderer.generate_dataset = _reuse_canonical_base" in source
    assert 'shutil.copy2(rotated / "train.tsv", output / "train.tsv")' in source
    assert '"base_utterance_reused": True' in source
    assert '"evaluation_seed_rotated": False' in source
    assert "training_code_sha256" in source

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        path = root / "rows.jsonl"
        module._write_jsonl(path, merged)
        roundtrip = module._read_jsonl(path)
        assert [row["wav_sha256"] for row in roundtrip] == ["x", "b", "c", "z", "e"]

    print("GRU train acoustic rotation contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
