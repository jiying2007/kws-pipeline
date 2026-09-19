from __future__ import annotations

import copy
import pathlib

from training.verify_training_entry_contract import read_json, verify

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    source = ROOT / "configs/training/xiaowo.torch-domain.json"
    shipping = ROOT / "configs/shipping.xiaowo.json"
    keywords = ROOT / "keywords/zh_cn_example.tsv"

    # The source/template config intentionally still has the legacy tone
    # generator. It must not qualify as a governed product-candidate input.
    try:
        verify(
            source,
            shipping,
            keywords,
            require_product_speech_like_base=True,
        )
    except ValueError as exc:
        assert "product_candidate_data" in str(exc)
    else:
        raise AssertionError("source tone template unexpectedly qualified for product training")

    config = copy.deepcopy(read_json(source))
    config["product_candidate_data"] = {
        "schema_version": 1,
        "policy": "external-speech-like-product-base-v1",
        "release_tag": "speech-like-base-fixture",
        "external_base_bundle_sha256": "1" * 64,
        "provider_identity_sha256": "2" * 64,
        "tone_fallback_allowed": False,
        "protected_evidence_used": False,
    }
    config["generator"]["external_base_dataset"] = {
        split: {
            "index": f"fixture/{split}/dataset-index.jsonl",
            "summary": f"fixture/{split}/dataset-summary.json",
            "index_sha256": "3" * 64,
            "summary_sha256": "4" * 64,
        }
        for split in ("train", "calibration", "test", "qualification")
    }
    import tempfile, json
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "effective.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        result = verify(
            path,
            shipping,
            keywords,
            require_product_speech_like_base=True,
        )
        assert result["product_speech_like_base_required"] is True
        assert result["product_external_base_bundle_sha256"] == "1" * 64

    workflow = (ROOT / ".github/workflows/model-training.yml").read_text(encoding="utf-8")
    assert "Materialize governed product speech-like training base" in workflow
    assert "materialize_product_training_config.py" in workflow
    assert "--require-product-speech-like-base" in workflow
    assert '--config "$KWS_EFFECTIVE_TRAINING_CONFIG"' in workflow
    assert "--config configs/training/xiaowo.torch-domain.json" not in workflow
    assert "governed model training must be dispatched from main" in workflow

    print("test_product_training_entry_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
