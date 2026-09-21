from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.training_request import (
    EVIDENCE_CLASS as TRAINING_INVOCATION_EVIDENCE_CLASS,
    self_test as training_request_self_test,
    verify_request,
)
from training.keyword_set_contract import (
    sha256_file,
    verify_keyword_set_contract,
)
from training.verify_training_entry_contract import read_json, verify


def main() -> int:
    training_request_self_test()
    request_path = ROOT / ".github/triggers/model-training-request.json"
    verified_request = verify_request(request_path)

    source = ROOT / "configs/training/xiaowo.torch-domain.json"
    shipping = ROOT / "configs/shipping.xiaowo.json"
    keywords = ROOT / "keywords/zh_cn_example.tsv"
    keyword_contract = ROOT / "configs/training/xiaowo-keyword-set-v1.json"
    keyword_identity = verify_keyword_set_contract(keyword_contract)

    with tempfile.TemporaryDirectory(prefix="custom-keyword-contract-") as td:
        fixture_root = pathlib.Path(td)
        (fixture_root / "keywords").mkdir()
        fixture_tokens = fixture_root / "keywords/tokens.txt"
        fixture_keywords = fixture_root / "keywords/custom.tsv"
        fixture_contract = fixture_root / "keyword-set.json"
        fixture_tokens.write_text(
            "<blk> 0\na 1\nb 2\nc 3\nd 4\n",
            encoding="utf-8",
        )
        fixture_keywords.write_text(
            "0\t自定义一\t0.50\ta b\n"
            "7\t自定义二\t0.55\tb c\n"
            "42\t自定义三\t0.60\tc d\n",
            encoding="utf-8",
        )
        fixture_contract.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": "kws-keyword-set-contract-v1",
                    "contract_id": "fixture-three-wake-v1",
                    "locale": "zh-CN",
                    "tokens_path": "keywords/tokens.txt",
                    "keywords_path": "keywords/custom.tsv",
                    "keywords": [
                        {
                            "id": 0,
                            "text": "自定义一",
                            "threshold": 0.50,
                            "tokens": ["a", "b"],
                        },
                        {
                            "id": 7,
                            "text": "自定义二",
                            "threshold": 0.55,
                            "tokens": ["b", "c"],
                        },
                        {
                            "id": 42,
                            "text": "自定义三",
                            "threshold": 0.60,
                            "tokens": ["c", "d"],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        custom = verify_keyword_set_contract(
            fixture_contract,
            root=fixture_root,
            expected_tokens_path=fixture_tokens,
            expected_keywords_path=fixture_keywords,
        )
        assert custom["keyword_ids"] == [0, 7, 42]
        assert custom["keyword_count"] == 3

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
        "keyword_set_contract_id": keyword_identity["contract_id"],
        "keyword_set_contract_path": "configs/training/xiaowo-keyword-set-v1.json",
        "keyword_set_contract_sha256": keyword_identity["contract_sha256"],
        "keyword_set_semantic_sha256": keyword_identity["semantic_sha256"],
        "keyword_tsv_sha256": keyword_identity["keywords_sha256"],
        "tokens_sha256": keyword_identity["tokens_sha256"],
        "corpus_plan_path": "configs/training/speech-like-corpus-plan-v1.json",
        "corpus_plan_sha256": sha256_file(
            ROOT / "configs/training/speech-like-corpus-plan-v1.json"
        ),
        "external_base_bundle_sha256": "1" * 64,
        "provider_identity_sha256": "2" * 64,
        "replay_provider_identity_sha256": "2" * 64,
        "replay_backend": "command",
        "replay_train_voice_slots": 2,
        "replay_tone_allowed": False,
        "tone_fallback_allowed": False,
        "protected_evidence_used": False,
    }
    config["generator"]["tts"] = {
        "backend": "command",
        "command": [
            "/tmp/fake-tts",
            "--speaker={speaker_id}",
            "--scale={length_scale}",
            "--output={output}",
            "{text}",
        ],
        "speaker_profiles": [
            {"speaker_id": 0, "length_scale": 1.0},
            {"speaker_id": 1, "length_scale": 1.1},
        ],
        "provider_identity_sha256": "2" * 64,
        "replay_voice_scope": "train-only",
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
        assert result["keyword_set_contract_id"] == "xiaowo-dual-wake-v1"

        tampered = copy.deepcopy(config)
        tampered["product_candidate_data"]["keyword_set_semantic_sha256"] = "f" * 64
        tampered_path = pathlib.Path(td) / "tampered-effective.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            verify(
                tampered_path,
                shipping,
                keywords,
                require_product_speech_like_base=True,
            )
        except ValueError as exc:
            assert "keyword_set_semantic_sha256" in str(exc)
        else:
            raise AssertionError("tampered keyword semantic identity was accepted")

    trainer = (ROOT / "training/train_ctc.py").read_text(encoding="utf-8")
    assert "ordered_token_sample_weighting" in trainer
    assert 'SAMPLE_WEIGHT_NORMALIZATION_POLICY = "dataset-mean-sample-weight-v1"' in trainer
    assert "normalized_weighted_mean(" in trainer
    assert "normalization_mean_weight=float(" in trainer

    workflow = (ROOT / ".github/workflows/model-training.yml").read_text(encoding="utf-8")
    materializer = (ROOT / "training/materialize_governed_product_base.sh").read_text(encoding="utf-8")
    config_materializer = (ROOT / "training/materialize_product_training_config.py").read_text(
        encoding="utf-8"
    )
    assert "Materialize governed product speech-like training base" in workflow
    assert "bash training/materialize_governed_product_base.sh" in workflow
    assert "training/materialize_product_training_config.py" in materializer
    assert "--require-product-speech-like-base" in materializer
    assert "--require-product-speech-like-base" in workflow
    assert '--config "$KWS_EFFECTIVE_TRAINING_CONFIG"' in workflow
    assert "--config configs/training/xiaowo.torch-domain.json" not in workflow
    assert "governed model training must run from current main" in workflow
    assert "github.event_name == 'workflow_dispatch' || github.event_name == 'push'" in workflow
    assert ".github/triggers/model-training-request.json" in workflow
    assert "Verify versioned training request" in workflow
    assert TRAINING_INVOCATION_EVIDENCE_CLASS == "governed-model-training-invocation-v1"
    assert verified_request["schema_version"] == 1
    assert verified_request["trigger_policy"] == "run-on-protected-main-change"
    assert verified_request["purpose"] == "governed-product-candidate-training"
    assert verified_request["source_policy"] == "exact-current-main"
    assert "training/training_request.py verify" in workflow
    assert "training/training_request.py write-receipt" in workflow
    assert "--provider-only" in materializer
    assert "--replay-provider" in materializer
    assert "--voice-inventory" in materializer
    assert "product-replay-provider" in materializer
    assert '"reuse_clean_across_rounds": True' in config_materializer
    entry_guard = (
        ROOT / "training/verify_training_entry_contract.py"
    ).read_text(encoding="utf-8")
    assert "shipping wake words must remain exactly the two four-character phrases" not in entry_guard
    assert "kws-keyword-set-contract-v1" in (
        ROOT / "configs/training/xiaowo-keyword-set-v1.json"
    ).read_text(encoding="utf-8")

    print("test_product_training_entry_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
