from __future__ import annotations

import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.training_request import (
    EVIDENCE_CLASS as TRAINING_INVOCATION_EVIDENCE_CLASS,
    self_test as training_request_self_test,
    verify_request,
)
from training.verify_training_entry_contract import read_json, verify


def main() -> int:
    training_request_self_test()
    request_path = ROOT / ".github/triggers/model-training-request.json"
    verified_request = verify_request(request_path)

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
        "replay_provider_identity_sha256": "5" * 64,
        "replay_provider_execution_identity_sha256": "6" * 64,
        "replay_provider_semantic_identity_policy": "product-replay-provider-semantic-v1",
        "replay_provider_semantic_contract_sha256": "7" * 64,
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
        "provider_identity_sha256": "5" * 64,
        "provider_execution_identity_sha256": "6" * 64,
        "provider_semantic_identity_policy": "product-replay-provider-semantic-v1",
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

    trainer = (ROOT / "training/train_ctc.py").read_text(encoding="utf-8")
    assert "ordered_token_sample_weighting" in trainer
    assert 'SAMPLE_WEIGHT_NORMALIZATION_POLICY = "dataset-mean-sample-weight-v1"' in trainer
    assert "normalized_weighted_mean(" in trainer
    assert "normalization_mean_weight=float(" in trainer
    assert 'PATH_PURITY_LOSS_WEIGHT_DEFAULT = 0.0' in (
        ROOT / "training/objective_contract.py"
    ).read_text(encoding="utf-8")
    assert "--path-purity-loss-weight" in trainer
    assert "--path-purity-margin" in trainer
    assert 'path_purity_policy": PATH_PURITY_POLICY' in trainer
    base_iterator = (ROOT / "training/iterate_domain.py").read_text(encoding="utf-8")
    refinement = (ROOT / "training/adversarial_refinement.py").read_text(
        encoding="utf-8"
    )
    assert "optional_objective_cli_args(train)" in base_iterator
    assert "optional_objective_cli_args(train)" in refinement

    workflow = (ROOT / ".github/workflows/model-training.yml").read_text(encoding="utf-8")
    materializer = (ROOT / "training/materialize_governed_product_base.sh").read_text(encoding="utf-8")
    config_materializer = (ROOT / "training/materialize_product_training_config.py").read_text(
        encoding="utf-8"
    )
    assert "Materialize governed product speech-like training base" in workflow
    assert "bash training/materialize_governed_product_base.sh" in workflow
    assert "training/materialize_product_training_config.py" in materializer
    assert "--replay-provider-summary" in materializer
    assert "--replay-provider-contract" in materializer
    assert "product-replay-provider-semantic-v1.json" in materializer
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
    preflight_workflow = (
        ROOT / ".github/workflows/model-training-preflight.yml"
    ).read_text(encoding="utf-8")
    assert "governed preflight request PR must be request-only" in preflight_workflow
    assert 'git diff --name-only "$BASE_SHA" "$HEAD_SHA"' in preflight_workflow
    assert 'if [[ "$path" != ".github/triggers/model-training-request.json" ]]' in preflight_workflow
    assert "--provider-only" in materializer
    assert "--replay-provider" in materializer
    assert "--voice-inventory" in materializer
    assert "product-replay-provider" in materializer
    assert '"reuse_clean_across_rounds": True' in config_materializer

    print("test_product_training_entry_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
