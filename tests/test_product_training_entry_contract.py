from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.training_request import (
    EVIDENCE_CLASS as TRAINING_INVOCATION_EVIDENCE_CLASS,
    self_test as training_request_self_test,
    verify_request,
)
from training.verify_training_entry_contract import read_json, verify
from training import base_stage_receipt
from training.verify_final_training_gates import PREFLIGHT_POLICY, verify as verify_final_gates



def verify_evidence_json_types() -> None:
    """Exercise the receipt and final gate with serialized temporary fixtures."""
    checked = 0
    missing = object()

    def reject(fn, label: str) -> None:
        nonlocal checked
        try:
            fn()
        except ValueError:
            checked += 1
            return
        raise AssertionError(f"malformed training evidence was accepted: {label}")

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        work = root / "base"
        work.mkdir()
        config_path = root / "base-config.json"
        config = {"qualification_holdout_seed": 271843}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (work / "domain-loop-manifest.json").write_text(
            json.dumps({"records": [{"round": 0}]}), encoding="utf-8"
        )
        receipt_path = work / "base-stage-receipt.json"
        github_output = root / "github-output.txt"
        with mock.patch.dict(os.environ, {"GITHUB_SHA": "1" * 40}), mock.patch.object(
            base_stage_receipt, "git_head", return_value="1" * 40
        ):
            for exit_code in (0, 1):
                baseline = base_stage_receipt.write_receipt(
                    config_path=config_path, work=work, output=receipt_path,
                    iteration_exit_code=exit_code,
                )
                verified = base_stage_receipt.verify_receipt(
                    config_path=config_path, work=work, receipt_path=receipt_path,
                    github_output=None,
                )
                if verified != baseline:
                    raise AssertionError("valid base receipt changed during verification")
                checked += 1
            baseline = base_stage_receipt.write_receipt(
                config_path=config_path, work=work, output=receipt_path, iteration_exit_code=0,
            )
            for field in ("schema_version", "qualification_holdout_seed", "iteration_exit_code",
                          "formal_seed_consumed"):
                value = baseline[field]
                invalid = (str(value), float(value), None, [], {}, missing)
                invalid += ("false", "", 0, 1) if isinstance(value, bool) else (True, False, -1)
                for bad in invalid:
                    changed = copy.deepcopy(baseline)
                    if bad is missing:
                        changed.pop(field)
                    else:
                        changed[field] = bad
                    receipt_path.write_text(json.dumps(changed), encoding="utf-8")
                    github_output.write_text("prior=ok\n", encoding="utf-8")
                    reject(
                        lambda: base_stage_receipt.verify_receipt(
                            config_path=config_path, work=work, receipt_path=receipt_path,
                            github_output=github_output,
                        ),
                        f"base receipt {field}={bad!r}",
                    )
                    if github_output.read_text(encoding="utf-8") != "prior=ok\n":
                        raise AssertionError("invalid base receipt wrote continuation outputs")
            receipt_path.write_text(json.dumps(baseline), encoding="utf-8")
            receipt_bytes = receipt_path.read_bytes()
            for bad in ("0", 0.0, True, False, None, -1, 2):
                reject(
                    lambda: base_stage_receipt.write_receipt(
                        config_path=config_path, work=work, output=receipt_path,
                        iteration_exit_code=bad,
                    ),
                    f"base writer iteration_exit_code={bad!r}",
                )
                if receipt_path.read_bytes() != receipt_bytes:
                    raise AssertionError("invalid base writer input replaced receipt")
            for bad in ("271843", 271843.0, True, False, None, -1):
                config_path.write_text(json.dumps({"qualification_holdout_seed": bad}), encoding="utf-8")
                reject(
                    lambda: base_stage_receipt.write_receipt(
                        config_path=config_path, work=work, output=receipt_path, iteration_exit_code=0,
                    ),
                    f"base writer qualification_holdout_seed={bad!r}",
                )
                if receipt_path.read_bytes() != receipt_bytes:
                    raise AssertionError("invalid base seed replaced receipt")

        final_root = root / "final"
        paths = {
            "config": final_root / "config.json",
            "summary": final_root / "training-run-summary.json",
            "cohort": final_root / "qualification-dataset/qualification-cohort.json",
            "far": final_root / "best/far-holdout-cohort.json",
            "shadow": final_root / "shadow-qualification/summary.json",
            "refinement": final_root / "adversarial-refinement/summary.json",
        }
        documents = {
            "config": {
                "retired_qualification_holdout_seeds": [11],
                "far_holdout_round_namespace": 20,
                "retired_far_holdout_round_namespaces": [19],
            },
            "summary": {
                "qualified": True, "training_exit_code": 0, "iteration_exit_code": 0,
                "best_round": 0, "best_frontend": "logmel", "artifacts": {},
                "candidate_selection": {
                    "qualification_used_for_selection": False,
                    "far_holdout_used_for_training": False, "far_holdout_used_for_selection": False,
                    "overlapping_training_holdout_wav_sha256": 0,
                    "overlapping_retired_active_far_holdout_wav_sha256": 0,
                    "far_holdout_round_namespace": 20, "retired_far_holdout_round_namespaces": [19],
                    "far_holdout_clip_count": 3, "far_holdout_sha256": "a" * 64,
                },
            },
            "cohort": {
                "formal_preflight_policy": PREFLIGHT_POLICY,
                "shadow_qualification_required": True, "shadow_qualification_qualified": True,
                "adversarial_refinement_required": True, "adversarial_formal_qualification_used": False,
                "model_provenance_adversarial_manifest_verified": True,
                "adversarial_overlap_guard_included": True, "failure_replay_required": True,
                "failure_replay_formal_qualification_used": False,
                "model_provenance_failure_replay_manifest_verified": True, "seed_disjoint": True,
                "qualification_seed": 21, "training_seed": 22,
                "overlapping_retired_active_wav_sha256": 0,
                "overlapping_exposed_active_wav_sha256": 0,
                "overlapping_development_active_wav_sha256": 0,
                "expected_wakes": 256, "retired_exposed_qualification_seeds": [11],
            },
            "far": {
                "active_namespace": 20, "retired": [{"namespace": 19}],
                "overlapping_retired_active_wav_sha256": 0, "active_clip_count": 3,
                "active_manifest_sha256": "a" * 64,
            },
            "shadow": {"qualified": True, "formal_qualification_seed_consumed": False},
            "refinement": {"qualified": True, "formal_qualification_used": False},
        }

        def materialize(values: dict) -> None:
            for key, path in paths.items():
                if key == "summary":
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(values[key], sort_keys=True) + "\n", encoding="utf-8")
            summary = copy.deepcopy(values["summary"])
            digest = hashlib.sha256(paths["far"].read_bytes()).hexdigest()
            summary["candidate_selection"]["far_holdout_cohort_sha256"] = digest
            summary["artifacts"]["far-holdout-cohort.json"] = digest
            paths["summary"].write_text(json.dumps(summary), encoding="utf-8")

        materialize(documents)
        result = verify_final_gates(paths["config"], final_root)
        if result != {"qualified": True, "best_round": 0, "best_frontend": "logmel",
                      "qualification_seed": 21, "far_holdout_clip_count": 3}:
            raise AssertionError("valid final gate result changed")
        checked += 1

        scalar_paths = []
        array_paths = []

        def collect(value, path=()) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    collect(child, path + (key,))
            elif isinstance(value, list):
                array_paths.append(path)
                for index, child in enumerate(value):
                    collect(child, path + (index,))
            elif isinstance(value, (bool, int)):
                scalar_paths.append((path, value))

        collect(documents)
        cases = []
        for path, value in scalar_paths:
            invalid = (str(value), float(value), None, [], {}, missing)
            invalid += ("false", "", 0, 1) if isinstance(value, bool) else (True, False, -1)
            for bad in invalid:
                cases.append((path, bad))
        for path in array_paths:
            for bad in (None, {}, "", 0, True, missing):
                cases.append((path, bad))
        for bad in (None, [], "", 0, True):
            cases.append((("far", "retired", 0), bad))

        for path, bad in cases:
            changed = copy.deepcopy(documents)
            parent = changed
            for key in path[:-1]:
                parent = parent[key]
            if bad is missing:
                del parent[path[-1]]
            else:
                parent[path[-1]] = bad
            materialize(changed)
            reject(lambda: verify_final_gates(paths["config"], final_root),
                   f"final {path}={bad!r}")
        materialize(documents)
        if verify_final_gates(paths["config"], final_root)["qualified"] is not True:
            raise AssertionError("invalid-case checks damaged valid fixture")
    print(f"strict training evidence JSON types: {checked} checks passed")


def main() -> int:
    verify_evidence_json_types()
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
    config["generator"]["external_base_admission_mode"] = "reviewed-real-v1"
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
        assert result["actual_review_receipts_verified"] is False
        config["generator"]["external_base_admission_mode"] = "historical-frozen-diagnostic-v1"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            verify(path, shipping, keywords, require_product_speech_like_base=True)
        except ValueError as exc:
            assert "historical frozen base is diagnostic-only" in str(exc)
        else:
            raise AssertionError("historical base was admitted for new product training")

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
