from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from product_development_experiment import materialize, self_test, verify_spec  # noqa: E402


def main() -> int:
    self_test()

    workflow = (
        ROOT / ".github/workflows/product-development-experiment.yml"
    ).read_text(encoding="utf-8")
    assert "source_policy':'exact-pr-head'" in workflow
    assert "development_only':True" in workflow
    assert "protected_evidence_used':False" in workflow
    assert ".github/triggers/model-training-request.json" in workflow
    assert "development experiment must not change governed model-training request" in workflow
    assert "configs/training/xiaowo.torch-domain.json" in workflow
    assert "configs/training/product-speech-like-base-v1.json" in workflow
    assert ".github/workflows/model-training.yml" in workflow
    assert "--defer-qualification" in workflow
    assert "--stop-after-development-eval" in workflow
    assert "kws_posterior_dump" in workflow
    assert "kws_decoder_replay" in workflow
    assert workflow.count("--posterior-dump build/kws_posterior_dump") == 3
    assert workflow.count("--decoder-replay build/kws_decoder_replay") == 3
    assert workflow.count('--posterior-cache "$KWS_EXPERIMENT_WORK/posterior-cache"') == 3
    assert "posterior-replay-cache-summary-v1" in workflow
    assert "'posterior_cache': posterior_cache is not None" in workflow
    assert "'posterior_cache':posterior_cache" in workflow
    assert "Diagnose saturated source threshold grid" in workflow
    assert "select_refinement_source" in workflow
    assert "development-threshold-saturation-trigger-v1" in workflow
    assert "refinement-source-saturation-v1" in workflow
    assert "tools/diagnose_kws_threshold_operating_curve.py" in workflow
    assert "round(0.20 + 0.05*index,2) for index in range(15)" in workflow
    assert "'threshold_saturation_trigger': threshold_trigger is not None" in workflow
    assert "required['threshold_wide_sweep']=threshold_sweep is not None" in workflow
    assert "'threshold_saturation_trigger':threshold_trigger" in workflow
    assert "'threshold_wide_sweep':threshold_sweep" in workflow
    assert "threshold-saturation-trigger.json" in workflow
    assert "threshold-wide-sweep.json" in workflow
    assert "posterior-cache/**/*.kwtr" in workflow
    assert "posterior-cache/**/*.json" in workflow
    assert "qualification.references.jsonl" not in workflow
    assert "formal-qualification" not in workflow
    assert workflow.count("ref: ${{ github.event.pull_request.head.sha }}") == 2
    assert "product-development-experiment-${{ github.event.pull_request.base.sha }}-${{ github.event.pull_request.head.sha }}" in workflow
    assert "steps.eligibility.outputs.eligible == 'true'" in workflow
    assert "'infrastructure_complete':all(required.values())" in workflow
    assert "'required_evidence_present':required" in workflow
    assert "'acoustic_alignment': acoustic is not None" in workflow
    assert "required['refinement_summary']=refinement is not None" in workflow
    assert "KWS_EXPERIMENT_RECEIPT: build/product-development-experiment-receipt.json" in workflow
    assert "load_optional(pathlib.Path(os.environ['KWS_EXPERIMENT_RECEIPT']))" in workflow
    assert "build/product-development-experiment-receipt.json" in workflow
    assert "build/product-development-experiment/experiment-receipt.json" not in workflow
    materializer = (
        ROOT / "training/materialize_governed_product_base.sh"
    ).read_text(encoding="utf-8")
    assert "retry_to_file()" in materializer
    assert "retry_release_download()" in materializer
    assert "KWS_DOWNLOAD_RETRY_ATTEMPTS" in materializer
    assert "--clobber" in materializer

    with tempfile.TemporaryDirectory(prefix="experiment-contract-") as tmp:
        root = pathlib.Path(tmp)
        spec = root / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "path-purity-contract-v1",
                    "development_only": True,
                    "source_policy": "exact-pr-head",
                    "protected_evidence_used": False,
                    "reason": "exercise bounded PR-head materialization",
                    "config_overrides": {
                        "train.path_purity_loss_weight": 0.1,
                        "train.path_purity_margin": 0.15,
                        "train.ordered_token_scope": "exact-configured-wake-targets-v1",
                    },
                }
            ),
            encoding="utf-8",
        )
        effective = root / "effective.json"
        effective.write_text(
            json.dumps(
                {
                    "product_candidate_data": {
                        "policy": "external-speech-like-product-base-v1",
                        "tone_fallback_allowed": False,
                        "protected_evidence_used": False,
                    },
                    "train": {"epochs": 36, "warm_start_epochs": 12},
                    "domain_iteration": {
                        "max_rounds": 4,
                        "min_rounds": 2,
                        "patience": 2,
                        "stop_on_gate": True,
                        "adversarial_lexicon": {
                            "enabled": True,
                            "refinement_epochs": 12,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        output = root / "experiment.json"
        receipt = root / "receipt.json"
        result = materialize(
            spec_path=spec,
            effective_config_path=effective,
            output_path=output,
            receipt_path=receipt,
            base_sha="1" * 40,
            head_sha="2" * 40,
        )
        cfg = json.loads(output.read_text(encoding="utf-8"))
        assert cfg["train"]["epochs"] == 12
        assert cfg["train"]["warm_start_epochs"] == 6
        assert cfg["train"]["path_purity_loss_weight"] == 0.1
        assert cfg["train"]["path_purity_margin"] == 0.15
        assert (
            cfg["train"]["ordered_token_scope"]
            == "exact-configured-wake-targets-v1"
        )
        assert cfg["domain_iteration"]["max_rounds"] == 2
        assert cfg["domain_iteration"]["min_rounds"] == 2
        assert cfg["domain_iteration"]["adversarial_lexicon"]["refinement_epochs"] == 6
        assert cfg["development_experiment"]["pr_base_sha"] == "1" * 40
        assert cfg["development_experiment"]["pr_head_sha"] == "2" * 40
        assert result["source_policy"] == "exact-pr-head"
        assert result["protected_evidence_used"] is False

        bad = json.loads(spec.read_text(encoding="utf-8"))
        bad["config_overrides"] = {
            "train.ordered_token_scope": "not-a-supported-scope"
        }
        spec.write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_spec(spec)
        except ValueError as exc:
            assert "must be one of" in str(exc)
        else:
            raise AssertionError("invalid ordered-token scope was accepted")

        bad = json.loads(spec.read_text(encoding="utf-8"))
        bad["config_overrides"] = {"train.epochs": 99}
        spec.write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_spec(spec)
        except ValueError as exc:
            assert "not allowed" in str(exc)
        else:
            raise AssertionError("non-whitelisted experiment override was accepted")

    print("product development experiment contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
