from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from product_development_experiment import materialize, self_test, verify_spec  # noqa: E402
from iterate_domain import posterior_replay_cli_args  # noqa: E402


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
    assert workflow.count("--posterior-dump build/kws_posterior_dump") == 2
    assert workflow.count("--decoder-replay build/kws_decoder_replay") == 2
    assert workflow.count('--posterior-cache "$KWS_EXPERIMENT_WORK/posterior-cache"') == 2
    assert "'--posterior-dump','build/kws_posterior_dump'" in workflow
    assert "'--decoder-replay','build/kws_decoder_replay'" in workflow
    assert "'--posterior-cache',str(root/'posterior-cache')" in workflow
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
    assert "Diagnose recalibrated decoder retention curve" in workflow
    assert "tools/diagnose_decoder_retention_recalibrated_curve.py" in workflow
    assert "'--development-manifest',str(root/'domain-loop-manifest.json')" in workflow
    assert "'--development-authority-receipt',os.environ['KWS_EXPERIMENT_RECEIPT']" in workflow
    assert "'--round-index',str(round_index)" in workflow
    assert "'--diagnostic-round-selection-policy','refinement-source-retention-v1'" in workflow
    retention_diagnostic = (
        ROOT / "tools/diagnose_decoder_retention_recalibrated_curve.py"
    ).read_text(encoding="utf-8")
    assert "development_round_evidence" in retention_diagnostic
    assert "calibration references do not match development manifest" in retention_diagnostic
    assert "test references do not match development manifest" in retention_diagnostic
    assert '"development_round_evidence": development_evidence' in retention_diagnostic
    assert "retentions=['0.85','0.90','0.94','0.97','0.99']" in workflow
    assert "'shipping_default_retention':0.94" in workflow
    assert "'selection_feedback_allowed':False" in workflow
    assert "'decoder_retention_recalibrated_curve': retention_curve is not None" in workflow
    assert "'decoder_retention_recalibrated_curve':retention_curve" in workflow
    assert "decoder-retention-recalibrated-curve.json" in workflow
    assert "Resolve optional decoder policy diagnostic" in workflow
    assert "experiment_id.startswith('decoder-policy-replay-')" in workflow
    assert "Diagnose decoder blank/fuzzy policy grid" in workflow
    assert "tools/diagnose_decoder_policy_replay.py" in workflow
    assert "'--blank-retentions','0.70','0.85','0.95'" in workflow
    assert "'--fuzzy-child-cost-logs','-8.25','-4.0'" in workflow
    assert "'--coordinate-rounds','1'" in workflow
    assert "'--diagnostic-round-selection-policy','refinement-source-decoder-policy-v1'" in workflow
    assert "required['decoder_policy_replay_grid']=decoder_policy is not None" in workflow
    assert "'decoder_policy_replay_grid':decoder_policy" in workflow
    assert "decoder-policy-replay-grid.json" in workflow
    assert "candidates/*/model.kwm" in workflow
    assert "datasets/round-*/calibration.references.jsonl" in workflow
    assert "datasets/round-*/test.references.jsonl" in workflow
    assert "posterior-cache/**/*.kwtr" in workflow
    assert "posterior-cache/**/*.json" in workflow
    assert "qualification.references.jsonl" not in workflow
    assert "formal-qualification" not in workflow
    assert workflow.count("ref: ${{ github.event.pull_request.head.sha }}") == 2
    assert "product-development-experiment-${{ github.event.pull_request.base.sha }}-${{ github.event.pull_request.head.sha }}" in workflow
    assert "steps.eligibility.outputs.eligible == 'true' && steps.decoder_policy.outputs.enabled != 'true'" in workflow
    assert "and not decoder_policy_enabled" in workflow
    assert "'refinement_skipped_for_decoder_policy':" in workflow
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

    replay_args = posterior_replay_cli_args(
        (pathlib.Path("dump"), pathlib.Path("replay"), pathlib.Path("cache")),
        decoder_blank_retention=0.85,
        decoder_fuzzy_child_cost_log=-4.0,
    )
    assert replay_args[-4:] == [
        "--decoder-blank-retention",
        "0.85",
        "--decoder-fuzzy-child-cost-log",
        "-4.0",
    ]
    for kwargs, message in (
        ({"decoder_blank_retention": 1.0}, "decoder_blank_retention"),
        ({"decoder_fuzzy_child_cost_log": 0.1}, "decoder_fuzzy_child_cost_log"),
    ):
        try:
            posterior_replay_cli_args(
                (pathlib.Path("dump"), pathlib.Path("replay"), pathlib.Path("cache")),
                **kwargs,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"invalid replay override accepted: {kwargs}")

    decoder_policy_diagnostic = (
        ROOT / "tools/diagnose_decoder_policy_replay.py"
    ).read_text(encoding="utf-8")
    assert "development_round_evidence" in decoder_policy_diagnostic
    assert '"selection_feedback_allowed": False' in decoder_policy_diagnostic
    assert "thresholds_recalibrated_per_policy_point" in decoder_policy_diagnostic

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
