#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-promotion.yml"
VERIFIER = ROOT / "tools" / "verify_model_promotion_bundle.py"
TRAINING_WORKFLOW = ROOT / ".github" / "workflows" / "model-training.yml"
DIAGNOSTICS = ROOT / "tools" / "build_training_diagnostics.py"
REFINEMENT = ROOT / "training" / "adversarial_refinement.py"
ITERATE_DOMAIN = ROOT / "training" / "iterate_domain.py"
FEATURE_CACHE = ROOT / "training" / "feature_cached_trainer.py"
TRAINING_CONFIG = ROOT / "configs" / "training" / "xiaowo.torch-domain.json"
EXPORTER = ROOT / "training" / "export_model.py"
REPAIR = ROOT / "training" / "qualification_failure_replay.py"
ROBUSTNESS = ROOT / "eval" / "gate_robustness.py"
BASE_STAGE = ROOT / "training" / "base_stage_receipt.py"
ENTRY_CONTRACT = ROOT / "training" / "verify_training_entry_contract.py"
FINAL_GATES = ROOT / "training" / "verify_final_training_gates.py"
FINALIZER = ROOT / "training" / "finalize_domain_candidate.py"
TRAINING_REQUEST = ROOT / "training" / "training_request.py"
FAR_GATE = ROOT / "eval" / "run_continuous_far_gate.py"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing required contract: {needle!r}")


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    training = TRAINING_WORKFLOW.read_text(encoding="utf-8")
    diagnostics = DIAGNOSTICS.read_text(encoding="utf-8")
    refinement = REFINEMENT.read_text(encoding="utf-8")
    iterate_domain = ITERATE_DOMAIN.read_text(encoding="utf-8")
    feature_cache = FEATURE_CACHE.read_text(encoding="utf-8")
    training_config = TRAINING_CONFIG.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    repair = REPAIR.read_text(encoding="utf-8")
    robustness = ROBUSTNESS.read_text(encoding="utf-8")
    base_stage = BASE_STAGE.read_text(encoding="utf-8")
    entry_contract = ENTRY_CONTRACT.read_text(encoding="utf-8")
    final_gates = FINAL_GATES.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    training_request = TRAINING_REQUEST.read_text(encoding="utf-8")
    far_gate = FAR_GATE.read_text(encoding="utf-8")

    for needle in (
        "workflow_dispatch:",
        "training_run_id:",
        "expected_head_sha:",
        "contents: write",
        "actions: read",
        "cancel-in-progress: false",
        "persist-credentials: false",
        "actions/runs/${TRAINING_RUN_ID}",
        "run.get('name') != 'model-training'",
        "run.get('status') != 'completed'",
        "run.get('conclusion') != 'success'",
        "run.get('head_sha') != expected",
        "xiaowo-torch-domain-model",
        "expected exactly one live xiaowo-torch-domain-model artifact",
        "tools/verify_model_promotion_bundle.py",
        "--training-run-id \"$TRAINING_RUN_ID\"",
        "--expected-head-sha \"$EXPECTED_HEAD_SHA\"",
        "sha256sum -c MODEL_SHA256SUMS",
        "subject-checksums: dist/MODEL_SHA256SUMS",
        "gh release view \"$tag\" --repo \"$GITHUB_REPOSITORY\"",
        "gh release create \"$MODEL_RELEASE_TAG\"",
        "dist/*",
        "--target \"$EXPECTED_HEAD_SHA\"",
    ):
        require(workflow, needle, "model-promotion workflow")

    for needle in (
        'PREFLIGHT_POLICY = "shadow-adversarial-failure-formal-preflight-v2"',
        'DATA_POLICY = "train-only-balanced-mining-v1"',
        'ADVERSARIAL_POLICY = "balanced-strict-prefix-anchor-topk-v2"',
        'FAILURE_REPLAY_POLICY = "development-failure-resynthesis-v1"',
        "EXPECTED_ADVERSARIAL_TOP_K = 64",
        "EXPECTED_ADVERSARIAL_PROBES = 2",
        "EXPECTED_ADVERSARIAL_REPLAY_PER_SEQUENCE = 8",
        "EXPECTED_ADVERSARIAL_REPLAY_EXAMPLES = 512",
        "EXPECTED_ADVERSARIAL_MIN_PER_KEYWORD = 24",
        "EXPECTED_SAFE_LEXICAL_POOL = 1330",
        '("ni3", "hao3", "xiao3")',
        '("xiao3", "wo1", "xiao3")',
        "xiaowo-model.kwm",
        "xiaowo-model.pt",
        "xiaowo-model-provenance.json",
        "xiaowo-keywords.kwk",
        "xiaowo-keywords.tsv",
        "development-failure-replay.json",
        "training-run-summary.json",
        "qualification-summary.json",
        "qualification-cohort.json",
        "robustness-summary.json",
        "continuous-far-summary.json",
        "continuous-far-stream-contract.json",
        "shadow-qualification-summary.json",
        "adversarial-refinement-summary.json",
        "adversarial-lexicon.json",
        "formal-preflight.json",
        "training-invocation.json",
        "governed-model-training-invocation-v1",
        "versioned governed training invocation lacks request",
        "manual governed training must not claim a versioned request",
        "model-promotion-manifest.json",
        "MODEL_SHA256SUMS",
        "model provenance SHA does not match promoted model.kwm",
        "model provenance checkpoint SHA does not match promoted model.pt",
        "training summary artifact SHA mismatch",
        "qualification evidence",
        "robustness evidence is not qualified with zero failures",
        "continuous FAR evidence is not strict zero-error/full-coverage",
        "latest-strict-gate-passing-round",
        "overlapping_development_active_wav_sha256",
        "overlapping_exposed_active_wav_sha256",
        "overlapping_retired_active_wav_sha256",
        "shadow_qualification_required",
        "shadow_qualification_qualified",
        "adversarial_refinement_required",
        "model_provenance_adversarial_manifest_verified",
        "adversarial_overlap_guard_included",
        "failure_replay_required",
        "model_provenance_failure_replay_manifest_verified",
        "failure_replay_formal_qualification_used",
        "failure_replay_development_source_wav_bytes_copied",
        "development-only-shadow-qualification",
        "development-only-adversarial-lexicon",
        "post-domain-adversarial-refinement-v1",
        "adversarial-hard-negatives.tsv",
        "promoted model provenance does not prove adversarial replay training",
        "promoted product candidate lacks ordered-token sample weighting",
        "promoted product candidate lacks deterministic feature-cache evidence",
        "two-character 小窝 is not a shipping wake word",
        "ni3 hao3 xiao3 wo1",
        "xiao3 wo1 xiao3 wo1",
        "model provenance repository tree differs from requested training HEAD tree",
        "model provenance per-keyword wake weights differ from refinement evidence",
        "adversarial refinement wake-pressure focus accounting drifted",
    ):
        require(verifier, needle, "promotion bundle verifier")

    push_section = training.split("  pull_request:", 1)[0]
    require(
        push_section,
        "- '.github/triggers/model-training-request.json'",
        "request-only model-training push trigger",
    )
    if "paths:\n  pull_request:" in push_section:
        raise AssertionError("model-training push paths must not be empty")

    # The model-training workflow is intentionally split across two independently
    # bounded hosted jobs. The base job must never expose the formal seed.
    for needle in (
        "base-domain-training:",
        "refinement-and-qualification:",
        "needs: base-domain-training",
        "xiaowo-domain-base-state",
        "Retain exact base-domain state",
        "Download exact base-domain state",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "training/base_stage_receipt.py write",
        "training/base_stage_receipt.py verify",
        "steps.base_stage.outputs.iteration_exit_code",
        "training/verify_final_training_gates.py",
        "eval/run_continuous_far_gate.py",
        "build/model-training/training-invocation.json",
        "training/training_request.py write-receipt",
    ):
        require(training, needle, "staged model-training workflow")
    timeouts = [int(value) for value in re.findall(r"timeout-minutes:\s*(\d+)", training)]
    if timeouts != [360, 360]:
        raise AssertionError(f"staged model-training jobs must remain independently capped at 360 minutes: {timeouts}")
    continuation = training.index("  refinement-and-qualification:")
    base_section = training[:continuation]
    for forbidden in (
        "Refine development candidate with model-mined adversarial lexicon",
        "Enforce development shadow qualification arena",
        "Rotate untouched qualification cohort",
        "training/render_qualification_guarded.py",
        "Finalize latest strict gate-passing candidate and run summary",
    ):
        if forbidden in base_section:
            raise AssertionError(f"base-domain job must not contain formal/refinement stage: {forbidden}")

    expected_order = [
        "Train and iterate domain rounds",
        "Refine development candidate with model-mined adversarial lexicon",
        "Enforce development shadow qualification arena",
        "Rotate untouched qualification cohort",
        "training/render_qualification_guarded.py",
        "Finalize latest strict gate-passing candidate and run summary",
    ]
    positions = []
    for needle in expected_order:
        require(training, needle, "model-training workflow")
        positions.append(training.index(needle))
    if positions != sorted(positions):
        raise AssertionError("model-training optimization/formal qualification ordering drifted")
    for needle in (
        "Verify decoder-aligned training objectives",
        "training/adversarial_refinement.py",
        "training/shadow_qualification.py",
        "training/render_qualification_guarded.py",
        "adversarial-refinement/",
        "adversarial-lexicon/",
        "development-failure-replay/",
        "qualification-repair/",
        "shadow-qualification/",
        "formal-preflight.json",
    ):
        require(training, needle, "model-training workflow")
    if "training/render_qualification_holdout.py \\\n            --config" in training:
        raise AssertionError("model-training workflow must not bypass the guarded formal renderer")

    for needle in (
        'REFINEMENT_SOURCE_POLICY = "development-recall-first-refinement-source-v1"',
        'WAKE_BALANCE_POLICY = "per-keyword-provenance-pressure-balance-v3"',
        'PRESSURE_ASSIGNMENT_POLICY = "explicit-replay-focus-then-token-edit-distance-v2"',
        "build_refinement_focus_rows",
        "derive_refinement_wake_balance",
        '"explicit_focus_nonwake_rows"',
        '"fallback_edit_distance_nonwake_rows"',
        '"--wake-example-weight"',
        '"--wake-keyword-weights"',
        '"wake_keyword_weights"',
        '"wake_balance"',
        "select_refinement_source",
        "refinement_source_policy",
        "source_was_strict",
        "refinement source selection must not use qualification",
    ):
        require(refinement, needle, "adversarial refinement source contract")

    for needle in (
        '"wake_keyword_weights"',
        "checkpoint wake_keyword_weights must be an object",
        '"ordered_token_sample_weighting"',
        '"deterministic-feature-cache-v1"',
        '"training_math_changed"',
    ):
        require(exporter, needle, "model exporter weighting provenance")

    for text, label in (
        (iterate_domain, "product domain iterator"),
        (refinement, "adversarial refinement"),
    ):
        for needle in (
            "feature_cache_max_items",
            "rewrite_training_command",
        ):
            require(text, needle, label)

    for needle in (
        'CACHE_POLICY = "deterministic-feature-cache-v1"',
        '"training_math_changed": False',
        '"feature_cache"',
    ):
        require(feature_cache, needle, "deterministic feature-cache wrapper")

    require(
        training_config,
        '"feature_cache_max_items": 8192',
        "product training feature-cache configuration",
    )

    for needle in (
        'EVIDENCE_CLASS = "governed-model-training-invocation-v1"',
        'SUPPORTED_EVENTS = {"push", "workflow_dispatch"}',
        'TRIGGER_POLICY = "run-on-protected-main-change"',
        'PURPOSE = "governed-product-candidate-training"',
        'SOURCE_POLICY = "exact-current-main"',
        'if ref != "refs/heads/main"',
    ):
        require(training_request, needle, "governed training request authority")

    for needle in (
        'root / "training-invocation.json"',
        'best_dir / "training-invocation.json"',
        'product_lineage.extend([effective_copy, base_contract_copy, invocation_copy])',
    ):
        require(finalizer, needle, "product lineage finalizer")

    for needle in (
        'POLICY = "staged-domain-base-handoff-v1"',
        '"workflow_sha": workflow_sha',
        '"checkout_commit_sha": git_head()',
        '"training_config_sha256": sha256_file(config_path)',
        '"domain_manifest_sha256": sha256_file(manifest_path)',
        '"formal_seed_consumed": False',
        "base stage workflow SHA differs from continuation workflow SHA",
        "base stage checkout commit differs from continuation checkout",
        "base stage domain manifest SHA differs from downloaded state",
        "base stage formal seed differs from continuation config",
        "iteration_exit_code=",
    ):
        require(base_stage, needle, "base-stage handoff receipt")

    for needle in (
        '"你好小窝"',
        '"小窝小窝"',
        '"0.55"',
        '"ni3 hao3 xiao3 wo1"',
        '"xiao3 wo1 xiao3 wo1"',
        "active != reserved",
        "active in retired",
    ):
        require(entry_contract, needle, "staged training entry contract")

    for needle in (
        'PREFLIGHT_POLICY = "shadow-adversarial-failure-formal-preflight-v2"',
        "formal qualification ran without qualified development-only adversarial refinement",
        "formal qualification ran without a qualified shadow arena",
        "model_provenance_adversarial_manifest_verified",
        "adversarial_overlap_guard_included",
        "model_provenance_failure_replay_manifest_verified",
        "failure_replay_formal_qualification_used",
        "overlapping_development_active_wav_sha256",
        "qualification leaked into candidate selection",
        "FAR holdout leaked into model training",
        "synthetic qualification gates were not met",
    ):
        require(final_gates, needle, "extracted final training gates")

    for needle in (
        '"coverage-capacity-v1"',
        '"semantic-negative-boundary-v1"',
        '"false_accepts"',
        '"far_per_hour"',
        '"full_negative_manifest_coverage"',
        '"min_observed_hard_negative_injections_per_clip"',
        '"hard_negative_rate_per_minute"',
        '"coverage_hard_negative_gain"',
        "actual payload rate fell below 8/min",
        "payload gap is inside decoder retention window",
    ):
        require(far_gate, needle, "extracted continuous FAR gate")

    for needle in (
        'POLICY = "development-qualification-repair-v1"',
        "MAX_UNIQUE_FAILURES = 8",
        "EXAMPLES_PER_FAILURE = 8",
        "REPAIR_EPOCHS = 6",
        "REPAIR_LR_SCALE = 0.25",
        '"qualification_repair_source_splits": ["qualification"]',
        '"source_splits": base_source_splits',
        '"formal_qualification_used": False',
        '"development_source_wav_bytes_copied": False',
        "qualification repair accidentally copied a development evaluation WAV",
    ):
        require(repair, needle, "qualification repair replay")

    for needle in (
        "REPAIR_VALIDATION_SEED_NAMESPACE = 171_000_003",
        "development-qualification-mining",
        "development-qualification-repair-validation-config.json",
        '"mining_cohort_used_for_training": True',
        '"validation_cohort_used_for_training": False',
        '"development_qualification_validation_used_for_training": False',
        '"formal_qualification_used": False',
        "development qualification repair did not reach strict triple-pass",
    ):
        require(refinement, needle, "qualification repair refinement")

    for needle in (
        '"blocked": True',
        '"blocked_reason": reason',
        'blocked_result("training-summary-missing")',
    ):
        require(robustness, needle, "robustness blocked diagnostic")

    for needle in (
        '"compact-model-training-diagnostics"',
        '"formal_seed"',
        '"consumed_by_this_run"',
        '"shadow_qualification"',
        '"development_failure_replay"',
        '"continuous_far"',
        '"diagnostic_errors"',
    ):
        require(diagnostics, needle, "compact training diagnostics")

    if workflow.count('--repo "$GITHUB_REPOSITORY"') < 2:
        raise AssertionError("all release CLI calls must bind the repository explicitly")
    scan = workflow.lower()
    for allowed in ("latest-strict-gate-passing-round", "strict/latest"):
        scan = scan.replace(allowed, "")
    if "latest" in scan:
        raise AssertionError("model promotion must never select an ambiguous training artifact")
    if "cancel-in-progress: true" in workflow.lower():
        raise AssertionError("model promotion must not cancel another promotion for the same provenance")
    if 'tags:\n      - "v*"' in workflow:
        raise AssertionError("model promotion must not masquerade as the SDK v* release workflow")

    print("model promotion + staged Data V3 qualification-repair provenance contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
