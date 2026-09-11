#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-promotion.yml"
VERIFIER = ROOT / "tools" / "verify_model_promotion_bundle.py"
TRAINING_WORKFLOW = ROOT / ".github" / "workflows" / "model-training.yml"
DIAGNOSTICS = ROOT / "tools" / "build_training_diagnostics.py"
REFINEMENT = ROOT / "training" / "adversarial_refinement.py"
REPAIR = ROOT / "training" / "qualification_failure_replay.py"
ROBUSTNESS = ROOT / "eval" / "gate_robustness.py"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing required contract: {needle!r}")


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    training = TRAINING_WORKFLOW.read_text(encoding="utf-8")
    diagnostics = DIAGNOSTICS.read_text(encoding="utf-8")
    refinement = REFINEMENT.read_text(encoding="utf-8")
    repair = REPAIR.read_text(encoding="utf-8")
    robustness = ROBUSTNESS.read_text(encoding="utf-8")

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
        "two-character 小窝 is not a shipping wake word",
        "ni3 hao3 xiao3 wo1",
        "xiao3 wo1 xiao3 wo1",
        "model provenance repository tree differs from requested training HEAD tree",
    ):
        require(verifier, needle, "promotion bundle verifier")

    expected_order = [
        "Train and iterate domain rounds",
        "Refine strict candidate with model-mined adversarial lexicon",
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
        "shadow-qualification/",
        "formal-preflight.json",
        "adversarial_overlap_guard_included",
        "model_provenance_adversarial_manifest_verified",
    ):
        require(training, needle, "model-training workflow")
    if "training/render_qualification_holdout.py \\\n            --config" in training:
        raise AssertionError("model-training workflow must not bypass the guarded formal renderer")

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

    print("model promotion + Data V3 qualification-repair provenance contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
