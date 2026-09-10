#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-promotion.yml"
VERIFIER = ROOT / "tools" / "verify_model_promotion_bundle.py"
TRAINING_WORKFLOW = ROOT / ".github" / "workflows" / "model-training.yml"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label} is missing required contract: {needle!r}")


def main() -> int:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    training = TRAINING_WORKFLOW.read_text(encoding="utf-8")

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
        "xiaowo-model.kwm",
        "xiaowo-model.pt",
        "xiaowo-model-provenance.json",
        "xiaowo-keywords.kwk",
        "xiaowo-keywords.tsv",
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
        "qualification keyword {keyword_id}",
        "robustness evidence is not qualified with zero failures",
        "continuous FAR evidence is not strict zero-error/full-coverage",
        "qualification cohort lacks strict development prerequisite evidence",
        "qualification cohort development manifest SHA",
        "latest-strict-gate-passing-round",
        "qualification cohort development round differs from finalized model",
        "qualification cohort development frontend differs from finalized model",
        "overlapping_development_active_wav_sha256",
        "overlapping_exposed_active_wav_sha256",
        "overlapping_retired_active_wav_sha256",
        "shadow-adversarial-formal-preflight-v1",
        "shadow_qualification_required",
        "shadow_qualification_qualified",
        "adversarial_refinement_required",
        "model_provenance_adversarial_manifest_verified",
        "adversarial_overlap_guard_included",
        "adversarial_formal_qualification_used",
        "development-only-shadow-qualification",
        "development-only-adversarial-lexicon",
        "post-domain-adversarial-refinement-v1",
        "adversarial-hard-negatives.tsv",
        "exhaustive safe lexical pool size drifted from 1330",
        "adversarial Top-K/replay product policy drifted",
        "int(adversarial.get(\"top_k\", 0)) != 24",
        "int(adversarial.get(\"replay_examples\", 0)) != 96",
        "int(adversarial.get(\"max_length\", -1)) != 5",
        "int(adversarial.get(\"probes_per_sequence\", -1)) != 1",
        "int(adversarial.get(\"replay_examples_per_sequence\", -1)) != 4",
        "shadow qualification arena is smaller than eight seeds",
        "shadow seed surrogate separation fell below the promoted floor",
        "promoted model provenance does not prove adversarial replay training",
        "two-character 小窝 is not a shipping wake word",
        "ni3 hao3 xiao3 wo1",
        "xiao3 wo1 xiao3 wo1",
        "model provenance repository tree differs from requested training HEAD tree",
        '"schema_version": 3',
        '"adversarial_refinement_qualified": True',
        '"shadow_qualification_qualified": True',
    ):
        require(verifier, needle, "promotion bundle verifier")

    # Formal-seed ownership must be a one-way chain: base training may finish with
    # development status 0/1, but refinement and all shadow gates must succeed
    # before the guarded renderer can ever expose the reserved formal seed.
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

    print("model promotion + optimized training provenance contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
