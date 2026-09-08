#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "model-promotion.yml"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"model promotion workflow is missing required contract: {needle!r}")


def main() -> int:
    text = WORKFLOW.read_text(encoding="utf-8")
    lowered = text.lower()

    for needle in (
        "workflow_dispatch:",
        "training_run_id:",
        "expected_head_sha:",
        "contents: write",
        "actions: read",
        "cancel-in-progress: false",
        "actions/runs/${TRAINING_RUN_ID}",
        "run.get('name') != 'model-training'",
        "run.get('status') != 'completed'",
        "run.get('conclusion') != 'success'",
        "run.get('head_sha') != expected",
        "xiaowo-torch-domain-model",
        "expected exactly one live xiaowo-torch-domain-model artifact",
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
        "model provenance SHA does not match promoted model.kwm",
        "model provenance checkpoint SHA does not match promoted model.pt",
        "training summary artifact SHA mismatch",
        "qualification evidence is not strict 256/256 zero-error",
        "qualification keyword {keyword_id} is not 128/128",
        "robustness evidence is not qualified with zero failures",
        "continuous FAR evidence is not zero-error",
        "full_negative_manifest_coverage",
        "two-character 小窝 is not a shipping wake word",
        "provenance_repository_sha",
        "provenance_tree_sha",
        "expected_tree_sha",
        "git_tree(os.environ['EXPECTED_HEAD_SHA'])",
        "model provenance repository tree differs from the requested training HEAD tree",
        "MODEL_SHA256SUMS",
        "model-promotion-manifest.json",
        "'schema_version': 2",
        "subject-checksums: dist/MODEL_SHA256SUMS",
        "--target \"$EXPECTED_HEAD_SHA\"",
    ):
        require(text, needle)

    if "latest" in lowered:
        raise AssertionError("model promotion must never select an ambiguous training artifact")
    if "cancel-in-progress: true" in lowered:
        raise AssertionError("model promotion must not cancel another promotion for the same provenance")
    if 'tags:\n      - "v*"' in text:
        raise AssertionError("model promotion must not masquerade as the SDK v* release workflow")

    print("model promotion provenance contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
