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
        "MODEL_SHA256SUMS",
        "model-promotion-manifest.json",
        "subject-checksums: dist/MODEL_SHA256SUMS",
        "--target \"$EXPECTED_HEAD_SHA\"",
    ):
        require(text, needle)

    if "latest" in lowered:
        raise AssertionError("model promotion must never select an ambiguous latest training artifact")
    if "cancel-in-progress: true" in lowered:
        raise AssertionError("model promotion must not cancel another promotion for the same provenance")
    if 'tags:\n      - "v*"' in text:
        raise AssertionError("model promotion must not masquerade as the SDK v* release workflow")

    print("model promotion provenance contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
