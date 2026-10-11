#!/usr/bin/env python3
"""Keep candidate-only numeric checks outside existing training/data jobs.

Existing product workflow path filters remain authoritative. Only the explicitly
added candidate paths are removed when deciding whether the original jobs are
required; any old path in a mixed PR/push preserves all original jobs. Selection
errors fail, never turn into a green skip. No models, artifacts or APIs are read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from check_workflow_path_filters import exact_path_filters
from research_ci_changes import changed_paths

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/product-training-data-contract.yml"
CANDIDATE_EXACT = frozenset({
    "tests/test_candidate_encoder_numeric.py",
    "tests/test_candidate_encoder_numeric_ci.py",
    "tools/candidate_numeric_ci_changes.py",
})
CANDIDATE_GLOB = "research/candidate_encoders/**"
# Shared source/control edits require both the old gates and the new contracts.
CANDIDATE_SHARED = frozenset({WORKFLOW, "training/model.py", "training/sitecustomize.py",
                              "tools/research_ci_changes.py", "tools/check_workflow_path_filters.py"})
SUPPORTED_PUSH_REFS = frozenset({"refs/heads/main", "refs/heads/fix/product-training-speech-like-v1"})


def scopes(paths, workflow=ROOT / WORKFLOW):
    filters = {path for _, path in exact_path_filters(workflow)}
    added = CANDIDATE_EXACT | {CANDIDATE_GLOB}
    if not added <= filters or not CANDIDATE_SHARED <= filters:
        raise ValueError("candidate workflow filters are incomplete")
    existing = filters - added
    if not existing or any(any(char in path for char in "*?[]{}$") for path in existing):
        raise ValueError("existing product filters must be nonempty exact paths")
    changed = {path for path in paths if path}
    # Changing selector logic is itself a gate change and must exercise the old
    # gates; ordinary candidate implementation/test/doc changes are bounded.
    existing_required = bool(changed & (existing | {"tools/candidate_numeric_ci_changes.py"}))
    candidate_required = bool(changed & (CANDIDATE_EXACT | CANDIDATE_SHARED)) or any(
        path.startswith("research/candidate_encoders/") for path in changed)
    return existing_required, candidate_required


def event_paths(event, payload, ref):
    if event == "push":
        if ref not in SUPPORTED_PUSH_REFS:
            raise ValueError("unexpected push ref for product contract workflow")
        # Reuse complete before..HEAD push-delta handling for both existing
        # allowed branches; this does not change any trigger or compare base.
        return changed_paths(event, payload, "refs/heads/main")
    if event == "pull_request":
        return changed_paths(event, payload, ref)
    raise ValueError("unexpected event for product contract workflow")


def main():
    event = os.environ["GITHUB_EVENT_NAME"]
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    existing, candidate = scopes(event_paths(event, payload, os.environ["GITHUB_REF"]))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"existing_required={str(existing).lower()}\n")
        output.write(f"candidate_required={str(candidate).lower()}\n")
    print(f"existing product gates required: {existing}; candidate numerics required: {candidate}")


if __name__ == "__main__":
    main()
