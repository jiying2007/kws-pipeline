from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from development_experiment_request import (  # noqa: E402
    EVIDENCE_CLASS,
    PURPOSE,
    SOURCE_POLICY,
    TRIGGER_POLICY,
    self_test,
    verify_request,
)


def main() -> int:
    self_test()

    marker = ROOT / ".github/triggers/product-development-experiment.json"
    request = verify_request(marker)
    assert request["enabled"] is False
    assert request["trigger_policy"] == TRIGGER_POLICY
    assert request["purpose"] == PURPOSE
    assert request["source_policy"] == SOURCE_POLICY
    assert SOURCE_POLICY == "exact-pr-head-development-only"
    assert EVIDENCE_CLASS == "product-development-pr-experiment-invocation-v1"

    workflow = (
        ROOT / ".github/workflows/product-development-pr-experiment.yml"
    ).read_text(encoding="utf-8")
    assert "product-development-base-experiment:" in workflow
    assert "product-development-refinement-experiment:" in workflow
    assert workflow.count("timeout-minutes: 120") == 2
    assert "github.event.pull_request.head.sha" in workflow
    assert 'test "$(git rev-parse HEAD)" = "$HEAD_SHA"' in workflow
    assert "development_experiment_request.py verify" in workflow
    assert "development_experiment_request.py write-receipt" in workflow
    assert "promotion=false" in workflow
    assert "formal_qualification_authorized" not in workflow
    assert "training/materialize_governed_product_base.sh" in workflow
    assert 'train["epochs"]=12' in workflow
    assert 'train["warm_start_epochs"]=6' in workflow
    assert 'adversarial["refinement_epochs"]=6' in workflow
    assert 'iteration["max_rounds"]=2' in workflow
    assert "--defer-qualification" in workflow
    assert "--stop-after-development-eval" in workflow
    assert "tools/diagnose_acoustic_alignment.py" in workflow
    assert "tools/build_training_diagnostics.py" in workflow
    assert "training/evaluate_refinement_eligibility.py" in workflow
    assert "product_preflight_handoff.py pack" in workflow
    assert "product_preflight_handoff.py restore" in workflow
    assert "product_preflight_handoff.py verify-materialization" in workflow
    assert ".github/triggers/model-training-request.json" in workflow
    assert "configs/training/product-speech-like-base-v1.json" in workflow
    assert "models/registry" in workflow
    assert "PR-head development experiment modified protected path" in workflow

    governed = (
        ROOT / ".github/workflows/model-training-preflight.yml"
    ).read_text(encoding="utf-8")
    assert "governed preflight request PR must be request-only" in governed
    assert ".github/triggers/product-development-experiment.json" not in governed

    raw = json.loads(marker.read_text(encoding="utf-8"))
    raw["enabled"] = True
    raw["source_policy"] = "exact-current-main"
    invalid = ROOT / ".tmp-invalid-development-experiment.json"
    try:
        invalid.write_text(json.dumps(raw), encoding="utf-8")
        try:
            verify_request(invalid)
        except ValueError as exc:
            assert "source_policy mismatch" in str(exc)
        else:
            raise AssertionError("governed source policy crossed into PR experiment lane")
    finally:
        invalid.unlink(missing_ok=True)

    print("product development PR experiment contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
