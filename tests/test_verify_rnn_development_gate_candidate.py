from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_rnn_development_gate as verify  # noqa: E402

# verify_candidate is what a frozen RNN candidate has to pass before it is
# allowed into qualification, and neither twin had a single test for it.
# The GRU twin's version of this function checks fewer things (no member
# hashes, no stability rounds, no candidate stage) but does check one thing
# the RNN version did not: that the selection evidence records its
# calibration and test gate flags as strict-pass. The gap that mattered is
# the second one, so that is what this file pins first.
#
# evaluate_development_split is stubbed. That is only honest because the
# split arithmetic itself is covered by tests/test_rnn_development_gate.py;
# what is exercised here is everything around it.

MEMBERS = {
    "model.kwm": b"model",
    "model.pt": b"checkpoint",
    "keywords.kwk": b"pack",
    "keywords.tsv": b"keywords",
    "model-provenance.json": b"{}",
}
FREEZE_FIELDS = (
    ("model.kwm", "model_sha256"),
    ("model.pt", "checkpoint_sha256"),
    ("keywords.kwk", "pack_sha256"),
    ("keywords.tsv", "keywords_sha256"),
    ("model-provenance.json", "provenance_sha256"),
)

STATE = {"qualified": True}


def install_stub() -> None:
    def split(calibration, domains, config):
        return {"qualified": STATE["qualified"], "policy": verify.DEVELOPMENT_GATE_POLICY}

    verify.evaluate_development_split = split


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_object(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def policy_value(**overrides) -> dict:
    payload = {
        "stable_strict_pass_rounds": 3,
        "candidate_freeze": {
            "fresh_validation_seed_namespace": 195000019,
            "shadow_arena": "rnn-shadow-arena-v1",
            "formal_qualification_seed": 77000019,
        },
    }
    payload.update(overrides)
    return payload


def selection_value(**overrides) -> dict:
    payload = {
        "calibration": {},
        "calibration_domains": {},
        "test": {},
        "test_domains": {},
        "selected_round": 7,
        "model_sha256": digest(MEMBERS["model.kwm"]),
        "calibration_gate": True,
        "test_gate": True,
    }
    payload.update(overrides)
    return payload


def stage_value(**overrides) -> dict:
    payload = {
        "fresh_validation_seed_namespace": 195000019,
        "shadow_arena": "rnn-shadow-arena-v1",
        "formal_qualification_seed": 77000019,
    }
    payload.update(overrides)
    return payload


def build(
    base: pathlib.Path,
    *,
    members=None,
    selection=None,
    freeze=None,
    policy=None,
    stage=stage_value(),
) -> pathlib.Path:
    """Write a frozen candidate that should pass, then apply one change."""
    contents = dict(MEMBERS)
    if members:
        contents.update(members)
    for name, payload in contents.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    selection_path = write_object(base / "selection-evidence.json", selection or selection_value())
    write_object(base / "source-config.json", {})
    write_object(base / "source-development-policy.json", policy or policy_value())

    value = {
        "schema_version": 1,
        "model_family": "rnn",
        "architecture": verify.loop.ARCHITECTURE,
        "development_gate_policy": verify.DEVELOPMENT_GATE_POLICY,
        "selection_evidence_sha256": digest(selection_path.read_bytes()),
        "stable_strict_pass_rounds_required": 3,
        "stable_strict_pass_rounds_observed": 3,
        "candidate_stage": stage,
    }
    for name, field in FREEZE_FIELDS:
        value[field] = digest(contents[name])
    if freeze:
        value.update(freeze)
    write_object(base / "freeze-manifest.json", value)
    return base


def expect_raises(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def check_happy(base: pathlib.Path) -> None:
    candidate = build(base / "ok")
    result = verify.verify_candidate(candidate)
    assert result["verified"] is True
    assert result["model_family"] == "rnn"
    assert result["selected_round"] == 7
    assert result["model_sha256"] == digest(MEMBERS["model.kwm"])
    assert result["mode"] == "candidate"


def check_gate_flags(base: pathlib.Path) -> None:
    # The gap: the RNN twin verified the recomputed robustness but never
    # looked at the gate flags the evidence records for itself. The GRU twin
    # required them to be the boolean True.
    for label, value in (("false", False), ("truthy int", 1), ("string", "true"), ("absent", None)):
        overrides = {"calibration_gate": value} if label != "absent" else {}
        selection = selection_value(**overrides)
        if label == "absent":
            selection.pop("calibration_gate")
        candidate = build(base / f"flag-{label.replace(' ', '-')}", selection=selection)
        expect_raises("gate flags are not strict-pass", lambda c=candidate: verify.verify_candidate(c))

    overrides = {"test_gate": False}
    candidate = build(base / "flag-test", selection=selection_value(**overrides))
    expect_raises("gate flags are not strict-pass", lambda: verify.verify_candidate(candidate))


def check_identity(base: pathlib.Path) -> None:
    candidate = build(base / "family", freeze={"model_family": "gru"})
    expect_raises("candidate identity mismatch", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "arch", freeze={"architecture": "other"})
    expect_raises("candidate identity mismatch", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "policy", freeze={"development_gate_policy": "other"})
    expect_raises("development gate policy mismatch", lambda: verify.verify_candidate(candidate))


def check_members(base: pathlib.Path) -> None:
    # Tamper after the build: the builder derives the manifest digests from
    # what it wrote, so a differently-sized member passed in up front would
    # just produce a consistent (and wrong) manifest.
    candidate = build(base / "member")
    (candidate / "model.kwm").write_bytes(b"tampered")
    expect_raises("member hash mismatch: model.kwm", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "missing")
    (candidate / "keywords.tsv").unlink()
    expect_raises("member hash mismatch: keywords.tsv", lambda: verify.verify_candidate(candidate))

    # The evidence file is pinned by digest, so swapping in a different one
    # that still parses must fail.
    candidate = build(base / "evidence")
    write_object(candidate / "selection-evidence.json", selection_value(selected_round=99))
    expect_raises("selection evidence hash mismatch", lambda: verify.verify_candidate(candidate))


def check_stability(base: pathlib.Path) -> None:
    candidate = build(base / "required", freeze={"stable_strict_pass_rounds_required": 2})
    expect_raises("required stability evidence mismatch", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "observed", freeze={"stable_strict_pass_rounds_observed": 2})
    expect_raises("lacks stable terminal strict evidence", lambda: verify.verify_candidate(candidate))


def check_stage(base: pathlib.Path) -> None:
    candidate = build(base / "nostage", freeze={"candidate_stage": None})
    expect_raises("candidate stage is missing", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "namespace", stage=stage_value(fresh_validation_seed_namespace=1))
    expect_raises("fresh namespace mismatch", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "arena", stage=stage_value(shadow_arena="other"))
    expect_raises("shadow arena mismatch", lambda: verify.verify_candidate(candidate))

    candidate = build(base / "seed", stage=stage_value(formal_qualification_seed=1))
    expect_raises("formal seed mismatch", lambda: verify.verify_candidate(candidate))


def check_robustness(base: pathlib.Path) -> None:
    candidate = build(base / "unqualified")
    STATE["qualified"] = False
    try:
        expect_raises("does not pass robustness gate", lambda: verify.verify_candidate(candidate))
    finally:
        STATE["qualified"] = True


def main() -> int:
    install_stub()
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        check_happy(base)
        check_gate_flags(base)
        check_identity(base)
        check_members(base)
        check_stability(base)
        check_stage(base)
        check_robustness(base)
    print("test_verify_rnn_development_gate_candidate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
