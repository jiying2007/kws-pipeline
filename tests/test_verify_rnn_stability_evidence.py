from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_rnn_stability_evidence as stability  # noqa: E402

# The GRU line's stability verifier is exercised by
# tests/test_gru_stability_evidence_contract.py; the RNN one was exercised by
# nothing but the three workflows that call it, so only its happy path ever ran.
# This verifier imports nothing outside the standard library, so every case
# below is a real call rather than a stubbed one.


def strict(round_index: int) -> dict:
    return {"round": round_index, "calibration_gate": True, "test_gate": True}


def count(gates) -> int:
    """Independent streak counter: a bug in the verifier's own cannot hide."""
    streak = 0
    for row in gates:
        streak = streak + 1 if (row.get("calibration_gate") and row.get("test_gate")) else 0
    return streak


def build(root: pathlib.Path, required: int = 2, gates=None) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    gates = [strict(0), strict(1)] if gates is None else gates
    observed = count(gates)
    (root / "source-development-policy.json").write_bytes(
        json.dumps(
            {
                "policy": stability.SOURCE_POLICY,
                "model_family": "rnn",
                "stable_strict_pass_rounds": required,
                "stable_strict_pass_rounds_required": required,
                "stable_strict_pass_rounds_observed": observed,
            }
        ).encode()
    )
    for name in ("freeze-manifest.json", "selection-evidence.json"):
        (root / name).write_bytes(
            json.dumps(
                {
                    "stable_strict_pass_rounds_required": required,
                    "stable_strict_pass_rounds_observed": observed,
                }
            ).encode()
        )
    set_in(
        root,
        "freeze-manifest.json",
        policy=stability.FREEZE_POLICY,
        model_family="rnn",
    )
    write_stability(
        root,
        {
            "policy": stability.STABILITY_POLICY,
            "source_policy": stability.SOURCE_POLICY,
            "model_family": "rnn",
            "development_only": True,
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
            "round_gates": gates,
            "stable_strict_pass_rounds_required": required,
            "stable_strict_pass_rounds_observed": observed,
        }
    )
    return root


def set_in(root: pathlib.Path, name: str, **changes) -> bytes:
    path = root / name
    value = json.loads(path.read_bytes().decode())
    value.update(changes)
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return data


def write_stability(root: pathlib.Path, value: dict) -> None:
    data = json.dumps(value).encode()
    (root / "stability-evidence.json").write_bytes(data)
    set_in(
        root,
        "freeze-manifest.json",
        stability_evidence_sha256=hashlib.sha256(data).hexdigest(),
    )


def set_stability(root: pathlib.Path, **changes) -> None:
    path = root / "stability-evidence.json"
    value = json.loads(path.read_bytes().decode())
    value.update(changes)
    write_stability(root, value)


def expect(root: pathlib.Path, needle: str) -> None:
    try:
        stability.verify(root)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def expect_streak(needle: str, gates) -> None:
    try:
        stability.terminal_strict_streak(gates)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)

        good = build(base / "good")
        value = stability.verify(good)
        assert value["verified"] is True, value
        assert value["model_family"] == "rnn", value
        assert value["stable_strict_pass_rounds_required"] == 2, value
        assert value["stable_strict_pass_rounds_observed"] == 2, value

        # Identity chain: freeze manifest, source policy, stability evidence.
        root = build(base / "freeze-policy")
        set_in(root, "freeze-manifest.json", policy="other")
        expect(root, "freeze identity mismatch")
        root = build(base / "freeze-family")
        set_in(root, "freeze-manifest.json", model_family="gru")
        expect(root, "freeze identity mismatch")
        root = build(base / "source-policy")
        set_in(root, "source-development-policy.json", policy="other")
        expect(root, "source policy identity mismatch")
        root = build(base / "source-family")
        set_in(root, "source-development-policy.json", model_family="gru")
        expect(root, "source policy identity mismatch")
        root = build(base / "stab-policy")
        set_stability(root, policy="other")
        expect(root, "stability evidence identity mismatch")
        root = build(base / "stab-source")
        set_stability(root, source_policy="other")
        expect(root, "stability evidence identity mismatch")
        root = build(base / "stab-family")
        set_stability(root, model_family="gru")
        expect(root, "scope mismatch")
        root = build(base / "stab-scope")
        set_stability(root, development_only=False)
        expect(root, "scope mismatch")
        for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
            root = build(base / f"stab-{field}")
            set_stability(root, **{field: True})
            expect(root, f"requires {field}=false")

        # A wrong digest and a malformed one are the same failure, not a pass.
        root = build(base / "digest-wrong")
        set_in(root, "freeze-manifest.json", stability_evidence_sha256="0" * 64)
        expect(root, "digest mismatch")
        root = build(base / "digest-short")
        set_in(root, "freeze-manifest.json", stability_evidence_sha256="abc")
        expect(root, "digest mismatch")

        # The required streak has to be a real positive integer: a truthy
        # non-integer would otherwise read as "some rounds required".
        for bad in (None, 0, True, "2"):
            root = build(base / f"required-{bad}")
            set_in(root, "source-development-policy.json", stable_strict_pass_rounds=bad)
            expect(root, "must be a positive integer")

        root = build(base / "short-streak", required=3)
        expect(root, "streak insufficient")

        # round_gates shape: non-empty, contiguous from zero, real booleans.
        root = build(base / "gates-empty")
        set_stability(root, round_gates=[])
        expect(root, "non-empty round_gates")
        root = build(base / "gates-skip")
        set_stability(root, round_gates=[strict(1)])
        expect(root, "contiguous from zero")
        root = build(base / "gates-nonbool")
        set_stability(root, round_gates=[{"round": 0, "calibration_gate": 1, "test_gate": True}])
        expect(root, "must be booleans")

        # All three containers have to agree on both numbers.
        for label, name in (
            ("stability", "stability-evidence.json"),
            ("freeze", "freeze-manifest.json"),
            ("selection", "selection-evidence.json"),
        ):
            root = build(base / f"req-{label}")
            if label == "stability":
                set_stability(root, stable_strict_pass_rounds_required=5)
            else:
                set_in(root, name, stable_strict_pass_rounds_required=5)
            expect(root, f"{label} RNN required streak mismatch")

            root = build(base / f"obs-{label}")
            if label == "stability":
                set_stability(root, stable_strict_pass_rounds_observed=5)
            else:
                set_in(root, name, stable_strict_pass_rounds_observed=5)
            expect(root, f"{label} RNN observed streak mismatch")

        root = build(base / "obs-bool")
        set_in(root, "selection-evidence.json", stable_strict_pass_rounds_observed=True)
        expect(root, "selection RNN observed streak mismatch")

    # The counter itself: a break resets it, and it rejects bad shapes.
    assert stability.terminal_strict_streak([strict(0), strict(1)]) == 2
    assert (
        stability.terminal_strict_streak(
            [strict(0), {"round": 1, "calibration_gate": False, "test_gate": True}, strict(2)]
        )
        == 1
    )
    expect_streak("non-empty round_gates", [])
    expect_streak("non-empty round_gates", None)
    expect_streak("contiguous from zero", [strict(1)])
    expect_streak("must be booleans", [{"round": 0, "calibration_gate": 1, "test_gate": True}])

    print("test_verify_rnn_stability_evidence: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
