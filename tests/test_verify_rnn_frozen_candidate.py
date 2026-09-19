from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_rnn_frozen_candidate as frozen  # noqa: E402

# The GRU line has five contract tests; the RNN line has none. Every RNN
# verifier is therefore exercised only by its workflow, which means only the
# happy path is ever executed -- a verifier that quietly starts accepting bad
# input stays green. This test pins the failure modes instead.
#
# The development-gate arithmetic is stubbed: it belongs to rnn_development_gate,
# not to the freeze contract, and stubbing it is what lets each case below be a
# single-field change instead of a full robust corpus.

MEMBERS = {
    "model.kwm": b"kwm",
    "model.pt": b"pt",
    "keywords.kwk": b"kwk",
    "keywords.tsv": b"tsv",
    "model-provenance.json": b"{}",
}

WAV_HASHES = ["a" * 64, "b" * 64]


class Gate:
    qualified = True


def evaluate_development_split(base: dict, domains: dict, config: dict) -> dict:
    return {"qualified": Gate.qualified}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(root: pathlib.Path, name: str, data: bytes) -> str:
    (root / name).write_bytes(data)
    return sha256(data)


def build(root: pathlib.Path) -> dict:
    """Write a minimal but complete frozen RNN candidate; return the manifest."""
    root.mkdir(parents=True, exist_ok=True)
    digest = {name: write(root, name, data) for name, data in MEMBERS.items()}

    config = json.dumps({"domain_gates": {}}).encode()
    policy = json.dumps(
        {
            "policy": frozen.SOURCE_POLICY,
            "model_family": "rnn",
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
        }
    ).encode()
    evidence = json.dumps(
        {
            "evidence_class": "rnn-frozen-development-selection",
            "model_family": "rnn",
            "architecture": frozen.ARCHITECTURE,
            "selection_policy": frozen.SELECTION_POLICY,
            "selected_round": 3,
            "selected_score": 0.5,
            "model_sha256": digest["model.kwm"],
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
            "calibration": {},
            "calibration_domains": {},
            "test": {},
            "test_domains": {},
            "calibration_gate": True,
            "test_gate": True,
        }
    ).encode()
    digest["source-config.json"] = write(root, "source-config.json", config)
    digest["source-development-policy.json"] = write(
        root, "source-development-policy.json", policy
    )
    digest["selection-evidence.json"] = write(
        root, "selection-evidence.json", evidence
    )
    digest["development-wav-sha256.json"] = write(
        root,
        "development-wav-sha256.json",
        json.dumps(
            {
                "evidence_class": "rnn-frozen-development-wav-identities",
                "model_family": "rnn",
                "wav_sha256": WAV_HASHES,
                "wav_sha256_count": len(WAV_HASHES),
            }
        ).encode(),
    )

    manifest = {
        "policy": frozen.FREEZE_POLICY,
        "source_policy": frozen.SOURCE_POLICY,
        "model_family": "rnn",
        "architecture": frozen.ARCHITECTURE,
        "selection_policy": frozen.SELECTION_POLICY,
        "evidence_scope": "development-only",
        "selection_evidence": ["development-calibration", "development-test"],
        "qualification_used_for_selection": False,
        "shadow_used_for_selection": False,
        "formal_qualification_used_for_selection": False,
        "candidate_stage": {
            "fresh_validation_required": True,
            "shadow_required": True,
            "formal_qualification_required": True,
            "bounded_repair_only": True,
            "validation_feedback_allowed": False,
            "threshold_feedback_allowed": False,
            "training_rule_feedback_allowed": False,
        },
        "selected_round": 3,
        "selected_score": 0.5,
        "selected_model_matches_selection_evidence": True,
    }
    manifest.update(
        {
            "model_sha256": digest["model.kwm"],
            "checkpoint_sha256": digest["model.pt"],
            "pack_sha256": digest["keywords.kwk"],
            "keywords_sha256": digest["keywords.tsv"],
            "provenance_sha256": digest["model-provenance.json"],
            "selection_evidence_sha256": digest["selection-evidence.json"],
            "development_wav_identities_sha256": digest["development-wav-sha256.json"],
            "source_config_snapshot_sha256": digest["source-config.json"],
            "source_development_policy_snapshot_sha256": digest[
                "source-development-policy.json"
            ],
            "config_sha256": digest["source-config.json"],
            "development_policy_sha256": digest["source-development-policy.json"],
        }
    )
    write(root, "freeze-manifest.json", json.dumps(manifest).encode())
    return root


def manifest_of(root: pathlib.Path) -> dict:
    return json.loads((root / "freeze-manifest.json").read_bytes().decode())


def patch(root: pathlib.Path, **changes: object) -> None:
    path = root / "freeze-manifest.json"
    manifest = json.loads(path.read_bytes().decode())
    manifest.update(changes)
    path.write_bytes(json.dumps(manifest).encode())


def replace(root: pathlib.Path, name: str, data: bytes, digest_field: str) -> None:
    """Rewrite a member and keep the manifest digest consistent with it."""
    (root / name).write_bytes(data)
    patch(root, **{digest_field: sha256(data)})


def expect(root: pathlib.Path, needle: str) -> None:
    try:
        frozen.verify(root)
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def main() -> int:
    frozen.evaluate_development_split = evaluate_development_split

    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        good = build(base / "good")
        assert isinstance(frozen.verify(good), dict), "a complete candidate must verify"

        # Each case is one field moved off the contract. Asserting the specific
        # message is the anti-vacuity guard: a verifier that rejected everything
        # with one error would fail every case but the first.
        cases = [
            ("policy", {"policy": "other"}, "policy identity mismatch"),
            ("family", {"model_family": "gru"}, "frozen model identity mismatch"),
            ("architecture", {"architecture": "other"}, "frozen model identity mismatch"),
            ("selection", {"selection_policy": "other"}, "freeze selection policy mismatch"),
            ("scope", {"evidence_scope": "production"}, "source scope must be development-only"),
            (
                "evidence",
                {"selection_evidence": ["development-test"]},
                "non-development evidence",
            ),
            (
                "qual-selection",
                {"qualification_used_for_selection": True},
                "qualification_used_for_selection=false",
            ),
            ("stage-missing", {"candidate_stage": None}, "candidate_stage contract missing"),
            ("config-snapshot", {"config_sha256": "0" * 64}, "config snapshot mismatch"),
            (
                "policy-snapshot",
                {"development_policy_sha256": "0" * 64},
                "policy snapshot mismatch",
            ),
            (
                "binding",
                {"selected_model_matches_selection_evidence": False},
                "model/selection binding not asserted",
            ),
        ]
        for name, change, needle in cases:
            root = build(base / f"case-{name}")
            patch(root, **change)
            expect(root, needle)

        stage = dict(manifest_of(good)["candidate_stage"])
        for field in (
            "fresh_validation_required",
            "shadow_required",
            "formal_qualification_required",
            "bounded_repair_only",
        ):
            root = build(base / f"stage-false-{field}")
            broken = dict(stage)
            broken[field] = False
            patch(root, candidate_stage=broken)
            expect(root, f"candidate stage requires {field}=true")
        for field in (
            "validation_feedback_allowed",
            "threshold_feedback_allowed",
            "training_rule_feedback_allowed",
        ):
            root = build(base / f"stage-true-{field}")
            broken = dict(stage)
            broken[field] = True
            patch(root, candidate_stage=broken)
            expect(root, f"candidate stage requires {field}=false")

        # Member set: absent, tampered, and present-but-empty have to be three
        # different failures, not one.
        root = build(base / "member-absent")
        (root / "model.kwm").unlink()
        expect(root, "member missing: model.kwm")

        root = build(base / "member-empty")
        (root / "model.pt").write_bytes(b"")
        expect(root, "member missing: model.pt")

        root = build(base / "member-tampered")
        (root / "model.kwm").write_bytes(b"different")
        expect(root, "digest mismatch: model.kwm")

        # Development WAV identities: sorted, unique, counted, well formed.
        def corpus(root: pathlib.Path, payload: dict, needle: str) -> None:
            replace(
                root,
                "development-wav-sha256.json",
                json.dumps(payload).encode(),
                "development_wav_identities_sha256",
            )
            expect(root, needle)

        def wavs(hashes: list, count: int) -> dict:
            return {
                "evidence_class": "rnn-frozen-development-wav-identities",
                "model_family": "rnn",
                "wav_sha256": hashes,
                "wav_sha256_count": count,
            }

        corpus(build(base / "wav-unsorted"), wavs(list(reversed(WAV_HASHES)), 2), "sorted, unique and counted")
        corpus(build(base / "wav-duplicate"), wavs([WAV_HASHES[0]] * 2, 2), "sorted, unique and counted")
        corpus(build(base / "wav-miscount"), wavs(WAV_HASHES, 3), "sorted, unique and counted")
        corpus(build(base / "wav-malformed"), wavs(["z" * 64], 1), "invalid SHA256")

        # Selection evidence has to agree with the manifest it is sealed into.
        def reseal(root: pathlib.Path, **changes: object) -> None:
            path = root / "selection-evidence.json"
            value = json.loads(path.read_bytes().decode())
            value.update(changes)
            replace(
                root,
                "selection-evidence.json",
                json.dumps(value).encode(),
                "selection_evidence_sha256",
            )

        root = build(base / "round-mismatch")
        reseal(root, selected_round=4)
        expect(root, "selection round mismatch")

        root = build(base / "score-mismatch")
        reseal(root, selected_score=0.9)
        expect(root, "selection score mismatch")

        root = build(base / "evidence-class")
        reseal(root, evidence_class="gru-frozen-development-selection")
        expect(root, "selection evidence class mismatch")

        root = build(base / "evidence-qualified")
        reseal(root, qualification_used=True)
        expect(root, "selection evidence requires qualification_used=false")

        # The one non-arithmetic use of the development gate: a split that does
        # not qualify must fail the freeze even when every digest matches.
        Gate.qualified = False
        expect(good, "fails full development robustness")
        Gate.qualified = True
        assert isinstance(frozen.verify(good), dict), "the gate must recover"

    print("test_verify_rnn_frozen_candidate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
