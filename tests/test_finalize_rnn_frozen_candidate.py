from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import finalize_rnn_frozen_candidate as finalize  # noqa: E402

# finalize_rnn_frozen_candidate is called by three workflows and turns a
# development loop into the frozen candidate that qualification consumes. It
# had no test at all, so every check below was exercised only by the happy
# path of a real run -- which is to say, never by anything that fails.
#
# These cases cover the identity chain, the drift checks, the stability gate
# and its destructive failure path, and the evidence it writes.

SELECTION = finalize.SELECTION_POLICY
FREEZE = finalize.FREEZE_POLICY
SOURCE = finalize.SOURCE_POLICY
ARCH = finalize.ARCHITECTURE

WAV_SHA = "a" * 64
MODEL_SHA = "b" * 64


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record(round_index: int, score: float = 1.0, sha: str = MODEL_SHA) -> dict:
    return {
        "round": round_index,
        "score": score,
        "model_sha256": sha,
        "frontend": "logmel",
        "calibration_gate": True,
        "test_gate": True,
        "calibration": {"frr": 0.0, "false_rejects_path": "cal.tsv"},
        "calibration_domains": {"domains": {}},
        "test": {"frr": 0.0, "false_rejects_path": "test.tsv"},
        "test_domains": {"domains": {}},
    }


def build(base: pathlib.Path, *, rounds: int = 3, required: int = 2) -> dict:
    """Lay out a workspace that finalize() should accept. Returns the handles."""
    work = base / "work"
    frozen = work / "frozen-candidate"
    frozen.mkdir(parents=True)

    config = base / "config.json"
    config.write_text('{"seed": 1337}\n', encoding="utf-8")
    policy = base / "policy.json"
    write_json(
        policy,
        {
            "candidate_freeze": {"selection_policy": SELECTION},
            "stable_strict_pass_rounds": required,
        },
    )

    # The WAV identity evidence comes from the round indices; a tsv would need
    # real audio files, an index only needs a 64-character digest.
    index = work / "datasets" / "round-0" / "domain-index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps({"split": "train", "wav_sha256": WAV_SHA}) + "\n",
        encoding="utf-8",
    )

    # select_record minimises on (score, -round, frontend), so round 0 has to
    # carry the best score to be the round the manifest claims was selected.
    records = [record(i, score=1.0 + i) for i in range(rounds)]
    development = {
        "policy": SOURCE,
        "model_family": "rnn",
        "selection_policy": SELECTION,
        "selected_round": 0,
        "selected_score": records[0]["score"],
        "records": records,
    }
    development_path = work / "development-loop-manifest.json"
    write_json(development_path, development)

    freeze = {
        "policy": FREEZE,
        "source_policy": SOURCE,
        "model_family": "rnn",
        "architecture": ARCH,
        "selection_policy": SELECTION,
        "selected_round": 0,
        "model_sha256": MODEL_SHA,
        "config_sha256": finalize.sha256_file(config),
        "development_policy_sha256": finalize.sha256_file(policy),
    }
    freeze_path = frozen / "freeze-manifest.json"
    write_json(freeze_path, freeze)

    return {
        "work": work,
        "frozen": frozen,
        "config": config,
        "policy": policy,
        "freeze_path": freeze_path,
        "development_path": development_path,
    }


def expect(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def run(base: pathlib.Path, **kwargs) -> dict:
    parts = build(base / "case", **kwargs)
    return parts, finalize.finalize(parts["work"], parts["config"], parts["policy"])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)

        # Happy path: the evidence lands in the frozen candidate and the
        # manifest gains the digests that bind them together.
        parts, value = run(base / "good")
        assert value["selected_model_matches_selection_evidence"] is True, value
        for key in (
            "selection_evidence_sha256",
            "stability_evidence_sha256",
            "development_wav_identities_sha256",
            "source_config_snapshot_sha256",
            "source_development_policy_snapshot_sha256",
        ):
            assert len(str(value[key])) == 64, (key, value.get(key))
        for name in (
            "selection-evidence.json",
            "stability-evidence.json",
            "development-wav-sha256.json",
            "source-config.json",
            "source-development-policy.json",
        ):
            assert (parts["frozen"] / name).is_file(), name

        # Missing inputs are named rather than surfacing as a KeyError.
        parts = build(base / "missing")
        parts["freeze_path"].unlink()
        expect(
            "did not produce frozen candidate evidence",
            lambda: finalize.finalize(parts["work"], parts["config"], parts["policy"]),
        )

        # Identity chain: four independent fields, each checked separately.
        for field, needle in (
            ("policy", "unexpected RNN freeze policy"),
            ("source_policy", "unexpected RNN freeze policy"),
            ("model_family", "unexpected RNN frozen model identity"),
            ("architecture", "unexpected RNN frozen model identity"),
            ("selection_policy", "unexpected RNN selection policy"),
        ):
            parts = build(base / f"freeze-{field}")
            freeze = json.loads(parts["freeze_path"].read_text(encoding="utf-8"))
            freeze[field] = "wrong"
            write_json(parts["freeze_path"], freeze)
            expect(
                needle,
                lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
            )

        for field in ("policy", "model_family"):
            parts = build(base / f"dev-{field}")
            development = json.loads(parts["development_path"].read_text(encoding="utf-8"))
            development[field] = "wrong"
            write_json(parts["development_path"], development)
            expect(
                "unexpected RNN development identity",
                lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
            )

        # The policy handed in on the command line has to agree with the freeze
        # contract, not just be well formed.
        for number, value in enumerate(("not-an-object", {"selection_policy": "wrong"})):
            parts = build(base / f"policy-mismatch-{number}")
            write_json(
                parts["policy"],
                {
                    "candidate_freeze": value,
                    "stable_strict_pass_rounds": 2,
                },
            )
            expect(
                "source development selection policy mismatch",
                lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
            )

        # Drift: the freeze manifest recorded a digest, so editing the file the
        # digest covers must fail even though the file itself is still valid.
        parts = build(base / "config-drift")
        parts["config"].write_text('{"seed": 7}\n', encoding="utf-8")
        expect(
            "config SHA drifted",
            lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
        )

        parts = build(base / "policy-drift")
        write_json(
            parts["policy"],
            {
                "candidate_freeze": {"selection_policy": SELECTION},
                "stable_strict_pass_rounds": 3,
            },
        )
        expect(
            "policy SHA drifted",
            lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
        )

        # A required streak that is not a positive integer: True == 1 and
        # "2" would sail through a bare comparison.
        # Built with the bad value rather than edited afterwards: the policy
        # digest is checked before this one, so an edited policy would fail on
        # drift first and never reach the round count at all.
        for number, bogus in enumerate((0, -1, True, "2", 2.0, None)):
            parts = build(base / f"required-{number}", required=bogus)
            expect(
                "must be positive",
                lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
            )

        # Too short a streak is destructive on purpose: the frozen candidate is
        # removed and the loop is marked unqualified, so a later run cannot
        # mistake the leftover directory for a passing one.
        parts = build(base / "short-streak", rounds=2, required=3)
        expect(
            "lacks stable strict-pass streak",
            lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
        )
        assert not parts["frozen"].exists(), "frozen candidate should have been removed"
        development = json.loads(parts["development_path"].read_text(encoding="utf-8"))
        assert development["development_qualified"] is False, development
        assert development["selected_round"] is None, development

        # No WAV identity evidence at all is a failure, not an empty corpus.
        parts = build(base / "no-wav")
        for index in sorted((parts["work"] / "datasets").glob("round-*/domain-index.jsonl")):
            index.unlink()
        expect(
            "no development/training WAV identity evidence",
            lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
        )

        # The model the freeze names has to be the model the selection picked.
        parts = build(base / "model-mismatch")
        freeze = json.loads(parts["freeze_path"].read_text(encoding="utf-8"))
        freeze["model_sha256"] = "c" * 64
        write_json(parts["freeze_path"], freeze)
        expect(
            "does not match selected development record",
            lambda p=parts: finalize.finalize(p["work"], p["config"], p["policy"]),
        )

    print("test_finalize_rnn_frozen_candidate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
