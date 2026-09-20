from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import finalize_gru_frozen_candidate as finalize  # noqa: E402

# The RNN twin of this finalizer has an end-to-end test; this one had none, so
# the checks below had never been exercised by anything that fails.
#
# Deliberately written against this file's own wording rather than copied from
# the RNN test: the two differ in every message, and in one real way -- the RNN
# finalizer also checks model_family and architecture, this one does not (the
# freeze policy string already implies the family). Asserting the RNN messages
# here would pass against an implementation that had quietly become the RNN one.

SELECTION = finalize.SELECTION_POLICY
FREEZE_POLICY = "gru-frozen-candidate-v1"
LOOP_POLICY = "gru-development-curriculum-loop-v1"

WAV_SHA = "a" * 64
MODEL_SHA = "b" * 64


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record(round_index: int, score: float, sha: str = MODEL_SHA) -> dict:
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

    index = work / "datasets" / "round-0" / "domain-index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps({"split": "train", "wav_sha256": WAV_SHA}) + "\n",
        encoding="utf-8",
    )

    # select_record minimises on (score, -round, frontend), so round 0 has to
    # carry the best score to be the round the manifest claims was selected.
    records = [record(i, score=1.0 + i) for i in range(rounds)]
    development_path = work / "development-loop-manifest.json"
    write_json(
        development_path,
        {
            "policy": LOOP_POLICY,
            "model_family": "gru",
            "selection_policy": SELECTION,
            "selected_round": 0,
            "selected_score": records[0]["score"],
            "records": records,
        },
    )

    freeze_path = frozen / "freeze-manifest.json"
    write_json(
        freeze_path,
        {
            "policy": FREEZE_POLICY,
            "source_policy": LOOP_POLICY,
            "model_family": "gru",
            "selection_policy": SELECTION,
            "selected_round": 0,
            "model_sha256": MODEL_SHA,
            "config_sha256": finalize.sha256_file(config),
            "development_policy_sha256": finalize.sha256_file(policy),
        },
    )

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


def run(parts: dict) -> dict:
    return finalize.finalize(parts["work"], parts["config"], parts["policy"])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)

        # Happy path: every digest is bound back into the freeze manifest and
        # all five artifacts land in the frozen candidate.
        value = run(build(base / "good"))
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
            assert (base / "good" / "work" / "frozen-candidate" / name).is_file(), name

        # Missing inputs are named rather than surfacing as a KeyError.
        parts = build(base / "missing")
        parts["freeze_path"].unlink()
        expect("did not produce frozen candidate evidence", lambda: run(parts))

        # Identity chain: each field is checked on its own, unlike the RNN twin
        # there is no model_family or architecture check here -- the freeze
        # policy string already implies the family.
        for field, needle in (
            ("policy", "unexpected freeze policy"),
            ("selection_policy", "unexpected freeze selection policy"),
        ):
            parts = build(base / f"freeze-{field}")
            freeze = json.loads(parts["freeze_path"].read_text(encoding="utf-8"))
            freeze[field] = "wrong"
            write_json(parts["freeze_path"], freeze)
            expect(needle, lambda p=parts: run(p))

        # Assert the absence explicitly: a later addition of a family check is a
        # change worth seeing, not something to let through silently.
        parts = build(base / "freeze-family")
        freeze = json.loads(parts["freeze_path"].read_text(encoding="utf-8"))
        freeze["model_family"] = "rnn"
        write_json(parts["freeze_path"], freeze)
        run(parts)

        parts = build(base / "dev-policy")
        development = json.loads(parts["development_path"].read_text(encoding="utf-8"))
        development["policy"] = "wrong"
        write_json(parts["development_path"], development)
        expect("unexpected development loop policy", lambda p=parts: run(p))

        for number, value in enumerate(("not-an-object", {"selection_policy": "wrong"})):
            parts = build(base / f"policy-mismatch-{number}")
            write_json(
                parts["policy"],
                {"candidate_freeze": value, "stable_strict_pass_rounds": 2},
            )
            expect("source development selection policy mismatch", lambda p=parts: run(p))

        # Drift: the digests were recorded at freeze time, so editing either
        # file must fail even though the file itself is still well formed.
        parts = build(base / "config-drift")
        parts["config"].write_text('{"seed": 7}\n', encoding="utf-8")
        expect("source config SHA drifted", lambda p=parts: run(p))

        parts = build(base / "policy-drift")
        write_json(
            parts["policy"],
            {
                "candidate_freeze": {"selection_policy": SELECTION},
                "stable_strict_pass_rounds": 3,
            },
        )
        expect("source development policy SHA drifted", lambda p=parts: run(p))

        # Built with the bad value rather than edited afterwards: the policy
        # digest is checked first, so an edited policy would fail on drift and
        # never reach the round count.
        for number, bogus in enumerate((0, -1, True, "2", 2.0, None)):
            parts = build(base / f"required-{number}", required=bogus)
            expect("must be a positive integer", lambda p=parts: run(p))

        # Too short a streak is destructive on purpose: the frozen candidate is
        # removed and the loop marked unqualified.
        parts = build(base / "short-streak", rounds=2, required=3)
        expect("lacks stable strict-pass streak", lambda p=parts: run(p))
        assert not parts["frozen"].exists(), "frozen candidate should have been removed"
        development = json.loads(parts["development_path"].read_text(encoding="utf-8"))
        assert development["development_qualified"] is False, development
        assert development["selected_round"] is None, development

        # No WAV identity evidence is a failure, not an empty corpus.
        parts = build(base / "no-wav")
        for index in sorted((parts["work"] / "datasets").glob("round-*/domain-index.jsonl")):
            index.unlink()
        expect("no development/training WAV identity evidence", lambda p=parts: run(p))

        # The model the freeze names has to be the model the selection picked.
        parts = build(base / "model-mismatch")
        freeze = json.loads(parts["freeze_path"].read_text(encoding="utf-8"))
        freeze["model_sha256"] = "c" * 64
        write_json(parts["freeze_path"], freeze)
        expect("does not match selected development record", lambda p=parts: run(p))

    print("test_finalize_gru_frozen_candidate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
