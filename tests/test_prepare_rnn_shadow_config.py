from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_rnn_shadow_config as prepare  # noqa: E402

# The shadow registrations are one-shot: the registry holds one
# reserved-untouched arena per family and consuming one cannot be undone. This
# side of the pair is the one that already checked the arena's family, so the
# suite pins that check as well as the contract around it.
#
# verify() is stubbed and ROOT is pointed at a temporary tree. Both are module
# attributes, so the tests exercise the real binding logic without having to
# build a frozen candidate that passes the full freeze verifier.
#
# Every expectation names this side's own wording ("RNN shadow arena ...").
# Copying the GRU suite would produce a test that still passes if this module
# silently became the GRU one, which is the failure this file exists to catch.

MODEL_SHA = "a" * 64
ARENA = "rnn-test-shadow-v9"
RNN_FORMAL = 4242
GRU_HOLDOUT = 555


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def arena_row(**overrides) -> dict:
    row = {
        "name": ARENA,
        "model_family": "rnn",
        "status": "reserved-untouched",
        "seeds": list(range(2001, 2009)),
    }
    row.update(overrides)
    return row


def build(base: pathlib.Path, *, arenas=None, config=None, candidate_freeze=None) -> dict:
    root = base / "repo"
    candidate = base / "candidate"
    candidate.mkdir(parents=True)
    write_json(
        candidate / "source-config.json",
        {
            "qualification_holdout_seed": GRU_HOLDOUT,
            "retired_qualification_holdout_seeds": [777],
        }
        if config is None
        else config,
    )
    write_json(
        candidate / "source-development-policy.json",
        {
            "candidate_freeze": {
                "shadow_arena": ARENA,
                "formal_qualification_seed": RNN_FORMAL,
            }
        }
        if candidate_freeze is None
        else {"candidate_freeze": candidate_freeze},
    )
    write_json(
        root / "experiments" / "model_family" / "shadow_arena_registry.json",
        {"arenas": [arena_row()] if arenas is None else arenas},
    )
    return {"root": root, "candidate": candidate, "output": base / "out" / "config.json"}


def run(parts: dict) -> dict:
    real_root, real_verify = prepare.ROOT, prepare.verify
    try:
        prepare.ROOT = parts["root"]
        prepare.verify = lambda candidate: {
            "model_sha256": MODEL_SHA,
            "policy": "rnn-frozen-candidate-v1",
        }
        return prepare.prepare(parts["candidate"], parts["output"])
    finally:
        prepare.ROOT, prepare.verify = real_root, real_verify


def expect(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)

        # Happy path: the arena binds, and the receipt names both formal seeds
        # -- this side reserves its own and reports the sibling's as untouched.
        parts = build(base / "good")
        value = run(parts)
        assert value["policy"] == "frozen-rnn-shadow-arena-binding-v1", value
        assert value["model_family"] == "rnn", value
        assert value["arena"] == ARENA, value
        assert value["arena_status"] == "reserved-untouched", value
        assert value["seeds"] == list(range(2001, 2009)), value
        assert value["model_sha256"] == MODEL_SHA, value
        assert value["candidate_policy"] == "rnn-frozen-candidate-v1", value
        assert value["rnn_formal_seed_reserved"] == RNN_FORMAL, value
        assert value["gru_formal_seed_untouched"] == GRU_HOLDOUT, value
        rendered = json.loads(parts["output"].read_text(encoding="utf-8"))
        assert rendered["shadow_qualification"]["seeds"] == list(range(2001, 2009)), rendered
        assert rendered["shadow_qualification"]["enabled"] is True, rendered
        assert rendered["shadow_qualification"]["expected_wakes_per_seed"] == 256, rendered

        # The policy has to name a candidate_freeze at all.
        parts = build(base / "no-freeze", candidate_freeze="not-an-object")
        expect("RNN source development policy has no candidate_freeze", lambda: run(parts))

        # An unknown arena is refused rather than silently binding nothing.
        parts = build(base / "unknown", arenas=[arena_row(name="somewhere-else")])
        expect("RNN shadow arena is not registered", lambda: run(parts))

        # The family check: an arena reserved for the other lane must not be
        # consumable from this one.
        parts = build(base / "wrong-family", arenas=[arena_row(model_family="gru")])
        expect("RNN shadow arena is not registered", lambda: run(parts))

        parts = build(base / "no-family", arenas=[arena_row(model_family="shared")])
        expect("RNN shadow arena is not registered", lambda: run(parts))

        # Already opened means spent, whether or not the family matches.
        parts = build(base / "opened", arenas=[arena_row(status="opened")])
        expect("RNN shadow arena is no longer reserved-untouched", lambda: run(parts))

        # A model that another arena already consumed cannot be reused.
        parts = build(
            base / "consumed",
            arenas=[
                arena_row(),
                arena_row(name="earlier", status="opened", candidate_model_sha256=MODEL_SHA),
            ],
        )
        expect("RNN model already consumed by an earlier shadow arena", lambda: run(parts))

        # ...but another lane's consumption of the same model is not this lane's
        # problem: the loop only inspects arenas of its own family.
        parts = build(
            base / "other-lane-consumed",
            arenas=[
                arena_row(),
                arena_row(
                    name="gru-earlier",
                    model_family="gru",
                    status="opened",
                    candidate_model_sha256=MODEL_SHA,
                ),
            ],
        )
        value = run(parts)
        assert value["arena"] == ARENA, value

        # The seed contract: eight to sixteen, all distinct.
        for seeds in (list(range(2001, 2008)), list(range(2001, 2009)) + [2001]):
            parts = build(base / f"seeds-{len(seeds)}", arenas=[arena_row(seeds=seeds)])
            expect("RNN shadow arena seed contract is invalid", lambda p=parts: run(p))

        # The formal qualification namespace is off limits, including the
        # sibling lane's holdout and every retired seed.
        for seeds in (
            list(range(GRU_HOLDOUT, GRU_HOLDOUT + 8)),
            list(range(RNN_FORMAL, RNN_FORMAL + 8)),
            list(range(777, 785)),
        ):
            parts = build(base / f"overlap-{seeds[0]}", arenas=[arena_row(seeds=seeds)])
            expect("RNN shadow arena overlaps formal qualification namespace", lambda p=parts: run(p))

        # This side also reserves a formal seed of its own, and refuses to run
        # without one -- or with one that is really the sibling lane's holdout.
        parts = build(
            base / "no-formal",
            candidate_freeze={"shadow_arena": ARENA},
        )
        expect("RNN formal seed is not independently reserved", lambda: run(parts))

        parts = build(
            base / "shared-formal",
            candidate_freeze={"shadow_arena": ARENA, "formal_qualification_seed": GRU_HOLDOUT},
        )
        expect("RNN formal seed is not independently reserved", lambda: run(parts))

    print("test_prepare_rnn_shadow_config: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
