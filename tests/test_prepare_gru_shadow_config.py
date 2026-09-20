from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_gru_shadow_config as prepare  # noqa: E402

# The shadow registrations are one-shot: the registry holds one
# reserved-untouched arena per family and consuming one cannot be undone. This
# finalizer had no test at all, including for the family check added when the
# RNN twin turned out to have one and this side did not.
#
# verify() is stubbed and ROOT is pointed at a temporary tree. Both are module
# attributes, so the tests exercise the real binding logic -- arena lookup,
# family, status, consumption, seeds, namespace overlap -- without having to
# build a frozen candidate that passes the full freeze verifier.

MODEL_SHA = "b" * 64
ARENA = "gru-test-shadow-v9"


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def arena_row(**overrides) -> dict:
    row = {
        "name": ARENA,
        "model_family": "gru",
        "status": "reserved-untouched",
        "seeds": list(range(1001, 1009)),
    }
    row.update(overrides)
    return row


def build(base: pathlib.Path, *, arenas=None, config=None, candidate_freeze=None) -> dict:
    root = base / "repo"
    candidate = base / "candidate"
    candidate.mkdir(parents=True)
    write_json(
        candidate / "source-config.json",
        {"qualification_holdout_seed": 555, "retired_qualification_holdout_seeds": [777]}
        if config is None
        else config,
    )
    write_json(
        candidate / "source-development-policy.json",
        {"candidate_freeze": {"shadow_arena": ARENA}}
        if candidate_freeze is None
        else {"candidate_freeze": candidate_freeze},
    )
    write_json(
        root / "experiments" / "model_family" / "shadow_arena_registry.json",
        {"arenas": [arena_row()] if arenas is None else arenas},
    )
    return {"root": root, "candidate": candidate, "output": base / "out" / "config.json"}


def run(parts: dict, *, registry_root=None) -> dict:
    real_root, real_verify = prepare.ROOT, prepare.verify
    try:
        prepare.ROOT = registry_root or parts["root"]
        prepare.verify = lambda candidate: {"model_sha256": MODEL_SHA, "policy": "gru-frozen-candidate-v1"}
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

        # Happy path: the arena binds and the rendered config carries its seeds.
        parts = build(base / "good")
        value = run(parts)
        assert value["arena"] == ARENA, value
        assert value["seeds"] == list(range(1001, 1009)), value
        assert value["model_sha256"] == MODEL_SHA, value
        rendered = json.loads(parts["output"].read_text(encoding="utf-8"))
        assert rendered["shadow_qualification"]["seeds"] == list(range(1001, 1009)), rendered
        assert rendered["shadow_qualification"]["enabled"] is True, rendered

        # The policy has to name a candidate_freeze at all.
        parts = build(base / "no-freeze", candidate_freeze="not-an-object")
        expect("has no candidate_freeze", lambda: run(parts))

        # An unknown arena is refused rather than silently binding nothing.
        parts = build(base / "unknown", arenas=[arena_row(name="somewhere-else")])
        expect("shadow arena is not registered", lambda: run(parts))

        # The family check: an arena reserved for the other lane must not be
        # consumable from this one. This is the check the RNN twin already had.
        parts = build(base / "wrong-family", arenas=[arena_row(model_family="rnn")])
        expect("shadow arena is not registered", lambda: run(parts))

        parts = build(base / "no-family", arenas=[arena_row(model_family="shared")])
        expect("shadow arena is not registered", lambda: run(parts))

        # An already opened arena is spent, whether or not the family matches.
        parts = build(base / "opened", arenas=[arena_row(status="opened")])
        expect("no longer reserved-untouched", lambda: run(parts))

        # A model that another arena already consumed cannot be reused.
        parts = build(
            base / "consumed",
            arenas=[
                arena_row(),
                arena_row(
                    name="earlier",
                    status="opened",
                    candidate_model_sha256=MODEL_SHA,
                ),
            ],
        )
        expect("already consumed by an earlier shadow arena", lambda: run(parts))

        # The seed contract: eight to sixteen, all distinct.
        for seeds in (list(range(1001, 1008)), list(range(1001, 1009)) + [1001]):
            parts = build(base / f"seeds-{len(seeds)}", arenas=[arena_row(seeds=seeds)])
            expect("seed contract is invalid", lambda p=parts: run(p))

        # The formal qualification namespace is off limits, including retired
        # seeds -- a shadow run must never touch a holdout.
        for seeds in (list(range(555, 563)), list(range(777, 785))):
            parts = build(base / f"overlap-{seeds[0]}", arenas=[arena_row(seeds=seeds)])
            expect("overlaps formal qualification namespace", lambda p=parts: run(p))

    print("test_prepare_gru_shadow_config: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
