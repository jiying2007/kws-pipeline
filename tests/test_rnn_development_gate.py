from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rnn_development_gate as gate  # noqa: E402

# The frozen-candidate contract test stubs this module's split arithmetic so
# each of its cases can be a one-field change. Stubbing is only honest if the
# stubbed thing is covered somewhere, so this file covers it.
#
# The asymmetry this closes: tests/test_gru_stability_evidence_contract.py
# exercises the GRU terminal_strict_streak, and nothing exercised the RNN one.
#
# evaluate_development_split composes three imports -- gate_values / base_gate /
# domain_gate from training/iterate_domain (8 test files) and evaluate_robustness
# from eval/gate_robustness (4 test files). Their own arithmetic is covered
# there; what is uncovered, and what actually decides whether a candidate is
# qualified, is the composition here: the AND between the base/domain gate and
# robustness, and the guard that a blocked robustness report is an error rather
# than a silent False.


class Stub:
    """Stand-ins for the two modules this one composes."""

    def __init__(self) -> None:
        self.base = True
        self.domain = True
        self.qualified = True
        self.blocked: object = False
        self.seen: object = None

    def gate_values(self, config: dict) -> dict:
        return dict(config or {})

    def base_gate(self, base: dict, gates: dict) -> bool:
        return self.base

    def domain_gate(self, domains: dict, gates: dict) -> bool:
        return self.domain

    def evaluate(self, summary: dict, config: dict) -> dict:
        self.seen = summary
        return {"qualified": self.qualified, "blocked": self.blocked}


def install(stub: Stub) -> Stub:
    gate.gate_values = stub.gate_values
    gate.base_gate = stub.base_gate
    gate.domain_gate = stub.domain_gate
    gate.evaluate_robustness = stub.evaluate
    return stub


def split(stub: Stub) -> dict:
    return gate.evaluate_development_split({}, {}, {})


def expect(needle: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), f"expected {needle!r}, got: {exc}"
        return
    raise AssertionError(f"expected a failure containing {needle!r}")


def strict(round_index: int) -> dict:
    return {"round": round_index, "calibration_gate": True, "test_gate": True}


def check_split() -> None:
    stub = install(Stub())
    value = split(stub)
    assert value["qualified"], value
    assert value["policy"] == gate.POLICY, value
    assert value["model_family"] == "rnn", value
    assert value["development_only"] is True, value
    # The domains report has to reach robustness under the key it expects.
    assert stub.seen == {"qualification_domains": {}}, stub.seen

    # qualified is the AND of both halves; neither half alone is enough.
    for base_ok, domain_ok, robust_ok in (
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, True),
    ):
        stub = install(Stub())
        stub.base = base_ok
        stub.domain = domain_ok
        stub.qualified = robust_ok
        value = split(stub)
        assert value["base_domain_qualified"] is (base_ok and domain_ok), value
        assert value["robustness_qualified"] is robust_ok, value

    # The robustness verdict has to already be a boolean. bool("false") is
    # True, so coercing it would certify a blocked evaluation; the landing
    # status reads this field with `is True`, which assumes a real boolean.
    for bogus in ("false", 1, 0, None, [], ""):
        stub = install(Stub())
        stub.qualified = bogus
        expect("must be a boolean", lambda: split(stub))
        assert value["qualified"] is (base_ok and domain_ok and robust_ok), value

    # A blocked robustness report is an error, not a False: something is wrong
    # with the run, and reporting it as "did not qualify" would hide that.
    for blocked in (True, None):
        stub = install(Stub())
        stub.blocked = blocked
        expect("unexpectedly blocked", lambda s=stub: split(s))


def check_streak() -> None:
    assert gate.terminal_strict_streak([]) == 0
    assert gate.terminal_strict_streak(None) == 0
    assert gate.terminal_strict_streak("not a list") == 0
    assert gate.terminal_strict_streak([strict(0), strict(1), strict(2)]) == 3

    # Only the trailing run counts: a break anywhere resets it.
    records = [
        strict(0),
        {"round": 1, "calibration_gate": True, "test_gate": False},
        strict(2),
        strict(3),
    ]
    assert gate.terminal_strict_streak(records) == 2, records
    assert gate.terminal_strict_streak([strict(0), strict(1), {"round": 2}]) == 0

    expect("must be an object", lambda: gate.terminal_strict_streak([strict(0), "nope"]))
    expect("round must be an integer", lambda: gate.terminal_strict_streak([{"round": True}]))
    expect("round must be an integer", lambda: gate.terminal_strict_streak([{"round": "0"}]))
    expect(
        "contiguous rounds starting at zero",
        lambda: gate.terminal_strict_streak([strict(1)]),
    )


def main() -> int:
    check_streak()
    check_split()
    print("test_rnn_development_gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
