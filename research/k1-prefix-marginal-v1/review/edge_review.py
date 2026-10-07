#!/usr/bin/env python3
"""Focused API, numeric, and lifecycle checks on synthetic observations only."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from independent_review import PREFIX, TOKENS, exact_ten_state, load_module, one, raw_path_oracle, row

HERE = Path(__file__).resolve().parent


def expect_error(verifier, function, error):
    before = verifier.snapshot()
    try:
        function()
    except error:
        pass
    else:
        raise AssertionError(f"expected {error.__name__}")
    assert verifier.snapshot() == before, "failed call changed state"


def run(module_path):
    module = load_module(module_path)
    checks = []

    def passed(name):
        checks.append(name)

    for invalid in (TOKENS[:-1], tuple(reversed(TOKENS)), (*TOKENS[:-1], "other"), "blank你好小窝屋", None):
        try:
            module.K1PrefixMarginal(token_order=invalid)
        except module.InputError:
            pass
        else:
            raise AssertionError(f"invalid token map accepted: {invalid}")
    passed("missing/reordered/unknown token maps rejected")

    verifier = module.K1PrefixMarginal()
    verifier.update_chunk(PREFIX)
    invalid_rows = [
        (), (1,2,3,4,5), (1,2,3,4,5,6,7), (0,)*6,
        (-1,1,1,1,1,1), (math.nan,1,1,1,1,1), (math.inf,1,1,1,1,1),
        (-math.inf,1,1,1,1,1), (True,1,1,1,1,1), ("1",1,1,1,1,1),
        (complex(1,0),1,1,1,1,1), (10**400,1,1,1,1,1),
        (Fraction(1,10**400),1,1,1,1,1), {t:1 for t in TOKENS}, None,
    ]
    for invalid in invalid_rows:
        expect_error(verifier, lambda value=invalid: verifier.update(value), module.InputError)
        expect_error(verifier, lambda value=invalid: verifier.update_chunk((one("窝"),value)), module.InputError)
    passed(f"{len(invalid_rows)} invalid row types rejected; single-row and chunk rollback")

    def broken_chunk():
        yield one("窝")
        raise RuntimeError("synthetic iteration failure")
    expect_error(verifier, lambda:verifier.update_chunk(broken_chunk()), RuntimeError)

    def broken_row():
        yield 1
        raise RuntimeError("synthetic row iteration failure")
    expect_error(verifier, lambda:verifier.update(broken_row()), RuntimeError)
    passed("row and chunk iterator failure rollback")

    before = verifier.snapshot()
    assert verifier.update_chunk(()) == before
    snap = verifier.update(one("窝"))
    result = verifier.finalize()
    assert result.snapshot.frames == snap.frames and result.snapshot.state_log_masses == snap.state_log_masses
    assert result.emitted and result.decision == "ACCEPT_K1"
    repeated = verifier.finalize()
    assert not repeated.emitted and repeated.snapshot == result.snapshot and repeated.decision == result.decision
    expect_error(verifier, lambda:verifier.update(one("blank")), module.FinalizedError)
    expect_error(verifier, lambda:verifier.update_chunk(()), module.FinalizedError)
    verifier.reset()
    assert verifier.snapshot() == module.K1PrefixMarginal().snapshot()
    verifier.update_chunk(PREFIX[:2])
    verifier.reset()
    assert verifier.snapshot() == module.K1PrefixMarginal().snapshot()
    passed("empty chunks, EOF no transition, one emission, post-EOF block, finalized/partial reset")

    cap = module.K1PrefixMarginal()
    cap.update_chunk((one("blank") for _ in range(module.MAX_FRAMES-1)))
    assert cap.snapshot().frames == 4095
    expect_error(cap, lambda:cap.update_chunk((one("blank"),one("blank"))), module.CapacityError)
    cap.update(one("blank"))
    assert cap.snapshot().frames == 4096
    cap.update_chunk(())
    expect_error(cap, lambda:cap.update(one("blank")), module.CapacityError)
    assert cap.finalize().decision == "REJECT_K1"
    passed("4096-row limit, exact-capacity EOF, overflowing chunk rollback")

    # This longer fixture has a single nonzero path, classified directly.
    first_confuser = module.K1PrefixMarginal()
    first_confuser.update_chunk(PREFIX+(one("屋"),one("blank"))+PREFIX+(one("窝"),))
    assert first_confuser.snapshot().masses == (0.,1.,0.)
    assert first_confuser.finalize().decision == "REJECT_K1"
    passed("nine-row deterministic initial confuser remains B after later wake")

    class EndlessRow:
        reads = 0
        def __iter__(self):
            while True:
                self.reads += 1
                yield 1
    endless = EndlessRow()
    fresh = module.K1PrefixMarginal()
    expect_error(fresh, lambda:fresh.update(endless), module.InputError)
    assert endless.reads == 7
    passed("unbounded malformed row consumes only seven values")

    # Finite representable extremes; oracle interprets literal floats exactly.
    subnormal = math.ulp(0.0)
    maximum = sys.float_info.max
    extreme_cases = [
        ("finite_sum_overflow", PREFIX+((maximum,)*6,)),
        ("positive_subnormal_terminal", PREFIX+((maximum,0,0,0,subnormal,0),)),
        ("positive_subnormal_chain", tuple(tuple(subnormal if t==wanted else maximum if t=="屋" else 0 for t in TOKENS) for wanted in ("你","好","小","窝"))),
        ("smallest_uniform", PREFIX+((subnormal,)*6,)),
    ]
    extremes = []
    for name, rows in extreme_cases:
        exact, paths, _ = raw_path_oracle(rows)
        verifier = module.K1PrefixMarginal()
        verifier.update_chunk(rows)
        snap = verifier.snapshot()
        _, trace = exact_ten_state(rows)
        assert all(low <= value*module.CERTIFICATE_SCALE <= high for low,value,high in zip(snap.state_lower_numerators, trace[-1], snap.state_upper_numerators))
        for actual, expected in zip(snap.masses,exact):
            assert abs(actual-float(expected)) <= 2e-12
        for actual_log, expected in zip(snap.log_masses,exact):
            if expected == 0:
                assert actual_log == -math.inf
            else:
                with localcontext() as context:
                    context.prec = 100
                    reference = (Decimal(expected.numerator)/Decimal(expected.denominator)).ln()
                assert math.isfinite(actual_log)
                assert abs(actual_log-float(reference)) < 1e-9, (name,actual_log,reference)
        target = "ACCEPT_K1" if exact[0] > exact[1]+exact[2] else "REJECT_K1"
        assert verifier.finalize().decision == target
        extremes.append({"case":name,"raw_paths":paths,"A":snap.A,"log_A":snap.log_A})
    passed("finite max/subnormal weights, full-row normalization, tiny positive logs")

    cases = []
    for terminal in (math.nextafter(1.,0.),1.,math.nextafter(1.,math.inf)):
        rows = PREFIX+(row(窝=terminal,屋=1),)
        verifier = module.K1PrefixMarginal()
        verifier.update_chunk(rows)
        expected = "ACCEPT_K1" if terminal > 1 else "REJECT_K1"
        result = verifier.finalize()
        assert result.decision == expected
        cases.append({"terminal_weight":terminal,"decision":result.decision})
    passed("exact tie and nearest binary64 weights on both sides")

    original = ((2,3,5,7,11,13),(13,11,7,5,3,2),(1,2,4,8,16,32),(3,5,7,11,13,17),(17,13,11,7,5,3))
    left, right = module.K1PrefixMarginal(), module.K1PrefixMarginal()
    left.update_chunk(original)
    right.update_chunk(tuple(tuple(math.ldexp(value, exponent) for value in values) for values,exponent in zip(original,(-900,-300,0,300,900))))
    assert all(abs(a-b)<1e-12 for a,b in zip(left.snapshot().masses,right.snapshot().masses))
    assert left.finalize().decision == right.finalize().decision
    passed("per-row common powers-of-two scaling invariance across exponent range")

    long = module.K1PrefixMarginal()
    previous = long.snapshot()
    for _ in range(module.MAX_FRAMES):
        snap = long.update((1,2,3,4,5,6))
        assert snap.log_A >= previous.log_A and snap.log_B >= previous.log_B
        assert snap.R <= previous.R+2e-14
        assert abs(sum(snap.masses)-1) < 1e-10
        previous = snap
    assert long.snapshot().frames == module.MAX_FRAMES
    passed("4096 nontrivial rows: conservation and absorbed-mass monotonicity")

    return {
        "checks_passed":checks,"extreme_cases":extremes,"boundary_cases":cases,
        "module_sha256":hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "review_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=HERE/"edge-results.json")
    args = parser.parse_args()
    report = run(args.module)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
