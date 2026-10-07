#!/usr/bin/env python3
"""Bounded synthetic review; oracle uses raw CTC collapse and string.startswith.

No model, decoder, waveform, or saved posterior input is opened by this script.
All 6**T paths are visited, including zero-weight paths. The exhaustive menu is
585 matrices: fixed deterministic 你 好 followed by 0..3 rows from eight rows.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
TOKENS = ("blank", "你", "好", "小", "窝", "屋")
HERE = Path(__file__).resolve().parent
TOL = 2e-12  # Verification tolerance only; never used for a decision.


def row(**weights):
    return tuple(weights.get(token, 0) for token in TOKENS)


def one(token):
    return tuple(int(t == token) for t in TOKENS)


PREFIX = (one("你"), one("好"), one("小"))


def integer_rows(rows):
    result = []
    for values in rows:
        fs = tuple(Fraction(x) for x in values)
        den = math.lcm(*(x.denominator for x in fs))
        result.append(tuple(x.numerator * (den // x.denominator) for x in fs))
    return result


def raw_path_oracle(rows):
    """Independent reference. No DFA or optimized recurrence is used here."""
    integers = integer_rows(rows)
    denominator = math.prod(sum(values) for values in integers)
    numerators = [0, 0, 0]
    visited = 0
    nonzero = 0
    for path in product(range(6), repeat=len(integers)):
        visited += 1
        weight = 1
        for frame, token in enumerate(path):
            weight *= integers[frame][token]
            if not weight:
                break
        if not weight:
            continue
        nonzero += 1
        previous = None
        collapsed = []
        for token in path:
            if token != previous and token != 0:
                collapsed.append(TOKENS[token])
            previous = token
        text = "".join(collapsed)
        bucket = 0 if text.startswith("你好小窝") else 1 if text.startswith("你好小屋") else 2
        numerators[bucket] += weight
    assert visited == 6 ** len(rows)
    result = tuple(Fraction(value, denominator) for value in numerators)
    assert sum(result) == 1
    return result, visited, nonzero


def exact_ten_state(rows):
    """Literal rational transcription of the design, separate from the oracle."""
    state = (Fraction(1),) + (Fraction(0),) * 9
    trace = [state]
    for values in rows:
        total = sum(map(Fraction, values))
        p = tuple(Fraction(value) / total for value in values)
        e, b1, r1, b2, r2, b3, r3, a, b, d = state
        h1, h2, h3 = b1+r1, b2+r2, b3+r3
        mismatch = e * sum(p[2:])
        for blank_mass, repeat_mass, last, expected in (
            (b1, r1, 1, {2}), (b2, r2, 2, {3}), (b3, r3, 3, {4, 5})
        ):
            mismatch += blank_mass * sum(p[c] for c in range(1, 6) if c not in expected)
            mismatch += repeat_mass * sum(p[c] for c in range(1, 6) if c not in expected | {last})
        state = (
            p[0]*e, p[0]*h1, p[1]*(r1+e), p[0]*h2, p[2]*(r2+h1),
            p[0]*h3, p[3]*(r3+h2), a+p[4]*h3, b+p[5]*h3, d+mismatch,
        )
        assert min(state) >= 0 and sum(state) == 1
        assert state[7] >= trace[-1][7] and state[8] >= trace[-1][8]
        trace.append(state)
    return (state[7], state[8], sum(state[:7])+state[9]), trace


def matrices():
    menu = tuple(one(token) for token in TOKENS) + ((1,1,1,1,1,1), (1,0,0,0,1,1))
    for count in range(4):
        for number, suffix in enumerate(product(menu, repeat=count)):
            yield f"menu_{count}_{number}", PREFIX[:2]+suffix, None
    cases = [
        ("empty", (), (0,0,1)),
        ("silence", (one("blank"),)*3, (0,0,1)),
        ("incomplete", PREFIX, (0,0,1)),
        ("true_prefix", PREFIX+(one("窝"),), (1,0,0)),
        ("confuser_prefix", PREFIX+(one("屋"),), (0,1,0)),
        ("leading_blanks", (one("blank"),)*2+PREFIX+(one("窝"),one("blank")), (1,0,0)),
        ("adjacent_repeat", (one("你"),)+PREFIX+(one("窝"),), (1,0,0)),
        ("blank_separated_repeat", (one("你"),one("blank"))+PREFIX+(one("窝"),), (0,0,1)),
        ("n1_suffix_protection", PREFIX+(one("窝"),one("blank"),one("屋"),(1,2,3,4,5,6)), (1,0,0)),
        ("incompatible_start", (one("屋"),one("blank"))+PREFIX+(one("窝"),), (0,0,1)),
        ("alignment_marginal", PREFIX+(row(窝=2,blank=3),)*2, (Fraction(16,25),0,Fraction(9,25))),
        ("six_competitors", PREFIX+((1,1,1,1,1,1),), (Fraction(1,6),Fraction(1,6),Fraction(2,3))),
        ("binary_grouping", PREFIX+(row(窝=4,屋=3,blank=3),), (Fraction(2,5),Fraction(3,10),Fraction(3,10))),
        ("exact_tie", PREFIX+(row(窝=1,屋=1),), (Fraction(1,2),Fraction(1,2),0)),
        ("split_negative_tie", PREFIX+(row(窝=2,屋=1,blank=1),), (Fraction(1,2),Fraction(1,4),Fraction(1,4))),
        ("alignment_tie_23", PREFIX+(row(窝=1,blank=2),row(窝=1,blank=3)), (Fraction(1,2),0,Fraction(1,2))),
        ("alignment_tie_48", PREFIX+(row(窝=1,blank=4),row(窝=3,blank=5)), (Fraction(1,2),0,Fraction(1,2))),
        ("alignment_tie_456", PREFIX+(row(窝=1,blank=3),row(窝=1,blank=4),row(窝=1,blank=5)), (Fraction(1,2),0,Fraction(1,2))),
        ("prefix_product_tie", (row(你=2,屋=1),row(好=3,屋=1),one("小"),one("窝")), (Fraction(1,2),0,Fraction(1,2))),
        ("irreversible_false_wo", PREFIX+(row(窝=3,blank=2),one("屋"),(1,2,3,4,5,6)), (Fraction(3,5),Fraction(2,5),0)),
        ("changing_conditional_odds", PREFIX+(row(窝=4,屋=3,blank=3),one("屋")), (Fraction(2,5),Fraction(3,5),0)),
        ("repeat_across_blank_mixture", (row(你=2,blank=1), row(你=1,blank=1), row(你=1,好=2), row(好=1,小=2), row(小=1,窝=2), row(屋=1,blank=1)), None),
        ("all_six_nonuniform_7", ((1,2,3,4,5,6),(6,5,4,3,2,1),(2,3,5,7,11,13),(13,11,7,5,3,2),(2,2,3,3,5,5),(5,3,1,2,4,6),(3,1,4,1,5,9)), None),
    ]
    cases += [(f"uniform_{count}", ((1,1,1,1,1,1),)*count, None) for count in range(4,8)]
    yield from cases


def load_module(path):
    spec = importlib.util.spec_from_file_location("reviewed_k1_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_module(module, rows, expected, trace):
    verifier = module.K1PrefixMarginal()
    for index, values in enumerate(rows, 1):
        verifier.update(values)
        snap = verifier.snapshot()
        assert snap.frames == index
        assert all(abs(a-float(b)) <= TOL for a,b in zip(snap.state_masses, trace[index]))
        assert abs(sum(snap.state_masses)-1) <= TOL
        scale = module.CERTIFICATE_SCALE
        assert all(low <= exact*scale <= high for low,exact,high in zip(snap.state_lower_numerators, trace[index], snap.state_upper_numerators))
        assert scale-10*index <= sum(snap.state_lower_numerators) <= scale
        assert scale <= sum(snap.state_upper_numerators) <= scale+10*index
    snap = verifier.snapshot()
    actual = (snap.A,snap.B,snap.R)
    assert all(abs(a-float(b)) <= TOL for a,b in zip(actual,expected)), (actual,expected)
    result = verifier.finalize()
    target = "ACCEPT_K1" if expected[0] > expected[1]+expected[2] else "REJECT_K1"
    assert result.decision == target, {"decision":result.decision,"expected":target,"masses":actual,"logs":(snap.log_A,snap.log_B,snap.log_R)}
    for (low,high),exact in zip(snap.mass_bounds,expected):
        assert low <= exact <= high
    if result.decision_status == "CERTIFIED_ACCEPT":
        assert expected[0] > Fraction(1,2) and snap.mass_bounds[0][0] > Fraction(1,2)
    elif result.decision_status == "CERTIFIED_REJECT":
        assert expected[0] <= Fraction(1,2) and snap.mass_bounds[0][1] <= Fraction(1,2)
    else:
        assert result.decision_status == "NUMERICALLY_UNRESOLVED"
        assert snap.mass_bounds[0][0] <= Fraction(1,2) < snap.mass_bounds[0][1]
        assert result.decision == "REJECT_K1"
    assert result.emitted is True
    assert verifier.finalize().emitted is False
    for split in range(len(rows)+1):
        chunked = module.K1PrefixMarginal()
        chunked.update_chunk(rows[:split])
        chunked.update_chunk(())
        chunked.update_chunk(rows[split:])
        other = chunked.snapshot()
        assert other.state_log_masses == snap.state_log_masses
        assert other.state_lower_numerators == snap.state_lower_numerators
        assert other.state_upper_numerators == snap.state_upper_numerators
        assert chunked.finalize().decision == result.decision
    return result.decision_status


def run(module_path):
    module = load_module(module_path) if module_path else None
    report = {"exhaustive_menu_matrices":585,"cases":0,"raw_paths_visited":0,"nonzero_paths":0,"failures":[],"tie_cases":[],"decision_status_counts":{},"numerically_unresolved_cases":[]}
    for name, rows, fixture_expected in matrices():
        expected, paths, nonzero = raw_path_oracle(rows)
        exact, trace = exact_ten_state(rows)
        assert exact == expected, (name,exact,expected)
        if fixture_expected is not None:
            assert expected == fixture_expected, (name,expected,fixture_expected)
        report["cases"] += 1
        report["raw_paths_visited"] += paths
        report["nonzero_paths"] += nonzero
        if expected[0] == expected[1]+expected[2]:
            report["tie_cases"].append(name)
        if module:
            try:
                status = check_module(module, rows, expected, trace)
                report["decision_status_counts"][status] = report["decision_status_counts"].get(status,0)+1
                if status == "NUMERICALLY_UNRESOLVED":
                    report["numerically_unresolved_cases"].append(name)
            except Exception as exc:
                report["failures"].append({"case":name,"error":str(exc),"type":type(exc).__name__})
    report["oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if module_path:
        report["module_sha256"] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"oracle-results.json")
    args = parser.parse_args()
    result = run(args.module)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(bool(result["failures"]))
