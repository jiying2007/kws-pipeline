#!/usr/bin/env python3
"""Report validation-strength drift between twin implementations.

This repository grows tools in pairs: verify_rnn_* next to verify_gru_*,
and a runtime bundle verifier next to a backend bundle verifier. Each pair
starts as a copy, and each pair has since drifted. Twice now the drift was
the interesting kind: one twin had a check the other did not, so one of the
two accepted input its sibling rejected.

This compares the checks, not the wording. Error strings, variable names and
log messages are expected to differ; a missing isinstance guard or a missing
`is not True` is not. The signal is deliberately narrow:

  * isinstance() type names per function
  * identity comparisons (is / is not) against True, False or None
  * membership tests (in / not in)
  * which functions exist at all

Divergence is not automatically a defect: the twins are allowed to differ.
So the current state is recorded in a baseline and the gate fails on two
things only -- a divergence that is not in the baseline, and a baseline
entry that no longer diverges (a stale waiver nobody removed). That way the
baseline doubles as the list of known differences, and fixing one forces a
baseline edit, which is the point.

Usage:
  check_twin_parity.py --root .                 # gate: compare to baseline
  check_twin_parity.py --root . --report        # print everything
  check_twin_parity.py --root . --update        # rewrite the baseline
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

BASELINE = pathlib.Path("configs/twin-parity-baseline.json")
SCAN_DIRS = ("tools", "training")
EXTRA_PAIRS = (
    ("tools/verify_speech_like_runtime_bundle.py", "tools/verify_speech_like_backend_bundle.py"),
)

# Twin names differ only by these tokens, so collapse them before comparing.
# Not \b: an underscore is a word character, so \b would miss the token in
# `_retain_rnn_identity` and two matching functions would look like one that
# exists in only one of the pair.
TOKEN_RE = re.compile(r"(?<![a-z0-9])(?:rnn|gru|runtime|backend)(?![a-z0-9])", re.IGNORECASE)

EXIT_OK = 0
EXIT_DIVERGENCE = 1
EXIT_USAGE = 2


def normalize(text: str) -> str:
    return TOKEN_RE.sub("*", text).strip().lower()


def guard_of(node: ast.AST) -> str | None:
    """Canonical form of a check, or None if this node is not one."""
    # `if not isinstance(x, dict): raise` is the dominant idiom here and is
    # the same check as the positive form, so look through the negation --
    # otherwise the scanner misses most of the guards in the repo.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return guard_of(node.operand)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "isinstance" and len(node.args) == 2:
            target = node.args[1]
            names: list[str] = []
            if isinstance(target, ast.Tuple):
                names = [ast.unparse(item) for item in target.elts]
            else:
                names = [ast.unparse(target)]
            return "isinstance:" + "|".join(sorted(normalize(n) for n in names))
        return None
    if isinstance(node, ast.Compare):
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(comparator, ast.Constant):
                if comparator.value in (True, False, None):
                    kind = "is" if isinstance(op, ast.Is) else "is-not"
                    return f"{kind}:{comparator.value}"
            if isinstance(op, (ast.In, ast.NotIn)):
                return "in" if isinstance(op, ast.In) else "not-in"
    return None


def guards_in(body: list[ast.stmt]) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        # Raise conditions and `if` tests are where checks live; walking the
        # whole function catches them wherever they sit.
        if isinstance(node, ast.If):
            test = guard_of(node.test)
            if test:
                found.add(test)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                test = guard_of(value)
                if test:
                    found.add(test)
        else:
            test = guard_of(node)
            if test:
                found.add(test)
    return found


def inventory(path: pathlib.Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        result[normalize(node.name)] = guards_in(node.body)
    return result


def discover_pairs(root: pathlib.Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for directory in SCAN_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*rnn*.py")):
            twin = path.with_name(path.name.replace("rnn", "gru"))
            if twin.is_file():
                pairs.append((path.relative_to(root).as_posix(), twin.relative_to(root).as_posix()))
    for left, right in EXTRA_PAIRS:
        if (root / left).is_file() and (root / right).is_file():
            pairs.append((left, right))
    return pairs


def divergences(root: pathlib.Path, pair: tuple[str, str]) -> dict[str, tuple[str, str, str]]:
    """Function -> (side, item) for every item present in only one twin."""
    left = inventory(root / pair[0])
    right = inventory(root / pair[1])
    found: dict[str, tuple[str, str, str]] = {}

    def owners(side: dict[str, set[str]], item: str) -> str:
        # The same check moved into a helper is still the same check. Naming
        # where the other twin performs it is what separates an organisation
        # difference from a genuinely missing guard.
        return ", ".join(sorted(n for n, guards in side.items() if item in guards))

    for name in sorted(set(left) | set(right)):
        # One-sided function is not drift: a twin may reuse the other's copy.
        # Comparing an absent function against a present one reports every
        # guard of the present side and reads like N missing checks, when the
        # fact is that there is nothing on this side to compare at all.
        if name not in left:
            found[f"{name}|right|missing-function"] = (pair[0], pair[1], f"{name}: only in right")
            continue
        if name not in right:
            found[f"{name}|left|missing-function"] = (pair[0], pair[1], f"{name}: only in left")
            continue
        only_left = left.get(name, set()) - right.get(name, set())
        only_right = right.get(name, set()) - left.get(name, set())
        for item in sorted(only_left):
            where = owners(right, item)
            note = f"{name}: only in left"
            if where:
                note += f" (right performs it in: {where})"
            found[f"{name}|left|{item}"] = (pair[0], pair[1], note)
        for item in sorted(only_right):
            where = owners(left, item)
            note = f"{name}: only in right"
            if where:
                note += f" (left performs it in: {where})"
            found[f"{name}|right|{item}"] = (pair[0], pair[1], note)
    return found


def key(pair: tuple[str, str], entry: str) -> str:
    return f"{pair[0]}|{pair[1]}|{entry}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path, default=None)
    parser.add_argument("--report", action="store_true", help="print every divergence, waived or not")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline from the current tree")
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_path = (args.baseline or root / BASELINE).resolve()
    pairs = discover_pairs(root)
    if not pairs:
        print("error: no twin pairs found", file=sys.stderr)
        return EXIT_USAGE

    current: dict[str, str] = {}
    for pair in pairs:
        for entry, note in divergences(root, pair).items():
            current[key(pair, entry)] = note[2]

    if args.update:
        payload = {"version": 1, "waived": sorted(current)}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"twin parity: baseline written with {len(current)} waived divergence(s) -> {baseline_path}")
        return EXIT_OK

    if not baseline_path.is_file():
        print(f"error: baseline is missing: {baseline_path}", file=sys.stderr)
        return EXIT_USAGE
    waived = set(json.loads(baseline_path.read_text(encoding="utf-8")).get("waived", []))

    unwaived = sorted(set(current) - waived)
    stale = sorted(waived - set(current))

    if args.report:
        for item in sorted(current):
            # Print the note for waived entries too: a waived divergence is
            # exactly the thing someone triages next, and the note is what says
            # whether it is a missing guard or just one moved into a helper.
            print(f"  {'waived ' if item in waived else 'OPEN   '} {item}  {current[item]}")

    print(f"twin parity: pairs={len(pairs)} divergences={len(current)} waived={len(waived) - len(stale)} open={len(unwaived)} stale={len(stale)}")
    if unwaived:
        print("error: new twin divergence (not in the baseline):", file=sys.stderr)
        for item in unwaived:
            print(f"  {item}  {current[item]}", file=sys.stderr)
    if stale:
        print("error: baseline entries that no longer diverge -- remove them:", file=sys.stderr)
        for item in stale:
            print(f"  {item}", file=sys.stderr)
    if unwaived or stale:
        return EXIT_DIVERGENCE
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
