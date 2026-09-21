#!/usr/bin/env python3
"""Report bool() coercions applied to evidence read from outside the process.

bool("false"), bool(0) and bool("no") are all True. That is harmless while
the value is already a real boolean -- which is why most of the ~180 coercions
in this tree are fine -- and it is a silent upgrade the moment the value comes
from a file: a recorded failure becomes a pass, in a receipt, a registry or a
manifest that the next gate reads.

So the signal is deliberately narrow. A coercion is reported only when both
hold:

  * the value comes from outside the function -- a parameter, or a local read
    out of JSON by load_object / json.loads / read_text;
  * the result is kept -- assigned to a name, written into a subscript, or
    returned -- so it can become evidence rather than just pick a branch.

Coercing something the function computed itself is a no-op, and coercing
something to choose a branch is a style question, not a correctness one.

Divergence is not automatically a defect, so the current state is recorded in
a baseline and the gate fails on two things only -- a coercion that is not in
the baseline, and a baseline entry that no longer exists.

Usage:
  check_bool_coercions.py --root .                  # gate: compare to baseline
  check_bool_coercions.py --root . --report         # print everything
  check_bool_coercions.py --root . --update         # rewrite the baseline
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

BASELINE = pathlib.Path("configs/bool-coercion-baseline.json")
SCAN_DIRS = ("tools", "training", "eval", "governance")

# Reads that bring a value in from a file.
LOADERS = ("load_object", "json.loads", "read_text", "loads")
# A parameter is untrusted by definition; a value built here is not.
EXTERNAL_ATTRS = ("get",)

EXIT_OK = 0
EXIT_DIVERGENCE = 1
EXIT_USAGE = 2


def base_name(node: ast.AST) -> str | None:
    """`row.get("k")` / `row["k"]` -> "row"."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in EXTERNAL_ATTRS and node.args:
            return base_name(func.value)
        return None
    if isinstance(node, ast.Subscript):
        return base_name(node.value)
    if isinstance(node, ast.Name):
        return node.id
    return None


def loader_call(node: ast.AST) -> bool:
    """True when this expression reads a value out of JSON or a file."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name in LOADERS:
            return True
        # manifest.get("records") is still the same object read from disk.
        return loader_call(func.value) if isinstance(func, ast.Attribute) else False
    return False


def assigned_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple):
        return [item.id for item in target.elts if isinstance(item, ast.Name)]
    if isinstance(target, ast.Subscript):
        # `record["gate"] = ...` mutates the dict the name refers to, so the
        # name itself is not rebound -- but the write is what matters elsewhere.
        return [target.value.id] if isinstance(target.value, ast.Name) else []
    return []


def external_names(fn: ast.AST) -> set[str]:
    names = {arg.arg for arg in list(fn.args.args) + list(fn.args.kwonlyargs)}
    # `for row in records:` -- row is one of the records, so it is as external
    # as records is. This is how every streak and filter in this repo reads,
    # so without it the scanner misses exactly the shape it exists to catch.
    for _ in range(3):
        for node in ast.walk(fn):
            if isinstance(node, ast.For):
                if base_name(node.iter) in names or loader_call(node.iter):
                    names.update(assigned_names(node.target))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is None or not loader_call(value):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(assigned_names(target))
    return names


def kept_targets(fn: ast.AST) -> dict[int, str]:
    """Map the line of a bool() call -> what its result is stored into."""
    out: dict[int, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            for target in targets:
                if ast.unparse(target) != "_":
                    out.setdefault(node.lineno, ast.unparse(target))
        elif isinstance(node, ast.Return) and node.value is not None:
            out.setdefault(node.lineno, "<return>")
    return out


def coercions(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        external = external_names(fn)
        kept = kept_targets(fn)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "bool":
                continue
            if not node.args:
                continue
            if base_name(node.args[0]) not in external:
                continue
            # Same line as the enclosing assignment counts as kept.
            if not any(abs(line - node.lineno) <= 1 for line in kept):
                continue
            target = min(kept, key=lambda line: abs(line - node.lineno))
            # No line number in the key: an edit anywhere above a coercion
            # would otherwise move it and report one stale plus one new entry
            # for a change that did not touch the coercion at all.
            found.append(f"{fn.name}|{kept[target]} = {ast.unparse(node)}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path, default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_path = (args.baseline or root / BASELINE).resolve()

    current: list[str] = []
    for directory in SCAN_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            for item in coercions(path):
                current.append(f"{relative}|{item}")
    current = sorted(set(current))

    if args.update:
        payload = {"version": 1, "waived": current}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"bool coercions: baseline written with {len(current)} entry(ies) -> {baseline_path}")
        return EXIT_OK

    if args.report:
        for item in current:
            print(f"  {item}")

    if not baseline_path.is_file():
        print(f"error: baseline is missing: {baseline_path}", file=sys.stderr)
        return EXIT_USAGE
    waived = set(json.loads(baseline_path.read_text(encoding="utf-8")).get("waived", []))
    unwaived = sorted(set(current) - waived)
    stale = sorted(waived - set(current))

    print(
        f"bool coercions: files={len({item.split('|')[0] for item in current})} "
        f"coercions={len(current)} waived={len(waived) - len(stale)} "
        f"open={len(unwaived)} stale={len(stale)}"
    )
    if unwaived:
        print("error: new bool coercion on external evidence (not in the baseline):", file=sys.stderr)
        for item in unwaived:
            print(f"  {item}", file=sys.stderr)
    if stale:
        print("error: baseline entries that no longer exist -- remove them:", file=sys.stderr)
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
