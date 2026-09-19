#!/usr/bin/env python3
"""Fail when a workflow references a repository source path that does not exist.

A workflow is wiring, not prose. When it names a script, a config or another
workflow, that name is a hard dependency: a step runs it, reads it, or waits
for a change to it. A name that no longer resolves is not a stale comment, it
is a job that fails the moment it is dispatched.

This repository retired a whole research lane in one cleanup commit. That commit
deleted the scripts, the configs and most of the workflows that drove them; what
it could not see was that one surviving workflow still executed a deleted
script, still consumed the file that script produced, and still listed deleted
files in its ``paths:`` filter. Nothing failed, because that workflow only
triggers on a branch prefix nobody pushes any more, so the breakage stayed
invisible until someone dispatched it by hand.

The check is deliberately narrow, because a broad one is unusable:

* only ``.github/workflows/*.yml`` are scanned. Documentation that narrates
  retired work is closure evidence, not wiring; scanning prose for path-like
  tokens would flag the retirement record itself.
* only paths under a known source directory are considered. A workflow is full
  of runtime paths (``build/...``, ``$RUNNER_TEMP/...``) that never exist in the
  worktree and are not supposed to.
* lines containing a ``${{ }}`` expression are skipped: the path is computed at
  run time and cannot be resolved by reading.
* a path that a step of the same workflow creates is not dangling. The approval
  workflow copies governance files into a staging directory and hashes them
  there; that name should never exist in the worktree. Creation is recognised by
  a producer command (``cp``, ``mv``, ``install``, ``tee``, a redirect) naming
  the same basename.

That last rule is a heuristic, so it is applied per-basename within one file
rather than as a list of exempt paths: the exemption is earned by the workflow
that needs it and cannot leak into another workflow.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Directories that hold tracked source. A reference into one of these is a
# dependency on the repository; a reference anywhere else is a runtime path.
SOURCE_DIRS = (
    ".github/",
    "bench/",
    "cmake/",
    "commercial/",
    "configs/",
    "docs/",
    "eval/",
    "governance/",
    "include/",
    "models/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
    "training/",
)

PATH_RE = re.compile(
    r"(?<![\w./-])"
    r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:c|h|json|py|sh|tsv|ya?ml))"
    r"(?![\w.-])"
)

# Commands that produce a file inside the job, plus a shell redirect. A path is
# not dangling when a step of the same workflow writes that basename.
PRODUCER_RE = re.compile(r"(?:^|[\s;&|`])(?:cp|mv|install|rsync|tee)\b|>>?")


def names(text: str, name: str) -> bool:
    """True when *text* contains *name* as a whole path component."""
    return re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])", text) is not None


def scan_workflow(path: pathlib.Path, root: pathlib.Path) -> tuple[list[str], int, int]:
    """Return (problems, references considered, references that resolved)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    producing = [line for line in lines if PRODUCER_RE.search(line)]
    problems: list[str] = []
    considered = 0
    resolved = 0
    for number, line in enumerate(lines, 1):
        if "${" in line and "}}" in line:
            # Computed at run time; it cannot be resolved by reading.
            continue
        for match in PATH_RE.finditer(line):
            candidate = match.group(1)
            if not candidate.startswith(SOURCE_DIRS):
                continue
            considered += 1
            if (root / candidate).exists():
                resolved += 1
                continue
            basename = pathlib.Path(candidate).name
            if any(names(producer, basename) for producer in producing):
                # Created by a step of this workflow, so not expected in the tree.
                resolved += 1
                continue
            problems.append(f"{path.name}:{number}: missing {candidate}")
    return problems, considered, resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="repository root the references resolve against",
    )
    parser.add_argument(
        "workflow",
        nargs="+",
        type=pathlib.Path,
        help="workflow files or directories containing them",
    )
    args = parser.parse_args()

    files: list[pathlib.Path] = []
    for entry in args.workflow:
        if entry.is_dir():
            files.extend(sorted(entry.glob("*.yml")))
        elif entry.is_file():
            files.append(entry)
        else:
            raise ValueError(f"not a file or directory: {entry}")
    if not files:
        raise ValueError("no workflow files given: refusing to pass vacuously")

    problems: list[str] = []
    considered = 0
    resolved = 0
    for path in files:
        found, seen, ok = scan_workflow(path, args.root)
        problems.extend(found)
        considered += seen
        resolved += ok

    if considered == 0:
        raise ValueError(
            "no source path reference found in any workflow: wrong input, "
            "not a clean result"
        )

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(
            f"check_workflow_references: dangling={len(problems)} "
            f"references={considered}",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_workflow_references: ok workflows={len(files)} "
        f"references={considered} resolved={resolved}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
