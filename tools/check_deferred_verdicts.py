#!/usr/bin/env python3
"""Fail-closed gate for deferred workflow verdicts.

Several workflows run a long step like this::

    - id: develop
      run: |
        set +e
        <long running command>
        code=$?
        echo "exit_code=$code" >> "$GITHUB_OUTPUT"
        exit 0

Deferring the verdict is legitimate and deliberate: it lets later steps collect
and upload evidence before the job ends. But the pattern is only correct when
**some later step redeems the captured code**. Nothing enforces that. One
workflow in this repository captured ``exit_code`` and never asserted it, so a
failing development loop still produced a green run whose only completion check
was ``test -f <some file>``.

The check is deliberately about the capture, not about the wrapping::
``continue-on-error`` and ``|| true`` are separate decisions with their own
(visible, reviewable) syntax. The deferred pattern is the dangerous one because
it looks like error handling while quietly discarding the result.

Matching is done on the raw file text for the "is it referenced?" half rather
than on the parsed structure, because the reference legitimately appears in
places the step list does not cover -- ``if:`` conditions on later steps, and
other jobs of the same workflow.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

CAPTURE_MARKER = "exit_code"
OUTPUT_MARKER = "GITHUB_OUTPUT"
USE_TEMPLATE = "steps.{}.outputs.exit_code"


def captures_in(job: dict) -> list[tuple[str, str]]:
    """Return [(step id, step name)] for steps that capture an exit code."""
    found: list[tuple[str, str]] = []
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        if not step_id:
            continue
        run = str(step.get("run") or "")
        if CAPTURE_MARKER in run and OUTPUT_MARKER in run:
            found.append((str(step_id), str(step.get("name") or step_id)))
    return found


def check_workflow(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    jobs = (data or {}).get("jobs") or {}
    problems: list[str] = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for step_id, label in captures_in(job):
            if USE_TEMPLATE.format(step_id) not in text:
                problems.append(
                    f"{path.name}: job '{job_name}' step '{label}' (id={step_id}) "
                    f"captures exit_code and exits 0, but nothing asserts "
                    f"{USE_TEMPLATE.format(step_id)}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    captures = 0
    for path in files:
        try:
            jobs = ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {})
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name}: invalid YAML: {exc}") from None
        for job in jobs.values():
            if isinstance(job, dict):
                captures += len(captures_in(job))
        problems.extend(check_workflow(path))

    if captures == 0:
        raise ValueError(
            "no deferred verdict found in any workflow: wrong input, not a clean result"
        )

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"check_deferred_verdicts: ok workflows={len(files)} deferred_verdicts={captures}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
