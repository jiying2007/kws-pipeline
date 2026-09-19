#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

GLOB_CHARS = set("*?[]{}$")
WORKFLOW_SUFFIXES = {".yml", ".yaml"}
ALLOW_MISSING_MARKER = "workflow-path-filter: allow-missing"


def exact_path_filters(workflow: pathlib.Path) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    paths_indent: int | None = None

    for line_no, raw in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))

        if paths_indent is not None:
            if indent <= paths_indent:
                paths_indent = None
            elif stripped.startswith("- "):
                if ALLOW_MISSING_MARKER in raw:
                    continue
                value = stripped[2:].strip()
                if not value:
                    continue
                if value[0] in {"'", '"'}:
                    quote = value[0]
                    end = value.find(quote, 1)
                    if end < 0:
                        raise ValueError(f"{workflow}:{line_no}: unterminated quoted path filter")
                    value = value[1:end]
                else:
                    value = value.split(" #", 1)[0].strip()
                if value and not value.startswith("!"):
                    refs.append((line_no, value))
                continue

        if stripped == "paths:":
            paths_indent = indent

    return refs


def validate(workflow_dir: pathlib.Path, repo_root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    workflows = sorted(
        path
        for path in workflow_dir.iterdir()
        if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
    )
    if not workflows:
        raise ValueError(f"no workflow files found in {workflow_dir}")

    checked = 0
    for workflow in workflows:
        for line_no, value in exact_path_filters(workflow):
            if any(char in value for char in GLOB_CHARS):
                continue
            candidate = pathlib.PurePosixPath(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"{workflow}:{line_no}: unsafe exact path filter: {value}")
                continue
            checked += 1
            if not (repo_root / pathlib.Path(*candidate.parts)).exists():
                errors.append(f"{workflow}:{line_no}: exact path filter does not exist: {value}")

    if checked == 0:
        raise ValueError("workflow path-filter check was vacuous: no exact paths were checked")
    return errors


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="workflow-path-filter-self-test-") as tmp:
        root = pathlib.Path(tmp)
        workflows = root / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (root / "tools").mkdir()
        (root / "tools" / "ok.py").write_text("print('ok')\n", encoding="utf-8")

        good = workflows / "good.yml"
        good.write_text(
            "on:\n  push:\n    paths:\n      - 'tools/ok.py'\n      - 'training/**'\n",
            encoding="utf-8",
        )
        assert validate(workflows, root) == []

        ephemeral = workflows / "ephemeral.yml"
        ephemeral.write_text(
            "on:\n  push:\n    paths:\n"
            "      - '.github/triggers/generated.json' # workflow-path-filter: allow-missing\n",
            encoding="utf-8",
        )
        assert validate(workflows, root) == []

        bad = workflows / "bad.yml"
        bad.write_text(
            "on:\n  pull_request:\n    paths:\n      - tools/missing.py\n",
            encoding="utf-8",
        )
        errors = validate(workflows, root)
        assert len(errors) == 1 and "tools/missing.py" in errors[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_dir", nargs="?", type=pathlib.Path, default=pathlib.Path(".github/workflows"))
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("workflow path-filter self-test: PASS")
        return 0

    workflow_dir = args.workflow_dir.resolve()
    repo_root = args.repo_root.resolve()
    errors = validate(workflow_dir, repo_root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("workflow exact path filters: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
