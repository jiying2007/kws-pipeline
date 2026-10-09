#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys


# Keep CI and release on the same complete test-source set. Other inventories
# remain explicit through --workflow (for example research-only test roots).
OFFICIAL_WORKFLOWS = (
    ".github/workflows/ci.yml",
    ".github/workflows/training-integration.yml",
    ".github/workflows/training-audit-closure.yml",
    ".github/workflows/speech-like-external-base-bundle-contract.yml",
    ".github/workflows/product-training-data-contract.yml",
    ".github/workflows/release.yml",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("tests"))
    workflows = parser.add_mutually_exclusive_group(required=True)
    workflows.add_argument("--workflow", action="append", type=pathlib.Path)
    workflows.add_argument("--official-workflows", action="store_true",
                           help="use the shared CI/release Python test inventory")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="test filename intentionally executed only outside Actions",
    )
    args = parser.parse_args()
    if args.official_workflows:
        args.workflow = [pathlib.Path(path) for path in OFFICIAL_WORKFLOWS]
    tests = sorted(path.name for path in args.root.glob("test_*.py"))
    if not tests:
        raise ValueError("no Python tests discovered")
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in args.workflow
    )
    excluded = set(args.exclude)
    missing = [name for name in tests if name not in workflow_text and name not in excluded]
    unknown_exclusions = sorted(excluded - set(tests))
    if unknown_exclusions:
        raise ValueError(
            "inventory exclusions do not exist: " + ", ".join(unknown_exclusions)
        )
    if missing:
        print(
            "Python tests missing from declared workflows: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print(
        f"inventory covered {len(tests)} Python test file(s) across "
        f"{len(args.workflow)} workflow(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
