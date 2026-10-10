#!/usr/bin/env python3
"""Read-only binding of a release tag (including annotated tags) to a commit.

GitHub's release target_commitish does not select the commit of an existing tag.
Use matching-refs to distinguish an absent tag from an API/authentication failure,
then dereference annotated tag objects. This never creates or moves a Git ref.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from urllib.parse import quote


def api(repository: str, path: str):
    result = subprocess.run(
        ["gh", "api", "--method", "GET", "-H", "Accept: application/vnd.github+json",
         "-H", "X-GitHub-Api-Version: 2022-11-28", f"repos/{repository}/git/{path}"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return json.loads(result.stdout)


def verify(repository: str, tag: str, expected_commit: str, *, allow_absent: bool = False) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository identity")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag)
            or ".." in tag or tag.endswith((".", ".lock"))):
        raise ValueError("invalid release tag")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("expected commit must be an exact 40-hex SHA")
    refs = api(repository, "matching-refs/tags/" + quote(tag, safe=""))
    if not isinstance(refs, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("ref"), str)
        or not isinstance(row.get("object"), dict) for row in refs
    ):
        raise ValueError("invalid matching-ref response")
    matches = [row for row in refs if row.get("ref") == "refs/tags/" + tag]
    if not matches:
        if allow_absent:
            return None
        raise ValueError("release tag is absent: " + tag)
    if len(matches) != 1:
        raise ValueError("duplicate release tag reference: " + tag)
    obj = matches[0].get("object")
    seen = set()
    for _ in range(32):
        if not isinstance(obj, dict) or not isinstance(obj.get("sha"), str):
            raise ValueError("invalid release tag object")
        sha, kind = obj["sha"], obj.get("type")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("invalid release tag object SHA")
        if kind == "commit":
            if sha != expected_commit:
                raise ValueError(f"release tag commit mismatch: {tag}: {sha} != {expected_commit}")
            return sha
        if kind != "tag" or sha in seen:
            raise ValueError("release tag does not resolve to a commit")
        seen.add(sha)
        annotated = api(repository, "tags/" + sha)
        if not isinstance(annotated, dict) or annotated.get("sha") != sha:
            raise ValueError("annotated tag object identity mismatch")
        obj = annotated.get("object")
    raise ValueError("release tag annotation depth exceeds limit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--allow-absent", action="store_true",
                        help="allow bootstrap creation only if the exact tag is absent")
    args = parser.parse_args()
    try:
        resolved = verify(args.repository, args.tag, args.expected_commit, allow_absent=args.allow_absent)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"release tag binding refused: {exc}", file=sys.stderr)
        return 1
    print(f"release tag binding: {args.tag}: {resolved or 'absent; bootstrap creation allowed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
