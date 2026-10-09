#!/usr/bin/env python3
"""Select existing research checks without an API call or a path-filtered gate."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

EXACT = frozenset({
    'tools/verify_research_sources.py', 'tools/verify_research_publication.py',
    'tools/run_research_source_checks.py', 'tools/check_workflow_path_filters.py',
    'tools/research_ci_changes.py', 'tests/test_research_ci_changes.py',
})


def relevant(paths: list[str]) -> bool:
    return any(path.startswith(('research/', '.github/workflows/')) or path in EXACT
               for path in paths)


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True)


def changed_paths(event: str, payload: dict, ref: str) -> list[str]:
    if event == 'pull_request':
        # Default actions/checkout checks out GitHub's PR merge commit. Its first
        # parent is the current base, so this includes the complete merged PR
        # delta, not just the most recent head commit or unrelated base changes.
        parents = git('show', '-s', '--format=%P', 'HEAD').split()
        if len(parents) != 2:
            raise ValueError('expected the GitHub pull-request merge commit')
        base = parents[0]
    elif event == 'push' and ref == 'refs/heads/main':
        base = payload['before']
        if not isinstance(base, str) or len(base) != 40 or any(c not in '0123456789abcdef' for c in base):
            raise ValueError('invalid push before SHA')
        if base == '0' * 40:
            return git('ls-files', '-z').split('\0')
        exists = subprocess.run(['git', 'cat-file', '-e', base + '^{commit}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if exists.returncode:
            # A multi-commit/force push may exceed checkout's two-commit depth.
            # Fetch only its exact before SHA once. Unavailable history fails
            # closed; it is never converted to "no relevant changes".
            subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', base], check=True)
    else:
        # Preserve the previous source workflow's event scope. Feature/codex
        # pushes are covered by their PR; consolidate pushes retain their own
        # path-filtered source workflow trigger.
        return []
    return git('diff', '--name-only', '--no-renames', '-z', base, 'HEAD', '--').split('\0')


def main() -> None:
    event = os.environ['GITHUB_EVENT_NAME']
    payload = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    needed = relevant(changed_paths(event, payload, os.environ['GITHUB_REF']))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'required={str(needed).lower()}\n')
    print(f'research source checks required: {needed}')


if __name__ == '__main__':
    main()
