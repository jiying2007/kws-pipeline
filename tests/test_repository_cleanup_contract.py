#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'repository-cleanup.yml'


def main() -> int:
    source = WORKFLOW.read_text(encoding='utf-8')

    for prefix in (
        'feature/', 'feat/', 'fix/', 'audit/', 'cleanup/', 'eval/',
        'exp/', 'experiment/', 'infra/', 'obs/', 'perf/', 'research/',
        'training/',
    ):
        assert prefix in source, f'repository cleanup safe prefix missing: {prefix}'

    assert 'CLOSED_PR_RETENTION_DAYS: "7"' in source

    for needle in (
        '[[ "${branch}" == "${default_branch}" ]] && continue',
        'if [[ "${protected}" == "true" ]]',
        '-f state=open -f head="${owner}:${branch}"',
        'skip open PR: ${branch}',
        'latest_sha=',
        'latest_open_prs=',
        'latest_active_runs=',
        '"${latest_sha}" != "${sha}"',
        '"${latest_open_prs}" != "0"',
        '"${latest_active_runs}" != "0"',
        'skip race: ${branch} changed while cleanup was running',
    ):
        assert needle in source, f'repository cleanup safety guard missing: {needle}'

    for needle in (
        '-f state=closed -f head="${owner}:${branch}"',
        'select(.merged_at != null and .head.sha == $sha)',
        'select(.merged_at == null and .head.sha == $sha)',
        '.closed_at',
        'retention_seconds="$((CLOSED_PR_RETENTION_DAYS * 86400))"',
        'age_seconds="$((now_epoch - closed_epoch))"',
        'closed-unmerged PR head older than ${CLOSED_PR_RETENTION_DAYS}d',
        'skip recently closed PR head: ${branch}',
    ):
        assert needle in source, f'closed-PR reclamation contract missing: {needle}'

    assert 'is_explicitly_retired "${branch}"' in source
    delete_line = 'git -C "${graph}" push --quiet origin ":refs/heads/${branch}"'
    assert source.count(delete_line) == 1, 'branch deletion must have one guarded owner'

    print('test_repository_cleanup_contract: ok')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
