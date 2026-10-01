#!/usr/bin/env python3
"""Bind one explicitly approved PR-label event to exact source and byte budgets.

No network/daemon operation happens here. The publishing owner places the exact
approval line in the PR body after reviewing the exact draft PR head, then adds the designated label.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller
import qualify as q

PREFIX = 'KWS_ASR_RUNTIME_APPROVAL='
REPOSITORY = 'jiying2007/kws-pipeline'
BRANCH = 'codex/github-asr-runtime-once-20261001'


def source_identity(root):
    root = Path(root)
    files = ['.github/workflows/research-asr-runtime.yml']
    for path in sorted((root / 'research/asr_runtime').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.pyc'):
            q.require(not path.is_symlink(), 'linked source file rejected')
            files.append(str(path.relative_to(root)))
    manifest = {name: {'sha256': q.sha256_file(root / name), 'bytes': (root / name).stat().st_size} for name in files}
    return hashlib.sha256(q.canonical(manifest)).hexdigest()


def validate_event(event, environ, admission, expected_source_sha256, actual_head, changed_paths, actual_parent):
    q.require(environ.get('GITHUB_EVENT_NAME') == 'pull_request' and event.get('action') == 'labeled', 'only the explicitly approved label event is eligible')
    q.require(event.get('label', {}).get('name') == admission['activation']['label'], 'wrong activation label')
    q.require(environ.get('GITHUB_RUN_ATTEMPT') == '1', 'automatic/manual rerun not admitted')
    pr = event.get('pull_request', {})
    q.require(event.get('repository', {}).get('full_name') == REPOSITORY and not event['repository'].get('private'), 'only the approved public repository is eligible')
    q.require(pr.get('head', {}).get('repo', {}).get('full_name') == REPOSITORY and pr['head'].get('ref') == BRANCH, 'untrusted execution branch')
    q.require(pr.get('user', {}).get('login') == 'jiying2007' and environ.get('GITHUB_ACTOR') == 'jiying2007', 'publishing owner identity mismatch')
    q.require(pr['head'].get('sha') == actual_head and re.fullmatch(r'[0-9a-f]{40}', actual_head), 'head SHA mismatch')
    q.require(actual_parent == admission['baseline_sha'] and pr.get('base', {}).get('sha') == actual_parent, 'reviewed PR base changed')
    number = event.get('number')
    q.require(type(number) is int and number > 0 and pr.get('number') == number, 'PR identity mismatch')
    q.require(changed_paths and all(p.startswith('research/asr_runtime/') or p == '.github/workflows/research-asr-runtime.yml' for p in changed_paths), 'changed files exceed runtime-only scope')
    approvals = [line[len(PREFIX):] for line in (pr.get('body') or '').splitlines() if line.startswith(PREFIX)]
    q.require(len(approvals) == 1 and len(approvals[0]) <= 1024, 'one bounded owner approval line is required')
    approval = json.loads(approvals[0])
    expected = {'armed': True, 'pr_number': number, 'base_sha': actual_parent, 'head_sha': actual_head,
                'nonce': admission['activation']['nonce'], 'source_sha256': expected_source_sha256,
                'dependency_bytes': admission['expected_input_bytes'], 'image_compressed_bytes': admission['image']['compressed_bytes'],
                'total_input_bytes': admission['expected_input_bytes'] + admission['image']['compressed_bytes'],
                'scope': 'single_no_model_dependency_cpu_qualification'}
    q.require(approval == expected, 'exact head/source/budget approval does not match')
    q.require(admission['execution_enabled'] is True, 'execution admission remains disabled')
    return expected


def git(root, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(root), *args], env=q.clean_env(), timeout=10, text=True).strip()


def main():
    root = Path(__file__).resolve().parents[2]
    admission = q.load_json(root / 'research/asr_runtime/locks/admission.json')
    event = q.load_json(Path(os.environ['GITHUB_EVENT_PATH']), maximum=1024 * 1024)
    head = git(root, 'rev-parse', 'HEAD')
    parent = admission['baseline_sha']
    q.require(re.fullmatch(r'[0-9a-f]{40}', parent), 'invalid fixed baseline')
    # Checkout is shallow. Fetch only the exact public base tree, never histories.
    git(root, 'fetch', '--no-tags', '--depth=1', 'origin', parent)
    q.require(git(root, 'status', '--porcelain') == '', 'working tree is not clean')
    paths = git(root, 'diff', '--name-only', parent, head).splitlines()
    approved = validate_event(event, os.environ, admission, source_identity(root), head, paths, parent)
    # All complete source/image/build locks must validate before a marker is issued.
    controller.validate_admission(root / 'research/asr_runtime')
    marker = Path(os.environ['RUNNER_TEMP']) / 'asr-runtime-admitted.json'
    q.write_receipt(marker, approved)
    print(json.dumps(approved, sort_keys=True))
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write('One no-model dependency/CPU run admitted for exact head ' + head + '.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
