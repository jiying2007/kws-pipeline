"""One-shot, aggregate-only runner storage measurement. Standard library only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = 'runner-storage-20261104-v1'
REPOSITORY = 'jiying2007/kws-pipeline'
REF = 'refs/heads/research/' + CAMPAIGN
WORKFLOW = '.github/workflows/' + CAMPAIGN + '.yml'
OUTPUT = 'runner-storage-public-' + CAMPAIGN
RESERVE_BYTES = 25 * 1024 ** 3
SOURCE_FILES = (
    WORKFLOW,
    'research/runner-storage-20261104/README.md',
    'research/runner-storage-20261104/collect_storage.py',
    'research/runner-storage-20261104/test_storage.py',
)
LOCK_PATH = 'research/runner-storage-20261104/source-lock.json'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bounded_read(path, maximum):
    with Path(path).open('rb') as stream:
        value = stream.read(maximum + 1)
    require(len(value) <= maximum, 'bounded input')
    return value


def admission(env, event, root):
    """Reject all but the first creation of the exact public campaign branch."""
    sha = env.get('GITHUB_SHA', '')
    run_id = env.get('GITHUB_RUN_ID', '')
    require(isinstance(event, dict), 'event object')
    repository = event.get('repository', {})
    require(isinstance(repository, dict), 'repository object')
    require(
        env.get('GITHUB_REPOSITORY') == REPOSITORY and
        env.get('GITHUB_REF') == REF and
        env.get('GITHUB_EVENT_NAME') == 'push' and
        env.get('GITHUB_RUN_NUMBER') == '1' and
        env.get('GITHUB_RUN_ATTEMPT') == '1' and
        env.get('GITHUB_ACTIONS') == 'true' and
        env.get('RUNNER_ENVIRONMENT') == 'github-hosted' and
        env.get('RUNNER_OS') == 'Linux' and
        env.get('RUNNER_ARCH') == 'X64' and
        env.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/' + WORKFLOW + '@' + REF and
        env.get('GITHUB_WORKFLOW_SHA') == sha and
        re.fullmatch('[0-9a-f]{40}', sha) is not None and
        re.fullmatch('[1-9][0-9]{0,19}', run_id) is not None and
        event.get('created') is True and event.get('deleted') is False and
        event.get('before') == '0' * 40 and event.get('after') == sha and
        event.get('ref') == REF and
        repository.get('private') is False and
        repository.get('full_name') == REPOSITORY,
        'exact first new public hosted campaign')
    require(env.get('GITHUB_WORKSPACE') and
            Path(env['GITHUB_WORKSPACE']).resolve() == Path(root).resolve(),
            'exact checked out workspace')
    image_os = env.get('ImageOS', '')
    image_version = env.get('ImageVersion', '')
    image_os = image_os if image_os == 'ubuntu24' else None
    image_version = image_version if isinstance(image_version, str) and re.fullmatch(
        '[0-9]{8}\\.[0-9]{1,3}\\.[0-9]{1,3}', image_version) is not None else None
    return dict(repository=REPOSITORY, ref=REF, head_sha=sha,
                workflow=WORKFLOW, workflow_sha=sha, run_id=int(run_id),
                run_number=1, run_attempt=1,
                runner_image=dict(os=image_os, version=image_version,
                                  metadata_complete=image_os is not None and image_version is not None),
                run_url='https://github.com/' + REPOSITORY + '/actions/runs/' + run_id)


def verify_sources(root):
    """Hash only the four fixed public source files; never walk a directory."""
    root = Path(root)
    for relative in SOURCE_FILES + (LOCK_PATH,):
        current = root
        for part in Path(relative).parts:
            current = current / part
            require(not current.is_symlink(), 'source symlink forbidden')
    raw = bounded_read(root / LOCK_PATH, 16384)
    lock = json.loads(raw)
    require(isinstance(lock, dict) and set(lock) == {'schema', 'campaign', 'files'},
            'closed source lock')
    require(lock['schema'] == 'runner-storage-source-lock-v1' and
            lock['campaign'] == CAMPAIGN and isinstance(lock['files'], dict) and
            set(lock['files']) == set(SOURCE_FILES), 'exact source set')
    for relative in SOURCE_FILES:
        digest = lock['files'][relative]
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                'public source digest')
        require(hashlib.sha256(bounded_read(root / relative, 1024 * 1024)).hexdigest() == digest,
                'source freeze')
    return hashlib.sha256(raw).hexdigest()


def nonnegative(value):
    require(type(value) is int and value >= 0, 'nonnegative aggregate')
    return value


def measure_one(label, path, stat_fn, statvfs_fn, disk_usage_fn):
    """Exactly one fixed-target stat and two filesystem-capacity snapshots."""
    require(label in ('root', 'workspace', 'docker_root'), 'fixed target label')
    try:
        stat = stat_fn(path)
        vfs = statvfs_fn(path)
        usage = disk_usage_fn(path)
        device = nonnegative(stat.st_dev)
        fragment_bytes = nonnegative(vfs.f_frsize)
        block_bytes = nonnegative(vfs.f_bsize)
        require(fragment_bytes > 0 and block_bytes > 0, 'positive block sizes')
        blocks = nonnegative(vfs.f_blocks)
        free_blocks = nonnegative(vfs.f_bfree)
        available_blocks = nonnegative(vfs.f_bavail)
        inodes = nonnegative(vfs.f_files)
        free_inodes = nonnegative(vfs.f_ffree)
        available_inodes = nonnegative(vfs.f_favail)
        require(available_blocks <= free_blocks <= blocks and
                available_inodes <= free_inodes <= inodes, 'filesystem counter bounds')
        total = nonnegative(usage.total)
        used = nonnegative(usage.used)
        free = nonnegative(usage.free)
        require(used + free <= total, 'disk usage bounds')
        conservative_available = min(free, available_blocks * fragment_bytes)
        return dict(label=label, status='measured', device_id=device,
                    device_major=nonnegative(os.major(device)),
                    device_minor=nonnegative(os.minor(device)),
                    disk_usage=dict(total_bytes=total, used_bytes=used,
                                    available_bytes=free),
                    statvfs=dict(fragment_bytes=fragment_bytes, block_bytes=block_bytes,
                                 total_bytes=blocks * fragment_bytes,
                                 free_bytes=free_blocks * fragment_bytes,
                                 available_bytes=available_blocks * fragment_bytes,
                                 total_inodes=inodes, free_inodes=free_inodes,
                                 available_inodes=available_inodes),
                    conservative_available_bytes=conservative_available,
                    meets_unchanged_prepare_reserve=(conservative_available >= RESERVE_BYTES))
    except OSError as error:
        number = error.errno if type(error.errno) is int and error.errno >= 0 else None
        return dict(label=label, status='unavailable', error_kind='os_error', errno=number)
    except (ValueError, OverflowError, AttributeError, TypeError):
        return dict(label=label, status='unavailable', error_kind='invalid_aggregate', errno=None)


def collect(root, identity, lock_sha, stat_fn, statvfs_fn, disk_usage_fn, observed_at):
    targets = (('root', '/'), ('workspace', str(root)), ('docker_root', '/var/lib/docker'))
    rows = [measure_one(label, path, stat_fn, statvfs_fn, disk_usage_fn)
            for label, path in targets]
    relations = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = rows[left], rows[right]
        known = a['status'] == b['status'] == 'measured'
        relations.append(dict(left=a['label'], right=b['label'],
                              same_device=(a['device_id'] == b['device_id']) if known else None))
    return dict(schema='runner-storage-aggregate-v1', campaign=CAMPAIGN,
                status='measured_all' if all(row['status'] == 'measured' for row in rows)
                else 'measured_partial',
                identity=identity, source_lock_sha256=lock_sha, observed_at_utc=observed_at,
                measurement_scope='fresh_runner_after_checkout_and_source_tests',
                previous_prepare_run_id=41822443746,
                previous_prepare_source_prefix='2784c3a',
                previous_prepare_disk_bytes_observed=False,
                measurement_is_previous_prepare_runner=False,
                authorizes_model_execution=False,
                prepare_reserve_bytes=RESERVE_BYTES, prepare_reserve_changed=False,
                reserve_comparison_uses='minimum_of_sequential_available_byte_snapshots',
                filesystem_snapshots_are_sequential=True,
                device_equality_is_not_mount_identity=True,
                targets=rows, shared_devices=relations,
                filesystem_target_count=3, recursive_scans=0, docker_commands=0,
                model_calls=0, model_downloads=0, package_installs=0,
                cleanup_actions=0, experiment_retries=0)


def run(env):
    event = json.loads(bounded_read(env['GITHUB_EVENT_PATH'], 1024 * 1024))
    identity = admission(env, event, ROOT)
    lock_sha = verify_sources(ROOT)
    output = ROOT / OUTPUT
    output.mkdir(exist_ok=False)  # Exclusive marker; no cleanup or repeated collection.
    observed_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    report = collect(ROOT, identity, lock_sha, os.stat, os.statvfs, shutil.disk_usage, observed_at)
    with (output / 'status.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['status'] == 'measured_all' else 2


def main():
    try:
        return run(os.environ)
    except Exception:
        # Exception text can contain event data, environment values or paths.
        print('Runner storage measurement stopped; no raw diagnostic exported.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
