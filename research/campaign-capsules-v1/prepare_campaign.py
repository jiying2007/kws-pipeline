"""Prepare/check a fresh offline, closed campaign bundle; never alter live source."""
import argparse
import os
from pathlib import Path
import stat
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_capsule as c

CAP = 1024 * 1024


def directory_fd(path):
    c.require('..' not in Path(path).parts, 'no parent traversal')
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:]:
            new = os.open(part, flags, dir_fd=fd)
            os.close(fd); fd = new
        return fd
    except BaseException:
        os.close(fd)
        raise


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_local(path):
    path = Path(path)
    parent = directory_fd(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(fd)
        c.require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= CAP,
                  'bounded unaliased ordinary JSON')
        chunks, left = [], CAP + 1
        while left:
            body = os.read(fd, min(left, 65536))
            if not body:
                break
            chunks.append(body); left -= len(body)
        raw = b''.join(chunks)
        c.require(len(raw) == before.st_size and _stamp(os.fstat(fd)) == _stamp(before) and
                  _stamp(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == _stamp(before),
                  'stable complete local document')
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def load_catalog():
    index = c.validate_index(c.decode(read_local(HERE / 'index.json')))
    policies = {row['policy_id']: c.validate_policy(read_local(HERE / row['path']), row)
                for row in index['policies']}
    return index, policies


def projected_bundle(draft, index, policy, *, campaign):
    c.validate_draft(draft, index, expected_campaign=campaign)
    controls = c.disabled_controls(draft, index, policy, expected_campaign=campaign)
    files = {'capsule.json': c.canonical(draft), **{name: c.canonical(value) for name, value in controls.items()}}
    manifest = {'schema': 'k1-asr18-closed-preparation-bundle-v1', 'status': 'draft_closed',
                'execution_authority': 'absent', 'scientific_sha256': draft['scientific_sha256'],
                'files': {name: c.identity(body) for name, body in sorted(files.items())}}
    files['derived-manifest.json'] = c.canonical(manifest)
    return files


def create_bundle(output, campaign, policy_id):
    index, policies = load_catalog()
    draft = c.new_draft(campaign, index, policy_id=policy_id)
    bodies = projected_bundle(draft, index, policies[policy_id], campaign=campaign)
    output = Path(output)
    c.require(output.name not in ('', '.', '..'), 'fresh output directory required')
    parent = directory_fd(output.parent)
    child = None
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent)
        child = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        os.fsync(parent)
        for name, body in bodies.items():
            fd = os.open(name, os.O_WRONLY | os.O_EXCL | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=child)
            try:
                offset = 0
                while offset < len(body):
                    count = os.write(fd, body[offset:]); c.require(count > 0, 'complete closed projection write')
                    offset += count
                os.fsync(fd)
            finally:
                os.close(fd)
        os.fsync(child)
    finally:
        if child is not None:
            os.close(child)
        os.close(parent)
    return check_bundle(output, campaign)


def check_bundle(bundle, campaign):
    index, policies = load_catalog()
    bundle = Path(bundle)
    draft = c.decode(read_local(bundle / 'capsule.json'))
    c.validate_draft(draft, index, expected_campaign=campaign)
    expected = projected_bundle(draft, index, policies[draft['policy_id']], campaign=campaign)
    fd = directory_fd(bundle)
    try:
        c.require(set(os.listdir(fd)) == set(expected), 'exact closed preparation member roster')
    finally:
        os.close(fd)
    for name, raw in expected.items():
        c.require(read_local(bundle / name) == raw, 'closed derived member drift: ' + name)
    return {'schema': 'k1-asr18-closed-preparation-check-v1', 'campaign': campaign,
            'status': 'closed_projection_verified', 'execution_authority': 'absent',
            'files': len(expected), 'scientific_sha256': draft['scientific_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create', help='create a fresh offline bundle; no current-source updates')
    create.add_argument('--campaign', required=True)
    create.add_argument('--policy', default='k1-asr18-scientific-v1')
    create.add_argument('--output', required=True, type=Path)
    check = commands.add_parser('check', help='read-only byte comparison to the closed deterministic projection')
    check.add_argument('--campaign', required=True)
    check.add_argument('--bundle', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'create':
        result = create_bundle(args.output, args.campaign, args.policy)
    else:
        result = check_bundle(args.bundle, args.campaign)
    print(c.canonical(result).decode())
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print('closed preparation rejected: ' + str(error), file=sys.stderr)
        raise SystemExit(2) from None
