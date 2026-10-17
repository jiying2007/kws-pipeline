#!/usr/bin/env python3
"""Bounded, service-independent experiment handoff. Never asserts durability."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

VERSION = 'experiment-handoff-v1'
PREFIX = 'KWS_HANDOFF_V1 '
MAX_FRAME = 1024 * 1024
MAX_INPUT = 32 * 1024 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def safe_name(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or len(name) > 256 or not name or '\\' in name or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('unsafe member name')
    if name.split('/')[0] in ('handoff-manifest.json', 'handoff.zip'):
        raise ValueError('reserved member name')
    return name


def read_regular(root, name, limit):
    # Walk descriptors with O_NOFOLLOW: also rejects symlink ancestors and FIFOs.
    parts = safe_name(name).split('/')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = new
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            before = os.fstat(leaf)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('not_regular')
            if before.st_size > limit:
                raise ValueError('input_size_limit')
            with os.fdopen(os.dup(leaf), 'rb') as source:
                raw = source.read(limit + 1)
            after = os.fstat(leaf)
            current = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError('path_replaced')
            if len(raw) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('changed_or_oversize')
            return raw
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


def zip_bytes(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    return out.getvalue()


def execute(command, output, files, max_bytes=262144):
    """One wrapper process owns spawn, terminal observation, collection and framing."""
    if not 2048 <= max_bytes <= 700000 or len(files) > 128 or not files or len(set(files)) != len(files):
        raise ValueError('invalid limits or allowlist')
    for name in files:
        safe_name(name)
        if any(other.startswith(name + '/') for other in files):
            raise ValueError('file_directory_collision')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('output must be fresh; stale results are forbidden')
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    terminal = {'returncode': None, 'signal': None, 'launch_error': None}
    streams = {}
    stream_incomplete = False
    proc = None
    def drain(stream, key, sink=streams):
        digest, size = hashlib.sha256(), 0
        while True:
            raw = stream.read(65536)
            if not raw:
                break
            digest.update(raw)
            size += len(raw)
        sink[key] = {'bytes': size, 'sha256': digest.hexdigest(), 'body_retained': False}
        stream.close()
    try:
        proc = subprocess.Popen([x.replace('{output}', str(output)) for x in command],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        terminal['launch_error'] = type(exc).__name__
    if proc is not None:
        threads = [threading.Thread(target=drain, args=(stream, key), daemon=True)
                   for stream, key in ((proc.stdout, 'stdout'), (proc.stderr, 'stderr'))]
        for thread in threads:
            thread.start()
        interrupted = []
        def forward(sig, frame):
            interrupted.append(sig)
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        old = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            terminal['returncode'] = proc.wait()
            deadline = time.monotonic() + 2.0
            for thread in threads:
                thread.join(max(0., deadline - time.monotonic()))
            stream_incomplete = any(thread.is_alive() for thread in threads)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        terminal['wrapper_signals'] = interrupted
        terminal['signal'] = -proc.returncode if proc.returncode < 0 else None
    streams = dict(streams)
    contents, entries, failures, total = {}, {}, {}, 0
    if stream_incomplete:
        failures['process_streams'] = 'incomplete_after_child_exit'
    try:
        root_identity = output.stat(follow_symlinks=False)
    except OSError:
        root_identity = None
    for name in files:
        try:
            raw = read_regular(output, name, MAX_INPUT - total)
            total += len(raw)
            entries[name] = {'bytes': len(raw), 'sha256': sha(raw)}
            contents[name] = raw
        except (OSError, ValueError) as exc:
            failures[name] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    if root_identity is not None:
        try:
            final_root = output.stat(follow_symlinks=False)
            if (root_identity.st_dev, root_identity.st_ino) != (final_root.st_dev, final_root.st_ino):
                failures['output'] = 'root_replaced'
        except OSError:
            failures['output'] = 'root_disappeared'
    summary = {}
    if 'run.json' in contents:
        try:
            run = json.loads(contents['run.json'])
            # Keep original run.json in package; display only bounded fields, not paths/errors.
            for key in ('status', 'trace_status', 'optimizer_steps', 'paired_steps', 'qualification_allowed', 'winner'):
                if key in run:
                    summary[key] = run[key]
            if len(encoded(summary)) > 4096:
                raise ValueError('summary_size')
            for group in ('checkpoints', 'traces', 'artifacts'):
                for name, digest in run.get(group, {}).items():
                    if name not in entries or entries[name]['sha256'] != digest:
                        failures['run_binding:' + name] = 'missing_or_mismatched'
        except (ValueError, TypeError, AttributeError):
            failures['run.json'] = 'invalid_report'
            summary = {}
    manifest = {'schema': VERSION, 'run_id': run_id, 'terminal': terminal,
                'streams': streams, 'summary': summary, 'files': entries,
                'failures': failures, 'delivery_status': 'unconfirmed',
                'persistence_claim': False, 'recovery_scope': 'all_selected_files',
                'allowed_files': files}
    archive = zip_bytes(dict(contents, **{'handoff-manifest.json': encoded(manifest)}))
    if len(archive) > max_bytes:
        manifest['failures']['package'] = 'package_size_limit'
        manifest['recovery_scope'] = 'manifest_only'
        archive = zip_bytes({'handoff-manifest.json': encoded(manifest)})
        if len(archive) > max_bytes:
            inventory_sha256 = sha(encoded(manifest))
            manifest = {'schema': VERSION, 'run_id': run_id, 'terminal': terminal,
                        'files': {}, 'summary': {}, 'failures': {'package': 'manifest_size_limit'},
                        'inventory_manifest_sha256': inventory_sha256,
                        'delivery_status': 'unconfirmed', 'persistence_claim': False,
                        'recovery_scope': 'manifest_only'}
            archive = zip_bytes({'handoff-manifest.json': encoded(manifest)})
    envelope = {'schema': VERSION, 'run_id': run_id, 'archive_bytes': len(archive),
                'archive_sha256': sha(archive), 'manifest_sha256': sha(encoded(manifest)),
                'manifest': manifest, 'archive_base64': base64.b64encode(archive).decode(),
                'end': 'COMPLETE_UNCONFIRMED'}
    line = PREFIX.encode() + encoded(envelope) + b'\n'
    if len(line) > MAX_FRAME:
        raise ValueError('frame_size_limit; no complete handoff')
    return line, manifest


def recover(frame, destination):
    """Validate entirely before extracting. Local reconstruction is not persistence."""
    if len(frame) > MAX_FRAME or not frame.startswith(PREFIX.encode()) or not frame.endswith(b'\n'):
        raise ValueError('missing_or_oversize_frame')
    value = json.loads(frame[len(PREFIX):])
    if value['schema'] != VERSION or value['end'] != 'COMPLETE_UNCONFIRMED':
        raise ValueError('incomplete_frame')
    raw = base64.b64decode(value['archive_base64'], validate=True)
    if len(raw) != value['archive_bytes'] or sha(raw) != value['archive_sha256']:
        raise ValueError('archive_integrity')
    manifest = value['manifest']
    if sha(encoded(manifest)) != value['manifest_sha256'] or manifest['run_id'] != value['run_id']:
        raise ValueError('manifest_integrity')
    if manifest['delivery_status'] != 'unconfirmed' or manifest['persistence_claim'] is not False:
        raise ValueError('invalid_persistence_claim')
    allowed = manifest.get('allowed_files', [])
    if not isinstance(allowed, list) or len(allowed) > 128 or len(set(allowed)) != len(allowed) or not set(manifest['files']).issubset(allowed):
        raise ValueError('allowlist_mismatch')
    for name in allowed:
        safe_name(name)
        if any(other.startswith(name + '/') for other in allowed):
            raise ValueError('file_directory_collision')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        expected = {'handoff-manifest.json'}
        if manifest['recovery_scope'] == 'all_selected_files':
            expected.update(manifest['files'])
        elif manifest['recovery_scope'] != 'manifest_only':
            raise ValueError('invalid_scope')
        if len(names) != len(set(names)) or set(names) != expected or sum(i.file_size for i in z.infolist()) > MAX_INPUT + 65536:
            raise ValueError('archive_members')
        if z.read('handoff-manifest.json') != encoded(manifest):
            raise ValueError('manifest_copy')
        contents = {}
        for name in names:
            if name != 'handoff-manifest.json':
                safe_name(name)
                member = z.read(name)
                if len(member) != manifest['files'][name]['bytes'] or sha(member) != manifest['files'][name]['sha256']:
                    raise ValueError('member_integrity')
                contents[name] = member
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    contents['handoff-manifest.json'] = encoded(manifest)
    contents['handoff.zip'] = raw
    root_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name, member in contents.items():
            parts = name.split('/')
            fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = new
                leaf = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
                with os.fdopen(leaf, 'wb') as target:
                    target.write(member)
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)
    return {'status': 'verified_local_only', 'archive_sha256': sha(raw),
            'archive_bytes': len(raw), 'run_id': manifest['run_id'],
            'complete_selected_recovery': not manifest['failures'] and manifest['recovery_scope'] == 'all_selected_files',
            'terminal': manifest['terminal'], 'delivery_status': 'unconfirmed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', required=True)
    run.add_argument('--file', action='append', required=True)
    run.add_argument('--max-bytes', type=int, default=262144)
    run.add_argument('command', nargs=argparse.REMAINDER)
    restore = sub.add_parser('recover')
    restore.add_argument('--frame', required=True)
    restore.add_argument('--destination', required=True)
    args = p.parse_args()
    if args.mode == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            p.error('command required')
        frame, manifest = execute(command, args.output, args.file, args.max_bytes)
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
        # Packaging status never replaces the true child status in the manifest.
        semantic_failure = any(manifest['summary'].get(key) == 'failed' for key in ('status', 'trace_status'))
        return 0 if manifest['terminal']['returncode'] == 0 and not manifest['failures'] and not semantic_failure else 1
    with open(args.frame, 'rb') as source:
        frame = source.read(MAX_FRAME + 1)
    print(json.dumps(recover(frame, args.destination), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
