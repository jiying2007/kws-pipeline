#!/usr/bin/env python3
"""Build, independently verify, then atomically publish one public package.

This reconstructed exporter does not run models, create campaign claims, or
publish remotely. It copies only bytes admitted by the strict evidence validator.
"""
from __future__ import annotations
import argparse
import ctypes
import errno
import os
from pathlib import Path
import shutil
import tempfile
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from execution_common import canonical, document, require, sha, sync_dir
from blind_pcm import read_regular, unaliased
from public_evidence import (MAX_FILE_BYTES, MAX_TOTAL_BYTES, collect, safe_name,
                             same, validate_admission, json_bytes)


def _write(root, name, raw):
    safe_name(name)
    require(type(raw) is bytes and len(raw) <= MAX_FILE_BYTES, 'public per-file cap')
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(fd)


def atomic_publish(source, destination):
    """Linux renameat2(RENAME_NOREPLACE); never emulate using racy exists+rename."""
    source, destination = Path(source), Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    require(rename is not None, 'atomic no-replacement rename is unavailable')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, 'public artifact already exists', str(destination))
        raise OSError(error, 'atomic no-replacement publish failed', str(destination))
    sync_dir(destination.parent)


def _stage_projection(stage):
    return dict(verified=stage['verified'], status=stage['status'],
                attempts=stage['attempts'], reserved_no_retry=True,
                receipts=stage['receipts'])


def export(plan, work, destination, admission=None):
    from verify_public import verify
    work = unaliased(work)
    require(work.is_dir(), 'durable work directory required')
    claim = validate_admission(document(work / 'campaign.claim.json'))
    if admission is not None:
        same(claim, validate_admission(admission), 'current admission mismatch')
    destination = Path(destination).absolute()
    unaliased(destination.parent)
    require(destination.parent.is_dir(), 'public parent directory required')
    require(destination != work and work not in destination.parents and destination not in work.parents,
            'public output must be separate from durable work')
    # Existing complete artifacts are read-only. A damaged one must never be
    # replaced by finalization, and a foreign admission may not be reused.
    if os.path.lexists(destination):
        return verify(destination, expected_admission=claim)
    plan_raw = read_regular(HERE / 'plan.json', MAX_FILE_BYTES)
    same(json_bytes(plan_raw), plan, 'export plan matches reviewed local source')
    same(claim['source_files'].get('research/kw1-preset-supplement-v1/plan.json'), sha(plan_raw),
         'admission reviewed plan source hash')
    host, stages, resources, result, attempts = collect(plan, work)
    files = {'plan.json': plan_raw, 'admission.json': canonical(claim),
             'report.json': canonical(result), 'ledger.json': canonical(attempts)}
    index = dict(schema='kw1-public-evidence-index-v1',
                 host=dict(verified=host['verified'], receipts=host['receipts']),
                 resources=dict(verified=resources['verified'], receipts=resources['receipts']),
                 stages={name: _stage_projection(stage) for name, stage in stages.items()})
    files['evidence-index.json'] = canonical(index)
    if host['verified']:
        for name, raw in host['files'].items():
            files['evidence/' + name] = raw
    if resources['verified']:
        files.update({'evidence/' + name: raw for name, raw in resources['files'].items()})
    for key, stage in stages.items():
        if stage['verified']:
            for name, raw in stage['files'].items():
                files['evidence/' + key + '/output/' + name] = raw
    total = sum(len(raw) for raw in files.values())
    require(total <= MAX_TOTAL_BYTES and all(len(raw) <= MAX_FILE_BYTES for raw in files.values()),
            'public artifact byte caps')
    manifest = dict(schema='kw1-public-package-v1', admission_sha256=sha(files['admission.json']),
                    plan_sha256=sha(files['plan.json']), payload_bytes=total,
                    files={name: dict(sha256=sha(raw), bytes=len(raw)) for name, raw in sorted(files.items())})
    files['manifest.json'] = canonical(manifest)
    require(sum(map(len, files.values())) <= MAX_TOTAL_BYTES and len(files['manifest.json']) <= MAX_FILE_BYTES,
            'public package including manifest exceeds cap')
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.admission-' + claim['config_sha256'][:12] + '-',
                                     dir=destination.parent))
    os.chmod(temporary, 0o700)
    try:
        for name, raw in files.items():
            _write(temporary, name, raw)
        # Explicit empty stage directories preserve verified NOT_RUN without
        # manufacturing any stage or model claim.
        for key, stage in stages.items():
            if stage['verified']:
                (temporary / 'evidence' / key / 'output').mkdir(parents=True, exist_ok=True, mode=0o700)
        (temporary / 'evidence').mkdir(exist_ok=True, mode=0o700)
        for path in sorted((p for p in temporary.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            sync_dir(path)
        sync_dir(temporary)
        verify(temporary, expected_admission=claim)
        try:
            atomic_publish(temporary, destination)
        except FileExistsError:
            # A concurrent publisher must be independently verified, never
            # overwritten or trusted because its name matches.
            return verify(destination, expected_admission=claim)
        return verify(destination, expected_admission=claim)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(canonical(export(document(args.plan), args.work, args.output)).decode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
