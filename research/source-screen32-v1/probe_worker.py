#!/usr/bin/env python3
"""One stdlib-only setup-container diagnostic; no installation or model work.

The fixed CLI has no arguments. Filesystem parameters exist only for offline
tests; the live kernel verifier is always called and cannot be supplied by a
caller. A complete receipt proves only these observed startup conditions.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
# Isolated Python omits the script directory. Import only the frozen sibling.
sys.path.insert(0, str(HERE))
import runtime_scope

MAX_INTERFACES = 16
MAX_SOURCE_BYTES = 65536
MAX_RECEIPT_BYTES = 8192
RECEIPT = "probe-kernel.json"
TEMPORARY = ".probe-kernel.json.tmp"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_sha256(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and before.st_size <= MAX_SOURCE_BYTES, "bounded unaliased source required")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                           value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(stamp(path.lstat()) == stamp(before) and len(raw) == before.st_size,
            "source changed while reading")
    return hashlib.sha256(raw).hexdigest()


def empty_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with os.scandir(directory) as entries:
            require(next(entries, None) is None, "empty directory required")
        return directory
    except BaseException:
        os.close(directory)
        raise


def write_receipt(output, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    require(len(raw) <= MAX_RECEIPT_BYTES, "oversized probe receipt")
    directory = empty_directory(output)
    created = published = False
    try:
        descriptor = os.open(TEMPORARY, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        created = True
        try:
            require(os.write(descriptor, raw) == len(raw), "incomplete probe receipt")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(TEMPORARY, RECEIPT, src_dir_fd=directory, dst_dir_fd=directory)
        published = True
        os.fsync(directory)
    except BaseException:
        if created:
            os.unlink(RECEIPT if published else TEMPORARY, dir_fd=directory)
        raise
    finally:
        os.close(directory)


def run(*, output=Path("/output"), runtime=Path("/runtime")):
    scope = runtime_scope.verify_runtime_scope(network_required=False)
    interfaces = scope["network_interfaces"]
    require(type(interfaces) is list and 2 <= len(interfaces) <= MAX_INTERFACES
            and all(type(name) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", name)
                    for name in interfaces)
            and interfaces == sorted(set(interfaces)) and "lo" in interfaces
            and scope["network_disabled"] is False,
            "bounded setup network inventory requires lo and non-loopback interface")
    implementation, version = platform.python_implementation(), platform.python_version()
    require((implementation, version) == ("CPython", "3.12.14"),
            "exact CPython 3.12.14 required")
    os.close(empty_directory(runtime))
    value = {"schema": "screen32-probe-kernel-v1", "status": "complete",
             "kernel": scope, "python_version": version,
             "worker_sha256": source_sha256(HERE / "probe_worker.py"),
             "runtime_scope_sha256": source_sha256(HERE / "runtime_scope.py"),
             "model_calls": 0, "runtime_installs": 0,
             "human_gold": False, "training_admitted": False}
    write_receipt(output, value)
    return value


def main():
    require(len(sys.argv) == 1, "probe worker accepts no arguments")
    run()


if __name__ == "__main__":
    main()
