"""Reconstructed host-only bounded I/O. No model or network side effects."""
import hashlib
import json
import os
from pathlib import Path
import stat

MAX_BYTES = 8 * 1024**2

def require(ok, message):
    if not ok:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(raw):
    require(type(raw) is bytes, 'hash bytes required')
    return hashlib.sha256(raw).hexdigest()

def read(path, maximum=MAX_BYTES):
    path = Path(path).absolute()
    require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'unaliased path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, 'bounded single-link regular file required')
        with os.fdopen(os.dup(fd), 'rb') as handle:
            raw = handle.read(maximum + 1)
        after = os.fstat(fd)
        entry = path.lstat()
        stamp = lambda x: (x.st_dev, x.st_ino, x.st_mode, x.st_nlink, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        require(stamp(before) == stamp(after) == stamp(entry) and len(raw) == before.st_size, 'file changed during read')
        return raw
    finally:
        os.close(fd)

def document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON')
    return json.loads(read(path), object_pairs_hook=unique, parse_constant=invalid)

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def claim(path, value):
    path = Path(path).absolute()
    require(path.parent.resolve() == path.parent and not any(p.is_symlink() for p in path.parents), 'unaliased claim directory')
    raw = canonical(value)
    require(len(raw) <= MAX_BYTES, 'claim size')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        sync_dir(path.parent)
        with os.fdopen(os.dup(fd), 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(fd)
    sync_dir(path.parent)
    return sha(raw)

def write(path, value):
    # Derived state still has a one-write contract. Recovery uses an explicitly
    # separate replacement path, never this function for authoritative receipts.
    return claim(path, value)

def exact_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(exact_equal(x,y) for x,y in zip(a,b))
    return a == b
