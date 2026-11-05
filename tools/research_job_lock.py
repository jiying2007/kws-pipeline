"""One local heavy-model slot, shared by direct and supervised research jobs.

An inherited flock belongs to an open file description. Reopening the same path
in a child would compete with its parent; unlocking the inherited descriptor
would release the parent's lock. Reuse it and let process/descriptor lifetime
retain the slot. This is cooperative local exclusion, not a process sandbox.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import stat

FD_ENV = "KWS_RESEARCH_HEAVY_LOCK_FD"
PATH_ENV = "KWS_RESEARCH_HEAVY_LOCK_PATH"
SUPERVISED_ENV = "KWS_RESEARCH_SUPERVISED_JOB"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_path(path):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            "HEAVY_LOCK_ALIAS")
    require(path.parent.is_dir(), "HEAVY_LOCK_PARENT_REQUIRED")
    return path


def same_file(fd, path):
    opened, named = os.fstat(fd), path.stat()
    require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1 and
            stat.S_ISREG(named.st_mode) and named.st_nlink == 1 and
            (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino),
            "HEAVY_LOCK_DESCRIPTOR_IDENTITY")


@dataclass(frozen=True)
class HeavyModelLease:
    path: Path
    fd: int
    inherited: bool
    supervised: bool = False

    @property
    def pass_fds(self):
        return (self.fd,)

    def child_environment(self, environment=None):
        result = dict(os.environ if environment is None else environment)
        result.update({FD_ENV: str(self.fd), PATH_ENV: str(self.path)})
        return result


@contextmanager
def heavy_model_slot(path, *, environment=None):
    """Acquire or validate/reuse the same explicit local flock, without waiting.

    Pass both ``lease.child_environment()`` and ``lease.pass_fds`` to every
    subprocess which continues the model job. A child that receives stale or
    mismatched inheritance metadata fails; it never silently opens another lock.
    The caller creates the chosen lock directory. Every cooperating entrypoint
    must use the same path; this cannot arbitrate remote machines or other locks.
    """
    path = checked_path(path)
    environment = os.environ if environment is None else environment
    has_fd, has_path = FD_ENV in environment, PATH_ENV in environment
    require(has_fd == has_path, "HEAVY_LOCK_INHERITANCE_INCOMPLETE")
    inherited = has_fd
    flag = environment.get(SUPERVISED_ENV)
    require(flag in (None, "1") and (flag is None or inherited),
            "SUPERVISED_JOB_REQUIRES_INHERITED_SLOT")
    if inherited:
        value = environment[FD_ENV]
        require(type(value) is str and value.isascii() and value.isdecimal() and
                value == str(int(value)) and int(value) >= 3,
                "HEAVY_LOCK_DESCRIPTOR_VALUE")
        fd = int(value)
        require(environment[PATH_ENV] == str(path), "HEAVY_LOCK_PATH_MISMATCH")
    else:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        same_file(fd, path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BlockingIOError("HEAVY_MODEL_SLOT_BUSY") from error
        same_file(fd, path)
        yield HeavyModelLease(path, fd, inherited, flag == "1")
    finally:
        if not inherited:
            # Closing this descriptor releases only our reference. A surviving
            # child with the same open file description still holds the slot.
            os.close(fd)
