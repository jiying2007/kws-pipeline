"""Bounded, no-follow output accounting under concurrent atomic replacement.

A completed observation enforces the caller's unchanged byte limit. This is a
polling guard, not a kernel quota or a frozen filesystem snapshot. A vanishing
entry invalidates the WHOLE observation; it is never counted as zero.
"""
from pathlib import Path
import errno
import os
import stat
import time

LIMIT=64*1024**2
MAX_ATTEMPTS=5
MAX_ENTRIES=4096
MAX_DEPTH=32
MAX_SECONDS=0.25

class OutputBoundError(ValueError):pass
class UnstableOutputError(ValueError):pass
class _Changed(Exception):pass


def _fingerprint(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_nlink,
            info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _check_clock(deadline):
    if time.monotonic()>=deadline:raise UnstableOutputError('output observation deadline exceeded')


def _open(name,parent,directory=False):
    flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
    if directory:flags|=os.O_DIRECTORY
    return os.open(name,flags,dir_fd=parent)


def _scan(root_fd,limit,deadline):
    directories=[];files=[];handles=[];seen=set();total=0;entries=0
    root_device=os.fstat(root_fd).st_dev
    def visit(fd,depth):
        nonlocal entries,total
        _check_clock(deadline)
        if depth>MAX_DEPTH:raise OutputBoundError('output depth bound exceeded')
        before=os.fstat(fd)
        if not stat.S_ISDIR(before.st_mode) or before.st_dev!=root_device:
            raise OutputBoundError('output directory boundary violated')
        key=(before.st_dev,before.st_ino)
        if key in seen:raise _Changed()
        seen.add(key);directories.append((fd,_fingerprint(before)))
        with os.scandir(fd) as iterator:
            for entry in iterator:
                name=entry.name
                _check_clock(deadline);entries+=1
                if entries>MAX_ENTRIES:raise OutputBoundError('output entry bound exceeded')
                if name in ('.','..') or '/' in name:raise OutputBoundError('invalid output entry')
                info=os.stat(name,dir_fd=fd,follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):raise OutputBoundError('output symlink prohibited')
                directory=stat.S_ISDIR(info.st_mode)
                if not directory and not stat.S_ISREG(info.st_mode):
                    raise OutputBoundError('output special file prohibited')
                child=_open(name,fd,directory)
                try:
                    actual=os.fstat(child)
                    if _fingerprint(actual)!=_fingerprint(info):raise _Changed()
                except BaseException:
                    os.close(child);raise
                if directory:
                    handles.append(child);visit(child,depth+1)
                else:
                    try:
                        if actual.st_dev!=root_device or actual.st_nlink!=1:
                            raise OutputBoundError('output regular-file boundary violated')
                        key=(actual.st_dev,actual.st_ino)
                        if key in seen:raise _Changed()  # a renamed inode is not double-counted
                        seen.add(key)
                        files.append((fd,name,_fingerprint(actual)))
                        total+=actual.st_size
                        if total>limit:raise OutputBoundError('output disk bound exceeded')
                    finally:os.close(child)
        if _fingerprint(os.fstat(fd))!=_fingerprint(before):raise _Changed()
    try:
        visit(root_fd,0)
        # Reopen every counted entry relative to its anchored directory. Changes
        # invalidate the entire sum, including atomic replacement and growth.
        for parent,name,expected in files:
            _check_clock(deadline)
            fd=_open(name,parent)
            try:
                if _fingerprint(os.fstat(fd))!=expected:raise _Changed()
            finally:os.close(fd)
        for fd,expected in directories:
            _check_clock(deadline)
            if _fingerprint(os.fstat(fd))!=expected:raise _Changed()
        return total
    finally:
        for fd in reversed(handles):os.close(fd)


def observed_output_size(output,limit=LIMIT):
    if type(limit) is not int or limit!=LIMIT:raise ValueError('unchanged 64 MiB output limit required')
    root=Path(output)
    if root.is_symlink() or root.resolve()!=root.absolute():raise OutputBoundError('unaliased output root required')
    deadline=time.monotonic()+MAX_SECONDS
    root_fd=_open(str(root),None,True)
    try:
        for attempt in range(MAX_ATTEMPTS):
            _check_clock(deadline)
            try:
                total=_scan(root_fd,limit,deadline)
                current=os.stat(str(root),follow_symlinks=False)
                actual=os.fstat(root_fd)
                if (current.st_dev,current.st_ino)!=(actual.st_dev,actual.st_ino):raise _Changed()
                _check_clock(deadline)
                return total
            except FileNotFoundError as error:
                if error.errno!=errno.ENOENT:raise
                # A vanished name is evidence of mutation, not permission to
                # skip bytes. Persistent ENOENT fails after bounded full scans.
                pass
            except _Changed:pass
            # EACCES/EIO/ELOOP/ENOTDIR and all other I/O errors propagate at once.
            if attempt+1<MAX_ATTEMPTS:time.sleep(0.005)
        raise UnstableOutputError('output did not stabilize within bounded observations')
    finally:os.close(root_fd)
