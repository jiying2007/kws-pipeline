"""Bounded pipe capture for an already-started trusted synthetic process.

No launcher, authorization issuer, CPU/RSS guarantee, or descendant containment.
Caller owns launching and process isolation; this helper is not a training guard.
"""
import os
from pathlib import Path
import selectors
import subprocess
import time

from .guard import require_unoptimized, sha, sync_directory, write_new


def capture(process, destination, *, timeout_seconds=1.0, max_output_bytes=65536):
    require_unoptimized()
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 60:
        raise ValueError('timeout must be finite and in (0, 60]')
    if type(max_output_bytes) is not int or not 0 < max_output_bytes <= 16 * 1024 * 1024:
        raise ValueError('invalid output cap')
    dest = Path(destination)
    dest.mkdir(exist_ok=False)
    write_new(dest / 'STARTED.json', {'complete': False, 'qualified': False})
    sync_directory(dest)
    sync_directory(dest.parent)
    total = 0
    reason = None
    truncated = False
    complete = False
    selector = selectors.DefaultSelector()
    streams = {}
    deadline = time.monotonic() + timeout_seconds
    try:
        for name in ('stdout', 'stderr'):
            pipe = getattr(process, name)
            if pipe is None:
                raise ValueError('both process output pipes are required')
            os.set_blocking(pipe.fileno(), False)
            stream = (dest / (name + '.bin')).open('xb')
            streams[name] = stream
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                reason = 'timeout'
                break
            for key, _ in selector.select(min(0.02, remaining_seconds)):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining = max_output_bytes - total
                retained = chunk[:remaining]
                streams[key.data].write(retained)
                total += len(retained)
                if len(chunk) > remaining:
                    truncated = True
                    reason = 'output_limit'
                    break
            if reason:
                break
        complete = not selector.get_map() and process.poll() is not None and reason is None
        # EOF and exit may arrive during the final select/read iteration.
        # Check the same absolute deadline before granting complete capture.
        if reason is None and time.monotonic() >= deadline:
            reason = 'timeout'
            complete = False
        if complete and process.returncode != 0:
            reason = 'exit_failure'
    except BaseException as exc:
        reason = 'capture_error:' + type(exc).__name__
        raise
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            reason = 'cleanup_timeout'
            complete = False
        selector.close()
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        for name in ('stdout', 'stderr'):
            pipe = getattr(process, name)
            if pipe is not None:
                pipe.close()
        result = {
            'complete': complete, 'qualified': False,
            'capture_ok': complete and reason is None,
            'reason': reason, 'returncode': process.returncode,
            'truncated': truncated, 'retained_bytes': total,
            'stdout_sha256': sha(dest / 'stdout.bin') if 'stdout' in streams else None,
            'stderr_sha256': sha(dest / 'stderr.bin') if 'stderr' in streams else None,
            'qualification': 'raw capture only; never numerical admission',
        }
        write_new(dest / 'TERMINAL.json', result)
        sync_directory(dest)
    return result
