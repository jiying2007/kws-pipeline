#!/usr/bin/env python3
"""Bound APT setup on the ephemeral Linux ARM CI runner; no build/result claims.

Run this helper under sudo, rather than spawning sudo children: the supervisor
must retain permission to signal every APT/dpkg process in its process group.
Only index refresh and download-only acquisition precede ONE no-download install.
An interrupted/failed install is terminal; never retry a possibly partial dpkg.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PACKAGES = ('gcc-arm-linux-gnueabihf', 'g++-arm-linux-gnueabihf', 'qemu-user-static')
APT_OPTIONS = (
    '-o', 'Acquire::Retries=0',
    '-o', 'Acquire::http::Timeout=20',
    '-o', 'Acquire::https::Timeout=20',
    '-o', 'APT::Get::Fix-Missing=false',
    '-o', 'APT::Get::AllowUnauthenticated=false',
)


@dataclass(frozen=True)
class Budget:
    # All durations are wall-clock seconds, INCLUDING TERM/KILL cleanup time.
    index: float = 60
    download: float = 240
    attempt: float = 120
    install: float = 120
    total: float = 440
    term: float = 2
    kill: float = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if min(self.index, self.download, self.attempt, self.install, self.total) <= self.term + self.kill:
            raise ValueError('phase budgets must reserve TERM and KILL cleanup')


def live_group_members(pgid):
    """Linux-only: zombies have exited; their adopter, not us, owns reaping.

    killpg(..., 0) alone counts orphan zombies as running forever. Inspect only
    /proc process metadata, and fail closed on unreadable metadata. APT children
    inherit the new session/group; this is not a sandbox for hostile programs
    deliberately escaping that group.
    """
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
            members.append(int(path.name))
    return sorted(members)


def exit_code(code):
    return 128 - code if code < 0 else code


def run_phase(command, phase, deadline, budget, cancelled, log):
    """Return (exit code, fatal cleanup/spawn/orphan failure). No shell involved."""
    started = time.monotonic()
    run_until = deadline - budget.term - budget.kill
    if started >= run_until or cancelled[0]:
        code = 128 + cancelled[0] if cancelled[0] else 124
        log(f'event=phase_end phase={phase} elapsed=0.000 exit_code={code} reason=no_budget_or_cancelled')
        return code, True
    if not log(f'event=phase_start phase={phase} budget_seconds={deadline - started:.3f}'):
        return 125, True
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except OSError as error:
        log(f'event=phase_end phase={phase} elapsed={time.monotonic() - started:.3f} '
            f'exit_code=127 reason=spawn_failed error={error!r}')
        return 127, True

    code, fatal, reason = 125, False, 'supervisor_error'
    try:
        while True:
            child_code = process.poll()
            if cancelled[0]:
                code, reason = 128 + cancelled[0], 'interrupted'
                break
            if child_code is not None:
                if live_group_members(process.pid):
                    code, fatal, reason = 125, True, 'orphaned_children'
                    break
                code, reason = exit_code(child_code), 'exited'
                break
            if time.monotonic() >= run_until:
                code, reason = 124, 'wall_clock_timeout'
                break
            time.sleep(min(0.02, max(0, run_until - time.monotonic())))
    except (OSError, ValueError, IndexError) as error:
        log(f'event=supervisor_error phase={phase} error={error!r}')
        fatal = True

    # Always inspect/clean the group, including when its original parent exited
    # with status zero. Never start a retry/install while old children run.
    inspection_error = None

    def remaining_members():
        nonlocal inspection_error
        process.poll()
        try:
            return live_group_members(process.pid)
        except (OSError, ValueError, IndexError) as error:
            # Failure to inspect is NOT evidence of exit. Still send TERM/KILL
            # to the known group, then fail closed rather than abandoning it.
            inspection_error = error
            return [process.pid]

    try:
        cleanup_deadline = min(deadline, time.monotonic() + budget.term + budget.kill)
        for sig, until in (
            (signal.SIGTERM, min(time.monotonic() + budget.term, cleanup_deadline - budget.kill)),
            (signal.SIGKILL, cleanup_deadline),
        ):
            if not remaining_members() and process.returncode is not None:
                break
            log(f'event=cleanup phase={phase} signal={sig.name} pgid={process.pid}')
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            while time.monotonic() < until:
                if not remaining_members() and process.returncode is not None:
                    break
                time.sleep(min(0.02, max(0, until - time.monotonic())))
        members = remaining_members()
        if inspection_error is not None:
            raise RuntimeError(f'cannot verify process group exit: {inspection_error!r}')
        log(f'event=child_exit phase={phase} child_exit_code={process.returncode} live_pids={members}')
        if process.returncode is None or members:
            raise RuntimeError('process group did not exit within cleanup budget')
    except (OSError, RuntimeError, ValueError, IndexError) as error:
        code, fatal, reason = 125, True, 'cleanup_failed'
        log(f'event=cleanup_failed phase={phase} error={error!r}')

    if cancelled[0] and not fatal:
        code, reason = 128 + cancelled[0], 'interrupted'
    log(f'event=phase_end phase={phase} elapsed={time.monotonic() - started:.3f} '
        f'exit_code={code} reason={reason}')
    return code, fatal


def install(*, apt=('/usr/bin/apt-get',), budget=None, output=None):
    """The injectable command/budget are for offline tests; CLI has no overrides."""
    budget = budget or Budget()
    output = output or sys.stdout
    started = time.monotonic()
    deadline = started + budget.total
    cancelled = [0]
    log_failed = [False]

    def log(message):
        try:
            print('ARM_APT ' + message, file=output, flush=True)
        except OSError:
            # A broken log pipe must not interrupt TERM/KILL cleanup or allow
            # more phases to run without their required exit evidence.
            log_failed[0] = True
        return not log_failed[0]

    def on_signal(signum, _frame):
        # Do not raise during cleanup or allow a second TERM to abandon children.
        cancelled[0] = signum

    handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    common = list(apt) + list(APT_OPTIONS)

    def phase(name, arguments, phase_deadline):
        result = run_phase(common + arguments, name, min(deadline, phase_deadline),
                           budget, cancelled, log)
        return (125, True) if log_failed[0] else result

    code = 125
    try:
        code, _ = phase('index', ['-o', 'APT::Update::Error-Mode=any', 'update'],
                        time.monotonic() + budget.index)
        if code == 0:
            download_deadline = min(deadline, time.monotonic() + budget.download)
            for attempt in (1, 2):
                code, fatal = phase(f'download-{attempt}',
                                    ['--download-only', 'install', '-y', *PACKAGES],
                                    min(download_deadline, time.monotonic() + budget.attempt))
                if code == 0 or fatal or cancelled[0]:
                    break
        if code == 0 and not cancelled[0]:
            # Even a timeout/TERM during this single attempt is terminal. No
            # dpkg recovery, no retry, and no fetching missing cached packages.
            code, _ = phase('install', ['--no-download', 'install', '-y', *PACKAGES],
                            time.monotonic() + budget.install)
        if cancelled[0] and code != 125:
            code = 128 + cancelled[0]
        elif code == 0 and time.monotonic() > deadline:
            code = 124
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        log(f'event=complete elapsed={time.monotonic() - started:.3f} exit_code={code} '
            f'result={"success" if code == 0 else "failure"}')
    return 125 if log_failed[0] else code


def main():
    if len(sys.argv) != 1 or not sys.platform.startswith('linux') or os.geteuid() != 0:
        print('Run without arguments under sudo on the ephemeral Linux CI runner.', file=sys.stderr)
        return 126
    return install()


if __name__ == '__main__':
    raise SystemExit(main())
