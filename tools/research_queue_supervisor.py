#!/usr/bin/env python3
"""Keep one explicitly declared research queue running; no new scheduler.

The existing queue owns claims, dependencies, outputs, model locks and budgets.
This process only hosts its controller, waits for live owners and writes a
replaceable heartbeat outside the immutable queue. It never retries a job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_queue as queue
from research_job_lock import SUPERVISED_ENV

SCHEMA = "kws-local-research-supervisor-v1"
HEARTBEAT_SCHEMA = "kws-local-research-supervisor-heartbeat-v1"
CONTROLLER = Path(__file__).resolve().with_name("research_queue.py")


def lock_active(path):
    """Observe an existing cooperative lock without creating queue files."""
    path = queue.safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    try:
        info = os.fstat(fd)
        queue.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "LOCK_FILE_IDENTITY")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def waiting_for(root, request, heavy_lock, *, check_slot=False):
    # The queue repeats these checks atomically when it actually dispatches.
    # A lost race returns 75; no lease or claim is created by this observation.
    if lock_active(root / "controller.lock"):
        return "waiting_for_controller"
    if any(lock_active(root / "jobs" / job["job_id"] / "job.lock") for job in request["jobs"]):
        return "waiting_for_worker"
    if check_slot and lock_active(heavy_lock):
        return "waiting_for_heavy_model_slot"
    return None


def replace_heartbeat(path, value):
    """Only this mutable liveness observation is replaced, never a receipt."""
    path = queue.safe_path(path)
    temporary = path.with_name(".heartbeat-" + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(queue.canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        queue.sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(report):
    queue.require(type(report) is dict and type(report.get("status")) is str, "CONTROLLER_REPORT_REQUIRED")
    result = {"status": report["status"]}
    for name in ("queue_id", "actual_known_job_nanoseconds", "actual_known_job_minutes",
                 "actual_total_is_complete", "remaining_budget_nanoseconds",
                 "global_budget_minutes", "budget_overrun_nanoseconds"):
        if name in report:
            result[name] = report[name]
    if "jobs" in report:
        result["jobs"] = {name: state["status"] for name, state in report["jobs"].items()}
        result["job_counts"] = dict(Counter(result["jobs"].values()))
    return result


def stop_controller(process):
    """Stop this controller's future dispatch, preserving its bounded worker."""
    if process is None or process.poll() is not None:
        return
    # Deliberately address only the controller. Its worker and command have
    # their own process groups, own the inherited leases and enforce deadlines.
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def supervise(root, state_dir, *, poll_seconds=1.0, heartbeat_seconds=5.0):
    queue.require(SUPERVISED_ENV not in os.environ, "NESTED_SUPERVISOR_NOT_SUPPORTED")
    for value, low, high in ((poll_seconds, .02, 30), (heartbeat_seconds, .05, 60)):
        queue.require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                      "BOUNDED_SUPERVISOR_INTERVAL_REQUIRED")
    root, metadata, request, queue_sha = queue.load_queue(root)
    state_dir = queue.safe_path(state_dir)
    queue.require(not state_dir.is_relative_to(root) and not root.is_relative_to(state_dir),
                  "SUPERVISOR_STATE_MUST_BE_SEPARATE_FROM_QUEUE")
    state_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    binding = {"schema": SCHEMA, "queue_root": str(root), "queue_sha256": queue_sha,
               "controller_source_sha256": metadata["source_sha256"],
               "supervisor_source_sha256": queue.sha(queue.read_bytes(Path(__file__).resolve()))}
    with queue.exclusive(state_dir / "supervisor.lock"):
        binding_path = state_dir / "binding.json"
        if binding_path.exists():
            queue.require(queue.loads(queue.read_bytes(binding_path)) == binding, "SUPERVISOR_BINDING_CHANGED")
        else:
            queue.require({p.name for p in state_dir.iterdir()} == {"supervisor.lock"},
                          "NEW_SUPERVISOR_STATE_REQUIRED")
            queue.write_json(binding_path, binding)
        sessions = queue.safe_path(state_dir / "sessions")
        sessions.mkdir(mode=0o700, exist_ok=True)
        session = sessions / uuid.uuid4().hex
        session.mkdir(mode=0o700)
        identity = {"schema": SCHEMA, "queue_id": request["queue_id"],
                    "queue_sha256": queue_sha, "session_id": session.name,
                    "pid": os.getpid(), "boot_id": queue.boot_id(),
                    "process_start_ticks": queue.process_start_ticks(os.getpid()),
                    "started_at": queue.utc(), "poll_seconds": poll_seconds,
                    "heartbeat_seconds": heartbeat_seconds}
        queue.write_json(session / "invocation.json", identity)
        stopped = threading.Event()
        stop_signal = []
        previous = {}

        def interrupted(signum, frame):
            if not stop_signal:
                stop_signal.append(signum)
            stopped.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        controller = None
        handles = []
        launches = 0
        recoveries = 0
        wait_for_slot = False
        last_queue = None
        last_queue_at = None
        last_state = None
        heartbeat_at = 0.0

        def publish(state, *, force=False, exit_code=None):
            nonlocal last_state, heartbeat_at
            now = time.monotonic()
            if not force and last_state == state and now - heartbeat_at < heartbeat_seconds:
                return
            value = {**identity, "schema": HEARTBEAT_SCHEMA, "updated_at": queue.utc(),
                     "state": state, "controller_pid": controller.pid if controller is not None else None,
                     "controller_launches": launches, "controller_recoveries": recoveries,
                     "automatic_job_retries": 0, "queue_snapshot": last_queue,
                     "queue_snapshot_is_live": False, "queue_snapshot_observed_at": last_queue_at,
                     "exit_code": exit_code}
            replace_heartbeat(state_dir / "heartbeat.json", value)
            last_state, heartbeat_at = state, now

        def finish(state, code):
            publish(state, force=True, exit_code=code)
            result = {"schema": SCHEMA, "session_id": session.name, "state": state,
                      "exit_code": code, "queue": last_queue, "finished_at": queue.utc(),
                      "controller_launches": launches, "controller_recoveries": recoveries,
                      "automatic_job_retries": 0, "heartbeat": str(state_dir / "heartbeat.json")}
            queue.write_json(session / "result.json", result)
            return result, code

        try:
            publish("starting", force=True)
            while True:
                if stopped.is_set():
                    stop_controller(controller)
                    return finish("stopped", 128 + stop_signal[0])
                if controller is not None:
                    code = controller.poll()
                    if code is None:
                        publish("controller_running")
                        stopped.wait(min(poll_seconds, heartbeat_seconds))
                        continue
                    for handle in handles:
                        handle.close()
                    handles.clear()
                    completed_pid = controller.pid
                    controller = None
                    output = session / f"controller-{launches:04d}.stdout.json"
                    raw = queue.read_bytes(output)
                    report = queue.loads(raw) if raw.strip() else None
                    if report is not None:
                        last_queue = summarize(report)
                        last_queue_at = queue.utc()
                    queue.write_json(session / f"controller-{launches:04d}.result.json",
                                     {"pid": completed_pid, "exit_code": code, "finished_at": queue.utc(),
                                      "queue": last_queue})
                    if code == 75:
                        wait_for_slot = report is not None and report["status"] == "waiting_for_heavy_model_slot"
                        publish("waiting_after_controller_race", force=True)
                        stopped.wait(poll_seconds)
                        continue
                    if code in (0, 1, 3):
                        queue.require(report is not None, "CONTROLLER_TERMINAL_REPORT_MISSING")
                        if code == 0:
                            queue.require(report["status"] == "completed", "INCOMPLETE_CONTROLLER_SUCCESS")
                        return finish(report["status"], code)
                    if code == 2:
                        return finish("invalid", 2)
                    # A controller process can die while its independently
                    # bounded worker is still alive. One controller recovery
                    # is allowed; the queue decides whether any job can run.
                    # A missing job result remains unknown, never a new try.
                    recoveries += 1
                    if recoveries > 1:
                        return finish("controller_failed", 1)
                    publish("recovering_controller", force=True)
                    continue
                waiting = waiting_for(root, request, metadata["heavy_lock"], check_slot=wait_for_slot)
                if waiting:
                    if waiting != "waiting_for_heavy_model_slot":
                        # Another owner may finish or invalidate the queue. On
                        # release, recheck its result before waiting on a slot
                        # that a terminal queue no longer needs.
                        wait_for_slot = False
                    publish(waiting)
                    stopped.wait(min(poll_seconds, heartbeat_seconds))
                    continue
                # Validate immutable declarations again before another launch.
                queue.load_queue(root)
                wait_for_slot = False
                launches += 1
                stdout = (session / f"controller-{launches:04d}.stdout.json").open("xb")
                handles.append(stdout)
                stderr = (session / f"controller-{launches:04d}.stderr.log").open("xb")
                handles.append(stderr)
                optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
                controller = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-B", *optimization, str(CONTROLLER),
                     "run", "--root", str(root)],
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True, start_new_session=True)
                queue.write_json(session / f"controller-{launches:04d}.started.json",
                                 {"pid": controller.pid, "started_at": queue.utc(),
                                  "process_start_ticks": queue.process_start_ticks(controller.pid)})
                publish("controller_running", force=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            stop_controller(controller)
            last_queue = {"status": "invalid", "reason": type(error).__name__ + ":" + str(error)}
            return finish("invalid", 2)
        finally:
            for handle in handles:
                handle.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def status(state_dir):
    state_dir = queue.safe_path(state_dir)
    binding = queue.loads(queue.read_bytes(state_dir / "binding.json"))
    queue.require(binding["schema"] == SCHEMA, "SUPERVISOR_SCHEMA")
    heartbeat = queue.loads(queue.read_bytes(state_dir / "heartbeat.json"))
    queue.require(heartbeat["schema"] == HEARTBEAT_SCHEMA and
                  heartbeat["queue_sha256"] == binding["queue_sha256"], "HEARTBEAT_BINDING_CHANGED")
    updated = datetime.fromisoformat(heartbeat["updated_at"])
    queue.require(updated.tzinfo is not None, "HEARTBEAT_TIMEZONE_REQUIRED")
    return {"schema": SCHEMA, "queue_root": binding["queue_root"],
            "supervisor_active": lock_active(state_dir / "supervisor.lock"),
            "heartbeat_age_seconds": (datetime.now(timezone.utc) - updated).total_seconds(),
            "heartbeat": heartbeat}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="host only this declared queue until terminal or stopped")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=1.0)
    run.add_argument("--heartbeat-seconds", type=float, default=5.0)
    show = commands.add_parser("status", help="read liveness without starting a queue or model")
    show.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result, code = status(args.state_dir), 0
        else:
            result, code = supervise(args.root, args.state_dir, poll_seconds=args.poll_seconds,
                                     heartbeat_seconds=args.heartbeat_seconds)
        print(queue.canonical(result).decode(), end="")
        return code
    except BlockingIOError:
        print(queue.canonical({"state": "supervisor_busy", "exit_code": 75}).decode(), end="")
        return 75
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(queue.canonical({"state": "invalid", "exit_code": 2,
                               "reason": type(error).__name__ + ":" + str(error)}).decode(), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
