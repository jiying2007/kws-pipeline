#!/usr/bin/env python3
"""Persistent local research queue; stdlib only, no remote scheduler or models.

One immutable declaration, one durable claim per job, completion-driven local
workers, and the shared heavy-model flock. No outcome is automatically retried.
Commands are cooperative local entrypoints, not untrusted sandboxed programs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

# Isolated ``python -I -S`` excludes the script directory from sys.path. Resolve
# this repository-owned sibling explicitly before importing it; never depend on
# the caller's working directory or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_job_lock import SUPERVISED_ENV, heavy_model_slot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/research/active-plan-20261105.json"
SCHEMA = "kws-local-research-queue-v1"
RECEIPT_SCHEMA = "kws-local-research-job-result-v1"
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
NS_PER_MINUTE = 60_000_000_000


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (
        _ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def safe_path(path):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "PATH_ALIAS")
    return path


def read_bytes(path, maximum=16 * 1024 * 1024):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                0 <= info.st_size <= maximum, "BOUNDED_REGULAR_FILE_REQUIRED")
        raw = stream.read(maximum + 1)
        require(len(raw) == info.st_size, "FILE_CHANGED_DURING_READ")
        return raw


def file_record(path, *, durable=False):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "REGULAR_OUTPUT_REQUIRED")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        final = os.fstat(stream.fileno())
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) ==
                (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns),
                "OUTPUT_CHANGED_DURING_READ")
        if durable:
            os.fsync(stream.fileno())
        return {"bytes": info.st_size, "sha256": digest}


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw):
    """Make complete bytes visible once; never replace a prior receipt."""
    path = safe_path(path)
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    sync_dir(path.parent)


def write_json(path, value):
    write_new(path, canonical(value))


def utc():
    return datetime.now(timezone.utc).isoformat()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_start_ticks(pid):
    try:
        # comm can contain spaces/parentheses; starttime is field 22.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def source_identity():
    return {name: sha(read_bytes(Path(__file__).parent / name))
            for name in ("research_queue.py", "research_job_lock.py")}


def minutes_ns(value, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0 and
            (maximum is None or value <= maximum), "INVALID_MINUTE_BUDGET")
    result = int(value * NS_PER_MINUTE)
    require(result > 0, "MINUTE_BUDGET_TOO_SMALL")
    return result


def relative(name):
    require(type(name) is str and 0 < len(name) <= 512 and "\\" not in name,
            "OUTPUT_RELATIVE_PATH")
    parsed = PurePosixPath(name)
    require(bool(parsed.parts) and not parsed.is_absolute() and str(parsed) == name and
            all(part not in (".", "..", "") for part in parsed.parts), "OUTPUT_RELATIVE_PATH")
    return name


@contextmanager
def exclusive(path):
    path = safe_path(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "LOCK_FILE_IDENTITY")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def active(path):
    try:
        with exclusive(path):
            return False
    except BlockingIOError:
        return True


def validate_plan(plan):
    require(plan["schema"] == "kws-active-research-plan-v1", "ACTIVE_PLAN_SCHEMA")
    execution = plan["execution"]
    require(execution["maximum_model_jobs_concurrently"] == 1 and
            execution["maximum_runner_minutes_per_item"] == 999 and
            execution["global_budget_is_sum_of_actual_job_minutes"] is True and
            execution["advance_on"] == "completed prerequisite outputs" and
            execution["reuse_identical_completed_outputs"] is True and
            execution["retry_quality_failures_automatically"] is False and
            execution["paid_model_apis_allowed"] is False and
            execution["external_codex_shared_quota_allowed"] is False and
            execution["maximum_infrastructure_attempts_per_new_stage"] == 2,
            "ACTIVE_EXECUTION_POLICY_CHANGED")


def validate_request(request):
    require(type(request) is dict and set(request) ==
            {"schema", "queue_id", "global_budget_minutes", "jobs"} and
            request["schema"] == SCHEMA and ID.fullmatch(request["queue_id"]), "QUEUE_SCHEMA")
    minutes_ns(request["global_budget_minutes"])
    jobs = request["jobs"]
    require(type(jobs) is list and 1 <= len(jobs) <= 256, "BOUNDED_JOB_ROSTER")
    ids = set()
    for job in jobs:
        require(type(job) is dict and set(job) == {"job_id", "recipe", "recipe_sha256", "argv",
                "cwd", "depends_on", "timeout_minutes", "outputs"}, "JOB_SCHEMA")
        require(type(job["job_id"]) is str and ID.fullmatch(job["job_id"]) and
                job["job_id"] not in ids, "UNIQUE_JOB_ID_REQUIRED")
        ids.add(job["job_id"])
        require(type(job["recipe_sha256"]) is str and SHA.fullmatch(job["recipe_sha256"]), "RECIPE_SHA256")
        require(type(job["recipe"]) is str and Path(job["recipe"]).is_absolute() and
                type(job["cwd"]) is str and Path(job["cwd"]).is_absolute(), "ABSOLUTE_JOB_PATHS")
        argv = job["argv"]
        require(type(argv) is list and 1 <= len(argv) <= 256 and
                all(type(arg) is str and len(arg) <= 32768 and "\0" not in arg for arg in argv) and
                Path(argv[0]).is_absolute(), "DIRECT_ARGV_REQUIRED")
        require(type(job["depends_on"]) is list and
                all(type(value) is str for value in job["depends_on"]) and
                len(set(job["depends_on"])) == len(job["depends_on"]) and
                job["job_id"] not in job["depends_on"], "JOB_DEPENDENCIES")
        require(type(job["outputs"]) is list and 1 <= len(job["outputs"]) <= 256 and
                len(set(job["outputs"])) == len(job["outputs"]), "DECLARED_OUTPUTS_REQUIRED")
        for name in job["outputs"]:
            relative(name)
        minutes_ns(job["timeout_minutes"], 999)
    require(all(set(job["depends_on"]) <= ids for job in jobs), "UNKNOWN_PREREQUISITE")
    visited = set()
    while True:
        ready = {job["job_id"] for job in jobs if set(job["depends_on"]) <= visited}
        if ready <= visited:
            break
        visited |= ready
    require(visited == ids, "CYCLIC_PREREQUISITES")


def create(root, request_path, plan_path, heavy_lock):
    root, heavy_lock = safe_path(root), safe_path(heavy_lock)
    require(not root.exists() and root.parent.is_dir() and heavy_lock.parent.is_dir(), "NEW_QUEUE_ROOT_REQUIRED")
    raw, plan_raw = read_bytes(request_path), read_bytes(plan_path)
    request, plan = loads(raw), loads(plan_raw)
    validate_plan(plan)
    validate_request(request)
    recipes = {}
    for job in request["jobs"]:
        body = read_bytes(job["recipe"])
        require(sha(body) == job["recipe_sha256"], "RECIPE_HASH_MISMATCH:" + job["job_id"])
        require(safe_path(job["cwd"]).is_dir() and Path(job["argv"][0]).is_file() and
                os.access(job["argv"][0], os.X_OK), "JOB_EXECUTABLE_OR_CWD_MISSING")
        recipes[job["job_id"]] = body
    root.mkdir(mode=0o700)
    (root / "jobs").mkdir(mode=0o700)
    (root / "recipes").mkdir(mode=0o700)
    for job in request["jobs"]:
        folder = root / "jobs" / job["job_id"]
        folder.mkdir(mode=0o700)
        (folder / "outputs").mkdir(mode=0o700)
        sync_dir(folder)
        write_new(root / "recipes" / (job["job_id"] + ".json"), recipes[job["job_id"]])
    sync_dir(root / "jobs")
    write_new(root / "request.json", raw)
    write_new(root / "active-plan.json", plan_raw)
    value = {"schema": SCHEMA, "created_at": utc(), "request_sha256": sha(raw),
             "active_plan_sha256": sha(plan_raw), "heavy_lock": str(heavy_lock),
             "source_sha256": source_identity(), "remote_watchdog_deployed": False,
             "historical_jobs_imported": 0, "automatic_retries": 0}
    write_json(root / "queue.json", value)
    sync_dir(root.parent)
    return inspect(root)


def load_queue(root):
    root = safe_path(root)
    queue_raw = read_bytes(root / "queue.json")
    queue = loads(queue_raw)
    require(queue["schema"] == SCHEMA and queue["source_sha256"] == source_identity(), "QUEUE_CONTROLLER_SOURCE_CHANGED")
    request_raw, plan_raw = read_bytes(root / "request.json"), read_bytes(root / "active-plan.json")
    require(sha(request_raw) == queue["request_sha256"] and sha(plan_raw) == queue["active_plan_sha256"], "FROZEN_QUEUE_CHANGED")
    request, plan = loads(request_raw), loads(plan_raw)
    validate_request(request)
    validate_plan(plan)
    for job in request["jobs"]:
        require(sha(read_bytes(root / "recipes" / (job["job_id"] + ".json"))) == job["recipe_sha256"], "FROZEN_RECIPE_CHANGED")
    return root, queue, request, sha(queue_raw)


def job_binding(job, queue_sha):
    return {"job_id": job["job_id"], "queue_sha256": queue_sha,
            "job_definition_sha256": sha(canonical(job)), "recipe_sha256": job["recipe_sha256"]}


def result_for(root, job, queue_sha, *, check_active=True):
    folder = root / "jobs" / job["job_id"]
    claim_path, result_path = folder / "claim.json", folder / "result.json"
    if not claim_path.exists():
        require(not result_path.exists(), "RESULT_WITHOUT_CLAIM")
        return {"status": "pending"}
    claim_raw = read_bytes(claim_path)
    claim = loads(claim_raw)
    binding = job_binding(job, queue_sha)
    require(all(claim.get(k) == v for k, v in binding.items()) and claim["attempt"] == 1,
            "CLAIM_IDENTITY_CHANGED")
    if check_active and active(folder / "job.lock"):
        return {"status": "running", "claim_sha256": sha(claim_raw)}
    if not result_path.exists():
        return {"status": "unknown", "reason": "claimed_without_terminal_receipt",
                "claim_sha256": sha(claim_raw), "actual_elapsed_ns": None}
    result = loads(read_bytes(result_path))
    require(result["schema"] == RECEIPT_SCHEMA and result["claim_sha256"] == sha(claim_raw) and
            all(result.get(k) == v for k, v in binding.items()) and
            result["status"] in ("completed", "failed", "timed_out", "unknown"), "RESULT_IDENTITY_CHANGED")
    elapsed = result["actual_elapsed_ns"]
    require(elapsed is None if result["status"] == "unknown" else
            type(elapsed) is int and elapsed >= 0, "RESULT_ACTUAL_DURATION")
    if result["status"] == "completed":
        require(result["exit_code"] == 0 and set(result["outputs"]) == set(job["outputs"]) and
                set(result["logs"]) == {"stdout.log", "stderr.log"}, "COMPLETED_OUTPUT_ROSTER")
        for name, expected in result["outputs"].items():
            require(file_record(folder / "outputs" / relative(name)) == expected, "COMPLETED_OUTPUT_CHANGED")
    for name, expected in result.get("logs", {}).items():
        require(name in ("stdout.log", "stderr.log") and file_record(folder / name) == expected, "JOB_LOG_CHANGED")
    result["receipt_sha256"] = sha(read_bytes(result_path))
    return result


def inspect(root, *, reconcile=False):
    root, queue, request, queue_sha = load_queue(root)
    states = {}
    for job in request["jobs"]:
        state = result_for(root, job, queue_sha)
        folder = root / "jobs" / job["job_id"]
        if reconcile and state["status"] == "unknown" and not (folder / "result.json").exists():
            with exclusive(folder / "job.lock"):
                state = result_for(root, job, queue_sha, check_active=False)
                if state["status"] == "unknown" and not (folder / "result.json").exists():
                    write_json(folder / "result.json", {"schema": RECEIPT_SCHEMA, **job_binding(job, queue_sha),
                        **state, "recorded_at": utc(), "exit_code": None, "automatic_retry": False,
                        "actual_duration_known": False, "outputs": {}, "logs": {}})
        states[job["job_id"]] = state
    used = sum(state.get("actual_elapsed_ns") or 0 for state in states.values())
    complete = all(state["status"] == "completed" for state in states.values())
    unresolved = [name for name, state in states.items() if state["status"] in ("unknown", "running")]
    return {"schema": "kws-local-research-queue-status-v1", "queue_id": request["queue_id"],
            "status": "completed" if complete else "incomplete", "jobs": states,
            "global_budget_minutes": request["global_budget_minutes"],
            "actual_known_job_nanoseconds": used, "actual_known_job_minutes": used / NS_PER_MINUTE,
            "actual_total_is_complete": not unresolved, "unresolved_jobs": unresolved,
            "budget_scope": "all_jobs_in_this_frozen_queue",
            "budget_overrun_nanoseconds": max(0, used - minutes_ns(request["global_budget_minutes"])),
            "remaining_budget_nanoseconds": max(0, minutes_ns(request["global_budget_minutes"]) - used),
            "heavy_lock": queue["heavy_lock"], "remote_watchdog_deployed": False,
            "automatic_retries": 0, "historical_jobs_imported": 0}


def expanded_argv(root, job):
    folder = root / "jobs" / job["job_id"]
    substitutions = {"{recipe}": str(root / "recipes" / (job["job_id"] + ".json")),
                     "{output_dir}": str(folder / "outputs"), "{job_dir}": str(folder)}
    return [substitutions.get(arg, arg) for arg in job["argv"]]


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def cleanup_recorded_command(folder, claim_sha):
    """After worker death, kill only a still-identical recorded live group leader."""
    path = folder / "command-started.json"
    if not path.exists():
        return
    record = loads(read_bytes(path))
    require(record["claim_sha256"] == claim_sha, "COMMAND_CLAIM_CHANGED")
    pid = record["pid"]
    try:
        if (record["boot_id"] == boot_id() and record["process_start_ticks"] is not None and
                process_start_ticks(pid) == record["process_start_ticks"] and os.getpgid(pid) == pid):
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def worker(root, job_id, lease_fd, claim_sha):
    root, queue, request, queue_sha = load_queue(root)
    job = next(job for job in request["jobs"] if job["job_id"] == job_id)
    folder = root / "jobs" / job_id
    info, named = os.fstat(lease_fd), (folder / "job.lock").stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
            (info.st_dev, info.st_ino) == (named.st_dev, named.st_ino), "WORKER_JOB_LOCK_IDENTITY")
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    claim_raw = read_bytes(folder / "claim.json")
    claim = loads(claim_raw)
    require(sha(claim_raw) == claim_sha and all(claim[k] == v for k, v in job_binding(job, queue_sha).items()) and
            claim["boot_id"] == boot_id() and not (folder / "result.json").exists() and
            sha(read_bytes(folder / "dependencies.json")) == claim["dependencies_sha256"], "WORKER_CLAIM_IDENTITY")
    with heavy_model_slot(queue["heavy_lock"]) as lease:
        write_json(folder / "worker-started.json", {"schema": "kws-local-research-worker-start-v1",
            "claim_sha256": claim_sha, "pid": os.getpid(), "started_at": utc()})
        command = expanded_argv(root, job)
        result = {"schema": RECEIPT_SCHEMA, **job_binding(job, queue_sha), "claim_sha256": claim_sha,
                  "status": "failed", "exit_code": None, "argv": command, "cwd": job["cwd"],
                  "outputs": {}, "logs": {}, "automatic_retry": False,
                  "allowed_runner_nanoseconds": claim["allowed_runner_nanoseconds"],
                  "started_at": claim["started_at"], "actual_duration_known": True}
        process = None
        previous = {}
        def interrupted(signum, frame):
            raise InterruptedError("WORKER_SIGNAL:" + str(signum))
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        try:
            with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
                environment = lease.child_environment()
                environment[SUPERVISED_ENV] = "1"
                environment.update(KWS_RESEARCH_JOB_ID=job_id,
                    KWS_RESEARCH_RECIPE=str(root / "recipes" / (job_id + ".json")),
                    KWS_RESEARCH_RECIPE_SHA256=job["recipe_sha256"],
                    KWS_RESEARCH_OUTPUT_DIR=str(folder / "outputs"),
                    KWS_RESEARCH_DEPENDENCIES=str(folder / "dependencies.json"))
                remaining = (claim["deadline_monotonic_ns"] - time.monotonic_ns()) / 1e9
                require(remaining > 0, "BUDGET_EXHAUSTED_BEFORE_COMMAND")
                process = subprocess.Popen(command, cwd=job["cwd"], env=environment, shell=False,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True,
                    pass_fds=tuple(set((*lease.pass_fds, lease_fd))))
                write_json(folder / "command-started.json", {"schema": "kws-local-research-command-start-v1",
                    "claim_sha256": claim_sha, "pid": process.pid, "process_group": process.pid,
                    "argv": command, "started_at": utc(), "boot_id": boot_id(),
                    "process_start_ticks": process_start_ticks(process.pid)})
                try:
                    process.wait(timeout=max(0, (claim["deadline_monotonic_ns"] - time.monotonic_ns()) / 1e9))
                except subprocess.TimeoutExpired:
                    kill_group(process)
                    process.wait()
                    result.update(status="timed_out", reason="actual_runner_deadline")
                else:
                    lingering = kill_group(process)
                    result.update(status="failed" if process.returncode != 0 or lingering else "completed",
                                  reason="lingering_process_group" if lingering else "command_exit")
                result["exit_code"] = process.returncode
                stdout.flush(); stderr.flush()
                os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
            if result["status"] == "completed":
                for name in job["outputs"]:
                    path = folder / "outputs" / name
                    result["outputs"][name] = file_record(path, durable=True)
                    for parent in path.parents:
                        sync_dir(parent)
                        if parent == folder:
                            break
        except BaseException as error:
            if process is not None:
                kill_group(process)
                process.wait()
                result["exit_code"] = process.returncode
            result.update(status="failed", reason=type(error).__name__ + ":" + str(error))
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        for name in ("stdout.log", "stderr.log"):
            if (folder / name).exists():
                result["logs"][name] = file_record(folder / name)
        elapsed = time.monotonic_ns() - claim["started_monotonic_ns"]
        result.update(finished_at=utc(), actual_elapsed_ns=elapsed,
                      actual_job_minutes=elapsed / NS_PER_MINUTE,
                      deadline_overrun_nanoseconds=max(0, elapsed - claim["allowed_runner_nanoseconds"]))
        write_json(folder / "result.json", result)
        return 0 if result["status"] == "completed" else 1


def run(root, *, max_jobs=None):
    """Advance ready jobs, optionally pausing after this many new claims.

    The limit is per invocation. Existing completed jobs never consume it,
    and a successful bounded pause still reports the queue as incomplete.
    """
    require(max_jobs is None or type(max_jobs) is int and max_jobs > 0, "MAX_JOBS_POSITIVE_INTEGER_REQUIRED")
    require(SUPERVISED_ENV not in os.environ, "NESTED_QUEUE_NOT_SUPPORTED")
    root, queue, request, queue_sha = load_queue(root)
    started_jobs = []
    with exclusive(root / "controller.lock"):
        while True:
            status = inspect(root, reconcile=True)
            if status["status"] == "completed":
                if status["budget_overrun_nanoseconds"]:
                    status["status"] = "completed_budget_overrun"
                    return status, 3
                return status, 0
            if status["unresolved_jobs"]:
                status["status"] = "running" if any(s["status"] == "running" for s in status["jobs"].values()) else "unknown"
                return status, 75 if status["status"] == "running" else 1
            ready = [job for job in request["jobs"] if status["jobs"][job["job_id"]]["status"] == "pending" and
                     all(status["jobs"][name]["status"] == "completed" for name in job["depends_on"])]
            if not ready:
                status["status"] = "failed_or_blocked"
                return status, 1
            if status["remaining_budget_nanoseconds"] == 0:
                status["status"] = "budget_exhausted"
                return status, 3
            if max_jobs is not None and len(started_jobs) >= max_jobs:
                status.update(stop_reason="max_jobs_reached", max_jobs=max_jobs,
                    jobs_started_this_invocation=len(started_jobs), started_job_ids=started_jobs)
                return status, 0 if all(status["jobs"][name]["status"] == "completed"
                                       for name in started_jobs) else 1
            job = ready[0]
            try:
                with heavy_model_slot(queue["heavy_lock"]) as lease, exclusive(root / "jobs" / job["job_id"] / "job.lock") as job_fd:
                    folder = root / "jobs" / job["job_id"]
                    dependencies = {name: {"receipt": str(root / "jobs" / name / "result.json"),
                        "receipt_sha256": status["jobs"][name]["receipt_sha256"],
                        "outputs_root": str(root / "jobs" / name / "outputs"),
                        "outputs": status["jobs"][name]["outputs"]} for name in job["depends_on"]}
                    allowed = min(minutes_ns(job["timeout_minutes"], 999), status["remaining_budget_nanoseconds"])
                    started = time.monotonic_ns()
                    claim = {"schema": "kws-local-research-job-claim-v1", **job_binding(job, queue_sha),
                        "attempt": 1, "started_at": utc(), "started_monotonic_ns": started,
                        "boot_id": boot_id(),
                        "dependencies_sha256": sha(canonical(dependencies)),
                        "allowed_runner_nanoseconds": allowed, "deadline_monotonic_ns": started + allowed,
                        "global_actual_before_nanoseconds": status["actual_known_job_nanoseconds"],
                        "automatic_retry": False}
                    write_json(folder / "claim.json", claim)
                    started_jobs.append(job["job_id"])
                    write_json(folder / "dependencies.json", dependencies)
                    optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
                    command = [sys.executable, "-I", "-S", "-B", *optimization, str(Path(__file__).resolve()), "worker", "--root", str(root),
                               "--job-id", job["job_id"], "--lease-fd", str(job_fd), "--claim-sha256", sha(canonical(claim))]
                    with (folder / "worker.log").open("xb") as log:
                        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            env=lease.child_environment(), pass_fds=tuple(set((*lease.pass_fds, job_fd))), start_new_session=True)
                        try:
                            process.wait(timeout=max(0, (started + allowed - time.monotonic_ns()) / 1e9) + 2)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        except KeyboardInterrupt:
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            cleanup_recorded_command(folder, sha(canonical(claim)))
                            raise
                        if process.returncode != 0:
                            cleanup_recorded_command(folder, sha(canonical(claim)))
                        log.flush(); os.fsync(log.fileno())
            except BlockingIOError:
                status["status"] = "waiting_for_heavy_model_slot"
                return status, 75


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="freeze an explicitly budgeted local queue, without executing jobs")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--request", type=Path, required=True)
    init.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    init.add_argument("--heavy-lock", type=Path, required=True)
    for name in ("run", "status", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "run":
            command.add_argument("--max-jobs", type=int, metavar="N",
                help="pause after at most N newly claimed jobs in this invocation (positive integer)")
        if name == "worker":
            command.add_argument("--job-id", required=True)
            command.add_argument("--lease-fd", type=int, required=True)
            command.add_argument("--claim-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result, code = create(args.root, args.request, args.plan, args.heavy_lock), 0
        elif args.command == "status":
            result, code = inspect(args.root), 0
        elif args.command == "run":
            result, code = run(args.root, max_jobs=args.max_jobs)
        else:
            return worker(args.root, args.job_id, args.lease_fd, args.claim_sha256)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return code
    except BlockingIOError as error:
        print(json.dumps({"status": "controller_busy", "reason": str(error)}))
        return 75
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(json.dumps({"status": "invalid", "reason": type(error).__name__ + ":" + str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
