#!/usr/bin/env python3
"""Real bounded subprocess recovery; no model imports, audio or network."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_queue as queue
import research_queue_supervisor as supervisor
from research_job_lock import FD_ENV, PATH_ENV, SUPERVISED_ENV, heavy_model_slot


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.recipe = self.base / "recipe.json"
        self.recipe.write_bytes(b'{"fixture":"stdlib-only","model_calls":0}\n')
        self.lock = self.base / "heavy.lock"
        self.environment = {k: v for k, v in os.environ.items()
                            if k not in (FD_ENV, PATH_ENV, SUPERVISED_ENV)}
        self.processes = []
        self.gates = []
        self.logs = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for gate in self.gates:
            gate.touch(exist_ok=True)
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        # A supervisor killed during the test may have left its deliberately
        # independent controller alive. Only address these recorded fixtures.
        for path in self.base.glob("*/sessions/*/controller-*.started.json"):
            record = json.loads(path.read_bytes())
            pid = record["pid"]
            if queue.process_start_ticks(pid) == record["process_start_ticks"]:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for log in self.logs:
            log.close()

    def job(self, name, *, gate=None, depends=(), timeout=.2, code=None):
        if code is None:
            lines = ["from pathlib import Path;import os,time",
                     "output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])",
                     "output.joinpath('started').open('x').close()"]
            if gate is not None:
                self.gates.append(gate)
                lines += [f"gate=Path({str(gate)!r})", "while not gate.exists(): time.sleep(.01)"]
            lines += ["output.joinpath('answer.json').write_text('once')"]
            code = "\n".join(lines)
        return {"job_id": name, "recipe": str(self.recipe),
                "recipe_sha256": queue.sha(self.recipe.read_bytes()),
                "argv": [sys.executable, "-I", "-S", "-B", "-c", code], "cwd": str(self.base),
                "depends_on": list(depends), "timeout_minutes": timeout,
                "outputs": ["answer.json"]}

    def init(self, jobs, *, budget=1):
        root = self.base / "queue"
        request = self.base / "request.json"
        request.write_bytes(queue.canonical({"schema": queue.SCHEMA, "queue_id": "supervisor-test",
                                             "global_budget_minutes": budget, "jobs": jobs}))
        queue.create(root, request, ROOT / "configs/research/active-plan-20261105.json", self.lock)
        return root, self.base / "supervisor"

    def argv(self, root, state):
        optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        return [sys.executable, "-I", "-S", "-B", *optimization,
                str(ROOT / "tools/research_queue_supervisor.py"), "run", "--root", str(root),
                "--state-dir", str(state), "--poll-seconds", ".025", "--heartbeat-seconds", ".05"]

    def launch(self, root, state):
        log = (self.base / f"supervisor-{len(self.processes)}.log").open("wb")
        self.logs.append(log)
        process = subprocess.Popen(self.argv(root, state), env=self.environment, cwd=self.base,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("bounded subprocess condition did not occur")

    def heartbeat(self, state):
        path = state / "heartbeat.json"
        return json.loads(path.read_bytes()) if path.is_file() else {}

    def started(self, root, name):
        return (root / "jobs" / name / "outputs/started").exists()

    def test_completion_heartbeat_and_restart_reuse_exact_queue_outputs(self):
        root, state = self.init([self.job("first"), self.job("second", depends=["first"])])
        self.assertEqual(self.launch(root, state).wait(timeout=5), 0)
        self.assertEqual(self.heartbeat(state)["state"], "completed")
        observed = supervisor.status(state)
        self.assertFalse(observed["supervisor_active"])
        self.assertLess(observed["heartbeat_age_seconds"], 5)
        snapshot = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                    for p in root.rglob("*") if p.is_file()}
        original = queue.inspect(root)
        # An already terminal queue needs no model slot, even if another queue
        # currently owns it. Hosting must not wait forever to report completion.
        with heavy_model_slot(self.lock, environment=self.environment):
            self.assertEqual(self.launch(root, state).wait(timeout=5), 0)
        for path, expected in snapshot.items():
            self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), expected)
        self.assertEqual(queue.inspect(root), original)
        self.assertEqual(len(list((state / "sessions").iterdir())), 2)
        summary = self.heartbeat(state)["queue_snapshot"]
        self.assertEqual(summary["job_counts"], {"completed": 2})
        self.assertEqual(summary["actual_known_job_nanoseconds"], original["actual_known_job_nanoseconds"])

    def test_busy_slot_waits_without_claims_or_budget_and_rejects_second_supervisor(self):
        root, state = self.init([self.job("later")])
        with heavy_model_slot(self.lock, environment=self.environment):
            process = self.launch(root, state)
            self.until(lambda: self.heartbeat(state).get("state") == "waiting_for_heavy_model_slot")
            first = self.heartbeat(state)
            self.until(lambda: self.heartbeat(state).get("updated_at") != first["updated_at"])
            self.assertTrue(supervisor.status(state)["supervisor_active"])
            self.assertFalse((root / "jobs/later/claim.json").exists())
            self.assertEqual(queue.inspect(root)["actual_known_job_nanoseconds"], 0)
            duplicate = self.launch(root, state)
            self.assertEqual(duplicate.wait(timeout=3), 75)
            self.assertEqual(self.heartbeat(state)["session_id"], first["session_id"])
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertEqual(queue.inspect(root)["jobs"]["later"]["status"], "completed")

    def test_killed_supervisor_restart_waits_for_existing_controller_without_repeating_jobs(self):
        gate = self.base / "release"
        root, state = self.init([self.job("first", gate=gate), self.job("second", depends=["first"])])
        first = self.launch(root, state)
        self.until(lambda: self.started(root, "first"))
        claim = (root / "jobs/first/claim.json").read_bytes()
        old_session = self.heartbeat(state)["session_id"]
        first.kill()
        first.wait(timeout=3)
        restored = self.launch(root, state)
        self.until(lambda: self.heartbeat(state).get("session_id") != old_session and
                   self.heartbeat(state).get("state") == "waiting_for_controller")
        self.assertFalse((root / "jobs/second/claim.json").exists())
        self.assertEqual((root / "jobs/first/claim.json").read_bytes(), claim)
        gate.touch()
        self.assertEqual(restored.wait(timeout=5), 0)
        self.assertEqual(queue.inspect(root)["status"], "completed")
        for name in ("first", "second"):
            self.assertEqual((root / "jobs" / name / "outputs/answer.json").read_text(), "once")

    def test_stop_halts_controller_dispatch_and_restart_rejoins_bounded_worker(self):
        gate = self.base / "release"
        root, state = self.init([self.job("first", gate=gate), self.job("second", depends=["first"])])
        first = self.launch(root, state)
        self.until(lambda: self.started(root, "first"))
        first.terminate()
        self.assertEqual(first.wait(timeout=4), 143)
        self.assertEqual(self.heartbeat(state)["state"], "stopped")
        self.assertTrue(supervisor.lock_active(root / "jobs/first/job.lock"))
        self.assertFalse((root / "jobs/second/claim.json").exists())
        restored = self.launch(root, state)
        self.until(lambda: self.heartbeat(state).get("state") == "waiting_for_worker")
        gate.touch()
        self.assertEqual(restored.wait(timeout=5), 0)
        self.assertEqual(queue.inspect(root)["status"], "completed")
        self.assertEqual(len(list((root / "jobs").glob("*/claim.json"))), 2)

    def test_controller_process_death_recovers_once_without_retrying_its_worker(self):
        gate = self.base / "release"
        root, state = self.init([self.job("first", gate=gate), self.job("second", depends=["first"])])
        process = self.launch(root, state)
        self.until(lambda: self.started(root, "first"))
        original_claim = (root / "jobs/first/claim.json").read_bytes()
        os.kill(self.heartbeat(state)["controller_pid"], signal.SIGKILL)
        self.until(lambda: self.heartbeat(state).get("state") == "waiting_for_worker")
        self.assertEqual(self.heartbeat(state)["controller_recoveries"], 1)
        self.assertFalse((root / "jobs/second/claim.json").exists())
        gate.touch()
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertEqual((root / "jobs/first/claim.json").read_bytes(), original_claim)
        self.assertEqual(self.heartbeat(state)["controller_recoveries"], 1)
        self.assertEqual(self.heartbeat(state)["automatic_job_retries"], 0)

    def test_real_worker_death_remains_unknown_across_supervisor_restarts(self):
        gate = self.base / "release"
        root, state = self.init([self.job("first", gate=gate), self.job("blocked", depends=["first"])])
        process = self.launch(root, state)
        self.until(lambda: self.started(root, "first"))
        worker = json.loads((root / "jobs/first/worker-started.json").read_bytes())
        os.kill(worker["pid"], signal.SIGKILL)
        # The original queue controller kills the surviving command group and
        # records the unresolved claim; the supervisor must not retry it.
        self.assertEqual(process.wait(timeout=5), 1)
        original = (root / "jobs/first/result.json").read_bytes()
        self.assertEqual(json.loads(original)["status"], "unknown")
        self.assertFalse((root / "jobs/blocked/claim.json").exists())
        self.assertEqual(self.launch(root, state).wait(timeout=5), 1)
        self.assertEqual((root / "jobs/first/result.json").read_bytes(), original)
        self.assertFalse((root / "jobs/blocked/claim.json").exists())
        self.assertFalse(self.heartbeat(state)["queue_snapshot"]["actual_total_is_complete"])

    def test_global_budget_exhaustion_is_terminal_and_not_reset_by_supervisor(self):
        # Leave enough time for an optimized child interpreter to start on a
        # loaded CI host; this case must exercise the running-command deadline,
        # not the distinct pre-command budget-exhaustion failure path.
        budget = .05
        root, state = self.init([self.job("long", code="import time;time.sleep(20)", timeout=1),
                                 self.job("next")], budget=budget)
        self.assertEqual(self.launch(root, state).wait(timeout=10), 3)
        observed = queue.inspect(root)
        receipt = observed["jobs"]["long"]
        self.assertEqual(receipt["status"], "timed_out")
        self.assertTrue((root / "jobs/long/command-started.json").is_file())
        self.assertEqual(receipt["allowed_runner_nanoseconds"], queue.minutes_ns(budget))
        self.assertFalse((root / "jobs/next/claim.json").exists())
        self.assertEqual(self.launch(root, state).wait(timeout=5), 3)
        self.assertEqual(queue.inspect(root), observed)
        self.assertEqual(self.heartbeat(state)["automatic_job_retries"], 0)

    def test_changed_queue_binding_and_nested_state_are_rejected_before_new_work(self):
        root, state = self.init([self.job("once")])
        self.assertEqual(self.launch(root, state).wait(timeout=5), 0)
        body = (root / "queue.json").read_bytes()
        (root / "queue.json").write_bytes(body + b"\n")
        self.assertEqual(self.launch(root, state).wait(timeout=5), 2)
        self.assertEqual(len(list((state / "sessions").iterdir())), 1)
        with self.assertRaisesRegex(ValueError, "SEPARATE_FROM_QUEUE"):
            supervisor.supervise(root, root / "invalid-state")


if __name__ == "__main__":
    unittest.main()
