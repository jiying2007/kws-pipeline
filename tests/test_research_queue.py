#!/usr/bin/env python3
"""Bounded real dummy commands, process death and shared locks; no ML imports."""
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
from research_job_lock import FD_ENV, PATH_ENV, SUPERVISED_ENV, heavy_model_slot


class QueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recipe = self.root / "declared-recipe.json"
        self.recipe.write_bytes(b'{"fixture":"stdlib-only","model_calls":0}\n')
        self.lock = self.root / "shared-heavy-model.lock"
        self.environment = {k: v for k, v in os.environ.items() if k not in (FD_ENV, PATH_ENV, SUPERVISED_ENV)}
        self.command = ROOT / "tools/research_queue.py"
        self.plan = ROOT / "configs/research/active-plan-20261105.json"

    def job(self, name, *, code=None, depends=(), timeout=0.1):
        code = code or "from pathlib import Path;import os;Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'answer.json').write_text('complete')"
        return {"job_id": name, "recipe": str(self.recipe), "recipe_sha256": queue.sha(self.recipe.read_bytes()),
                "argv": [sys.executable, "-S", "-B", "-c", code], "cwd": str(self.root),
                "depends_on": list(depends), "timeout_minutes": timeout, "outputs": ["answer.json"]}

    def init(self, jobs, *, budget=1, name="queue"):
        request = {"schema": queue.SCHEMA, "queue_id": name, "global_budget_minutes": budget, "jobs": jobs}
        path = self.root / (name + "-request.json")
        path.write_bytes(queue.canonical(request))
        root = self.root / name
        queue.create(root, path, self.plan, self.lock)
        return root

    def launch(self, root):
        log = (self.root / (root.name + "-controller.log")).open("wb")
        self.addCleanup(log.close)
        process = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(self.command), "run", "--root", str(root)],
                                   env=self.environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.addCleanup(lambda: self.stop(process))
        return process

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)

    def until(self, predicate, seconds=4):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("bounded dummy condition did not occur")

    def test_completion_drives_dependencies_and_resume_reuses_exact_outputs(self):
        first = self.job("first")
        code = "\n".join((
            "from pathlib import Path;import os,json",
            "deps=json.loads(Path(os.environ['KWS_RESEARCH_DEPENDENCIES']).read_text())",
            "value=Path(deps['first']['outputs_root'],'answer.json').read_text()",
            "print('dependency='+value)",
            "Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'answer.json').write_text(value+'-second')",
        ))
        root = self.init([self.job("second", code=code, depends=["first"]), first])
        self.assertEqual(queue.inspect(root)["actual_known_job_minutes"], 0)
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("completed", 0))
        output = root / "jobs/second/outputs/answer.json"
        self.assertEqual(output.read_text(), "complete-second")
        saved = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                 for p in root.rglob("*") if p.is_file()}
        repeated, code = queue.run(root)
        self.assertEqual(code, 0)
        self.assertEqual(status, repeated)
        for path, identity in saved.items():
            self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), identity)
        durations = [state["actual_elapsed_ns"] for state in status["jobs"].values()]
        self.assertEqual(status["actual_known_job_nanoseconds"], sum(durations))
        self.assertGreater(status["actual_known_job_minutes"], 0)
        self.assertFalse(status["remote_watchdog_deployed"])

    def test_isolated_cli_init_status_run_and_resume_from_foreign_directory(self):
        request = {"schema": queue.SCHEMA, "queue_id": "isolated", "global_budget_minutes": 1,
                   "jobs": [self.job("once")]}
        request_path = self.root / "cli-request.json"
        request_path.write_bytes(queue.canonical(request))
        root = self.root / "isolated-queue"
        # The public CLI must import its own helper, even from an unrelated cwd
        # containing a module with the same name and a hostile PYTHONPATH.
        (self.root / "research_job_lock.py").write_text("raise RuntimeError('untrusted helper imported')\n")
        environment = {**self.environment, "PYTHONPATH": str(self.root)}
        optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        prefix = [sys.executable, "-I", "-S", "-B", *optimization, str(self.command)]

        def cli(*arguments):
            result = subprocess.run([*prefix, *arguments], cwd=self.root, env=environment,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)

        created = cli("init", "--root", str(root), "--request", str(request_path),
                      "--plan", str(self.plan), "--heavy-lock", str(self.lock))
        self.assertEqual(created["status"], "incomplete")
        self.assertEqual(cli("status", "--root", str(root))["jobs"]["once"]["status"], "pending")
        completed = cli("run", "--root", str(root))
        self.assertEqual(completed["status"], "completed")
        receipt = root / "jobs/once/result.json"
        identity = (receipt.read_bytes(), receipt.stat().st_ino, receipt.stat().st_mtime_ns)
        self.assertEqual(cli("run", "--root", str(root)), completed)
        self.assertEqual(cli("status", "--root", str(root)), completed)
        self.assertEqual((receipt.read_bytes(), receipt.stat().st_ino, receipt.stat().st_mtime_ns), identity)

    def test_max_jobs_cli_pauses_before_next_claim_and_resumes_the_same_request(self):
        first_code = "\n".join((
            "from pathlib import Path;import os",
            "output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])",
            "output.joinpath('executed-once').open('x').close()",
            "output.joinpath('answer.json').write_text('prepared-for-verification')",
        ))
        next_code = "\n".join((
            "from pathlib import Path;import os,json",
            "deps=json.loads(Path(os.environ['KWS_RESEARCH_DEPENDENCIES']).read_text())",
            "value=Path(deps['prepare']['outputs_root'],'answer.json').read_text()",
            "output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])",
            "output.joinpath('executed-once').open('x').close()",
            "output.joinpath('answer.json').write_text(value+'-consumed')",
        ))
        request = {"schema": queue.SCHEMA, "queue_id": "bounded", "global_budget_minutes": 1,
                   "jobs": [self.job("prepare", code=first_code),
                            self.job("train", code=next_code, depends=["prepare"])]}
        request_path = self.root / "bounded-request.json"
        request_path.write_bytes(queue.canonical(request))
        root = self.root / "bounded-queue"
        optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        prefix = [sys.executable, "-I", "-S", "-B", *optimization, str(self.command)]

        def cli(*arguments):
            result = subprocess.run([*prefix, *arguments], cwd=self.root, env=self.environment,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)

        cli("init", "--root", str(root), "--request", str(request_path),
            "--plan", str(self.plan), "--heavy-lock", str(self.lock))
        immutable = [request_path, root / "request.json", root / "queue.json", root / "active-plan.json",
                     *sorted((root / "recipes").iterdir())]
        frozen = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in immutable}
        paused = cli("run", "--root", str(root), "--max-jobs", "1")
        self.assertEqual(paused["status"], "incomplete")
        self.assertEqual(paused["stop_reason"], "max_jobs_reached")
        self.assertEqual(paused["max_jobs"], 1)
        self.assertEqual(paused["jobs_started_this_invocation"], 1)
        self.assertEqual(paused["started_job_ids"], ["prepare"])
        self.assertEqual(paused["jobs"]["prepare"]["status"], "completed")
        self.assertEqual(paused["jobs"]["train"]["status"], "pending")
        # No next-job lock, claim, command, worker, output, or log has begun.
        self.assertEqual({p.name for p in (root / "jobs/train").iterdir()}, {"outputs"})
        self.assertEqual(list((root / "jobs/train/outputs").iterdir()), [])
        self.assertEqual(queue.inspect(root)["status"], "incomplete")
        first_folder = root / "jobs/prepare"
        self.assertEqual((first_folder / "outputs/answer.json").read_text(), "prepared-for-verification")
        first = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                 for p in first_folder.rglob("*") if p.is_file()}
        first_ns = paused["jobs"]["prepare"]["actual_elapsed_ns"]
        self.assertGreater(first_ns, 0)
        self.assertEqual(paused["actual_known_job_nanoseconds"], first_ns)
        self.assertEqual(paused["remaining_budget_nanoseconds"], queue.minutes_ns(1) - first_ns)
        with heavy_model_slot(self.lock, environment=self.environment):
            pass  # The pause leaves the real global slot free for verification.

        # A caller can verify preparation here, then advance the same request.
        completed = cli("run", "--root", str(root), "--max-jobs", "1")
        self.assertEqual(completed["status"], "completed")
        self.assertNotIn("stop_reason", completed)
        self.assertEqual((root / "jobs/train/outputs/answer.json").read_text(),
                         "prepared-for-verification-consumed")
        second_ns = completed["jobs"]["train"]["actual_elapsed_ns"]
        self.assertGreater(second_ns, 0)
        self.assertEqual(completed["actual_known_job_nanoseconds"], first_ns + second_ns)
        self.assertEqual(completed["remaining_budget_nanoseconds"], queue.minutes_ns(1) - first_ns - second_ns)
        claim = json.loads((root / "jobs/train/claim.json").read_bytes())
        self.assertEqual(claim["global_actual_before_nanoseconds"], first_ns)
        for path, identity in {**frozen, **first}.items():
            self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), identity)
        self.assertEqual(cli("run", "--root", str(root)), completed)
        self.assertEqual(cli("run", "--root", str(root), "--max-jobs", "1"), completed)

    def test_max_jobs_rejects_nonpositive_or_noninteger_values_before_claiming(self):
        root = self.init([self.job("untouched")])
        for invalid in (0, -1, True, False, 1.5, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "MAX_JOBS_POSITIVE_INTEGER_REQUIRED"):
                queue.run(root, max_jobs=invalid)
        optimization = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        for invalid in ("0", "-1", "1.5"):
            result = subprocess.run([sys.executable, "-I", "-S", "-B", *optimization, str(self.command),
                "run", "--root", str(root), "--max-jobs", invalid], cwd=self.root, env=self.environment,
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse((root / "jobs/untouched/claim.json").exists())
        self.assertEqual(queue.inspect(root)["actual_known_job_nanoseconds"], 0)

    def test_max_jobs_does_not_turn_a_failed_job_into_success(self):
        root = self.init([self.job("failed", code="import sys;sys.exit(7)"), self.job("next")])
        status, code = queue.run(root, max_jobs=1)
        self.assertEqual((status["status"], status["stop_reason"], code),
                         ("incomplete", "max_jobs_reached", 1))
        self.assertEqual(status["jobs"]["failed"]["exit_code"], 7)
        self.assertFalse((root / "jobs/next/claim.json").exists())

    def test_max_jobs_does_not_hide_global_budget_exhaustion(self):
        root = self.init([self.job("long", code="import time;time.sleep(30)", timeout=1),
                          self.job("next")], budget=.006)
        status, code = queue.run(root, max_jobs=1)
        self.assertEqual((status["status"], code), ("budget_exhausted", 3))
        self.assertNotIn("stop_reason", status)
        self.assertEqual(status["jobs"]["long"]["status"], "timed_out")
        self.assertFalse((root / "jobs/next/claim.json").exists())

    def test_shell_metacharacters_are_literal_argv(self):
        job = self.job("literal", code="from pathlib import Path;import sys;Path(sys.argv[1],'answer.json').write_text(sys.argv[2])")
        literal = "$(touch SHOULD_NOT_EXIST); `printf surprise` *"
        job["argv"] += ["{output_dir}", literal]
        root = self.init([job])
        self.assertEqual(queue.run(root)[1], 0)
        self.assertEqual((root / "jobs/literal/outputs/answer.json").read_text(), literal)
        self.assertFalse((self.root / "SHOULD_NOT_EXIST").exists())

    def test_failure_is_not_retried_and_only_its_dependents_are_blocked(self):
        failed = self.job("failed", code="import sys;print('recorded failure',file=sys.stderr);sys.exit(7)")
        root = self.init([failed, self.job("blocked", depends=["failed"]), self.job("independent")])
        status, code = queue.run(root)
        self.assertEqual(code, 1)
        self.assertEqual(status["jobs"]["failed"]["exit_code"], 7)
        self.assertEqual(status["jobs"]["independent"]["status"], "completed")
        self.assertEqual(status["jobs"]["blocked"]["status"], "pending")
        receipt = (root / "jobs/failed/result.json").read_bytes()
        self.assertIn("recorded failure", (root / "jobs/failed/stderr.log").read_text())
        self.assertEqual(queue.run(root)[1], 1)
        self.assertEqual((root / "jobs/failed/result.json").read_bytes(), receipt)
        self.assertFalse((root / "jobs/blocked/claim.json").exists())

    def test_actual_global_budget_bounds_command_and_leaves_next_job_unclaimed(self):
        root = self.init([self.job("long", code="import time;time.sleep(30)", timeout=1), self.job("next")], budget=.006)
        started = time.monotonic()
        status, code = queue.run(root)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual((status["status"], code), ("budget_exhausted", 3))
        receipt = status["jobs"]["long"]
        self.assertEqual(receipt["status"], "timed_out")
        self.assertEqual(receipt["allowed_runner_nanoseconds"], queue.minutes_ns(.006))
        self.assertGreaterEqual(receipt["actual_elapsed_ns"], receipt["allowed_runner_nanoseconds"])
        self.assertEqual(status["actual_known_job_nanoseconds"], receipt["actual_elapsed_ns"])
        self.assertFalse((root / "jobs/next/claim.json").exists())

    def test_shared_external_slot_is_not_bypassed_or_charged_as_execution(self):
        root = self.init([self.job("waiting")])
        with heavy_model_slot(self.lock, environment=self.environment):
            status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("waiting_for_heavy_model_slot", 75))
        self.assertEqual(status["actual_known_job_nanoseconds"], 0)
        self.assertFalse((root / "jobs/waiting/claim.json").exists())
        self.assertEqual(queue.run(root)[1], 0)

    def test_distinct_queues_use_one_real_global_slot(self):
        code = "from pathlib import Path;import os,time;Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'started').touch();time.sleep(.4);Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'answer.json').write_text('done')"
        first = self.init([self.job("held", code=code)], name="first")
        second = self.init([self.job("other")], name="second")
        process = self.launch(first)
        self.until(lambda: (first / "jobs/held/outputs/started").exists())
        self.assertEqual(queue.run(second)[1], 75)
        self.assertFalse((second / "jobs/other/claim.json").exists())
        self.assertEqual(process.wait(timeout=3), 0)
        self.assertEqual(queue.run(second)[1], 0)

    def test_controller_death_keeps_bounded_worker_and_resume_does_not_repeat(self):
        code = "from pathlib import Path;import os,time;Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'started').touch();time.sleep(.4);Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'answer.json').write_text('once')"
        root = self.init([self.job("survives", code=code), self.job("after", depends=["survives"])])
        process = self.launch(root)
        self.until(lambda: (root / "jobs/survives/outputs/started").exists())
        process.kill(); process.wait(timeout=2)
        self.assertEqual(queue.run(root)[1], 75)
        self.until(lambda: (root / "jobs/survives/result.json").exists() and not queue.active(root / "jobs/survives/job.lock"))
        receipt = (root / "jobs/survives/result.json").read_bytes()
        status, code = queue.run(root)
        self.assertEqual(code, 0)
        self.assertEqual(status["jobs"]["after"]["status"], "completed")
        self.assertEqual((root / "jobs/survives/result.json").read_bytes(), receipt)

    def test_orphaned_claim_is_durable_unknown_and_never_spends_unknown_as_zero(self):
        root = self.init([self.job("unknown"), self.job("later")])
        _, _, request, queue_sha = queue.load_queue(root)
        claim = {"schema": "kws-local-research-job-claim-v1", **queue.job_binding(request["jobs"][0], queue_sha),
                 "attempt": 1, "started_at": queue.utc(), "started_monotonic_ns": time.monotonic_ns(),
                 "allowed_runner_nanoseconds": 1000000000, "deadline_monotonic_ns": time.monotonic_ns() + 1000000000,
                 "global_actual_before_nanoseconds": 0, "automatic_retry": False}
        queue.write_json(root / "jobs/unknown/claim.json", claim)
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("unknown", 1))
        self.assertFalse(status["actual_total_is_complete"])
        result = json.loads((root / "jobs/unknown/result.json").read_bytes())
        self.assertIsNone(result["actual_elapsed_ns"])
        self.assertFalse((root / "jobs/later/claim.json").exists())
        self.assertEqual(queue.run(root)[1], 1)

    def test_real_worker_death_cleans_known_command_and_preserves_unknown(self):
        code = "from pathlib import Path;import os,time;Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'started').touch();time.sleep(30)"
        root = self.init([self.job("crash", code=code), self.job("later")])
        process = self.launch(root)
        folder = root / "jobs/crash"
        self.until(lambda: (folder / "outputs/started").exists())
        worker = json.loads((folder / "worker-started.json").read_bytes())
        os.kill(worker["pid"], signal.SIGKILL)
        self.assertIn(process.wait(timeout=3), (1, 75))
        self.until(lambda: not queue.active(folder / "job.lock"))
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("unknown", 1))
        self.assertFalse(status["actual_total_is_complete"])
        self.assertIsNone(status["jobs"]["crash"]["actual_elapsed_ns"])
        self.assertFalse((root / "jobs/later/claim.json").exists())
        with heavy_model_slot(self.lock, environment=self.environment):
            pass

    def test_future_trainer_stage_stays_in_outer_group_and_deadline_covers_it(self):
        stage = "\n".join((
            "import sys,os,json,time;from pathlib import Path",
            "sys.path.insert(0,sys.argv[1]);from research_job_lock import heavy_model_slot,PATH_ENV",
            "with heavy_model_slot(os.environ[PATH_ENV]) as lease:",
            " Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'stage.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'supervised':lease.supervised,'inherited':lease.inherited}))",
            " time.sleep(30)",
        ))
        outer = "\n".join((
            "import sys,os;from pathlib import Path",
            "sys.path.insert(0,sys.argv[1]);from research_weak_train import stage_command",
            "from research_job_lock import heavy_model_slot,PATH_ENV",
            "with heavy_model_slot(os.environ[PATH_ENV]) as lease:",
            " with open(Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'stage.log'),'wb') as log:",
            "  code=stage_command([sys.executable,'-S','-B','-c',sys.argv[2],sys.argv[3]],log=log,lease=lease,seconds=30)",
            " raise SystemExit(code)",
        ))
        job = self.job("nested", code=outer, timeout=1)
        job["argv"] += [str(ROOT / "training"), stage, str(ROOT / "tools")]
        root = self.init([job, self.job("later")], budget=.02)
        status, code = queue.run(root)
        self.assertEqual((status["status"], code), ("budget_exhausted", 3))
        self.assertEqual(status["jobs"]["nested"]["status"], "timed_out")
        folder = root / "jobs/nested"
        actual = json.loads((folder / "outputs/stage.json").read_bytes())
        command = json.loads((folder / "command-started.json").read_bytes())
        self.assertEqual(actual["pgid"], command["process_group"])
        self.assertTrue(actual["supervised"] and actual["inherited"])
        proc = Path(f"/proc/{actual['pid']}/stat")
        self.until(lambda: not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z")
        with heavy_model_slot(self.lock, environment=self.environment):
            pass
        self.assertFalse((root / "jobs/later/claim.json").exists())

    def test_completed_artifact_and_frozen_recipe_drift_do_not_rerun(self):
        root = self.init([self.job("bound")])
        self.assertEqual(queue.run(root)[1], 0)
        path = root / "jobs/bound/outputs/answer.json"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "COMPLETED_OUTPUT_CHANGED"):
            queue.run(root)
        path.write_text("complete")
        (root / "recipes/bound.json").write_text("changed recipe")
        with self.assertRaisesRegex(ValueError, "FROZEN_RECIPE_CHANGED"):
            queue.run(root)

    def test_false_zero_exit_without_outputs_is_failed(self):
        root = self.init([self.job("missing", code="print('no output')")])
        status, code = queue.run(root)
        self.assertEqual(code, 1)
        self.assertEqual(status["jobs"]["missing"]["exit_code"], 0)
        self.assertEqual(status["jobs"]["missing"]["status"], "failed")

    def test_invalid_budgets_cycles_and_recipe_identity_fail_before_execution(self):
        for value in (0, -1, True, float("nan"), float("inf"), 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                queue.minutes_ns(value, 999)
        one, two = self.job("one", depends=["two"]), self.job("two", depends=["one"])
        with self.assertRaisesRegex(ValueError, "CYCLIC_PREREQUISITES"):
            self.init([one, two])
        job = self.job("bad")
        job["recipe_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "RECIPE_HASH_MISMATCH"):
            self.init([job])


if __name__ == "__main__":
    unittest.main()
