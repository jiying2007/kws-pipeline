"""Offline qualification tests: real spawn contenders, atomic SQLite service, no sockets.

Fake service responses are fixtures, never live GitHub evidence. The fake does
not choose a winner: INSERT under BEGIN IMMEDIATE does, after a process barrier.
"""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

# Explicit sibling import also works under -I -S; this is only the frozen offline test script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import git_claim_qualification as q

SOURCE_SHA = "a" * 40
SOURCE_TREE = "b" * 40
RUN_ID = 9001
WORKFLOW_ID = 501
TOKEN = "test-token-never-persist-7Hn4p9"


_AUDIT_CAPTURE = None
_SAVED_STDOUT = None


def setUpModule():
    # Real spawned children inherit this FD. Synthetic receipts never enter hosted job logs.
    global _AUDIT_CAPTURE, _SAVED_STDOUT
    _AUDIT_CAPTURE = tempfile.TemporaryFile()
    _SAVED_STDOUT = os.dup(1)
    os.dup2(_AUDIT_CAPTURE.fileno(), 1)


def tearDownModule():
    global _AUDIT_CAPTURE, _SAVED_STDOUT
    os.dup2(_SAVED_STDOUT, 1)
    os.close(_SAVED_STDOUT)
    _AUDIT_CAPTURE.seek(0)
    raw = _AUDIT_CAPTURE.read()
    _AUDIT_CAPTURE.close()
    if TOKEN.encode() in raw:
        raise AssertionError("synthetic credential leaked into audit")
    lines = raw.splitlines()
    for line in lines:
        if len(line) + 1 > 4096:
            raise AssertionError("non-atomic audit line")
        item = q.strict_json(line)
        if type(item) is not dict or "event" not in item:
            raise AssertionError("invalid audit projection")
    summary = {"event": "offline_test_audit", "synthetic": True, "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(), "lines": len(lines), "token_absent": True}
    os.write(1, q.encoded(summary) + b"\n")


def reference(ref, target):
    return {"ref": ref, "node_id": "REF_public_fixture", "url": q.API_ORIGIN + q.ROOT + "/git/refs/" + ref[5:],
            "object": {"type": "commit", "sha": target,
                       "url": q.API_ORIGIN + q.ROOT + "/git/commits/" + target}}


def run_row(number=1):
    return {"id": RUN_ID + number - 1, "workflow_id": WORKFLOW_ID, "path": q.WORKFLOW,
            "head_branch": q.BRANCH, "event": "push", "head_sha": SOURCE_SHA,
            "run_number": number, "run_attempt": 1, "status": "in_progress", "conclusion": None,
            "repository": {"full_name": q.REPOSITORY, "private": False},
            "head_repository": {"full_name": q.REPOSITORY, "private": False}}


@dataclass
class AtomicFake:
    database: str
    history_count: int = 1
    mode: str = "healthy"
    deny_status: int = 403

    def connect(self):
        return sqlite3.connect(Path(self.database).resolve().as_uri() + "?mode=rw",
                               uri=True, isolation_level=None, timeout=10)

    def request(self, method, path, body, token, budget):
        # Never store credentials, even in the private synthetic request audit.
        workflow_id = 10 ** 1000 if self.mode == "oversized_workflow_id" else WORKFLOW_ID
        with closing(self.connect()) as db:
            db.execute("INSERT INTO requests(method,path,body,pid,flags) VALUES(?,?,?,?,?)",
                       (method, path, body, os.getpid(), q.encoded([sys.flags.isolated, sys.flags.no_site,
                                                                 sys.flags.dont_write_bytecode, sys.flags.optimize])))
            if self.mode == "readfail" and method == "GET":
                raise OSError("unsafe exception contents " + token)
            if self.mode == "read_denied" and method == "GET":
                status, data = self.deny_status, {"message": "denied"}
            elif method == "POST":
                need_body = q.strict_json(body)
                ref, target = need_body["ref"], need_body["sha"]
                if self.mode == "post_denied":
                    status, data = self.deny_status, {"message": "denied"}
                elif self.mode == "post_timeout":
                    raise TimeoutError("unsafe timeout " + token)
                else:
                    if ref == q.RACE_REF:
                        # Both real calls must have entered before the atomic INSERT race.
                        # This coordinates arrival, never selects the successful owner.
                        deadline = time.monotonic() + 5
                        while db.execute("SELECT COUNT(*) FROM requests WHERE method='POST'").fetchone()[0] < 2:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("local fake race arrival deadline")
                            time.sleep(0.005)
                    db.execute("BEGIN IMMEDIATE")
                    old = db.execute("SELECT target FROM refs WHERE ref=?", (ref,)).fetchone()
                    if old is None:
                        db.execute("INSERT INTO refs VALUES(?,?)", (ref, target))
                        status, data = 201, reference(ref, target)
                    else:
                        status, data = 422, {"message": "Reference already exists", "documentation_url": "https://docs.github.com/rest/git/refs"}
                    db.execute("COMMIT")
                    if self.mode == "both201" and ref == q.RACE_REF:
                        status, data = 201, reference(ref, target)
                    if self.mode == "owner_wrong" and status == 201:
                        data = reference(ref, q.TARGET_B if target == q.TARGET_A else q.TARGET_A)
                    if self.mode == "post_redirect":
                        status, data = 302, {"message": "redirect"}
            elif path == q.workflow_endpoint():
                status, data = 200, {"id": workflow_id, "path": q.WORKFLOW, "state": "active"}
            elif path == q.ROOT + "/actions/runs/" + str(RUN_ID):
                status, data = 200, run_row()
                data["workflow_id"] = workflow_id
                if self.mode == "api_run2":
                    data["run_number"] = 2
            elif "/attempts/1/jobs?" in path:
                status, data = 200, {"total_count": 1, "jobs": [{"id": 7001, "name": q.JOB,
                    "run_id": RUN_ID, "run_attempt": 1, "head_sha": SOURCE_SHA,
                    "status": "in_progress", "conclusion": None}]}
                if self.mode == "job_wrong":
                    data["jobs"][0]["name"] = "foreign"
                if self.mode == "oversized_job_id":
                    data["jobs"][0]["id"] = 10 ** 1000
            elif path == q.history_endpoint(workflow_id):
                rows = [run_row(number) for number in range(1, self.history_count + 1)]
                for row in rows:
                    row["workflow_id"] = workflow_id
                if self.mode == "oversized_ordinal":
                    rows[-1]["run_number"] = 10 ** 1000
                status, data = 200, {"total_count": self.history_count, "workflow_runs": rows}
                if self.mode == "history_duplicate_ordinal":
                    rows[-1]["run_number"] = 1
                if self.mode == "history_truncated":
                    data["total_count"] += 1
                if self.mode == "history_foreign":
                    rows[-1]["head_sha"] = "f" * 40
            elif "/git/commits/" in path:
                target = path.rsplit("/", 1)[1]
                status, data = 200, {"sha": target, "tree": {"sha": SOURCE_TREE if target == SOURCE_SHA else q.FIXTURE_TREE},
                    "parents": [{"sha": p} for p in q.FIXTURE_PARENTS.get(target, ())]}
                if self.mode == "source_wrong" and target == SOURCE_SHA:
                    data["tree"]["sha"] = "f" * 40
                if self.mode == "parent_wrong" and target == q.TARGET_A:
                    data["parents"] = []
            elif path in (q.ref_endpoint(q.RACE_REF), q.ref_endpoint(q.LOSS_REF)):
                ref = "refs/" + path.split("/git/ref/", 1)[1]
                row = db.execute("SELECT target FROM refs WHERE ref=?", (ref,)).fetchone()
                status, data = (200, reference(ref, row[0])) if row else (404, {"message": "Not Found"})
            else:
                raise ValueError("unexpected fake endpoint")
            raw = q.encoded(data)
            if self.mode == "secret_response":
                raw = q.encoded({"message": token})
            if self.mode == "bad_json":
                raw = b'{"id":1,"id":2}'
            budget.observe(len(raw))
            return status, raw


class ForbiddenToken(dict):
    def pop(self, key, *args):
        if key == "GITHUB_TOKEN":
            raise AssertionError("token touched before guard")
        return super().pop(key, *args)

    def get(self, key, *args):
        if key == "GITHUB_TOKEN":
            raise AssertionError("token touched before guard")
        return super().get(key, *args)


class Fixture:
    def __init__(self, directory, *, enabled=True, count=1, mode="healthy", deny=403):
        self.root = Path(directory) / "checkout"
        self.root.mkdir()
        self.temp = Path(directory) / "runner"
        self.temp.mkdir()
        self.control = q.default_control()
        self.control["enabled"] = enabled
        for relative in q.FROZEN_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith("/git_claim_qualification.py"):
                path.write_bytes(Path(q.__file__).read_bytes())
            elif relative == q.CONTROL_PATH:
                path.write_bytes(q.encoded(self.control))
            else:
                path.write_text("local synthetic frozen fixture\n")
        self.freeze()
        self.database = str(Path(directory) / "service.sqlite")
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript("CREATE TABLE refs(ref TEXT PRIMARY KEY,target TEXT NOT NULL);"
                             "CREATE TABLE requests(method TEXT,path TEXT,body BLOB,pid INTEGER,flags BLOB);")
        self.transport = AtomicFake(self.database, count, mode, deny)
        self.env = {"GITHUB_RUN_NUMBER": "1", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": str(RUN_ID),
                    "GITHUB_REPOSITORY": q.REPOSITORY, "GITHUB_REF": "refs/heads/" + q.BRANCH,
                    "GITHUB_EVENT_NAME": "push", "GITHUB_JOB": q.JOB, "GITHUB_ACTIONS": "true",
                    "GITHUB_WORKFLOW_REF": q.REPOSITORY + "/" + q.WORKFLOW + "@refs/heads/" + q.BRANCH,
                    "GITHUB_WORKFLOW_SHA": SOURCE_SHA, "GITHUB_SHA": SOURCE_SHA,
                    "RUNNER_TEMP": str(self.temp), "GITHUB_TOKEN": TOKEN}

    def freeze(self):
        files = []
        for relative in q.FROZEN_PATHS:
            raw = (self.root / relative).read_bytes()
            files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        self.freeze_path = self.root / q.FREEZE_PATH
        self.freeze_path.write_bytes(q.encoded({"schema_version": 1, "files": files}))
        self.freeze_hash = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()

    def execute(self):
        return q.execute(control_path=str(self.root / q.CONTROL_PATH), checkout_sha=SOURCE_SHA,
                         checkout_tree=SOURCE_TREE, source_freeze=str(self.freeze_path),
                         source_freeze_sha=self.freeze_hash, checkout_root=str(self.root),
                         env=self.env, transport=self.transport)

    def requests(self):
        with closing(sqlite3.connect(self.database)) as db:
            return db.execute("SELECT method,path,body,pid FROM requests").fetchall()

    def evidence(self, name):
        return self.temp / (q.CAMPAIGN + "-run-" + str(RUN_ID)) / name


def run_source_event(fixture, number, barrier, sender):
    fixture.env["GITHUB_RUN_NUMBER"] = str(number)
    fixture.env["GITHUB_RUN_ID"] = str(RUN_ID + number - 1)
    if number == 1:
        fixture.env["GITHUB_TOKEN"] = TOKEN  # Synthetic test secret, never a hosted credential.
    else:
        fixture.env = ForbiddenToken(fixture.env)
    barrier.wait(timeout=20)
    sender.send(fixture.execute())
    sender.close()


def die_holding_lock(budget, sender):
    budget.mutex.acquire()
    sender.send(True)
    sender.close()
    time.sleep(60)  # Parent kills this process while it owns the mutex.


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name

    def fixture(self, **kwargs):
        return Fixture(self.directory, **kwargs)

    def assert_no_post(self, fixture, result):
        self.assertNotEqual(result["status"], "passed", result)
        self.assertEqual(result["post_dispatched"], 0)
        self.assertFalse(any(row[0] == "POST" for row in fixture.requests()))

    def test_real_process_atomic_race_and_loss(self):
        f = self.fixture()
        os.environ["GITHUB_TOKEN"] = TOKEN
        result = f.execute()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual((result["get_requests"], result["post_reserved"], result["post_dispatched"]), (17, 3, 3))
        requests = f.requests()
        self.assertEqual(len(requests), 20)
        posts = [row for row in requests if row[0] == "POST"]
        self.assertEqual(len(set(row[3] for row in posts)), 3)
        with closing(sqlite3.connect(f.database)) as db:
            flags = [q.strict_json(row[0]) for row in db.execute("SELECT flags FROM requests WHERE method='POST'")]
        self.assertTrue(all(row == [sys.flags.isolated, sys.flags.no_site, sys.flags.dont_write_bytecode,
                                    sys.flags.optimize] for row in flags))
        self.assertEqual({q.strict_json(row[2])["sha"] for row in posts[:2]}, {q.TARGET_A, q.TARGET_B})
        children = [q.strict_json(f.evidence(owner + ".result.json").read_bytes()) for owner in ("A", "B")]
        self.assertEqual(sorted(row["status"] for row in children), [201, 422])
        self.assertEqual(sum(row["permit"] for row in children), 1)
        self.assertEqual(sum(row["witness"] for row in children), 1)
        overlap_ns = min(row["post_end_ns"] for row in children) - max(row["post_start_ns"] for row in children)
        self.assertGreater(overlap_ns, 0)
        self.assertEqual(result["race_call_overlap_ns"], overlap_ns)
        self.assertTrue(all(row["clean_environment"] for row in children))
        self.assertNotIn("GITHUB_TOKEN", os.environ)
        same = q.strict_json(f.evidence("same-owner-reentry.result.json").read_bytes())
        wrong = q.strict_json(f.evidence("wrong-owner-reentry.result.json").read_bytes())
        self.assertNotEqual(same["offered_target"], wrong["offered_target"])
        for row in (same, wrong):
            self.assertFalse(row["permit"])
            self.assertFalse(row["witness"])
            self.assertNotIn(row["pid"], {child["pid"] for child in children})
        loss = q.strict_json(f.evidence("loss-injection.json").read_bytes())
        self.assertEqual(loss["mode"], "injected_response_loss_plus_kill")
        self.assertFalse(loss["permit"] or loss["witness"])
        self.assertFalse(f.evidence("loss-A.result.json").exists())
        self.assertFalse(f.evidence("loss-A.witness.json").exists())
        raw = f.evidence("loss-A.response.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), loss["response_sha256"])
        self.assertEqual(q.validate_ref(q.strict_json(raw), q.LOSS_REF, q.TARGET_A), q.TARGET_A)
        for path in Path(self.directory).rglob("*"):
            if path.is_file():
                self.assertNotIn(TOKEN.encode(), path.read_bytes(), str(path))
        self.assertLessEqual(result["response_bytes"], q.MAX_RESPONSE_TOTAL)
        self.assertLessEqual(result["metadata_bytes"] + result["terminal_summary_max_bytes"], q.MAX_METADATA)
        self.assertEqual(result["source_sha"], SOURCE_SHA)
        self.assertEqual(result["source_freeze_sha256"], f.freeze_hash)
        self.assertEqual(result["run_id"], RUN_ID)
        self.assertEqual(result["workflow_id"], WORKFLOW_ID)
        self.assertEqual(result["job_id"], 7001)
        self.assertEqual(result["write_body_bytes"], sum(len(row[2]) for row in posts))

    def concurrent_events(self, count):
        f = self.fixture(count=count)
        f.env.pop("GITHUB_TOKEN")
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(count)
        processes, receivers = [], []
        for number in range(1, count + 1):
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(target=run_source_event, args=(f, number, barrier, sender))
            process.start()
            sender.close()
            processes.append(process)
            receivers.append(receiver)
        results = []
        for process, receiver in zip(processes, receivers):
            self.assertTrue(receiver.poll(20))
            results.append(receiver.recv())
            receiver.close()
            process.join(20)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(results[0]["status"], "passed", results)
        self.assertEqual(results[0]["history_initial_total"], count)
        self.assertEqual(results[0]["history_final_total"], count)
        for result in results[1:]:
            self.assertEqual(result["status"], "skipped")
            self.assertEqual((result["get_requests"], result["post_dispatched"]), (0, 0))
        self.assertEqual(sum(result["post_dispatched"] for result in results), 3)
        self.assertEqual(sum(result["get_requests"] for result in results), 17)
        self.assertEqual(len(f.requests()), 20)

    def test_two_concurrent_source_events_truthful(self):
        self.concurrent_events(2)

    def test_three_concurrent_source_events_truthful(self):
        self.concurrent_events(3)

    def test_maximum_history_projection_stays_bounded(self):
        f = self.fixture(count=100)
        result = f.execute()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["history_initial_total"], 100)
        self.assertEqual(result["history_final_total"], 100)
        self.assertLessEqual(result["metadata_bytes"] + 4096, q.MAX_METADATA)
        self.assertEqual(result["get_requests"], 17)

    def test_history_over_page_cap_closed(self):
        f = self.fixture(count=101)
        self.assert_no_post(f, f.execute())

    def test_run2_zero_api_before_token(self):
        f = self.fixture()
        f.env = ForbiddenToken(f.env)
        f.env["GITHUB_RUN_NUMBER"] = "2"
        result = f.execute()
        self.assertEqual(result["status"], "skipped", result)
        self.assertEqual(f.requests(), [])

    def test_attempt2_zero_api_before_token(self):
        f = self.fixture()
        f.env = ForbiddenToken(f.env)
        f.env["GITHUB_RUN_ATTEMPT"] = "2"
        self.assertEqual(f.execute()["status"], "skipped")
        self.assertEqual(f.requests(), [])

    def test_disabled_zero_api_before_token(self):
        f = self.fixture(enabled=False)
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "control_disabled")
        self.assertEqual(f.requests(), [])

    def test_same_run_restart_refused_before_token(self):
        f = self.fixture(mode="readfail")
        first = f.execute()
        self.assertEqual(first["get_requests"], 1)
        f.env = ForbiddenToken(f.env)
        second = f.execute()
        self.assertEqual(second["reason"], "invocation_already_exists", second)
        self.assertEqual(second["get_requests"], 0)
        self.assertEqual(len(f.requests()), 1)

    def test_read_failure_closed_no_exception_leak(self):
        f = self.fixture(mode="readfail")
        result = f.execute()
        self.assert_no_post(f, result)
        self.assertNotIn(TOKEN, json.dumps(result))
        self.assertNotIn("unsafe", json.dumps(result))

    def test_existing_claim_blocks_all_posts(self):
        f = self.fixture()
        with closing(sqlite3.connect(f.database)) as db:
            db.execute("INSERT INTO refs VALUES(?,?)", (q.RACE_REF, q.TARGET_A))
            db.commit()
        result = f.execute()
        self.assert_no_post(f, result)
        self.assertEqual(result["get_requests"], 8)

    def test_existing_loss_blocks_all_posts(self):
        f = self.fixture()
        with closing(sqlite3.connect(f.database)) as db:
            db.execute("INSERT INTO refs VALUES(?,?)", (q.LOSS_REF, q.TARGET_A))
            db.commit()
        result = f.execute()
        self.assert_no_post(f, result)
        self.assertEqual(result["get_requests"], 9)

    def test_source_wrong(self):
        f = self.fixture(mode="source_wrong")
        self.assert_no_post(f, f.execute())

    def test_fixture_parent_wrong(self):
        f = self.fixture(mode="parent_wrong")
        self.assert_no_post(f, f.execute())

    def test_missing_nullable_metadata_is_unknown(self):
        row = run_row()
        del row["conclusion"]
        with self.assertRaisesRegex(q.Closed, "run_identity"):
            q.validate_run(row, {"run_id": RUN_ID, "source_sha": SOURCE_SHA}, WORKFLOW_ID, current=True)

    def test_api_ordinal_wrong(self):
        f = self.fixture(mode="api_run2")
        self.assert_no_post(f, f.execute())

    def test_job_wrong(self):
        f = self.fixture(mode="job_wrong")
        self.assert_no_post(f, f.execute())

    def test_oversized_numeric_api_identities_closed_before_writes(self):
        for mode, expected_gets, reason in (("oversized_workflow_id", 1, "workflow_identity"),
                                             ("oversized_job_id", 3, "job_identity"),
                                             ("oversized_ordinal", 4, "run_ordinals")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                f = Fixture(directory, count=2, mode=mode)
                result = f.execute()
                self.assert_no_post(f, result)
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["get_requests"], expected_gets)
                self.assertEqual(result["post_reserved"], 0)
                self.assertEqual(len(f.requests()), expected_gets)

    def test_history_duplicate_ordinal_closed_before_writes(self):
        f = self.fixture(count=2, mode="history_duplicate_ordinal")
        result = f.execute()
        self.assert_no_post(f, result)
        self.assertEqual(result["reason"], "history_duplicate_ordinal")
        self.assertEqual(result["get_requests"], 4)
        self.assertEqual(result["post_reserved"], 0)
        self.assertEqual(len(f.requests()), 4)

    def test_history_truncated(self):
        f = self.fixture(count=2, mode="history_truncated")
        self.assert_no_post(f, f.execute())

    def test_history_foreign_source(self):
        f = self.fixture(count=2, mode="history_foreign")
        self.assert_no_post(f, f.execute())

    def test_owner_wrong_never_grants(self):
        f = self.fixture(mode="owner_wrong")
        result = f.execute()
        self.assertEqual(result["status"], "failed", result)
        self.assertLessEqual(result["post_dispatched"], 2)
        self.assertFalse(f.evidence("A.witness.json").exists())
        self.assertFalse(f.evidence("B.witness.json").exists())
        self.assertFalse(f.evidence("loss-A.response.json").exists())

    def test_both201_negative_stops_loss(self):
        f = self.fixture(mode="both201")
        result = f.execute()
        self.assertEqual(result["reason"], "race_not_one_winner", result)
        self.assertEqual(result["post_dispatched"], 2)
        self.assertEqual(result["get_requests"], 9)

    def test_disjoint_client_call_evidence_stops_later_phases(self):
        f = self.fixture()
        actual_children = q.children
        def disjoint_evidence(*args, **kwargs):
            results = actual_children(*args, **kwargs)
            if kwargs.get("barrier"):
                # Deliberately synthetic timestamp evidence; actual service and winner remain real.
                results[0].update(post_start_ns=100, post_end_ns=200)
                results[1].update(post_start_ns=300, post_end_ns=400)
            return results
        with patch.object(q, "children", side_effect=disjoint_evidence):
            result = f.execute()
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["reason"], "qualification_not_concurrent")
        self.assertEqual((result["get_requests"], result["post_dispatched"]), (9, 2))
        self.assertFalse(f.evidence("loss-A.response.json").exists())
        self.assertEqual(sum(row[0] == "POST" for row in f.requests()), 2)

    def test_post_timeout_no_retry_or_later_phase(self):
        f = self.fixture(mode="post_timeout")
        result = f.execute()
        self.assertEqual(result["status"], "failed")
        self.assertLessEqual(result["post_dispatched"], 2)
        self.assertEqual(result["get_requests"], 9)
        self.assertFalse(f.evidence("loss-A.response.json").exists())

    def test_denied_statuses_no_retry(self):
        for status in (401, 403, 429):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                f = Fixture(directory, mode="post_denied", deny=status)
                result = f.execute()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["get_requests"], 9)
                self.assertLessEqual(result["post_dispatched"], 2)
                self.assertFalse(any(q.strict_json(row[2])["ref"] == q.LOSS_REF for row in f.requests() if row[0] == "POST"))

    def test_read_denied_no_retry(self):
        for status in (401, 403, 429):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                f = Fixture(directory, mode="read_denied", deny=status)
                result = f.execute()
                self.assert_no_post(f, result)
                self.assertEqual(len(f.requests()), 1)

    def test_redirect_no_fallback(self):
        f = self.fixture(mode="post_redirect")
        result = f.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["get_requests"], 9)
        self.assertLessEqual(result["post_dispatched"], 2)

    def test_secret_echo_never_written(self):
        f = self.fixture(mode="secret_response")
        result = f.execute()
        self.assertEqual(result["reason"], "response_secret_rejected")
        for path in f.temp.rglob("*"):
            if path.is_file():
                self.assertNotIn(TOKEN.encode(), path.read_bytes())

    def test_duplicate_json_closed(self):
        f = self.fixture(mode="bad_json")
        self.assert_no_post(f, f.execute())

    def test_freeze_changed_file_no_token(self):
        f = self.fixture()
        (f.root / q.WORKFLOW).write_text("changed")
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "freeze_file_size")
        self.assertEqual(f.requests(), [])

    def test_freeze_extra_key_no_token(self):
        f = self.fixture()
        data = q.strict_json(f.freeze_path.read_bytes())
        data["extra"] = "rejected"
        f.freeze_path.write_bytes(q.encoded(data))
        f.freeze_hash = hashlib.sha256(f.freeze_path.read_bytes()).hexdigest()
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "freeze_schema")
        self.assertEqual(f.requests(), [])

    def test_freeze_symlink_rejected(self):
        f = self.fixture()
        path = f.root / q.WORKFLOW
        raw = path.read_bytes()
        path.unlink()
        elsewhere = Path(self.directory) / "elsewhere"
        elsewhere.write_bytes(raw)
        path.symlink_to(elsewhere)
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "freeze_symlink")
        self.assertEqual(f.requests(), [])

    def test_freeze_path_escape_rejected(self):
        f = self.fixture()
        data = q.strict_json(f.freeze_path.read_bytes())
        data["files"][0]["path"] = "../../outside"
        f.freeze_path.write_bytes(q.encoded(data))
        f.freeze_hash = hashlib.sha256(f.freeze_path.read_bytes()).hexdigest()
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "freeze_entry")
        self.assertEqual(f.requests(), [])

    def test_control_extra_field_rejected(self):
        f = self.fixture()
        f.control["retry"] = True
        (f.root / q.CONTROL_PATH).write_bytes(q.encoded(f.control))
        f.freeze()
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "control_schema")
        self.assertEqual(f.requests(), [])

    def test_checkout_sha_mismatch_before_token(self):
        f = self.fixture()
        f.env["GITHUB_SHA"] = "f" * 40
        f.env = ForbiddenToken(f.env)
        self.assertEqual(f.execute()["reason"], "checkout_identity")
        self.assertEqual(f.requests(), [])


class DirectClientTests(unittest.TestCase):
    def budget(self):
        return q.Budget(mp.get_context("spawn"), time.monotonic() + 30)

    def test_direct_https_pins_origin_no_proxy_or_redirect(self):
        budget = self.budget()
        class Response:
            status = 302
            def getheader(self, key, default):
                return default
            def read1(self, size):
                return b""
        class Connection:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                seen.append((args, kwargs))
            def request(self, *args, **kwargs):
                seen.append((args, kwargs))
            def getresponse(self):
                return Response()
            def close(self):
                pass
        seen = []
        with patch.object(q.http.client, "HTTPSConnection", Connection), patch.dict(os.environ, {"HTTPS_PROXY": "http://invalid.example"}):
            result = q.DirectHTTPS().request("GET", q.workflow_endpoint(), None, TOKEN, budget)
        self.assertEqual(result, (302, b""))
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], ("api.github.com", 443))
        self.assertLessEqual(seen[0][1]["timeout"], 30)
        self.assertEqual(seen[1][1]["headers"]["X-GitHub-Api-Version"], q.API_VERSION)

    def test_request_alarm_covers_whole_request(self):
        start = time.monotonic()
        with self.assertRaisesRegex(q.Closed, "request_timeout"):
            with q.request_alarm(0.02):
                time.sleep(0.2)
        self.assertLess(time.monotonic() - start, 0.15)

    def test_deadline_and_response_total_budget(self):
        budget = self.budget()
        budget.observe(q.MAX_RESPONSE_TOTAL)
        with self.assertRaisesRegex(q.Closed, "response_total_cap"):
            budget.observe(1)
        self.assertEqual(budget.snapshot()["response_bytes"], q.MAX_RESPONSE_TOTAL)
        budget.deadline = time.monotonic() - 1
        with self.assertRaisesRegex(q.Closed, "global_deadline"):
            budget.get()

    def test_dead_child_lock_never_blocks_forever(self):
        ctx = mp.get_context("spawn")
        budget = self.budget()
        receiver, sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=die_holding_lock, args=(budget, sender))
        process.start()
        sender.close()
        self.assertTrue(receiver.poll(10))
        self.assertTrue(receiver.recv())
        receiver.close()
        process.kill()
        process.join(10)
        self.assertEqual(process.exitcode, -9)
        start = time.monotonic()
        self.assertIsNone(budget.snapshot()["response_bytes"])
        budget.fail()
        with self.assertRaisesRegex(q.Closed, "budget_lock_unknown"):
            budget.get()
        self.assertLess(time.monotonic() - start, 2.5)

    def test_response_reservations_are_separate_from_observed(self):
        budget = self.budget()
        budget.reserve_response()
        budget.reserve_response()
        self.assertEqual(budget.snapshot()["response_bytes"], 0)
        self.assertEqual(budget.snapshot()["response_reserved_or_unknown_bytes"], 2 * q.MAX_RESPONSE)
        budget.observe(32, reserved=True)
        budget.release_response(q.MAX_RESPONSE - 32)
        self.assertEqual(budget.snapshot()["response_bytes"], 32)
        self.assertEqual(budget.snapshot()["response_reserved_or_unknown_bytes"], q.MAX_RESPONSE)
        budget.reserve_response()
        budget.reserve_response()
        with self.assertRaisesRegex(q.Closed, "response_total_cap"):
            budget.reserve_response()

    def test_direct_response_cap_and_unknown_reservation(self):
        budget = self.budget()
        class Response:
            status = 200
            def getheader(self, key, default):
                return default
            def read1(self, size):
                return b"x" * size
        class Connection:
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return Response()
            def close(self):
                pass
        with patch.object(q.http.client, "HTTPSConnection", Connection):
            with self.assertRaisesRegex(q.Closed, "response_cap"):
                q.DirectHTTPS().request("GET", q.workflow_endpoint(), None, TOKEN, budget)
        self.assertEqual(budget.snapshot()["response_bytes"], q.MAX_RESPONSE)
        self.assertEqual(budget.snapshot()["response_reserved_or_unknown_bytes"], 0)

    def test_metadata_cap_without_printing_server_body(self):
        budget = self.budget()
        budget.metadata(q.MAX_METADATA - 4096)
        with self.assertRaisesRegex(q.Closed, "metadata_cap"):
            budget.metadata(1)

    def test_post_tickets_nonreusable_and_bounded(self):
        budget = self.budget()
        for ticket in range(3):
            budget.reserve_post(ticket)
            budget.consume_post(ticket)
        with self.assertRaisesRegex(q.Closed, "post_already_reserved"):
            budget.reserve_post(0)
        with self.assertRaisesRegex(q.Closed, "post_ticket"):
            budget.consume_post(0)
        self.assertEqual(budget.snapshot()["post_dispatched"], 3)
        self.assertLessEqual(budget.snapshot()["write_body_bytes"], 1536)

    def test_get_budget_and_no_mutation_method_fallback(self):
        budget = self.budget()
        for _ in range(17):
            budget.get()
        with self.assertRaisesRegex(q.Closed, "request_budget"):
            budget.get()
        client = q.Client(TOKEN, self.budget(), {"run_id": RUN_ID, "source_sha": SOURCE_SHA})
        with self.assertRaisesRegex(q.Closed, "fixed_get"):
            client.request("PATCH", q.ROOT + "/git/refs", (200,))


if __name__ == "__main__":
    unittest.main()
