"""Bounded, zero-model Git create-only qualification; disabled unless pinned control enables it.

This is a qualification of observed REST behavior, not production admission, a
lease, or malicious-replay protection. Trusted workflow integrity, a single
invocation, and Actions' workflow ordinal contract are explicit assumptions.
"""
import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import ssl
import time
from urllib.parse import quote

REPOSITORY = "jiying2007/kws-pipeline"
CAMPAIGN = "k1-atomic-claim-qualification-v1"
WORKFLOW = ".github/workflows/" + CAMPAIGN + ".yml"
BRANCH = "research/" + CAMPAIGN
JOB = "qualify"
RACE_REF = "refs/heads/evidence/" + CAMPAIGN + "-race"
LOSS_REF = "refs/heads/evidence/" + CAMPAIGN + "-lost-response"
TARGET_A = "8b805105810518f849a42fcc6db5d2fd331b8e33"
TARGET_B = "1fd1b76275b22c0501d8be5bf1087ead7b1fca60"
FIXTURE_TREE = "aeea9ae50c067b11b1d2cabf5bb9000a31d2d1f2"
FIXTURE_PARENTS = {TARGET_A: ("276e56062b8d9eeac11ea93348a792b6bd008c8c",), TARGET_B: (TARGET_A,)}
ROOT = "/repos/" + REPOSITORY
API_ORIGIN = "https://api.github.com"
API_VERSION = "2022-11-28"
SOURCE_DIR = "research/" + CAMPAIGN
CONTROL_PATH = SOURCE_DIR + "/qualification-control.json"
FREEZE_PATH = SOURCE_DIR + "/source-freeze.json"
FROZEN_PATHS = tuple(sorted((WORKFLOW, CONTROL_PATH, SOURCE_DIR + "/README.md",
                            SOURCE_DIR + "/git_claim_qualification.py",
                            SOURCE_DIR + "/test_git_claim_qualification.py")))
MAX_SECONDS = 300
CLEANUP_SECONDS = 15
MAX_WORK_SECONDS = MAX_SECONDS - CLEANUP_SECONDS
MAX_REQUEST_SECONDS = 30
MAX_RESPONSE = 1024 * 1024
MAX_RESPONSE_TOTAL = 4 * MAX_RESPONSE
MAX_WRITE_BYTES = 1536
MAX_METADATA = 128 * 1024
TICKETS = ((RACE_REF, TARGET_A), (RACE_REF, TARGET_B), (LOSS_REF, TARGET_A))


class Closed(Exception):
    """Only constant, non-server-derived reason codes may be raised."""


def need(condition, code):
    if not condition:
        raise Closed(code)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Closed("nonfinite_json")))
    except Closed:
        raise
    except Exception:
        raise Closed("invalid_json") from None


def sha(value, size=40):
    return type(value) is str and re.fullmatch("[0-9a-f]{%d}" % size, value) is not None


def positive(value):
    return type(value) is int and 0 < value < 10 ** 20


def default_control():
    return {"schema_version": 1, "enabled": False, "repository": REPOSITORY,
            "workflow_path": WORKFLOW, "source_branch": BRANCH, "job": JOB,
            "race_ref": RACE_REF, "loss_ref": LOSS_REF, "target_a": TARGET_A,
            "target_b": TARGET_B, "fixture_tree": FIXTURE_TREE}


def validate_control(control):
    expected = default_control()
    need(type(control) is dict and set(control) == set(expected), "control_schema")
    need(type(control["enabled"]) is bool, "control_enabled_type")
    for key, value in expected.items():
        if key != "enabled":
            need(type(control[key]) is type(value) and control[key] == value, "control_identity")
    return control["enabled"]


def regular_file(root, relative):
    need(type(relative) is str and relative in (*FROZEN_PATHS, FREEZE_PATH), "freeze_path")
    path = root
    for part in relative.split("/"):
        path = path / part
        need(not path.is_symlink(), "freeze_symlink")
    need(path.is_file() and path.resolve().is_relative_to(root), "freeze_file")
    return path


def verify_freeze(root, freeze_path, freeze_sha, control_path):
    root = Path(root).resolve(strict=True)
    need(sha(freeze_sha, 64), "freeze_digest")
    frozen = regular_file(root, FREEZE_PATH)
    need(Path(freeze_path).absolute() == frozen, "freeze_location")
    need(Path(control_path).absolute() == root / CONTROL_PATH, "control_location")
    need(frozen.stat().st_size <= 65536, "freeze_size")
    raw = frozen.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == freeze_sha, "freeze_digest")
    data = strict_json(raw)
    need(type(data) is dict and set(data) == {"schema_version", "files"}
         and type(data["schema_version"]) is int and data["schema_version"] == 1, "freeze_schema")
    files = data["files"]
    need(type(files) is list and len(files) == len(FROZEN_PATHS), "freeze_roster")
    for item, expected in zip(files, FROZEN_PATHS):
        need(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "freeze_entry")
        need(item["path"] == expected and type(item["bytes"]) is int
             and 0 <= item["bytes"] <= MAX_RESPONSE and sha(item["sha256"], 64), "freeze_entry")
        path = regular_file(root, expected)
        need(path.stat().st_size == item["bytes"], "freeze_file_size")
        need(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "freeze_file_digest")
        if expected.endswith("/git_claim_qualification.py"):
            need(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == item["sha256"], "running_code_digest")


class Budget:
    """Shared counters use bounded explicit locks; dead children cannot deadlock cleanup."""
    def __init__(self, ctx, deadline, total_deadline=None):
        self.total_deadline = deadline + CLEANUP_SECONDS if total_deadline is None else total_deadline
        self.values = ctx.Array("q", 10, lock=False)
        self.mutex = ctx.RLock()
        self.deadline = deadline

    @contextmanager
    def locked(self):
        timeout = min(0.5, max(0, self.total_deadline - time.monotonic()))
        need(self.mutex.acquire(timeout=timeout), "budget_lock_unknown")
        try:
            yield
        finally:
            self.mutex.release()

    def remaining(self):
        value = self.deadline - time.monotonic()
        need(value > 0, "global_deadline")
        return value

    def check(self):
        self.remaining()
        need(not self.values[4], "previous_failure")

    def fail(self):
        try:
            with self.locked():
                self.values[4] = 1
        except Closed:
            pass  # A dead child's unreleased lock is itself a closed/unknown state.

    def get(self):
        with self.locked():
            self.check()
            need(self.values[0] < 17 and self.values[0] + self.values[1] < 20, "request_budget")
            self.values[0] += 1

    def reserve_post(self, ticket):
        body = encoded({"ref": TICKETS[ticket][0], "sha": TICKETS[ticket][1]})
        with self.locked():
            self.check()
            need(not self.values[5] & (1 << ticket), "post_already_reserved")
            need(self.values[1] < 3 and self.values[0] + self.values[1] < 20
                 and self.values[2] + len(body) <= MAX_WRITE_BYTES, "write_budget")
            self.values[1] += 1
            self.values[2] += len(body)
            self.values[5] |= 1 << ticket

    def consume_post(self, ticket):
        with self.locked():
            self.check()
            need(self.values[5] & (1 << ticket) and not self.values[6] & (1 << ticket), "post_ticket")
            self.values[6] |= 1 << ticket
            self.values[7] += 1

    def reserve_response(self):
        with self.locked():
            need(self.values[3] + self.values[8] + MAX_RESPONSE <= MAX_RESPONSE_TOTAL, "response_total_cap")
            self.values[8] += MAX_RESPONSE

    def observe(self, count, reserved=False):
        with self.locked():
            need(0 <= count and self.values[3] + self.values[8] + (0 if reserved else count)
                 <= MAX_RESPONSE_TOTAL, "response_total_cap")
            if reserved:
                need(self.values[8] >= count, "response_reservation")
                self.values[8] -= count
            self.values[3] += count

    def release_response(self, count):
        with self.locked():
            need(0 <= count <= self.values[8], "response_reservation")
            self.values[8] -= count

    def metadata(self, count):
        with self.locked():
            # Leave one bounded line available for the terminal summary.
            need(self.values[9] + count <= MAX_METADATA - 4096, "metadata_cap")
            self.values[9] += count

    def snapshot(self):
        keys = ("get_requests", "post_reserved", "write_body_bytes", "response_bytes",
                "failed", "issued_mask", "consumed_mask", "post_dispatched",
                "response_reserved_or_unknown_bytes", "metadata_bytes")
        try:
            with self.locked():
                return dict(zip(keys, self.values[:]))
        except Closed:
            return dict.fromkeys(keys, None)


def emit(budget, event):
    """Only explicit safe projections enter this function; one <=4096-byte POSIX write."""
    raw = encoded(event) + b"\n"
    need(len(raw) <= 4096, "metadata_line_cap")
    budget.metadata(len(raw))
    need(os.write(1, raw) == len(raw), "metadata_write_unknown")


@contextmanager
def request_alarm(seconds):
    def expired(_signum, _frame):
        raise Closed("request_timeout")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class DirectHTTPS:
    """Fixed origin, standard certificate verification, no proxy, redirect or retry path."""
    def request(self, method, path, body, token, budget):
        connection = None
        try:
            timeout = min(MAX_REQUEST_SECONDS, budget.remaining())
            budget.reserve_response()
            with request_alarm(timeout):
                connection = http.client.HTTPSConnection("api.github.com", 443, timeout=timeout,
                                                         context=ssl.create_default_context())
                connection.request(method, path, body=body, headers={
                    "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": API_VERSION, "User-Agent": CAMPAIGN,
                    "Content-Type": "application/json", "Accept-Encoding": "identity"})
                response = connection.getresponse()
                need(response.getheader("Content-Encoding", "identity") == "identity", "response_encoding")
                chunks, length = [], 0
                while True:
                    budget.remaining()
                    room = min(65536, MAX_RESPONSE - length)
                    need(room > 0, "response_cap")
                    chunk = response.read1(room)  # Never hold a shared lock across network I/O.
                    budget.observe(len(chunk), reserved=True)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    length += len(chunk)
                budget.release_response(MAX_RESPONSE - length)
                return response.status, b"".join(chunks)
        except Closed:
            raise
        except Exception:
            raise Closed("transport_unknown") from None
        finally:
            if connection is not None:
                connection.close()


def workflow_endpoint():
    return ROOT + "/actions/workflows/" + CAMPAIGN + ".yml"


def history_endpoint(workflow_id):
    return ROOT + "/actions/workflows/" + str(workflow_id) + "/runs?branch=" + quote(BRANCH, safe="") + "&event=push&per_page=100&page=1"


def ref_endpoint(ref):
    need(ref in (RACE_REF, LOSS_REF), "fixed_ref")
    return ROOT + "/git/ref/" + ref.removeprefix("refs/")


class Client:
    def __init__(self, token, budget, authority, transport=None):
        self.token, self.budget, self.authority = token, budget, authority
        self.transport = DirectHTTPS() if transport is None else transport
        self.workflow_id = None
        self.post_window = {"post_start_ns": None, "post_end_ns": None}

    def request(self, method, path, allowed, ticket=None):
        try:
            if method == "POST":
                need(path == ROOT + "/git/refs" and type(ticket) is int and 0 <= ticket < 3, "fixed_post")
                self.budget.consume_post(ticket)
                body = encoded({"ref": TICKETS[ticket][0], "sha": TICKETS[ticket][1]})
            else:
                need(method == "GET" and path in self.get_paths(), "fixed_get")
                self.budget.get()
                body = None
            if method == "POST":
                emit(self.budget, {"event": "api_attempt", "method": "POST", "pid": os.getpid(),
                     "ref": TICKETS[ticket][0], "target": TICKETS[ticket][1]})
                # Start after audit output: logging cannot manufacture interval overlap.
                self.post_window["post_start_ns"] = time.monotonic_ns()
            try:
                status, raw = self.transport.request(method, path, body, self.token, self.budget)
            finally:
                if method == "POST":
                    self.post_window["post_end_ns"] = time.monotonic_ns()
            need(type(status) is int and 100 <= status <= 599 and type(raw) is bytes
                 and len(raw) <= MAX_RESPONSE, "transport_result")
            emit(self.budget, {"event": "api_response", "method": method, "path": path,
                 "status": status, "body_bytes": len(raw), "body_sha256": hashlib.sha256(raw).hexdigest(),
                 "pid": os.getpid(), **(self.post_window if method == "POST" else {})})
            need(self.token.encode() not in raw, "response_secret_rejected")
            need(status in allowed, "http_status_rejected")
            return status, raw, strict_json(raw)
        except Closed as error:
            self.budget.fail()
            try:
                emit(self.budget, {"event": "api_stopped", "method": method, "reason": str(error),
                     "pid": os.getpid(), **(self.post_window if method == "POST" else {})})
            except Closed:
                pass
            raise
        except Exception:
            self.budget.fail()
            raise Closed("request_unknown") from None

    def get_paths(self):
        a = self.authority
        result = {workflow_endpoint(), ROOT + "/actions/runs/" + str(a["run_id"]),
                  ROOT + "/actions/runs/" + str(a["run_id"]) + "/attempts/1/jobs?per_page=100",
                  *(ROOT + "/git/commits/" + value for value in (a["source_sha"], TARGET_A, TARGET_B)),
                  ref_endpoint(RACE_REF), ref_endpoint(LOSS_REF)}
        if self.workflow_id is not None:
            result.add(history_endpoint(self.workflow_id))
        return result

    def get(self, path, allowed=(200,)):
        return self.request("GET", path, allowed)


def validate_ref(data, ref, target=None):
    need(type(data) is dict and data.get("ref") == ref and type(data.get("object")) is dict,
         "ref_identity")
    obj = data["object"]
    need(obj.get("type") == "commit" and obj.get("sha") in (TARGET_A, TARGET_B), "ref_object")
    need(target is None or obj["sha"] == target, "ref_target")
    return obj["sha"]


def validate_commit(data, expected_sha, expected_tree, parents=None):
    need(type(data) is dict and data.get("sha") == expected_sha
         and type(data.get("tree")) is dict and data["tree"].get("sha") == expected_tree, "commit_identity")
    rows = data.get("parents")
    need(type(rows) is list and len(rows) <= 16 and all(type(p) is dict and sha(p.get("sha")) for p in rows), "commit_parents")
    if parents is not None:
        need(tuple(p["sha"] for p in rows) == parents, "fixture_parents")


def validate_run(row, authority, workflow_id, current=False):
    need(type(row) is dict, "run_schema")
    fixed = {"workflow_id": workflow_id, "path": WORKFLOW, "head_branch": BRANCH,
             "event": "push", "head_sha": authority["source_sha"]}
    if current:
        fixed.update(id=authority["run_id"], run_number=1, run_attempt=1, status="in_progress", conclusion=None)
    for key, value in fixed.items():
        need(key in row and type(row[key]) is type(value) and row[key] == value, "run_identity")
    need(positive(row.get("id")) and positive(row.get("run_number")) and positive(row.get("run_attempt")), "run_ordinals")
    for key in ("repository", "head_repository"):
        repo = row.get(key)
        need(type(repo) is dict and repo.get("full_name") == REPOSITORY and repo.get("private") is False, "run_repository")


def history(client):
    _, _, data = client.get(history_endpoint(client.workflow_id))
    need(type(data) is dict and type(data.get("total_count")) is int
         and type(data.get("workflow_runs")) is list, "history_schema")
    rows = data["workflow_runs"]
    need(1 <= data["total_count"] == len(rows) <= 100, "history_incomplete")
    for row in rows:
        validate_run(row, client.authority, client.workflow_id)
    ids = [row["id"] for row in rows]
    ordinals = [row["run_number"] for row in rows]
    need(len(set(ordinals)) == len(ordinals), "history_duplicate_ordinal")
    need(len(set(ids)) == len(ids) and ids.count(client.authority["run_id"]) == 1, "history_current")
    current = rows[ids.index(client.authority["run_id"])]
    validate_run(current, client.authority, client.workflow_id, current=True)
    emit(client.budget, {"event": "history", "total_count": data["total_count"]})
    for row in rows:
        emit(client.budget, {"event": "history_row", "run_event": "push", **{key: row[key] for key in
             ("id", "workflow_id", "path", "head_branch", "head_sha", "run_number", "run_attempt")}})
    return data["total_count"]


def preflight(client):
    _, _, wf = client.get(workflow_endpoint())
    need(type(wf) is dict and positive(wf.get("id")) and wf.get("path") == WORKFLOW
         and wf.get("state") == "active", "workflow_identity")
    client.workflow_id = wf["id"]
    client.authority["workflow_id"] = wf["id"]
    a = client.authority
    _, _, run = client.get(ROOT + "/actions/runs/" + str(a["run_id"]))
    validate_run(run, a, wf["id"], current=True)
    _, _, data = client.get(ROOT + "/actions/runs/" + str(a["run_id"]) + "/attempts/1/jobs?per_page=100")
    need(type(data) is dict and type(data.get("total_count")) is int and data["total_count"] == 1
         and type(data.get("jobs")) is list and len(data["jobs"]) == 1, "job_count")
    job = data["jobs"][0]
    need(type(job) is dict and positive(job.get("id")), "job_identity")
    for key, value in {"name": JOB, "run_id": a["run_id"], "run_attempt": 1,
                       "head_sha": a["source_sha"], "status": "in_progress", "conclusion": None}.items():
        need(key in job and type(job[key]) is type(value) and job[key] == value, "job_identity")
    client.authority["job_id"] = job["id"]
    emit(client.budget, {"event": "validated_identity", **client.authority,
         "repository": REPOSITORY, "workflow_path": WORKFLOW, "source_branch": BRANCH,
         "run_number": 1, "run_attempt": 1, "job": JOB})
    count = history(client)
    _, _, source = client.get(ROOT + "/git/commits/" + a["source_sha"])
    validate_commit(source, a["source_sha"], a["source_tree"])
    for target in (TARGET_A, TARGET_B):
        _, _, data = client.get(ROOT + "/git/commits/" + target)
        validate_commit(data, target, FIXTURE_TREE, FIXTURE_PARENTS[target])
    for ref in (RACE_REF, LOSS_REF):
        client.get(ref_endpoint(ref), allowed=(404,))
    return count


def save(path, raw):
    with open(path, "xb") as output:
        os.chmod(path, 0o600)
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def child_main(mode, owner, ticket, directory, budget, authority, transport, barrier, token_pipe):
    """Fresh children decide locally; the parent never elects a winner for them."""
    result = {"pid": os.getpid(), "owner": owner, "permit": False, "witness": False,
              "status": None, "offered_target": TICKETS[ticket][1], "reason": "child_unknown", "clean_environment": "GITHUB_TOKEN" not in os.environ,
              "post_start_ns": None, "post_end_ns": None}
    client = None
    try:
        need(result["clean_environment"], "child_token_environment")
        token = token_pipe.recv()
        token_pipe.close()
        client = Client(token, budget, authority, transport)
        if mode == "reentry":
            ref = LOSS_REF if ticket == 2 else RACE_REF
            _, _, data = client.get(ref_endpoint(ref))
            validate_ref(data, ref)
            result.update(status=200, reason="existing_ref_no_permit")
        else:
            if barrier is not None:
                barrier.wait(timeout=budget.remaining())
            status, raw, data = client.request("POST", ROOT + "/git/refs", (201, 409, 422), ticket)
            result["status"] = status
            result.update(client.post_window)
            if status == 201:
                validate_ref(data, TICKETS[ticket][0], TICKETS[ticket][1])
                emit(budget, {"event": "validated_create_201", "pid": os.getpid(),
                     "owner": owner, "ref": TICKETS[ticket][0], "target": TICKETS[ticket][1]})
            save(Path(directory) / (owner + ".response.json"), raw)
            if mode == "loss":
                need(status == 201, "loss_create_failed")
                # Keep exact observed public 201, discard application acknowledgment, exit BEFORE permit.
                save(Path(directory) / "loss-injection.json", encoded({
                    "status": 201, "pid": os.getpid(), "mode": "injected_response_loss_plus_kill",
                    "response_sha256": hashlib.sha256(raw).hexdigest(), "response_bytes": len(raw),
                    "permit": False, "witness": False, **client.post_window}))
                emit(budget, {"event": "loss_injection", "pid": os.getpid(), "status": 201,
                     "ref": LOSS_REF, "target": TARGET_A, "mode": "injected_response_loss_plus_kill",
                     "planned_exit_code": 86, "permit": False, "witness": False, **client.post_window})
                os._exit(86)
            if status == 201:
                result.update(permit=True, reason="fresh_201_only")
                save(Path(directory) / (owner + ".witness.json"), encoded({
                    "pid": os.getpid(), "owner": owner, "target": TICKETS[ticket][1],
                    "kind": "synthetic_downstream_witness"}))
                result["witness"] = True
            else:
                result["reason"] = "create_rejected_no_permit"
    except Closed as error:
        budget.fail()
        result["reason"] = str(error)
    except BaseException:
        budget.fail()
        result["reason"] = "child_unknown"
    if client is not None:
        result.update(client.post_window)
    try:
        save(Path(directory) / (owner + ".result.json"), encoded(result))
        emit(budget, {"event": "child_result", **result})
    except BaseException:
        budget.fail()


def children(ctx, client, directory, specifications, barrier=False):
    processes = []
    gate = ctx.Barrier(len(specifications)) if barrier else None
    try:
        for mode, owner, ticket in specifications:
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(target=child_main, args=(mode, owner, ticket, str(directory),
                client.budget, client.authority, client.transport, gate, receiver))
            process.start()
            receiver.close()
            sender.send(client.token)
            sender.close()
            processes.append((process, mode, owner))
        for process, mode, owner in processes:
            process.join(timeout=client.budget.remaining())
            need(not process.is_alive(), "child_deadline")
            need(process.exitcode == (86 if mode == "loss" else 0), "child_exit")
            emit(client.budget, {"event": "child_exit", "pid": process.pid,
                 "owner": owner, "mode": mode, "exit_code": process.exitcode})
        client.budget.check()
        return [strict_json((directory / (owner + ".result.json")).read_bytes())
                for _, mode, owner in processes if mode != "loss"]
    finally:
        for process, _, _ in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=min(2, max(0, client.budget.total_deadline - time.monotonic())))
                if process.is_alive():
                    process.kill()
                    process.join(timeout=min(2, max(0, client.budget.total_deadline - time.monotonic())))


def scenarios(ctx, client, directory):
    initial_count = preflight(client)
    for ticket in (0, 1):
        client.budget.reserve_post(ticket)
    race = children(ctx, client, directory, (("race", "A", 0), ("race", "B", 1)), barrier=True)
    need(len({item["pid"] for item in race}) == 2, "race_process_identity")
    winners = [item for item in race if item["status"] == 201 and item["permit"] and item["witness"]]
    losers = [item for item in race if item["status"] in (409, 422) and not item["permit"] and not item["witness"]]
    need(len(winners) == len(losers) == 1, "race_not_one_winner")
    for item in race:
        need(positive(item.get("post_start_ns")) and positive(item.get("post_end_ns"))
             and item["post_start_ns"] < item["post_end_ns"], "race_call_window_unknown")
    overlap_ns = min(item["post_end_ns"] for item in race) - max(item["post_start_ns"] for item in race)
    need(overlap_ns > 0, "qualification_not_concurrent")
    emit(client.budget, {"event": "race_call_overlap", "overlap_ns": overlap_ns,
         "scope": "client_transport_calls_including_tls"})
    winner = TARGET_A if winners[0]["owner"] == "A" else TARGET_B
    _, _, ref = client.get(ref_endpoint(RACE_REF))
    validate_ref(ref, RACE_REF, winner)
    emit(client.budget, {"event": "validated_ref", "ref": RACE_REF, "target": winner, "phase": "race"})
    winner_ticket = 0 if winner == TARGET_A else 1
    for owner, ticket in (("same-owner-reentry", winner_ticket), ("wrong-owner-reentry", 1 - winner_ticket)):
        result = children(ctx, client, directory, (("reentry", owner, ticket),))[0]
        need(result["reason"] == "existing_ref_no_permit" and not result["permit"] and not result["witness"], "reentry_admitted")
    client.budget.reserve_post(2)
    children(ctx, client, directory, (("loss", "loss-A", 2),))
    injected = strict_json((directory / "loss-injection.json").read_bytes())
    need(injected["status"] == 201 and not injected["permit"] and not injected["witness"], "loss_injection")
    need(not (directory / "loss-A.witness.json").exists(), "loss_witness")
    _, _, ref = client.get(ref_endpoint(LOSS_REF))
    validate_ref(ref, LOSS_REF, TARGET_A)
    emit(client.budget, {"event": "validated_ref", "ref": LOSS_REF, "target": TARGET_A, "phase": "loss"})
    result = children(ctx, client, directory, (("reentry", "loss-same-owner-reentry", 2),))[0]
    need(not result["permit"] and not result["witness"], "loss_reentry_admitted")
    final_count = history(client)
    need(final_count >= initial_count, "history_regressed")
    for ref, target in ((RACE_REF, winner), (LOSS_REF, TARGET_A)):
        _, _, data = client.get(ref_endpoint(ref))
        validate_ref(data, ref, target)
        emit(client.budget, {"event": "validated_ref", "ref": ref, "target": target, "phase": "final"})
    return {"history_initial_total": initial_count, "history_final_total": final_count,
            "race_winner": "A" if winner == TARGET_A else "B", "race_permits": 1,
            "race_call_overlap_ns": overlap_ns,
            "race_witnesses": 1, "loss_permits": 0, "loss_witnesses": 0,
            "loss_mode": "injected_response_loss_plus_kill", "loss_exit_code": 86}


def execute(*, control_path, checkout_sha, checkout_tree, source_freeze,
            source_freeze_sha, checkout_root=".", env=None, transport=None):
    """transport is an internal test seam, never selectable through CLI/control/environment."""
    env = os.environ if env is None else env
    result = {"schema_version": 1, "status": "failed", "reason": "local_guard_unknown",
              "get_requests": 0, "post_reserved": 0, "post_dispatched": 0,
              "write_body_bytes": 0, "response_bytes": 0}
    budget = None
    directory = None
    authority = None
    started = time.monotonic()
    try:
        if env.get("GITHUB_RUN_NUMBER") != "1" or env.get("GITHUB_RUN_ATTEMPT") != "1":
            result.update(status="skipped", reason="ordinal_or_attempt_ineligible")
            return result
        # No access to GITHUB_TOKEN before all local validation and exclusive invocation guard.
        verify_freeze(checkout_root, source_freeze, source_freeze_sha, control_path)
        control = strict_json(Path(control_path).read_bytes())
        if not validate_control(control):
            result.update(status="skipped", reason="control_disabled")
            return result
        expected = {"GITHUB_REPOSITORY": REPOSITORY, "GITHUB_REF": "refs/heads/" + BRANCH,
                    "GITHUB_EVENT_NAME": "push", "GITHUB_JOB": JOB, "GITHUB_ACTIONS": "true",
                    "GITHUB_WORKFLOW_REF": REPOSITORY + "/" + WORKFLOW + "@refs/heads/" + BRANCH}
        need(all(env.get(key) == value for key, value in expected.items()), "environment_identity")
        need(sha(checkout_sha) and sha(checkout_tree) and env.get("GITHUB_SHA") == checkout_sha
             and env.get("GITHUB_WORKFLOW_SHA") == checkout_sha, "checkout_identity")
        run_id = env.get("GITHUB_RUN_ID", "")
        need(type(run_id) is str and re.fullmatch("[1-9][0-9]{0,19}", run_id) is not None, "run_id")
        temporary = Path(env.get("RUNNER_TEMP", ""))
        need(temporary.is_absolute() and temporary.is_dir() and not temporary.is_symlink(), "runner_temp")
        directory = temporary / (CAMPAIGN + "-run-" + run_id)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            raise Closed("invocation_already_exists") from None
        token = env.pop("GITHUB_TOKEN", None)
        # Also clean the real process environment when a test supplied a separate mapping.
        os.environ.pop("GITHUB_TOKEN", None)
        need(type(token) is str and 8 <= len(token) <= 4096 and not any(c.isspace() for c in token), "token_unavailable")
        ctx = mp.get_context("spawn")
        budget = Budget(ctx, started + MAX_WORK_SECONDS, started + MAX_SECONDS)
        authority = {"run_id": int(run_id), "source_sha": checkout_sha, "source_tree": checkout_tree,
                     "source_freeze_sha256": source_freeze_sha}
        client = Client(token, budget, authority, transport)
        details = scenarios(ctx, client, directory)
        result.update(status="passed", reason="qualification_observed", **details)
    except Closed as error:
        result["reason"] = str(error)
        if budget is not None:
            budget.fail()
    except BaseException:
        result["reason"] = "unexpected_failure"
        if budget is not None:
            budget.fail()
    finally:
        if budget is not None:
            snapshot = budget.snapshot()
            if authority is not None:
                result.update(**authority, repository=REPOSITORY, workflow_path=WORKFLOW,
                              source_branch=BRANCH, run_number=1, run_attempt=1, job=JOB)
            for key in ("get_requests", "post_reserved", "post_dispatched", "write_body_bytes", "response_bytes",
                        "response_reserved_or_unknown_bytes", "metadata_bytes"):
                result[key] = snapshot[key]
            result["work_window_seconds"] = MAX_WORK_SECONDS
            result["total_window_seconds"] = MAX_SECONDS
            result["metadata_limit_bytes"] = MAX_METADATA
            result["terminal_summary_max_bytes"] = 4096
        if directory is not None and directory.is_dir() and result["reason"] != "invocation_already_exists":
            try:
                save(directory / "summary.json", encoded(result))
            except BaseException:
                result.update(status="failed", reason="summary_write_failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--checkout-sha", required=True)
    parser.add_argument("--checkout-tree", required=True)
    parser.add_argument("--source-freeze", required=True)
    parser.add_argument("--source-freeze-sha", required=True)
    parser.add_argument("--checkout-root", default=".")
    args = parser.parse_args()
    result = execute(control_path=args.control, checkout_sha=args.checkout_sha,
                     checkout_tree=args.checkout_tree, source_freeze=args.source_freeze,
                     source_freeze_sha=args.source_freeze_sha, checkout_root=args.checkout_root)
    raw = encoded({"event": "summary", **result}) + b"\n"
    if len(raw) > 4096:
        raw = encoded({"event": "summary", "status": "failed", "reason": "summary_cap"}) + b"\n"
    os.write(1, raw)
    return 0 if result["status"] in ("passed", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
