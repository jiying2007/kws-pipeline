"""Disabled trusted-host K1 ASR18 same-repository publisher; no inference imports.

Only explicit mock transports are used in tests. CLI gates fixed disabled source-local controls before reading any token. Actual
source/workflow/release activation is separate and has not been authorized here.
"""
import base64
from datetime import datetime, timezone
import http.client
import os
from pathlib import Path
import re
import signal
import ssl
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import evidence_transport as t
import campaign_contract as c
import failure_observer


def _observe(observer, stage):
    if observer is not None:
        try: observer.stage(stage)
        except BaseException: pass  # Diagnostic failure cannot truncate publication or replace its original failure.

BASE = "/repos/" + t.REPOSITORY
MAX_REQUEST = 512 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_API_REQUESTS = 64
MAX_API_BODY = 8 * 1024 * 1024
MAX_API_UPLOAD = 18 * 1024 * 1024
TIMEOUT = 180
AUTHOR = {"name": "K1 ASR18 bounded publisher", "email": "noreply@github.com"}


class PublicationError(RuntimeError):
    def __init__(self, status=None):
        self.status = status if type(status) is int else None
        super().__init__("publication failed or unconfirmed; no retry")


class PublicationDisabled(ValueError):
    """Only the fixed source-local approval gate emits this safe CLI reason."""


def validate_release(release):
    t.require(type(release) is dict and set(release) == {"schema", "purpose", "source_sha", "source_tree",
        "source_closure_sha256", "prior_claim_history_sha256", "commit_timestamp"}, "reviewed detached release fields")
    t.require(release["schema"] == "k1-asr18-publisher-bound-release-v1", "release schema")
    t.policy(release["purpose"])
    for key in ("source_sha", "source_tree"):
        t.require(type(release[key]) is str and t.HEAD.fullmatch(release[key]), "reviewed source Git binding")
    for key in ("source_closure_sha256", "prior_claim_history_sha256"):
        t.require(type(release[key]) is str and t.SHA.fullmatch(release[key]), "reviewed source/history binding")
    stamp = release["commit_timestamp"]
    t.require(type(stamp) is str and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", stamp), "fixed commit timestamp")
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def context(env, event, release):
    validate_release(release)
    p = t.policy(release["purpose"])
    t.require(type(env) is dict and type(event) is dict and type(event.get("repository")) is dict, "push context")
    t.require(env.get("GITHUB_REPOSITORY") == t.REPOSITORY
        and event["repository"].get("full_name") == t.REPOSITORY
        and event["repository"].get("private") is False, "canonical public repository")
    t.require(env.get("GITHUB_EVENT_NAME") == "push" and event.get("created") is True
        and event.get("deleted") is False and event.get("forced") is False and event.get("before") == "0" * 40, "fresh push-created branch only")
    t.require(env.get("GITHUB_REF") == "refs/heads/" + p["source_branch"]
        and event.get("ref") == env["GITHUB_REF"], "fixed unique source ref")
    t.require(env.get("GITHUB_RUN_ATTEMPT") == "1" and type(env.get("GITHUB_RUN_ID")) is str
        and re.fullmatch(r"[1-9][0-9]{0,19}", env["GITHUB_RUN_ID"]), "actual run attempt one")
    value = {"repository": t.REPOSITORY, "run_id": int(env["GITHUB_RUN_ID"]), "run_attempt": 1,
        "job_key": env.get("GITHUB_JOB"), "executed_head_sha": env.get("GITHUB_SHA")}
    t.identity(value, release["purpose"])
    t.require(env.get("GITHUB_WORKFLOW_REF") == t.REPOSITORY + "/" + p["workflow"] + "@refs/heads/" + p["source_branch"]
        and env.get("GITHUB_WORKFLOW_SHA") == release["source_sha"], "same exact executed workflow source")
    t.require(value["executed_head_sha"] == release["source_sha"]
        and event.get("after") == release["source_sha"], "reviewed executed push SHA")
    return value


def endpoints(identity, release):
    p = t.policy(release["purpose"])
    run = str(identity["run_id"])
    return {"repo": BASE, "run": BASE + "/actions/runs/" + run,
        "jobs": BASE + "/actions/runs/" + run + "/attempts/1/jobs?per_page=100",
        "history": BASE + "/actions/workflows/" + p["workflow"].split("/")[-1] + "/runs?event=push&branch=" + p["source_branch"] + "&per_page=100",
        "source": BASE + "/git/commits/" + release["source_sha"],
        "source_tree": BASE + "/git/trees/" + release["source_tree"] + "?recursive=1",
        "ref": BASE + "/git/ref/heads/" + p["result_branch"]}


class GitHubAPI:
    """Fixed API TLS host/routes, count/body caps, finite timeout, no retry/redirect.

    Token must be supplied only by a future reviewed trusted publisher step.
    This class is NOT wired to CLI, environment lookup or an active workflow.
    """
    def __init__(self, token, identity, release):
        validate_release(release); t.identity(identity, release["purpose"])
        t.require(type(token) is str and token.startswith("ghs_") and 5 <= len(token) <= 4096 and "\r" not in token and "\n" not in token,
                  "publisher-only short-lived token")
        self._token = token
        self.release = t.decode(t.canonical(release))
        self.allowed = {("GET", route) for route in endpoints(identity, release).values()}
        self.allowed |= {("POST", BASE + "/git/" + noun) for noun in ("blobs", "trees", "commits", "refs")}
        self.deadline = time.monotonic() + TIMEOUT
        self.count = 0; self.body_bytes = 0; self.request_bytes = 0; self.seen = set(); self.posts = set()

    def bind_result(self, commit_sha, tree_sha):
        for value in (commit_sha, tree_sha):
            t.require(type(value) is str and t.HEAD.fullmatch(value), "locally computed immutable Git object")
        t.require(not hasattr(self, "result"), "result binding once")
        self.result = (commit_sha, tree_sha)
        self.allowed |= {("GET", BASE + "/git/commits/" + commit_sha),
                         ("GET", BASE + "/git/trees/" + tree_sha + "?recursive=1")}

    def __call__(self, method, route, body=None, expected=200):
        t.require((method, route) in self.allowed, "fixed method/route only")
        t.require(type(expected) is int and expected in (200, 201, 404), "HTTP status policy")
        t.require((method == "POST" and expected == 201 and type(body) is dict)
                  or (method == "GET" and expected in (200, 404) and body is None), "HTTP method/body/status")
        raw = None if body is None else t.canonical(body)
        t.require(raw is None or len(raw) <= MAX_REQUEST, "request body cap")
        t.require(self.request_bytes + len(raw or b"") <= MAX_API_UPLOAD, "total API upload body cap")
        if method == "POST" and route != BASE + "/git/blobs":
            t.require(route not in self.posts, "create-only object attempted once")
            if route == BASE + "/git/refs":
                t.require(hasattr(self, "result") and body == {"ref": "refs/heads/" + t.policy(self.release["purpose"])["result_branch"],
                    "sha": self.result[0]}, "only exact reviewed result ref and computed commit")
            elif route == BASE + "/git/commits":
                t.require(hasattr(self, "result") and body == commit_body(self.result[1], self.release)[0], "only computed result commit")
            elif route == BASE + "/git/trees":
                t.require(set(body) == {"base_tree", "tree"} and body["base_tree"] == self.release["source_tree"]
                    and type(body["tree"]) is list and 2 <= len(body["tree"]) <= 49, "bounded source-preserving result tree")
                names = [row.get("path") for row in body["tree"] if type(row) is dict]
                allowed = {t.manifest_path(self.release["purpose"])} | {t.chunk_path(i, self.release["purpose"]) for i in range(t.MAX_CHUNKS)}
                t.require(len(names) == len(body["tree"]) == len(set(names)) and set(names) <= allowed
                    and t.manifest_path(self.release["purpose"]) in names, "only bounded result paths")
                for row in body["tree"]:
                    t.require(set(row) == {"path", "mode", "type", "sha"} and row["mode"] == "100644" and row["type"] == "blob"
                        and type(row["sha"]) is str and t.HEAD.fullmatch(row["sha"]), "exact result tree rows")
        # Identical request cannot be attempted again, including uncertain failures.
        request = (method, route, t.sha(raw or b""), expected)
        t.require(request not in self.seen, "request already consumed")
        remaining = self.deadline - time.monotonic()
        t.require(remaining > 0 and self.count < MAX_API_REQUESTS, "API deadline/count cap")
        self.seen.add(request); self.count += 1; self.request_bytes += len(raw or b"")
        if method == "POST" and route != BASE + "/git/blobs": self.posts.add(route)
        connection = http.client.HTTPSConnection("api.github.com", timeout=min(15, remaining), context=ssl.create_default_context())
        try:
            connection.request(method, route, body=raw, headers={
                "Authorization": "Bearer " + self._token, "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "k1-asr18-bounded-evidence-publisher",
                "Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read(min(MAX_RESPONSE, MAX_API_BODY - self.body_bytes) + 1)
            self.body_bytes += len(body)
            if len(body) > MAX_RESPONSE or self.body_bytes > MAX_API_BODY or response.status != expected:
                raise PublicationError(response.status)
            if expected == 404: return None
            value = t.decode(body)
            t.require(type(value) is dict, "API object response")
            return value
        except PublicationError:
            raise
        except Exception:
            raise PublicationError() from None
        finally:
            connection.close()


class RawReader:
    """Public immutable raw-content only; at most 49 paths once, 16 MiB bodies."""
    def __init__(self, commit_sha, purpose):
        t.require(type(commit_sha) is str and t.HEAD.fullmatch(commit_sha), "immutable result commit")
        t.policy(purpose)
        self.prefix = "/" + t.REPOSITORY + "/" + commit_sha + "/"
        self.allowed = {t.manifest_path(purpose): t.MAX_MANIFEST,
                        **{t.chunk_path(i, purpose): t.MAX_CHUNK for i in range(t.MAX_CHUNKS)}}
        self.seen = set(); self.body_bytes = 0; self.deadline = time.monotonic() + TIMEOUT

    def __call__(self, name):
        t.require(name in self.allowed and name not in self.seen and len(self.seen) < 49, "fixed raw path once")
        remaining = self.deadline - time.monotonic()
        t.require(remaining > 0, "raw total deadline")
        self.seen.add(name)
        connection = http.client.HTTPSConnection("raw.githubusercontent.com", timeout=min(15, remaining), context=ssl.create_default_context())
        try:
            connection.request("GET", self.prefix + name, headers={"User-Agent": "k1-asr18-bounded-evidence-reader"})
            response = connection.getresponse()
            raw = response.read(min(self.allowed[name], t.MAX_BODY - self.body_bytes) + 1)
            self.body_bytes += len(raw)
            if response.status != 200 or len(raw) > self.allowed[name] or self.body_bytes > t.MAX_BODY:
                raise PublicationError(response.status)
            return raw
        except PublicationError:
            raise
        except Exception:
            raise PublicationError() from None
        finally:
            connection.close()


def _equal(record, expected, label):
    t.require(type(record) is dict, label)
    for key, value in expected.items():
        t.require(type(record.get(key)) is type(value) and record[key] == value, label)


def verify_run(api, identity, release):
    routes = endpoints(identity, release); p = t.policy(release["purpose"])
    repo = api("GET", routes["repo"])
    _equal(repo, {"full_name": t.REPOSITORY, "private": False}, "API canonical public repository")
    run = api("GET", routes["run"])
    expected = {"id": identity["run_id"], "run_attempt": 1, "head_sha": release["source_sha"],
        "head_branch": p["source_branch"], "event": "push", "path": p["workflow"], "status": "in_progress"}
    _equal(run, expected, "actual run binding")
    _equal(run.get("repository"), {"full_name": t.REPOSITORY, "private": False}, "actual run repository")
    history = api("GET", routes["history"])
    _equal(history, {"total_count": 1}, "exact one run for fresh source ref")
    t.require(type(history.get("workflow_runs")) is list and len(history["workflow_runs"]) == 1, "complete one-use history")
    _equal(history["workflow_runs"][0], expected, "history actual run binding")
    jobs = api("GET", routes["jobs"])
    t.require(type(jobs.get("total_count")) is int and 1 <= jobs["total_count"] <= 100
        and type(jobs.get("jobs")) is list and len(jobs["jobs"]) == jobs["total_count"], "bounded complete jobs")
    matches = [job for job in jobs["jobs"] if type(job) is dict and job.get("name") == p["job_key"]]
    t.require(len(matches) == 1, "one nonmatrix job")
    job = matches[0]
    _equal(job, {"run_id": identity["run_id"], "head_sha": release["source_sha"], "head_branch": p["source_branch"],
        "run_url": "https://api.github.com" + routes["run"], "status": "in_progress", "conclusion": None}, "actual numeric job binding")
    if "run_attempt" in job: _equal(job, {"run_attempt": 1}, "job attempt one")
    t.require(type(job.get("id")) is int and job["id"] > 0, "numeric job ID")
    return job["id"]


def verify_tree(value, expected_sha):
    t.require(type(value) is dict and value.get("sha") == expected_sha and value.get("truncated") is False,
              "complete exact tree response")
    rows = value.get("tree")
    t.require(type(rows) is list and len(rows) <= 4096, "bounded recursive tree")
    flat = []; directories = {}; names = set()
    for row in rows:
        t.require(type(row) is dict, "tree entry")
        name = t.path(row.get("path"))
        t.require(name not in names and type(row.get("sha")) is str and t.HEAD.fullmatch(row["sha"]), "unique tree entry")
        names.add(name)
        if row.get("type") == "tree":
            t.require(row.get("mode") == "040000", "tree mode")
            directories[name] = row["sha"]
        else:
            flat.append({k: row.get(k) for k in ("path", "mode", "type", "sha")})
    measured, measured_directories = t.tree_identity(flat)
    t.require(measured == expected_sha and measured_directories == directories, "real nested Git tree identity")
    return flat


def commit_body(tree_sha, release):
    author = {**AUTHOR, "date": release["commit_timestamp"]}
    message = "Retain bounded K1 ASR18 audio-free evidence\n"
    body = {"message": message, "tree": tree_sha, "parents": [release["source_sha"]],
            "author": author, "committer": author}
    timestamp = int(datetime.strptime(release["commit_timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    person = AUTHOR["name"] + " <" + AUTHOR["email"] + "> " + str(timestamp) + " +0000"
    raw = ("tree " + tree_sha + "\nparent " + release["source_sha"] + "\nauthor " + person
           + "\ncommitter " + person + "\n\n" + message).encode("utf-8")
    return body, t.git_sha(raw, "commit")


def verify_commit(value, expected_sha, expected_body):
    _equal(value, {"sha": expected_sha}, "true precomputed commit identity")
    message = expected_body["message"]
    # GitHub's API GET may omit this fixed single terminal LF in its presentation.
    # The exact precomputed Git object SHA still binds the original commit bytes.
    t.require(type(message) is str and message.endswith("\n") and not message.endswith("\n\n"),
              "fixed commit message terminal LF")
    t.require(value.get("message") in (message, message[:-1]), "exact commit message API presentation")
    _equal(value.get("tree"), {"sha": expected_body["tree"]}, "commit tree")
    t.require(type(value.get("parents")) is list and len(value["parents"]) == 1, "single source parent")
    _equal(value["parents"][0], {"sha": expected_body["parents"][0]}, "source commit parent")
    for key in ("author", "committer"):
        _equal(value.get(key), expected_body[key], "exact commit metadata")



_TRANSFER_STAGES = frozenset(("validate_local_package", "verify_actual_run_source", "verify_destination_absent",
    "create_blobs", "create_tree", "create_commit", "create_ref", "immutable_postpublish_readback", "readback_verified"))
_TRANSFER_STATUSES = frozenset(("upload_plan_prepared", "publication_failed_or_unconfirmed_no_retry", "same_identity_readback_verified"))


def transfer_record(identity, progress, status, error=None):
    """Fixed public metadata only. An unconfirmed POST may have written its object.

    This is transfer observation, never recognition or recovery success. Source
    bytes and planned blobs are already validated before upload_plan_prepared.
    """
    t.identity(identity)
    t.require(type(progress) is dict and status in _TRANSFER_STATUSES, "fixed transfer observation")
    stage = progress.get("stage")
    t.require(stage in _TRANSFER_STAGES, "fixed transfer stage")
    rows = progress.get("planned_blobs")
    if rows is not None:
        t.require(type(rows) is list and 2 <= len(rows) <= t.MAX_CHUNKS + 1, "bounded planned blobs")
        allowed = {t.manifest_path()} | {t.chunk_path(i) for i in range(t.MAX_CHUNKS)}
        names = []; copied = []
        for row in rows:
            t.require(type(row) is dict and set(row) == {"path", "bytes", "sha256", "git_blob_sha"}, "exact planned blob fields")
            name = row["path"]
            t.require(type(name) is str and name in allowed, "fixed planned blob path")
            cap = t.MAX_MANIFEST if name == t.manifest_path() else t.MAX_CHUNK
            t.require(type(row["bytes"]) is int and 0 < row["bytes"] <= cap, "planned blob bytes")
            t.require(type(row["sha256"]) is str and t.SHA.fullmatch(row["sha256"]) and
                type(row["git_blob_sha"]) is str and t.HEAD.fullmatch(row["git_blob_sha"]), "planned blob digests")
            names.append(name); copied.append(dict(row))
        t.require(names == sorted(set(names)) and t.manifest_path() in names, "complete ordered planned paths")
        chunks = [name for name in names if name != t.manifest_path()]
        t.require(chunks == [t.chunk_path(i) for i in range(len(chunks))], "contiguous planned chunks")
        rows = copied
    numeric = {}
    for key in ("attempted_blob_count", "confirmed_blob_count"):
        value = progress.get(key)
        t.require(value is None or (type(value) is int and rows is not None and 0 <= value <= len({r["git_blob_sha"] for r in rows})), "bounded blob observation count")
        numeric[key] = value
    if numeric["confirmed_blob_count"] is not None:
        t.require(numeric["attempted_blob_count"] is not None and numeric["confirmed_blob_count"] <= numeric["attempted_blob_count"], "confirmed versus attempted blobs")
    digests = {}
    for key, pattern in (("archive_sha256", t.SHA), ("expected_result_commit", t.HEAD), ("expected_result_tree", t.HEAD)):
        value = progress.get(key)
        t.require(value is None or (type(value) is str and pattern.fullmatch(value)), "fixed transfer digest")
        digests[key] = value
    size = progress.get("archive_bytes")
    t.require(size is None or (type(size) is int and 0 < size <= t.MAX_ARCHIVE), "transfer archive bytes")
    if rows is not None:
        t.require(size is not None and all(v is not None for v in digests.values()) and
            sum(row["bytes"] for row in rows if row["path"] != t.manifest_path()) == size, "complete transfer byte plan")
    code = None; http_status = None
    if error is not None:
        code = next((known.__name__ for known in (ValueError, TypeError, OSError, TimeoutError, PublicationError)
            if type(error) is known), "OtherException")
        if type(error) is PublicationError:
            fields = BaseException.__dict__["__dict__"].__get__(error)
            candidate = dict.get(fields, "status")
            if type(candidate) is int and 100 <= candidate <= 599: http_status = candidate
    for key in ("publication_confirmed", "readback_confirmed"):
        t.require(progress.get(key) is None or type(progress[key]) is bool, "transfer confirmation observation")
    return {"schema": "k1-asr18-publisher-transfer-observation-v1", "identity": dict(identity),
        "status": status, "stage": stage, "planned_blobs": rows, **numeric, **digests, "archive_bytes": size,
        "http_status": http_status, "error_type": code,
        "publication_confirmed": progress.get("publication_confirmed"), "readback_confirmed": progress.get("readback_confirmed"),
        "attempted_blob_count_scope": "blob_api_invocations_not_confirmed_http_sends",
        "observation_delivery": "best_effort_stdout",
        "unconfirmed_request_may_have_written": True, "original_http_body_included": False,
        "native_success_established": False, "retry_authorized": False}


def _emit_transfer(observer, identity, progress, status, error=None, directory=None):
    # Diagnostics cannot alter publication, replace its primary failure, or retry.
    if observer is None: return False
    try:
        record = transfer_record(identity, progress, status, error)
        raw = t.canonical(record)
        t.require(len(raw) <= 16 * 1024, "bounded transfer observation")
        persistence = {"status": "not_attempted", "error_type": None}
        if directory is not None:
            leaf = "publication.plan.json" if status == "upload_plan_prepared" else "publication.transfer.json"
            try:
                write_once(directory, leaf, record)
                persistence["status"] = "local_exclusive_write_and_fsync_completed"
            except BaseException as local_error:
                persistence = {"status": "local_write_failed_or_unconfirmed",
                    "error_type": failure_observer.error_metadata(local_error)["error_type"]}
        envelope = {"schema": "k1-asr18-publisher-transfer-envelope-v1", "record": record,
            "record_bytes": len(raw), "record_sha256": t.sha(raw), "persistence": persistence}
        out = t.canonical(envelope)
        t.require(len(out) <= 17 * 1024, "bounded transfer envelope")
        print(out.decode("ascii"), flush=True)
        return True
    except BaseException:
        return False


def publish(env, event, release, payload, api, raw_get, progress=None, observer=None, transfer_directory=None):
    progress = {} if progress is None else progress
    _observe(observer, "validate_local_package")
    progress.update(stage="validate_local_package", publication_confirmed=False, readback_confirmed=False)
    identity = context(env, event, release); purpose = release["purpose"]; p = t.policy(purpose)
    t.validate_archive(payload, identity)
    _observe(observer, "verify_actual_run_source")
    progress["stage"] = "verify_actual_run_source"
    numeric_job_id = verify_run(api, identity, release)
    routes = endpoints(identity, release)
    source = api("GET", routes["source"])
    _equal(source, {"sha": release["source_sha"]}, "reviewed source commit")
    _equal(source.get("tree"), {"sha": release["source_tree"]}, "reviewed source tree")
    original = verify_tree(api("GET", routes["source_tree"]), release["source_tree"])
    t.validate_source_binding(payload, identity, release["source_closure_sha256"], original, numeric_job_id, release["source_tree"])
    directory = p["directory"]
    t.require(not any(row["path"] == directory or row["path"].startswith(directory + "/")
        or directory.startswith(row["path"] + "/") for row in original), "result directory absent; no replacement")
    _observe(observer, "verify_destination_absent")
    progress["stage"] = "verify_destination_absent"
    api("GET", routes["ref"], expected=404)
    files = t.build_file_set(payload, identity, release["source_closure_sha256"], numeric_job_id, purpose)
    tree_rows = [{"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for name, raw in sorted(files.items())]
    expected_tree, _ = t.tree_identity(original + tree_rows)
    expected_body, expected_commit = commit_body(expected_tree, release)
    api.bind_result(expected_commit, expected_tree)
    _observe(observer, "create_blobs")
    progress.update(stage="create_blobs", attempted_blob_count=0, confirmed_blob_count=0, expected_result_commit=expected_commit,
                    expected_result_tree=expected_tree, archive_sha256=t.sha(payload), archive_bytes=len(payload),
                    planned_blobs=[{"path": row["path"], "bytes": len(files[row["path"]]),
                        "sha256": t.sha(files[row["path"]]), "git_blob_sha": row["sha"]} for row in tree_rows])
    _emit_transfer(observer, identity, progress, "upload_plan_prepared", directory=transfer_directory)
    sent_blobs = set()
    for row in tree_rows:
        if row["sha"] not in sent_blobs:
            progress["attempted_blob_count"] += 1
            blob = api("POST", BASE + "/git/blobs", {"content": base64.b64encode(files[row["path"]]).decode("ascii"), "encoding": "base64"}, expected=201)
            _equal(blob, {"sha": row["sha"]}, "true Git blob identity")
            sent_blobs.add(row["sha"])
            progress["confirmed_blob_count"] = len(sent_blobs)
    _observe(observer, "create_tree")
    progress["stage"] = "create_tree"
    tree = api("POST", BASE + "/git/trees", {"base_tree": release["source_tree"], "tree": tree_rows}, expected=201)
    _equal(tree, {"sha": expected_tree}, "true computed merged Git tree")
    _observe(observer, "create_commit")
    progress["stage"] = "create_commit"
    commit = api("POST", BASE + "/git/commits", expected_body, expected=201)
    verify_commit(commit, expected_commit, expected_body)
    _observe(observer, "create_ref")
    progress["stage"] = "create_ref"
    ref = "refs/heads/" + p["result_branch"]
    created = api("POST", BASE + "/git/refs", {"ref": ref, "sha": expected_commit}, expected=201)
    _equal(created, {"ref": ref}, "created exact ref")
    _equal(created.get("object"), {"type": "commit", "sha": expected_commit}, "created ref exact commit")
    _observe(observer, "immutable_postpublish_readback")
    progress.update(stage="immutable_postpublish_readback", publication_confirmed=True)
    final_ref = api("GET", routes["ref"])
    _equal(final_ref, {"ref": ref}, "readback ref")
    _equal(final_ref.get("object"), {"type": "commit", "sha": expected_commit}, "readback commit")
    verify_commit(api("GET", BASE + "/git/commits/" + expected_commit), expected_commit, expected_body)
    actual = verify_tree(api("GET", BASE + "/git/trees/" + expected_tree + "?recursive=1"), expected_tree)
    t.require(sorted(actual, key=lambda row: row["path"]) == sorted(original + tree_rows, key=lambda row: row["path"]), "exact preserved source and new output paths")
    # Factory receives immutable commit and fixed purpose; no arbitrary remote URL.
    read = raw_get(expected_commit, purpose)
    outer = read(t.manifest_path(purpose))
    t.require(outer == files[t.manifest_path(purpose)], "exact outer manifest readback")
    m = t.parse_manifest(outer, identity, release["source_closure_sha256"], numeric_job_id, purpose)
    chunks = [(row["path"], read(row["path"])) for row in m["chunks"]]
    got = t.reassemble(outer, chunks, identity, release["source_closure_sha256"], numeric_job_id, purpose)
    t.require(got == payload, "same complete audio-free evidence archive readback")
    t.validate_archive(got, identity)
    _observe(observer, "readback_verified")
    progress.update(stage="readback_verified", readback_confirmed=True)
    return {"schema": "k1-asr18-evidence-publication-receipt-v1", "status": "same_identity_readback_verified",
        "identity": identity, "purpose": purpose, "numeric_job_id": numeric_job_id,
        "result_branch": p["result_branch"], "result_commit": expected_commit, "result_tree": expected_tree,
        "source_closure_sha256": release["source_closure_sha256"], "archive_sha256": t.sha(payload),
        "archive_bytes": len(payload), "outer_manifest_sha256": t.sha(outer),
        "raw_gets": 1 + len(chunks), "readback_body_bytes": len(outer) + len(payload),
        "semantic_inference_success_not_established_by_publisher": True, "retry_authorized": False}


def _safe_directory(root):
    root = Path(root)
    t.require(root.is_absolute() and str(root) == os.path.abspath(root), "canonical receipt directory")
    for part in [*reversed(root.parents), root]:
        t.require(stat.S_ISDIR(part.lstat().st_mode), "non-symlink receipt directory")
    return root


def write_once(root, name, value):
    root = _safe_directory(root)
    t.require(type(name) is str and t.path(name) == name and "/" not in name, "fixed receipt basename")
    # Resolve each directory with an openat no-follow walk; keep that descriptor
    # through creation and fsync so parent symlink swaps cannot redirect a write.
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(t.canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)


def run_once(env, event, release, payload, api, raw_get, receipt_directory, observer=None):
    identity = context(env, event, release)
    old_handler = signal.getsignal(signal.SIGALRM)
    t.require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "publisher owns separate bounded host process timer")
    _observe(observer, "publication_started")
    write_once(receipt_directory, "publication.started.json", {"schema": "k1-asr18-evidence-publication-started-v1",
        "identity": identity, "purpose": release["purpose"], "archive_sha256": t.sha(payload)})
    progress = {}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(PublicationError()))
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        _observe(observer, "remote_publish")
        result = publish(env, event, release, payload, api, raw_get, progress, observer=observer, transfer_directory=receipt_directory)
    except Exception as error:
        if observer is not None:
            try: observer.failure(error)
            except BaseException: pass  # A diagnostic must never replace the original failure.
        signal.setitimer(signal.ITIMER_REAL, 0)
        result = {"schema": "k1-asr18-evidence-publication-receipt-v1", "status": "publication_failed_or_unconfirmed_no_retry",
            "identity": identity, "purpose": release["purpose"], "error_type": type(error).__name__ if type(error) in (ValueError, TypeError, OSError, TimeoutError, PublicationError) else "OtherException",
            "retry_authorized": False, "new_native_calls_authorized": False, **progress}
        if type(error) is PublicationError:
            fields = BaseException.__dict__["__dict__"].__get__(error)
            status = dict.get(fields, "status")
            if type(status) is int and 100 <= status <= 599: result["http_status"] = status
        _emit_transfer(observer, identity, progress, "publication_failed_or_unconfirmed_no_retry", error, directory=receipt_directory)
        try: _observe(observer, "receipt_write")
        except BaseException: pass  # Preserve the already recorded primary publish exception.
        try:
            write_once(receipt_directory, "publication.receipt.json", result)
        except BaseException as receipt_error:
            if observer is not None:
                try: observer.failure(receipt_error)
                except BaseException: pass
        try: error.publication_receipt = result
        except BaseException: pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    _emit_transfer(observer, identity, progress, "same_identity_readback_verified", directory=receipt_directory)
    _observe(observer, "receipt_write")
    write_once(receipt_directory, "publication.receipt.json", result)
    return result


def disabled_publication_release():
    return {"schema": "k1-asr18-publication-control-v1", "enabled": False,
        "root_publication_release": False, "independent_source_review": False,
        "same_repository_ephemeral_token_only": True, "source_branch": t.POLICY["source_branch"],
        "result_branch": t.POLICY["result_branch"], "workflow": t.POLICY["workflow"],
        "job_key": "asr18", "run_attempt": 1, "retry_authorized": False}


def publication_gate(control):
    fixed = disabled_publication_release()
    approvals = ("enabled", "root_publication_release", "independent_source_review")
    t.require(type(control) is dict and set(control) == set(fixed), "fixed publication control fields")
    t.require(all(type(control[k]) is type(v) and control[k] == v for k, v in fixed.items() if k not in approvals),
        "fixed same-repository create-only publication scope")
    if not all(control[k] is True for k in approvals):
        raise PublicationDisabled("ASR18 publication disabled or review absent")



_CONTEXT_KEYS = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH", "GITHUB_REF",
    "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA", "GITHUB_WORKFLOW_REF", "GITHUB_WORKFLOW_SHA",
    "RUNNER_TEMP", "KWS_SOURCE_FREEZE_SHA256")


def _source_git(*args):
    t.require(args in (("status", "--porcelain", "--untracked-files=all"), ("rev-parse", "HEAD"),
        ("rev-parse", "HEAD^{tree}"), ("log", "-1", "--format=%H%n%cI", "HEAD")), "fixed local source Git observations only")
    safe_env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1", "LC_ALL": "C"}
    return subprocess.check_output(["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
        "-C", str(ROOT), *args], env=safe_env, stderr=subprocess.DEVNULL, timeout=15).decode("ascii").strip()

def _utc_timestamp(stamp):
    """Strict whole-second source timestamp; normalize an explicit known offset."""
    t.require(type(stamp) is str and re.fullmatch(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})", stamp),
        "whole-second source timestamp with explicit offset")
    if not stamp.endswith("Z"):
        hours, minutes = int(stamp[-5:-3]), int(stamp[-2:])
        t.require(stamp[-6:] != "-00:00" and minutes < 60 and
            (hours < 14 or (hours == 14 and minutes == 0)), "known source offset within fourteen hours")
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_timestamp(event, head):
    t.require(type(event) is dict and type(event.get("head_commit")) is dict, "actual push source timestamp")
    stamp = _utc_timestamp(event["head_commit"].get("timestamp"))
    commit = _source_git("log", "-1", "--format=%H%n%cI", "HEAD").splitlines()
    t.require(len(commit) == 2 and commit[0] == head, "timestamp from exact checked-out source commit")
    t.require(stamp == _utc_timestamp(commit[1]), "push timestamp matches source committer instant")
    return stamp


def bound_release(env, event, observer=None):
    """Clean exact source + separately approved freeze; never trust a self hash.

    The source-local control intentionally contains no hash of its own source
    commit or source freeze. KWS_SOURCE_FREEZE_SHA256 is the independently
    reviewed trusted-host workflow input and is not copied from a candidate.
    """
    _observe(observer, "source_release")
    publication_gate(c.read_json(HERE / "publication-release.json", cap=65536))
    execution = c.read_json(HERE / "execution-release.json", cap=65536)
    freeze_raw = c.read_bytes(HERE / "execution-freeze.json", cap=8 * 1024 ** 2)
    freeze = t.decode(freeze_raw)
    expected = env.get("KWS_SOURCE_FREEZE_SHA256")
    c.require_execution_release(execution, freeze, expected_freeze_sha256=expected)
    t.require(t.sha(freeze_raw) == expected, "exact independently approved source freeze bytes")
    t.require(env.get("GITHUB_ACTIONS") == "true", "trusted Actions poststep only")
    _observe(observer, "source_git_clean")
    t.require(_source_git("status", "--porcelain", "--untracked-files=all") == "", "clean executed source checkout")
    _observe(observer, "source_git_identity")
    head = _source_git("rev-parse", "HEAD")
    tree = _source_git("rev-parse", "HEAD^{tree}")
    t.require(head == env.get("GITHUB_SHA") and t.HEAD.fullmatch(head) and t.HEAD.fullmatch(tree), "actual clean executed Git source")
    _observe(observer, "source_freeze")
    import prepare_freeze
    t.require(Path(prepare_freeze.__file__).resolve() == HERE / "prepare_freeze.py", "fixed source-freeze builder")
    t.require(t.canonical(prepare_freeze.build(ROOT)) == freeze_raw, "all exact executed source closure bytes")
    _observe(observer, "source_timestamp")
    # Bind the push timestamp to the exact local commit instant, never a clock
    # or remote-server timestamp, and keep the fixed final UTC release contract.
    stamp = _source_timestamp(event, head)
    release = {"schema": "k1-asr18-publisher-bound-release-v1", "purpose": t.PURPOSE,
        "source_sha": head, "source_tree": tree, "source_closure_sha256": expected,
        "prior_claim_history_sha256": execution["prior_public_claim_history_sha256"], "commit_timestamp": stamp}
    validate_release(release)
    context(env, event, release)
    return release


def main(argv=None):
    # Construction does no IO. The initial checkpoint has only a fixed literal
    # stage; release admission still precedes environment/token/runtime access.
    observer = failure_observer.Observation(role="publisher")
    try:
        _observe(observer, "publication_gate")
        publication_gate(c.read_json(HERE / "publication-release.json", cap=65536))
        _observe(observer, "arguments")
        argv = sys.argv[1:] if argv is None else argv
        t.require(argv == [], "fixed trusted-host poststep paths only")
        env = {key: os.environ[key] for key in _CONTEXT_KEYS if key in os.environ}
        _observe(observer, "event_read")
        event = c.read_json(Path(env.get("GITHUB_EVENT_PATH", "")), cap=1024 * 1024)
        release = bound_release(env, event, observer=observer)
        _observe(observer, "context")
        run_identity = context(env, event, release)
        observer.attach(run_identity, Path(env["RUNNER_TEMP"]))
        _observe(observer, "package_read")
        payload = t.read_package(Path(env["RUNNER_TEMP"]), run_identity)
        # The worker never sees this token. Only the separate reviewed Actions
        # publisher step receives the same-repository short-lived GITHUB_TOKEN.
        _observe(observer, "token_admission")
        api = GitHubAPI(os.environ.get("GITHUB_TOKEN"), run_identity, release)
        _observe(observer, "receipt_directory")
        root = Path(env["RUNNER_TEMP"]) / ("k1-asr18-publication-" + str(run_identity["run_id"]))
        c.durable_mkdir(root)
        result = run_once(env, event, release, payload, api, RawReader, root, observer=observer)
    except BaseException as error:
        # A secondary diagnostic failure never replaces the original exception.
        try: observer.failure(error)
        except BaseException: pass
        try: observer.finish(success=False)
        except BaseException: pass
        raise
    _observe(observer, "complete")
    observer.finish(success=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except PublicationDisabled:
        raise SystemExit("ASR18 publication disabled or review absent.") from None
    except BaseException:
        # No arbitrary error strings, HTTP bodies, token, base64 or local paths.
        raise SystemExit("ASR18 publication blocked or unconfirmed; no retry.") from None
