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
MAX_API_REQUESTS = 67
MAX_API_BODY = 8 * 1024 * 1024
MAX_API_UPLOAD = 18 * 1024 * 1024
TIMEOUT = c.EFFECTIVE_EXECUTION_ENVELOPE["publisher_seconds"]
AUTHOR = {"name": "K1 ASR18 bounded publisher", "email": "noreply@github.com"}


class PublicationError(RuntimeError):
    def __init__(self, status=None, *, body_bounded=True):
        self.body_bounded = body_bounded is True
        self.status = status if type(status) is int else None
        super().__init__("publication failed or unconfirmed; no further retry")


class PublicationDisabled(ValueError):
    """Only the fixed source-local approval gate emits this safe CLI reason."""


def validate_release(release):
    fields = {"schema", "purpose", "source_sha", "source_tree", "source_closure_sha256",
        "prior_claim_history_sha256", "commit_timestamp"}
    t.require(type(release) is dict, "reviewed detached release object")
    bound = release.get("schema") == "k1-asr18-publisher-bound-release-v2"
    if bound:
        fields |= {"claim_admission_mode", "execution_release_sha256", "claim_admission_control_sha256",
            "source_branch", "workflow", "result_branch"}
        for key in ("source_branch", "result_branch"):
            _branch(release.get(key))
        t.require(type(release.get("workflow")) is str and re.fullmatch(r"\.github/workflows/[A-Za-z0-9_-]+\.ya?ml", release["workflow"]), "fixed source-bound workflow path")
        t.require(release.get("claim_admission_mode") == "git-owner-v2", "source-bound admission mode")
        for key in ("execution_release_sha256", "claim_admission_control_sha256"):
            c.sha(release.get(key))
        t.require(all(release[key] == c.current_scope()[key]
            for key in ("source_branch", "workflow", "result_branch")), "publisher compiled source scope")
    else:
        t.require(release.get("schema") == "k1-asr18-publisher-bound-release-v1", "release schema")
    t.require(set(release) == fields, "reviewed detached release fields")
    t.policy(release["purpose"])
    for key in ("source_sha", "source_tree"):
        t.require(type(release[key]) is str and t.HEAD.fullmatch(release[key]), "reviewed source Git binding")
    for key in ("source_closure_sha256", "prior_claim_history_sha256"):
        t.require(type(release[key]) is str and t.SHA.fullmatch(release[key]), "reviewed source/history binding")
    stamp = release["commit_timestamp"]
    t.require(type(stamp) is str and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", stamp), "fixed commit timestamp")
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def publisher_policy(release):
    policy = t.policy(release["purpose"])
    if release.get("schema") == "k1-asr18-publisher-bound-release-v2":
        t.require(all(release[key] == c.current_scope()[key]
            for key in ("source_branch", "workflow", "result_branch")), "publisher compiled source scope")
    elif release["purpose"] == t.PURPOSE:
        policy = dict(t.LEGACY_POLICY)
    return policy


def producer_history_route(owner):
    c.validate_bound_identity(owner)
    c.require_current_scope(owner)
    return BASE + "/actions/workflows/" + owner["workflow"].split("/")[-1] + "/runs?branch=" + owner["source_branch"] + "&per_page=100"


def context(env, event, release):
    validate_release(release)
    p = publisher_policy(release)
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
    p = publisher_policy(release)
    run = str(identity["run_id"])
    return {"repo": BASE, "run": BASE + "/actions/runs/" + run,
        "jobs": BASE + "/actions/runs/" + run + "/attempts/1/jobs?per_page=100",
        "history": BASE + "/actions/workflows/" + p["workflow"].split("/")[-1] +
            ("/runs?branch=" if release["schema"] == "k1-asr18-publisher-bound-release-v2" else "/runs?event=push&branch=") + p["source_branch"] + "&per_page=100",
        "source": BASE + "/git/commits/" + release["source_sha"],
        "source_tree": BASE + "/git/trees/" + release["source_tree"] + "?recursive=1",
        "ref": BASE + "/git/ref/heads/" + p["result_branch"],
        "checkpoint_ref": BASE + "/git/ref/heads/" + p["result_branch"] + "-checkpoint"}


class GitHubAPI:
    """Fixed TLS/API scope with one globally bounded uncertain-blob recovery.

    The normal constructor admits only the reviewed ASR profile. The private
    initializer is shared with separately admitted, source-fixed transfer tests;
    it never reads credentials or bypasses normal publication validation.
    """
    def __init__(self, token, identity, release):
        validate_release(release); t.identity(identity, release["purpose"])
        self.release = t.decode(t.canonical(release))
        self._initialize(token, get_routes=set(endpoints(identity, release).values()),
            source_sha=release["source_sha"], source_tree=release["source_tree"],
            result_branch=publisher_policy(release)["result_branch"],
            paths={t.manifest_path(release["purpose"]): t.MAX_MANIFEST,
                **{t.chunk_path(i, release["purpose"]): t.MAX_CHUNK for i in range(t.MAX_CHUNKS)}})

    def _initialize(self, token, *, get_routes, source_sha, source_tree, result_branch, paths, expected_body=None,
            max_requests=MAX_API_REQUESTS, timeout=TIMEOUT, max_upload=MAX_API_UPLOAD, max_body=MAX_API_BODY, final_ref_only=False, checkpoint_branch=None):
        t.require(not hasattr(self, "_token"), "API fixed scope initialized once")
        t.require(type(token) is str and token.startswith("ghs_") and 5 <= len(token) <= 4096
            and all(32 <= ord(char) < 127 for char in token), "publisher-only short-lived token")
        for value in (source_sha, source_tree):
            t.require(type(value) is str and t.HEAD.fullmatch(value), "exact source Git identity")
        _branch(result_branch)
        checkpoint_branch = result_branch + "-checkpoint" if checkpoint_branch is None else _branch(checkpoint_branch)
        t.require(checkpoint_branch != result_branch, "distinct fixed checkpoint ref")
        t.require(type(get_routes) in (set, frozenset) and len(get_routes) <= 16 and all(type(route) is str
            and route.startswith(BASE + "/") or route == BASE for route in get_routes), "fixed same repository API routes")
        t.require(all("\r" not in route and "\n" not in route and "#" not in route
            and ".." not in route and "%" not in route for route in get_routes), "canonical fixed API routes")
        t.require(type(paths) is dict and 2 <= len(paths) <= 49, "bounded fixed output paths")
        for name, cap in paths.items():
            t.path(name)
            t.require(type(cap) is int and 0 < cap <= t.MAX_CHUNK, "bounded fixed path cap")
        self._token = token
        self.source_sha = source_sha; self.source_tree = source_tree
        self.result_branch = result_branch; self.checkpoint_branch = checkpoint_branch; self.paths = dict(paths)
        self._file_caps = dict(paths)
        self.required_paths = {t.manifest_path(self.release["purpose"])} if hasattr(self, "release") else set(paths)
        self.expected_body = None if expected_body is None else t.decode(t.canonical(expected_body))
        t.require(type(final_ref_only) is bool, "fixed API write mode")
        self.final_ref_only = final_ref_only
        refs = {BASE + "/git/ref/heads/" + result_branch, BASE + "/git/ref/heads/" + checkpoint_branch}
        self.allowed = {("GET", route) for route in get_routes | refs}
        self.allowed |= {("POST", BASE + "/git/" + noun) for noun in (("refs",) if final_ref_only else ("blobs", "trees", "commits", "refs"))}
        for value, cap in ((max_requests, MAX_API_REQUESTS), (timeout, TIMEOUT), (max_upload, MAX_API_UPLOAD), (max_body, MAX_API_BODY)):
            t.require(type(value) is int and 0 < value <= cap, "scope only narrows transport limits")
        self.max_requests = max_requests; self.max_upload = max_upload; self.max_body = max_body
        self.deadline = time.monotonic() + timeout
        self.count = 0; self.body_bytes = 0; self.request_bytes = 0; self.seen = set(); self.posts = set()
        self.response_bodies = {}
        self.blob_api_attempt_count = 0; self.blob_confirmed_count = 0
        self.blob_recovery_get_count = 0; self.blob_retry_count = 0; self._recovery_consumed = False
        self._retry_request = None

    def bind_result(self, commit_sha, tree_sha):
        for value in (commit_sha, tree_sha):
            t.require(type(value) is str and t.HEAD.fullmatch(value), "locally computed immutable Git object")
        t.require(not hasattr(self, "result"), "result binding once")
        if self.expected_body is None:
            self.expected_body, computed = commit_body(tree_sha, self.release)
            t.require(computed == commit_sha, "precomputed result commit binding")
        t.require(self.expected_body.get("tree") == tree_sha
            and self.expected_body.get("parents") == [self.source_sha], "fixed result tree and sole source parent")
        t.require(_commit_identity(self.expected_body) == commit_sha, "locally computed exact commit bytes")
        self.result = (commit_sha, tree_sha)
        self.allowed |= {("GET", BASE + "/git/commits/" + commit_sha),
                         ("GET", BASE + "/git/trees/" + tree_sha + "?recursive=1")}

    def __call__(self, method, route, body=None, expected=200):
        if method != "POST" or route != BASE + "/git/blobs":
            return self._request(method, route, body, expected)
        t.require(type(body) is dict and set(body) == {"content", "encoding"} and body["encoding"] == "base64"
            and type(body["content"]) is str and len(body["content"]) <= 4 * ((t.MAX_CHUNK + 2) // 3), "bounded exact blob request")
        raw = base64.b64decode(body["content"], validate=True)
        t.require(0 < len(raw) <= t.MAX_CHUNK and base64.b64encode(raw).decode("ascii") == body["content"], "canonical blob request")
        digest = t.git_sha(raw)
        try:
            result = self._request(method, route, body, expected)
        except PublicationError as error:
            if type(error) is not PublicationError: raise
            fields = BaseException.__dict__["__dict__"].__get__(error)
            status = dict.get(fields, "status"); bounded = dict.get(fields, "body_bounded")
            if self._recovery_consumed or bounded is not True or not (status is None or type(status) is int
                    and status in (500, 502, 503, 504)): raise
            self._recovery_consumed = True
            recovery_route = BASE + "/git/blobs/" + digest
            self.allowed.add(("GET", recovery_route))
            self.blob_recovery_get_count += 1
            try:
                recovered = self._request("GET", recovery_route)
            except PublicationError as recovery_error:
                if type(recovery_error) is not PublicationError: raise
                fields = BaseException.__dict__["__dict__"].__get__(recovery_error)
                if type(dict.get(fields, "status")) is not int or dict.get(fields, "status") != 404 or dict.get(fields, "body_bounded") is not True: raise
                # Only an actual authenticated 404 can release this exact body
                # for one extra POST. A second uncertainty is terminal.
                self._retry_request = ("POST", route, t.sha(t.canonical(body)), 201)
                self.blob_retry_count += 1
                result = self._request(method, route, body, expected)
            else:
                _equal(recovered, {"sha": digest, "encoding": "base64", "size": len(raw)}, "exact recovered blob identity")
                content = recovered.get("content")
                # GitHub wraps base64 with LF; admit only LF and canonical bytes.
                t.require(type(content) is str and len(content) <= MAX_RESPONSE, "bounded recovered blob content")
                content = content.replace("\n", "")
                recovered_raw = base64.b64decode(content, validate=True)
                t.require(len(recovered_raw) == len(raw) and recovered_raw == raw and t.git_sha(recovered_raw) == digest
                    and base64.b64encode(recovered_raw).decode("ascii") == content, "exact recovered blob bytes")
                result = {"sha": digest}
        _equal(result, {"sha": digest}, "true Git blob identity")
        self.blob_confirmed_count += 1
        return result

    def _request(self, method, route, body=None, expected=200):
        t.require((method, route) in self.allowed, "fixed method/route only")
        t.require(type(expected) is int and expected in (200, 201, 404), "HTTP status policy")
        t.require((method == "POST" and expected == 201 and type(body) is dict)
                  or (method == "GET" and expected in (200, 404) and body is None), "HTTP method/body/status")
        raw = None if body is None else t.canonical(body)
        t.require(raw is None or len(raw) <= MAX_REQUEST, "request body cap")
        t.require(self.request_bytes + len(raw or b"") <= self.max_upload, "total API upload body cap")
        post_key = route
        if method == "POST" and route != BASE + "/git/blobs":
            if route == BASE + "/git/refs":
                refs = {"refs/heads/" + self.result_branch}
                if not self.final_ref_only: refs.add("refs/heads/" + self.checkpoint_branch)
                t.require(hasattr(self, "result") and set(body) == {"ref", "sha"} and body["ref"] in refs
                    and body["sha"] == self.result[0], "only exact reviewed refs and computed commit")
                post_key = (route, body["ref"])
            elif route == BASE + "/git/commits":
                t.require(hasattr(self, "result") and body == self.expected_body, "only computed result commit")
            elif route == BASE + "/git/trees":
                t.require(set(body) == {"base_tree", "tree"} and body["base_tree"] == self.source_tree
                    and type(body["tree"]) is list and 2 <= len(body["tree"]) <= 49, "bounded source-preserving result tree")
                names = [row.get("path") for row in body["tree"] if type(row) is dict]
                t.require(len(names) == len(body["tree"]) == len(set(names)) and set(names) <= set(self.paths)
                    and self.required_paths <= set(names), "only bounded result paths")
                for row in body["tree"]:
                    t.require(set(row) == {"path", "mode", "type", "sha"} and row["mode"] == "100644" and row["type"] == "blob"
                        and type(row["sha"]) is str and t.HEAD.fullmatch(row["sha"]), "exact result tree rows")
            t.require(post_key not in self.posts, "create-only object attempted once")
        request = (method, route, t.sha(raw or b""), expected)
        retry = self._retry_request == request
        t.require(request not in self.seen or retry, "request already consumed")
        remaining = self.deadline - time.monotonic()
        t.require(remaining > 0 and self.count < self.max_requests, "API deadline/count cap")
        t.require(0 <= self.body_bytes < self.max_body, "remaining API response budget")
        if retry: self._retry_request = None
        self.seen.add(request); self.count += 1; self.request_bytes += len(raw or b"")
        if method == "POST" and route == BASE + "/git/blobs": self.blob_api_attempt_count += 1
        if method == "POST" and route != BASE + "/git/blobs": self.posts.add(post_key)
        connection = None; status = None
        try:
            connection = http.client.HTTPSConnection("api.github.com", timeout=min(15, remaining), context=ssl.create_default_context())
            connection.request(method, route, body=raw, headers={
                "Authorization": "Bearer " + self._token, "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "k1-asr18-bounded-evidence-publisher",
                "Content-Type": "application/json"})
            response = connection.getresponse(); status = response.status
            response_body = response.read(min(MAX_RESPONSE, self.max_body - self.body_bytes) + 1)
            self.body_bytes += len(response_body)
            if len(response_body) > MAX_RESPONSE or self.body_bytes > self.max_body:
                raise PublicationError(status, body_bounded=False)
            if status != expected: raise PublicationError(status)
            if expected == 404: return None
            value = t.decode(response_body)
            t.require(type(value) is dict, "API object response")
            self.response_bodies[route] = response_body
            return value
        except PublicationError:
            raise
        except (OSError, http.client.HTTPException):
            raise PublicationError(status) from None
        except Exception:
            raise PublicationError(status, body_bounded=False) from None
        finally:
            if connection is not None:
                try: connection.close()
                except Exception: pass


def _branch(value):
    t.require(type(value) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]{0,199}", value)
        and not value.endswith("/") and "//" not in value, "exact bounded branch")
    return value


class RawReader:
    """Public immutable raw-content only; at most 49 paths once, 16 MiB bodies."""
    def __init__(self, commit_sha, purpose):
        t.policy(purpose)
        self._initialize(commit_sha, {t.manifest_path(purpose): t.MAX_MANIFEST,
            **{t.chunk_path(i, purpose): t.MAX_CHUNK for i in range(t.MAX_CHUNKS)}})

    def _initialize(self, commit_sha, paths, *, shared_deadline=None):
        t.require(not hasattr(self, "prefix"), "raw fixed scope initialized once")
        t.require(type(commit_sha) is str and t.HEAD.fullmatch(commit_sha), "immutable result commit")
        t.require(type(paths) is dict and 2 <= len(paths) <= 49, "bounded raw paths")
        for name, cap in paths.items():
            t.path(name)
            t.require(type(cap) is int and 0 < cap <= t.MAX_CHUNK, "bounded raw path cap")
        self.prefix = "/" + t.REPOSITORY + "/" + commit_sha + "/"
        self.allowed = dict(paths)
        self.seen = set(); self.body_bytes = 0
        self.deadline = min(time.monotonic() + TIMEOUT, shared_deadline) if shared_deadline is not None else time.monotonic() + TIMEOUT

    def __call__(self, name):
        t.require(name in self.allowed and name not in self.seen and len(self.seen) < 49, "fixed raw path once")
        remaining = self.deadline - time.monotonic()
        t.require(remaining > 0, "raw total deadline")
        t.require(0 <= self.body_bytes < t.MAX_BODY, "remaining raw response budget")
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


def history_observation(api, route, owner, page, phase):
    raw = getattr(api, "response_bodies", {}).get(route)
    t.require(type(raw) is bytes and 0 < len(raw) <= MAX_RESPONSE and
        c.exact_json(t.decode(raw), page), "original bounded history response bytes required")
    return c.api_bound_history(owner, page, phase, raw_sha256=t.sha(raw))


def verify_run(api, identity, release, *, source_claim=None, observations=None):
    if source_claim is not None:
        c.require_current_scope(source_claim["identity"])
    routes = endpoints(identity, release); p = publisher_policy(release)
    repo = api("GET", routes["repo"])
    _equal(repo, {"full_name": t.REPOSITORY, "private": False}, "API canonical public repository")
    run = api("GET", routes["run"])
    expected = {"id": identity["run_id"], "run_attempt": 1, "head_sha": release["source_sha"],
        "head_branch": p["source_branch"], "event": "push", "path": p["workflow"], "status": "in_progress"}
    _equal(run, expected, "actual run binding")
    _equal(run.get("repository"), {"full_name": t.REPOSITORY, "private": False}, "actual run repository")
    history = api("GET", routes["history"])
    if source_claim is None:
        t.require(release["schema"] == "k1-asr18-publisher-bound-release-v1", "v2 publisher requires source-bound admission")
        _equal(history, {"total_count": 1}, "exact one run for fresh source ref")
        t.require(type(history.get("workflow_runs")) is list and len(history["workflow_runs"]) == 1, "complete one-use history")
        _equal(history["workflow_runs"][0], expected, "history actual run binding")
    else:
        owner = source_claim["identity"]
        c.validate_bound_admission(source_claim["admission"], owner)
        t.require(owner["run_id"] == identity["run_id"] and owner["source_sha"] == identity["executed_head_sha"]
            and owner["source_tree"] == release["source_tree"], "same source-bound publisher owner")
        _equal(run, c.bound_run_row(owner), "same actual owner run identity")
        _equal(run.get("head_repository"), {"full_name": t.REPOSITORY, "private": False}, "public owner head repository")
        observation = history_observation(api, routes["history"], owner, history, "publication")
        if observations is not None: observations["producer_history"] = observation
        _equal(run, {"run_number": 1, "run_attempt": 1, "conclusion": None}, "eligible live immutable owner")
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
    if source_claim is not None:
        t.require(job["id"] == source_claim["identity"]["job_id"], "same independently observed owner job")
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


def _commit_identity(body):
    """Exact fixed-author single-parent commit bytes, including the terminal LF."""
    t.require(type(body) is dict and set(body) == {"message", "tree", "parents", "author", "committer"}, "exact commit body fields")
    t.require(type(body["message"]) is str and 0 < len(body["message"].encode("utf-8")) <= 4096
        and body["message"].endswith("\n") and not body["message"].endswith("\n\n") and "\r" not in body["message"], "fixed bounded commit message")
    t.require(type(body["tree"]) is str and t.HEAD.fullmatch(body["tree"])
        and type(body["parents"]) is list and len(body["parents"]) == 1
        and type(body["parents"][0]) is str and t.HEAD.fullmatch(body["parents"][0]), "fixed commit Git identities")
    author = body["author"]
    t.require(type(author) is dict and set(author) == {"name", "email", "date"} and body["committer"] == author,
        "fixed identical commit persons")
    _equal(author, AUTHOR, "fixed publisher commit author")
    stamp = author["date"]
    t.require(type(stamp) is str and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", stamp), "fixed commit timestamp")
    timestamp = int(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    person = author["name"] + " <" + author["email"] + "> " + str(timestamp) + " +0000"
    raw = ("tree " + body["tree"] + "\nparent " + body["parents"][0] + "\nauthor " + person
        + "\ncommitter " + person + "\n\n" + body["message"]).encode("utf-8")
    return t.git_sha(raw, "commit")


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
    "create_blobs", "create_tree", "create_commit", "create_ref", "immutable_postpublish_readback", "readback_verified",
    "create_checkpoint_ref", "immutable_checkpoint_readback", "checkpoint_verified"))
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
    wire = {}
    for key, cap in (("blob_api_attempt_count", len(rows) + 1 if rows else 50),
            ("blob_recovery_get_count", 1), ("blob_retry_count", 1)):
        value = progress.get(key)
        t.require(value is None or (type(value) is int and 0 <= value <= cap), "bounded wire attempt counts")
        wire[key] = value
    checkpoint = {}
    branch = progress.get("checkpoint_branch")
    t.require(branch is None or branch == t.POLICY["result_branch"] + "-checkpoint", "fixed observation checkpoint branch")
    checkpoint["checkpoint_branch"] = branch
    for key, expected_key in (("checkpoint_commit", "expected_result_commit"), ("checkpoint_tree", "expected_result_tree")):
        value = progress.get(key)
        t.require(value is None or (type(value) is str and t.HEAD.fullmatch(value)
            and (progress.get(expected_key) is None or value == progress[expected_key])), "fixed checkpoint identity")
        checkpoint[key] = value
    for key in ("checkpoint_created", "checkpoint_verified", "checkpoint_preserved"):
        value = progress.get(key)
        t.require(value is None or type(value) is bool, "fixed checkpoint confirmation")
        checkpoint[key] = value
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
        "status": status, "stage": stage, "planned_blobs": rows, **numeric, **wire, **digests, **checkpoint, "archive_bytes": size,
        "http_status": http_status, "error_type": code,
        "publication_confirmed": progress.get("publication_confirmed"), "readback_confirmed": progress.get("readback_confirmed"),
        "attempted_blob_count_scope": "blob_api_invocations_not_confirmed_http_sends",
        "observation_delivery": "best_effort_stdout",
        "unconfirmed_request_may_have_written": True, "original_http_body_included": False,
        "native_success_established": False, "retry_authorized": False, "limited_blob_retry_cap": 1,
        "retry_authorized_scope": "whole_publication_or_host_rerun"}


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


def finalize_exact_ref(api, result_branch, expected_commit, *, progress=None, observer=None):
    """Create one exact final ref, then read it back; never update an existing ref.

    The admitted caller must already have verified its complete pinned immutable
    checkpoint and checked final absence. This primitive grants no admission.
    """
    _branch(result_branch)
    t.require(type(expected_commit) is str and t.HEAD.fullmatch(expected_commit), "immutable final commit")
    ref = "refs/heads/" + result_branch
    # The server may persist a ref before its response is lost. Null means the
    # attempted creation is unknown, not that the ref was never created.
    if progress is not None: progress["publication_confirmed"] = None
    created = api("POST", BASE + "/git/refs", {"ref": ref, "sha": expected_commit}, expected=201)
    _equal(created, {"ref": ref}, "created exact final ref")
    _equal(created.get("object"), {"type": "commit", "sha": expected_commit}, "created final ref exact commit")
    if progress is not None: progress.update(publication_confirmed=True, stage="immutable_postpublish_readback")
    _observe(observer, "immutable_postpublish_readback")
    final = api("GET", BASE + "/git/ref/heads/" + result_branch)
    _equal(final, {"ref": ref}, "readback final ref")
    _equal(final.get("object"), {"type": "commit", "sha": expected_commit}, "readback final commit")
    return final


def _transport_counts(api, progress):
    # Fakes may not expose wire counters. Never confuse absent observations with zero.
    for key in ("blob_api_attempt_count", "blob_recovery_get_count", "blob_retry_count"):
        value = getattr(api, key, None)
        if type(value) is int: progress[key] = value


def write_checked_files(*, api, raw_get, files, original, source_tree, expected_body, expected_commit,
        result_branch, checkpoint_branch, verify_readback, progress=None, observer=None, before_write=None,
        checkpoint_only=False):
    """Shared bounded wire writer after caller-specific admission and validation.

    The normal ASR entry always validates the whole scientific package first.
    Qualification callers have separate fixed synthetic admission. No callback
    can skip exact-byte/tree checks or make an ASR schema claim through this path.
    checkpoint_only is internal and unavailable from the normal publisher CLI.
    """
    progress = {} if progress is None else progress
    _branch(result_branch)
    t.require(checkpoint_branch == getattr(api, "checkpoint_branch", result_branch + "-checkpoint"), "exact checkpoint branch")
    t.require(type(checkpoint_only) is bool and callable(verify_readback), "fixed shared writer mode")
    t.require(type(files) is dict and 2 <= len(files) <= 49, "bounded complete output file set")
    for name, raw in files.items():
        t.path(name)
        t.require(type(raw) is bytes and 0 < len(raw) <= t.MAX_CHUNK, "bounded immutable output bytes")
        scope = getattr(api, "_file_caps", None)
        if type(scope) is dict:
            t.require(name in scope and len(raw) <= scope[name], "exact admitted output path and cap")
    t.require(sum(map(len, files.values())) <= t.MAX_BODY, "complete output body cap")
    files = dict(files)
    t.require(t.tree_identity(original)[0] == source_tree, "exact preserved source tree")
    tree_rows = [{"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for name, raw in sorted(files.items())]
    expected_tree, _ = t.tree_identity(original + tree_rows)
    t.require(expected_body.get("tree") == expected_tree and type(expected_body.get("parents")) is list
        and len(expected_body["parents"]) == 1, "exact computed merged tree and sole source parent")
    api.bind_result(expected_commit, expected_tree)
    _observe(observer, "verify_destination_absent")
    progress.update(stage="verify_destination_absent", publication_confirmed=False, readback_confirmed=False,
        checkpoint_branch=checkpoint_branch, checkpoint_commit=expected_commit, checkpoint_tree=expected_tree,
        checkpoint_created=False, checkpoint_verified=False, checkpoint_preserved=False)
    api("GET", BASE + "/git/ref/heads/" + result_branch, expected=404)
    api("GET", BASE + "/git/ref/heads/" + checkpoint_branch, expected=404)
    _observe(observer, "create_blobs")
    progress.update(stage="create_blobs", attempted_blob_count=0, confirmed_blob_count=0,
        expected_result_commit=expected_commit, expected_result_tree=expected_tree,
        planned_blobs=[{"path": row["path"], "bytes": len(files[row["path"]]),
            "sha256": t.sha(files[row["path"]]), "git_blob_sha": row["sha"]} for row in tree_rows])
    _transport_counts(api, progress)
    if before_write is not None: before_write()
    sent_blobs = set()
    for row in tree_rows:
        if row["sha"] not in sent_blobs:
            progress["attempted_blob_count"] += 1
            try:
                blob = api("POST", BASE + "/git/blobs", {"content": base64.b64encode(files[row["path"]]).decode("ascii"), "encoding": "base64"}, expected=201)
            finally:
                _transport_counts(api, progress)
            _equal(blob, {"sha": row["sha"]}, "true Git blob identity")
            sent_blobs.add(row["sha"])
            progress["confirmed_blob_count"] = len(sent_blobs)
    _observe(observer, "create_tree"); progress["stage"] = "create_tree"
    tree = api("POST", BASE + "/git/trees", {"base_tree": source_tree, "tree": tree_rows}, expected=201)
    _equal(tree, {"sha": expected_tree}, "true computed merged Git tree")
    _observe(observer, "create_commit"); progress["stage"] = "create_commit"
    verify_commit(api("POST", BASE + "/git/commits", expected_body, expected=201), expected_commit, expected_body)
    _observe(observer, "create_checkpoint_ref"); progress["stage"] = "create_checkpoint_ref"
    checkpoint_ref = "refs/heads/" + checkpoint_branch
    progress["checkpoint_created"] = None  # An uncertain response may follow a persisted ref.
    created = api("POST", BASE + "/git/refs", {"ref": checkpoint_ref, "sha": expected_commit}, expected=201)
    _equal(created, {"ref": checkpoint_ref}, "created exact checkpoint ref")
    _equal(created.get("object"), {"type": "commit", "sha": expected_commit}, "created checkpoint exact commit")
    progress["checkpoint_created"] = True
    _observe(observer, "immutable_checkpoint_readback"); progress["stage"] = "immutable_checkpoint_readback"
    checkpoint = api("GET", BASE + "/git/ref/heads/" + checkpoint_branch)
    _equal(checkpoint, {"ref": checkpoint_ref}, "readback checkpoint ref")
    _equal(checkpoint.get("object"), {"type": "commit", "sha": expected_commit}, "readback checkpoint commit")
    verify_commit(api("GET", BASE + "/git/commits/" + expected_commit), expected_commit, expected_body)
    actual = verify_tree(api("GET", BASE + "/git/trees/" + expected_tree + "?recursive=1"), expected_tree)
    t.require(sorted(actual, key=lambda row: row["path"]) == sorted(original + tree_rows, key=lambda row: row["path"]), "exact preserved source and new output paths")
    read = raw_get(expected_commit)
    got_files = {}
    for name, expected_raw in sorted(files.items()):
        raw = read(name)
        t.require(type(raw) is bytes and raw == expected_raw, "exact complete immutable file readback")
        got_files[name] = raw
    t.require(verify_readback(got_files) is True, "caller-specific complete readback validation")
    _observe(observer, "checkpoint_verified")
    progress.update(stage="checkpoint_verified", checkpoint_verified=True, checkpoint_preserved=True)
    result = {"checkpoint_branch": checkpoint_branch, "checkpoint_commit": expected_commit, "checkpoint_tree": expected_tree,
        "checkpoint_created": True, "checkpoint_verified": True, "checkpoint_preserved": True,
        "result_commit": expected_commit, "result_tree": expected_tree, "raw_gets": len(got_files),
        "readback_body_bytes": sum(map(len, got_files.values()))}
    if not checkpoint_only:
        _observe(observer, "create_ref"); progress["stage"] = "create_ref"
        finalize_exact_ref(api, result_branch, expected_commit, progress=progress, observer=observer)
        _observe(observer, "readback_verified")
        progress.update(stage="readback_verified", publication_confirmed=True, readback_confirmed=True)
    result.update(publication_confirmed=progress["publication_confirmed"], readback_confirmed=progress["readback_confirmed"])
    _transport_counts(api, progress)
    result.update({key: progress.get(key) for key in ("attempted_blob_count", "confirmed_blob_count", "blob_api_attempt_count", "blob_recovery_get_count", "blob_retry_count")})
    return result


def publish(env, event, release, payload, api, raw_get, progress=None, observer=None, transfer_directory=None):
    progress = {} if progress is None else progress
    _observe(observer, "validate_local_package")
    progress.update(stage="validate_local_package", publication_confirmed=False, readback_confirmed=False)
    identity = context(env, event, release); purpose = release["purpose"]; p = publisher_policy(release)
    t.validate_archive(payload, identity)
    _observe(observer, "verify_actual_run_source"); progress["stage"] = "verify_actual_run_source"
    routes = endpoints(identity, release)
    source = api("GET", routes["source"])
    _equal(source, {"sha": release["source_sha"]}, "reviewed source commit")
    _equal(source.get("tree"), {"sha": release["source_tree"]}, "reviewed source tree")
    if release["schema"] == "k1-asr18-publisher-bound-release-v2":
        t.require(type(source.get("committer")) is dict and
            _utc_timestamp(source["committer"].get("date")) == release["commit_timestamp"], "actual source commit timestamp")
    original = verify_tree(api("GET", routes["source_tree"]), release["source_tree"])
    # The independently read source tree selects the history policy. A claim's
    # own schema, or an untrusted caller flag, can never promote legacy source.
    local_job = t.archive_job_identity(payload, identity)
    source_claim = t.validate_source_binding(payload, identity, release["source_closure_sha256"],
        original, local_job, release["source_tree"])
    t.validate_publisher_source_release(payload, release, source_claim)
    observations = {}
    numeric_job_id = verify_run(api, identity, release, source_claim=source_claim, observations=observations)
    t.require(numeric_job_id == local_job, "archive job equals independent current job")
    directory = p["directory"]
    t.require(not any(row["path"] == directory or row["path"].startswith(directory + "/")
        or directory.startswith(row["path"] + "/") for row in original), "result directory absent; no replacement")
    files = t.build_file_set(payload, identity, release["source_closure_sha256"], numeric_job_id, purpose)
    tree_rows = [{"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for name, raw in sorted(files.items())]
    expected_tree, _ = t.tree_identity(original + tree_rows)
    expected_body, expected_commit = commit_body(expected_tree, release)
    progress.update(archive_sha256=t.sha(payload), archive_bytes=len(payload))
    def checked_readback(got_files):
        outer = got_files[t.manifest_path(purpose)]
        manifest = t.parse_manifest(outer, identity, release["source_closure_sha256"], numeric_job_id, purpose)
        got = t.reassemble(outer, [(row["path"], got_files[row["path"]]) for row in manifest["chunks"]],
            identity, release["source_closure_sha256"], numeric_job_id, purpose)
        t.require(got == payload, "same complete audio-free evidence archive readback")
        t.validate_archive(got, identity)
        return True
    result = write_checked_files(api=api, raw_get=lambda commit: raw_get(commit, purpose), files=files,
        original=original, source_tree=release["source_tree"], expected_body=expected_body, expected_commit=expected_commit,
        result_branch=p["result_branch"], checkpoint_branch=p["result_branch"] + "-checkpoint",
        verify_readback=checked_readback, progress=progress, observer=observer,
        before_write=lambda: _emit_transfer(observer, identity, progress, "upload_plan_prepared", directory=transfer_directory))
    return {"schema": "k1-asr18-evidence-publication-receipt-v1", "status": "same_identity_readback_verified",
        "identity": identity, "purpose": purpose, "numeric_job_id": numeric_job_id,
        **observations,
        "result_branch": p["result_branch"], **result,
        "source_closure_sha256": release["source_closure_sha256"], "archive_sha256": t.sha(payload),
        "archive_bytes": len(payload), "outer_manifest_sha256": t.sha(files[t.manifest_path(purpose)]),
        "semantic_inference_success_not_established_by_publisher": True, "retry_authorized": False, "limited_blob_retry_cap": 1,
        "retry_authorized_scope": "whole_publication_or_host_rerun"}


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
            "retry_authorized": False, "limited_blob_retry_cap": 1, "retry_authorized_scope": "whole_publication_or_host_rerun",
            "new_native_calls_authorized": False, **progress}
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
    return {"schema": "k1-asr18-publication-control-v2", "claim_admission_mode": "git-owner-v2", "enabled": False,
        "root_publication_release": False, "independent_source_review": False,
        "same_repository_ephemeral_token_only": True, "source_branch": t.POLICY["source_branch"],
        "result_branch": t.POLICY["result_branch"], "workflow": t.POLICY["workflow"],
        "job_key": "asr18", "run_attempt": 1, "retry_authorized": False, "limited_blob_retry_cap": 1,
        "effective_execution_envelope": dict(c.EFFECTIVE_EXECUTION_ENVELOPE),
        "scientific_campaign_baseline_minutes": 70}


def _publication_scope(control):
    fixed = disabled_publication_release()
    if type(control) is dict and control.get("schema") == "k1-asr18-publication-control-v1":
        fixed = {key: value for key, value in fixed.items() if key != "claim_admission_mode"}
        fixed["schema"] = "k1-asr18-publication-control-v1"
        fixed.update({key: t.LEGACY_POLICY[key] for key in ("source_branch", "result_branch", "workflow")})
    approvals = ("enabled", "root_publication_release", "independent_source_review")
    t.require(type(control) is dict and set(control) == set(fixed), "fixed publication control fields")
    t.require(all(type(control[k]) is type(v) and control[k] == v for k, v in fixed.items() if k not in approvals),
        "fixed same-repository create-only publication scope")
    envelope = control["effective_execution_envelope"]
    t.require(type(envelope) is dict and set(envelope) == set(c.EFFECTIVE_EXECUTION_ENVELOPE)
        and all(type(value) is int and value == c.EFFECTIVE_EXECUTION_ENVELOPE[key] for key, value in envelope.items()),
        "exact effective execution envelope integer limits")
    t.require(all(type(control[key]) is bool for key in approvals), "publication approval boolean fields")


def publication_gate(control):
    _publication_scope(control)
    if control["schema"] != "k1-asr18-publication-control-v2" or not all(control[key] is True for key in ("enabled", "root_publication_release", "independent_source_review")):
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


def validate_transfer_workflow(raw, control=None, *, admission_control=None):
    """Fixed workflow inspection, not a permissive general YAML interpreter.

    The scientific 70-minute baseline remains bound by historical proofs. The
    effective current job is 80 minutes, including its separately bounded
    publisher; neither the baseline nor the ASR/model execution budget expands.
    """
    t.require(type(c.LIMITS.get("workflow_minutes")) is int and c.LIMITS["workflow_minutes"] == 70
        and type(c.EFFECTIVE_EXECUTION_ENVELOPE) is dict
        and c.EFFECTIVE_EXECUTION_ENVELOPE == {"workflow_minutes": 80, "publisher_seconds": 600, "host_seconds": 3600}
        and all(type(value) is int for value in c.EFFECTIVE_EXECUTION_ENVELOPE.values()),
        "fixed historical scientific and effective transfer workflow limits")
    if control is not None: _publication_scope(control)
    t.require(type(raw) is bytes and 0 < len(raw) <= 256 * 1024, "bounded exact workflow bytes")
    text = raw.decode("ascii")
    t.require("\r" not in text and "\t" not in text and text.count("    steps:\n") == 1,
        "fixed workflow layout")
    head, steps = text.split("    steps:\n")
    t.require(head.count("jobs:\n") == 1 and head.split("jobs:\n")[1].startswith("  asr18:\n")
        and re.findall(r"(?m)^    timeout-minutes:.*$", head) == ["    timeout-minutes: 80"],
        "effective current workflow job timeout eighty minutes")
    blocks = re.split(r"(?m)(?=^      - )", steps)
    bound = (admission_control is not None or
        type(control) is dict and control.get("schema") == "k1-asr18-publication-control-v2" or
        "      - name: Reserve fixed campaign owner without models\n" in steps)
    if admission_control is not None:
        c.validate_claim_control(admission_control)
    expected = (
        ("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", 2),
        ("      - name: K1 punctuation6 ASR18 offline source contracts", 4 if bound else 5),
        *(((("      - name: Reserve fixed campaign owner without models", 1),)) if bound else ()) ,
        ("      - name: Admit complete new input and run fixed ASR18 once", 61),
        ("      - name: Publish complete audio-free bounded evidence", 11),
    )
    t.require(blocks[0] == "" and len(blocks) == len(expected) + 1 and text.count("timeout-minutes") == len(expected) + 1,
        "exact fixed bounded workflow step roster")
    for block, (first, minutes) in zip(blocks[1:], expected):
        t.require(block.splitlines()[0] == first and re.findall(r"(?m)^        timeout-minutes:.*$", block)
            == ["        timeout-minutes: " + str(minutes)], "fixed checkout contracts host publisher step timeouts")
    if bound:
        issuer, host, publisher = blocks[3:6]
        issuer_lines = issuer.splitlines()
        t.require(issuer_lines == [
            "      - name: Reserve fixed campaign owner without models", "        id: campaign_claim",
            "        if: ${{ success() && steps.asr18_contracts.outcome == 'success' && github.run_number == 1 && github.run_attempt == 1 }}",
            "        timeout-minutes: 1", "        env:", "          GITHUB_TOKEN: ${{ github.token }}",
            "        run: |", "          python3 -I -B research/k1-punctuation6-asr18-v1/claim_admission.py"],
            "one isolated fixed claim issuer invocation and token scope")
        t.require("        if: ${{ success() && steps.asr18_contracts.outcome == 'success' && steps.campaign_claim.outcome == 'success' && steps.campaign_claim.outputs.ready == 'true' }}" in host.splitlines(),
            "host requires durable issuer readiness")
        t.require("GITHUB_TOKEN" not in head + blocks[1] + blocks[2] + host
            and text.count("          GITHUB_TOKEN: ${{ github.token }}") == 2
            and publisher.count("          GITHUB_TOKEN: ${{ github.token }}") == 1,
            "token only in isolated issuer and publisher")
    t.require(sum(minutes for _, minutes in expected) == 79 < c.EFFECTIVE_EXECUTION_ENVELOPE["workflow_minutes"]
        and TIMEOUT == c.EFFECTIVE_EXECUTION_ENVELOPE["publisher_seconds"] == 600 < 11 * 60
        and c.LIMITS["host_seconds"] == c.EFFECTIVE_EXECUTION_ENVELOPE["host_seconds"] == 3600 < 61 * 60,
        "separate host publisher and workflow timeout margins")


def _validate_workflow_transfer_scope(execution=None, admission_control=None):
    workflow=t.POLICY["workflow"] if execution is None else execution["workflow"]
    actual = c.read_bytes(ROOT / workflow, cap=256 * 1024)
    template = c.read_bytes(HERE / "workflow.yml", cap=256 * 1024)
    t.require(actual == template, "exact actual workflow and template bytes")
    validate_transfer_workflow(actual, admission_control=admission_control)


def bound_release(env, event, observer=None):
    """Clean exact source + separately approved freeze; never trust a self hash.

    The source-local control intentionally contains no hash of its own source
    commit or source freeze. KWS_SOURCE_FREEZE_SHA256 is the independently
    reviewed trusted-host workflow input and is not copied from a candidate.
    """
    _observe(observer, "source_release")
    publication_gate(c.read_json(HERE / "publication-release.json", cap=65536))
    execution_raw = c.read_bytes(HERE / "execution-release.json", cap=65536)
    execution = t.decode(execution_raw)
    control_raw = c.read_bytes(HERE / "claim-admission-control.json", cap=65536)
    admission_control = c.require_claim_admission(execution, control_raw)
    c.require_current_scope(admission_control)
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
    _validate_workflow_transfer_scope(execution, admission_control)
    import prepare_freeze
    t.require(Path(prepare_freeze.__file__).resolve() == HERE / "prepare_freeze.py", "fixed source-freeze builder")
    t.require(t.canonical(prepare_freeze.build(ROOT)) == freeze_raw, "all exact executed source closure bytes")
    _observe(observer, "source_timestamp")
    # Bind the push timestamp to the exact local commit instant, never a clock
    # or remote-server timestamp, and keep the fixed final UTC release contract.
    stamp = _source_timestamp(event, head)
    release = {"schema": "k1-asr18-publisher-bound-release-v2", "purpose": t.PURPOSE,
        "claim_admission_mode": "git-owner-v2", "execution_release_sha256": t.sha(execution_raw),
        "claim_admission_control_sha256": t.sha(control_raw),
        "source_branch": execution["source_ref"], "workflow": execution["workflow"], "result_branch": execution["evidence_ref"],
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
        t.validate_archive(payload, run_identity)
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
