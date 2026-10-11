"""Trusted HOST-only, create-once public Git evidence publisher.

The workflow provides KWS_EVIDENCE_TOKEN only to this separate post-capture
process. Never import this from inference. No artifact/cache/LFS, updates,
redirects, arbitrary URLs, retries, checkout changes or recognition calls.
HTTP is injectable for offline tests; tests never need a real credential.
"""
from pathlib import Path, PurePosixPath
import base64
import hashlib
import http.client
import os
import re
import signal
import ssl
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_transport import HEAD, canonical, decode, require, sha
from branch_transport import (REPOSITORY, TRANSPORT, MAX_MANIFEST, bound_identity,
    manifest_path, package_path, purpose_policy, read_package, safe_directory)

BASE = "/repos/" + REPOSITORY
MAX_RESPONSE = 1024 * 1024
MAX_REQUEST = 512 * 1024
WORKFLOW = ".github/workflows/qwen12-native-admission-v1.yml"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


class PublicationError(RuntimeError):
    def __init__(self, status=None):
        self.status = status if type(status) is int else None
        super().__init__("evidence publication failed")


def context(env, event, purpose):
    policy = purpose_policy(purpose)
    require(type(event) is dict and type(event.get("repository")) is dict, "push event required")
    require(env.get("GITHUB_REPOSITORY") == REPOSITORY and event["repository"].get("full_name") == REPOSITORY
            and event["repository"].get("private") is False, "canonical public repository")
    require(env.get("GITHUB_EVENT_NAME") == "push" and event.get("created") is True
            and event.get("deleted") is False, "fresh branch push only")
    require(env.get("GITHUB_REF") == "refs/heads/" + policy["source_branch"]
            and event.get("ref") == env["GITHUB_REF"], "purpose/source ref mismatch")
    require(env.get("GITHUB_RUN_ATTEMPT") == "1", "single run attempt")
    value = env.get("GITHUB_RUN_ID")
    require(type(value) is str and re.fullmatch(r"[1-9][0-9]{0,19}", value), "real run id")
    run_identity = {"repository": REPOSITORY, "run_id": int(value), "run_attempt": 1,
        "job_key": env.get("GITHUB_JOB"), "executed_head_sha": env.get("GITHUB_SHA")}
    bound_identity(run_identity, purpose)
    require(event.get("after") == run_identity["executed_head_sha"], "actual executed push head")
    return run_identity


def frozen_sources():
    """Local reviewed source integrity, independent of real-ASR release enablement."""
    from campaign_contract import read_json, identity
    freeze = read_json(HERE / "execution-freeze.json")
    release = read_json(HERE / "execution-release.json")
    require(release.get("source_freeze_sha256") == identity(HERE / "execution-freeze.json")["sha256"],
            "publisher source-freeze binding")
    required = {"research/qwen12-native-admission-v1/" + name for name in
                ("branch_transport.py", "evidence_publisher.py", "emit_synthetic_branch_fixture.py", "log_transport.py")}
    require(type(freeze.get("host_code")) is dict and required <= set(freeze["host_code"]), "publisher frozen source coverage")
    for name, row in freeze["host_code"].items():
        require(type(name) is str and not PurePosixPath(name).is_absolute()
                and str(PurePosixPath(name)) == name and all(part not in (".", "..", "") for part in name.split("/"))
                and name.startswith(("research/", "tools/", ".github/workflows/")), "frozen code scope")
        require(identity(ROOT / name) == row, "publisher frozen source changed")
    require(release.get("evidence_transport") == TRANSPORT, "reviewed branch transport required")
    require(release.get("new_transcripts_publication_approved") is True, "explicit public evidence approval required")


def endpoints(run_identity, purpose):
    policy = purpose_policy(purpose)
    head = run_identity["executed_head_sha"]
    run = str(run_identity["run_id"])
    return {"repo": BASE, "run": BASE + "/actions/runs/" + run,
        "jobs": BASE + "/actions/runs/" + run + "/attempts/1/jobs?per_page=100",
        "source": BASE + "/git/commits/" + head,
        "ref": BASE + "/git/ref/heads/" + policy["evidence_branch"],
        "directory": BASE + "/contents/" + policy["directory"] + "?ref=" + head}


class GitHubAPI:
    """Fixed TLS host, exact route allowlist, bounded bodies, no redirect/retry."""
    def __init__(self, token, run_identity, purpose):
        require(type(token) is str and 1 <= len(token) <= 4096 and "\r" not in token and "\n" not in token,
                "publisher-only token required")
        self._token = token
        self._allowed = {("GET", path) for path in endpoints(run_identity, purpose).values()}
        self._allowed |= {("POST", BASE + "/git/" + kind) for kind in ("blobs", "trees", "commits", "refs")}
        self._deadline = time.monotonic() + 180

    def __call__(self, method, path, body=None, expected=200):
        require((method, path) in self._allowed, "fixed API method/path only")
        require(type(expected) is int and expected in (200, 201, 404), "API expected status")
        remaining = self._deadline - time.monotonic()
        require(remaining > 0, "publisher total deadline")
        data = None if body is None else canonical(body)
        require(data is None or len(data) <= MAX_REQUEST, "API request cap")
        connection = http.client.HTTPSConnection("api.github.com", timeout=min(15, remaining),
                                                 context=ssl.create_default_context())
        try:
            connection.request(method, path, body=data, headers={
                "Authorization": "Bearer " + self._token,
                "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "kws-bounded-evidence-publisher", "Content-Type": "application/json"})
            response = connection.getresponse()
            status = response.status
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE or status != expected:
                raise PublicationError(status)
            if expected == 404:
                return None
            result = decode(raw)
            require(type(result) is dict, "API response type")
            return result
        except PublicationError:
            raise
        except Exception:
            raise PublicationError() from None
        finally:
            connection.close()


def git_sha(raw, kind="blob"):
    return hashlib.sha1(kind.encode("ascii") + b" " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def publication_path(purpose):
    return purpose_policy(purpose)["directory"] + "/publication.json"


def validate_publication(raw, expected_identity, purpose, numeric_job_id, outer_manifest_raw):
    require(type(raw) is bytes and 1 <= len(raw) <= MAX_MANIFEST, "publication metadata cap/type")
    value = decode(raw)
    require(type(value) is dict and set(value) == {"schema", "identity", "purpose", "source_tree_sha",
        "numeric_job_id", "job_status_at_publication", "outer_manifest_sha256", "evidence_branch",
        "native_completion_not_established"}, "publication metadata fields")
    require(canonical(value) == raw and value["schema"] == "native-evidence-publication-v1", "canonical publication metadata")
    bound_identity(value["identity"], purpose)
    require(canonical(value["identity"]) == canonical(expected_identity) and value["purpose"] == purpose,
            "publication identity mismatch")
    require(type(numeric_job_id) is int and numeric_job_id > 0 and type(value["numeric_job_id"]) is int
            and value["numeric_job_id"] == numeric_job_id, "publication numeric job binding")
    require(type(value["source_tree_sha"]) is str and HEAD.fullmatch(value["source_tree_sha"]), "source tree identity")
    require(value["job_status_at_publication"] == "in_progress"
            and value["native_completion_not_established"] is True, "publication does not establish ASR completion")
    require(value["outer_manifest_sha256"] == sha(outer_manifest_raw)
            and value["evidence_branch"] == purpose_policy(purpose)["evidence_branch"], "publication package/ref binding")
    return value


def publish(env, event, purpose, api, progress=None):
    """Offline-testable publication protocol; no token access and no retries."""
    progress = {} if progress is None else progress
    progress.update(stage="validate_local_package", publication_confirmed=False)
    run_identity = context(env, event, purpose)
    policy = purpose_policy(purpose)
    files, payload, _, custody = read_package(env["RUNNER_TEMP"], run_identity, purpose)
    if purpose == "synthetic":
        from emit_synthetic_branch_fixture import verify_synthetic
        verify_synthetic(payload, run_identity)
    paths = endpoints(run_identity, purpose)
    progress["stage"] = "verify_api_identity"
    repository = api("GET", paths["repo"])
    require(repository.get("full_name") == REPOSITORY and repository.get("private") is False,
            "API canonical public repository")
    run = api("GET", paths["run"])
    for key, expected in (("id", run_identity["run_id"]), ("run_attempt", 1),
            ("head_sha", run_identity["executed_head_sha"]), ("head_branch", policy["source_branch"]),
            ("event", "push"), ("path", WORKFLOW), ("status", "in_progress")):
        require(type(run.get(key)) is type(expected) and run[key] == expected, "API actual run mismatch")
    require(type(run.get("repository")) is dict and run["repository"].get("full_name") == REPOSITORY
            and run["repository"].get("private") is False, "API run repository mismatch")
    jobs = api("GET", paths["jobs"])
    require(type(jobs.get("total_count")) is int and 1 <= jobs["total_count"] <= 100
            and type(jobs.get("jobs")) is list and len(jobs["jobs"]) == jobs["total_count"], "complete bounded jobs list")
    matches = [job for job in jobs["jobs"] if type(job) is dict and job.get("name") == policy["job_key"]]
    require(len(matches) == 1, "unique nonmatrix actual job required")
    job = matches[0]
    for key, expected in (("run_id", run_identity["run_id"]),
            ("head_sha", run_identity["executed_head_sha"]), ("status", "in_progress"),
            ("run_url", "https://api.github.com" + paths["run"]), ("head_branch", policy["source_branch"])):
        require(type(job.get(key)) is type(expected) and job[key] == expected, "API actual job mismatch")
    if "run_attempt" in job:
        require(type(job["run_attempt"]) is int and job["run_attempt"] == 1, "job attempt mismatch")
    require(type(job.get("id")) is int and job["id"] > 0 and job.get("conclusion") is None,
            "actual in-progress numeric job required")
    source = api("GET", paths["source"])
    require(source.get("sha") == run_identity["executed_head_sha"] and type(source.get("tree")) is dict
            and type(source["tree"].get("sha")) is str and HEAD.fullmatch(source["tree"]["sha"]), "actual source commit/tree")
    progress.update(stage="verify_destination_absent", source_tree_sha=source["tree"]["sha"])
    api("GET", paths["ref"], expected=404)
    api("GET", paths["directory"], expected=404)
    publication = {"schema": "native-evidence-publication-v1", "identity": run_identity,
        "purpose": purpose, "source_tree_sha": source["tree"]["sha"], "numeric_job_id": job["id"],
        "job_status_at_publication": job["status"], "outer_manifest_sha256": sha(files[manifest_path(purpose)]),
        "evidence_branch": policy["evidence_branch"], "native_completion_not_established": True}
    publication_raw = canonical(publication)
    validate_publication(publication_raw, run_identity, purpose, job["id"], files[manifest_path(purpose)])
    files[publication_path(purpose)] = publication_raw
    tree_rows = []
    progress.update(stage="create_blobs", confirmed_blob_count=0)
    for path, raw in sorted(files.items()):
        result = api("POST", BASE + "/git/blobs", {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"}, expected=201)
        expected_blob = git_sha(raw)
        require(result.get("sha") == expected_blob, "created blob identity")
        tree_rows.append({"path": path, "mode": "100644", "type": "blob", "sha": expected_blob})
        progress.update(last_confirmed_blob_sha=expected_blob, confirmed_blob_count=len(tree_rows))
    progress["stage"] = "create_tree"
    tree = api("POST", BASE + "/git/trees", {"base_tree": source["tree"]["sha"], "tree": tree_rows}, expected=201)
    require(type(tree.get("sha")) is str and HEAD.fullmatch(tree["sha"]), "created tree identity")
    progress.update(stage="create_commit", evidence_tree_sha=tree["sha"])
    commit = api("POST", BASE + "/git/commits", {"message": "Retain bounded " + purpose + " Qwen12 evidence",
        "tree": tree["sha"], "parents": [run_identity["executed_head_sha"]]}, expected=201)
    require(type(commit.get("sha")) is str and HEAD.fullmatch(commit["sha"])
            and type(commit.get("tree")) is dict and commit["tree"].get("sha") == tree["sha"]
            and type(commit.get("parents")) is list and len(commit["parents"]) == 1
            and commit["parents"][0].get("sha") == run_identity["executed_head_sha"], "created evidence commit binding")
    # Sole visibility/branch mutation: CREATE, once, after every file is in a single tree.
    ref = "refs/heads/" + policy["evidence_branch"]
    progress.update(stage="create_ref", evidence_commit=commit["sha"])
    created = api("POST", BASE + "/git/refs", {"ref": ref, "sha": commit["sha"]}, expected=201)
    require(created.get("ref") == ref and type(created.get("object")) is dict
            and created["object"].get("type") == "commit" and created["object"].get("sha") == commit["sha"],
            "created ref binding")
    progress.update(stage="evidence_ref_confirmed", publication_confirmed=True)
    return {"schema": "native-evidence-publication-receipt-v1", "status": "evidence_branch_created",
        "identity": run_identity, "numeric_job_id": job["id"], "purpose": purpose,
        "evidence_branch": policy["evidence_branch"], "evidence_commit": commit["sha"],
        "archive_sha256": custody["archive_sha256"], "archive_bytes": custody["archive_bytes"],
        "file_count": len(files), "native_completion_not_established": True,
        "independent_readback_required": True}


def _write_once(path, value):
    with path.open("xb") as target:
        target.write(canonical(value))


def run_once(env, event, purpose, api):
    run_identity = context(env, event, purpose)
    root = package_path(env["RUNNER_TEMP"], run_identity, purpose).parent
    safe_directory(root)
    # A terminal claim prevents process reruns even after a network-uncertain failure.
    _write_once(root / "evidence-publication-started.json", {"identity": run_identity, "purpose": purpose})
    progress = {}
    try:
        result = publish(env, event, purpose, api, progress)
    except Exception as error:
        result = {"schema": "native-evidence-publication-receipt-v1", "status": "publication_failed_or_unconfirmed_no_retry",
            "error_type": type(error).__name__, "native_completion_not_established": True, **progress}
        if isinstance(error, PublicationError) and error.status is not None:
            result["http_status"] = error.status
        _write_once(root / "evidence-publication-receipt.json", result)
        error.publication_receipt = result
        raise
    _write_once(root / "evidence-publication-receipt.json", result)
    return result


def _deadline_expired(_signum, _frame):
    raise PublicationError()


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ("synthetic", "native"), "fixed publisher purpose only")
    purpose = sys.argv[1]
    # Copy only explicitly needed nonsecret CI values. Never dump the environment.
    names = ("GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_EVENT_NAME", "GITHUB_RUN_ATTEMPT",
             "GITHUB_RUN_ID", "GITHUB_JOB", "GITHUB_SHA", "RUNNER_TEMP")
    env = {name: os.environ[name] for name in names}
    with open(os.environ["GITHUB_EVENT_PATH"], "rb") as source:
        raw = source.read(MAX_RESPONSE + 1)
    require(len(raw) <= MAX_RESPONSE, "event response cap")
    event = decode(raw)
    run_identity = context(env, event, purpose)
    frozen_sources()
    read_package(env["RUNNER_TEMP"], run_identity, purpose)
    signal.signal(signal.SIGALRM, _deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, 180)
    try:
        api = GitHubAPI(os.environ["KWS_EVIDENCE_TOKEN"], run_identity, purpose)
        print(canonical(run_once(env, event, purpose, api)).decode("ascii"), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        result = {"status": "publication_failed_or_unconfirmed_no_retry", "error_type": type(error).__name__}
        if isinstance(error, PublicationError) and error.status is not None:
            result["http_status"] = error.status
        receipt = getattr(error, "publication_receipt", None)
        print(canonical(receipt if type(receipt) is dict else result).decode("ascii"), flush=True)
        raise SystemExit(2)
