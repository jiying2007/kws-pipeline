"""Disabled, create-only ASR evidence transfer recovery. No model execution.

A future separately reviewed source may bind one original complete immutable
checkpoint. This entry cannot recover a lost runner ZIP or a sealed ASR run.
The current source has no recovery workflow and no release authority.
"""
from pathlib import Path
import os
import re
import signal
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import campaign_contract as c
import evidence_transport as t
import evidence_publisher as p

SOURCE_BRANCH = "research/k1-asr18-transfer-recovery-v1"
WORKFLOW = ".github/workflows/k1-asr18-transfer-recovery-v1.yml"
JOB_KEY = "transfer_recovery"
# Publicly recorded lost runner ZIP; never turn metadata into recoverable bytes.
LOST_R2_RUN = 38286042984
LOST_R2_SOURCE = "d3b7ce54c8909df8e962d8547a546c3d460dcbc8"
MAX_API_REQUESTS = 17
MAX_RAW_BODY = 32 * 1024 ** 2
TIMEOUT = 600
HOST_STEP = "Admit complete new input and run fixed ASR18 once"
PUBLISHER_STEP = "Publish complete audio-free bounded evidence"
CONCLUSIONS = frozenset(("success", "failure", "cancelled", "timed_out", "neutral", "skipped", "action_required", "stale", "startup_failure"))
APPROVALS = ("enabled", "root_transfer_recovery_release", "independent_source_review")
_CONTEXT_KEYS = ("GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH", "GITHUB_REF",
    "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB", "GITHUB_SHA", "GITHUB_WORKFLOW_REF", "GITHUB_WORKFLOW_SHA",
    "RUNNER_ENVIRONMENT", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "KWS_SOURCE_FREEZE_SHA256")


class RecoveryDisabled(ValueError):
    pass


def disabled_release():
    return {"schema": "k1-asr18-transfer-recovery-control-v1", "enabled": False,
        "root_transfer_recovery_release": False, "independent_source_review": False,
        "same_repository_ephemeral_token_only": True, "source_branch": SOURCE_BRANCH,
        "workflow": WORKFLOW, "job_key": JOB_KEY, "run_attempt": 1,
        "result_branch": t.POLICY["result_branch"], "new_native_calls_authorized": False,
        "retry_authorized": False, "authority": None}


def original_release(authority):
    # The original timestamp and sole parent reproduce its existing Git object.
    return {"source_sha": authority["producer_identity"]["executed_head_sha"],
        "source_tree": authority["source_tree"], "commit_timestamp": authority["source_commit_timestamp"]}


def validate_authority(value):
    keys = {"schema", "producer_identity", "numeric_job_id", "source_tree", "source_closure_sha256",
        "source_commit_timestamp", "archive_sha256", "archive_bytes", "outer_manifest_sha256",
        "checkpoint_ref", "checkpoint_commit", "checkpoint_tree", "terminal"}
    t.require(type(value) is dict and set(value) == keys, "complete exact original checkpoint authority")
    t.require(value["schema"] == "k1-asr18-transfer-recovery-authority-v1", "recovery authority schema")
    t.identity(value["producer_identity"])
    t.require(value["producer_identity"]["run_id"] != LOST_R2_RUN
        and value["producer_identity"]["executed_head_sha"] != LOST_R2_SOURCE, "lost r2 producer is not recoverable")
    t.require(type(value["numeric_job_id"]) is int and 0 < value["numeric_job_id"] < 10 ** 20, "original numeric job")
    for key in ("source_tree", "checkpoint_commit", "checkpoint_tree"):
        t.require(type(value[key]) is str and t.HEAD.fullmatch(value[key]), "pinned original Git identity")
    for key in ("source_closure_sha256", "archive_sha256", "outer_manifest_sha256"):
        t.require(type(value[key]) is str and t.SHA.fullmatch(value[key]), "pinned complete evidence digest")
    stamp = value["source_commit_timestamp"]
    t.require(type(stamp) is str and stamp.endswith("Z") and p._utc_timestamp(stamp) == stamp, "original UTC source timestamp")
    t.require(type(value["archive_bytes"]) is int and 0 < value["archive_bytes"] <= t.MAX_ARCHIVE, "pinned archive size")
    t.require(value["checkpoint_ref"] == "refs/heads/" + t.POLICY["result_branch"] + "-checkpoint", "only fixed original checkpoint ref")
    terminal = value["terminal"]
    t.require(type(terminal) is dict and set(terminal) == {"run_conclusion", "job_conclusion", "host_step_conclusion",
        "publisher_step_conclusion", "host_receipt_status"}, "exact original terminal observations")
    for key in ("run_conclusion", "job_conclusion", "host_step_conclusion", "publisher_step_conclusion"):
        t.require(type(terminal[key]) is str and terminal[key] in CONCLUSIONS, "pinned terminal conclusion")
    t.require(terminal["publisher_step_conclusion"] in ("failure", "cancelled", "timed_out"), "original publisher failure remains failed")
    t.require(terminal["host_receipt_status"] in ("stopped_no_retry", "all_planned_native_complete", "cleanup_unconfirmed"), "original host terminal status")
    t.require(terminal["host_step_conclusion"] != "success" or terminal["host_receipt_status"] == "all_planned_native_complete",
        "successful original host requires complete native host receipt")
    _, computed = p.commit_body(value["checkpoint_tree"], original_release(value))
    t.require(computed == value["checkpoint_commit"], "checkpoint equals original precomputed result commit")
    return value


def recovery_gate(control):
    fixed = disabled_release()
    t.require(type(control) is dict and set(control) == set(fixed), "fixed recovery control fields")
    t.require(all(type(control[key]) is type(value) and control[key] == value for key, value in fixed.items()
        if key not in (*APPROVALS, "authority")), "fixed transfer-only scope")
    if not all(control[key] is True for key in APPROVALS):
        raise RecoveryDisabled("ASR transfer recovery disabled or review absent")
    return validate_authority(control["authority"])


def context(env, event, control):
    authority = recovery_gate(control)
    t.require(type(env) is dict and type(event) is dict and type(event.get("repository")) is dict, "exact recovery context")
    t.require(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == t.REPOSITORY
        and event["repository"].get("full_name") == t.REPOSITORY and event["repository"].get("private") is False,
        "canonical public recovery repository")
    t.require(env.get("GITHUB_EVENT_NAME") == "push" and event.get("created") is True and event.get("deleted") is False
        and event.get("forced") is False and event.get("before") == "0" * 40, "fresh push-created recovery branch only")
    t.require(env.get("GITHUB_REF") == "refs/heads/" + SOURCE_BRANCH and event.get("ref") == env["GITHUB_REF"], "fixed recovery source ref")
    t.require(env.get("GITHUB_RUN_ATTEMPT") == "1" and type(env.get("GITHUB_RUN_ID")) is str
        and re.fullmatch(r"[1-9][0-9]{0,19}", env["GITHUB_RUN_ID"]), "recovery attempt one")
    head = env.get("GITHUB_SHA")
    t.require(type(head) is str and t.HEAD.fullmatch(head) and event.get("after") == head, "exact recovery source SHA")
    t.require(env.get("GITHUB_JOB") == JOB_KEY and env.get("GITHUB_WORKFLOW_REF") == t.REPOSITORY + "/" + WORKFLOW
        + "@refs/heads/" + SOURCE_BRANCH and env.get("GITHUB_WORKFLOW_SHA") == head, "separate fixed recovery workflow/job")
    t.require(env.get("RUNNER_ENVIRONMENT") == "github-hosted" and env.get("RUNNER_OS") == "Linux"
        and env.get("RUNNER_ARCH") == "X64" and env.get("ImageOS") == "ubuntu24", "fixed trusted recovery host")
    run = {"repository": t.REPOSITORY, "run_id": int(env["GITHUB_RUN_ID"]), "run_attempt": 1,
        "job_key": JOB_KEY, "executed_head_sha": head}
    t.require(run["run_id"] != authority["producer_identity"]["run_id"] and head != authority["producer_identity"]["executed_head_sha"],
        "recovery identity distinct from original producer")
    return run


def bound_source(env, event, control):
    """Only recovery approval, clean source and independent freeze; no model gates."""
    run = context(env, event, control)
    for module, leaf in ((c, "campaign_contract.py"), (t, "evidence_transport.py"), (p, "evidence_publisher.py")):
        t.require(Path(module.__file__).resolve() == HERE / leaf, "fixed recovery import origin")
    t.require(p._source_git("status", "--porcelain", "--untracked-files=all") == "", "clean recovery source checkout")
    head = p._source_git("rev-parse", "HEAD")
    tree = p._source_git("rev-parse", "HEAD^{tree}")
    t.require(head == run["executed_head_sha"] and type(tree) is str and t.HEAD.fullmatch(tree), "exact recovery Git source")
    raw = c.read_bytes(HERE / "execution-freeze.json", cap=8 * 1024 ** 2)
    expected = env.get("KWS_SOURCE_FREEZE_SHA256")
    t.require(type(expected) is str and t.SHA.fullmatch(expected) and t.sha(raw) == expected, "independently reviewed recovery freeze")
    import prepare_freeze
    t.require(Path(prepare_freeze.__file__).resolve() == HERE / "prepare_freeze.py", "fixed freeze builder origin")
    t.require(t.canonical(prepare_freeze.build(ROOT)) == raw, "complete recovery source closure")
    freeze = t.decode(raw)
    names = {WORKFLOW, "research/k1-punctuation6-asr18-v1/transfer_recovery.py",
        "research/k1-punctuation6-asr18-v1/transfer-recovery-release.json"}
    t.require(type(freeze.get("host_code")) is dict and names <= set(freeze["host_code"]), "reviewed installed recovery entry/control/workflow")
    stamp = p._source_timestamp(event, head)
    return {"source_sha": head, "source_tree": tree, "source_closure_sha256": expected, "commit_timestamp": stamp}


def endpoints(run, source, authority):
    policy = t.policy()
    base = p.BASE
    routes = {"repo": base, "checkpoint_ref": base + "/git/ref/heads/" + policy["result_branch"] + "-checkpoint",
        "checkpoint_commit": base + "/git/commits/" + authority["checkpoint_commit"],
        "checkpoint_tree": base + "/git/trees/" + authority["checkpoint_tree"] + "?recursive=1",
        "final_ref": base + "/git/ref/heads/" + policy["result_branch"]}
    for prefix, identity, branch, workflow, sha, tree in (
        ("recovery", run, SOURCE_BRANCH, WORKFLOW, source["source_sha"], source["source_tree"]),
        ("producer", authority["producer_identity"], policy["source_branch"], policy["workflow"],
            authority["producer_identity"]["executed_head_sha"], authority["source_tree"])):
        routes.update({prefix + "_run": base + "/actions/runs/" + str(identity["run_id"]),
            prefix + "_jobs": base + "/actions/runs/" + str(identity["run_id"]) + "/attempts/1/jobs?per_page=100",
            prefix + "_history": base + "/actions/workflows/" + workflow.split("/")[-1] + "/runs?event=push&branch=" + branch + "&per_page=100",
            prefix + "_source": base + "/git/commits/" + sha,
            prefix + "_tree": base + "/git/trees/" + tree + "?recursive=1"})
    return routes


class RecoveryAPI(p.GitHubAPI):
    """Shared fixed TLS client, narrowed to 17 requests and one final-ref POST."""
    def __init__(self, token, run, source, control):
        authority = recovery_gate(control)
        t.require(run["job_key"] == JOB_KEY and source["source_sha"] == run["executed_head_sha"], "admitted recovery source")
        routes = endpoints(run, source, authority)
        self.authority = t.decode(t.canonical(authority))
        self.producer_history_route = routes["producer_history"]
        self.producer_history_bound = False
        body, commit = p.commit_body(authority["checkpoint_tree"], original_release(authority))
        self._initialize(token, get_routes=set(routes.values()), source_sha=authority["producer_identity"]["executed_head_sha"],
            source_tree=authority["source_tree"], result_branch=t.POLICY["result_branch"],
            paths={t.manifest_path(): t.MAX_MANIFEST, **{t.chunk_path(i): t.MAX_CHUNK for i in range(t.MAX_CHUNKS)}}, expected_body=body, max_requests=MAX_API_REQUESTS, timeout=TIMEOUT,
            max_upload=p.MAX_REQUEST, max_body=p.MAX_API_BODY, final_ref_only=True)
        self.bind_result(commit, authority["checkpoint_tree"])
        self.allowed = {("GET", route) for route in routes.values()} | {("POST", p.BASE + "/git/refs")}

    def bind_producer_history(self, payload, source_rows):
        """Replace only the producer-history ticket after immutable proof binding."""
        a = self.authority
        t.require(not self.producer_history_bound and
            not any(row[0] == "GET" and row[1] == self.producer_history_route for row in self.seen),
            "producer history rebound once before any history request")
        t.require(type(payload) is bytes and len(payload) == a["archive_bytes"]
            and t.sha(payload) == a["archive_sha256"] and
            t.tree_identity(source_rows)[0] == a["source_tree"],
            "independently pinned original archive and complete source tree")
        proof = t.validate_source_binding(payload, a["producer_identity"], a["source_closure_sha256"],
            source_rows, a["numeric_job_id"], a["source_tree"])
        t.require(proof is not None, "source-bound producer proof required for route replacement")
        route = p.producer_history_route(proof["identity"])
        self.allowed.remove(("GET", self.producer_history_route))
        self.allowed.add(("GET", route))
        self.producer_history_route = route
        self.producer_history_bound = True
        return route

    def __call__(self, method, route, body=None, expected=200):
        t.require(self.count < MAX_API_REQUESTS, "recovery exact API budget")
        if method == "POST":
            t.require(route == p.BASE + "/git/refs" and not self.posts and body == {
                "ref": "refs/heads/" + t.POLICY["result_branch"], "sha": self.result[0]}, "one exact final-ref create only")
        return self._request(method, route, body, expected)


class CheckpointReader(p.RawReader):
    """One complete immutable read; inherited 16 MiB limit is stricter than 32."""
    def __init__(self, commit, *, deadline):
        self._initialize(commit, {t.manifest_path(): t.MAX_MANIFEST,
            **{t.chunk_path(i): t.MAX_CHUNK for i in range(t.MAX_CHUNKS)}}, shared_deadline=deadline)

    def __call__(self, name):
        raw = super().__call__(name)
        t.require(self.body_bytes <= MAX_RAW_BODY, "hard aggregate recovery raw byte cap")
        return raw


def _verify_run(api, routes, prefix, identity, branch, workflow, job_key, terminal=None, *, source_claim=None, observations=None):
    if source_claim is not None:
        c.require_current_scope(source_claim["identity"])
    expected = {"id": identity["run_id"], "run_attempt": 1, "head_sha": identity["executed_head_sha"],
        "head_branch": branch, "event": "push", "path": workflow,
        "status": "completed" if terminal else "in_progress", "conclusion": terminal["run_conclusion"] if terminal else None}
    run = api("GET", routes[prefix + "_run"])
    p._equal(run, expected, "exact current or original run")
    p._equal(run.get("repository"), {"full_name": t.REPOSITORY, "private": False}, "public run repository")
    history = api("GET", routes[prefix + "_history"])
    if source_claim is None:
        p._equal(history, {"total_count": 1}, "one source run only")
        t.require(type(history.get("workflow_runs")) is list and len(history["workflow_runs"]) == 1, "complete one-use source history")
        p._equal(history["workflow_runs"][0], expected, "same exact original/recovery history")
    else:
        t.require(prefix == "producer" and terminal is not None, "v2 never changes recovery's own history policy")
        owner = source_claim["identity"]
        c.validate_bound_admission(source_claim["admission"], owner)
        t.require(owner["run_id"] == identity["run_id"] and owner["source_sha"] == identity["executed_head_sha"]
            and (owner["source_branch"], owner["workflow"], owner["job_key"]) == (branch, workflow, job_key),
            "same source-bound original owner and scope")
        p._equal(run, c.bound_run_row(owner), "same actual original owner run identity")
        p._equal(run.get("head_repository"), {"full_name": t.REPOSITORY, "private": False}, "public original head repository")
        observation = p.history_observation(api, routes[prefix + "_history"], owner, history, "posterior")
        if observations is not None: observations["producer_history"] = observation
        p._equal(run, {"run_number": 1, "run_attempt": 1}, "eligible original immutable owner")
    jobs = api("GET", routes[prefix + "_jobs"])
    t.require(type(jobs.get("total_count")) is int and 1 <= jobs["total_count"] <= 100
        and type(jobs.get("jobs")) is list and len(jobs["jobs"]) == jobs["total_count"], "complete bounded jobs")
    matches = [job for job in jobs["jobs"] if type(job) is dict and job.get("name") == job_key]
    t.require(len(matches) == 1, "unique fixed nonmatrix job")
    job = matches[0]
    p._equal(job, {"run_id": identity["run_id"], "head_sha": identity["executed_head_sha"], "head_branch": branch,
        "run_url": "https://api.github.com" + routes[prefix + "_run"], "status": expected["status"],
        "conclusion": terminal["job_conclusion"] if terminal else None}, "actual numeric job binding")
    if "run_attempt" in job:
        p._equal(job, {"run_attempt": 1}, "job attempt one")
    t.require(type(job.get("id")) is int and 0 < job["id"] < 10 ** 20, "actual numeric job identity")
    if source_claim is not None:
        t.require(job["id"] == source_claim["identity"]["job_id"], "same independent original owner job")
    if terminal:
        steps = job.get("steps")
        t.require(type(steps) is list and 2 <= len(steps) <= 100 and all(type(step) is dict for step in steps), "complete original steps")
        for name, key in ((HOST_STEP, "host_step_conclusion"), (PUBLISHER_STEP, "publisher_step_conclusion")):
            rows = [step for step in steps if step.get("name") == name]
            t.require(len(rows) == 1, "one exact original host/publisher step")
            p._equal(rows[0], {"status": "completed", "conclusion": terminal[key]}, "preserve original step conclusion")
    return job["id"]


def recover(env, event, control, source, api, raw_factory=CheckpointReader):
    """Read/validate the whole original checkpoint before the sole allowed POST."""
    authority = recovery_gate(control)
    run = context(env, event, control)
    t.require(type(source) is dict and set(source) == {"source_sha", "source_tree", "source_closure_sha256", "commit_timestamp"}
        and source["source_sha"] == run["executed_head_sha"], "bound recovery source")
    for key, pattern in (("source_sha", t.HEAD), ("source_tree", t.HEAD), ("source_closure_sha256", t.SHA)):
        t.require(type(source[key]) is str and pattern.fullmatch(source[key]), "exact recovery source pins")
    routes = endpoints(run, source, authority)
    p._equal(api("GET", routes["repo"]), {"full_name": t.REPOSITORY, "private": False}, "canonical public API repository")
    recovery_job = _verify_run(api, routes, "recovery", run, SOURCE_BRANCH, WORKFLOW, JOB_KEY)
    original = authority["producer_identity"]
    # Preserve the original job pin while reading its immutable checkpoint.
    # Independent producer run/history/jobs are read only after archive binding.
    producer_job = authority["numeric_job_id"]
    source_rows = None
    for prefix, sha, tree, stamp in (("recovery", source["source_sha"], source["source_tree"], source["commit_timestamp"]),
        ("producer", original["executed_head_sha"], authority["source_tree"], authority["source_commit_timestamp"])):
        commit = api("GET", routes[prefix + "_source"])
        p._equal(commit, {"sha": sha}, "exact reviewed source commit")
        p._equal(commit.get("tree"), {"sha": tree}, "exact reviewed source tree")
        t.require(type(commit.get("committer")) is dict and p._utc_timestamp(commit["committer"].get("date")) == stamp, "exact original/recovery source UTC")
        rows = p.verify_tree(api("GET", routes[prefix + "_tree"]), tree)
        if prefix == "producer": source_rows = rows
    checkpoint = api("GET", routes["checkpoint_ref"])
    p._equal(checkpoint, {"ref": authority["checkpoint_ref"]}, "exact pinned checkpoint ref")
    p._equal(checkpoint.get("object"), {"type": "commit", "sha": authority["checkpoint_commit"]}, "checkpoint ref immutable target")
    body, commit = p.commit_body(authority["checkpoint_tree"], original_release(authority))
    p.verify_commit(api("GET", routes["checkpoint_commit"]), commit, body)
    complete = p.verify_tree(api("GET", routes["checkpoint_tree"]), authority["checkpoint_tree"])
    read = raw_factory(commit, deadline=api.deadline)
    outer = read(t.manifest_path())
    t.require(t.sha(outer) == authority["outer_manifest_sha256"], "pinned whole original outer manifest")
    manifest = t.parse_manifest(outer, original, authority["source_closure_sha256"], producer_job)
    t.require(manifest["archive_sha256"] == authority["archive_sha256"] and manifest["archive_bytes"] == authority["archive_bytes"], "pinned original archive identity")
    chunks = [(row["path"], read(row["path"])) for row in manifest["chunks"]]
    t.require(len(outer) + sum(len(raw) for _, raw in chunks) <= MAX_RAW_BODY, "aggregate bounded complete immutable read")
    payload = t.reassemble(outer, chunks, original, authority["source_closure_sha256"], producer_job)
    t.validate_archive(payload, original)
    proof = t.validate_source_binding(payload, original, authority["source_closure_sha256"], source_rows, producer_job, authority["source_tree"])
    producer_policy = t.LEGACY_POLICY if proof is None else c.require_current_scope(proof["identity"])
    if proof is not None:
        route = p.producer_history_route(proof["identity"])
        if isinstance(api, RecoveryAPI):
            t.require(api.bind_producer_history(payload, source_rows) == route, "only verified producer history ticket")
        routes["producer_history"] = route
    observations = {}
    observed_job = _verify_run(api, routes, "producer", original, producer_policy["source_branch"],
        producer_policy["workflow"], "asr18", authority["terminal"], source_claim=proof, observations=observations)
    t.require(observed_job == producer_job, "original pinned independent numeric job")
    # This is the same real native/host validator as ordinary publication, never
    # a synthetic transport selector or caller-supplied schema callback.
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        host = t.decode(archive.read("host-receipt.json"))
    t.require(host["status"] == authority["terminal"]["host_receipt_status"], "preserve original host receipt status")
    files = {t.manifest_path(): outer, **dict(chunks)}
    rebuilt = t.build_file_set(payload, original, authority["source_closure_sha256"], producer_job)
    t.require(files == rebuilt, "complete original canonical file set")
    directory = t.POLICY["directory"]
    t.require(not any(row["path"] == directory or row["path"].startswith(directory + "/") or directory.startswith(row["path"] + "/")
        for row in source_rows), "original source has no result directory")
    added = [{"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for name, raw in sorted(files.items())]
    t.require(sorted(complete, key=lambda row: row["path"]) == sorted(source_rows + added, key=lambda row: row["path"]), "checkpoint exact preserved original source and complete package")
    t.require(t.tree_identity(source_rows + added)[0] == authority["checkpoint_tree"], "same original precomputed tree")
    api("GET", routes["final_ref"], expected=404)
    p.finalize_exact_ref(api, t.POLICY["result_branch"], commit)
    return {"schema": "k1-asr18-transfer-recovery-receipt-v1", "status": "exact_original_checkpoint_final_ref_verified",
        "producer_identity": dict(original), "producer_numeric_job_id": producer_job,
        **observations,
        "original_terminal": dict(authority["terminal"]), "original_run_status": "completed",
        "original_job_status": "completed", "original_publisher_reclassified": False,
        "recovery_identity": run, "recovery_numeric_job_id": recovery_job,
        "recovery_source_closure_sha256": source["source_closure_sha256"],
        "checkpoint_ref": authority["checkpoint_ref"], "checkpoint_commit": commit, "checkpoint_tree": authority["checkpoint_tree"],
        "final_ref": "refs/heads/" + t.POLICY["result_branch"], "archive_sha256": authority["archive_sha256"],
        "archive_bytes": authority["archive_bytes"], "outer_manifest_sha256": authority["outer_manifest_sha256"],
        "native_success_established": False, "new_native_calls_authorized": False, "retry_authorized": False}


def main(argv=None):
    # No environment, token, process, network or output before the disabled gate.
    control = c.read_json(HERE / "transfer-recovery-release.json", cap=65536)
    recovery_gate(control)
    argv = sys.argv[1:] if argv is None else argv
    t.require(argv == [] and sys.flags.isolated and sys.dont_write_bytecode, "fixed isolated -I -B recovery CLI only")
    env = {key: os.environ[key] for key in _CONTEXT_KEYS if key in os.environ}
    event = c.read_json(Path(env.get("GITHUB_EVENT_PATH", "")), cap=1024 * 1024)
    run = context(env, event, control)
    source = bound_source(env, event, control)
    # This is the first token access. No native host, runtime or setup is run.
    api = RecoveryAPI(os.environ.get("GITHUB_TOKEN"), run, source, control)
    def timeout(_signum, _frame):
        raise TimeoutError("bounded transfer recovery deadline")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        result = recover(env, event, control, source, api)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    raw = t.canonical(result)
    t.require(len(raw) <= 16 * 1024, "metadata-only recovery receipt cap")
    print(raw.decode("ascii"), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except RecoveryDisabled:
        raise SystemExit("ASR transfer recovery disabled or review absent.") from None
    except BaseException:
        raise SystemExit("ASR transfer recovery blocked or unconfirmed; no retry.") from None
