"""Fixed trusted-host publisher wrapper. Disabled release is the first gate.

No model imports, subprocess credential inheritance, arbitrary paths/purposes,
tracebacks, environment dumps or audio-bearing logs. Root separately releases a
single fixture or real TTS publication after complete source review.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_TERMINAL = 16 * 1024
MAX_RELEASE = 16 * 1024
PURPOSES = ("fixture", "tts")
SAFE_CLASSES = {"ValueError", "RuntimeError", "TimeoutError", "TimeoutExpired", "OSError", "FileNotFoundError",
    "FileExistsError", "PermissionError", "ImportError", "ModuleNotFoundError", "MemoryError", "CalledProcessError",
    "AttributeError", "TypeError", "KeyboardInterrupt", "SystemExit", "PublicationError"}
WRAPPER_STAGES = {"release_gate", "verify_source", "verify_tts_release", "fresh_context", "clean_checkout", "read_members", "build_package", "token_access", "publication"}
PUBLISHER_STAGES = {"validate_local_package", "verify_actual_run_source", "verify_destination_absent", "create_blobs",
    "create_tree", "create_commit", "create_ref", "immutable_postpublish_readback", "readback_verified"}


class Disabled(RuntimeError):
    pass


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def open_directory(root):
    root = Path(root)
    require(root.is_absolute() and str(root) == os.path.abspath(root), "canonical directory")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_regular(directory_fd, name, maximum):
    require(type(name) is str and name and "/" not in name and name not in (".", ".."), "fixed basename")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, "bounded unaliased file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(fd)
        require(len(raw) == before.st_size and len(raw) <= maximum and all(getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "stable file bytes")
        return raw
    finally:
        os.close(fd)


def read_path(path, maximum):
    path = Path(path)
    fd = open_directory(path.parent)
    try: return read_regular(fd, path.name, maximum)
    finally: os.close(fd)


def publication_release(purpose):
    """Before any environment/token access, source imports, subprocess or network."""
    require(purpose in PURPOSES and type(purpose) is str, "fixed purpose")
    try:
        value = decode(read_path(HERE / "publication-release.json", MAX_RELEASE))
    except FileNotFoundError:
        raise Disabled() from None
    require(type(value) is dict, "release object")
    # Missing, false, integer-like or null gates all remain disabled.
    gates = value.get("purposes")
    if value.get("enabled") is not True or type(gates) is not dict or gates.get(purpose) is not True:
        raise Disabled()
    require(set(value) == {"schema", "experiment", "enabled", "purposes", "independent_source_review",
        "root_publication_release", "prior_claim_history_sha256", "commit_timestamp"}, "exact publication release")
    require(value["schema"] == "k1-punctuation6-publication-release-v1" and value["experiment"] == "k1-punctuation6-v1", "release scope")
    require(set(gates) == set(PURPOSES) and all(type(v) is bool for v in gates.values()), "exact boolean purpose gates")
    require(value["independent_source_review"] is True and value["root_publication_release"] is True, "review and root release")
    require(type(value["prior_claim_history_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["prior_claim_history_sha256"]), "frozen prior history")
    require(type(value["commit_timestamp"]) is str and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value["commit_timestamp"]), "frozen commit timestamp")
    return value


def components():
    # Deliberately after the disabled gate; direct python -I CLI works even before
    # sibling-module resolution. Every imported file belongs to the source freeze.
    sys.path.insert(0, str(HERE))
    import data_transport
    import data_publisher
    import tts_gate
    return data_transport, data_publisher, tts_gate


def clean_git_env():
    return {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}


def clean_checkout(identity):
    def git(*args):
        return subprocess.run(["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-C", str(ROOT), *args],
            env=clean_git_env(), check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    require(git("rev-parse", "HEAD") == identity["executed_head_sha"], "actual checkout commit")
    require(not git("status", "--porcelain", "--untracked-files=all"), "clean complete checkout")
    tree = git("rev-parse", "HEAD^{tree}")
    require(re.fullmatch(r"[0-9a-f]{40}", tree), "actual checkout tree")
    return tree


def safe_context_environment(environment):
    # Read fixed non-secret values only; never copy or iterate the host environment.
    keys = ("GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_JOB", "GITHUB_RUN_ATTEMPT",
        "GITHUB_RUN_ID", "GITHUB_SHA", "RUNNER_ENVIRONMENT", "RUNNER_OS", "RUNNER_ARCH", "GITHUB_EVENT_PATH", "RUNNER_TEMP")
    return {key: environment[key] for key in keys}


def read_tts_members(runtime, transport):
    """Read only the 23 fixed paths; reject extras, aliases, devices and races."""
    root = runtime / "publication-members"
    fd = open_directory(root)
    try:
        names = set(os.listdir(fd))
        require(transport.TTS_JSON <= names <= transport.TTS_JSON | {"raw", "derived", "claims"}, "exact publication root entries")
        result = {name: read_regular(fd, name, 64 * 1024) for name in sorted(transport.TTS_JSON)}
        for kind in ("raw", "derived", "claims"):
            if kind not in names: continue
            child = os.open(kind, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                entries = os.listdir(child)
                suffix = ".started.json" if kind == "claims" else ".wav"
                allowed = {cell + suffix for cell in transport.CELL_IDS}
                require(set(entries) <= allowed, "exact publication leaf names")
                cap = 64 * 1024 if kind == "claims" else (24000 * 12 * 4 + 256 if kind == "raw" else 16000 * 12 * 2 + 256)
                for name in sorted(entries): result[kind + "/" + name] = read_regular(child, name, cap)
            finally:
                os.close(child)
        require(set(os.listdir(fd)) == names, "publication root did not change")
        require(set(result) <= transport.TTS_ALLOWED and sum(len(raw) for raw in result.values()) <= transport.MAX_ARCHIVE, "bounded exact members")
        return result
    finally:
        os.close(fd)


def receipt_directory(runtime, purpose):
    if purpose == "fixture":
        parent = open_directory(runtime.parent)
        try:
            os.mkdir(runtime.name, mode=0o700, dir_fd=parent)  # Exclusive fresh fixture workspace.
            os.fsync(parent)
        finally: os.close(parent)
    fd = open_directory(runtime)
    os.close(fd)
    return runtime


class PinnedJobAPI:
    """No additional request: bind API numeric job to the original host claim."""
    def __init__(self, api, jobs_route, purpose, numeric_job_id=None):
        require(purpose in PURPOSES, "fixed publication job")
        if purpose == "tts": require(type(numeric_job_id) is int and 0 < numeric_job_id < 10**20, "original numeric job")
        else: require(numeric_job_id is None, "fixture has no generation job claim")
        self.api = api; self.jobs_route = jobs_route; self.numeric_job_id = numeric_job_id
        self.job_key = "k1-tts" if purpose == "tts" else "writer-fixture"

    def bind_result(self, commit, tree):
        return self.api.bind_result(commit, tree)

    def __call__(self, method, route, body=None, expected=200):
        result = self.api(method, route, body, expected)
        if method == "GET" and route == self.jobs_route:
            require(type(result) is dict and type(result.get("jobs")) is list, "actual job response")
            matches = [row for row in result["jobs"] if type(row) is dict and row.get("name") == self.job_key]
            require(len(matches) == 1 and type(matches[0].get("labels")) is list and matches[0]["labels"] == ["ubuntu-24.04"],
                "actual fixed standard hosted job")
            if self.numeric_job_id is not None:
                require(type(matches[0].get("id")) is int and matches[0]["id"] == self.numeric_job_id,
                    "original generation job numeric identity")
        return result


def execute(purpose, environment=None, progress=None):
    progress = {} if progress is None else progress
    progress["stage"] = "release_gate"
    released = publication_release(purpose)
    transport, publisher, gate = components()
    progress["stage"] = "verify_source"
    source = gate.verify_source(ROOT)
    require(type(source) is dict and type(source.get("source_freeze_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", source["source_freeze_sha256"]), "verified source closure")
    if purpose == "tts":
        progress["stage"] = "verify_tts_release"
        _, tts_source = gate.verify_release(ROOT)
        require(type(tts_source) is dict and tts_source.get("source_freeze_sha256") == source["source_freeze_sha256"],
            "same TTS execution/publication source closure")
    progress["stage"] = "fresh_context"
    environment = os.environ if environment is None else environment
    env = safe_context_environment(environment)
    event = decode(read_path(Path(env["GITHUB_EVENT_PATH"]), 1024 * 1024))
    require(env["RUNNER_ENVIRONMENT"] == "github-hosted" and env["RUNNER_OS"] == "Linux" and env["RUNNER_ARCH"] == "X64", "fixed standard hosted context")
    require(type(env["GITHUB_SHA"]) is str and re.fullmatch(r"[0-9a-f]{40}", env["GITHUB_SHA"]), "actual source head")
    provisional = {"schema": "k1-publisher-testable-release-v1", "purpose": purpose,
        "source_sha": env["GITHUB_SHA"], "source_tree": "0" * 40,
        "source_closure_sha256": source["source_freeze_sha256"],
        "prior_claim_history_sha256": released["prior_claim_history_sha256"], "commit_timestamp": released["commit_timestamp"]}
    identity = publisher.context(env, event, provisional)
    if purpose == "tts": require(gate.fresh_context(env, event) == identity, "shared actual TTS identity")
    progress["stage"] = "clean_checkout"
    source_tree = clean_checkout(identity)
    release = {**provisional, "source_tree": source_tree}
    publisher.validate_release(release)
    runtime = Path(env["RUNNER_TEMP"]) / ("k1-punctuation6-writer-fixture-r1" if purpose == "fixture" else "k1-punctuation6-tts-v1")
    require(runtime.is_absolute() and str(runtime) == os.path.abspath(runtime), "fixed canonical runtime")
    semantic_validator = None
    if purpose == "fixture":
        progress["stage"] = "build_package"
        payload = transport.fixture_archive(identity, source["source_freeze_sha256"])
    else:
        progress["stage"] = "read_members"
        members = read_tts_members(runtime, transport)
        # Only fixed stdlib validator; never import or invoke tts_worker/model code.
        from tts_semantics import validate_tts_members
        semantic_validator = validate_tts_members
        claim = decode(members["claim.json"])
        require(claim.get("identity") == identity and claim.get("source_freeze_sha256") == source["source_freeze_sha256"]
            and claim.get("source_tree") == source_tree, "current source/claim binding")
        progress["stage"] = "build_package"
        payload = transport.build_archive(members, identity, "tts", semantic_validator)
    directory = receipt_directory(runtime, purpose)
    progress["stage"] = "token_access"
    token = environment["KWS_DATA_TOKEN"]  # Last; never passed to any subprocess.
    api = publisher.GitHubAPI(token, identity, release)
    api = PinnedJobAPI(api, publisher.endpoints(identity, release)["jobs"], purpose,
        claim["preflight"]["numeric_job_id"] if purpose == "tts" else None)
    progress["stage"] = "publication"
    return publisher.run_once(env, event, release, payload, api, publisher.RawReader, directory, semantic_validator)


def safe_identity(value, purpose):
    if type(value) is not dict or set(value) != {"repository", "run_id", "run_attempt", "job_key", "executed_head_sha"}: return None
    if value["repository"] != "jiying2007/kws-data" or type(value["run_id"]) is not int or not 0 < value["run_id"] < 10**20: return None
    if type(value["run_attempt"]) is not int or value["run_attempt"] != 1: return None
    if value["job_key"] != ("writer-fixture" if purpose == "fixture" else "k1-tts"): return None
    if type(value["executed_head_sha"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", value["executed_head_sha"]): return None
    return value


def safe_receipt(value, purpose):
    """Projection only: never echo unknown strings, arbitrary errors or contents."""
    if type(value) is not dict: return {}
    result = {}
    identity = safe_identity(value.get("identity"), purpose)
    if identity is not None: result["identity"] = identity
    for key in ("result_commit", "result_tree", "expected_result_commit", "expected_result_tree"):
        item = value.get(key)
        if type(item) is str and re.fullmatch(r"[0-9a-f]{40}", item): result[key] = item
    for key in ("source_closure_sha256", "archive_sha256", "outer_manifest_sha256"):
        item = value.get(key)
        if type(item) is str and re.fullmatch(r"[0-9a-f]{64}", item): result[key] = item
    for key, maximum in (("numeric_job_id", 10**20 - 1), ("archive_bytes", 12 * 1024 * 1024),
        ("raw_gets", 49), ("readback_body_bytes", 16 * 1024 * 1024), ("confirmed_blob_count", 49)):
        item = value.get(key)
        if type(item) is int and 0 <= item <= maximum: result[key] = item
    if type(value.get("http_status")) is int and 100 <= value["http_status"] <= 599: result["http_status"] = value["http_status"]
    for key in ("publication_confirmed", "readback_confirmed", "semantic_inference_success_not_established_by_publisher"):
        if type(value.get(key)) is bool: result[key] = value[key]
    if type(value.get("stage")) is str and value["stage"] in PUBLISHER_STAGES: result["publisher_stage"] = value["stage"]
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    purpose = argv[1] if len(argv) == 2 and argv[0] == "--purpose" and type(argv[1]) is str and argv[1] in PURPOSES else None
    progress = {"stage": "release_gate"}
    result = {"schema": "k1-punctuation6-publication-terminal-v1", "purpose": purpose,
        "retry_authorized": False, "new_generation_authorized": False}
    code = 2
    try:
        require(purpose is not None, "fixed publisher argument")
        receipt = execute(purpose, progress=progress)
        require(type(receipt) is dict and receipt.get("status") == "same_identity_readback_verified", "exact successful terminal receipt")
        result.update(status="same_identity_readback_verified", **safe_receipt(receipt, purpose)); code = 0
    except Disabled:
        result.update(status="disabled", stage="release_gate")
    except BaseException as error:
        result.update(status="publication_failed_or_unconfirmed_no_retry",
            stage=progress["stage"] if progress.get("stage") in WRAPPER_STAGES else "publication",
            error_class=type(error).__name__ if type(error).__name__ in SAFE_CLASSES else "OtherException",
            **safe_receipt(getattr(error, "publication_receipt", None), purpose))
    raw = canonical(result)
    if len(raw) > MAX_TERMINAL:
        raw = canonical({"schema": "k1-punctuation6-publication-terminal-v1", "status": "publication_failed_or_unconfirmed_no_retry",
            "retry_authorized": False, "new_generation_authorized": False})
        code = 2
    sys.stdout.write(raw.decode("utf-8") + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
