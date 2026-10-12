"""Fixed, independent audio-free prepare diagnostic transport; no model/network IO.

Unchanged reviewed ASR metadata validators are imported read-only. No ASR host,
ledger, started slot, terminal slot or recognition receipt is fabricated/admitted.
Only complete fixed source/native closures and allowlisted metadata are retained.
"""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import os
import random
import re
import stat
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "research/k1-punctuation6-asr18-v1"
for folder in (ROOT / "tools", OLD):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
import campaign_contract as c
if Path(c.__file__).resolve() != OLD / "campaign_contract.py":
    raise ValueError("fixed reviewed metadata contract origin")
_spec = importlib.util.spec_from_file_location("_k1_diagnostic_reviewed_transport", OLD / "evidence_transport.py")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
REPOSITORY = "jiying2007/kws-pipeline"
EXPERIMENT = "k1-punctuation6-prepare-diagnostic-r2"
SOURCE_REL = "research/k1-punctuation6-prepare-diagnostic-v1/"
PURPOSE = "prepare_diagnostic"
MIB = 1024 ** 2
MAX_ARCHIVE = MAX_COMPRESSED = 12 * MIB
MAX_UNCOMPRESSED = 32 * MIB
MAX_MEMBER = 8 * MIB
MAX_MEMBERS = 4096
MAX_CHUNK = 256 * 1024
MAX_CHUNKS = 48
MAX_MANIFEST = 16 * 1024
MAX_BODY = 16 * MIB
SHA, HEAD = legacy.SHA, legacy.HEAD
MEMBER_MANIFEST = "__prepare_diagnostic_member_manifest__.json"
POLICY = {"source_branch": "research/" + EXPERIMENT,
    "result_branch": "evidence/" + EXPERIMENT,
    "directory": "research/" + EXPERIMENT + "/evidence/prepare-diagnostic-v1",
    "workflow": ".github/workflows/k1-punctuation6-prepare-diagnostic-v1.yml", "job_key": PURPOSE}
require, canonical, decode, sha = legacy.require, legacy.canonical, legacy.decode, legacy.sha
path, git_sha, tree_identity = legacy.path, legacy.git_sha, legacy.tree_identity
FAMILIES = ("qwen06", "sensevoice", "whisper")
FREEZE = "authority/prepare-freeze.json"
SOURCE_FREEZE = "source/" + SOURCE_REL + "prepare-freeze.json"
PACKET_NAMES = frozenset(("job.json", "decoder-inputs.json", "worker-input-binding.json", "execution-plan.json", "host-binding.json"))
ROOT_JSON = frozenset(("prepare-host-receipt.json", "prepare-run-preflight.json", "input-freeze.json",
    "public-screening-receipt.json", "acquisition-receipt.json", "image-transfer.json", "asr-setup-public.json",
    "cpu-receipt.json", "cleanup-receipt.json", "supervision-receipt.json", FREEZE,
    "authority/prepare-claim.json", "policy/policy.json", "policy/policy-freeze.json", "policy/capacity-proof.json"))
WORKER_JSON = (legacy.WORKER_JSON - {"prepared.json", "run-identity.json", "blind-job.json", "execution-plan.json", "worker-complete.json", "worker-failure.json", "progress.json"}) | {"prepare-receipt.json", "worker-complete.json"}

def _prepare_contract():
    spec = importlib.util.spec_from_file_location("_k1_diagnostic_prepare_contract", HERE / "prepare_contract.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def policy(purpose=PURPOSE):
    require(type(purpose) is str and purpose == PURPOSE, "fixed prepare diagnostic purpose")
    return dict(POLICY)

def identity(value, purpose=PURPOSE):
    policy(purpose)
    require(type(value) is dict and set(value) == {"repository", "run_id", "run_attempt", "job_key", "executed_head_sha"}, "fixed diagnostic identity")
    require(value["repository"] == REPOSITORY and value["job_key"] == PURPOSE, "same repo independent prepare job")
    require(type(value["run_id"]) is int and 0 < value["run_id"] < 10 ** 20 and type(value["run_attempt"]) is int and value["run_attempt"] == 1, "actual run attempt one")
    require(type(value["executed_head_sha"]) is str and HEAD.fullmatch(value["executed_head_sha"]), "source Git identity")
    _prepare_contract().validate_identity(value)

def _fields(value, keys, reason):
    require(type(value) is dict and set(value) == set(keys), reason)

def _source_members(files):
    raw = files.get(FREEZE)
    require(raw is not None, "complete independent prepare freeze required")
    freeze = decode(raw)
    _fields(freeze, {"schema", "experiment", "worker_code", "inference_code", "setup_code", "host_code"}, "fixed prepare source freeze fields")
    require(freeze["schema"] == "k1-prepare-source-freeze-v1" and freeze["experiment"] == EXPERIMENT and canonical(freeze) == raw, "canonical independent prepare freeze")
    for key in ("worker_code", "inference_code", "setup_code", "host_code"):
        require(type(freeze[key]) is dict and freeze[key], "complete fixed source maps")
        for name, row in freeze[key].items():
            path(name)
            _fields(row, {"bytes", "sha256"}, "source identity fields")
            require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_MEMBER and type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]), "bounded source identity")
            require(freeze["host_code"].get(name) == row, "whole source subclosure bound by host closure")
    require(freeze["worker_code"] == freeze["inference_code"], "exact isolated worker closure")
    names = {"source/" + name: row for name, row in freeze["host_code"].items()}
    require(SOURCE_FREEZE not in names, "freeze self-reference prohibited")
    names[SOURCE_FREEZE] = {"bytes": len(raw), "sha256": sha(raw)}
    require({name for name in files if name.startswith("source/")} == set(names), "complete exact source roster; no omitted or extra source")
    for member, row in names.items():
        require(member in files and {"bytes": len(files[member]), "sha256": sha(files[member])} == row, "every exact public source body retained")
        require(PurePosixPath(member).suffix in (".py", ".json", ".md", ".yml", ".yaml", ".txt") or legacy.SOURCE_TEXT_EXCEPTIONS.get(member) == row, "reviewed public source text only")
        files[member].decode("utf-8")
    return names

def _worker_path(name):
    match = re.fullmatch(r"output/(qwen06|sensevoice|whisper)/(.+)", name)
    return bool(match and match[2] in WORKER_JSON)

def _public_failure(value):
    if value is None: return
    import failure_observer as fo
    require(Path(fo.__file__).resolve() == OLD / "failure_observer.py", "fixed public error validator origin")
    require(type(value) is dict and {"error_type", "error_text"} <= set(value) <= {"error_type", "error_text", "errno", "returncode", "validation_code"}, "fixed public error fields")
    require(value["error_type"] in {kind.__name__ for kind in fo.ERROR_TYPES} | {"UnlistedError"} and value["error_text"] == "omitted_by_fixed_privacy_policy", "no arbitrary exception content")
    if "errno" in value: require(type(value["errno"]) is int and value["errno"] in fo.ERRNOS, "fixed public errno")
    if "returncode" in value: require(type(value["returncode"]) is int and -255 <= value["returncode"] <= 255 and value["error_type"] == "CalledProcessError", "bounded public subprocess status")
    if "validation_code" in value: require(value["error_type"] == "ValueError" and value["validation_code"] in frozenset(fo.VALIDATION_CODES.values()), "exact source-fixed public enum")


def _zip_files(files, run_identity):
    """Internal bounded compressor; no admission/privacy claim by itself."""
    identity(run_identity)
    require(type(files) is dict and 1 <= len(files) <= MAX_MEMBERS and MEMBER_MANIFEST not in files, "member count cap")
    require(all(type(raw) is bytes and len(raw) <= MAX_MEMBER for raw in files.values()), "member cap")
    manifest = {"schema": "k1-prepare-diagnostic-member-manifest-v1", "identity": run_identity,
        "members": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(files.items())}}
    m = canonical(manifest)
    expanded = len(m) + sum(map(len, files.values()))
    require(len(m) <= MAX_MEMBER and expanded <= MAX_UNCOMPRESSED, "complete expanded evidence cap; do not omit records")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in [(MEMBER_MANIFEST, m), *sorted(files.items())]:
            path(name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, raw)
    payload = buffer.getvalue()
    require(len(payload) <= MAX_COMPRESSED, "complete compressed evidence cap; do not omit records")
    return payload, manifest, expanded



def validate_archive(payload, expected_identity):
    identity(expected_identity)
    require(type(payload) is bytes and 1 <= len(payload) <= MAX_COMPRESSED, "archive compressed cap")
    require(len(payload) >= 22 and payload[-22:-18] == b"PK\x05\x06" and payload[-2:] == b"\0\0", "no ZIP comment/trailer")
    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        require(2 <= len(infos) <= MAX_MEMBERS + 1 and len({i.filename for i in infos}) == len(infos), "complete unique member count")
        expanded = sum(i.file_size for i in infos)
        require(0 < expanded <= MAX_UNCOMPRESSED and infos[0].header_offset == 0 and not archive.comment, "no preamble/expanded cap")
        for item in infos:
            path(item.filename)
            require(not item.is_dir() and not item.flag_bits & 1 and item.compress_type == zipfile.ZIP_DEFLATED
                and stat.S_IFMT(item.external_attr >> 16) == stat.S_IFREG and 0 <= item.file_size <= MAX_MEMBER,
                "safe bounded ZIP regular member")
            with archive.open(item) as stream: raw = stream.read(MAX_MEMBER + 1)
            require(len(raw) == item.file_size, "whole member read")
            files[item.filename] = raw
    require(MEMBER_MANIFEST in files, "member manifest required")
    mraw = files.pop(MEMBER_MANIFEST)
    manifest = decode(mraw)
    require(type(manifest) is dict and set(manifest) == {"schema", "identity", "members"}
        and manifest["schema"] == "k1-prepare-diagnostic-member-manifest-v1" and canonical(manifest) == mraw
        and canonical(manifest["identity"]) == canonical(expected_identity), "exact canonical member manifest")
    measured = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}
    require(canonical(manifest["members"]) == canonical(measured), "all members bound")
    validate_members(files, expected_identity)
    # Reconstructing the canonical container rejects covert unindexed ZIP bytes,
    # duplicate local headers, trailing data and comments, with no extraction.
    rebuilt, _, _ = _zip_files(files, expected_identity)
    require(rebuilt == payload, "canonical complete ZIP only")
    return manifest, {"archive_bytes": len(payload), "archive_sha256": sha(payload),
        "manifest_sha256": sha(mraw), "member_count": len(files), "uncompressed_bytes": expanded}



def manifest_path(purpose=PURPOSE):
    return policy(purpose)["directory"] + "/manifest.json"


def chunk_path(index, purpose=PURPOSE):
    require(type(index) is int and 0 <= index < MAX_CHUNKS, "chunk index")
    return policy(purpose)["directory"] + f"/chunks/{index:04d}.bin"


def build_file_set(payload, run_identity, freeze_sha256, numeric_job_id, purpose=PURPOSE):
    identity(run_identity, purpose)
    manifest, measured = validate_archive(payload, run_identity)
    require(manifest["members"]["authority/prepare-freeze.json"]["sha256"] == freeze_sha256, "archive independently approved source closure")
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256), "source closure hash")
    require(type(numeric_job_id) is int and 0 < numeric_job_id < 10 ** 20, "numeric job identity")
    chunks = [payload[i:i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
    m = {"schema": "k1-prepare-diagnostic-evidence-branch-manifest-v1", "identity": run_identity, "purpose": purpose,
        "source_closure_sha256": freeze_sha256, "numeric_job_id": numeric_job_id, **measured,
        "chunk_bytes": MAX_CHUNK, "chunk_count": len(chunks),
        "chunks": [{"index": i, "path": chunk_path(i), "bytes": len(raw), "sha256": sha(raw)} for i, raw in enumerate(chunks)]}
    raw = canonical(m)
    require(len(raw) <= MAX_MANIFEST and len(chunks) <= MAX_CHUNKS, "outer manifest/chunk cap")
    return {manifest_path(): raw, **{chunk_path(i): raw for i, raw in enumerate(chunks)}}


def parse_manifest(raw, run_identity, freeze_sha256, numeric_job_id, purpose=PURPOSE):
    identity(run_identity, purpose)
    require(type(raw) is bytes and 0 < len(raw) <= MAX_MANIFEST, "bounded outer manifest")
    m = decode(raw)
    require(type(m) is dict and set(m) == {"schema", "identity", "purpose", "source_closure_sha256", "numeric_job_id",
        "archive_bytes", "archive_sha256", "manifest_sha256", "member_count", "uncompressed_bytes", "chunk_bytes", "chunk_count", "chunks"}, "outer exact fields")
    require(canonical(m) == raw and m["schema"] == "k1-prepare-diagnostic-evidence-branch-manifest-v1" and m["purpose"] == purpose,
        "canonical prepare diagnostic outer manifest")
    identity(m["identity"])
    require(canonical(m["identity"]) == canonical(run_identity), "same run identity")
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256) and m["source_closure_sha256"] == freeze_sha256,
        "independent source closure binding")
    require(type(numeric_job_id) is int and 0 < numeric_job_id < 10 ** 20 and type(m["numeric_job_id"]) is int
        and m["numeric_job_id"] == numeric_job_id, "exact numeric job identity")
    for key, cap in (("archive_bytes", MAX_ARCHIVE), ("uncompressed_bytes", MAX_UNCOMPRESSED), ("member_count", MAX_MEMBERS), ("chunk_count", MAX_CHUNKS)):
        require(type(m[key]) is int and 1 <= m[key] <= cap, "bounded outer measurement")
    require(type(m["chunk_bytes"]) is int and m["chunk_bytes"] == MAX_CHUNK and
        m["chunk_count"] == (m["archive_bytes"] + MAX_CHUNK - 1) // MAX_CHUNK, "fixed chunk geometry")
    for key in ("archive_sha256", "manifest_sha256"):
        require(type(m[key]) is str and SHA.fullmatch(m[key]), "outer digest")
    require(type(m["chunks"]) is list and len(m["chunks"]) == m["chunk_count"], "complete ordered chunks")
    for i, row in enumerate(m["chunks"]):
        require(type(row) is dict and set(row) == {"index", "path", "bytes", "sha256"} and type(row["index"]) is int
            and row["index"] == i and row["path"] == chunk_path(i), "exact chunk fields/order/path")
        require(type(row["bytes"]) is int and row["bytes"] == min(MAX_CHUNK, m["archive_bytes"] - i * MAX_CHUNK), "exact chunk bytes")
        require(type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]), "chunk digest")
    return m


def reassemble(raw, chunks, run_identity, freeze_sha256, numeric_job_id, purpose=PURPOSE):
    m = parse_manifest(raw, run_identity, freeze_sha256, numeric_job_id, purpose)
    require(type(chunks) is list and len(chunks) == m["chunk_count"], "complete chunks")
    for pair, row in zip(chunks, m["chunks"]):
        require(type(pair) is tuple and len(pair) == 2 and pair[0] == row["path"] and type(pair[1]) is bytes
            and len(pair[1]) == row["bytes"] and sha(pair[1]) == row["sha256"], "same exact ordered chunk bytes")
    payload = b"".join(pair[1] for pair in chunks)
    _, measured = validate_archive(payload, run_identity)
    require(all(type(m[k]) is type(v) and m[k] == v for k, v in measured.items()), "whole evidence identity/measurement")
    return payload



def package_path(runner_temp, run_identity):
    identity(run_identity)
    root = c.safe_path(runner_temp)
    require(root.is_dir(), "existing trusted runner temporary directory")
    return root / (EXPERIMENT + "-evidence-" + str(run_identity["run_id"]) + "-attempt-1-" +
        run_identity["executed_head_sha"] + ".zip")


def write_package(runner_temp, payload, run_identity):
    import os
    validate_archive(payload, run_identity)
    target = package_path(runner_temp, run_identity)
    parent = c._open_directory(target.parent)
    try:
        fd = os.open(target.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            require(stream.write(payload) == len(payload), "whole evidence package write")
            stream.flush(); os.fsync(stream.fileno())
        os.fsync(parent)
    finally:
        os.close(parent)
    require(c.read_bytes(target, cap=MAX_COMPRESSED) == payload, "exact same retained evidence package")
    return target



def read_package(runner_temp, run_identity):
    payload = c.read_bytes(package_path(runner_temp, run_identity), cap=MAX_COMPRESSED)
    validate_archive(payload, run_identity)
    return payload



def validate_prepare_receipt(value, family, run, claim):
    _fields(value, {"schema", "identity", "family", "status", "model_loads_started", "model_loads_completed",
        "maximum_new_asr_calls", "native_decode_attempts", "audio_forward_calls", "audio_mounted", "recognition_result",
        "legacy_execution_contract_grants_asr", "metadata_packet_sha256", "source_freeze_sha256", "guard", "wall_seconds"}, "independent prepare receipt fields")
    expected = {"schema": "k1-prepare-worker-result-v1", "identity": run, "family": family,
        "status": "prepared_configuration_only", "model_loads_started": 1, "model_loads_completed": 1,
        "maximum_new_asr_calls": 0, "native_decode_attempts": 0, "audio_forward_calls": 0,
        "audio_mounted": False, "recognition_result": False, "legacy_execution_contract_grants_asr": False,
        "metadata_packet_sha256": claim["metadata_packet_sha256"], "source_freeze_sha256": claim["source_freeze_sha256"],
        "guard": {"schema": "k1-prepare-forward-guard-v1", "forbidden_attempts": 0,
            "python_entry_guard": "active_through_constructor_and_config",
            "coverage": "reviewed_fixed_python_native_and_module_forward_entrypoints", "universal_native_extension_proof": False}}
    require(all(c.exact_json(value[k], v) for k, v in expected.items()), "exact separate zero-forward preparation semantics")
    require(type(value["wall_seconds"]) in (int, float) and 0 <= value["wall_seconds"] <= 600, "bounded model preparation duration")


def _acquisition(value):
    _fields(value, {"schema", "data", "asr_response_body_bytes", "whisper", "reserved_total_bytes", "cap_bytes", "hard_wire_byte_cap"}, "fixed acquisition fields")
    require(value["schema"] == "k1-asr18-reserved-acquisition-v1" and value["reserved_total_bytes"] == c.LIMITS["acquisition_reserved_bytes"] and value["cap_bytes"] == 6*c.GIB and value["hard_wire_byte_cap"] is False, "unchanged reserved acquisition scope")
    require(value["asr_response_body_bytes"] is None or type(value["asr_response_body_bytes"]) is int and 0 <= value["asr_response_body_bytes"] <= 4*c.GIB, "bounded model setup bodies")
    data = value["data"]
    _fields(data, {"gets", "response_body_bytes", "response_body_reservation_bytes", "retries", "redirects", "extra_source_gets", "failed", "execution_enabled"}, "fixed data acquisition fields")
    require(type(data["gets"]) is int and 1 <= data["gets"] <= 49 and type(data["response_body_bytes"]) is int and 0 < data["response_body_bytes"] <= MAX_BODY and data["response_body_reservation_bytes"] == MAX_BODY and all(type(data[k]) is int and data[k] == 0 for k in ("retries", "redirects", "extra_source_gets")) and data["failed"] is False and data["execution_enabled"] is False, "bounded fixed source acquisition")
    if value["whisper"] is not None:
        w = value["whisper"]
        _fields(w, {"schema", "status", "aggregate_cap", "total_read_bytes", "asset_count", "network_teardown_complete", "model_calls", "setup_deadline_seconds"}, "fixed Whisper acquisition fields")
        require(w["schema"] == "bounded-host-setup-v1" and w["status"] == "complete" and w["aggregate_cap"] == 1536*MIB and type(w["total_read_bytes"]) is int and 0 <= w["total_read_bytes"] <= 1536*MIB and type(w["asset_count"]) is int and w["asset_count"] == 24 and w["network_teardown_complete"] is True and type(w["model_calls"]) is int and w["model_calls"] == 0 and w["setup_deadline_seconds"] == 1200, "bounded load setup acquisition")


# Fixed production-written receipts. Presence never substitutes for the
# independent schemas, source/body bindings and stage validation below.
PREPARED_COMMON = frozenset(("prepare-receipt.json", "worker-complete.json",
    "native-source-lock.json", "native-asset-lock.json", "native-config.json",
    "kernel-scope.json", "cpu-process-guard.json"))
PREPARED_EXTRA = {
    "qwen06": frozenset(("contract.json", "decoder-inputs.json", "model-load.json",
        "native-resolved-generation-config.json")),
    "sensevoice": frozenset(("contract.json", "decoder-inputs.json", "model-load.json",
        "native-frontend-config.json", "optional-tool-probes.json")),
    "whisper": frozenset(("load-receipt.json",)),
}
SETUP_COMPLETE_MEMBERS = frozenset(("acquisition-receipt.json", "image-transfer.json",
    "asr-setup-public.json", "whisper-setup/runtime-lock.json", "whisper-setup/source-lock.json",
    "whisper-setup/asset-lock.json", "whisper-setup/installed-inventory.json",
    "whisper-setup/download-receipt.json", "whisper-setup/setup-receipt.json",
    "whisper-setup/bounded-acquisition/receipt.json",
    *("whisper-setup/phase-" + phase + "-complete.json" for phase in ("install", "source", "model", "finalize"))))
PREPARED_HOST_MEMBERS = frozenset(("policy/policy.json", "policy/policy-freeze.json",
    "cpu-receipt.json", "cleanup-receipt.json", "supervision-receipt.json"))


def _require_completed_layout(files, host, prepared, phase):
    """Phase-aware no-omission gate; failed future stages need no fake receipts."""
    require(not prepared or host["stages"]["setup"] == "complete",
        "completed preparation requires completed setup")
    for family in prepared:
        require({"output/" + family + "/" + name for name in PREPARED_COMMON | PREPARED_EXTRA[family]} <= set(files),
            "complete production-written family prepare receipts")
    if host["stages"]["setup"] == "complete":
        require(SETUP_COMPLETE_MEMBERS <= set(files), "complete production-written setup receipts")
        require(set(legacy.official_whisper_source_inventory()) <= set(files),
            "complete locked official source after setup")
    if phase == "prepared" or host["status"] == "all3_prepared_no_recognition":
        require(all(value == "complete" for value in host["stages"].values()) and
            PREPARED_HOST_MEMBERS <= set(files), "complete prepared policy and resource receipts")
        cpu, cleanup = decode(files["cpu-receipt.json"]), decode(files["cleanup-receipt.json"])
        require(cpu["all_final_usage_verified"] is True and cpu["threshold_exceeded"] is False and
            cleanup["all_removed"] is True and cleanup["all_cleanup_verified"] is True,
            "complete prepare resource and cleanup observations")
        run_id = str(host["identity"]["run_id"])
        expected = [(c.EXPERIMENT + "-" + run_id + "-setup", "setup", None)] + [
            (EXPERIMENT + "-" + run_id + "-" + family, family, "prepare") for family in FAMILIES]
        rows = decode(files["supervision-receipt.json"])["stages"]
        require([(row["name"], row["family"], row["mode"]) for row in rows] == expected and
            all(row["status"] == "complete" and row["worker_exit_success"] is True and
                row["primary_error"] is None and row["cleanup_error"] is None for row in rows),
            "successful prepare requires four exact completed supervision stages")
    if host["status"] == "all3_prepared_no_recognition":
        require("policy/capacity-proof.json" in files, "complete final capacity receipt")


def validate_roots(files, run, *, phase="final"):
    required = {"prepare-host-receipt.json", "prepare-run-preflight.json", "input-freeze.json", "public-screening-receipt.json", FREEZE,
                "source/" + SOURCE_REL + "prepare-release.json"}
    require(required <= set(files), "complete independent prepare evidence core")
    val = lambda name: decode(files[name])
    pc = _prepare_contract()
    release = val("source/" + SOURCE_REL + "prepare-release.json")
    pc.validate_release(release)
    host = val("prepare-host-receipt.json")
    _fields(host, {"schema", "identity", "source_freeze_sha256", "executed_tree_sha", "source_claim_sha256", "screening_receipt_sha256", "input_verified", "status", "stage", "stages", "max_asr_calls", "asr_forward_calls", "tts_calls", "prepared_families", "primary_error", "cleanup_error", "cleanup_verified"}, "exact independent host receipt")
    require(host["schema"] == "k1-prepare-diagnostic-host-receipt-v1" and c.exact_json(host["identity"], run) and host["source_freeze_sha256"] == sha(files[FREEZE]) and type(host["executed_tree_sha"]) is str and HEAD.fullmatch(host["executed_tree_sha"]), "bound diagnostic host identity/source")
    require(host["status"] in ("started", "failed_no_retry", "all3_prepared_no_recognition") and host["stage"] in ("admitted", "input", "setup", "prepare", "policy_freeze", "capacity_proof", "cleanup", "package"), "prepare host stage/status only")
    for k in ("input_verified", "cleanup_verified"): require(type(host[k]) is bool, "observed host boolean")
    require(type(host["max_asr_calls"]) is int and host["max_asr_calls"] == 0 and type(host["tts_calls"]) is int and host["tts_calls"] == 0 and (host["asr_forward_calls"] is None or type(host["asr_forward_calls"]) is int and host["asr_forward_calls"] == 0), "zero authorization; incomplete observation remains unknown")
    require(type(host["stages"]) is dict and set(host["stages"]) == {"setup", *(f+"_prepare" for f in FAMILIES)} and all(v in ("not_run", "started", "complete", "failed_no_retry") for v in host["stages"].values()), "fixed preparation stages only")
    require(type(host["prepared_families"]) is list and host["prepared_families"] == [f for f in FAMILIES if f in host["prepared_families"]], "ordered unique completed preparations")
    _public_failure(host["primary_error"]); _public_failure(host["cleanup_error"])
    preflight = val("prepare-run-preflight.json")
    _fields(preflight, {"schema", "identity", "numeric_job_id", "workflow", "source_branch", "evidence_branch", "source_branch_workflow_runs", "evidence_ref_absent", "prior_prepare_history_sha256", "public_standard_hosted_ubuntu24", "api_metadata_accounting"}, "diagnostic preflight fields")
    expected = {"schema": "k1-prepare-run-preflight-v1", "identity": run, "workflow": POLICY["workflow"], "source_branch": POLICY["source_branch"], "evidence_branch": POLICY["result_branch"], "source_branch_workflow_runs": 1, "evidence_ref_absent": True, "prior_prepare_history_sha256": release["prior_prepare_history_sha256"], "public_standard_hosted_ubuntu24": True}
    require(all(c.exact_json(preflight[k], v) for k, v in expected.items()) and type(preflight["numeric_job_id"]) is int and 0 < preflight["numeric_job_id"] < 10**20, "actual one-use prepare job")
    accounting = preflight["api_metadata_accounting"]
    _fields(accounting, {"gets", "response_body_bytes", "response_body_cap_bytes", "scope", "hard_wire_byte_cap"}, "public API accounting fields")
    require(type(accounting["gets"]) is int and accounting["gets"] == 6 and type(accounting["response_body_bytes"]) is int and 0 <= accounting["response_body_bytes"] <= MIB and accounting["response_body_cap_bytes"] == MIB and accounting["scope"] == "public_api_metadata_separate_from_model_audio_bodies" and accounting["hard_wire_byte_cap"] is False, "unchanged bounded actual public API accounting")
    require(sha(files["input-freeze.json"]) == pc.BINDING_PINS["input_freeze_sha256"] and sha(files["public-screening-receipt.json"]) == pc.BINDING_PINS["screening_receipt_sha256"] == host["screening_receipt_sha256"], "exact unchanged independently reviewed input authority")
    packet_present = {name for name in files if name.startswith("metadata/")}
    claim = None
    if "authority/prepare-claim.json" in files:
        claim = val("authority/prepare-claim.json")
        _fields(claim, {"schema", "experiment", "identity", "source_freeze_sha256", "release_sha256", "metadata_packet_sha256", "maximum_model_constructions", "maximum_new_asr_calls", "prior_prepare_history_sha256"}, "separate once prepare claim fields")
        packet = {name: files["metadata/"+name] for name in PACKET_NAMES if "metadata/"+name in files}
        values = pc.validate_metadata_packet(packet, release)
        expected = {"schema": "k1-prepare-once-claim-v1", "experiment": EXPERIMENT, "identity": run,
            "source_freeze_sha256": sha(files[FREEZE]), "release_sha256": sha(canonical(release)),
            "metadata_packet_sha256": sha(canonical({name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(packet.items())})),
            "maximum_model_constructions": 3, "maximum_new_asr_calls": 0, "prior_prepare_history_sha256": release["prior_prepare_history_sha256"]}
        require(c.exact_json(claim, expected) and host["input_verified"] is True and host["source_claim_sha256"] == sha(files["authority/prepare-claim.json"]) and values["host-binding.json"]["identity"] == run, "same exact host metadata and independent prepare authority")
    else:
        require(host["source_claim_sha256"] is None and not any(n.startswith("output/") for n in files), "no model preparation without independent once claim")
        if packet_present:
            packet = {name: files["metadata/"+name] for name in PACKET_NAMES if "metadata/"+name in files}
            pc.validate_metadata_packet(packet, release)
    if packet_present: require(packet_present == {"metadata/"+name for name in PACKET_NAMES}, "whole metadata packet only")
    prepared = []
    for family in FAMILIES:
        prefix = "output/" + family + "/"
        for leaf in ("prepare-receipt.json", "worker-complete.json"):
            if prefix+leaf in files:
                require(claim is not None, "prepare receipt requires separate claim")
                validate_prepare_receipt(val(prefix+leaf), family, run, claim)
        if prefix+"worker-complete.json" in files:
            mandatory = PREPARED_COMMON | PREPARED_EXTRA[family]
            require(all(prefix+n in files for n in mandatory) and files[prefix+"prepare-receipt.json"] == files[prefix+"worker-complete.json"], "complete actual native prepare receipts")
            source, assets, config = (val(prefix+n) for n in ("native-source-lock.json", "native-asset-lock.json", "native-config.json"))
            import worker_receipts as wr
            wr._inventory(source.get("files")); wr._inventory(assets.get("assets")); wr.validate_native_config(config)
            require(source.get("executed_head_sha") == run["executed_head_sha"] and source.get("workflow_run_id") == str(run["run_id"]) and config["source_lock_sha256"] == sha(files[prefix+"native-source-lock.json"]) and config["asset_lock_sha256"] == sha(files[prefix+"native-asset-lock.json"]) and config["adapter_id"] == wr.ADAPTERS[family] and config["model_family"] == assets["model_family"] and assets["model_sha256"] == sha(canonical(assets["assets"])), "prepared actual source/assets/config identity")
            prepared.append(family)
    require(all(f in prepared for f in host["prepared_families"]), "host acknowledged preparations backed by complete receipts")
    if host["asr_forward_calls"] == 0:
        require(prepared == list(FAMILIES) and host["prepared_families"] == prepared, "only all-three acknowledged completed guards establish zero observed forwards")
    if phase == "prepared" or host["status"] == "all3_prepared_no_recognition":
        require(prepared == list(FAMILIES) and host["asr_forward_calls"] == 0, "all-three actual zero-forward preparation required")
    if host["status"] == "all3_prepared_no_recognition":
        require(host["cleanup_verified"] and host["primary_error"] is None and host["cleanup_error"] is None and "policy/capacity-proof.json" in files, "success requires bounded package proof and clean shutdown")
    if "policy/policy.json" in files:
        import worker_receipts as wr
        rows = wr._policy(files["policy/policy.json"])
        require(prepared == list(FAMILIES), "complete measured native policies only")
        for family in FAMILIES:
            require(c.exact_json(rows[family]["native_decode_config"], val("output/"+family+"/native-config.json")), "unchanged actual native configuration policy")
    if "policy/policy-freeze.json" in files:
        pf = val("policy/policy-freeze.json")
        expected = {"schema": "k1-prepare-all3-policy-freeze-v1", "identity": run,
            "recognizer_policy_sha256": sha(files["policy/policy.json"]), "asr_forward_calls": 0,
            "prepared_config_files": {f: {"bytes": len(files["output/"+f+"/native-config.json"]), "sha256": sha(files["output/"+f+"/native-config.json"])} for f in FAMILIES}}
        require(c.exact_json(pf, expected), "exact independent prepare policy freeze")
    if "policy/capacity-proof.json" in files: _capacity_proof(val("policy/capacity-proof.json"), files, run)
    if "acquisition-receipt.json" in files: _acquisition(val("acquisition-receipt.json"))
    _require_completed_layout(files, host, prepared, phase)
    return True


def validate_members(files, run_identity, *, phase="final"):
    identity(run_identity)
    require(phase in ("prepared", "final") and type(files) is dict and 1 <= len(files) <= MAX_MEMBERS, "bounded whole diagnostic evidence map")
    require(all(type(raw) is bytes and len(raw) <= MAX_MEMBER for raw in files.values()), "member byte cap")
    for name, raw in files.items():
        path(name)
        require(not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE") and not raw.startswith((b"fLaC", b"OggS", b"ID3")), "audio bytes prohibited")
        require(PurePosixPath(name).suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg", ".pt", ".pth", ".safetensors", ".whl", ".pkl", ".pickle", ".key", ".pem", ".zip", ".gz", ".bin"), "audio/weights/packages/credentials prohibited")
    source = _source_members(files)
    native = legacy._native_body_members(files)
    official = legacy.official_whisper_source_members(files)
    setup = legacy._host().public_setup_members()
    for name, raw in files.items():
        if name in source or name in official: continue
        if name in native:
            require(name.startswith("output/"), "no duplicate frozen or inference evidence admitted")
            if "/".join(name.split("/")[1:]) not in legacy.BINARY_ASSETS: raw.decode("utf-8")
            continue
        require(name in ROOT_JSON or _worker_path(name) or name in setup or name in {"metadata/"+n for n in PACKET_NAMES}, "unknown/ASR evidence cannot be omitted or published")
        value = decode(raw)
        require(type(value) is dict, "public metadata object")
        legacy._public_metadata(value)
        if _worker_path(name) and name.rsplit("/", 1)[-1] not in ("prepare-receipt.json", "worker-complete.json") or name in setup or name in ("cpu-receipt.json", "cleanup-receipt.json", "supervision-receipt.json", "image-transfer.json", "asr-setup-public.json"):
            legacy.validate_member_json(name, value)
        if name == "supervision-receipt.json":
            require(all(row["mode"] in ("prepare", "setup", None) and "decode" not in row["name"] for row in value["stages"]), "no recognition supervision in independent prepare evidence")
    validate_roots(files, run_identity, phase=phase)
    return files


def _walk(folder):
    require(stat.S_ISDIR(folder.lstat().st_mode), "plain public evidence directory")
    result = set()
    for root, dirs, names in os.walk(folder, followlinks=False):
        for name in dirs: require(stat.S_ISDIR((Path(root)/name).lstat().st_mode), "no aliased source directory")
        for name in names:
            file = Path(root)/name
            require(stat.S_ISREG(file.lstat().st_mode), "no special public evidence member")
            result.add(file.relative_to(folder).as_posix())
    return result


def _collect(root, run_identity, *, phase="final"):
    root = c.safe_path(root)
    require(root.is_dir(), "existing trusted host evidence directory")
    folders = {"authority", "metadata", "policy", "source", "output", "whisper-setup"}
    private = {"input", "code", "setup-code", "asr-runtime", "asr-setup", "whisper-runtime", "bounded-setup", "gates"}
    names = set()
    for entry in root.iterdir():
        if entry.name in private:
            require(stat.S_ISDIR(entry.lstat().st_mode), "fixed private workspace must be plain directory")
            continue
        if entry.name in folders:
            names.update(entry.name + "/" + n for n in _walk(entry))
        elif entry.name in ROOT_JSON:
            names.add(entry.name)
        else:
            # The host keeps private/runtime material outside this evidence root.
            require(False, "unknown root member cannot be silently dropped")
    require(1 <= len(names) <= MAX_MEMBERS, "complete evidence member count cap")
    files = {path(name): c.read_bytes(root/name, cap=MAX_MEMBER) for name in sorted(names)}
    return validate_members(files, run_identity, phase=phase)


def _failure_record(run_identity, stage, error):
    try: identity(run_identity); bound_identity = run_identity
    except BaseException: bound_identity = None
    value = {"schema": "k1-prepare-package-failure-v1", "identity": bound_identity,
        "stage": stage, "status": "failed_closed_no_pruning", "maximum_new_asr_calls": 0,
        "observed_asr_calls": None, "retry_authorized": False,
        "error_type": type(error).__name__ if type(error) in (ValueError, TypeError, OSError, RuntimeError, MemoryError) else "OtherError"}
    print(canonical(value).decode("ascii"), flush=True)
    return value


def build_package(root, run_identity):
    try:
        files = _collect(root, run_identity)
        payload, manifest, _ = _zip_files(files, run_identity)
        validate_archive(payload, run_identity)
        return payload, manifest
    except BaseException as error:
        try: _failure_record(run_identity, "package", error)
        except BaseException: pass  # Preserve the primary package failure if stdout is unavailable.
        raise


def validate_source_binding(payload, run_identity, freeze_sha256, source_rows, numeric_job_id, source_tree):
    manifest, _ = validate_archive(payload, run_identity)
    require(manifest["members"][FREEZE]["sha256"] == freeze_sha256, "independent prepare source freeze binding")
    require(type(numeric_job_id) is int and 0 < numeric_job_id < 10**20 and type(source_tree) is str and HEAD.fullmatch(source_tree), "actual numeric job/source tree")
    by_path = {row["path"]: row for row in source_rows}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in manifest["members"]:
            if name.startswith("source/"):
                actual = by_path.get(name[7:])
                require(actual is not None and actual["mode"] in ("100644", "100755") and actual["type"] == "blob" and actual["sha"] == git_sha(archive.read(name)), "every archived source body equals actual Git source")
        preflight = decode(archive.read("prepare-run-preflight.json"))
        host = decode(archive.read("prepare-host-receipt.json"))
        require(type(preflight["numeric_job_id"]) is int and preflight["numeric_job_id"] == numeric_job_id and host["executed_tree_sha"] == source_tree, "same actual independent prepare job/source tree")

def _package_budget_preflight(root, run_identity, *, job, execution_plan, input_binding, policy_path):
    """Actually pack the complete prepared bytes plus fixed six-clip worst prefixes.

    The fixture is internal sizing data, never admissible public native evidence.
    It proves this prepared closure and stress fixture only, not a universal
    compression bound. The whole actual terminal package is checked again.
    """
    c.validate_input_binding(input_binding, job, execution_plan)
    require(Path(policy_path) == Path(root) / "policy/policy.json", "fixed prepared policy path")
    policy_raw = c.read_bytes(policy_path, cap=MAX_MEMBER)
    actual = _collect(root, run_identity, phase="prepared")
    require(actual.get("policy/policy.json") == policy_raw, "actual prepared policy bytes")
    require(not any(re.search(r"/clip-00000[1-6](?:/|\.)", name) for name in actual), "zero forward before budget proof")
    prepared_payload, _, prepared_expanded = _zip_files(actual, run_identity)
    trial = dict(actual)
    rng = random.Random(20261012)
    for family in c.FAMILIES:
        prefix = "output/" + family + "/"
        require(prefix + "native-config.json" in actual and prefix + "prepare-receipt.json" in actual, "all three actual prepared configurations")
        prepared = decode(actual[prefix + "prepare-receipt.json"])
        require(prepared.get("status") == "prepared_configuration_only" and prepared.get("audio_forward_calls") == 0
            and prepared.get("native_decode_attempts") == 0 and prepared.get("recognition_result") is False, "prepare-only zero native calls")
        config = decode(actual[prefix + "native-config.json"])
        # Validate actual native config/policy without importing a model or worker.
        import worker_receipts
        policy_rows = worker_receipts._policy(policy_raw)
        require(canonical(policy_rows[family]["native_decode_config"]) == canonical(config), "actual prepared native policy/config binding")
        for name, raw in actual.items():
            if name.startswith(prefix):
                frozen_name = "frozen/" + name[len("output/"):]
                require(frozen_name not in trial or trial[frozen_name] == raw, "complete prepare/frozen identity")
                trial[frozen_name] = raw
        for clip in job["clips"]:
            base = prefix + clip["audio_id"] + "/"
            if family == "qwen06":
                # The retained generation-return capture includes the prompt:
                # its 65536 total-ID cap is separate from 256 generated tokens.
                # Seeded diverse ordinary IDs avoid a single-value compression
                # shortcut. This fixed stress record is never public evidence.
                trial[prefix + clip["audio_id"] + ".native-generation-return.json"] = canonical({
                    "schema": "synthetic-size-only-not-ASR-evidence",
                    "sequence_shape": [1, 65536], "prompt_token_count": 65536 - 256,
                    "returned_sequence_ids": [rng.randrange(1, 151000) for _ in range(65536)],
                    "capture_truncated": False})
            if family == "whisper":
                ids = [50258, 50260, 50359, 50363]
                for step in range(1, 225):
                    ids.append(rng.randrange(1, 50256))
                    trial[base + f"update-{step:03d}.json"] = canonical({"step": step, "token_ids": list(ids), "completed": step == 224})
            for leaf in ("started.json", "completed.json", "raw-native-output.json", "raw-transcript.json"):
                body = {"schema": "synthetic-size-only-not-ASR-evidence", "config": config,
                    "token_ids": [rng.randrange(1, 151000) for _ in range(256)]}
                if leaf in ("raw-native-output.json", "raw-transcript.json"):
                    body["raw_text"] = "".join(chr(0x4e00 + rng.randrange(20480)) for _ in range(32768))
                # ASCII escaping conservatively reserves six bytes per CJK char.
                trial[base + leaf] = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    trial["size-reserve.json"] = canonical({"schema": "synthetic-size-reserve", "padding": "x" * MIB})
    payload, _, expanded = _zip_files(trial, run_identity)
    return {"schema": "k1-prepare-package-capacity-v1", "status": "actual_prepared_and_private_pressure_fit",
        "identity": run_identity, "synthetic_pressure_is_not_asr_evidence": True, "new_native_calls": 0,
        "prepared_compressed_bytes": len(prepared_payload), "prepared_expanded_member_bytes": prepared_expanded, "eligible_clip_count": len(job["clips"]),
        "whisper_full_prefix_steps_per_clip": 224, "transcript_reserve_characters_each_copy": 32768,
        "transcript_copies_per_clip": 2, "extra_json_reserve_bytes": MIB,
        "recognizer_policy_sha256": sha(policy_raw), "worker_input_binding_sha256": sha(canonical(input_binding)),
        "execution_plan_sha256": sha(canonical(execution_plan)), "job_sha256": sha(canonical(job)),
        "compressed_bytes": len(payload), "expanded_member_bytes": expanded,
        "actual_prepared_members": len(actual), "worst_fixture_members": len(trial)}



def _capacity_proof(value, files, run):
    fixed = {"schema": "k1-prepare-package-capacity-v1", "status": "actual_prepared_and_private_pressure_fit",
        "identity": run, "synthetic_pressure_is_not_asr_evidence": True, "new_native_calls": 0,
        "eligible_clip_count": 6, "whisper_full_prefix_steps_per_clip": 224,
        "transcript_reserve_characters_each_copy": 32768, "transcript_copies_per_clip": 2,
        "extra_json_reserve_bytes": MIB, "recognizer_policy_sha256": sha(files["policy/policy.json"]),
        "worker_input_binding_sha256": sha(files["metadata/worker-input-binding.json"]),
        "execution_plan_sha256": sha(files["metadata/execution-plan.json"]), "job_sha256": sha(files["metadata/job.json"])}
    caps = {"prepared_compressed_bytes": MAX_COMPRESSED, "prepared_expanded_member_bytes": MAX_UNCOMPRESSED,
        "compressed_bytes": MAX_COMPRESSED, "expanded_member_bytes": MAX_UNCOMPRESSED,
        "actual_prepared_members": MAX_MEMBERS, "worst_fixture_members": MAX_MEMBERS}
    _fields(value, set(fixed)|set(caps), "exact private sizing proof fields")
    require(all(c.exact_json(value[k], v) for k, v in fixed.items()) and all(type(value[k]) is int and 1 <= value[k] <= cap for k, cap in caps.items()), "complete actual prepared and fixed pressure measurements within unchanged caps")
    require(value["actual_prepared_members"] < value["worst_fixture_members"] and value["prepared_expanded_member_bytes"] < value["expanded_member_bytes"], "actual plus full fixed pressure fixture retained")


def package_budget_preflight(root, run_identity, *, job, execution_plan, input_binding, policy_path):
    try:
        return _package_budget_preflight(root, run_identity, job=job, execution_plan=execution_plan,
            input_binding=input_binding, policy_path=policy_path)
    except BaseException as error:
        try: _failure_record(run_identity, "capacity_proof", error)
        except BaseException: pass  # Preserve the primary sizing failure if stdout is unavailable.
        raise


def package_capacity_preflight(root, run_identity):
    """Trusted host fixed interface; no caller-selected files or input authority."""
    try:
        root = c.safe_path(root)
        job = decode(c.read_bytes(root / "metadata/job.json", cap=MAX_MEMBER))
        plan = decode(c.read_bytes(root / "metadata/execution-plan.json", cap=MAX_MEMBER))
        binding = decode(c.read_bytes(root / "metadata/worker-input-binding.json", cap=MAX_MEMBER))
    except BaseException as error:
        try: _failure_record(run_identity, "capacity_proof", error)
        except BaseException: pass
        raise
    return package_budget_preflight(root, run_identity, job=job, execution_plan=plan,
        input_binding=binding, policy_path=root / "policy/policy.json")
