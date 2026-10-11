"""Bounded public Git file-set transport. No network, execution or extraction.

Readers must fetch manifest and ordered chunks from ONE independently verified,
immutable evidence commit, not a moving branch. The expected identity comes from
independent GitHub run/job metadata. These hashes provide custody, not ASR success.
"""
from pathlib import Path
import os
import stat

from log_transport import (HEAD, MAX_COMPRESSED, MAX_MEMBERS, MAX_UNCOMPRESSED,
    SHA, _read, canonical, decode, identity, require, sha, validate_archive)

TRANSPORT = "github-repository-zip-chunks-v1"
MAX_CHUNK = 256 * 1024
MAX_CHUNKS = 32
MAX_MANIFEST = 16 * 1024
REPOSITORY = "jiying2007/kws-pipeline"
PURPOSES = {
    "synthetic": {"source_branch": "research/qwen12-native-transport-preflight-v1",
        "evidence_branch": "evidence/qwen12-native-transport-synthetic-v1",
        "directory": "research/qwen12-native-admission-v1/evidence/synthetic-v1",
        "scope": "qwen12-native-transport-preflight-v1", "job_key": "transport-fixture"},
    "native": {"source_branch": "research/qwen12-native-admission-recovery2",
        "evidence_branch": "evidence/qwen12-native-admission-recovery2",
        "directory": "research/qwen12-native-admission-v1/evidence/recovery2",
        "scope": "qwen12-native-admission-v1", "job_key": "qwen12"},
}


def purpose_policy(purpose):
    require(type(purpose) is str and purpose in PURPOSES, "fixed transport purpose")
    return PURPOSES[purpose]


def bound_identity(run_identity, purpose):
    identity(run_identity)
    require(run_identity["job_key"] == purpose_policy(purpose)["job_key"], "purpose/job mismatch")


def manifest_path(purpose):
    return purpose_policy(purpose)["directory"] + "/manifest.json"


def chunk_path(purpose, index):
    require(type(index) is int and 0 <= index < MAX_CHUNKS, "chunk index")
    return purpose_policy(purpose)["directory"] + f"/chunks/{index:04d}.bin"


def build_file_set(payload, run_identity, purpose):
    """Retain the exact original ZIP; return full repository paths to bytes."""
    bound_identity(run_identity, purpose)
    _, measured = validate_archive(payload, run_identity)
    chunks = [payload[i:i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
    require(1 <= len(chunks) <= MAX_CHUNKS, "chunk count cap")
    manifest = {"schema": "native-evidence-branch-manifest-v1", "transport": TRANSPORT,
        "purpose": purpose, "identity": run_identity, **measured,
        "chunk_bytes": MAX_CHUNK, "chunk_count": len(chunks),
        "chunks": [{"index": i, "path": chunk_path(purpose, i), "bytes": len(raw),
                    "sha256": sha(raw)} for i, raw in enumerate(chunks)]}
    raw = canonical(manifest)
    require(len(raw) <= MAX_MANIFEST, "outer manifest cap")
    return {manifest_path(purpose): raw,
            **{chunk_path(purpose, i): raw for i, raw in enumerate(chunks)}}


def parse_manifest(manifest_raw, expected_identity, purpose):
    bound_identity(expected_identity, purpose)
    require(type(manifest_raw) is bytes and 1 <= len(manifest_raw) <= MAX_MANIFEST,
            "outer manifest cap/type")
    m = decode(manifest_raw)
    require(type(m) is dict and set(m) == {"schema", "transport", "purpose", "identity",
        "archive_bytes", "archive_sha256", "manifest_sha256", "member_count",
        "uncompressed_bytes", "chunk_bytes", "chunk_count", "chunks"}, "outer manifest fields")
    require(canonical(m) == manifest_raw, "outer manifest must be canonical")
    require(m["schema"] == "native-evidence-branch-manifest-v1" and m["transport"] == TRANSPORT
            and m["purpose"] == purpose, "outer manifest kind/purpose")
    bound_identity(m["identity"], purpose)
    require(canonical(m["identity"]) == canonical(expected_identity), "run/head/job identity mismatch")
    for key, cap in (("archive_bytes", MAX_COMPRESSED), ("uncompressed_bytes", MAX_UNCOMPRESSED),
                     ("member_count", MAX_MEMBERS), ("chunk_count", MAX_CHUNKS)):
        require(type(m[key]) is int and 1 <= m[key] <= cap, "outer size/count cap")
    require(type(m["chunk_bytes"]) is int and m["chunk_bytes"] == MAX_CHUNK, "fixed chunk size")
    require(m["chunk_count"] == (m["archive_bytes"] + MAX_CHUNK - 1) // MAX_CHUNK,
            "exact chunk count")
    for key in ("archive_sha256", "manifest_sha256"):
        require(type(m[key]) is str and SHA.fullmatch(m[key]), "outer digest")
    require(type(m["chunks"]) is list and len(m["chunks"]) == m["chunk_count"], "chunk list count")
    for i, row in enumerate(m["chunks"]):
        require(type(row) is dict and set(row) == {"index", "path", "bytes", "sha256"}, "chunk fields")
        require(type(row["index"]) is int and row["index"] == i and row["path"] == chunk_path(purpose, i),
                "noncanonical/reordered/unsafe chunk")
        expected_bytes = min(MAX_CHUNK, m["archive_bytes"] - i * MAX_CHUNK)
        require(type(row["bytes"]) is int and row["bytes"] == expected_bytes, "chunk exact bytes")
        require(type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]), "chunk digest")
    return m


def reassemble_file_set(manifest_raw, ordered_chunks, expected_identity, purpose,
                        *, evidence_commit=None):
    """ordered_chunks is an exact list of (full_repository_path, raw_bytes).

    evidence_commit is optional only for local tests. Remote custody callers must
    provide the independently resolved immutable commit they used for every fetch.
    """
    m = parse_manifest(manifest_raw, expected_identity, purpose)
    require(evidence_commit is None or type(evidence_commit) is str and HEAD.fullmatch(evidence_commit),
            "immutable evidence commit")
    require(type(ordered_chunks) is list and len(ordered_chunks) == m["chunk_count"], "missing/extra chunks")
    bodies = []
    for pair, row in zip(ordered_chunks, m["chunks"]):
        require(type(pair) in (tuple, list) and len(pair) == 2, "chunk pair type")
        name, raw = pair
        require(type(name) is str and name == row["path"], "unordered/duplicate/mixed chunk path")
        require(type(raw) is bytes and len(raw) == row["bytes"] and sha(raw) == row["sha256"],
                "chunk size/digest mismatch")
        bodies.append(raw)
    payload = b"".join(bodies)
    internal, _ = validate_archive(payload, expected_identity, m)
    return payload, internal, {"transport": TRANSPORT, "identity": expected_identity,
        "purpose": purpose, "evidence_commit": evidence_commit,
        "outer_manifest_sha256": sha(manifest_raw), "archive_sha256": sha(payload),
        "archive_bytes": len(payload), "member_manifest_sha256": m["manifest_sha256"],
        "chunk_count": m["chunk_count"], "github_actions_artifact_id": None,
        "numeric_job_id_requires_external_verified_binding": True}


def safe_directory(path):
    """Reject symlinked/noncanonical parents, including the directory itself."""
    path = Path(path)
    require(path.is_absolute() and str(path) == os.path.abspath(path), "absolute canonical directory required")
    for part in [*reversed(path.parents), path]:
        require(stat.S_ISDIR(part.lstat().st_mode), "non-symlink directory required")
    return path


def package_path(runner_temp, run_identity, purpose):
    bound_identity(run_identity, purpose)
    root = safe_directory(runner_temp)
    return root / (purpose_policy(purpose)["scope"] + "-" + str(run_identity["run_id"])) / "evidence-package"


def write_package(runner_temp, payload, run_identity, purpose):
    file_set = build_file_set(payload, run_identity, purpose)
    root = package_path(runner_temp, run_identity, purpose)
    root.parent.mkdir(mode=0o700, exist_ok=True)
    safe_directory(root.parent)
    root.mkdir(mode=0o700)  # No rewrite, retry, alternate path or overwrite.
    (root / "chunks").mkdir(mode=0o700)
    prefix = purpose_policy(purpose)["directory"] + "/"
    for name, raw in file_set.items():
        with (root / name[len(prefix):]).open("xb") as stream:
            stream.write(raw)
    read_package(runner_temp, run_identity, purpose)
    return root


def read_package(runner_temp, run_identity, purpose):
    root = safe_directory(package_path(runner_temp, run_identity, purpose))
    require(sorted(p.name for p in root.iterdir()) == ["chunks", "manifest.json"], "exact package root entries")
    safe_directory(root / "chunks")
    manifest_raw = _read(root, "manifest.json", MAX_MANIFEST)
    m = parse_manifest(manifest_raw, run_identity, purpose)
    expected = [f"{i:04d}.bin" for i in range(m["chunk_count"])]
    require(sorted(p.name for p in (root / "chunks").iterdir()) == expected, "exact local chunks")
    chunks = [(row["path"], _read(root, "chunks/" + expected[i], MAX_CHUNK)) for i, row in enumerate(m["chunks"])]
    payload, internal, receipt = reassemble_file_set(manifest_raw, chunks, run_identity, purpose)
    return {manifest_path(purpose): manifest_raw, **dict(chunks)}, payload, internal, receipt
