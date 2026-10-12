"""Bounded K1 public-data transport, stdlib-only. No model or network imports.

A byte-custody check is not semantic audio/generation/source admission. The TTS
adapter is deliberately unavailable until its receipt schema is reviewed.
"""
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import stat
import zipfile

REPOSITORY = "jiying2007/kws-data"
MAX_ARCHIVE = 12 * 1024 * 1024
MAX_CHUNK = 256 * 1024
MAX_CHUNKS = 48
MAX_MANIFEST = 16 * 1024
MAX_BODY = 16 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
HEAD = re.compile(r"[0-9a-f]{40}\Z")
POLICIES = {"fixture": {
    "source_branch": "research/k1-punctuation6-writer-fixture-r1",
    "result_branch": "data/k1-punctuation6-writer-fixture-r1",
    "directory": "datasets/k1-punctuation6-v1/writer-fixture-r1",
    "workflow": ".github/workflows/k1-punctuation6-tts-v1.yml",
    "job_key": "writer-fixture",
}, "tts": {
    "source_branch": "research/k1-punctuation6-tts-v1",
    "result_branch": "data/k1-punctuation6-tts-v1",
    "directory": "datasets/k1-punctuation6-v1/tts-v1",
    "workflow": ".github/workflows/k1-punctuation6-tts-v1.yml",
    "job_key": "k1-tts",
}}

def policy(purpose):
    require(type(purpose) is str and purpose in POLICIES, "fixed purpose only")
    return POLICIES[purpose]
MEMBER_MANIFEST = "__k1_member_manifest__.json"
FIXTURE_MEMBERS = ("claim.json", "fixture.json", "source.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_sha(raw, kind="blob"):
    require(type(raw) is bytes and kind in ("blob", "tree", "commit"), "Git object input")
    return hashlib.sha1(kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def identity(value, purpose="fixture"):
    require(type(value) is dict and set(value) == {"repository", "run_id", "run_attempt",
            "job_key", "executed_head_sha"}, "identity fields")
    require(value["repository"] == REPOSITORY, "same public data repository")
    require(type(value["run_id"]) is int and 0 < value["run_id"] < 10**20, "run ID")
    require(type(value["run_attempt"]) is int and value["run_attempt"] == 1, "attempt one")
    require(value["job_key"] == policy(purpose)["job_key"], "fixed fixture job")
    require(type(value["executed_head_sha"]) is str and HEAD.fullmatch(value["executed_head_sha"]), "source commit")


def path(name):
    require(type(name) is str and 0 < len(name) <= 240 and name.isascii()
            and "\\" not in name and all(32 < ord(c) < 127 for c in name), "safe path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and str(p) == name and all(c not in ("", ".", "..")
            for c in name.split("/")), "canonical relative path")
    return name


def manifest_path(purpose="fixture"):
    return policy(purpose)["directory"] + "/manifest.json"


def chunk_path(index, purpose="fixture"):
    require(type(index) is int and 0 <= index < MAX_CHUNKS, "chunk index")
    return policy(purpose)["directory"] + f"/chunks/{index:04d}.bin"


def fixture_files(run_identity, freeze_sha256):
    identity(run_identity)
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256), "source closure SHA256")
    return {
        "claim.json": canonical({"schema": "k1-writer-fixture-claim-v1", "identity": run_identity,
            "fixture_only": True, "tts_claim_touched": False, "asr_claim_touched": False}),
        "fixture.json": canonical({"schema": "k1-fixed-zero-model-fixture-v1", "fixture_only": True,
            "description": "Fixed public non-audio JSON fixture; transport custody only.",
            "model_downloads": 0, "model_loads": 0, "tts_calls": 0, "asr_calls": 0,
            "training_runs": 0, "audio_members": 0, "fixed_values": list(range(32))}),
        "source.json": canonical({"schema": "k1-writer-fixture-source-v1", "identity": run_identity,
            "source_closure_sha256": freeze_sha256}),
    }


def fixture_archive(run_identity, freeze_sha256):
    files = fixture_files(run_identity, freeze_sha256)
    manifest = canonical({"schema": "k1-data-member-manifest-v1", "identity": run_identity,
        "members": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # Stored JSON makes the small fixture byte-identical across zlib versions.
        for name, raw in [(MEMBER_MANIFEST, manifest), *sorted(files.items())]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, raw)
    raw = buffer.getvalue()
    require(1024 <= len(raw) <= 4096, "kilobyte-only fixture")
    return raw


def verify_fixture(payload, run_identity, freeze_sha256):
    # Exact reconstruction excludes hidden members, secret/log/audio payloads,
    # duplicate ZIP entries, trailing bytes, encryption and compression bombs.
    require(type(payload) is bytes and payload == fixture_archive(run_identity, freeze_sha256),
            "exact fixed no-audio fixture required")
    return {name: {"bytes": len(raw), "sha256": sha(raw)}
            for name, raw in fixture_files(run_identity, freeze_sha256).items()}


def build_file_set(payload, run_identity, freeze_sha256, numeric_job_id, purpose="fixture"):
    identity(run_identity, purpose)
    require(type(payload) is bytes and 1 <= len(payload) <= MAX_ARCHIVE, "archive cap")
    require(type(numeric_job_id) is int and numeric_job_id > 0, "numeric job")
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256), "source freeze")
    chunks = [payload[i:i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
    m = {"schema": "k1-data-branch-manifest-v1", "identity": run_identity,
        "purpose": purpose, "fixture_only": purpose == "fixture", "source_closure_sha256": freeze_sha256,
        "numeric_job_id": numeric_job_id, "archive_bytes": len(payload), "archive_sha256": sha(payload),
        "chunk_bytes": MAX_CHUNK, "chunk_count": len(chunks),
        "chunks": [{"index": i, "path": chunk_path(i, purpose), "bytes": len(raw), "sha256": sha(raw)}
                   for i, raw in enumerate(chunks)]}
    raw = canonical(m)
    require(len(raw) <= MAX_MANIFEST, "outer manifest cap")
    return {manifest_path(purpose): raw, **{chunk_path(i, purpose): raw for i, raw in enumerate(chunks)}}


def parse_manifest(raw, run_identity, freeze_sha256, numeric_job_id, purpose="fixture"):
    identity(run_identity, purpose)
    require(type(raw) is bytes and 0 < len(raw) <= MAX_MANIFEST, "outer manifest cap/type")
    m = decode(raw)
    require(type(m) is dict and set(m) == {"schema", "identity", "purpose", "fixture_only", "source_closure_sha256",
        "numeric_job_id", "archive_bytes", "archive_sha256", "chunk_bytes", "chunk_count", "chunks"}, "outer fields")
    require(canonical(m) == raw and m["schema"] == "k1-data-branch-manifest-v1"
            and m["purpose"] == purpose and m["fixture_only"] is (purpose == "fixture"), "canonical fixed manifest")
    identity(m["identity"], purpose)
    require(canonical(m["identity"]) == canonical(run_identity), "identity mismatch")
    require(type(numeric_job_id) is int and numeric_job_id > 0 and type(m["numeric_job_id"]) is int
            and m["numeric_job_id"] == numeric_job_id, "numeric job mismatch")
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256)
            and m["source_closure_sha256"] == freeze_sha256, "source closure mismatch")
    require(type(m["archive_bytes"]) is int and 1 <= m["archive_bytes"] <= MAX_ARCHIVE, "archive cap")
    require(type(m["chunk_bytes"]) is int and m["chunk_bytes"] == MAX_CHUNK, "chunk size")
    require(type(m["chunk_count"]) is int and 1 <= m["chunk_count"] <= MAX_CHUNKS
            and m["chunk_count"] == (m["archive_bytes"] + MAX_CHUNK - 1) // MAX_CHUNK, "chunk count")
    require(type(m["archive_sha256"]) is str and SHA.fullmatch(m["archive_sha256"]), "archive SHA256")
    require(type(m["chunks"]) is list and len(m["chunks"]) == m["chunk_count"], "chunk list")
    for i, row in enumerate(m["chunks"]):
        require(type(row) is dict and set(row) == {"index", "path", "bytes", "sha256"}, "chunk fields")
        require(type(row["index"]) is int and row["index"] == i and row["path"] == chunk_path(i, purpose), "chunk order/path")
        require(type(row["bytes"]) is int and row["bytes"] == min(MAX_CHUNK, m["archive_bytes"] - i * MAX_CHUNK), "chunk bytes")
        require(type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]), "chunk SHA256")
    return m


def reassemble(raw, chunks, run_identity, freeze_sha256, numeric_job_id, purpose="fixture"):
    m = parse_manifest(raw, run_identity, freeze_sha256, numeric_job_id, purpose)
    require(type(chunks) is list and len(chunks) == m["chunk_count"], "exact chunks")
    for pair, row in zip(chunks, m["chunks"]):
        require(type(pair) is tuple and len(pair) == 2 and pair[0] == row["path"], "ordered unique chunks")
        require(type(pair[1]) is bytes and len(pair[1]) == row["bytes"] and sha(pair[1]) == row["sha256"], "chunk custody")
    payload = b"".join(pair[1] for pair in chunks)
    require(len(payload) == m["archive_bytes"] and sha(payload) == m["archive_sha256"], "archive custody")
    return payload


def tree_identity(flat_files):
    """Recompute actual nested Git trees from canonical file rows, no Git process."""
    require(type(flat_files) is list and len(flat_files) <= 4096, "bounded tree files")
    root = {}
    for row in flat_files:
        require(type(row) is dict and set(row) == {"path", "mode", "type", "sha"}, "tree row fields")
        name = path(row["path"])
        require(row["mode"] in ("100644", "100755", "120000", "160000")
                and row["type"] == ("commit" if row["mode"] == "160000" else "blob")
                and type(row["sha"]) is str and HEAD.fullmatch(row["sha"]), "tree leaf")
        node = root
        for part in name.split("/")[:-1]:
            node = node.setdefault(part, {})
            require(type(node) is dict, "tree path collision")
        leaf = name.split("/")[-1]
        require(leaf not in node, "duplicate tree path")
        node[leaf] = (row["mode"], row["sha"])
    directories = {}
    def fold(node, prefix):
        entries = []
        for name, entry in node.items():
            if type(entry) is dict:
                digest = fold(entry, prefix + name + "/")
                mode, sort_name = "40000", name + "/"
                directories[prefix + name] = digest
            else:
                mode, digest = entry
                sort_name = name
            entries.append((sort_name.encode("ascii"), mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(digest)))
        return git_sha(b"".join(raw for _, raw in sorted(entries)), "tree")
    return fold(root, ""), directories


TTS_JSON = {"claim.json", "outcomes.json", "source.json", "eos.json", "resources.json"}
CELL_IDS = tuple(f"k1p6-{i:03d}" for i in range(1, 7))
TTS_ALLOWED = TTS_JSON | {f"claims/{c}.started.json" for c in CELL_IDS} | {
    f"{kind}/{c}.wav" for c in CELL_IDS for kind in ("raw", "derived")}


def _metadata(value):
    if type(value) is str:
        require(not value.startswith(("/", "\\\\")) and not re.match(r"^[A-Za-z]:[\\/]", value),
                "private absolute path prohibited")
    elif type(value) is dict:
        for key, item in value.items():
            _metadata(key); _metadata(item)
    elif type(value) is list:
        for item in value: _metadata(item)


def validate_members(members, run_identity, purpose, semantic_validator=None):
    identity(run_identity, purpose)
    require(type(members) is dict and all(type(v) is bytes for v in members.values()), "member bytes")
    if purpose == "fixture":
        require(set(members) == set(FIXTURE_MEMBERS), "exact fixture members")
        for raw in members.values(): require(len(raw) <= 2048, "fixture member cap")
    else:
        require(TTS_JSON <= set(members) <= TTS_ALLOWED, "exact bounded TTS member allowlist")
        for name, raw in members.items():
            if name.startswith("raw/"): cap = 24000 * 12 * 4 + 256
            elif name.startswith("derived/"): cap = 16000 * 12 * 2 + 256
            else: cap = 64 * 1024
            require(0 < len(raw) <= cap, "TTS member cap")
        require(callable(semantic_validator), "reviewed TTS semantic validator required")
    for name, raw in members.items():
        path(name)
        if name.endswith(".json"):
            decoded = decode(raw)
            require(type(decoded) is dict and canonical(decoded) == raw, "canonical typed JSON member")
            _metadata(decoded)
    if purpose == "tts":
        # The reviewed adapter must bind all six outcomes, raw claim bytes,
        # WAV/float/PCM geometry, EOS/source/resources and allowlisted schemas.
        require(semantic_validator(members, run_identity) is True, "TTS semantic validation failed")


def build_archive(members, run_identity, purpose, semantic_validator=None):
    validate_members(members, run_identity, purpose, semantic_validator)
    manifest = canonical({"schema": "k1-data-member-manifest-v1", "identity": run_identity,
        "members": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(members.items())}})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in [(MEMBER_MANIFEST, manifest), *sorted(members.items())]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, raw)
    payload = buffer.getvalue()
    require(len(payload) <= MAX_ARCHIVE, "archive cap")
    return payload


def validate_archive(payload, run_identity, purpose, semantic_validator=None):
    identity(run_identity, purpose)
    require(type(payload) is bytes and 1 <= len(payload) <= MAX_ARCHIVE, "archive cap")
    members = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        require(2 <= len(infos) <= len(TTS_ALLOWED) + 1 and len({i.filename for i in infos}) == len(infos),
                "ZIP exact member count")
        require(sum(i.file_size for i in infos) <= MAX_ARCHIVE, "expanded bound")
        for info in infos:
            path(info.filename)
            require(info.filename == MEMBER_MANIFEST or info.filename in (set(FIXTURE_MEMBERS) if purpose == "fixture" else TTS_ALLOWED),
                    "unexpected ZIP member")
            require(not info.is_dir() and info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 1
                    and stat.S_IFMT(info.external_attr >> 16) == stat.S_IFREG
                    and info.file_size <= 24000 * 12 * 4 + 256, "safe bounded ZIP member")
            with archive.open(info) as stream:
                raw = stream.read(24000 * 12 * 4 + 257)
            require(len(raw) == info.file_size, "member size")
            members[info.filename] = raw
    require(MEMBER_MANIFEST in members, "internal member manifest")
    manifest = decode(members.pop(MEMBER_MANIFEST))
    require(manifest == {"schema": "k1-data-member-manifest-v1", "identity": run_identity,
        "members": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(members.items())}}, "exact internal identities")
    validate_members(members, run_identity, purpose, semantic_validator)
    require(payload == build_archive(members, run_identity, purpose, semantic_validator), "canonical exact ZIP; no trailer")
    return members
