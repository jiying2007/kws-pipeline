"""Bounded native-evidence transport in GitHub job logs. No network or execution.

The caller supplies a reviewed, explicit public-output member allowlist. This
module cannot establish publication permission or classify arbitrary secrets.
The consumer supplies run/head/job identities from independently fetched GitHub
job metadata and must preserve the numeric job ID and full decoded log hash in
its custody receipt. A log package is never called a GitHub Actions artifact.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

PREFIX = "KWS_NATIVE_EVIDENCE_V1 "
MAX_COMPRESSED = 12 * 1024 * 1024
MAX_UNCOMPRESSED = 32 * 1024 * 1024
MAX_MEMBER = 8 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_LOG_BYTES = 24 * 1024 * 1024
CHUNK_CHARS = 4096
MANIFEST = "__native_evidence_manifest__.json"
TRANSPORT = "github-job-log-chunks-v1"
TIMESTAMP = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ")
SHA = re.compile(r"[a-f0-9]{64}\Z")
HEAD = re.compile(r"[a-f0-9]{40}\Z")

PACKAGE_LIMIT_REASONS = {
    "compressed_cap": "compressed evidence cap; do not omit native evidence",
    "expanded_cap": "uncompressed evidence cap",
    "member_cap": "bounded member size cap",
    "member_count_cap": "explicit bounded member allowlist required",
}


class PackageLimitError(ValueError):
    """Fixed, public sizing diagnostics; unknown aggregate measurements are None."""
    def __init__(self, reason_code, *, compressed_bytes=None, expanded_bytes=None,
                 member_count=None):
        if type(reason_code) is not str or reason_code not in PACKAGE_LIMIT_REASONS:
            raise ValueError("package limit reason")
        if any(value is not None and (type(value) is not int or value < 0)
               for value in (compressed_bytes, expanded_bytes, member_count)):
            raise ValueError("package limit measurement")
        self.reason_code = reason_code
        self.compressed_bytes = compressed_bytes
        self.expanded_bytes = expanded_bytes
        self.member_count = member_count
        super().__init__(PACKAGE_LIMIT_REASONS[reason_code])

def require(condition, message):
    if not condition:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def identity(value):
    require(type(value) is dict and set(value) == {"repository", "run_id", "run_attempt",
        "job_key", "executed_head_sha"}, "identity fields")
    require(value["repository"] == "jiying2007/kws-pipeline", "repository identity")
    require(type(value["run_id"]) is int and value["run_id"] > 0, "real run ID required")
    require(type(value["run_attempt"]) is int and value["run_attempt"] == 1, "single run attempt required")
    require(type(value["job_key"]) is str and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value["job_key"]), "job key")
    require(type(value["executed_head_sha"]) is str and HEAD.fullmatch(value["executed_head_sha"]), "executed head")

def relative(name):
    require(type(name) is str and name and "\\" not in name and len(name) <= 240
            and all(ord(char) >= 32 and ord(char) != 127 for char in name), "member path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and all(p not in (".", "..", "")
        for p in name.split("/")), "unsafe member path")
    require(not any(p.startswith(".") for p in path.parts), "hidden member prohibited")
    require(path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg", ".pt", ".pth",
        ".kwm", ".safetensors", ".whl", ".pem", ".key", ".pkl", ".pickle"}, "audio/weights/credential member prohibited")
    return name

def _read(root, name, cap=MAX_MEMBER):
    require(type(cap) is int and 0 < cap <= MAX_MEMBER, "read cap")
    parts = PurePosixPath(relative(name)).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for part in parts[:-1]:
            new = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = new
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "bounded unaliased member required")
        if before.st_size > cap:
            raise PackageLimitError("member_cap")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            raw = stream.read(cap + 1)
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in fields), "member changed during read")
        require(not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"), "WAV bytes prohibited")
        return raw
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)

def build_package(root, allowed_members, run_identity):
    identity(run_identity)
    require(type(allowed_members) is list and 1 <= len(allowed_members), "explicit bounded member allowlist required")
    if len(allowed_members) > MAX_MEMBERS:
        raise PackageLimitError("member_count_cap", member_count=len(allowed_members))
    names = [relative(name) for name in allowed_members]
    require(names == sorted(set(names)) and MANIFEST not in names, "canonical unique member allowlist required")
    files, total = {}, 0
    for name in names:
        try:
            raw = _read(root, name)
        except PackageLimitError as error:
            raise PackageLimitError(error.reason_code, member_count=len(names)) from None
        total += len(raw)
        if total > MAX_UNCOMPRESSED:
            # Stop before reading the remainder; do not present a partial sum as
            # the complete package's expanded size.
            raise PackageLimitError("expanded_cap", member_count=len(names))
        files[name] = raw
    manifest = {"schema": "native-evidence-member-manifest-v1", "identity": run_identity,
                "members": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}}
    manifest_raw = canonical(manifest)
    if len(manifest_raw) > MAX_MEMBER:
        raise PackageLimitError("member_cap", expanded_bytes=total + len(manifest_raw),
                                member_count=len(names))
    if total + len(manifest_raw) > MAX_UNCOMPRESSED:
        raise PackageLimitError("expanded_cap", expanded_bytes=total + len(manifest_raw),
                                member_count=len(names))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in [(MANIFEST, manifest_raw), *files.items()]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, raw)
    payload = buffer.getvalue()
    if len(payload) > MAX_COMPRESSED:
        raise PackageLimitError("compressed_cap", compressed_bytes=len(payload),
                                expanded_bytes=total + len(manifest_raw), member_count=len(names))
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = [encoded[i:i + CHUNK_CHARS] for i in range(0, len(encoded), CHUNK_CHARS)]
    header = {"kind": "header", "transport": TRANSPORT, "identity": run_identity,
              "archive_bytes": len(payload), "archive_sha256": sha(payload),
              "manifest_sha256": sha(manifest_raw), "member_count": len(files),
              "uncompressed_bytes": total + len(manifest_raw), "chunk_count": len(chunks),
              "chunk_chars": CHUNK_CHARS}
    h = sha(canonical(header))
    records = [header]
    for i, chunk in enumerate(chunks):
        records.append({"kind": "chunk", "header_sha256": h, "index": i,
                        "data": chunk, "data_sha256": sha(chunk.encode("ascii"))})
    records.append({"kind": "end", "header_sha256": h, "archive_sha256": sha(payload), "chunk_count": len(chunks)})
    return payload, manifest, [PREFIX + canonical(record).decode("ascii") for record in records]

def validate_archive(payload, expected_identity, binding=None):
    """Pure bounded validation shared by log and Git transports; never extract.

    binding, if supplied, contains externally authenticated size/digest fields.
    None measures those fields while still validating every exact member.
    """
    identity(expected_identity)
    require(type(payload) is bytes and 1 <= len(payload) <= MAX_COMPRESSED,
            "compressed archive cap/type")
    require(binding is None or type(binding) is dict, "archive binding type")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        require(2 <= len(infos) <= MAX_MEMBERS + 1, "archive member count cap")
        measured_header = {"archive_bytes": len(payload), "archive_sha256": sha(payload),
            "member_count": len(infos) - 1,
            "uncompressed_bytes": sum(i.file_size for i in infos)}
        require(1 <= measured_header["uncompressed_bytes"] <= MAX_UNCOMPRESSED,
                "archive expanded size cap")
        header = measured_header if binding is None else binding
        for key, value in measured_header.items():
            require(type(header.get(key)) is type(value) and header[key] == value,
                    "archive identity/size mismatch")
        require(len(infos) == header["member_count"] + 1 and len({i.filename for i in infos}) == len(infos), "archive members duplicate/count")
        require(sum(i.file_size for i in infos) == header["uncompressed_bytes"], "archive expanded size")
        files = {}
        for info in infos:
            relative(info.filename)
            require(not info.is_dir() and not info.flag_bits & 1 and info.compress_type == zipfile.ZIP_DEFLATED
                    and stat.S_IFMT(info.external_attr >> 16) == stat.S_IFREG
                    and info.file_size <= MAX_MEMBER, "unsafe archive member")
            with archive.open(info) as stream:
                body = stream.read(MAX_MEMBER + 1)
            require(len(body) == info.file_size, "archive member size")
            files[info.filename] = body
    require(MANIFEST in files, "member manifest missing")
    measured_header["manifest_sha256"] = sha(files[MANIFEST])
    if binding is not None:
        require(measured_header["manifest_sha256"] == binding.get("manifest_sha256"), "member manifest binding")
    manifest = decode(files.pop(MANIFEST))
    require(type(manifest) is dict and set(manifest) == {"schema", "identity", "members"}
            and manifest["schema"] == "native-evidence-member-manifest-v1"
            and canonical(manifest["identity"]) == canonical(expected_identity), "manifest identity")
    measured = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}
    require(canonical(manifest["members"]) == canonical(measured), "exact member identities")
    return manifest, measured_header

def parse_job_log(log_text, expected_identity):
    identity(expected_identity)
    require(type(log_text) is str and len(log_text.encode("utf-8")) <= MAX_LOG_BYTES, "decoded job log cap/type")
    records = []
    for raw_line in log_text.splitlines():
        line = TIMESTAMP.sub("", raw_line)
        if line.startswith("\ufeff"):
            line = line[1:]
        if line.startswith(PREFIX):
            require(len(line) <= CHUNK_CHARS + 2048, "transport record cap")
            record = decode(line[len(PREFIX):])
            require(type(record) is dict, "transport record type")
            records.append(record)
            require(len(records) <= (4 * ((MAX_COMPRESSED + 2) // 3) + CHUNK_CHARS - 1) // CHUNK_CHARS + 2, "transport record count cap")
    require(len(records) >= 3, "missing transport header/chunks/end")
    header, end = records[0], records[-1]
    require(set(header) == {"kind", "transport", "identity", "archive_bytes", "archive_sha256",
        "manifest_sha256", "member_count", "uncompressed_bytes", "chunk_count", "chunk_chars"}
        and header["kind"] == "header" and header["transport"] == TRANSPORT, "transport header")
    require(canonical(header["identity"]) == canonical(expected_identity), "run/head/job identity mismatch")
    for key, cap in (("archive_bytes", MAX_COMPRESSED), ("uncompressed_bytes", MAX_UNCOMPRESSED),
                     ("member_count", MAX_MEMBERS), ("chunk_count", 1 + 4 * MAX_COMPRESSED // (3 * CHUNK_CHARS))):
        require(type(header[key]) is int and 1 <= header[key] <= cap, "header size/count cap")
    require(type(header["chunk_chars"]) is int and header["chunk_chars"] == CHUNK_CHARS, "fixed chunk size")
    for key in ("archive_sha256", "manifest_sha256"):
        require(type(header[key]) is str and SHA.fullmatch(header[key]), "header digest")
    h = sha(canonical(header))
    require(canonical(end) == canonical({"kind": "end", "header_sha256": h,
        "archive_sha256": header["archive_sha256"], "chunk_count": header["chunk_count"]}), "end record mismatch")
    require(len(records) == header["chunk_count"] + 2, "missing/duplicate/mixed transport records")
    chunks = []
    for i, row in enumerate(records[1:-1]):
        require(set(row) == {"kind", "header_sha256", "index", "data", "data_sha256"}
                and row["kind"] == "chunk" and row["header_sha256"] == h
                and type(row["index"]) is int and row["index"] == i, "unordered/duplicate/mixed chunk")
        chunk = row["data"]
        require(type(chunk) is str and 0 < len(chunk) <= CHUNK_CHARS
                and (i == header["chunk_count"] - 1 or len(chunk) == CHUNK_CHARS), "chunk length")
        require(row["data_sha256"] == sha(chunk.encode("ascii")), "chunk digest")
        chunks.append(chunk)
    payload = base64.b64decode("".join(chunks), validate=True)
    manifest, _ = validate_archive(payload, expected_identity, header)
    return payload, manifest, {"transport": TRANSPORT, "identity": expected_identity,
        "decoded_log_sha256": sha(log_text.encode("utf-8")), "archive_sha256": sha(payload),
        "archive_bytes": len(payload), "member_manifest_sha256": header["manifest_sha256"],
        "github_actions_artifact_id": None, "numeric_job_id_requires_external_verified_binding": True}
