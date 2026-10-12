"""Host-only, disabled K1 complete-data reader. Standard library only.

This module does not grant execution or review authority. Real identities and
independently reviewed expected hashes are external inputs; there are no defaults
that replace them with this preparation's source snapshot. No source GETs occur.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

import asr18_metadata as metadata

require = metadata.require
canonical = metadata.canonical
MAX_ARCHIVE = metadata.MAX_ARCHIVE
MAX_MANIFEST = metadata.MAX_MANIFEST
MAX_CHUNK = metadata.MAX_CHUNK
MAX_BODY = metadata.MAX_RESPONSE_BODIES
DATA_DIRECTORY = "datasets/k1-punctuation6-v1/tts-v1"
MEMBER_MANIFEST = "__k1_member_manifest__.json"
SOURCE_DIRECTORY = "research/k1-punctuation6-v1"
FREEZE_PATH = SOURCE_DIRECTORY + "/tts-source-freeze.json"
HOST_TTS_VERIFIER_ROOT = Path(__file__).resolve().parents[2] / "host-tts-verifier-source"
EXECUTION_ENABLED = False
VERIFIER_PINS = {'contract.py': {'bytes': 21656, 'sha256': '68180483211325b1d634a0cc64ab47d1655d2beac6792a6e6d2d93656c5832ec'}, 'data_transport.py': {'bytes': 16321, 'sha256': '42dd0b5091714ed98d226db32d919ef458ae468aee82aee1d94beffa43a7e110'}, 'tts_semantics.py': {'bytes': 16586, 'sha256': '671f3e173563e38b02a7796fae9e7885c8082742ff4ea72a92e5facdb6ccc77c'}, 'tts_common.py': {'bytes': 3897, 'sha256': '3b53f16ee7eaeda8b1fe0d26cdb4b31964f7e802b8767f1bc291b9ee8814d7d3'}, 'tts_public_schema.py': {'bytes': 36108, 'sha256': 'a94af5f17c2d6b43e7415d9d85975d5533198303ecef743286860ce4e8f30678'}, 'tts_gate.py': {'bytes': 11847, 'sha256': '50fb28d89e556547e604be239af3449c961680b551b8132426719eea89f248a3'}, 'tts_finalize.py': {'bytes': 19309, 'sha256': 'c3b2d323a0edaca48d042ae480c17e0ac7490ef59fc461384c931c511e8a3532'}, 'tts_worker.py': {'bytes': 16325, 'sha256': '72cdad4ef81e8d41c5218c1ff2f87953c5ba7e765f4489f085b5a4a91db751b7'}}
SOURCE_ROSTER = tuple(['.github/workflows/k1-punctuation6-tts-v1.yml', 'research/k1-punctuation6-v1/IMPLEMENTATION.md', 'research/k1-punctuation6-v1/README.md', 'research/k1-punctuation6-v1/contract.py', 'research/k1-punctuation6-v1/coverage.proposed.json', 'research/k1-punctuation6-v1/data_publisher.py', 'research/k1-punctuation6-v1/data_transport.py', 'research/k1-punctuation6-v1/entry.py', 'research/k1-punctuation6-v1/execution-release.disabled.json', 'research/k1-punctuation6-v1/model-provenance.json', 'research/k1-punctuation6-v1/plan.proposed.json', 'research/k1-punctuation6-v1/policy-amendment.json', 'research/k1-punctuation6-v1/publication-release.json', 'research/k1-punctuation6-v1/source-reuse-lock.json', 'research/k1-punctuation6-v1/test_contract.py', 'research/k1-punctuation6-v1/test_data_publisher.py', 'research/k1-punctuation6-v1/test_tts_finalize.py', 'research/k1-punctuation6-v1/test_tts_host.py', 'research/k1-punctuation6-v1/test_tts_public_schema.py', 'research/k1-punctuation6-v1/test_tts_publish.py', 'research/k1-punctuation6-v1/test_tts_semantics.py', 'research/k1-punctuation6-v1/test_tts_worker.py', 'research/k1-punctuation6-v1/tts-release.json', 'research/k1-punctuation6-v1/tts_common.py', 'research/k1-punctuation6-v1/tts_finalize.py', 'research/k1-punctuation6-v1/tts_gate.py', 'research/k1-punctuation6-v1/tts_host.py', 'research/k1-punctuation6-v1/tts_public_schema.py', 'research/k1-punctuation6-v1/tts_publish.py', 'research/k1-punctuation6-v1/tts_semantics.py', 'research/k1-punctuation6-v1/tts_worker.py', 'research/qwen6_tts/generate_six.py', 'research/qwen6_tts/import_preflight.py', 'research/qwen6_tts/model-lock.json', 'research/qwen6_tts/model-locks.json', 'research/qwen6_tts/runtime-lock.json', 'research/qwen6_tts/runtime_gate.py', 'research/qwen6_tts/setup_diagnostics.py', 'research/qwen6_tts/setup_locked.py', 'research/qwen6_tts/supervision.py', 'research/qwen6_tts/test_supervision.py', 'research/qwen6_tts/wheel_identity.py'])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw, *, maximum=2 * 1024 ** 2, canonical_only=True):
    require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded JSON bytes")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    require(type(value) is dict, "JSON object required")
    require(not canonical_only or canonical(value) == raw, "canonical JSON bytes")
    return value


def validate_authority(value, *, expected_authority_sha256):
    """Check a separately reviewed authority, never establish it from fetched data."""
    metadata._hash(expected_authority_sha256)
    metadata._exact(value, ("schema", "input", "source_tree", "data_parent", "source_files"),
                    "exact independently frozen TTS authority")
    require(value["schema"] == "k1-asr18-reviewed-tts-authority-v1", "actual authority schema")
    metadata.validate_source_authority_shape(value["input"])
    metadata._hash(value["source_tree"], metadata.HEAD)
    require(value["data_parent"] == value["input"]["source_commit"], "data result has sole actual source parent")
    require(metadata.digest(value) == expected_authority_sha256, "independent actual TTS authority hash")
    files = value["source_files"]
    require(type(files) is dict and sorted(files) == list(SOURCE_ROSTER), "exact 42-file source freeze roster")
    for item in files.values():
        metadata._exact(item, ("bytes", "sha256"), "source file byte identity")
        metadata._integer(item["bytes"], 0, 8 * 1024 ** 2, "bounded public source file")
        metadata._hash(item["sha256"])
    for name, pin in VERIFIER_PINS.items():
        require(files[SOURCE_DIRECTORY + "/" + name] == pin, "reviewed verifier implementation pin")
    return json.loads(canonical(value))


def run_identity(authority):
    inp = authority["input"]
    return {"repository": inp["repository"], "run_id": inp["run_id"],
            "run_attempt": inp["run_attempt"], "job_key": inp["job_key"],
            "executed_head_sha": inp["source_commit"]}


def parse_manifest(raw, authority):
    inp = authority["input"]
    require(sha(raw) == inp["outer_manifest_sha256"], "reviewed outer manifest identity")
    value = decode(raw, maximum=MAX_MANIFEST)
    metadata._exact(value, ("schema", "identity", "purpose", "fixture_only", "source_closure_sha256",
        "numeric_job_id", "archive_bytes", "archive_sha256", "chunk_bytes", "chunk_count", "chunks"), "manifest fields")
    require(value["schema"] == "k1-data-branch-manifest-v1" and value["purpose"] == "tts" and
            value["fixture_only"] is False, "actual TTS data only")
    require(canonical(value["identity"]) == canonical(run_identity(authority)), "actual source/run/attempt/job")
    require(type(value["numeric_job_id"]) is int and value["numeric_job_id"] == inp["job_id"], "actual numeric job")
    for manifest_key, input_key in (("source_closure_sha256", "source_freeze_sha256"),
        ("archive_sha256", "archive_sha256"), ("archive_bytes", "archive_bytes")):
        require(canonical(value[manifest_key]) == canonical(inp[input_key]), "manifest independent authority binding")
    require(type(value["chunk_bytes"]) is int and value["chunk_bytes"] == MAX_CHUNK, "fixed chunk size")
    expected_count = (inp["archive_bytes"] + MAX_CHUNK - 1) // MAX_CHUNK
    require(type(value["chunk_count"]) is int and value["chunk_count"] == expected_count <= 48, "complete chunk count")
    require(type(value["chunks"]) is list and len(value["chunks"]) == expected_count, "complete ordered chunk list")
    for index, row in enumerate(value["chunks"]):
        metadata._exact(row, ("index", "path", "bytes", "sha256"), "chunk exact fields")
        require(type(row["index"]) is int and row["index"] == index and
                row["path"] == DATA_DIRECTORY + f"/chunks/{index:04d}.bin", "fixed raw-data chunk path/order")
        require(type(row["bytes"]) is int and row["bytes"] == min(MAX_CHUNK, inp["archive_bytes"] - index * MAX_CHUNK), "chunk byte count")
        metadata._hash(row["sha256"])
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


class RawDataReader:
    """One manifest plus <=48 immutable raw GETs, no retry or redirect handler.

    The private opener argument exists solely for stdlib fake-response tests.
    Count bytes actually returned by every read before checking status/size.
    Once any failure occurs the instance is terminal, including uncertain GETs.
    """
    def __init__(self, *, _opener=None):
        self._opener = _opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.gets = 0
        self.response_body_bytes = 0
        self.used = False
        self.failed = False

    def _get(self, commit, relative, cap):
        require(not self.failed and self.gets < 49, "terminal or exhausted reader")
        allowed = relative == DATA_DIRECTORY + "/manifest.json" or relative in {
            DATA_DIRECTORY + f"/chunks/{index:04d}.bin" for index in range(48)}
        require(allowed and metadata.HEAD.fullmatch(commit), "fixed immutable raw path only")
        require(self.response_body_bytes < MAX_BODY, "complete response-body reservation exhausted")
        url = "https://raw.githubusercontent.com/jiying2007/kws-data/" + commit + "/" + relative
        request = urllib.request.Request(url, method="GET", headers={"Accept-Encoding": "identity"})
        self.gets += 1  # An uncertain open still consumes its GET; never retry.
        response = None
        try:
            try:
                response = self._opener.open(request, timeout=30)
            except urllib.error.HTTPError as error:
                response = error
            body = bytearray()
            while True:
                allowance = min(65536, cap + 1 - len(body), MAX_BODY - self.response_body_bytes)
                require(allowance > 0, "response body reached bounded read ceiling")
                block = response.read(allowance)
                require(type(block) is bytes and len(block) <= allowance, "bounded raw read contract")
                self.response_body_bytes += len(block)
                body.extend(block)
                require(len(body) <= cap, "response body exceeds member ceiling")
                if not block:
                    break
            require(response.getcode() == 200 and response.geturl() == url, "HTTP error or redirected URL")
            encoding = response.headers.get("Content-Encoding", "identity")
            require(encoding.lower() == "identity", "no compressed HTTP body")
            return bytes(body)
        except BaseException:
            self.failed = True
            raise
        finally:
            if response is not None:
                response.close()

    def read_complete(self, authority, *, expected_authority_sha256):
        require(not self.used, "one reader invocation; no resume or retry")
        self.used = True
        try:
            authority = validate_authority(authority, expected_authority_sha256=expected_authority_sha256)
            inp = authority["input"]
            manifest_raw = self._get(inp["data_commit"], DATA_DIRECTORY + "/manifest.json", MAX_MANIFEST)
            manifest = parse_manifest(manifest_raw, authority)
            chunks = []
            for row in manifest["chunks"]:
                raw = self._get(inp["data_commit"], row["path"], MAX_CHUNK)
                require(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "actual complete chunk custody")
                chunks.append(raw)
            payload = b"".join(chunks)
            require(len(payload) == inp["archive_bytes"] and sha(payload) == inp["archive_sha256"], "actual complete ZIP custody")
            return {"archive": payload, "manifest": manifest_raw, "accounting": self.accounting()}
        except BaseException:
            self.failed = True
            raise

    def accounting(self):
        return {"gets": self.gets, "response_body_bytes": self.response_body_bytes,
                "response_body_reservation_bytes": MAX_BODY, "retries": 0,
                "redirects": 0, "extra_source_gets": 0, "failed": self.failed,
                "execution_enabled": False}


def _read_regular(path, maximum):
    require(not any(part.is_symlink() for part in (path, *path.parents)), "no source aliases")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, "bounded unaliased source")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == before.st_size, "stable complete source bytes")
    return raw


def verified_source_snapshot(authority, *, expected_authority_sha256, expected_source_freeze_sha256):
    authority = validate_authority(authority, expected_authority_sha256=expected_authority_sha256)
    metadata._hash(expected_source_freeze_sha256)
    require(expected_source_freeze_sha256 == authority["input"]["source_freeze_sha256"], "independently frozen actual source-freeze hash")
    root = HOST_TTS_VERIFIER_ROOT
    require(root.is_dir() and not root.is_symlink(), "fixed host-only verifier source root")
    names = []
    for path in root.rglob("*"):
        require(not path.is_symlink(), "no source symlinks")
        if path.is_file():
            names.append(path.relative_to(root).as_posix())
        else:
            require(path.is_dir() and path.name != "__pycache__", "source files/directories only")
    require(sorted(names) == sorted((*SOURCE_ROSTER, FREEZE_PATH)), "complete exact 43-source closure")
    snapshot = {name: _read_regular(root / name, 8 * 1024 ** 2) for name in names}
    freeze = decode(snapshot[FREEZE_PATH], canonical_only=False)
    require(sha(snapshot[FREEZE_PATH]) == expected_source_freeze_sha256, "actual source-freeze byte identity")
    require(freeze == {"schema": "k1-punctuation6-tts-source-freeze-v1", "files": authority["source_files"]}, "actual source file authority")
    for name, pin in authority["source_files"].items():
        require({"bytes": len(snapshot[name]), "sha256": sha(snapshot[name])} == pin, "complete source byte custody")
    return snapshot


# Fixed child program. Imports resolve only inside the verified private snapshot
# plus the interpreter stdlib. Nothing downloaded is used as Python source.
_VERIFY_CHILD = r'''import json, pathlib, sys
root, archive_path, authority_path = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root / "research/k1-punctuation6-v1"))
from data_transport import validate_archive, canonical
from tts_semantics import validate_tts_members
from tts_gate import verify_source
value = json.loads(authority_path.read_bytes())
i = value["input"]
identity = {"repository": i["repository"], "run_id": i["run_id"], "run_attempt": i["run_attempt"], "job_key": i["job_key"], "executed_head_sha": i["source_commit"]}
verified = verify_source(root)
if verified["source_freeze_sha256"] != i["source_freeze_sha256"]:
    raise ValueError("actual source authority mismatch")
members = validate_archive(archive_path.read_bytes(), identity, "tts", validate_tts_members)
source = json.loads(members["source.json"])
if source["runtime_setup"].get("status") != "verified" or source["runtime_model_load"].get("status") != "loaded_validated":
    raise ValueError("complete TTS runtime setup/load receipts required")
claim = json.loads(members["claim.json"])
if source["source_tree"] != value["source_tree"] or claim["preflight"]["numeric_job_id"] != i["job_id"]:
    raise ValueError("actual tree/job authority mismatch")
outcomes = json.loads(members["outcomes.json"])
if outcomes["status"] != "completed_six_once" or outcomes["consumed_attempts"] != 6 or any(row["status"] != "complete" or type(row["attempts"]) is not int or row["attempts"] != 1 for row in outcomes["cells"]):
    raise ValueError("six natural complete cells required before ASR")
print(canonical({"schema": "k1-asr18-tts-verifier-result-v1", "verified": True}).decode(), end="")
'''


def verify_complete_package(payload, authority, *, expected_authority_sha256, expected_source_freeze_sha256):
    """Run only the pinned pure-stdlib semantic verifier; no model setup/inference."""
    snapshot = verified_source_snapshot(authority, expected_authority_sha256=expected_authority_sha256,
                                        expected_source_freeze_sha256=expected_source_freeze_sha256)
    require(type(payload) is bytes and len(payload) == authority["input"]["archive_bytes"] <= MAX_ARCHIVE and
            sha(payload) == authority["input"]["archive_sha256"], "reviewed complete archive bytes")
    # Source reads are snapshotted first: imports cannot race a mutable stage.
    with tempfile.TemporaryDirectory(prefix="k1-asr18-verify-") as directory:
        root = Path(directory)
        source = root / "source"
        for name, raw in snapshot.items():
            path = source / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o400)
        archive_path, authority_path = root / "archive.zip", root / "authority.json"
        archive_path.write_bytes(payload)
        authority_path.write_bytes(canonical(authority))
        # A regular stdout file enforces the fixed small-result protocol after
        # child completion; RLIMIT_FSIZE is set before any retained import.
        command = [sys.executable, "-I", "-S", "-B", "-c",
            "import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(4096,4096));\n" + _VERIFY_CHILD,
            str(source), str(archive_path), str(authority_path)]
        with (root / "result").open("w+b") as output:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                cwd=root, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                timeout=20, check=False)
            require(result.returncode == 0, "pinned TTS semantic/source verification failed")
            output.seek(0)
            result_raw = output.read(4097)
            require(result_raw == canonical({"schema": "k1-asr18-tts-verifier-result-v1", "verified": True}), "bounded verifier result")
    # Retained verifier has already proven canonical ZIP, allowlists and bounds.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member_raw = archive.read(MEMBER_MANIFEST)
        require(sha(member_raw) == authority["input"]["member_manifest_sha256"], "independent member manifest hash")
        members = {info.filename: archive.read(info) for info in archive.infolist() if info.filename != MEMBER_MANIFEST}
    return members
