"""Fixed audio-free K1 ASR18 evidence custody and pre-forward capacity proof.

Only the local reviewed host's exact collector/schema validator can admit public
members. There is no public arbitrary-files packaging API, log/base64 emitter,
model import, network operation, inference claim, or history inventory input.
"""
from __future__ import annotations
import hashlib
import ast
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import stat
import zipfile

import campaign_contract as c

REPOSITORY = "jiying2007/kws-pipeline"
PURPOSE = "asr18"
MIB = 1024 ** 2
MAX_ARCHIVE = MAX_COMPRESSED = 12 * MIB
MAX_UNCOMPRESSED = 32 * MIB
MAX_MEMBER = 8 * MIB
MAX_MEMBERS = 4096
MAX_CHUNK = 256 * 1024
MAX_CHUNKS = 48
MAX_MANIFEST = 16 * 1024
MAX_BODY = 16 * MIB
SHA = re.compile(r"[0-9a-f]{64}\Z")
HEAD = re.compile(r"[0-9a-f]{40}\Z")
MEMBER_MANIFEST = "__native_evidence_manifest__.json"
HERE = Path(__file__).resolve().parent
_POLICY_KEYS = ("source_branch", "result_branch", "directory", "workflow", "job_key")
POLICY = {key: c.current_scope()[key] for key in _POLICY_KEYS}
LEGACY_POLICY = {key: c.LEGACY_V1_SCOPE[key] for key in _POLICY_KEYS}
require = c.require
canonical = c.canonical
decode = lambda raw: c.decode_json(raw, require_canonical=False)
sha = c.digest


def policy(purpose=PURPOSE):
    require(type(purpose) is str and purpose == PURPOSE, "fixed ASR18 transport purpose")
    return dict(POLICY)


def identity(value, purpose=PURPOSE):
    policy(purpose)
    require(type(value) is dict and set(value) == {"repository", "run_id", "run_attempt",
        "job_key", "executed_head_sha"}, "fixed evidence identity fields")
    require(value["repository"] == REPOSITORY and value["job_key"] == "asr18", "same repository and ASR18 job")
    require(type(value["run_id"]) is int and 0 < value["run_id"] < 10 ** 20, "actual numeric run identity")
    require(type(value["run_attempt"]) is int and value["run_attempt"] == 1, "attempt one only")
    require(type(value["executed_head_sha"]) is str and HEAD.fullmatch(value["executed_head_sha"]), "source Git identity")
    require(value["run_id"] not in c.SEALED_ASR_RUNS and value["executed_head_sha"] not in c.SEALED_ASR_SOURCES,
        "sealed earlier ASR identity is not reusable")


def path(name):
    require(type(name) is str and 0 < len(name) <= 240 and name.isascii()
        and "\\" not in name and all(32 < ord(ch) < 127 for ch in name), "canonical public relative path")
    require(str(PurePosixPath(name)) == name and not name.startswith("/") and
        all(part not in ("", ".", "..") for part in name.split("/")), "canonical public relative path")
    return name


def git_sha(raw, kind="blob"):
    require(type(raw) is bytes and kind in ("blob", "tree", "commit"), "Git object type")
    return hashlib.sha1(kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _host():
    module = importlib.import_module("hosted_run")
    require(Path(module.__file__).resolve() == HERE / "hosted_run.py", "fixed reviewed host validator origin")
    return module


ROOT_JSON = frozenset(("host-receipt.json", "terminal-outcomes.json", "run-preflight.json",
    "input-freeze.json", "public-screening-receipt.json", "new-cell-projection.json",
    "acquisition-receipt.json", "image-transfer.json", "asr-setup-public.json", "cpu-receipt.json",
    "cleanup-receipt.json", "supervision-receipt.json", "authority/execution-freeze.json",
    "authority/source-claim.json", "admission.json", "order/execution-plan.json", "order/worker-input-binding.json",
    "policy/policy.json", "policy/policy-freeze.json", "policy/budget-proof.json"))
WORKER_JSON = frozenset(("prepared.json", "run-identity.json", "kernel-scope.json", "cpu-process-guard.json",
    "blind-job.json", "execution-plan.json", "progress.json", "contract.json", "decoder-inputs.json",
    "model-load.json", "native-source-lock.json", "native-asset-lock.json", "native-config.json",
    "native-resolved-generation-config.json", "native-frontend-config.json", "optional-tool-probes.json",
    "load-receipt.json", "worker-complete.json", "worker-failure.json"))
CLIP_JSON = frozenset(("started.json", "completed.json", "raw-output.json", "raw-native-output.json",
    "raw-transcript.json", "raw-native-row.json", "input-capture.json", "pre-finalize.json"))
NATIVE_SUFFIX = frozenset(("native-observation.json", "native-generation-return.json", "native-tokenizer-input.json",
    "native-inference-return.json", "native-returned-raw.json"))
ASSETS = {
    "qwen06": frozenset(("chat_template.json", "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json")),
    "sensevoice": frozenset(("am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model", "config.yaml", "configuration.json")),
    "whisper": frozenset(("whisper/assets/mel_filters.npz", "whisper/assets/multilingual.tiktoken")),
}
BINARY_ASSETS = frozenset(("sensevoice/retained-assets/chn_jpn_yue_eng_ko_spectok.bpe.model",
    "whisper/retained-assets/whisper/assets/mel_filters.npz",
    "whisper/retained-assets/whisper/assets/multilingual.tiktoken"))
# One already-reviewed source license has no suffix. Its independent fixed
# identity is required in addition to the complete source-freeze binding below;
# a caller-supplied/rebound freeze cannot authorize different license bytes.
SOURCE_TEXT_EXCEPTIONS = {
    "source/research/qwen6_asr/FUNASR_MODEL_LICENSE": {
        "bytes": 5306, "sha256": "7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8"},
}
# Independently fixed official setup source, copied by the retained finalizer.
# These identities are anchored to the ORIGINAL source-lock bytes, never to a
# caller-supplied source freeze or mutable setup/native lock. Only the locked
# mel filter is binary; LICENSE and both tokenizer vocabularies are UTF-8 text.
OFFICIAL_WHISPER_SOURCE_LOCK_SHA256 = "0635fb8759fb56439f59db6956612f78cc8903c239f81f3eb7f9b0c47eb99649"
OFFICIAL_WHISPER_SOURCE_PREFIX = "whisper-setup/official-source/"
_OFFICIAL_WHISPER_SOURCE_ROWS = (
    ('LICENSE', 1063, 'b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d'),
    ('README.md', 8246, '38c180c2a8d8ba628a3e131ad42bf06c8a51bb681efb30fbe28d5a0f4b65f288'),
    ('pyproject.toml', 1418, '9c8111da3701070095304ea5e12b6301eddcca3bb65e491a982b102d6042b1c4'),
    ('requirements.txt', 140, '70063b7cde70331c8d9187bb459c27fbe87bf3335cebc3ee55675b9d91262599'),
    ('whisper/__init__.py', 7432, '79ecbed714783b6230931a3dba6d5da6d386e2d44f9fc27256e4fc72751e1f62'),
    ('whisper/__main__.py', 35, 'c519d586c3a6b639f64e59d2d6fd6b591a67ecd739f9727492911fc4ce4d545a'),
    ('whisper/assets/gpt2.tiktoken', 835554, '306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930'),
    ('whisper/assets/mel_filters.npz', 4271, '7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c'),
    ('whisper/assets/multilingual.tiktoken', 816730, 'b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126'),
    ('whisper/audio.py', 4945, '36ed287b2de3eaf40207cd2a44a8d45d57ea5c212a39df07401d4fcb4d5a5a1a'),
    ('whisper/decoding.py', 32155, '94084fedc5af74cabf8e1883a85552700057081f00ea669ad49435ecb29039db'),
    ('whisper/model.py', 12209, 'f82b0f40e29076506f1e34607a8008782f0586750a8e8bd6d8e1cee5c41b9329'),
    ('whisper/normalizers/__init__.py', 130, 'f18fcdce4caee7f2e80e4c01ab0457d3a080a872a29489542c6d23c4b0bba572'),
    ('whisper/normalizers/basic.py', 2064, '4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38'),
    ('whisper/normalizers/english.json', 56128, '6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465'),
    ('whisper/normalizers/english.py', 21001, 'd31a52685b5b42628ea4e3343c3a7925ce5b2f317e6c0ec4575677dd5c8b5d2c'),
    ('whisper/timing.py', 12697, 'bb091c973c428697012f0f8d1b4f5f3a605cd93c5d1c129957f712a6a878a59c'),
    ('whisper/tokenizer.py', 12338, '3b48e361a7e95b4ec0356ca6d72bba635778aa10269153136ee7bc34cae30b85'),
    ('whisper/transcribe.py', 30375, '176f5ab30e961bec2a7872f5701a1f5bf3cbfc77946e50512f0f0cf00a1ceacf'),
    ('whisper/triton_ops.py', 3646, '26c5082f029bbd2b7ae77ecf5a3f7f196f220e5850e2c9d0ec42961caf0dde4c'),
    ('whisper/utils.py', 11839, 'f46aa555363ab42a455cd7c7f8c108dc2f69c5a94c6354c73ae96d671d9b1c37'),
    ('whisper/version.py', 25, 'c4d27ecccdd52a9cded762f1a82da3152db12cbc0aa6ea781f945446a143cfaf'),
)


def official_whisper_source_inventory():
    """Return a fresh exact 22-member identity map from the original fixed lock."""
    return {OFFICIAL_WHISPER_SOURCE_PREFIX + name: {"bytes": size, "sha256": digest}
        for name, size, digest in _OFFICIAL_WHISPER_SOURCE_ROWS}


def official_whisper_source_members(files):
    """Validate retained official setup bodies without filesystem/model access.

    Interrupted copytree may retain only fully copied, independently locked
    members. A completed setup/finalize receipt requires all 22 source bodies.
    Metadata schemas are still checked by the caller's full evidence validator.
    """
    require(type(files) is dict, "official Whisper source evidence map")
    for name in ("whisper-setup/source-lock.json",
            "source/research/source-screen32-whisper-v1/source-lock.json"):
        if name in files:
            require(type(files[name]) is bytes and sha(files[name]) == OFFICIAL_WHISPER_SOURCE_LOCK_SHA256,
                "original fixed official Whisper source lock required; caller freeze cannot rebind it")
    expected = official_whisper_source_inventory()
    result = {}
    for name, raw in files.items():
        if type(name) is not str or not (name == OFFICIAL_WHISPER_SOURCE_PREFIX.rstrip("/") or
                name.startswith(OFFICIAL_WHISPER_SOURCE_PREFIX)):
            continue
        path(name)
        require(name in expected, "member outside fixed official Whisper source roster")
        require(type(raw) is bytes and {"bytes": len(raw), "sha256": sha(raw)} == expected[name],
            "exact original official Whisper source bytes/hash required")
        if name != OFFICIAL_WHISPER_SOURCE_PREFIX + "whisper/assets/mel_filters.npz":
            raw.decode("utf-8")
        result[name] = expected[name]
    for name in ("whisper-setup/setup-receipt.json", "whisper-setup/phase-finalize-complete.json",
            "whisper-setup/phase-finalize-progress.json"):
        if name in files:
            value = decode(files[name])
            require(type(value) is dict, "official Whisper setup completion object")
            if value.get("status") == "complete":
                require(set(result) == set(expected), "complete Whisper setup requires all 22 official source bodies")
    return result


FORBIDDEN_KEYS = frozenset(("environment", "env", "environ", "credentials", "credential", "password", "secret",
    "access_token", "api_key", "authorization", "private_history", "historical_inventory", "inventory_sha256",
    "history_sha256", "history_paths", "history_count", "history_matches", "matched_history", "private_path",
    "audio_bytes", "wav_bytes", "audio_base64", "samples", "waveform"))


def _public_metadata(value, depth=0):
    require(depth <= 48, "bounded JSON nesting")
    if type(value) is dict:
        for key, item in value.items():
            require(type(key) is str and key.lower() not in FORBIDDEN_KEYS, "nonpublic field prohibited")
            _public_metadata(item, depth + 1)
    elif type(value) is list:
        for item in value: _public_metadata(item, depth + 1)
    elif type(value) is str:
        require(not value.startswith(("/workspace/", "/home/", "/Users/", "/root/", "file://"))
            and not re.match(r"^[A-Za-z]:[\\/]", value), "private host path prohibited")
        require(not value.startswith(("ghp_", "github_pat_", "Bearer ", "-----BEGIN PRIVATE KEY")), "credential payload prohibited")
    else:
        require(value is None or type(value) in (int, float, bool), "public JSON scalar")


def _source_members(files):
    raw = files.get("authority/execution-freeze.json")
    require(raw is not None, "complete immutable source freeze required")
    freeze = decode(raw)
    require(type(freeze) is dict and set(freeze) == {"schema", "inference_code", "setup_code", "host_code"}
        and freeze["schema"] == "k1-asr18-source-freeze-v1", "fixed source freeze schema")
    require(all(type(freeze[k]) is dict and freeze[k] for k in ("inference_code", "setup_code", "host_code")), "complete source closure maps")
    names = {"source/" + path(n): row for n, row in freeze["host_code"].items()}
    name = "source/research/k1-punctuation6-asr18-v1/execution-freeze.json"
    names[name] = {"bytes": len(raw), "sha256": sha(raw)}
    for member, row in names.items():
        require(type(row) is dict and set(row) == {"bytes", "sha256"} and type(row["bytes"]) is int
            and 0 <= row["bytes"] <= MAX_MEMBER and type(row["sha256"]) is str and SHA.fullmatch(row["sha256"]), "bounded source identity")
        require(member in files and {"bytes": len(files[member]), "sha256": sha(files[member])} == row,
            "every exact source body retained")
        require(PurePosixPath(member).suffix in (".py", ".json", ".md", ".yml", ".yaml", ".txt") or
            SOURCE_TEXT_EXCEPTIONS.get(member) == row, "public source text or exact reviewed license only")
        files[member].decode("utf-8")
    return names


def _native_body_members(files):
    result = {}
    for storage in ("output", "frozen"):
        for family in c.FAMILIES:
            prefix = storage + "/" + family + "/"
            for leaf, field in (("native-source-lock.json", "files"), ("native-asset-lock.json", "assets")):
                if prefix + leaf not in files: continue
                lock = decode(files[prefix + leaf])
                expected_schema = "retained-asr-source-lock-v1" if field == "files" else "retained-asr-model-lock-v1"
                require(type(lock) is dict and lock.get("schema") == expected_schema, "native dependency lock schema")
                rows = lock.get(field)
                require(type(rows) is list and rows, "complete native dependency roster")
                if field == "files": rows = rows + lock.get("empty_source_files", [])
                for row in rows:
                    require(type(row) is dict, "native dependency row")
                    if field == "assets" and row.get("body_retained") is not True: continue
                    member = path(row.get("path"))
                    old_prefix = "output/" + family + "/"
                    require(member.startswith(old_prefix), "same family native dependency")
                    suffix = member[len(old_prefix):]
                    if field == "files":
                        require((suffix.startswith("retained-source/") and suffix.endswith(".py")) or
                            (family == "sensevoice" and suffix == "retained-source/optional-tool-policy.json" and
                             row.get("role") == "runtime-execution-policy"), "source-only native dependency")
                    else:
                        require((suffix.startswith("retained-assets/") and suffix[len("retained-assets/"):] in ASSETS[family]) or
                            (family == "sensevoice" and suffix == "native-frontend-config.json"), "fixed public asset names only")
                    physical = prefix + suffix
                    require(physical in files and type(row.get("bytes")) is int and row["bytes"] == len(files[physical]) and
                        row.get("sha256") == sha(files[physical]), "full exact native source/dependency bytes required")
                    require(physical not in result or result[physical] == row, "nonconflicting native dependency")
                    result[physical] = row
    return result


def _worker_path(name):
    match = re.fullmatch(r"(output|frozen)/(qwen06|sensevoice|whisper)/(.+)", name)
    if not match: return False
    storage, family, suffix = match.groups()
    if suffix in WORKER_JSON: return True
    if storage != "output": return False
    match = re.fullmatch(r"clip-00000([1-6])/(.+)", suffix)
    if match:
        leaf = match[2]
        if leaf in CLIP_JSON: return True
        step = re.fullmatch(r"update-([0-9]{3})\.json", leaf)
        return family == "whisper" and step is not None and 1 <= int(step[1]) <= 224
    match = re.fullmatch(r"clip-00000[1-6]\.(.+)", suffix)
    return family in ("qwen06", "sensevoice") and match is not None and match[1] in NATIVE_SUFFIX


def validate_members(files, run_identity, *, phase="final"):
    identity(run_identity)
    require(phase in ("prepared", "final") and type(files) is dict and 1 <= len(files) <= MAX_MEMBERS,
        "bounded complete evidence file map")
    require(all(type(raw) is bytes and len(raw) <= MAX_MEMBER for raw in files.values()), "member byte cap")
    for name, raw in files.items():
        path(name)
        require(not (raw[:4] == b"RIFF" and raw[8:12] == b"WAVE") and not raw.startswith((b"fLaC", b"OggS", b"ID3")), "audio bytes prohibited")
        require(PurePosixPath(name).suffix.lower() not in (".wav", ".flac", ".mp3", ".ogg", ".pt", ".pth",
            ".safetensors", ".whl", ".pkl", ".pickle", ".key", ".pem", ".zip", ".gz", ".bin"), "audio/weights/packages/credentials prohibited")
    official = official_whisper_source_members(files)
    source = _source_members(files)
    _source_claim_mode(files)
    native = _native_body_members(files)
    for name, raw in files.items():
        if name in source or name in official: continue
        if name in native:
            if "/".join(name.split("/")[1:]) not in BINARY_ASSETS: raw.decode("utf-8")
            continue
        # Setup metadata's precise finite roster and JSON schemas belong to the
        # fixed host validator. A .json suffix alone never admits any member.
        require(name in ROOT_JSON or _worker_path(name) or name in _host().public_setup_members(),
            "member outside exact public allowlist")
        value = decode(raw)
        require(type(value) is dict, "public metadata object required")
        _public_metadata(value)
        if _worker_path(name) or name in ("cpu-receipt.json", "cleanup-receipt.json", "supervision-receipt.json", "image-transfer.json", "asr-setup-public.json") or name.startswith("whisper-setup/"):
            validate_member_json(name, value)
    _host().validate_public_evidence(files, run_identity, phase=phase)
    return files


def _zip_files(files, run_identity):
    """Internal bounded compressor; no admission/privacy claim by itself."""
    identity(run_identity)
    require(type(files) is dict and 1 <= len(files) <= MAX_MEMBERS and MEMBER_MANIFEST not in files, "member count cap")
    require(all(type(raw) is bytes and len(raw) <= MAX_MEMBER for raw in files.values()), "member cap")
    manifest = {"schema": "native-evidence-member-manifest-v1", "identity": run_identity,
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


def _collect(root, run_identity, *, phase):
    root = c.safe_path(root)
    names = _host().collect_public(root)
    require(type(names) is list and names == sorted(set(names)) and 1 <= len(names) <= MAX_MEMBERS,
        "fixed complete host collector required")
    files = {path(name): c.read_bytes(root / name, cap=MAX_MEMBER) for name in names}
    return validate_members(files, run_identity, phase=phase)


def build_package(root, run_identity):
    files = _collect(root, run_identity, phase="final")
    payload, manifest, _ = _zip_files(files, run_identity)
    validate_archive(payload, run_identity)
    return payload, manifest


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
        and manifest["schema"] == "native-evidence-member-manifest-v1" and canonical(manifest) == mraw
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


def package_budget_preflight(root, run_identity, *, job, execution_plan, input_binding, policy_path):
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
    trial = dict(actual)
    rng = random.Random(20261012)
    for family in c.FAMILIES:
        prefix = "output/" + family + "/"
        require(prefix + "native-config.json" in actual and prefix + "prepared.json" in actual, "all three actual prepared configurations")
        prepared = decode(actual[prefix + "prepared.json"])
        require(prepared.get("mode") == "prepare" and prepared.get("audio_forward_calls") == 0
            and prepared.get("completed") == 0 and prepared.get("audio_ids") == [], "prepare-only zero native calls")
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
    return {"schema": "k1-asr18-predecode-package-budget-v1", "status": "passed_before_any_audio_forward",
        "new_native_calls": 0, "eligible_clip_count": len(job["clips"]),
        "whisper_full_prefix_steps_per_clip": 224, "transcript_reserve_characters_each_copy": 32768,
        "transcript_copies_per_clip": 2, "extra_json_reserve_bytes": MIB,
        "recognizer_policy_sha256": sha(policy_raw), "worker_input_binding_sha256": sha(canonical(input_binding)),
        "execution_plan_sha256": sha(canonical(execution_plan)), "job_sha256": sha(canonical(job)),
        "compressed_bytes": len(payload), "expanded_member_bytes": expanded,
        "actual_prepared_members": len(actual), "worst_fixture_members": len(trial)}


def manifest_path(purpose=PURPOSE):
    return policy(purpose)["directory"] + "/manifest.json"


def chunk_path(index, purpose=PURPOSE):
    require(type(index) is int and 0 <= index < MAX_CHUNKS, "chunk index")
    return policy(purpose)["directory"] + f"/chunks/{index:04d}.bin"


def build_file_set(payload, run_identity, freeze_sha256, numeric_job_id, purpose=PURPOSE):
    identity(run_identity, purpose)
    manifest, measured = validate_archive(payload, run_identity)
    require(manifest["members"]["authority/execution-freeze.json"]["sha256"] == freeze_sha256, "archive independently approved source closure")
    require(type(freeze_sha256) is str and SHA.fullmatch(freeze_sha256), "source closure hash")
    require(type(numeric_job_id) is int and 0 < numeric_job_id < 10 ** 20, "numeric job identity")
    chunks = [payload[i:i + MAX_CHUNK] for i in range(0, len(payload), MAX_CHUNK)]
    m = {"schema": "k1-asr18-evidence-branch-manifest-v1", "identity": run_identity, "purpose": purpose,
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
    require(canonical(m) == raw and m["schema"] == "k1-asr18-evidence-branch-manifest-v1" and m["purpose"] == purpose,
        "canonical ASR18 outer manifest")
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


def package_path(runner_temp, run_identity):
    identity(run_identity)
    root = c.safe_path(runner_temp)
    require(root.is_dir(), "existing trusted runner temporary directory")
    return root / ("k1-asr18-evidence-" + str(run_identity["run_id"]) + "-attempt-1-" +
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


# Retained public metadata schemas are strict structural grammars: keys, known
# schema names and JSON types only. No earlier run identity, input hash, source
# value or transcript is copied. Native semantic validators add their unchanged
# stronger checks where applicable. Unknown variants stop the whole publication.
_RETAINED_SHAPES = {'0': ['number'],
 '1': ['str'],
 '2': ['object',
       {'bytes': '0', 'filename': '1', 'metadata_sha256': '1', 'sha256': '1', 'source': '1', 'version': '1'}],
 '3': ['object', {'bytes': '0', 'sha256': '1'}],
 '4': ['object', {'setup.cfg': '3', 'setup.py': '3'}],
 '5': ['list', ['1']],
 '6': ['object', {'build-backend': '1', 'requires': '5'}],
 '7': ['scalar'],
 '8': ['list', ['7']],
 '9': ['list', ['1', '8']],
 '10': ['list', ['9']],
 '11': ['object',
        {'bytes': '0',
         'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '10',
         'path': '1',
         'requires_dist': '5',
         'retained_metadata': '1',
         'sha256': '1',
         'version': '1'}],
 '12': ['object',
        {'archive': '3',
         'build_files': '4',
         'build_system': '6',
         'expanded_source_bytes': '0',
         'name': '1',
         'wheel': '11'}],
 '13': ['object', {'setup.py': '3'}],
 '14': ['object',
        {'bytes': '0',
         'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '8',
         'path': '1',
         'requires_dist': '8',
         'retained_metadata': '1',
         'sha256': '1',
         'version': '1'}],
 '15': ['object',
        {'archive': '3',
         'build_files': '13',
         'build_system': '6',
         'expanded_source_bytes': '0',
         'name': '1',
         'wheel': '14'}],
 '16': ['list', ['12', '15']],
 '17': ['bool'],
 '18': ['object',
        {'bytes': '0',
         'codes': '8',
         'exception_types': '8',
         'raw_text_disclosed': '17',
         'stream_scope': '1',
         'tail_bytes': '0',
         'tail_sha256': '1',
         'tail_truncated': '17'}],
 '19': ['object',
        {'package': '1',
         'returncode': '0',
         'stage': '1',
         'status': '1',
         'stderr_diagnostic': '18',
         'wall_seconds': '0'}],
 '20': ['null'],
 '21': ['object',
        {'package': '20',
         'returncode': '0',
         'stage': '1',
         'status': '1',
         'stderr_diagnostic': '18',
         'wall_seconds': '0'}],
 '22': ['list', ['19', '21']],
 '23': ['object', {'buildtmp_bytes': '0', 'free_bytes': '0', 'installed_bytes': '0', 'workspace_bytes': '0'}],
 '24': ['list', ['23']],
 '25': ['object', {'bytes': '0', 'filename': '1', 'sha256': '1'}],
 '26': ['list', ['25']],
 '27': ['list', ['0']],
 '28': ['object',
        {'bundled_native_module': '1',
         'bundled_native_sha256': '1',
         'empty_nonfinite_rejection_checked': '17',
         'float32_values_sha256': '1',
         'float_wav_sha256': '1',
         'frames': '0',
         'libsndfile_version': '1',
         'pcm16_values': '27',
         'pcm_wav_sha256': '1'}],
 '29': ['literal', 'native-import-preflight-v1'],
 '30': ['object',
        {'cffi': '1',
         'funasr': '1',
         'numpy': '1',
         'qwen-asr': '1',
         'scipy': '1',
         'soundfile': '1',
         'sox': '1',
         'torch': '1',
         'transformers': '1'}],
 '31': ['object',
        {'forward_calls': '0',
         'model_constructions': '0',
         'native_io': '28',
         'profile': '1',
         'raw_text_disclosed': '17',
         'schema': '29',
         'stage': '1',
         'status': '1',
         'versions': '30',
         'weight_loads': '0'}],
 '32': ['object',
        {'buildtmp_bytes': '0',
         'image_transfer_reserve_bytes': '0',
         'initial_free_bytes': '0',
         'input_download_bytes': '0',
         'installed_bytes': '0',
         'workspace_bytes': '0'}],
 '33': ['object',
        {'import_preflight.py': '3',
         'model-locks.json': '3',
         'runtime-lock.json': '3',
         'setup_diagnostics.py': '3',
         'setup_locked.py': '3',
         'wheel_identity.py': '3'}],
 '34': ['object',
        {'adapter': '3',
         'bound_models_sha256': '1',
         'limits': '32',
         'profile': '1',
         'retained_files': '33',
         'retained_model_locks': '3',
         'runtime_lock': '3',
         'runtime_scope': '3'}],
 '35': ['literal', 'native-optional-tool-probe-receipt-v1'],
 '36': ['object',
        {'actual_installed_status': '1',
         'availability': '1',
         'completed': '17',
         'external_backend': '1',
         'external_process_executed': '17',
         'intercepted_probes': '0',
         'maximum_import_probes': '0',
         'module_backend_enabled': '17',
         'policy': '1',
         'schema': '35',
         'source': '1',
         'source_sha256': '1'}],
 '37': ['object', {'bytes': '0', 'record': '36', 'sha256': '1'}],
 '38': ['literal', 'qwen12-inference-process-guard-v1'],
 '39': ['object',
        {'clone3': '1',
         'family': '1',
         'fork_vfork_nonthread_clone': '1',
         'rlimit_cpu_seconds': '0',
         'scheduler_granularity_not_zero_overshoot': '17',
         'schema': '38',
         'scope': '1'}],
 '40': ['literal', 'qwen12-import-process-guard-v1'],
 '41': ['object',
        {'container_matches_decode_scope': '17',
         'cpu_accounting': '1',
         'extra_child_cpu_limit_seconds': '0',
         'guard': '39',
         'schema': '40',
         'scope': '1'}],
 '42': ['object', {'bytes': '0', 'record': '41', 'sha256': '1'}],
 '43': ['object', {'import-optional-tool-policy.json': '37', 'import-process-guard.json': '42'}],
 '44': ['literal', 'qwen12-asr-setup-public-projection-v1'],
 '45': ['list', ['1', '5']],
 '46': ['list', ['9', '45']],
 '47': ['object',
        {'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '46',
         'requires_dist': '5',
         'version': '1'}],
 '48': ['object',
        {'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '10',
         'requires_dist': '5',
         'version': '1'}],
 '49': ['object',
        {'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '8',
         'requires_dist': '8',
         'version': '1'}],
 '50': ['list', ['45', '9']],
 '51': ['object',
        {'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'normalized_requires_dist': '50',
         'requires_dist': '5',
         'version': '1'}],
 '52': ['list', ['47', '48', '49', '51']],
 '53': ['object', {'qwen06': '26', 'sensevoice': '26'}],
 '54': ['object', {'byteorder': '1', 'implementation': '1', 'machine': '1', 'system': '1', 'version': '1'}],
 '55': ['object', {'bytes': '0', 'distribution': '1', 'relative_path': '1', 'sha256': '1'}],
 '56': ['list', ['55']],
 '57': ['object', {'disk': '23', 'installed': '52', 'model_assets': '53', 'python': '54', 'source_members': '56'}],
 '58': ['object',
        {'bytes': '0',
         'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'path': '1',
         'requires_dist': '5',
         'sha256': '1',
         'version': '1'}],
 '59': ['object',
        {'bytes': '0',
         'metadata_bytes': '0',
         'metadata_sha256': '1',
         'name': '1',
         'path': '1',
         'requires_dist': '8',
         'sha256': '1',
         'version': '1'}],
 '60': ['map', ['58', '59', '11', '14']],
 '61': ['object',
        {'actual_download_bytes': '0',
         'bootstrap_pip': '2',
         'builds': '16',
         'commands': '22',
         'disk_checkpoints': '24',
         'downloads': '26',
         'import_preflight': '31',
         'locks': '34',
         'model_calls': '0',
         'pre_model_process_compatibility': '43',
         'retry_count': '0',
         'schema': '44',
         'source_receipt_sha256': '1',
         'status': '1',
         'verification': '57',
         'wall_seconds': '0',
         'wheels': '60'}],
 '62': ['literal', 'qwen12-gated-command-v1'],
 '63': ['object', {'exit_code': '0', 'schema': '62', 'status': '1'}],
 '64': ['object',
        {'operation': '1', 'returncode': '0', 'status': '1', 'timeout_seconds': '0', 'wall_seconds': '0'}],
 '65': ['object', {'operation': '1', 'status': '1', 'wall_seconds': '0'}],
 '66': ['list', ['64', '65']],
 '67': ['object',
        {'accounting_status': '1',
         'cleanup_budget_seconds': '0',
         'cleanup_ok': '17',
         'cleanup_overrun_seconds': '0',
         'cleanup_seconds': '0',
         'container_id': '1',
         'errors': '8',
         'gate_result': '63',
         'host_deadline_expired_at_start': '17',
         'host_deadline_overrun': '17',
         'name': '1',
         'operations': '66',
         'ownership': '1',
         'pause_acknowledged': '17',
         'receipt_errors': '8',
         'removal_proof': '1',
         'stage': '1',
         'status': '1'}],
 '68': ['list', ['67']],
 '69': ['object', {'all_cleanup_verified': '17', 'all_removed': '17', 'containers': '68'}],
 '70': ['object', {'closed': '17', 'final_verified': '17', 'stage': '1', 'usage_usec': '0'}],
 '71': ['map', ['70']],
 '72': ['literal', 'qwen12-cumulative-cpu-threshold-v1'],
 '73': ['object', {'qwen06': '0', 'sensevoice': '0', 'setup': '0', 'whisper': '0'}],
 '74': ['object',
        {'all_final_usage_verified': '17',
         'configured_poll_seconds': '0',
         'containers': '71',
         'hard_cumulative_kernel_quota': '17',
         'kill_command_timeout_seconds': '0',
         'maximum_observed_poll_gap_seconds': '0',
         'overshoot_bound': '1',
         'schema': '72',
         'threshold_exceeded': '17',
         'thresholds_seconds': '73',
         'total_threshold_seconds': '0',
         'total_usage_usec': '0',
         'usage_scope': '1'}],
 '75': ['object', {'ifindex': '0', 'rx_bytes': '0'}],
 '76': ['object', {'docker0': '75', 'enP56469s1': '75', 'eth0': '75'}],
 '77': ['object', {'default_interface': '1', 'interfaces': '76'}],
 '78': ['literal', 'screen32-image-transfer-observation-v2'],
 '79': ['object',
        {'all_interface_delta_diagnostic_only_bytes': '0',
         'byte_scope': '1',
         'cumulative_image_transfer_bytes_verified': '17',
         'daemon_completion_verified': '17',
         'declared_config_and_compressed_layer_bytes': '0',
         'declared_manifest_bytes': '0',
         'default_interface_received_delta_bytes': '0',
         'image': '1',
         'image_config_digest': '1',
         'network_after': '77',
         'network_before': '77',
         'pull_client_attempts': '0',
         'pull_client_exit_code': '0',
         'reservation_bytes': '0',
         'reservation_kind': '1',
         'schema': '78',
         'status': '1'}],
 '80': ['object', {'audio_id': '1', 'audio_path': '1', 'wav_sha256': '1'}],
 '81': ['list', ['80']],
 '82': ['literal', 'blind-source-screen32-job-v1'],
 '83': ['object', {'clips': '81', 'schema': '82'}],
 '84': ['object', {'bytes': '0', 'path': '1', 'role': '1', 'sha256': '1'}],
 '85': ['object', {'bytes': '0', 'distribution': '1', 'path': '1', 'role': '1', 'sha256': '1'}],
 '86': ['list', ['84', '85']],
 '87': ['object', {'body_retained': '17', 'bytes': '0', 'kind': '1', 'path': '1', 'sha256': '1'}],
 '88': ['list', ['87']],
 '89': ['object', {'batch_size': '0', 'device': '1', 'dtype': '1'}],
 '90': ['object',
        {'context': '1', 'do_sample': '17', 'eos_token_id': '27', 'force_language': '20', 'max_new_tokens': '0'}],
 '91': ['object',
        {'bad_words_ids': '20',
         'begin_suppress_tokens': '20',
         'custom_logits_processors': '8',
         'custom_stopping_criteria': '8',
         'execution': '89',
         'force_words_ids': '20',
         'forced_bos_token_id': '20',
         'forced_eos_token_id': '20',
         'generation_mode': '1',
         'native_request': '90',
         'num_beams': '0',
         'num_return_sequences': '0',
         'resolved_generation_config_sha256': '1',
         'sequence_bias': '20',
         'stop_strings': '20',
         'suppress_tokens': '20'}],
 '92': ['literal', 'speech-asr-native-capture-v2'],
 '93': ['object',
        {'audio_frames': '0',
         'channels': '0',
         'padding_frames': '0',
         'pcm_format': '1',
         'preprocessing': '1',
         'processed_end_frame': '0',
         'processed_start_frame': '0',
         'sample_rate': '0',
         'trimmed_frames': '0'}],
 '94': ['object',
        {'capture_stage': '1',
         'generated_token_ids': '27',
         'generated_tokens': '0',
         'kind': '1',
         'stop_reason': '1',
         'truncated': '17'}],
 '95': ['object',
        {'acoustic_verbatim_completeness': '1',
         'asset_lock_sha256': '1',
         'effective_config': '91',
         'effective_config_sha256': '1',
         'evidence_class': '92',
         'input': '93',
         'native_config_sha256': '1',
         'peer_transcript_received': '17',
         'schema_version': '0',
         'source_lock_sha256': '1',
         'target_context_received': '17',
         'termination': '94',
         'transcript_postcorrection_applied': '17'}],
 '96': ['literal', 'retained-asr-completed-v1'],
 '97': ['object',
        {'acoustic_verbatim_completeness': '1',
         'audio_frames': '0',
         'audio_id': '1',
         'decoder_completed': '17',
         'executed_head_sha': '1',
         'executed_source_files': '86',
         'file_sha256': '1',
         'loaded_model_assets': '88',
         'model_lock_sha256': '1',
         'native_capture': '95',
         'native_decode_config_sha256': '1',
         'original_job_sha256': '1',
         'pcm_sha256': '1',
         'processed_end_frame': '0',
         'processed_start_frame': '0',
         'raw_native_output_sha256': '1',
         'raw_output_sha256': '1',
         'recognizer_id': '1',
         'recognizer_identity_sha256': '1',
         'recognizer_policy_sha256': '1',
         'remaining_plan_sha256': '1',
         'request_id': '1',
         'run_id': '1',
         'schema': '96',
         'source_id': '1',
         'source_lock_sha256': '1',
         'started_sha256': '1',
         'status': '1'}],
 '98': ['object',
        {'bytes': '0',
         'channels': '0',
         'dtype': '1',
         'encoding': '1',
         'frame_count': '0',
         'normalization': '1',
         'sample_order': '1',
         'sample_rate_hz': '0',
         'sha256': '1',
         'shape': '27'}],
 '99': ['object', {'mix': '17', 'resample': '17', 'trim': '17'}],
 '100': ['object',
         {'bytes': '0',
          'channels': '0',
          'dtype': '1',
          'encoding': '1',
          'frame_count': '0',
          'sample_order': '1',
          'sample_rate_hz': '0',
          'sha256': '1',
          'shape': '27'}],
 '101': ['literal', 'cosyvoice12-pcm-binding-v1'],
 '102': ['object', {'bytes': '0', 'format': '1', 'header_bytes': '0', 'sha256': '1'}],
 '103': ['object',
         {'float32_pcm': '98',
          'float32_pcm_descriptor_sha256': '1',
          'opaque_id': '1',
          'operations': '99',
          'raw_pcm16': '100',
          'raw_pcm16_descriptor_sha256': '1',
          'schema': '101',
          'wav': '102'}],
 '104': ['object',
         {'completeness_scope': '1',
          'eos_terminated': '17',
          'generated_tokens': '0',
          'language': '1',
          'max_new_tokens': '0'}],
 '105': ['object',
         {'_from_model_config': '17',
          'assistant_confidence_threshold': '0',
          'assistant_early_exit': '20',
          'assistant_lookbehind': '0',
          'bad_words_ids': '20',
          'begin_suppress_tokens': '20',
          'bos_token_id': '20',
          'cache_config': '20',
          'cache_implementation': '20',
          'constraints': '20',
          'decoder_start_token_id': '20',
          'disable_compile': '17',
          'diversity_penalty': '0',
          'do_sample': '17',
          'dola_layers': '20',
          'early_stopping': '17',
          'encoder_no_repeat_ngram_size': '0',
          'encoder_repetition_penalty': '0',
          'eos_token_id': '27',
          'epsilon_cutoff': '0',
          'eta_cutoff': '0',
          'exponential_decay_length_penalty': '20',
          'force_words_ids': '20',
          'forced_bos_token_id': '20',
          'forced_eos_token_id': '20',
          'guidance_scale': '20',
          'is_assistant': '17',
          'length_penalty': '0',
          'low_memory': '20',
          'max_length': '0',
          'max_matching_ngram_size': '20',
          'max_new_tokens': '0',
          'max_time': '20',
          'min_length': '0',
          'min_new_tokens': '20',
          'min_p': '20',
          'no_repeat_ngram_size': '0',
          'num_assistant_tokens': '0',
          'num_assistant_tokens_schedule': '1',
          'num_beam_groups': '0',
          'num_beams': '0',
          'num_return_sequences': '0',
          'output_attentions': '17',
          'output_hidden_states': '17',
          'output_logits': '20',
          'output_scores': '17',
          'pad_token_id': '20',
          'penalty_alpha': '20',
          'prefill_chunk_size': '20',
          'prompt_lookup_num_tokens': '20',
          'remove_invalid_values': '17',
          'renormalize_logits': '17',
          'repetition_penalty': '0',
          'return_dict_in_generate': '17',
          'return_legacy_cache': '20',
          'sequence_bias': '20',
          'stop_strings': '20',
          'suppress_tokens': '20',
          'target_lookbehind': '0',
          'temperature': '0',
          'token_healing': '17',
          'top_k': '0',
          'top_p': '0',
          'transformers_version': '1',
          'typical_p': '0',
          'use_cache': '17',
          'watermarking_config': '20'}],
 '106': ['object', {'device': '1', 'dtype': '1', 'shape': '27', 'values': '27'}],
 '107': ['object', {'device': '1', 'dtype': '1', 'shape': '8', 'values': '0'}],
 '108': ['object',
         {'_bos_token_tensor': '20',
          '_decoder_start_token_tensor': '20',
          '_eos_token_tensor': '106',
          '_pad_token_tensor': '107'}],
 '109': ['object', {'padding': '17', 'return_attention_mask': '17', 'sampling_rate': '0', 'truncation': '17'}],
 '110': ['literal', 'source-screen-native-v2-observation-v1'],
 '111': ['object',
         {'acoustic_verbatim_completeness': '1',
          'custom_logits_processors': '8',
          'custom_logits_processors_resolved_generation_config': '105',
          'custom_logits_processors_runtime_special_tokens': '108',
          'custom_stopping_criteria': '8',
          'custom_stopping_criteria_resolved_generation_config': '105',
          'custom_stopping_criteria_runtime_special_tokens': '108',
          'frontend_call': '109',
          'generation_input_ids_seq_length': '0',
          'generation_mode': '1',
          'generation_returned': '17',
          'inference_returned': '17',
          'input': '93',
          'input_binding_sha256': '1',
          'logits_input_ids_seq_length': '0',
          'opaque_id': '1',
          'prepared': '105',
          'returned_generated_token_ids': '27',
          'schema': '110'}],
 '112': ['object', {'contiguous_copy': '17', 'original_stride': '27'}],
 '113': ['object',
         {'attention_mask': '112', 'feature_attention_mask': '112', 'input_features': '112', 'input_ids': '112'}],
 '114': ['object', {'dtype': '1', 'sha256': '1', 'shape': '27'}],
 '115': ['object',
         {'attention_mask': '114', 'feature_attention_mask': '114', 'input_features': '114', 'input_ids': '114'}],
 '116': ['object',
         {'decoded_skip_special_tokens': '1',
          'decoded_with_special_tokens': '1',
          'force_language': '20',
          'input_binding': '103',
          'metadata': '104',
          'native_capture': '95',
          'native_v2_observation': '111',
          'native_v2_validated': '17',
          'processor_input_layout': '113',
          'processor_tensor_evidence': '115',
          'prompt_context': '1',
          'token_ids': '27',
          'transcript_repair_applied': '17'}],
 '117': ['object', {'completeness': '1', 'quality_flags': '8', 'raw_text': '1'}],
 '118': ['object', {'bytes': '0', 'path': '1', 'sha256': '1'}],
 '119': ['literal', 'retained-asr-raw-transcript-v1'],
 '120': ['object',
         {'acoustic_verbatim_completeness': '1',
          'actual_text': '1',
          'native_output': '118',
          'native_row': '118',
          'output_scope': '1',
          'request_id': '1',
          'schema': '119',
          'truncated': '17'}],
 '121': ['literal', 'retained-asr-started-v1'],
 '122': ['object',
         {'audio_frames': '0',
          'audio_id': '1',
          'executed_head_sha': '1',
          'file_sha256': '1',
          'model_lock_sha256': '1',
          'native_decode_config_sha256': '1',
          'original_job_sha256': '1',
          'pcm_sha256': '1',
          'recognizer_id': '1',
          'recognizer_identity_sha256': '1',
          'recognizer_policy_sha256': '1',
          'remaining_plan_sha256': '1',
          'request_id': '1',
          'run_id': '1',
          'schema': '121',
          'source_id': '1',
          'source_lock_sha256': '1'}],
 '123': ['literal', 'bounded-source-screen-asr-execution-v1'],
 '124': ['object',
         {'asset_lock_sha256': '1',
          'batch_size': '0',
          'candidate_count': '0',
          'context': '1',
          'decoder_manifest_sha256': '1',
          'expected_schema_sha256': '1',
          'hotwords': '8',
          'human_gold': '17',
          'language': '1',
          'model_dtype': '1',
          'model_id': '1',
          'run_id': '1',
          'schema': '123',
          'scope': '1',
          'source_lock_sha256': '1',
          'source_manifest_sha256': '1',
          'training_admitted': '17'}],
 '125': ['object', {'binding_sha256': '1', 'descriptor': '103', 'opaque_id': '1'}],
 '126': ['list', ['125']],
 '127': ['literal', 'bounded-source-screen-asr-decoder-v1'],
 '128': ['object', {'clips': '126', 'order': '5', 'schema': '127', 'scope': '1', 'source_manifest_sha256': '1'}],
 '129': ['object', {'cpu.max': '1', 'memory.max': '1', 'memory.swap.max': '1', 'pids.max': '1'}],
 '130': ['literal', 'screen32-kernel-scope-v1'],
 '131': ['object',
         {'capabilities': '1',
          'limits': '129',
          'network_disabled': '17',
          'network_interfaces': '5',
          'no_new_privileges': '17',
          'private_cgroup_root': '17',
          'read_only_root': '17',
          'schema': '130',
          'seccomp_filter': '17',
          'uid': '0'}],
 '132': ['object',
         {'asset_lock_sha256': '1',
          'model_id': '1',
          'revision': '1',
          'verified_bytes': '0',
          'verified_files': '26'}],
 '133': ['object', {'class': '1', 'implementation': '1', 'name': '1'}],
 '134': ['list', ['133']],
 '135': ['object', {'dtype': '1', 'shape': '27'}],
 '136': ['map', ['135']],
 '137': ['object', {'format': '1'}],
 '138': ['object', {'data_offsets': '27', 'dtype': '1', 'shape': '27'}],
 '139': ['map', ['137', '138']],
 '140': ['literal', 'qwen-safetensors-content-evidence-v1'],
 '141': ['object',
         {'checkpoint_content_sha256': '1',
          'checkpoint_dtype': '1',
          'conversion': '1',
          'data_offsets': '27',
          'dtype': '1',
          'sha256': '1',
          'shape': '27'}],
 '142': ['map', ['141']],
 '143': ['object',
         {'approved_aliases': '8',
          'approved_aliases_sha256': '1',
          'checkpoint_bytes': '0',
          'checkpoint_file_sha256': '1',
          'expected_state_schema': '136',
          'expected_state_schema_sha256': '1',
          'header': '139',
          'header_sha256': '1',
          'schema': '140',
          'tensors': '142'}],
 '144': ['object', {'error_msgs': '8', 'mismatched_keys': '8', 'missing_keys': '8', 'unexpected_keys': '8'}],
 '145': ['object',
         {'_dynamic_tied_weights_keys': '20',
          '_keys_to_ignore_on_load_missing': '20',
          '_keys_to_ignore_on_load_unexpected': '20',
          '_tied_weights_keys': '20',
          'class': '1',
          'module': '1'}],
 '146': ['object',
         {'_dynamic_tied_weights_keys': '20',
          '_keys_to_ignore_on_load_missing': '20',
          '_keys_to_ignore_on_load_unexpected': '20',
          '_tied_weights_keys': '5',
          'class': '1',
          'module': '1'}],
 '147': ['list', ['145', '146']],
 '148': ['literal', 'qwen-complete-load-receipt-v1'],
 '149': ['map', ['114']],
 '150': ['object',
         {'checkpoint_evidence': '143',
          'complete_checkpoint_verification_passed': '17',
          'execution_authorized': '17',
          'loading_info': '144',
          'runtime_ignore_hazards': '147',
          'schema': '148',
          'verified_alias_proofs': '8',
          'verified_loaded_state': '149'}],
 '151': ['object',
         {'funasr': '1',
          'kaldi-native-fbank': '1',
          'numpy': '1',
          'qwen-asr': '1',
          'torch': '1',
          'transformers': '1'}],
 '152': ['object',
         {'assets': '132',
          'device': '1',
          'dtype': '1',
          'effective_attention': '134',
          'load': '150',
          'versions': '151'}],
 '153': ['literal', 'retained-asr-model-lock-v1'],
 '154': ['object',
         {'asset_verification': '1',
          'assets': '88',
          'model_family': '1',
          'model_id': '1',
          'model_revision': '1',
          'model_sha256': '1',
          'original_asset_lock_sha256': '1',
          'provider_id': '1',
          'schema': '153'}],
 '155': ['literal', 'speech-asr-native-config-v2'],
 '156': ['object',
         {'channels': '0', 'max_audio_frames': '0', 'pcm_format': '1', 'preprocessing': '1', 'sample_rate': '0'}],
 '157': ['object',
         {'adapter_id': '1',
          'adapter_version': '0',
          'asset_lock_sha256': '1',
          'conditioning_policy': '1',
          'effective_config': '91',
          'effective_config_sha256': '1',
          'evidence_class': '155',
          'execution': '89',
          'input_policy': '156',
          'model_family': '1',
          'native_request': '90',
          'schema_version': '0',
          'source_lock_sha256': '1'}],
 '158': ['object',
         {'capture_truncated': '17',
          'prompt_token_count': '0',
          'returned_sequence_ids': '27',
          'sequence_shape': '27'}],
 '159': ['object',
         {'decoded_skip_special_tokens': '1',
          'decoded_with_special_tokens': '1',
          'force_language': '20',
          'input_binding': '103',
          'metadata': '104',
          'processor_input_layout': '113',
          'processor_tensor_evidence': '115',
          'prompt_context': '1',
          'token_ids': '27',
          'transcript_repair_applied': '17'}],
 '160': ['list', ['84']],
 '161': ['literal', 'retained-asr-source-lock-v1'],
 '162': ['object',
         {'empty_source_files': '160',
          'executed_head_sha': '1',
          'files': '86',
          'model_id': '1',
          'original_source_lock_sha256': '1',
          'schema': '161',
          'workflow_run_id': '1'}],
 '163': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_forward_calls': '0',
          'human_gold': '17',
          'scope': '1',
          'shipping_approved': '17',
          'status': '1',
          'training_admitted': '17'}],
 '164': ['object', {'index': '0', 'stage': '1', 'started_monotonic': '0'}],
 '165': ['literal', 'qwen12-real-actions-identity-v1'],
 '166': ['object', {'executed_head_sha': '1', 'run_id': '0', 'schema': '165'}],
 '167': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_ids': '8',
          'calls': '0',
          'completed': '0',
          'human_gold': '17',
          'scope': '1',
          'shipping_approved': '17',
          'status': '1',
          'training_admitted': '17'}],
 '168': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_ids': '5',
          'calls': '0',
          'completed': '0',
          'human_gold': '17',
          'scope': '1',
          'shipping_approved': '17',
          'status': '1',
          'training_admitted': '17'}],
 '169': ['object', {'body_retained': '17', 'bytes': '0', 'kind': '1', 'path': '1', 'role': '1', 'sha256': '1'}],
 '170': ['list', ['169', '87']],
 '171': ['object',
         {'ban_emo_unk': '17',
          'data_type': '1',
          'device': '1',
          'fs': '0',
          'language': '1',
          'output_timestamp': '17',
          'text_norm': '1',
          'use_itn': '17'}],
 '172': ['object',
         {'decoding': '1',
          'execution': '89',
          'frontend_config_sha256': '1',
          'lexical_unknown_id': '0',
          'metadata_prefix_tokens': '0',
          'native_request': '171',
          'parser_contract_sha256': '1',
          'rich_postprocess_applied': '17',
          'tokenizer_sha256': '1',
          'vad_used': '17',
          'vocabulary_size': '0'}],
 '173': ['object',
         {'inference_returned': '17',
          'kind': '1',
          'lexical_token_ids': '27',
          'metadata_boundary_verified': '17',
          'metadata_token_ids': '27',
          'raw_redecode_matches_capture': '17',
          'raw_token_ids': '27',
          'token_scope': '1',
          'truncated': '17'}],
 '174': ['object',
         {'acoustic_verbatim_completeness': '1',
          'asset_lock_sha256': '1',
          'effective_config': '172',
          'effective_config_sha256': '1',
          'evidence_class': '92',
          'input': '93',
          'native_config_sha256': '1',
          'peer_transcript_received': '17',
          'schema_version': '0',
          'source_lock_sha256': '1',
          'target_context_received': '17',
          'termination': '173',
          'transcript_postcorrection_applied': '17'}],
 '175': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_frames': '0',
          'audio_id': '1',
          'decoder_completed': '17',
          'executed_head_sha': '1',
          'executed_source_files': '86',
          'file_sha256': '1',
          'loaded_model_assets': '170',
          'model_lock_sha256': '1',
          'native_capture': '174',
          'native_decode_config_sha256': '1',
          'original_job_sha256': '1',
          'pcm_sha256': '1',
          'processed_end_frame': '0',
          'processed_start_frame': '0',
          'raw_native_output_sha256': '1',
          'raw_output_sha256': '1',
          'recognizer_id': '1',
          'recognizer_identity_sha256': '1',
          'recognizer_policy_sha256': '1',
          'remaining_plan_sha256': '1',
          'request_id': '1',
          'run_id': '1',
          'schema': '96',
          'source_id': '1',
          'source_lock_sha256': '1',
          'started_sha256': '1',
          'status': '1'}],
 '176': ['object',
         {'acoustic_verbatim_completeness': '1',
          'inference_returned': '17',
          'input': '93',
          'input_binding_sha256': '1',
          'opaque_id': '1',
          'request': '171',
          'schema': '110',
          'tokenizer_input_ids': '27'}],
 '177': ['object', {'key': '1', 'text': '1'}],
 '178': ['list', ['177']],
 '179': ['object', {'batch_data_time': '0', 'extract_feat': '1', 'load_data': '1'}],
 '180': ['list', ['178', '179']],
 '181': ['literal', 'cosyvoice12-sense-token-binding-v1'],
 '182': ['object',
         {'emo_unknown_id': '0',
          'emo_unknown_is_lexical_unknown': '17',
          'lexical_unknown_id': '0',
          'lexical_unknown_piece': '1',
          'sha256': '1',
          'vocabulary_size': '0'}],
 '183': ['object',
         {'lexical_token_ids': '27',
          'lexical_unknown_present': '17',
          'metadata_boundary_verified': '17',
          'metadata_pieces': '5',
          'metadata_token_ids': '27',
          'parser_contract_sha256': '1',
          'raw_decoded_sha256': '1',
          'raw_redecode_matches_capture': '17',
          'raw_token_ids': '27',
          'schema': '181',
          'token_scope': '1',
          'tokenizer': '182'}],
 '184': ['object', {'decoded': '1', 'token_ids': '27', 'token_scope': '1'}],
 '185': ['object',
         {'completeness_scope': '1',
          'input_binding': '103',
          'metadata': '5',
          'native_capture': '174',
          'native_v2_observation': '176',
          'native_v2_validated': '17',
          'raw_result': '180',
          'rich_postprocess_applied': '17',
          'token_binding': '183',
          'token_evidence': '184',
          'use_itn': '17'}],
 '186': ['object', {'version': '0'}],
 '187': ['map', ['186']],
 '188': ['object', {'encoder.rwkv_encoders': '186', 'frontend': '186'}],
 '189': ['object', {'funasr': '1', 'torch': '1'}],
 '190': ['object',
         {'all_checkpoint_metadata_preserved': '17',
          'architecture': '187',
          'architecture_sha256': '1',
          'checkpoint': '187',
          'checkpoint_only_module_records': '188',
          'checkpoint_sha256': '1',
          'module_version_migration_allowed': '17',
          'qualified_runtime_versions': '189',
          'recipe': '1'}],
 '191': ['literal', 'sensevoice-exact-checkpoint-contract-v1'],
 '192': ['object',
         {'checkpoint_state': '149',
          'checkpoint_state_sha256': '1',
          'expected_schema': '136',
          'expected_schema_sha256': '1',
          'identity_key_mapping_only': '17',
          'metadata': '190',
          'schema': '191'}],
 '193': ['object', {'missing_keys': '8', 'unexpected_keys': '8'}],
 '194': ['literal', 'sensevoice-exact-apply-receipt-v1'],
 '195': ['object',
         {'complete_checkpoint_verification_passed': '17',
          'contract': '192',
          'execution_authorized': '17',
          'loading_info': '193',
          'runtime_tensor_checks_delegated_to_trusted_callback': '17',
          'schema': '194',
          'verified_loaded_state': '149',
          'verified_loaded_state_sha256': '1'}],
 '196': ['object',
         {'assets': '132',
          'device': '1',
          'dither': '0',
          'dither_rng_reproducibility': '1',
          'dtype': '1',
          'frontend_backend': '1',
          'load': '195',
          'no_vad': '17',
          'tokenizer_identity': '182',
          'use_itn': '17',
          'versions': '151'}],
 '197': ['object',
         {'asset_verification': '1',
          'assets': '170',
          'model_family': '1',
          'model_id': '1',
          'model_revision': '1',
          'model_sha256': '1',
          'original_asset_lock_sha256': '1',
          'provider_id': '1',
          'schema': '153'}],
 '198': ['object',
         {'adapter_id': '1',
          'adapter_version': '0',
          'asset_lock_sha256': '1',
          'conditioning_policy': '1',
          'effective_config': '172',
          'effective_config_sha256': '1',
          'evidence_class': '155',
          'execution': '89',
          'input_policy': '156',
          'model_family': '1',
          'native_request': '171',
          'schema_version': '0',
          'source_lock_sha256': '1'}],
 '199': ['object',
         {'dither': '0',
          'filter_length_max': '0',
          'filter_length_min': '0',
          'frame_length': '0',
          'frame_shift': '0',
          'fs': '0',
          'lfr_m': '0',
          'lfr_n': '0',
          'n_mels': '0',
          'snip_edges': '17',
          'upsacle_samples': '17',
          'window': '1'}],
 '200': ['object', {'attributes': '199', 'class': '1', 'cmvn_sha256': '1', 'loaded_cmvn_tensor': '114'}],
 '201': ['object', {'raw_result': '180', 'representation': '1'}],
 '202': ['object',
         {'completeness_scope': '1',
          'input_binding': '103',
          'metadata': '5',
          'raw_result': '180',
          'rich_postprocess_applied': '17',
          'token_binding': '183',
          'token_evidence': '184',
          'use_itn': '17'}],
 '203': ['object', {'raw_token_ids': '27', 'token_scope': '1'}],
 '204': ['literal', 'qwen12-supervision-v1'],
 '205': ['object', {'cleanup': '0', 'create': '0', 'inspect': '0', 'start': '0', 'validate': '0', 'wait': '0'}],
 '206': ['object',
         {'cleanup_error': '20',
          'family': '1',
          'gate_result': '63',
          'mode': '20',
          'name': '1',
          'phase_wall_seconds': '205',
          'primary_error': '20',
          'status': '1',
          'wall_seconds': '0'}],
 '207': ['object',
         {'cleanup_error': '20',
          'family': '1',
          'gate_result': '63',
          'mode': '1',
          'name': '1',
          'phase_wall_seconds': '205',
          'primary_error': '20',
          'status': '1',
          'wall_seconds': '0'}],
 '208': ['list', ['206', '207']],
 '209': ['object', {'schema': '204', 'stages': '208'}],
 '210': ['object',
         {'small.pt': '3',
          'whisper/assets/gpt2.tiktoken': '3',
          'whisper/assets/mel_filters.npz': '3',
          'whisper/assets/multilingual.tiktoken': '3'}],
 '211': ['literal', 'whisper-screen-loaded-assets-lock-v1'],
 '212': ['object', {'files': '210', 'model_id': '1', 'schema': '211'}],
 '213': ['object', {'container': '1', 'deadline_overrun': '17', 'status': '1'}],
 '214': ['list', ['213']],
 '215': ['literal', 'bounded-host-cleanup-v1'],
 '216': ['object', {'containers': '214', 'schema': '215', 'status': '1'}],
 '217': ['object',
         {'allowance': '0',
          'asset_id': '1',
          'budget_charge_bytes': '0',
          'charge_kind': '1',
          'kind': '1',
          'observed_ciphertext_bytes': '0',
          'status': '1'}],
 '218': ['object', {'allowance': '0', 'asset_id': '1', 'kind': '1', 'status': '1'}],
 '219': ['literal', 'bounded-host-budget-v1'],
 '220': ['object', {'cap': '0', 'schema': '219'}],
 '221': ['object', {'cpus': '0', 'memory': '0', 'network': '1', 'pids': '0'}],
 '222': ['literal', 'bounded-container-inspect-v1'],
 '223': ['object',
         {'argv_sha256': '1',
          'limits': '221',
          'role': '1',
          'schema': '222',
          'socket_mounted': '17',
          'status': '1'}],
 '224': ['literal', 'bounded-kernel-scope-v1'],
 '225': ['object',
         {'CapAmb': '1',
          'CapBnd': '1',
          'CapEff': '1',
          'CapInh': '1',
          'CapPrm': '1',
          'NSpid': '1',
          'NoNewPrivs': '1',
          'Seccomp': '1'}],
 '226': ['object',
         {'cgroup': '1',
          'interfaces': '5',
          'limits': '129',
          'private_readonly_cgroup': '17',
          'readonly_root': '17',
          'role': '1',
          'schema': '224',
          'status': '225',
          'uid': '0'}],
 '227': ['object',
         {'complete': '17',
          'connections_started': '0',
          'downstream_read_bytes': '0',
          'total_read_bytes': '0',
          'transport_closed': '0',
          'upstream_read_bytes': '0'}],
 '228': ['object', {'model': '227'}],
 '229': ['literal', 'bounded-connect-ledger-v1'],
 '230': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '228',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '231': ['list', ['217']],
 '232': ['literal', 'bounded-host-acquisition-v1'],
 '233': ['object',
         {'aggregate_cap': '0',
          'assets': '231',
          'byte_scope': '1',
          'schema': '232',
          'status': '1',
          'total_read_bytes': '0'}],
 '234': ['object', {'source': '227'}],
 '235': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '234',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '236': ['object', {'wheel-01': '227'}],
 '237': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '236',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '238': ['object', {'wheel-02': '227'}],
 '239': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '238',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '240': ['object', {'wheel-03': '227'}],
 '241': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '240',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '242': ['object', {'wheel-04': '227'}],
 '243': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '242',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '244': ['object', {'wheel-05': '227'}],
 '245': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '244',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '246': ['object', {'wheel-06': '227'}],
 '247': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '246',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '248': ['object', {'wheel-07': '227'}],
 '249': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '248',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '250': ['object', {'wheel-08': '227'}],
 '251': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '250',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '252': ['object', {'wheel-09': '227'}],
 '253': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '252',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '254': ['object', {'wheel-10': '227'}],
 '255': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '254',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '256': ['object', {'wheel-11': '227'}],
 '257': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '256',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '258': ['object', {'wheel-12': '227'}],
 '259': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '258',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '260': ['object', {'wheel-13': '227'}],
 '261': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '260',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '262': ['object', {'wheel-14': '227'}],
 '263': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '262',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '264': ['object', {'wheel-15': '227'}],
 '265': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '264',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '266': ['object', {'wheel-16': '227'}],
 '267': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '266',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '268': ['object', {'wheel-17': '227'}],
 '269': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '268',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '270': ['object', {'wheel-18': '227'}],
 '271': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '270',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '272': ['object', {'wheel-19': '227'}],
 '273': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '272',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '274': ['object', {'wheel-20': '227'}],
 '275': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '274',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '276': ['object', {'wheel-21': '227'}],
 '277': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '276',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '278': ['object', {'wheel-22': '227'}],
 '279': ['object',
         {'aggregate_cap': '0',
          'asset_cap': '0',
          'asset_id': '1',
          'assets': '278',
          'byte_scope': '1',
          'charged_bytes': '0',
          'expected_size': '0',
          'failure_code': '20',
          'kind': '1',
          'plan_complete': '17',
          'prior_charged_bytes': '0',
          'schema': '229',
          'status': '1',
          'total_read_bytes': '0'}],
 '280': ['literal', 'bounded-pip-worker-result-v2'],
 '281': ['object',
         {'asset_index': '0',
          'filename': '1',
          'help_exit_code': '0',
          'missing_options': '5',
          'pip_version': '1',
          'reason': '20',
          'resume_policy': '1',
          'reviewed_source_matches': '17',
          'schema': '280',
          'sha256': '1',
          'size_bytes': '0',
          'stage': '1',
          'status': '1'}],
 '282': ['object',
         {'budget_charge_bytes': '0', 'index': '0', 'kind': '1', 'observed_ciphertext_bytes': '0', 'status': '1'}],
 '283': ['list', ['282']],
 '284': ['object',
         {'aggregate_cap': '0',
          'assets': '283',
          'byte_scope': '1',
          'excludes': '1',
          'retry_count': '0',
          'schema': '232',
          'status': '1',
          'total_read_bytes': '0'}],
 '285': ['object',
         {'model/small.pt': '3',
          'source/LICENSE': '3',
          'source/README.md': '3',
          'source/pyproject.toml': '3',
          'source/requirements.txt': '3',
          'source/whisper/__init__.py': '3',
          'source/whisper/__main__.py': '3',
          'source/whisper/assets/gpt2.tiktoken': '3',
          'source/whisper/assets/mel_filters.npz': '3',
          'source/whisper/assets/multilingual.tiktoken': '3',
          'source/whisper/audio.py': '3',
          'source/whisper/decoding.py': '3',
          'source/whisper/model.py': '3',
          'source/whisper/normalizers/__init__.py': '3',
          'source/whisper/normalizers/basic.py': '3',
          'source/whisper/normalizers/english.json': '3',
          'source/whisper/normalizers/english.py': '3',
          'source/whisper/timing.py': '3',
          'source/whisper/tokenizer.py': '3',
          'source/whisper/transcribe.py': '3',
          'source/whisper/triton_ops.py': '3',
          'source/whisper/utils.py': '3',
          'source/whisper/version.py': '3',
          'wheels/certifi-2026.7.22-py3-none-any.whl': '3',
          'wheels/charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': '3',
          'wheels/filelock-4.0.8-py3-none-any.whl': '3',
          'wheels/fsspec-2026.9.0-py3-none-any.whl': '3',
          'wheels/idna-3.20-py3-none-any.whl': '3',
          'wheels/jinja2-3.1.6-py3-none-any.whl': '3',
          'wheels/llvmlite-0.50.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': '3',
          'wheels/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': '3',
          'wheels/more_itertools-10.8.0-py3-none-any.whl': '3',
          'wheels/mpmath-1.3.0-py3-none-any.whl': '3',
          'wheels/networkx-3.7-py3-none-any.whl': '3',
          'wheels/numba-0.68.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl': '3',
          'wheels/numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl': '3',
          'wheels/regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': '3',
          'wheels/requests-2.34.2-py3-none-any.whl': '3',
          'wheels/setuptools-81.0.0-py3-none-any.whl': '3',
          'wheels/sympy-1.14.0-py3-none-any.whl': '3',
          'wheels/tiktoken-0.14.0-cp312-cp312-manylinux_2_28_x86_64.whl': '3',
          'wheels/torch-2.12.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl': '3',
          'wheels/tqdm-4.70.1-py3-none-any.whl': '3',
          'wheels/typing_extensions-4.16.0-py3-none-any.whl': '3',
          'wheels/urllib3-2.8.0-py3-none-any.whl': '3'}],
 '286': ['literal', 'whisper-screen-download-bounded-tls-v1'],
 '287': ['object',
         {'acoustic_verbatim_completeness': '1',
          'acquisition': '284',
          'files': '285',
          'human_gold': '17',
          'model_loaded': '17',
          'retry_count': '0',
          'schema': '286',
          'shipping_approved': '17',
          'source_role': '1',
          'training_admitted': '17'}],
 '288': ['map', ['3']],
 '289': ['literal', 'whisper-screen-installed-runtime-v1'],
 '290': ['object', {'files': '288', 'schema': '289'}],
 '291': ['object', {'runtime-lock.json': '1', 'source-lock.json': '1'}],
 '292': ['literal', 'whisper-screen-phase-complete-v1'],
 '293': ['object',
         {'acoustic_verbatim_completeness': '1',
          'human_gold': '17',
          'lock_sha256': '291',
          'model_calls': '0',
          'phase': '1',
          'schema': '292',
          'selected_source_model_body_bytes': '0',
          'shipping_approved': '17',
          'source_builds': '0',
          'source_role': '1',
          'status': '1',
          'training_admitted': '17'}],
 '294': ['literal', 'whisper-screen-phased-progress-v1'],
 '295': ['object',
         {'acoustic_verbatim_completeness': '1',
          'asset': '20',
          'assets': '8',
          'byte_scope': '1',
          'completed_stages': '8',
          'human_gold': '17',
          'lock_sha256': '291',
          'model_calls': '0',
          'network_transfer_total_known': '17',
          'phase': '1',
          'response_body_bytes_observed': '0',
          'retry_count': '0',
          'schema': '294',
          'shipping_approved': '17',
          'source_role': '1',
          'stage': '1',
          'status': '1',
          'training_admitted': '17',
          'verified_assets': '0'}],
 '296': ['object',
         {'acoustic_verbatim_completeness': '1',
          'asset': '20',
          'assets': '8',
          'byte_scope': '1',
          'completed_stages': '5',
          'human_gold': '17',
          'lock_sha256': '291',
          'model_calls': '0',
          'network_transfer_total_known': '17',
          'phase': '1',
          'response_body_bytes_observed': '0',
          'retry_count': '0',
          'schema': '294',
          'shipping_approved': '17',
          'source_role': '1',
          'stage': '1',
          'status': '1',
          'training_admitted': '17',
          'verified_assets': '0'}],
 '297': ['object',
         {'expected_bytes': '0',
          'expected_sha256': '1',
          'group': '1',
          'http_status': '0',
          'index': '0',
          'observed_sha256': '1',
          'response_body_bytes_observed': '0',
          'status': '1'}],
 '298': ['list', ['297']],
 '299': ['object',
         {'acoustic_verbatim_completeness': '1',
          'asset': '297',
          'assets': '298',
          'byte_scope': '1',
          'completed_stages': '5',
          'human_gold': '17',
          'lock_sha256': '291',
          'model_calls': '0',
          'network_transfer_total_known': '17',
          'phase': '1',
          'response_body_bytes_observed': '0',
          'retry_count': '0',
          'schema': '294',
          'shipping_approved': '17',
          'source_role': '1',
          'stage': '1',
          'status': '1',
          'training_admitted': '17',
          'verified_assets': '0'}],
 '300': ['object',
         {'filename': '1',
          'name': '1',
          'requires_dist': '8',
          'sha256': '1',
          'size_bytes': '0',
          'url': '1',
          'version': '1'}],
 '301': ['object',
         {'filename': '1',
          'name': '1',
          'requires_dist': '5',
          'sha256': '1',
          'size_bytes': '0',
          'url': '1',
          'version': '1'}],
 '302': ['list', ['300', '301']],
 '303': ['object', {'retained_path': '1', 'retained_sha256': '1'}],
 '304': ['literal', 'whisper-screen-runtime-lock-v1'],
 '305': ['object',
         {'metadata_observed': '17', 'sha256': '1', 'url': '1', 'wheel_body_downloaded_during_preparation': '17'}],
 '306': ['object',
         {'cpu_source_import_policy': '1',
          'files': '302',
          'provenance': '303',
          'schema': '304',
          'supplemental_metadata': '305',
          'target': '1'}],
 '307': ['literal', 'whisper-screen-setup-v1'],
 '308': ['object',
         {'acoustic_verbatim_completeness': '1',
          'acquisition': '284',
          'asset_lock_sha256': '1',
          'human_gold': '17',
          'installed_inventory_sha256': '1',
          'kernel_scope': '131',
          'model_calls': '0',
          'runtime_lock_sha256': '1',
          'schema': '307',
          'shipping_approved': '17',
          'source_builds': '0',
          'source_lock_sha256': '1',
          'source_role': '1',
          'status': '1',
          'training_admitted': '17'}],
 '309': ['object', {'path': '1', 'sha256': '1', 'size_bytes': '0', 'url': '1'}],
 '310': ['list', ['309']],
 '311': ['literal', 'whisper-screen-official-source-lock-v1'],
 '312': ['object',
         {'files': '310',
          'local_preparation_scope': '1',
          'schema': '311',
          'source_bytes_verified': '17',
          'source_license': '1',
          'source_repository': '1',
          'source_revision': '1'}],
 '313': ['object', {'bytes': '0', 'path': '1', 'role': '1', 'sha256': '1', 'source_revision': '1'}],
 '314': ['list', ['84', '313']],
 '315': ['object',
         {'language': '0',
          'no_speech': '0',
          'no_timestamps': '0',
          'sot': '0',
          'sot_lm': '0',
          'sot_prev': '0',
          'transcribe': '0',
          'translate': '0'}],
 '316': ['object',
         {'beam_size': '20',
          'best_of': '20',
          'fp16': '17',
          'language': '1',
          'length_penalty': '20',
          'max_initial_timestamp': '20',
          'patience': '20',
          'prefix': '20',
          'prompt': '20',
          'sample_len': '20',
          'suppress_blank': '17',
          'suppress_tokens': '1',
          'task': '1',
          'temperature': '0',
          'without_timestamps': '17'}],
 '317': ['object',
         {'control_token_ids': '315',
          'decoder': '1',
          'effective_sample_len': '0',
          'eot_token_id': '0',
          'execution': '89',
          'initial_blank_suppress_token_ids': '27',
          'initial_token_ids': '27',
          'n_text_ctx': '0',
          'native_request': '316',
          'non_speech_token_ids': '27',
          'resolved_suppress_token_ids': '27',
          'space_token_ids': '27',
          'tokenizer_sha256': '1'}],
 '318': ['object',
         {'capture_stage': '1',
          'completed': '17',
          'generated_token_ids': '27',
          'initial_token_ids': '27',
          'kind': '1',
          'pre_finalize_token_count': '0',
          'step_count': '0',
          'stop_reason': '1',
          'truncated': '17'}],
 '319': ['object',
         {'acoustic_verbatim_completeness': '1',
          'asset_lock_sha256': '1',
          'effective_config': '317',
          'effective_config_sha256': '1',
          'evidence_class': '92',
          'input': '93',
          'native_config_sha256': '1',
          'peer_transcript_received': '17',
          'schema_version': '0',
          'source_lock_sha256': '1',
          'target_context_received': '17',
          'termination': '318',
          'transcript_postcorrection_applied': '17'}],
 '320': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_frames': '0',
          'audio_id': '1',
          'decoder_completed': '17',
          'executed_head_sha': '1',
          'executed_source_files': '314',
          'file_sha256': '1',
          'loaded_model_assets': '88',
          'model_lock_sha256': '1',
          'native_capture': '319',
          'native_decode_config_sha256': '1',
          'original_job_sha256': '1',
          'pcm_sha256': '1',
          'processed_end_frame': '0',
          'processed_start_frame': '0',
          'raw_native_output_sha256': '1',
          'raw_output_sha256': '1',
          'recognizer_id': '1',
          'recognizer_identity_sha256': '1',
          'recognizer_policy_sha256': '1',
          'remaining_plan_sha256': '1',
          'request_id': '1',
          'run_id': '1',
          'schema': '96',
          'source_id': '1',
          'source_lock_sha256': '1',
          'started_sha256': '1',
          'status': '1'}],
 '321': ['object',
         {'float32_padded_sha256': '1',
          'float32_source_sha256': '1',
          'frontend_source_sha256': '1',
          'mel_filters_sha256': '1',
          'mel_float32_sha256': '1',
          'mel_shape': '27',
          'native_input': '93',
          'pcm16_sha256': '1'}],
 '322': ['object', {'last_update_completed': '17', 'token_ids': '27', 'trace_count': '0'}],
 '323': ['object', {'device': '1', 'dtype': '1', 'float32_sha256': '1', 'shape': '27'}],
 '324': ['literal', 'whisper-screen-native-result-projection-v1'],
 '325': ['object',
         {'acoustic_verbatim_completeness': '1',
          'audio_features': '323',
          'avg_logprob': '0',
          'compression_ratio': '0',
          'decoder_completion_validated': '17',
          'full_input_processed': '17',
          'language': '1',
          'language_probs': '20',
          'native_capture': '319',
          'no_speech_prob': '0',
          'process_peak_rss_kib': '0',
          'projection_scope': '1',
          'raw_text': '1',
          'returned_token_ids': '27',
          'schema': '324',
          'temperature': '0',
          'trace_files': '0',
          'wall_seconds': '0'}],
 '326': ['object',
         {'acoustic_verbatim_completeness': '1',
          'actual_text': '1',
          'native_output': '118',
          'native_row': '20',
          'output_scope': '1',
          'request_id': '1',
          'schema': '119',
          'truncated': '17'}],
 '327': ['object', {'completed': '17', 'step': '0', 'token_ids': '27'}],
 '328': ['object',
         {'n_audio_ctx': '0',
          'n_audio_head': '0',
          'n_audio_layer': '0',
          'n_audio_state': '0',
          'n_mels': '0',
          'n_text_ctx': '0',
          'n_text_head': '0',
          'n_text_layer': '0',
          'n_text_state': '0',
          'n_vocab': '0'}],
 '329': ['object',
         {'asset_lock_sha256': '1',
          'audio_calls': '0',
          'config_sha256': '1',
          'installed_inventory_sha256': '1',
          'model_dimensions': '328',
          'model_parameters': '0',
          'model_training': '17',
          'numpy_version': '1',
          'source_lock_sha256': '1',
          'status': '1',
          'torch_interop_threads': '0',
          'torch_threads': '0',
          'torch_version': '1'}],
 '330': ['object',
         {'adapter_id': '1',
          'adapter_version': '0',
          'asset_lock_sha256': '1',
          'conditioning_policy': '1',
          'effective_config': '317',
          'effective_config_sha256': '1',
          'evidence_class': '155',
          'execution': '89',
          'input_policy': '156',
          'model_family': '1',
          'native_request': '316',
          'schema_version': '0',
          'source_lock_sha256': '1'}],
 '331': ['object',
         {'empty_source_files': '160',
          'executed_head_sha': '1',
          'files': '314',
          'model_id': '1',
          'official_source_revision': '1',
          'original_source_lock_canonical_sha256': '1',
          'schema': '161',
          'workflow_run_id': '1'}]}
_RETAINED_MEMBERS = {'asr-setup-public.json': ['61'],
 'cleanup-receipt.json': ['69'],
 'cpu-receipt.json': ['74'],
 'image-transfer.json': ['79'],
 'qwen06/blind-job.json': ['83'],
 'qwen06/clip/completed.json': ['97'],
 'qwen06/clip/raw-native-output.json': ['116'],
 'qwen06/clip/raw-native-row.json': ['117'],
 'qwen06/clip/raw-transcript.json': ['120'],
 'qwen06/clip/started.json': ['122'],
 'qwen06/contract.json': ['124'],
 'qwen06/cpu-process-guard.json': ['39'],
 'qwen06/decoder-inputs.json': ['128'],
 'qwen06/kernel-scope.json': ['131'],
 'qwen06/model-load.json': ['152'],
 'qwen06/native-asset-lock.json': ['154'],
 'qwen06/native-config.json': ['157'],
 'qwen06/native-generation-return.json': ['158'],
 'qwen06/native-observation.json': ['111'],
 'qwen06/native-resolved-generation-config.json': ['105'],
 'qwen06/native-returned-raw.json': ['159'],
 'qwen06/native-source-lock.json': ['162'],
 'qwen06/prepared.json': ['163'],
 'qwen06/progress.json': ['164'],
 'qwen06/run-identity.json': ['166'],
 'qwen06/worker-complete.json': ['167', '168'],
 'sensevoice/blind-job.json': ['83'],
 'sensevoice/clip/completed.json': ['175'],
 'sensevoice/clip/raw-native-output.json': ['185'],
 'sensevoice/clip/raw-native-row.json': ['117'],
 'sensevoice/clip/raw-transcript.json': ['120'],
 'sensevoice/clip/started.json': ['122'],
 'sensevoice/contract.json': ['124'],
 'sensevoice/cpu-process-guard.json': ['39'],
 'sensevoice/decoder-inputs.json': ['128'],
 'sensevoice/kernel-scope.json': ['131'],
 'sensevoice/model-load.json': ['196'],
 'sensevoice/native-asset-lock.json': ['197'],
 'sensevoice/native-config.json': ['198'],
 'sensevoice/native-frontend-config.json': ['200'],
 'sensevoice/native-inference-return.json': ['201'],
 'sensevoice/native-observation.json': ['176'],
 'sensevoice/native-returned-raw.json': ['202'],
 'sensevoice/native-source-lock.json': ['162'],
 'sensevoice/native-tokenizer-input.json': ['203'],
 'sensevoice/optional-tool-probes.json': ['36'],
 'sensevoice/prepared.json': ['163'],
 'sensevoice/progress.json': ['164'],
 'sensevoice/run-identity.json': ['166'],
 'sensevoice/worker-complete.json': ['167', '168'],
 'supervision-receipt.json': ['209'],
 'whisper-setup/asset-lock.json': ['212'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-000.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-001.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-002.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-003.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-004.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-005.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-006.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-007.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-008.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-009.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-010.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-011.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-012.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-013.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-014.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-015.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-016.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-017.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-018.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-019.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-020.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-021.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-022.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-023.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-024.json': ['216'],
 'whisper-setup/bounded-acquisition/bounded-host-cleanup-025.json': ['216'],
 'whisper-setup/bounded-acquisition/budget-000-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-000-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-001-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-001-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-002-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-002-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-003-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-003-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-004-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-004-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-005-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-005-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-006-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-006-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-007-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-007-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-008-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-008-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-009-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-009-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-010-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-010-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-011-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-011-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-012-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-012-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-013-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-013-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-014-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-014-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-015-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-015-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-016-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-016-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-017-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-017-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-018-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-018-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-019-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-019-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-020-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-020-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-021-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-021-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-022-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-022-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-023-final.json': ['217'],
 'whisper-setup/bounded-acquisition/budget-023-started.json': ['218'],
 'whisper-setup/bounded-acquisition/budget-initial.json': ['220'],
 'whisper-setup/bounded-acquisition/finalize-offline-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/install-offline-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/model-broker-kernel.json': ['226'],
 'whisper-setup/bounded-acquisition/model-broker-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/model-broker.json': ['230'],
 'whisper-setup/bounded-acquisition/model-worker-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/receipt.json': ['233'],
 'whisper-setup/bounded-acquisition/source-broker-kernel.json': ['226'],
 'whisper-setup/bounded-acquisition/source-broker-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/source-broker.json': ['235'],
 'whisper-setup/bounded-acquisition/source-worker-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/wheel-NN-broker-kernel.json': ['226'],
 'whisper-setup/bounded-acquisition/wheel-NN-broker-scope.json': ['223'],
 'whisper-setup/bounded-acquisition/wheel-NN-broker.json': ['237',
                                                            '239',
                                                            '241',
                                                            '243',
                                                            '245',
                                                            '247',
                                                            '249',
                                                            '251',
                                                            '253',
                                                            '255',
                                                            '257',
                                                            '259',
                                                            '261',
                                                            '263',
                                                            '265',
                                                            '267',
                                                            '269',
                                                            '271',
                                                            '273',
                                                            '275',
                                                            '277',
                                                            '279'],
 'whisper-setup/bounded-acquisition/wheel-NN-worker-kernel.json': ['226'],
 'whisper-setup/bounded-acquisition/wheel-NN-worker-result.json': ['281'],
 'whisper-setup/bounded-acquisition/wheel-NN-worker-scope.json': ['223'],
 'whisper-setup/download-receipt.json': ['287'],
 'whisper-setup/installed-inventory.json': ['290'],
 'whisper-setup/phase-finalize-complete.json': ['293'],
 'whisper-setup/phase-finalize-progress.json': ['295'],
 'whisper-setup/phase-install-complete.json': ['293'],
 'whisper-setup/phase-install-progress.json': ['296'],
 'whisper-setup/phase-model-complete.json': ['293'],
 'whisper-setup/phase-model-progress.json': ['299'],
 'whisper-setup/phase-source-complete.json': ['293'],
 'whisper-setup/phase-source-progress.json': ['299'],
 'whisper-setup/runtime-lock.json': ['306'],
 'whisper-setup/setup-receipt.json': ['308'],
 'whisper-setup/source-lock.json': ['312'],
 'whisper/blind-job.json': ['83'],
 'whisper/clip/completed.json': ['320'],
 'whisper/clip/input-capture.json': ['321'],
 'whisper/clip/pre-finalize.json': ['322'],
 'whisper/clip/raw-native-output.json': ['325'],
 'whisper/clip/raw-output.json': ['325'],
 'whisper/clip/raw-transcript.json': ['326'],
 'whisper/clip/started.json': ['122'],
 'whisper/clip/update.json': ['327'],
 'whisper/cpu-process-guard.json': ['39'],
 'whisper/kernel-scope.json': ['131'],
 'whisper/load-receipt.json': ['329'],
 'whisper/native-asset-lock.json': ['154'],
 'whisper/native-config.json': ['330'],
 'whisper/native-source-lock.json': ['331'],
 'whisper/prepared.json': ['163'],
 'whisper/progress.json': ['164'],
 'whisper/run-identity.json': ['166'],
 'whisper/worker-complete.json': ['167', '168']}


def _matches_shape(value, reference, *, subset=False):
    rule = _RETAINED_SHAPES[reference]
    kind = rule[0]
    if kind == "literal": return type(value) is type(rule[1]) and value == rule[1]
    if kind == "str": return type(value) is str
    if kind == "number": return type(value) in (int, float)
    if kind == "bool": return type(value) is bool
    if kind == "null": return value is None
    if kind == "scalar": return value is None or type(value) in (str, int, float, bool)
    if kind == "list": return type(value) is list and all(any(_matches_shape(item, ref) for ref in rule[1]) for item in value)
    if kind == "map":
        return type(value) is dict and all(type(k) is str and len(k) <= 240 and not k.startswith("/") and ".." not in k.split("/") and re.fullmatch(r"[A-Za-z0-9_./:@+() -]*", k) and any(_matches_shape(v, ref) for ref in rule[1]) for k, v in value.items())
    if kind == "object":
        fields = rule[1]
        if type(value) is not dict or (not set(value) <= set(fields) if subset else set(value) != set(fields)): return False
        return all(_matches_shape(item, fields[key]) for key, item in value.items())
    raise ValueError("unknown fixed schema rule")


def _schema_key(name):
    if name.startswith(("output/", "frozen/")):
        _, family, leaf = name.split("/", 2)
        if leaf.startswith("clip-"):
            leaf = "clip/" + leaf.split("/", 1)[1] if "/" in leaf else leaf[12:]
        leaf = re.sub(r"update-[0-9]{3}\.json", "update.json", leaf)
        return family + "/" + leaf
    return re.sub(r"wheel-[0-9]{2}-", "wheel-NN-", name)


def _fields(value, fields, label):
    require(type(value) is dict and set(value) == set(fields), label)


START_FIELDS = frozenset(("schema", "source_id", "file_sha256", "pcm_sha256", "audio_frames", "audio_id",
    "job_sha256", "execution_plan_sha256", "worker_input_binding_sha256", "screening_receipt_sha256",
    "tts_authority_sha256", "request_id", "recognizer_id", "recognizer_identity_sha256", "recognizer_policy_sha256",
    "model_lock_sha256", "source_lock_sha256", "native_decode_config_sha256", "run_id", "run_attempt", "executed_head_sha"))
COMPLETION_FIELDS = START_FIELDS | {"started_sha256", "raw_output_sha256", "raw_native_output_sha256", "status",
    "decoder_completed", "processed_start_frame", "processed_end_frame", "loaded_model_assets", "executed_source_files",
    "native_capture", "acoustic_verbatim_completeness"}


def _started(value, family, audio_id):
    _fields(value, START_FIELDS, "exact new ASR18 start schema")
    require(value["schema"] == "retained-asr-started-v1" and value["recognizer_id"] == family and value["audio_id"] == audio_id,
        "same fixed native slot")
    for k, v in value.items():
        if k.endswith("sha256"): require(type(v) is str and SHA.fullmatch(v), "native slot digest")
    require(type(value["audio_frames"]) is int and 0 < value["audio_frames"] < 192000 and
        type(value["run_attempt"]) is int and value["run_attempt"] == 1, "new bounded audio and attempt one")
    require(type(value["run_id"]) is str and re.fullmatch(r"[1-9][0-9]{0,19}", value["run_id"]), "slot run ID")
    require(type(value["executed_head_sha"]) is str and HEAD.fullmatch(value["executed_head_sha"]), "slot executed source")
    require(value["source_id"] == "k1asr18-" + value["file_sha256"] and value["request_id"] ==
        "k1asr18-" + value["run_id"] + "-" + family + "-" + audio_id, "new unique request/source identity")


def validate_member_json(name, value):
    """Fixed native/retained schemas, not suffix-based JSON approval.

    Host/source/control schemas and cross-record 18-slot reconciliation are
    additionally required by hosted_run.validate_public_evidence.
    """
    path(name); _public_metadata(value)
    if _special_retained(name, value): return
    key = _schema_key(name)
    parts = name.split("/")
    family = parts[1] if name.startswith(("output/", "frozen/")) else None
    leaf = parts[-1]
    if name == "supervision-receipt.json":
        _fields(value, ("schema", "stages"), "exact ASR18 supervision schema")
        require(value["schema"] == "k1-asr18-supervision-v1" and type(value["stages"]) is list and len(value["stages"]) <= 32, "fixed bounded supervision")
        for row in value["stages"]:
            _fields(row, ("name", "family", "mode", "status", "worker_exit_success", "primary_error", "cleanup_error", "wall_seconds", "gate_result"), "exact supervision stage")
            require(type(row["name"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["name"]), "fixed stage name")
            require(row["family"] in (*c.FAMILIES, "setup") and row["mode"] in ("prepare", "decode", "setup", None) and
                row["status"] in ("started", "complete", "failed", "cleanup_failed", "failed_no_retry") and
                type(row["worker_exit_success"]) is bool and type(row["wall_seconds"]) in (int, float) and
                0 <= row["wall_seconds"] <= 86400, "bounded observed supervision")
            for key in ("primary_error", "cleanup_error"): _host()._safe_failure(row[key])
            _gate_shape(row["gate_result"])
        return
    if family and leaf == "run-identity.json":
        raise ValueError("old run identity receipt is not emitted by ASR18")
    if family and leaf in ("prepared.json", "worker-complete.json"):
        _fields(value, ("schema", "family", "mode", "status", "audio_forward_calls", "completed", "audio_ids",
            "job_sha256", "execution_plan_sha256"), "exact new worker result")
        require(value["schema"] == "k1-asr18-worker-result-v1" and value["family"] == family and
            value["mode"] in ("prepare", "decode"), "new fixed worker result")
        for k in ("audio_forward_calls", "completed"):
            require(type(value[k]) is int and 0 <= value[k] <= 6, "native count at most six")
        require(type(value["audio_ids"]) is list and value["audio_forward_calls"] == value["completed"] == len(value["audio_ids"])
            and len(set(value["audio_ids"])) == len(value["audio_ids"]) and
            all(type(x) is str and re.fullmatch(r"clip-00000[1-6]", x) for x in value["audio_ids"]), "unique complete opaque native roster")
        if value["mode"] == "prepare":
            require(value["status"] == "prepared_no_recognition" and value["audio_forward_calls"] == 0, "prepare zero forwards")
        else: require(value["status"] == "all_planned_native_complete", "decode terminal status")
        for k in ("job_sha256", "execution_plan_sha256"): require(type(value[k]) is str and SHA.fullmatch(value[k]), "result input binding")
        return
    if family and leaf == "execution-plan.json":
        _fields(value, ("schema", "execution_enabled", "job_sha256", "family_order", "audio_order"), "opaque plan schema")
        require(value["schema"] == "k1-asr18-anonymous-order-metadata-proposal-v1" and value["execution_enabled"] is False
            and value["family_order"] == list(c.FAMILIES), "fixed opaque order schema")
        require(type(value["audio_order"]) is list and 1 <= len(value["audio_order"]) <= 6 and
            len(set(value["audio_order"])) == len(value["audio_order"]) and
            all(type(x) is str and re.fullmatch(r"clip-00000[1-6]", x) for x in value["audio_order"]), "anonymous order only")
        require(type(value["job_sha256"]) is str and SHA.fullmatch(value["job_sha256"]), "bound opaque job")
        return
    if family and leaf == "worker-failure.json":
        _fields(value, ("status", "error_type", "diagnostic"), "bounded worker failure only")
        require(value["status"] == "failed_no_retry" and value["error_type"] in ("ValueError", "TypeError", "OSError",
            "PermissionError", "FileNotFoundError", "RuntimeError", "TimeoutError", "ImportError", "ModuleNotFoundError",
            "MemoryError", "KeyError", "IndexError", "AttributeError", "AssertionError", "RecursionError", "KeyboardInterrupt",
            "SystemExit", "OtherException"), "fixed public error type")
        d = value["diagnostic"]
        _fields(d, ("schema", "stage", "index", "family", "mode", "frames", "frames_truncated", "chain_followed",
            "message_disclosed", "source_lines_disclosed", "locals_disclosed"), "bounded failure diagnostic schema")
        require(d["schema"] == "k1-asr18-bounded-failure-v1" and d["family"] in (*c.FAMILIES, "unknown")
            and d["mode"] in ("prepare", "decode", "unknown") and d["stage"] in ("startup", "pre_model_import", "release", "scope", "guard", "inputs", "load_model", "prepare_receipt", "compare_frozen", "bind_predecode", "bind_audio", "start_attempt", "infer", "complete_attempt", "worker_complete", "unknown") and
            all(d[k] is False for k in ("chain_followed", "message_disclosed", "source_lines_disclosed", "locals_disclosed")), "no exception/private disclosure")
        require(d["index"] is None or type(d["index"]) is int and 0 <= d["index"] <= 6, "fixed failed slot index")
        require(type(d["frames"]) is list and len(d["frames"]) <= 16 and type(d["frames_truncated"]) is bool,
            "bounded traceback metadata")
        for frame in d["frames"]:
            _fields(frame, ("file", "function", "line"), "bounded public code frame")
            require(type(frame["file"]) is str and (frame["file"] == "<redacted>" or
                frame["file"].startswith(("research/", "tools/", "runtime/site-packages/"))), "public relative frame only")
            require(type(frame["function"]) is str and re.fullmatch(r"(?:<redacted>|[A-Za-z_][A-Za-z0-9_]*)", frame["function"]) and len(frame["function"]) <= 160 and
                (frame["line"] is None or type(frame["line"]) is int and frame["line"] > 0), "bounded frame metadata")
        return
    if family and leaf == "started.json":
        _started(value, family, parts[2]); return
    if family and leaf == "completed.json":
        _fields(value, COMPLETION_FIELDS, "complete new native receipt schema")
        started = {k: value[k] for k in START_FIELDS}; started["schema"] = "retained-asr-started-v1"
        _started(started, family, parts[2])
        require(value["schema"] == "retained-asr-completed-v1" and value["status"] == "success" and
            value["decoder_completed"] is True and value["processed_start_frame"] == 0 and
            value["processed_end_frame"] == value["audio_frames"] and value["acoustic_verbatim_completeness"] == "unknown", "complete native terminal semantics")
        # Validate the unchanged nested receipt shape after replacing only the
        # explicit campaign-binding fields; no old identity value is substituted.
        old = {k: v for k, v in value.items() if k not in ("job_sha256", "execution_plan_sha256",
            "worker_input_binding_sha256", "screening_receipt_sha256", "tts_authority_sha256", "run_attempt")}
        old["original_job_sha256"] = value["job_sha256"]; old["remaining_plan_sha256"] = value["execution_plan_sha256"]
        require(any(_matches_shape(old, ref) for ref in _RETAINED_MEMBERS.get(key, [])), "complete retained native nested shape")
        return
    if family and leaf == "native-config.json":
        import worker_receipts
        worker_receipts.validate_native_config(value)
        require(value["adapter_id"] == worker_receipts.ADAPTERS[family], "native config same family")
        return
    if family and leaf.startswith("update-"):
        _fields(value, ("step", "token_ids", "completed"), "full native prefix fields")
        require(family == "whisper" and type(value["step"]) is int and 1 <= value["step"] <= 224
            and leaf == f"update-{value['step']:03d}.json" and type(value["completed"]) is bool, "full native prefix number")
        require(type(value["token_ids"]) is list and len(value["token_ids"]) == value["step"] + 4 and
            all(type(x) is int and 0 <= x < 51865 for x in value["token_ids"]), "full observed native prefix retained")
        return
    if key.endswith("/native-observation.json"):
        observed = dict(value)
        error = observed.pop("exception_type", None)
        require(error is None or type(error) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", error), "bounded native exception class")
        require({"schema", "opaque_id", "input_binding_sha256", "inference_returned", "acoustic_verbatim_completeness"} <= set(observed), "failure observation minimum retained")
        require(any(_matches_shape(observed, ref, subset=True) for ref in _RETAINED_MEMBERS.get(key, [])), "bounded actual partial native observation")
        return
    choices = _RETAINED_MEMBERS.get(key)
    require(choices is not None and any(_matches_shape(value, ref) for ref in choices), "exact retained public metadata schema required")



def archive_job_identity(payload, run_identity):
    """Read a job candidate only after complete native/archive validation.

    This local observation is still cross-checked against the independent API
    job before publication. It does not independently establish Actions truth.
    """
    manifest, _ = validate_archive(payload, run_identity)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        preflight = decode(archive.read("run-preflight.json"))
    number = preflight.get("numeric_job_id")
    require(type(number) is int and 0 < number < 10 ** 20, "bounded archived actual job")
    return number


def _source_claim_mode(files):
    """Read one literal source-compiled policy, never execute archived code."""
    prefix = "source/research/k1-punctuation6-asr18-v1/"
    release = c.decode_json(files[prefix + "execution-release.json"], require_canonical=False)
    source = files[prefix + "campaign_contract.py"]
    declarations = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CURRENT_CLAIM_MODE" for target in node.targets):
            require(isinstance(node.value, ast.Constant) and type(node.value.value) is str,
                "literal archived source admission policy")
            declarations.append(node.value.value)
    bound = c.is_bound_release(release)
    require(declarations == (["git-owner-v2"] if bound else []),
        "actual archived source policy and release version must agree")
    return bound


def bind_archive_admission(files, run_identity, freeze_sha256, numeric_job_id, source_tree):
    """Bind public owner proof to exact archived source bytes, never its schema alone.

    Callers must additionally verify every source member against the original
    immutable Git tree. Native/archive validation retains the same obligations.
    """
    prefix = "source/research/k1-punctuation6-asr18-v1/"
    release_raw = files[prefix + "execution-release.json"]
    release = c.decode_json(release_raw, require_canonical=False)
    if not _source_claim_mode(files):
        require("admission.json" not in files, "legacy source cannot self-promote using public admission")
        if "authority/source-claim.json" in files:
            c.validate_legacy_source_claim(files["authority/source-claim.json"],
                expected_sha256=sha(files["authority/source-claim.json"]), release=release,
                source_freeze_sha256=freeze_sha256, run_id=str(run_identity["run_id"]),
                executed_head_sha=run_identity["executed_head_sha"])
        return None
    control_raw = files[prefix + c.CLAIM_CONTROL_FILE]
    control = c.require_claim_admission(release, control_raw)
    c.require_current_scope(control)
    admission_raw = files["admission.json"]
    require(type(admission_raw) is bytes and len(admission_raw) <= c.MAX_BOUND_ADMISSION,
        "existing bounded public admission bytes")
    admission = c.decode_json(admission_raw)
    owner = admission["owner"]["identity"]
    preflight = decode(files["run-preflight.json"])
    # Workflow/job are local candidates here, independently checked by the
    # existing current run/jobs GETs. Freeze/release are exact source bodies.
    identity = c.sourcebound_identity(run_identity, numeric_job_id, release, source_tree,
        owner["source_timestamp"], freeze_sha256, sha(release_raw), control,
        preflight.get("workflow_id"))
    c.validate_bound_admission(admission, identity)
    c.require_current_scope(identity)
    claim = None
    if "authority/source-claim.json" in files:
        claim = c.bind_source_claim(files["authority/source-claim.json"], run_identity, release,
            numeric_job_id, source_tree, release_raw=release_raw, control_raw=control_raw,
            source_freeze_sha256=freeze_sha256)
        require(c.exact_json(claim["admission"], admission), "same fixed public admission and worker claim")
    return {"mode": "git-owner-v2", "identity": identity, "admission": admission,
        "source_claim": claim, "release_sha256": sha(release_raw), "control_sha256": sha(control_raw)}


def validate_publisher_source_release(payload, release, proof):
    """Do not let detached publisher metadata downgrade an actual v2 source."""
    if proof is None:
        require(release["schema"] == "k1-asr18-publisher-bound-release-v1", "legacy publisher source policy")
        return
    c.require_current_scope(proof["identity"])
    require(release["commit_timestamp"] == proof["identity"]["source_timestamp"], "publisher actual source timestamp")
    require(all(release[key] == proof["identity"][key] for key in ("source_branch", "workflow", "result_branch")),
        "publisher immutable source selectors")
    require(release["schema"] == "k1-asr18-publisher-bound-release-v2" and
        release["claim_admission_mode"] == proof["mode"] and
        release["execution_release_sha256"] == proof["release_sha256"] and
        release["claim_admission_control_sha256"] == proof["control_sha256"],
        "publisher binds exact immutable release and control bytes")


def validate_source_binding(payload, run_identity, freeze_sha256, source_rows, numeric_job_id, source_tree):
    """Bind archived source bytes to the independently read exact Git source tree."""
    manifest, _ = validate_archive(payload, run_identity)
    require(manifest["members"]["authority/execution-freeze.json"]["sha256"] == freeze_sha256,
        "archive independent source-freeze binding")
    require(type(numeric_job_id) is int and 0 < numeric_job_id < 10 ** 20, "actual numeric publisher job")
    by_path = {row["path"]: row for row in source_rows}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in manifest["members"]:
            if name.startswith("source/"):
                actual = by_path.get(name[len("source/"):])
                require(actual is not None and actual["mode"] in ("100644", "100755") and actual["type"] == "blob"
                    and actual["sha"] == git_sha(archive.read(name)), "archived public source equals actual Git source")
        preflight = decode(archive.read("run-preflight.json"))
        host = decode(archive.read("host-receipt.json"))
        require(preflight.get("numeric_job_id") == numeric_job_id and type(preflight.get("numeric_job_id")) is int
            and host.get("executed_tree_sha") == source_tree, "same independently verified actual job/source tree")
        if "authority/source-claim.json" in manifest["members"]:
            claim = decode(archive.read("authority/source-claim.json"))
            require(type(claim.get("numeric_job_id")) is int and claim["numeric_job_id"] == numeric_job_id,
                "same actual once-source-claim job")
        files = {name: archive.read(name) for name in manifest["members"]}
    return bind_archive_admission(files, run_identity, freeze_sha256, numeric_job_id, source_tree)



_SETUP_REASONS = frozenset(("entry_contract", "python_version", "fresh_directories", "disk_reserve",
    "source_revision", "source_path", "wheel_set", "wheel_filename", "unapproved_url", "unapproved_redirect",
    "byte_cap", "declared_length", "sha256_mismatch", "installed_size", "diagnostic_identity", "duplicate_asset",
    "validation_rejected", "http_error", "url_error", "tls_error", "timeout", "file_exists", "permission_denied",
    "connection_error", "incomplete_read", "os_error", "subprocess_exit", "interrupted", "runtime_error", "unexpected_exception"))
_BROKER_FAILURES = ("plan", "state", "request", "destination", "dns", "connection", "connection_limit", "deadline", "byte_cap", "io", "poisoned")


def _public_reason(value):
    require(type(value) is dict and "reason_code" in value and set(value) <= {"reason_code", "http_status", "returncode", "errno"}
        and value["reason_code"] in _SETUP_REASONS, "exact bounded setup reason")
    for k, low, high in (("http_status", 100, 599), ("returncode", -255, 255), ("errno", 0, 4096)):
        if k in value: require(type(value[k]) is int and low <= value[k] <= high, "bounded numeric setup reason")


def _setup_boundary(value):
    expected = {"source_role": "source_screen_only", "training_admitted": False, "shipping_approved": False,
        "human_gold": False, "acoustic_verbatim_completeness": "unknown"}
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()), "setup is not inference/admission")
    return set(expected)


def _gate_shape(value):
    if value is None: return
    require(type(value) is dict and value.get("status") in ("missing", "valid", "invalid"), "fixed observed gate variant")
    if value["status"] == "missing":
        _fields(value, ("status",), "exact missing gate observation")
    elif value["status"] == "valid":
        _fields(value, ("status", "schema", "exit_code"), "exact retained gate outcome")
        require(value["schema"] == "qwen12-gated-command-v1" and type(value["exit_code"]) is int and
            -255 <= value["exit_code"] <= 255, "bounded gate outcome")
    else:
        _host()._safe_failure({k:v for k,v in value.items() if k != "status"})


def _special_retained(name, value):
    if name == "asr-setup-public.json":
        _fields(value, ("schema", "status", "model_calls", "actual_download_bytes", "setup_receipt_sha256"), "exact ASR18 setup projection")
        require(value["schema"] == "k1-asr18-setup-summary-v1" and value["status"] == "complete" and
            type(value["model_calls"]) is int and value["model_calls"] == 0 and
            type(value["actual_download_bytes"]) is int and 0 <= value["actual_download_bytes"] <= 4 * 1024**3 and
            type(value["setup_receipt_sha256"]) is str and SHA.fullmatch(value["setup_receipt_sha256"]), "setup observation only")
        return True
    if name == "cleanup-receipt.json":
        _fields(value, ("containers", "all_removed", "all_cleanup_verified"), "whole cleanup observation")
        require(type(value["containers"]) is list and len(value["containers"]) <= 128 and
            type(value["all_removed"]) is bool and type(value["all_cleanup_verified"]) is bool, "bounded cleanup roster")
        mandatory = {"name", "container_id", "status", "stage", "ownership", "operations", "cleanup_budget_seconds",
            "host_deadline_expired_at_start", "host_deadline_overrun", "pause_acknowledged", "errors", "accounting_status",
            "gate_result", "cleanup_seconds", "cleanup_overrun_seconds", "cleanup_ok", "receipt_errors"}
        for row in value["containers"]:
            require(type(row) is dict and mandatory <= set(row) <= mandatory | {"removal_proof"}, "complete cleanup failure/success fields")
            require(type(row["name"]) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["name"]) and
                (row["container_id"] is None or type(row["container_id"]) is str and SHA.fullmatch(row["container_id"])), "bounded container identity")
            require(row["stage"] in (*c.FAMILIES, "setup") and row["status"] in ("removed", "removal_unconfirmed") and
                row["ownership"] in ("verified", "unknown") and row["accounting_status"] in
                ("final_usage_verified", "not_started", "final_usage_unverified") and row["cleanup_budget_seconds"] == 15, "cleanup scope/status")
            for key in ("host_deadline_expired_at_start", "host_deadline_overrun", "pause_acknowledged", "cleanup_ok"):
                require(type(row[key]) is bool, "observed cleanup boolean")
            for key in ("cleanup_seconds", "cleanup_overrun_seconds"):
                require(type(row[key]) in (int, float) and 0 <= row[key] <= 86400, "bounded cleanup duration")
            if "removal_proof" in row: require(row["removal_proof"] in ("acknowledged_remove", "successful_empty_exact_id_query"), "fixed removal evidence")
            _gate_shape(row["gate_result"])
            for key in ("errors", "receipt_errors"):
                require(type(row[key]) is list and len(row[key]) <= 16, "bounded cleanup errors")
                for error in row[key]:
                    require(type(error) is dict and error.get("operation") in ("final_accounting", "force_remove", "verify_absence",
                        "ownership", "cpu_receipt", "cleanup_receipt"), "bounded cleanup error operation")
                    _host()._safe_failure({k: v for k, v in error.items() if k != "operation"})
            require(type(row["operations"]) is list and len(row["operations"]) <= 16, "bounded cleanup operations")
            for operation in row["operations"]:
                require(type(operation) is dict and {"operation", "status", "wall_seconds"} <= set(operation) and
                    operation["operation"] in ("pause", "inspect_paused", "final_cpu_close", "force_remove", "verify_absence") and
                    operation["status"] in ("not_started", "started", "succeeded", "failed"), "fixed cleanup operation")
                for key in ("wall_seconds", "timeout_seconds"):
                    if key in operation: require(type(operation[key]) in (int, float) and 0 <= operation[key] <= 86400, "bounded cleanup operation duration")
                error = {k:v for k,v in operation.items() if k not in ("operation", "status", "wall_seconds", "timeout_seconds", "returncode")}
                if error: _host()._safe_failure(error)
                if "returncode" in operation: require(type(operation["returncode"]) is int and -255 <= operation["returncode"] <= 255, "bounded cleanup return code")
        return True
    if name == "image-transfer.json":
        required = {"schema", "status", "image", "reservation_bytes", "reservation_kind", "declared_config_and_compressed_layer_bytes",
            "declared_manifest_bytes", "default_interface_received_delta_bytes", "daemon_completion_verified",
            "cumulative_image_transfer_bytes_verified", "pull_client_attempts", "byte_scope"}
        optional = {"network_before", "network_after", "all_interface_delta_diagnostic_only_bytes", "pull_client_exit_code", "image_config_digest",
            "error_type", "daemon_fetch_may_continue", "after_snapshot_available"}
        require(type(value) is dict and required <= set(value) <= required | optional and
            value["schema"] == "screen32-image-transfer-observation-v2", "complete bounded image observation")
        require(value["status"] in ("started", "completed_observed_within_reservation", "stopped_before_setup_daemon_completion_unverified") and
            value["image"] == "docker.io/library/python@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a" and
            value["reservation_bytes"] == 128 * MIB and type(value["pull_client_attempts"]) is int and 0 <= value["pull_client_attempts"] <= 1,
            "fixed image and single attempted pull")
        require(value["reservation_kind"] == "monitored_default_interface_receive_observation_not_hard_transfer_limit" and
            value["byte_scope"] == "selected_default_interface_includes_protocol_and_unrelated_host_traffic_not_exact_daemon_accounting",
            "fixed honest image measurement scope")
        if "image_config_digest" in value: require(type(value["image_config_digest"]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_config_digest"]), "actual image config digest")
        if "pull_client_exit_code" in value: require(type(value["pull_client_exit_code"]) is int and -255 <= value["pull_client_exit_code"] <= 255, "bounded image client exit")
        for key in ("daemon_completion_verified", "cumulative_image_transfer_bytes_verified", "daemon_fetch_may_continue", "after_snapshot_available"):
            if key in value: require(type(value[key]) is bool, "explicit image observation boolean")
        for key in ("declared_config_and_compressed_layer_bytes", "declared_manifest_bytes", "default_interface_received_delta_bytes", "all_interface_delta_diagnostic_only_bytes"):
            if key in value: require(value[key] is None or type(value[key]) is int and value[key] >= 0, "nonnegative image observation or unknown")
        for key in ("network_before", "network_after"):
            if key not in value: continue
            net = value[key]; _fields(net, ("default_interface", "interfaces"), "network counter projection only")
            require(type(net["interfaces"]) is dict and 1 <= len(net["interfaces"]) <= 64 and net["default_interface"] in net["interfaces"], "bounded interface counter map")
            for interface, row in net["interfaces"].items():
                require(type(interface) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", interface), "bounded interface name")
                _fields(row, ("rx_bytes", "ifindex"), "counter fields only")
                require(type(row["rx_bytes"]) is int and row["rx_bytes"] >= 0 and type(row["ifindex"]) is int and row["ifindex"] > 0, "observed counters")
        if "error_type" in value: require(type(value["error_type"]) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value["error_type"]), "image error class only")
        return True
    if name == "whisper-setup/phased-setup-failure.json":
        boundary = _setup_boundary(value)
        fixed = {"schema", "phase", "status", "model_calls", "retry_count"} | boundary
        require(fixed <= set(value) <= fixed | {"reason_code", "http_status", "returncode", "errno"}, "phased failure fields only")
        require(value["schema"] == "whisper-screen-phased-failure-v1" and value["phase"] in ("install", "source", "model", "finalize") and
            value["status"] == "failed_no_retry" and type(value["model_calls"]) is int and value["model_calls"] == 0 and
            type(value["retry_count"]) is int and value["retry_count"] == 0, "zero-model no-retry phased failure")
        _public_reason({k:v for k,v in value.items() if k not in fixed}); return True
    if re.fullmatch(r"whisper-setup/phase-(install|source|model|finalize)-progress\.json", name):
        boundary = _setup_boundary(value)
        fields = {"schema", "status", "stage", "completed_stages", "lock_sha256", "asset", "assets", "response_body_bytes_observed",
            "verified_assets", "model_calls", "retry_count", "network_transfer_total_known", "byte_scope", "phase"} | boundary
        require(fields <= set(value) <= fields | {"failure"} and value["schema"] == "whisper-screen-phased-progress-v1" and
            name == "whisper-setup/phase-" + value["phase"] + "-progress.json", "exact phased progress shape")
        require(value["status"] in ("started", "complete", "failed_no_retry") and type(value["stage"]) is str and
            re.fullmatch(r"[a-z_]{1,80}", value["stage"]), "bounded setup stage")
        require(type(value["completed_stages"]) is list and len(value["completed_stages"]) <= 64 and
            all(type(x) is str and re.fullmatch(r"[a-z_]{1,80}", x) for x in value["completed_stages"]), "bounded completed stages")
        require(type(value["lock_sha256"]) is dict and set(value["lock_sha256"]) <= {"runtime-lock.json", "source-lock.json"} and
            all(type(x) is str and SHA.fullmatch(x) for x in value["lock_sha256"].values()), "fixed public setup lock hashes")
        require(type(value["assets"]) is list and len(value["assets"]) <= 128 and
            (value["asset"] is None or value["asset"] in value["assets"]), "full bounded observed asset roster")
        for asset in value["assets"]:
            _fields(asset, ("group", "index", "expected_sha256", "expected_bytes", "status", "response_body_bytes_observed", "observed_sha256", "http_status"), "bounded asset observation")
            require(asset["group"] in ("source", "model", "wheel") and type(asset["index"]) is int and 1 <= asset["index"] <= 128 and
                asset["status"] in ("started", "verified", "failed_no_retry"), "bounded asset scope")
            for key in ("expected_sha256", "observed_sha256"):
                require((key == "observed_sha256" and asset[key] is None) or type(asset[key]) is str and SHA.fullmatch(asset[key]), "public fixed asset hash")
            for key in ("expected_bytes", "response_body_bytes_observed"):
                require(asset[key] is None or type(asset[key]) is int and 0 <= asset[key] <= 1024**3, "bounded asset byte observation")
            require(asset["http_status"] is None or type(asset["http_status"]) is int and 100 <= asset["http_status"] <= 599, "bounded HTTP status only")
        for key in ("response_body_bytes_observed", "verified_assets", "model_calls", "retry_count"):
            require(type(value[key]) is int and value[key] >= 0, "nonnegative setup counts")
        require(value["model_calls"] == value["retry_count"] == 0 and value["network_transfer_total_known"] is False, "no setup model calls/retries or false all-wire claim")
        if "failure" in value: _public_reason(value["failure"])
        return True
    match = re.fullmatch(r"whisper-setup/bounded-acquisition/(wheel-[0-9]{2}|source|model)-failure\.json", name)
    if match:
        _fields(value, ("schema", "asset_id", "status", "broker_failure_code", "broker_observed_ciphertext_bytes", "budget_charge_kind"), "exact broker failure observation")
        require(value["schema"] == "bounded-host-asset-failure-v1" and value["asset_id"] == match[1] and
            value["status"] == "failed_no_retry" and value["broker_failure_code"] in (None, *_BROKER_FAILURES) and
            value["budget_charge_kind"] in ("conservative_reserved_allowance", "observed") and
            (value["broker_observed_ciphertext_bytes"] is None or type(value["broker_observed_ciphertext_bytes"]) is int and
             0 <= value["broker_observed_ciphertext_bytes"] <= 1536 * MIB), "no retry broker failure retained")
        return True
    return False
