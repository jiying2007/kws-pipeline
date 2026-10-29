"""Resolve retained ASR evidence against code-reviewed, immutable source roots.

Locations are untrusted storage, never authority. There are no production train
roots today. Adding one requires a reviewed code change after its actual source,
rights and capture provenance are checked. This module only reads bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import weakref
import wave
import zipfile

from speech_label_admission import canonical_sha256, require_sha
from source_identity import LINEAGE_FIELDS, validate_source_identity_values

_REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_POLICY = "immutable-asr-source-resolver-v1"
CAPTURE_CONTRACT = "explicit-complete-asr-capture-v1"


@dataclass(frozen=True)
class _SourceRoot:
    root_id: str
    trust_root_commit: str
    provenance_path: str
    provenance_sha256: str
    allowed_role: str
    capture_contract: str | None = None


# A tiny trust-root index, not a second artifact/member registry. Every archive
# and member identity is read from the existing pinned provenance document.
_SOURCE_ROOTS = (
    _SourceRoot("screen32-generation-qwen-38018029787",
                "d899c65c9b89b7d3b23b197b879f7d9afa688c58",
                "research/source-screen32-v1/evidence/run-38018029787/provenance.json",
                "bff000137b4bb851a49b7115c4ebdd78c39fd0445b17180e957dcb5540892bb3", "source_screen_only"),
    _SourceRoot("screen32-sense-continuation-38021467257",
                "d899c65c9b89b7d3b23b197b879f7d9afa688c58",
                "research/source-screen32-v1/evidence/run-38021467257/provenance.json",
                "f135d0131aee89a14630a15006225b3be3a8a4e5d6ffc6efba6938cb1da3d489", "source_screen_only"),
)
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_TOKEN = object()
_CONTEXTS = weakref.WeakKeyDictionary()


class VerifiedSourceContext:
    """Opaque per-process handle. Consumption rechecks bytes, not just identity."""
    __slots__ = ("__weakref__",)

    def __new__(cls, token=None):
        if token is not _TOKEN:
            raise ValueError("verified source context must be created by the controlled resolver")
        return super().__new__(cls)

    def __reduce__(self):
        raise TypeError("verified source contexts cannot be serialized")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes, label: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected object")
    return value


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("evidence path must be canonical and relative")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts) or str(path) != value:
        raise ValueError("unsafe evidence path")
    return value


def _read_bounded(root: Path, relative: str, limit: int) -> bytes:
    relative = _relative(relative)
    parts = (*Path(root).absolute().parts[1:], *PurePosixPath(relative).parts)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("safe retained-evidence reading requires no-follow directory descriptors")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    fd = None
    try:
        # Traverse with directory FDs, rather than a symlink precheck followed
        # by a second path lookup. A renamed directory does not redirect an
        # already opened FD. The batch-end strong rehash also checks the path.
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("missing, nonregular or oversized retained evidence")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(raw) > limit:
            raise ValueError("retained evidence grew past size bound")
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError("retained evidence changed while reading")
        return raw
    except OSError as exc:
        raise ValueError("missing, symlinked or unreadable retained evidence") from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def evidence_location(value: object = None) -> Path:
    """Operator-selected storage only; no root/authority is accepted from it."""
    if value is None:
        value = os.environ.get("KWS_ASR_EVIDENCE_DIR")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: local KWS_ASR_EVIDENCE_DIR is required")
    return Path(value).absolute()


class _VerificationSession:
    def __init__(self, evidence_dir, token):
        if token is not _TOKEN:
            raise ValueError("verification session requires the controlled factory")
        self.location = evidence_dir
        self.active = True
        self.raw = {}
        self.members = {}
        self.total_bytes = 0

    def add_size(self, size):
        self.total_bytes += size
        if self.total_bytes > _MAX_TOTAL_BYTES:
            raise ValueError("ASR verification session snapshot exceeds bounded memory")

    def read(self, root, relative, limit):
        if not self.active:
            raise ValueError("ASR verification session is closed")
        key = (Path(root).absolute(), relative, limit)
        if key not in self.raw:
            data = _read_bounded(*key)
            self.add_size(len(data))
            self.raw[key] = data
        return self.raw[key]

    def checkpoint(self):
        if not self.active:
            raise ValueError("ASR verification session is closed")
        # Every independent entry reopens and strongly rehashes each storage
        # object once before returning. Stat/hash claims supplied by callers
        # are never cache authority. Within the batch only pinned immutable
        # byte snapshots are read; no sample-count times archive-size rereads.
        for (root, relative, limit), prior in self.raw.items():
            if _sha(_read_bounded(root, relative, limit)) != _sha(prior):
                raise ValueError("retained ASR evidence changed during verification session")


@contextmanager
def source_verification_session(evidence_dir=None):
    session = _VerificationSession(evidence_dir, _TOKEN)
    try:
        yield session
        session.checkpoint()
    finally:
        session.active = False
        session.raw.clear()
        session.members.clear()


def _read(root, relative, limit, session=None):
    return session.read(root, relative, limit) if session is not None else _read_bounded(root, relative, limit)


def _archives(provenance: dict) -> dict:
    archives = provenance.get("archives")
    if archives is None and isinstance(provenance.get("archive"), dict):
        archives = {"archive": provenance["archive"]}
    if not isinstance(archives, dict) or not archives:
        raise ValueError("trusted provenance has no archive inventory")
    return archives


def _load_archive(store: Path, row: dict, session=None) -> dict[str, bytes]:
    if not isinstance(row, dict):
        raise ValueError("trusted archive inventory entry must be an object")
    digest = require_sha(row.get("sha256"), "trusted archive sha256")
    # Stable content-addressed locations avoid leaking private paths in receipts.
    cache_key = (store, digest, canonical_sha256(row))
    if session is not None and cache_key in session.members:
        return session.members[cache_key]
    raw = _read(store, f"{digest}.zip", _MAX_ARCHIVE_BYTES, session)
    if type(row.get("bytes")) is not int or len(raw) != row["bytes"] or _sha(raw) != digest:
        raise ValueError("retained archive identity changed")
    expected = row.get("exact_members")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("trusted archive has no exact member inventory")
    for name in expected:
        _relative(name)
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or {item.filename for item in infos} != set(expected):
            raise ValueError("archive members differ from exact trusted inventory")
        if sum(item.file_size for item in infos) > _MAX_TOTAL_BYTES:
            raise ValueError("archive uncompressed size exceeds bound")
        for info in infos:
            _relative(info.filename)
            if (info.is_dir() or info.file_size > _MAX_MEMBER_BYTES or info.flag_bits & 1
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("unsupported/unsafe archive member")
            declared = expected[info.filename]
            if not isinstance(declared, dict) or type(declared.get("bytes")) is not int or declared["bytes"] != info.file_size:
                raise ValueError("retained member declared size differs")
            with archive.open(info, "r") as member_stream:
                content = member_stream.read(min(_MAX_MEMBER_BYTES, declared["bytes"]) + 1)
            if (not isinstance(declared, dict) or type(declared.get("bytes")) is not int
                    or len(content) != declared["bytes"] or info.file_size != len(content)
                    or _sha(content) != require_sha(declared.get("sha256"), "trusted member hash")):
                raise ValueError("retained member identity changed")
            if session is not None:
                session.add_size(len(content))
            result[info.filename] = content
    if session is not None:
        session.members[cache_key] = result
    return result


def _source_identity(raw: bytes) -> dict:
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio:
            if (audio.getnchannels() != 1 or audio.getframerate() != 16000
                    or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE"):
                raise ValueError("trusted source must be mono 16-kHz PCM16")
            frames = audio.getnframes()
            pcm = audio.readframes(frames + 1)
            if frames <= 0 or len(pcm) != 2 * frames:
                raise ValueError("trusted source frames are missing/truncated")
    except (EOFError, wave.Error) as exc:
        raise ValueError("malformed trusted source WAV") from exc
    return {"file_sha256": _sha(raw), "pcm_sha256": _sha(pcm), "audio_frames": frames}


def _same(actual, expected):
    return type(actual) is type(expected) and canonical_sha256(actual) == canonical_sha256(expected)


def _equal_fields(actual: dict, expected: dict, fields, label: str) -> None:
    for field in fields:
        if not _same(actual.get(field), expected.get(field)):
            raise ValueError(f"{label}: {field} differs from retained evidence")


def _resolve(record: dict, store: Path, session=None) -> dict:
    from speech_asr_consensus import validate_consensus_record
    validate_consensus_record(record)
    bindings = record.get("evidence_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "root_id", "source_policy", "rights", "source_audio", "recognizers",
    }:
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: exact retained evidence bindings required")
    roots = [root for root in _SOURCE_ROOTS if root.root_id == bindings["root_id"]]
    if len(roots) != 1:
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: source root is not trusted by reviewed code")
    root = roots[0]
    raw_provenance = _read(_REPO_ROOT, root.provenance_path, _MAX_MEMBER_BYTES, session)
    if _sha(raw_provenance) != root.provenance_sha256:
        raise ValueError("trusted source provenance changed")
    provenance = _json(raw_provenance, "source provenance")
    if provenance.get("repository") != "jiying2007/kws-pipeline":
        raise ValueError("trusted source repository differs")
    executed_head = provenance.get("executed_head_sha")
    if (type(provenance.get("run_id")) is not int or provenance["run_id"] < 1
            or not isinstance(executed_head, str) or len(executed_head) != 40
            or any(char not in "0123456789abcdef" for char in executed_head)):
        raise ValueError("trusted provenance must bind the actual run and executed source commit")
    if root.allowed_role != "train" or root.capture_contract != CAPTURE_CONTRACT:
        raise ValueError("trusted source is screen-only/unsupported, never training-admitted")
    archives = _archives(provenance)
    cache = {}
    used = {}

    def read(ref):
        if not isinstance(ref, dict) or set(ref) != {"archive", "member"}:
            raise ValueError("evidence reference must select an archive and exact member")
        name = ref["archive"]
        if not isinstance(name, str) or name not in archives:
            raise ValueError("evidence archive not in trusted provenance")
        member = _relative(ref["member"])
        if name not in cache:
            cache[name] = _load_archive(store, archives[name], session)
        if member not in cache[name]:
            raise ValueError("evidence member not in trusted provenance")
        raw = cache[name][member]
        used[(name, member)] = _sha(raw)
        return raw

    policy = _json(read(bindings["source_policy"]), "source policy")
    if (policy.get("schema") != "retained-asr-source-policy-v1"
            or policy.get("allowed_roles") != ["train"] or policy.get("human_gold") is not False
            or not _same(policy.get("source_provenance"), record["source_provenance"])):
        raise ValueError("retained source policy is not eligible train-only machine evidence")
    validate_source_identity_values(policy, "source policy")
    validate_source_identity_values(record, "source record")
    _equal_fields(policy, record, ("source_id", "intended_text", *LINEAGE_FIELDS), "source policy")
    if (not _same(policy.get("human_review_history"), record["human_review_history"])
            or policy.get("human_review_history_sha256") != canonical_sha256(record["human_review_history"])):
        raise ValueError("known human review history differs from trusted source policy")
    for alias in ("role", "allowed_role", "split_role", "source_role", "split"):
        if alias in policy and policy[alias] != "train":
            raise ValueError("retained source role cannot be relabeled")
    if (policy.get("source_screen_only") is True or policy.get("review_origin") == "human"
            or policy.get("training_use_prohibited") is True or policy.get("training_use_allowed") is False):
        raise ValueError("retained source policy has contradictory machine/source roles")
    if record["schema_version"] == 2:
        if (not _same(policy.get("human_observation_history"), record["human_observation_history"])
                or policy.get("human_observation_history_sha256") != canonical_sha256(record["human_observation_history"])):
            raise ValueError("raw human observations differ from trusted source history")
        from speech_asr_consensus import human_conflicts, raw_human_conflicts
        observations, typed_reviews, typed_conflicts = [], [], {}
        for reference in record["human_observation_history"]:
            observation = _json(read(reference["artifact"]), "private raw human observation")
            if (canonical_sha256(observation) != reference["observation_sha256"]
                    or not _same(observation.get("observation_revision"), reference["observation_revision"])
                    or observation.get("supersedes_observation_sha256") != reference["supersedes_observation_sha256"]):
                raise ValueError("private human observation differs from retained reference")
            if "original_observation_artifact" in observation:
                original = read(observation["original_observation_artifact"])
                if not original or _sha(original) != observation.get("original_observation_sha256"):
                    raise ValueError("original human note differs from retained bytes")
                try:
                    original_object = _json(original, "original human observation")
                except (ValueError, UnicodeError):
                    original_object = None
                is_typed = isinstance(original_object, dict) and original_object.get("evidence_class") == "speech-like-audio-review-v2"
                if observation.get("original_format", "raw-note") == "human-review-v2":
                    if not is_typed:
                        raise ValueError("typed human reference must retain its original review bytes")
                    typed_reviews.append(original_object)
                    conflict, _ = human_conflicts(typed_reviews, record)
                    typed_conflicts[observation["observation_revision"]] = conflict
                    # Optional known transcript summaries cannot rewrite the
                    # original typed review. Unknown summaries remain unknown;
                    # actual conflict/use constraints come from original bytes.
                    if (observation.get("observed_text") is not None
                            and observation["observed_text"] != original_object.get("actual_text")):
                        raise ValueError("human wrapper rewrites original typed transcript")
                elif is_typed:
                    raise ValueError("known typed human history cannot be downgraded to an opaque raw note")
            elif observation.get("original_format") == "human-review-v2":
                raise ValueError("typed human review must retain original bytes")
            observations.append(observation)
        current, historical = human_conflicts(record["human_review_history"], record)
        raw_current, raw_historical = raw_human_conflicts(observations, record, typed_conflicts=typed_conflicts)
        if (record["human_conflict"] is not (current or raw_current)
                or record["historical_human_conflict"] is not (historical or raw_historical)):
            raise ValueError("human conflict audit differs from private trusted observations")
        if policy.get("training_use_prohibited") is not False:
            raise ValueError("trusted native-v2 source policy must explicitly permit training use")
        _equal_fields(policy, record, ("label_selection_policy",), "trusted label-selection policy")
    rights_raw = read(bindings["rights"])
    rights = _json(rights_raw, "source rights")
    if (_sha(rights_raw) != record["source_provenance"]["rights_evidence_sha256"]
            or rights.get("schema") != "retained-asr-source-rights-v1"
            or rights.get("training_use_allowed") is not True
            or rights.get("license_id") != record["source_provenance"]["license_id"]):
        raise ValueError("retained source rights do not authorize training")
    _equal_fields(rights, record, ("source_id", "source_family_id"), "source rights")
    measured = _source_identity(read(bindings["source_audio"]))
    _equal_fields(record, measured, measured, "source WAV/PCM/frame identity")
    _equal_fields(policy, record, ("file_sha256", "pcm_sha256", "audio_frames"), "source policy")
    refs = bindings["recognizers"]
    identities = {item["recognizer_id"]: item for item in record["recognizer_policy"]["recognizers"]}
    observations = {item["recognizer_id"]: item for item in record["observations"]}
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError("exactly three retained recognizer evidence chains required")
    seen = set()
    for refs_row in refs:
        expected_refs = {"recognizer_id", "model_lock", "source_lock", "native_decode_config",
                         "started", "receipt", "raw_output"}
        if record["schema_version"] == 2:
            expected_refs.add("native_dependencies")
        if not isinstance(refs_row, dict) or set(refs_row) != expected_refs:
            raise ValueError("incomplete retained recognizer evidence chain")
        name = refs_row["recognizer_id"]
        if not isinstance(name, str) or name not in identities or name in seen:
            raise ValueError("retained recognizer chains must cover each model exactly once")
        seen.add(name)
        identity, observation = identities[name], observations[name]
        model_raw, code_raw, config_raw = (read(refs_row[key]) for key in
                                           ("model_lock", "source_lock", "native_decode_config"))
        model, code, config = (_json(raw, label) for raw, label in
                              ((model_raw, "model lock"), (code_raw, "source lock"), (config_raw, "native config")))
        if model.get("schema") != "retained-asr-model-lock-v1" or code.get("schema") != "retained-asr-source-lock-v1":
            raise ValueError("unsupported retained model/source lock")
        _equal_fields(model, identity, ("provider_id", "model_id", "model_revision", "model_family", "model_sha256"), "model lock")
        for lock, field in ((model, "assets"), (code, "files")):
            items = lock.get(field)
            if not isinstance(items, list) or not items:
                raise ValueError("retained lock must list measured source/model assets")
            paths = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("invalid retained lock item")
                path = _relative(item.get("path"))
                if path in paths or type(item.get("bytes")) is not int or item["bytes"] < 1:
                    raise ValueError("duplicate/invalid retained lock item")
                paths.add(path)
                require_sha(item.get("sha256"), "retained lock item hash")
        if canonical_sha256(model["assets"]) != identity["model_sha256"]:
            raise ValueError("model identity must bind the actual loaded asset inventory")
        if code.get("executed_head_sha") != provenance.get("executed_head_sha"):
            raise ValueError("source lock differs from independently pinned executed source")
        for item in code["files"]:
            body = read({"archive": refs_row["source_lock"]["archive"], "member": item["path"]})
            if len(body) != item["bytes"] or _sha(body) != item["sha256"]:
                raise ValueError("executed source lock differs from retained source bytes")
        native_v2 = record["schema_version"] == 2
        if native_v2:
            from asr_native_contracts import validate_native_config
            validate_native_config(config)
            if (not _same(config, identity["native_decode_config"])
                    or canonical_sha256(config) != identity["native_decode_config_sha256"]
                    or config.get("source_lock_sha256") != _sha(code_raw)
                    or config.get("asset_lock_sha256") != _sha(model_raw)
                    or config.get("model_family") != identity["model_family"]):
                raise ValueError("native config differs from retained model/source/config identity")
            effective = config["effective_config"]
            dependencies = refs_row["native_dependencies"]
            required = {key for key in effective if key.endswith("_sha256")}
            if not isinstance(dependencies, dict) or set(dependencies) != required:
                raise ValueError("native effective hashes require exact retained dependency references")
            for key, ref in dependencies.items():
                body = read(ref)
                if _sha(body) != effective[key]:
                    raise ValueError("native effective dependency differs from retained bytes")
                if key == "resolved_generation_config_sha256":
                    generation = _json(body, "resolved runtime GenerationConfig")
                    # Actual runtime snapshot, not request defaults. Other
                    # native GenerationConfig keys remain preserved verbatim.
                    for field in ("num_beams", "num_return_sequences", "forced_bos_token_id",
                                  "forced_eos_token_id", "suppress_tokens", "begin_suppress_tokens",
                                  "bad_words_ids", "force_words_ids", "sequence_bias", "stop_strings"):
                        _equal_fields(generation, effective, (field,), "resolved runtime GenerationConfig")
                    _equal_fields(generation, config["native_request"],
                                  ("do_sample", "max_new_tokens", "eos_token_id"), "resolved runtime GenerationConfig")
                else:
                    # The recorded tokenizer/parser/frontend must also occur
                    # in the actually loaded asset/source inventory. Full
                    # weights need not be copied into this bounded bundle.
                    matches = [item for item in model["assets"] + code["files"]
                               if item["path"] == ref["member"] and item["sha256"] == _sha(body)
                               and item["bytes"] == len(body)]
                    if len(matches) != 1:
                        raise ValueError("native dependency is not a loaded asset/executed source binding")
        else:
            # Legacy generic-v1 captures are separate. Never map auto-language
            # Qwen, greedy SenseVoice or Whisper into manufactured v1 defaults.
            if (config.get("schema") != "retained-asr-native-six-field-config-v1"
                    or config.get("adapter_id") != "explicit-six-field-v1"
                    or not _same(config.get("parameters"), identity["decode_config"])
                    or _sha(config_raw) != identity["native_decode_config_sha256"]):
                raise ValueError("unsupported/unbound native decoding configuration; no defaults may be invented")
        started_raw, receipt_raw, output_raw = (read(refs_row[key]) for key in ("started", "receipt", "raw_output"))
        started, receipt, output = (_json(raw, label) for raw, label in
                                   ((started_raw, "started"), (receipt_raw, "receipt"), (output_raw, "raw output")))
        if (started.get("schema") != "retained-asr-started-v1"
                or receipt.get("schema") != "retained-asr-completed-v1"
                or output.get("schema") != "retained-asr-raw-transcript-v1"):
            raise ValueError("unsupported original ASR capture format")
        common = {key: observation[key] for key in ("source_id", "file_sha256", "pcm_sha256", "audio_frames", "request_id")}
        common.update(recognizer_id=name, recognizer_identity_sha256=canonical_sha256(identity),
                      recognizer_policy_sha256=record["recognizer_policy_sha256"],
                      model_lock_sha256=_sha(model_raw), source_lock_sha256=_sha(code_raw),
                      native_decode_config_sha256=identity["native_decode_config_sha256"])
        _equal_fields(started, common, common, "original ASR start")
        _equal_fields(receipt, common, common, "original ASR completion")
        if (receipt.get("started_sha256") != _sha(started_raw)
                or receipt.get("raw_output_sha256") != _sha(output_raw)
                or observation["raw_output_sha256"] != _sha(output_raw)
                or receipt.get("status") != "success" or receipt.get("decoder_completed") is not True
                or type(receipt.get("processed_start_frame")) is not int or receipt["processed_start_frame"] != 0
                or type(receipt.get("processed_end_frame")) is not int or receipt["processed_end_frame"] != record["audio_frames"]
                or not _same(receipt.get("loaded_model_assets"), model["assets"])
                or not _same(receipt.get("executed_source_files"), code["files"])):
            raise ValueError("original ASR execution does not bind complete input/output/model/source evidence")
        if native_v2:
            from asr_native_contracts import validate_native_capture
            capture = receipt.get("native_capture")
            validate_native_capture(config, capture)
            _equal_fields(capture["input"], {"audio_frames": record["audio_frames"],
                          "processed_start_frame": 0, "processed_end_frame": record["audio_frames"]},
                          ("audio_frames", "processed_start_frame", "processed_end_frame"), "native capture input")
        _equal_fields(output, observation, ("request_id", "actual_text"), "complete raw transcript")
        if output.get("output_scope") != "complete-decoder-transcript" or output.get("truncated") is not False:
            raise ValueError("partial/unknown raw decoder output is not consensus evidence")
    return {"policy": RESOLVER_POLICY, "source_root_id": root.root_id,
            "trust_root_commit": root.trust_root_commit, "provenance_sha256": root.provenance_sha256,
            "executed_head_sha": provenance.get("executed_head_sha"), "run_id": provenance.get("run_id"),
            "record_sha256": canonical_sha256(record), "source_identity": measured,
            "bound_members": [{"archive": name, "member": member, "sha256": sha}
                              for (name, member), sha in sorted(used.items())]}


def resolve_source_context(record: dict, *, evidence_dir=None, session=None) -> VerifiedSourceContext:
    if session is not None and (type(session) is not _VerificationSession or not session.active):
        raise ValueError("invalid or closed source verification session")
    store = evidence_location(session.location if session is not None else evidence_dir)
    proof = _resolve(record, store, session)
    context = VerifiedSourceContext(_TOKEN)
    _CONTEXTS[context] = (store, proof, session)
    return context


def verify_source_context(context: object, record: dict) -> dict:
    if type(context) is not VerifiedSourceContext or context not in _CONTEXTS:
        raise ValueError("PENDING_EXTERNAL_VERIFICATION: caller JSON is not a verified source context")
    store, prior, session = _CONTEXTS[context]
    if prior["record_sha256"] != canonical_sha256(record):
        raise ValueError("verified source context belongs to a different record")
    if session is not None and not session.active:
        raise ValueError("verified source context belongs to a closed batch")
    current = _resolve(record, store, session)
    if not _same(current, prior):
        raise ValueError("retained evidence changed since source verification")
    return {**current, "proof_sha256": canonical_sha256(current)}
