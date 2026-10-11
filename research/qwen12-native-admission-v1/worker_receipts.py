"""Prospective original native receipts. No model calls or admission authority.

Host order: three prepare-only processes; build_policy; freeze full policy and
own prepared files read-only; fresh-process load; bind_predecode; emit_start
immediately before ONE actual model call; emit_completion on its actual return.
Exceptions/incomplete returns consume that start, never authorize a retry. The
host owns cross-process one-use accounting, read-only mounts and source/runtime
verification. This helper binds observations; it cannot prove a call occurred.
"""
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from asr_native_contracts import (QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER,
                                  validate_native_capture, validate_native_config)
from speech_asr_consensus import (NORMALIZATION, POLICY_CLASS_V2,
                                  validate_recognizer_policy)

from campaign_contract import validate_original_job, remaining_plan, validate_start_slot

FILES = ("native-source-lock.json", "native-asset-lock.json", "native-config.json")
FAMILIES = ("qwen06", "sensevoice", "whisper")
ADAPTERS = dict(zip(FAMILIES, (QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER)))
IDENTITY = ("provider_id", "model_id", "model_revision", "model_family", "model_sha256")
MAX_BYTES = 8 * 1024 ** 2
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(body):
    return hashlib.sha256(body).hexdigest()


def _json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(body, object_pairs_hook=unique)
    require(type(value) is dict and canonical(value) == body, "canonical JSON object required")
    return value


def _sha(value):
    require(type(value) is str and SHA.fullmatch(value), "lowercase SHA256 required")
    return value


def _safe_path(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "aliased path prohibited")
    return path


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read(path, *, readonly=False):
    path = _safe_path(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_BYTES,
            "bounded unaliased regular file required")
    if readonly:
        require(not before.st_mode & 0o222 or bool(os.statvfs(path).f_flag & os.ST_RDONLY),
                "read-only frozen file required")
    with path.open("rb") as stream:
        require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "file changed before read")
        body = stream.read(MAX_BYTES + 1)
        require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "file changed during read")
    require(_stamp(path.stat()) == _stamp(before) and len(body) == before.st_size, "file changed after read")
    return body


def _write(path, value):
    path = _safe_path(path)
    body = canonical(value)
    require(len(body) <= MAX_BYTES, "receipt output bound")
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    require(_read(path, readonly=True) == body, "retained output drift")
    return {"path": path.name, "bytes": len(body), "sha256": digest(body)}


def _inventory(rows):
    require(type(rows) is list and rows, "nonempty measured inventory required")
    seen = set()
    for row in rows:
        require(type(row) is dict, "inventory row required")
        name = row.get("path")
        require(type(name) is str and name and "\\" not in name and "\0" not in name and
                not PurePosixPath(name).is_absolute() and
                all(part not in ("", ".", "..") for part in name.split("/")) and name not in seen,
                "unique relative inventory path required")
        seen.add(name)
        require(type(row.get("bytes")) is int and row["bytes"] > 0, "measured inventory length required")
        _sha(row.get("sha256"))


def _prepared(directory, recognizer_id, *, readonly=False):
    require(recognizer_id in FAMILIES, "fixed recognizer ID required")
    bodies = tuple(_read(Path(directory) / name, readonly=readonly) for name in FILES)
    source, model, config = map(_json, bodies)
    require(source.get("schema") == "retained-asr-source-lock-v1" and
            model.get("schema") == "retained-asr-model-lock-v1", "resolver-compatible locks required")
    _inventory(source.get("files")); _inventory(model.get("assets"))
    require(type(source.get("executed_head_sha")) is str and
            re.fullmatch(r"[0-9a-f]{40}", source["executed_head_sha"]), "actual executed source head required")
    require(type(source.get("workflow_run_id")) is str and
            re.fullmatch(r"[1-9][0-9]*", source["workflow_run_id"]), "actual source workflow run required")
    validate_native_config(config)
    require(config["adapter_id"] == ADAPTERS[recognizer_id] and
            config["source_lock_sha256"] == digest(bodies[0]) and
            config["asset_lock_sha256"] == digest(bodies[1]), "native configuration/lock identity drift")
    require(model.get("model_sha256") == digest(canonical(model["assets"])) and
            model.get("model_family") == config["model_family"], "loaded model inventory identity drift")
    identity = {"recognizer_id": recognizer_id, **{key: model[key] for key in IDENTITY},
                "native_decode_config": config, "native_decode_config_sha256": digest(bodies[2])}
    return bodies, source, model, config, identity


def _policy(body):
    policy = _json(body)
    require(set(policy) == {"schema_version", "evidence_class", "normalization", "frozen_before_recognition",
                           "target_conditioning_allowed", "recognizers"} and
            policy.get("evidence_class") == POLICY_CLASS_V2, "public full native-v2 policy required")
    identities = validate_recognizer_policy(policy)
    require(set(identities) == set(FAMILIES), "full three-recognizer policy required")
    for family, identity in identities.items():
        require(set(identity) == {"recognizer_id", *IDENTITY, "native_decode_config", "native_decode_config_sha256"}
                and identity["native_decode_config"]["adapter_id"] == ADAPTERS[family],
                "public recognizer identity contains unsupported fields/family")
    return identities


def build_policy(prepared):
    """Build the full public policy after all three prepare-only loads, no calls.

    prepared maps qwen06/sensevoice/whisper to their prepared output directories.
    Host exclusively retains canonical(result) then read-only freezes it before
    starting any decode process. No target/source labels enter this function.
    """
    require(type(prepared) is dict and set(prepared) == set(FAMILIES), "all three prepared recognizers required")
    rows = [_prepared(prepared[family], family) for family in FAMILIES]
    require(len({(row[1]["executed_head_sha"], row[1]["workflow_run_id"]) for row in rows}) == 1,
            "prepared recognizers belong to different runs/source heads")
    policy = {"schema_version": 2, "evidence_class": POLICY_CLASS_V2, "normalization": NORMALIZATION,
              "frozen_before_recognition": True, "target_conditioning_allowed": False,
              "recognizers": [row[4] for row in rows]}
    _policy(canonical(policy))
    return policy


@dataclass(frozen=True)
class Binding:
    """Immutable byte identities plus a private process-local one-attempt ledger."""
    recognizer_id: str
    policy_path: Path
    policy_bytes: bytes
    frozen_dir: Path
    current_dir: Path
    prepared_bytes: tuple
    run_id: str
    executed_head_sha: str
    original_job_bytes: bytes
    remaining_plan_bytes: bytes
    _requests: set = field(default_factory=set, repr=False, compare=False)
    _sources: set = field(default_factory=set, repr=False, compare=False)


def bind_predecode(*, recognizer_id, policy_path, expected_policy_sha256, frozen_dir,
                   current_dir, run_id, executed_head_sha, original_job):
    """Before any call, compare fresh-load bytes to read-only prepared copies."""
    require(type(run_id) is str and re.fullmatch(r"[1-9][0-9]*", run_id), "actual run ID required")
    body = _read(policy_path, readonly=True)
    require(digest(body) == _sha(expected_policy_sha256), "full-policy hash drift")
    identities = _policy(body)
    frozen = _prepared(frozen_dir, recognizer_id, readonly=True)
    current = _prepared(current_dir, recognizer_id)
    require(current[0] == frozen[0], "fresh current config/locks differ from frozen preparation")
    require(canonical(current[4]) == canonical(identities[recognizer_id]), "own identity differs from full policy")
    require(current[1]["executed_head_sha"] == executed_head_sha and
            current[1]["workflow_run_id"] == run_id, "actual run/source lineage drift")
    job_bytes = canonical(validate_original_job(original_job))
    plan_bytes = canonical(remaining_plan(original_job, recognizer_id))
    return Binding(recognizer_id, Path(policy_path), body, Path(frozen_dir), Path(current_dir),
                   frozen[0], run_id, executed_head_sha, job_bytes, plan_bytes)


def _unchanged(binding):
    require(type(binding) is Binding, "predecode binding required")
    job = _json(binding.original_job_bytes)
    validate_original_job(job)
    require(canonical(remaining_plan(job, binding.recognizer_id)) == binding.remaining_plan_bytes,
            "fixed original job/remaining plan binding drift")
    require(_read(binding.policy_path, readonly=True) == binding.policy_bytes, "full-policy bytes drift")
    identities = _policy(binding.policy_bytes)
    for directory, readonly in ((binding.frozen_dir, True), (binding.current_dir, False)):
        bodies = tuple(_read(directory / name, readonly=readonly) for name in FILES)
        require(bodies == binding.prepared_bytes, "current/frozen configuration or locks drift")
    return identities[binding.recognizer_id]


def _wave(path, expected_sha):
    body = _read(path, readonly=True)
    require(digest(body) == _sha(expected_sha), "source WAV hash drift")
    try:
        with wave.open(io.BytesIO(body), "rb") as audio:
            require(audio.getnchannels() == 1 and audio.getframerate() == 16000 and
                    audio.getsampwidth() == 2 and audio.getcomptype() == "NONE", "mono PCM16 16k WAV required")
            frames = audio.getnframes()
            pcm = audio.readframes(frames + 1)
            require(0 < frames <= 192000 and len(pcm) == 2 * frames, "complete bounded PCM frames required")
    except (wave.Error, EOFError) as exc:
        raise ValueError("malformed source WAV") from exc
    return {"file_sha256": digest(body), "pcm_sha256": digest(pcm), "audio_frames": frames}


@dataclass(frozen=True)
class Attempt:
    binding: Binding
    directory: Path
    wav_path: Path
    started_bytes: bytes
    _completion_attempted: set = field(default_factory=set, repr=False, compare=False)


def emit_start(binding, directory, *, wav_path, expected_file_sha256, request_id, audio_id, source_id=None):
    """Exclusive FIRST-ATTEMPT start. Call immediately before the actual model."""
    identity = _unchanged(binding)
    measured = _wave(wav_path, expected_file_sha256)
    source_id = "qwen12-" + measured["file_sha256"] if source_id is None else source_id
    require(type(source_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", source_id),
            "opaque non-target source ID required")
    require(type(request_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", request_id),
            "opaque request ID required")
    validate_start_slot(_json(binding.original_job_bytes), binding.recognizer_id, audio_id=audio_id,
        measured=measured, source_id=source_id, request_id=request_id, run_id=binding.run_id)
    require(request_id not in binding._requests and measured["file_sha256"] not in binding._sources,
            "duplicate source/request attempt prohibited")
    directory = _safe_path(directory)
    require(directory.is_dir(), "existing per-call output directory required")
    require(not any((directory / name).exists() for name in ("started.json", "completed.json", "raw-transcript.json")),
            "existing attempt cannot be relabeled/retried")
    binding._requests.add(request_id); binding._sources.add(measured["file_sha256"])
    source_raw, model_raw, config_raw = binding.prepared_bytes
    started = {"schema": "retained-asr-started-v1", "source_id": source_id, **measured,
        "audio_id": audio_id, "original_job_sha256": digest(binding.original_job_bytes),
        "remaining_plan_sha256": digest(binding.remaining_plan_bytes),
        "request_id": request_id, "recognizer_id": binding.recognizer_id,
        "recognizer_identity_sha256": digest(canonical(identity)),
        "recognizer_policy_sha256": digest(binding.policy_bytes), "model_lock_sha256": digest(model_raw),
        "source_lock_sha256": digest(source_raw), "native_decode_config_sha256": digest(config_raw),
        "run_id": binding.run_id, "executed_head_sha": binding.executed_head_sha}
    _write(directory / "started.json", started)
    return Attempt(binding, directory, Path(wav_path), canonical(started))


def _text(value):
    require(type(value) is str and len(value) <= 32768 and
            not any(0xD800 <= ord(char) <= 0xDFFF for char in value), "bounded original transcript required")
    return value


def _actual_text(family, config, row, raw):
    """Check the unchanged native extraction against the actual returned row."""
    capture = raw["native_capture"]
    if family == "whisper":
        require(row is None and raw.get("decoder_completion_validated") is True and
                raw.get("full_input_processed") is True, "actual Whisper complete return required")
        require(canonical(raw.get("returned_token_ids")) == canonical(capture["termination"]["generated_token_ids"][:-1]),
                "Whisper raw returned tokens differ from observed pre-finalize tokens")
        return _text(raw.get("raw_text"))
    require(type(row) is dict and raw.get("native_v2_validated") is True, "native adapter rejected completion")
    if family == "qwen06":
        from asr_stage.decoding import qwen_transcript
        require(canonical(raw.get("token_ids")) == canonical(capture["termination"]["generated_token_ids"]),
                "Qwen raw tokens differ from native generation capture")
        expected, metadata = qwen_transcript(raw.get("decoded_skip_special_tokens"), raw.get("token_ids"))
        require(canonical(row) == canonical(expected) and canonical(raw.get("metadata")) == canonical(metadata),
                "Qwen returned row/raw transcript mismatch")
    else:
        from sense_parser import corrected_sense_extract
        result = raw.get("raw_result")
        require(type(result) in (list, tuple) and len(result) == 2 and type(result[0]) is list and
                len(result[0]) == 1 and type(result[0][0]) is dict, "actual SenseVoice single raw return required")
        decoded = _text(result[0][0].get("text"))
        evidence, token_binding = raw.get("token_evidence"), raw.get("token_binding")
        require(type(evidence) is dict and type(token_binding) is dict and evidence.get("decoded") == decoded,
                "SenseVoice raw decoder text association mismatch")
        termination = capture["termination"]
        for key in ("raw_token_ids", "metadata_token_ids", "lexical_token_ids"):
            require(canonical(token_binding.get(key)) == canonical(termination[key]), "SenseVoice native token binding mismatch")
        require(canonical(evidence.get("token_ids")) == canonical(termination["raw_token_ids"]) and
                token_binding.get("raw_decoded_sha256") == digest(decoded.encode("utf-8")) and
                token_binding.get("parser_contract_sha256") == config["effective_config"]["parser_contract_sha256"] and
                token_binding.get("metadata_boundary_verified") is True and
                token_binding.get("raw_redecode_matches_capture") is True, "SenseVoice raw text/token provenance mismatch")
        expected = corrected_sense_extract(decoded, termination["lexical_token_ids"])
        metadata = expected.pop("metadata")
        require(canonical(row) == canonical(expected) and canonical(raw.get("metadata")) == canonical(metadata),
                "SenseVoice returned row/raw transcript mismatch")
    return _text(row.get("raw_text"))


def emit_completion(attempt, *, row, raw):
    """Retain raw diagnostics first; no success until unchanged native validation.

    raw-transcript.json binds unmodified native JSON and returned row by their
    actual retained byte hashes. Whisper's existing raw-output.json is checked
    byte-for-byte and never overwritten. A failure consumes this completion.
    """
    require(type(attempt) is Attempt and not attempt._completion_attempted, "duplicate/invalid completion attempt")
    attempt._completion_attempted.add(True)
    directory, binding = attempt.directory, attempt.binding
    require(type(raw) is dict, "actual raw return object required")
    native_ref = _write(directory / "raw-native-output.json", raw)
    row_ref = _write(directory / "raw-native-row.json", row) if row is not None else None
    _unchanged(binding)
    require(_read(directory / "started.json", readonly=True) == attempt.started_bytes, "immutable started receipt drift")
    started = _json(attempt.started_bytes)
    require(_wave(attempt.wav_path, started["file_sha256"]) == {key: started[key] for key in
            ("file_sha256", "pcm_sha256", "audio_frames")}, "input WAV/PCM drift after call")
    config = _json(binding.prepared_bytes[2])
    capture = raw.get("native_capture")
    validate_native_capture(config, capture)
    require(capture["input"]["audio_frames"] == started["audio_frames"] and
            capture["input"]["processed_start_frame"] == 0 and
            capture["input"]["processed_end_frame"] == started["audio_frames"], "native full-source coverage mismatch")
    actual_text = _actual_text(binding.recognizer_id, config, row, raw)
    if binding.recognizer_id == "whisper":
        require(_read(directory / "raw-output.json") == canonical(raw), "original Whisper raw output mismatch")
    transcript = {"schema": "retained-asr-raw-transcript-v1", "request_id": started["request_id"],
        "actual_text": actual_text, "output_scope": "complete-decoder-transcript", "truncated": False,
        "native_output": native_ref, "native_row": row_ref, "acoustic_verbatim_completeness": "unknown"}
    output_ref = _write(directory / "raw-transcript.json", transcript)
    common = {key: value for key, value in started.items() if key != "schema"}
    receipt = {"schema": "retained-asr-completed-v1", **common,
        "started_sha256": digest(attempt.started_bytes), "raw_output_sha256": output_ref["sha256"],
        "raw_native_output_sha256": native_ref["sha256"], "status": "success", "decoder_completed": True,
        "processed_start_frame": 0, "processed_end_frame": started["audio_frames"],
        "loaded_model_assets": _json(binding.prepared_bytes[1])["assets"],
        "executed_source_files": _json(binding.prepared_bytes[0])["files"], "native_capture": capture,
        "acoustic_verbatim_completeness": "unknown"}
    _write(directory / "completed.json", receipt)
    return receipt
