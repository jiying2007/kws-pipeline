"""K1 ASR18 native receipts with exclusive durable once-only claim boundaries.

This module never loads a model, invokes a recognizer or grants execution. A
start is consumed even on uncertain/partial persistence; all failures poison the
binding. A matching capture proves consistency, not call authenticity: genuine
fixed native call sites and trusted source/runtime custody are also required.

Host reconciliation MUST treat any slot directory or uncertain start fault as
conservatively consumed even when started.json is absent. A valid-looking
completed.json can remain after a failed fsync/close: without the worker's
successful durable-completion return, the failed current slot stays
failed_no_retry. Preserve its evidence; never promote by existence alone. Any
such fault stops every later eligible slot, including other families.
"""
from dataclasses import dataclass, field
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import wave

from campaign_contract import (canonical, digest, require, read_bytes, safe_path,
    decode_json, write_once, durable_mkdir, validate_input_binding, validate_start_slot)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"

def _check_import_origin(name, expected):
    loaded = sys.modules.get(name)
    if loaded is not None:
        require(Path(getattr(loaded, "__file__", "")).absolute() == expected.absolute(),
                "foreign retained module import prohibited")
    spec = importlib.util.find_spec(name)
    require(spec is not None and spec.origin is not None and Path(spec.origin).absolute() == expected.absolute(),
            "explicit retained module source required")

# Only exact, staged pure validators. Never add the historical campaign folder.
sys.path.insert(0, str(TOOLS))
for _name in ("asr_native_contracts", "speech_label_admission", "speech_asr_consensus"):
    _check_import_origin(_name, TOOLS / (_name + ".py"))
from asr_native_contracts import (QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER,
                                  validate_native_capture, validate_native_config)
from speech_asr_consensus import (NORMALIZATION, POLICY_CLASS_V2,
                                  validate_recognizer_policy)

FILES = ("native-source-lock.json", "native-asset-lock.json", "native-config.json")
FAMILIES = ("qwen06", "sensevoice", "whisper")
ADAPTERS = dict(zip(FAMILIES, (QWEN_ADAPTER, SENSE_ADAPTER, WHISPER_ADAPTER)))
IDENTITY = ("provider_id", "model_id", "model_revision", "model_family", "model_sha256")
MAX_BYTES = 8 * 1024 ** 2
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _json(body):
    value = decode_json(body)
    require(type(value) is dict, "canonical JSON object required")
    return value


def _sha(value):
    require(type(value) is str and SHA.fullmatch(value), "lowercase SHA256 required")
    return value


def _read(path, *, readonly=False):
    return read_bytes(path, cap=MAX_BYTES, readonly=readonly)


def _write(path, value):
    body = canonical(value)
    require(len(body) <= MAX_BYTES, "receipt output bound")
    result = write_once(path, value)
    return {"path": Path(path).name, "bytes": len(body), "sha256": result}


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
    recognizer_id: str
    policy_path: Path
    policy_bytes: bytes
    frozen_dir: Path
    current_dir: Path
    prepared_bytes: tuple
    run_id: str
    executed_head_sha: str
    job_bytes: bytes
    execution_plan_bytes: bytes
    input_binding_bytes: bytes
    # Mutable process-local state is not an authority and cannot resume a run.
    _state: dict = field(default_factory=lambda: {"stopped": False, "next": 0, "active": None,
                                                "consumed": set()}, repr=False, compare=False)


def bind_predecode(*, recognizer_id, policy_path, expected_policy_sha256, frozen_dir,
                   current_dir, run_id, executed_head_sha, job, execution_plan, input_binding):
    require(type(run_id) is str and re.fullmatch(r"[1-9][0-9]{0,19}", run_id), "actual run ID required")
    require(type(executed_head_sha) is str and re.fullmatch(r"[0-9a-f]{40}", executed_head_sha),
            "actual executed source identity required")
    validate_input_binding(input_binding, job, execution_plan)
    body = _read(policy_path, readonly=True)
    require(digest(body) == _sha(expected_policy_sha256), "full-policy hash drift")
    identities = _policy(body)
    frozen = _prepared(frozen_dir, recognizer_id, readonly=True)
    current = _prepared(current_dir, recognizer_id)
    require(current[0] == frozen[0], "fresh current config/locks differ from frozen preparation")
    require(canonical(current[4]) == canonical(identities[recognizer_id]), "own identity differs from full policy")
    require(current[1]["executed_head_sha"] == executed_head_sha and
            current[1]["workflow_run_id"] == run_id, "actual run/source lineage drift")
    current_path = safe_path(current_dir)
    # Reject restart, even when the previous failure left only a partial directory.
    require(not any(path.name.startswith("clip-") for path in current_path.iterdir()),
            "prior or uncertain attempt directory prohibits restart")
    return Binding(recognizer_id, safe_path(policy_path), body, safe_path(frozen_dir), current_path,
        frozen[0], run_id, executed_head_sha, canonical(job), canonical(execution_plan), canonical(input_binding))


def _unchanged(binding):
    require(type(binding) is Binding and not binding._state["stopped"], "live predecode binding required")
    validate_input_binding(_json(binding.input_binding_bytes), _json(binding.job_bytes),
                           _json(binding.execution_plan_bytes))
    require(_read(binding.policy_path, readonly=True) == binding.policy_bytes, "full-policy bytes drift")
    identities = _policy(binding.policy_bytes)
    for directory, readonly in ((binding.frozen_dir, True), (binding.current_dir, False)):
        require(tuple(_read(directory / name, readonly=readonly) for name in FILES) == binding.prepared_bytes,
                "current/frozen configuration or locks drift")
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
            require(0 < frames < 192000 and len(pcm) == 2 * frames, "complete strictly bounded PCM frames required")
    except (wave.Error, EOFError) as exc:
        raise ValueError("malformed source WAV") from exc
    return {"file_sha256": digest(body), "pcm_sha256": digest(pcm), "audio_frames": frames}


@dataclass(frozen=True)
class Attempt:
    binding: Binding
    directory: Path
    wav_path: Path
    started_bytes: bytes
    _state: dict = field(default_factory=lambda: {"native_entered": False, "native_returned": False,
        "completion_attempted": False, "completed": False}, repr=False, compare=False)


def emit_start(binding, directory, *, wav_path, expected_file_sha256, request_id, audio_id, source_id=None):
    """Only a full canonical file + filefsync + parentfsync returns a claim.

    No writes are rolled back. The target is fixed to current_dir/audio_id, so a
    different directory cannot relabel the same slot within a source workspace.
    """
    require(type(binding) is Binding, "predecode binding required")
    try:
        identity = _unchanged(binding)
        require(binding._state["active"] is None, "previous native attempt must complete first")
        plan = _json(binding.execution_plan_bytes)
        index = binding._state["next"]
        require(index < len(plan["audio_order"]) and audio_id == plan["audio_order"][index],
                "fixed eligible opaque order; no retry or refill")
        measured = _wave(wav_path, expected_file_sha256)
        source_id = "k1asr18-" + measured["file_sha256"] if source_id is None else source_id
        validate_start_slot(_json(binding.job_bytes), plan, _json(binding.input_binding_bytes),
            binding.recognizer_id, audio_id=audio_id, measured=measured, source_id=source_id,
            request_id=request_id, run_id=binding.run_id)
        require(audio_id not in binding._state["consumed"], "attempt already consumed")
        directory = safe_path(directory)
        require(directory == binding.current_dir / audio_id, "fixed per-slot output directory required")
        durable_mkdir(directory)
        source_raw, model_raw, config_raw = binding.prepared_bytes
        anonymous = _json(binding.input_binding_bytes)
        started = {"schema": "retained-asr-started-v1", "source_id": source_id, **measured,
            "audio_id": audio_id, "job_sha256": digest(binding.job_bytes),
            "execution_plan_sha256": digest(binding.execution_plan_bytes),
            "worker_input_binding_sha256": digest(binding.input_binding_bytes),
            "screening_receipt_sha256": anonymous["screening_receipt_sha256"],
            "tts_authority_sha256": anonymous["tts_authority_sha256"],
            "request_id": request_id, "recognizer_id": binding.recognizer_id,
            "recognizer_identity_sha256": digest(canonical(identity)),
            "recognizer_policy_sha256": digest(binding.policy_bytes), "model_lock_sha256": digest(model_raw),
            "source_lock_sha256": digest(source_raw), "native_decode_config_sha256": digest(config_raw),
            "run_id": binding.run_id, "run_attempt": 1, "executed_head_sha": binding.executed_head_sha}
        # Conservatively consume before exclusive create: any uncertain I/O is terminal.
        binding._state["consumed"].add(audio_id)
        _write(directory / "started.json", started)
        attempt = Attempt(binding, directory, safe_path(wav_path), canonical(started))
        binding._state["active"] = attempt
        return attempt
    except BaseException:
        binding._state["stopped"] = True
        raise


def _active(attempt):
    require(type(attempt) is Attempt and attempt.binding._state["active"] is attempt and
            not attempt.binding._state["stopped"] and not attempt._state["completed"],
            "exact live first attempt required")


def mark_native_enter(attempt):
    """Fixed worker calls this immediately before its single actual native call."""
    try:
        _active(attempt)
        require(not attempt._state["native_entered"], "native invocation already consumed")
        _unchanged(attempt.binding)
        require(_read(attempt.directory / "started.json", readonly=True) == attempt.started_bytes,
                "durable canonical start changed before native call")
        started = _json(attempt.started_bytes)
        require(_wave(attempt.wav_path, started["file_sha256"]) == {key: started[key] for key in
            ("file_sha256", "pcm_sha256", "audio_frames")}, "audio changed before native call")
        attempt._state["native_entered"] = True
    except BaseException:
        fail_attempt(attempt)
        raise


def mark_native_return(attempt):
    try:
        _active(attempt)
        require(attempt._state["native_entered"] and not attempt._state["native_returned"],
                "single actual native return required")
        attempt._state["native_returned"] = True
    except BaseException:
        fail_attempt(attempt)
        raise


def fail_attempt(attempt):
    if type(attempt) is Attempt:
        attempt.binding._state["stopped"] = True


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
        _check_import_origin("asr_stage.decoding", ROOT / "research/qwen6_asr/core/asr_stage/decoding.py")
        from asr_stage.decoding import qwen_transcript
        require(canonical(raw.get("token_ids")) == canonical(capture["termination"]["generated_token_ids"]),
                "Qwen raw tokens differ from native generation capture")
        expected, metadata = qwen_transcript(raw.get("decoded_skip_special_tokens"), raw.get("token_ids"))
        require(canonical(row) == canonical(expected) and canonical(raw.get("metadata")) == canonical(metadata),
                "Qwen returned row/raw transcript mismatch")
    else:
        _check_import_origin("sense_parser", ROOT / "research/qwen6_asr/sense_parser.py")
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


def _emit_completion(attempt, *, row, raw):
    """Retain raw diagnostics first; no success until unchanged native validation.

    raw-transcript.json binds unmodified native JSON and returned row by their
    actual retained byte hashes. Whisper's existing raw-output.json is checked
    byte-for-byte and never overwritten. A failure consumes this completion.
    """
    _active(attempt)
    require(attempt._state["native_returned"] and not attempt._state["completion_attempted"],
            "single matching native return required before completion")
    attempt._state["completion_attempted"] = True
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
    attempt._state["completed"] = True
    binding._state["active"] = None
    binding._state["next"] += 1
    return receipt


def emit_completion(attempt, *, row, raw):
    try:
        return _emit_completion(attempt, row=row, raw=raw)
    except BaseException:
        fail_attempt(attempt)
        raise
