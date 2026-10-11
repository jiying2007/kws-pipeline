"""Prospective native ASR contracts, standard-library metadata validation only.

These v2 contracts do not reinterpret historical six-field/v1 records. They do
not load models, create observations, authenticate captures, or grant training
admission. The caller must resolve source/asset locks and captured bytes through
its independently trusted archive. A matching hash or a plausible trace here
proves internal consistency, not that recognition actually happened.

``validate_native_config(config)`` validates a frozen, fully specified request
and expected effective summary. ``validate_native_capture(config, capture)``
validates an independently resolved ``native_capture`` object and compares its
observed effective summary to that frozen object. The caller must also bind
``capture['input']['audio_frames']`` to its independently measured source.
No missing, null, unknown, or unsupported value is replaced with a default.

The three separate scopes are full-source input processing, decoder completion
without truncation, and acoustic verbatim completeness. The last always remains
``unknown``: neither a natural EOS nor a returned full-input CTC call proves it.
Qwen records native auto-language/context/generation options. SenseVoice records
full-input greedy CTC and collapsed nonblank IDs, never fictitious EOS/logits.
Whisper requires a trusted greedy-update trace *before* finalize; its public
DecodingResult strips EOS, and finalize can append EOS to unfinished sequences.

Effective summaries must be measured from the resolved runtime/config/tokenizer,
not synthesized from this validator. In particular, inherited Qwen controls and
Whisper's resolved suppression token sets are mandatory observed fields. The
tokenizer/source/asset authenticity and full raw-output binding belong to the
caller; this module only checks the supplied metadata's exact semantics.
The caller must resolve Qwen's resolved_generation_config_sha256 to the retained
effective GenerationConfig snapshot, SenseVoice tokenizer/parser/frontend hashes
to retained asset/source/runtime-config bytes, and Whisper tokenizer_sha256 to
the retained tokenizer asset binding. A digest without those bytes is not ready
for authentication. Source/asset lock hashes are byte hashes of retained lock
files; config/effective hashes are canonical JSON hashes.
"""
from __future__ import annotations

import hashlib
import json

CONFIG_CLASS = "speech-asr-native-config-v2"
CAPTURE_CLASS = "speech-asr-native-capture-v2"
CONDITIONING_POLICY = "blind-no-target-or-peer-context-v1"
QWEN_ADAPTER = "qwen3-asr-transformers-greedy"
SENSE_ADAPTER = "sensevoice-full-input-greedy-ctc"
WHISPER_ADAPTER = "openai-whisper-decode-short"
ADAPTER_FAMILIES = {QWEN_ADAPTER: "qwen3-asr", SENSE_ADAPTER: "sensevoice",
                    WHISPER_ADAPTER: "whisper"}
EXECUTION = {"device": "cpu", "dtype": "float32", "batch_size": 1}
QWEN_REQUEST = {"context": "", "force_language": None, "max_new_tokens": 256,
                "do_sample": False, "eos_token_id": [151645, 151643]}
SENSE_REQUEST = {"device": "cpu", "language": "auto", "use_itn": False,
                 "text_norm": "woitn", "fs": 16000, "data_type": "sound",
                 "output_timestamp": False, "ban_emo_unk": False}
WHISPER_REQUEST = {"task": "transcribe", "language": "zh", "temperature": 0.0,
                   "sample_len": None, "beam_size": None, "best_of": None,
                   "patience": None, "length_penalty": None, "prompt": None,
                   "prefix": None, "suppress_tokens": "-1", "suppress_blank": True,
                   "without_timestamps": True, "max_initial_timestamp": None,
                   "fp16": False}
REQUESTS = {QWEN_ADAPTER: QWEN_REQUEST, SENSE_ADAPTER: SENSE_REQUEST,
            WHISPER_ADAPTER: WHISPER_REQUEST}
MAX_AUDIO_FRAMES = 16_000 * 60 * 60  # Metadata bound, not a tested runtime capacity.
MAX_TOKEN_IDS = 65_536


def _json(value: object) -> None:
    """Reject non-JSON Python types and NaN before any equality/hash comparison."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("native metadata keys must be strings")
        for item in value.values():
            _json(item)
    elif type(value) is list:
        for item in value:
            _json(item)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("native metadata must contain exact JSON types")
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("native metadata must contain finite valid JSON values") from exc


def canonical_sha256(value: object) -> str:
    _json(value)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _same(actual: object, expected: object, label: str) -> None:
    # Python considers False == 0 and True == 1; a native contract must not.
    if type(actual) is not type(expected):
        raise ValueError(f"{label} type mismatch")
    if type(expected) is dict:
        _fields(actual, set(expected), label)
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(f"{label} length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label} value mismatch")


def _fields(value: object, fields: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} has missing or unsupported fields")


def _sha(value: object, label: str) -> None:
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be lowercase SHA256")


def _int(value: object, label: str, low: int = 0, high: int = MAX_AUDIO_FRAMES) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be a bounded integer")


def _ids(value: object, label: str, *, empty: bool = False) -> None:
    if type(value) is not list or not (0 if empty else 1) <= len(value) <= MAX_TOKEN_IDS:
        raise ValueError(f"{label} must be a bounded token ID list")
    for token in value:
        _int(token, label, high=1_000_000)


def _sorted_ids(value: object, label: str, *, empty: bool = False) -> None:
    _ids(value, label, empty=empty)
    if value != sorted(set(value)):
        raise ValueError(f"{label} must be a sorted unique token ID set")


def _qwen_effective(value: dict) -> None:
    fixed = {"execution": EXECUTION, "native_request": QWEN_REQUEST,
        "generation_mode": "greedy_search", "num_beams": 1, "num_return_sequences": 1,
        "forced_bos_token_id": None, "forced_eos_token_id": None,
        "suppress_tokens": None, "begin_suppress_tokens": None, "bad_words_ids": None,
        "force_words_ids": None, "sequence_bias": None, "stop_strings": None,
        "custom_logits_processors": [], "custom_stopping_criteria": []}
    _fields(value, set(fixed) | {"resolved_generation_config_sha256"}, "Qwen effective config")
    for key, expected in fixed.items():
        _same(value[key], expected, f"Qwen effective.{key}")
    _sha(value["resolved_generation_config_sha256"], "Qwen resolved generation config")


def _sense_effective(value: dict) -> None:
    fixed = {"execution": EXECUTION, "native_request": SENSE_REQUEST,
        "decoding": "full_input_greedy_ctc", "vad_used": False,
        "rich_postprocess_applied": False, "vocabulary_size": 25055,
        "lexical_unknown_id": 0, "metadata_prefix_tokens": 4}
    hashes = {"tokenizer_sha256", "parser_contract_sha256", "frontend_config_sha256"}
    _fields(value, set(fixed) | hashes, "SenseVoice effective config")
    for key, expected in fixed.items():
        _same(value[key], expected, f"SenseVoice effective.{key}")
    for key in hashes:
        _sha(value[key], f"SenseVoice {key}")


def _whisper_effective(value: dict) -> None:
    _fields(value, {"execution", "native_request", "decoder", "n_text_ctx",
        "effective_sample_len", "tokenizer_sha256", "non_speech_token_ids",
        "control_token_ids", "resolved_suppress_token_ids", "space_token_ids",
        "initial_blank_suppress_token_ids", "eot_token_id", "initial_token_ids"},
        "Whisper effective config")
    _same(value["execution"], EXECUTION, "Whisper execution")
    _same(value["native_request"], WHISPER_REQUEST, "Whisper native request")
    _same(value["decoder"], "greedy", "Whisper decoder")
    _int(value["n_text_ctx"], "Whisper n_text_ctx", 8, 65_536)
    _int(value["effective_sample_len"], "Whisper effective_sample_len", 1, 32_768)
    if value["effective_sample_len"] != value["n_text_ctx"] // 2:
        raise ValueError("Whisper null sample_len must resolve to observed n_text_ctx // 2")
    _sha(value["tokenizer_sha256"], "Whisper tokenizer")
    _sorted_ids(value["non_speech_token_ids"], "Whisper non_speech tokens")
    controls = value["control_token_ids"]
    _fields(controls, {"transcribe", "translate", "sot", "sot_prev", "sot_lm",
                       "no_speech", "language", "no_timestamps"}, "Whisper tokenizer controls")
    for key, token in controls.items():
        if key != "no_speech" or token is not None:
            _int(token, f"Whisper {key} token", high=1_000_000)
    concrete = [token for token in controls.values() if token is not None]
    if len(concrete) != len(set(concrete)):
        raise ValueError("Whisper tokenizer control IDs must be distinct")
    _int(value["eot_token_id"], "Whisper EOT token", high=1_000_000)
    if value["eot_token_id"] in concrete:
        raise ValueError("Whisper EOT cannot alias a control token")
    expected_suppression = set(value["non_speech_token_ids"])
    expected_suppression.update(controls[key] for key in
        ("transcribe", "translate", "sot", "sot_prev", "sot_lm"))
    if controls["no_speech"] is not None:
        expected_suppression.add(controls["no_speech"])
    _sorted_ids(value["resolved_suppress_token_ids"], "Whisper resolved suppression")
    if value["resolved_suppress_token_ids"] != sorted(expected_suppression):
        raise ValueError("Whisper '-1' suppression differs from native tokenizer-derived set")
    if value["eot_token_id"] in expected_suppression:
        raise ValueError("Whisper permanent suppression cannot suppress EOT")
    _ids(value["space_token_ids"], "Whisper encoded space")
    _sorted_ids(value["initial_blank_suppress_token_ids"], "Whisper initial blank suppression")
    if value["initial_blank_suppress_token_ids"] != sorted(set(
            value["space_token_ids"] + [value["eot_token_id"]])):
        raise ValueError("Whisper suppress_blank must resolve separately for the initial step")
    _same(value["initial_token_ids"], [controls[key] for key in
        ("sot", "language", "transcribe", "no_timestamps")], "Whisper unconditioned initial tokens")


def validate_native_config(config: dict) -> None:
    """Validate one frozen native-v2 configuration; never fill in defaults."""
    _json(config)
    _fields(config, {"schema_version", "evidence_class", "adapter_id", "adapter_version",
        "model_family", "source_lock_sha256", "asset_lock_sha256", "conditioning_policy",
        "execution", "input_policy", "native_request", "effective_config",
        "effective_config_sha256"}, "native config")
    _same(config["schema_version"], 2, "native config schema_version")
    _same(config["evidence_class"], CONFIG_CLASS, "native config evidence_class")
    _same(config["adapter_version"], 1, "native adapter_version")
    adapter = config["adapter_id"]
    if type(adapter) is not str or adapter not in ADAPTER_FAMILIES:
        raise ValueError("unsupported native adapter")
    _same(config["model_family"], ADAPTER_FAMILIES[adapter], "native model family")
    for key in ("source_lock_sha256", "asset_lock_sha256", "effective_config_sha256"):
        _sha(config[key], key)
    _same(config["conditioning_policy"], CONDITIONING_POLICY, "native conditioning policy")
    _same(config["execution"], EXECUTION, "native execution")
    _same(config["native_request"], REQUESTS[adapter], "native request")
    input_policy = config["input_policy"]
    _fields(input_policy, {"sample_rate", "channels", "max_audio_frames", "pcm_format",
                           "preprocessing"}, "native input policy")
    _same(input_policy["sample_rate"], 16000, "input sample rate")
    _same(input_policy["channels"], 1, "input channels")
    _same(input_policy["pcm_format"], "float32_pcm16_scale_32768", "input PCM format")
    _int(input_policy["max_audio_frames"], "input frame limit", 1,
         480000 if adapter == WHISPER_ADAPTER else MAX_AUDIO_FRAMES)
    _same(input_policy["preprocessing"], "zero_pad_only_480000"
          if adapter == WHISPER_ADAPTER else "full_input", "input preprocessing")
    {QWEN_ADAPTER: _qwen_effective, SENSE_ADAPTER: _sense_effective,
     WHISPER_ADAPTER: _whisper_effective}[adapter](config["effective_config"])
    if canonical_sha256(config["effective_config"]) != config["effective_config_sha256"]:
        raise ValueError("native frozen effective config hash mismatch")


def _input(config: dict, value: dict) -> None:
    _fields(value, {"sample_rate", "channels", "audio_frames", "processed_start_frame",
        "processed_end_frame", "padding_frames", "trimmed_frames", "preprocessing",
        "pcm_format"}, "native captured input")
    for key in ("sample_rate", "channels", "preprocessing", "pcm_format"):
        _same(value[key], config["input_policy"][key], f"captured input.{key}")
    _int(value["audio_frames"], "captured source frames", 1,
         config["input_policy"]["max_audio_frames"])
    for key in ("processed_start_frame", "processed_end_frame", "padding_frames", "trimmed_frames"):
        _int(value[key], f"captured input.{key}")
    _same(value["processed_start_frame"], 0, "captured input start")
    _same(value["processed_end_frame"], value["audio_frames"], "captured full input end")
    _same(value["trimmed_frames"], 0, "captured input trimming")
    _same(value["padding_frames"], 480000 - value["audio_frames"]
          if config["adapter_id"] == WHISPER_ADAPTER else 0, "captured input padding")


def _qwen_termination(value: dict, effective: dict) -> None:
    _fields(value, {"kind", "capture_stage", "generated_token_ids", "generated_tokens",
        "stop_reason", "truncated"}, "Qwen termination")
    _same(value["kind"], "autoregressive_eos", "Qwen termination kind")
    _same(value["capture_stage"], "returned_generation_sequence", "Qwen capture stage")
    _same(value["stop_reason"], "eos_token", "Qwen stop reason")
    _same(value["truncated"], False, "Qwen truncated")
    ids = value["generated_token_ids"]
    _ids(ids, "Qwen generated tokens")
    _int(value["generated_tokens"], "Qwen generated_tokens", 1, 256)
    if value["generated_tokens"] != len(ids):
        raise ValueError("Qwen generated token count differs from captured IDs")
    eos = effective["native_request"]["eos_token_id"]
    if ids[-1] not in eos or any(token in eos for token in ids[:-1]):
        raise ValueError("Qwen missing natural terminal EOS or tokens after EOS")


def _sense_termination(value: dict, effective: dict) -> None:
    _fields(value, {"kind", "inference_returned", "token_scope", "raw_token_ids",
        "metadata_token_ids", "lexical_token_ids", "metadata_boundary_verified",
        "raw_redecode_matches_capture", "truncated"}, "SenseVoice termination")
    fixed = {"kind": "full_input_ctc_return", "inference_returned": True,
        "token_scope": "ctc_collapsed_nonblank_ids_not_framewise_logits",
        "metadata_boundary_verified": True, "raw_redecode_matches_capture": True,
        "truncated": False}
    for key, expected in fixed.items():
        _same(value[key], expected, f"SenseVoice termination.{key}")
    for key in ("raw_token_ids", "metadata_token_ids", "lexical_token_ids"):
        _ids(value[key], f"SenseVoice {key}")
    ids = value["raw_token_ids"]
    count = effective["metadata_prefix_tokens"]
    if (any(token >= effective["vocabulary_size"] for token in ids)
            or value["metadata_token_ids"] != ids[:count]
            or len(value["metadata_token_ids"]) != count
            or value["lexical_token_ids"] != ids[count:]
            or effective["lexical_unknown_id"] in value["lexical_token_ids"]):
        raise ValueError("SenseVoice captured metadata/lexical token boundary is invalid or unknown")


def _whisper_termination(value: dict, effective: dict) -> None:
    _fields(value, {"kind", "capture_stage", "initial_token_ids", "generated_token_ids",
        "step_count", "pre_finalize_token_count", "completed", "stop_reason", "truncated"},
        "Whisper termination")
    fixed = {"kind": "whisper_greedy_pre_finalize", "capture_stage": "decoder_update_before_finalize",
             "completed": True, "stop_reason": "eot", "truncated": False}
    for key, expected in fixed.items():
        _same(value[key], expected, f"Whisper termination.{key}")
    _same(value["initial_token_ids"], effective["initial_token_ids"], "Whisper initial token trace")
    ids = value["generated_token_ids"]
    _ids(ids, "Whisper generated pre-finalize tokens")
    _int(value["step_count"], "Whisper step count", 1, effective["effective_sample_len"])
    _int(value["pre_finalize_token_count"], "Whisper token count", 1, MAX_TOKEN_IDS)
    initial_count = len(value["initial_token_ids"])
    if value["step_count"] != len(ids) or value["pre_finalize_token_count"] != initial_count + len(ids):
        raise ValueError("Whisper pre-finalize token trace/count mismatch")
    eot = effective["eot_token_id"]
    if ids[-1] != eot or eot in ids[:-1]:
        raise ValueError("Whisper requires actual terminal EOT before finalize, with no later tokens")
    # _main_loop breaks after update when completed OR total tokens > n_ctx.
    # EOS on the last permitted step is legitimate; a returned/finalized EOS is not.
    if initial_count + len(ids) - 1 > effective["n_text_ctx"]:
        raise ValueError("Whisper trace continued after context-limit termination")
    if ids[0] in effective["initial_blank_suppress_token_ids"]:
        raise ValueError("Whisper first-step trace violates suppress_blank")
    suppressed = set(effective["resolved_suppress_token_ids"])
    if any(token in suppressed for token in ids):
        raise ValueError("Whisper token trace violates resolved native suppression")


def validate_native_capture(config: dict, capture: dict) -> None:
    """Validate a trusted archive's native_capture projection, not its authenticity."""
    validate_native_config(config)
    _json(capture)
    _fields(capture, {"schema_version", "evidence_class", "native_config_sha256",
        "source_lock_sha256", "asset_lock_sha256", "effective_config", "effective_config_sha256",
        "input", "termination", "transcript_postcorrection_applied", "target_context_received",
        "peer_transcript_received", "acoustic_verbatim_completeness"}, "native capture")
    fixed = {"schema_version": 2, "evidence_class": CAPTURE_CLASS,
        "native_config_sha256": canonical_sha256(config),
        "source_lock_sha256": config["source_lock_sha256"],
        "asset_lock_sha256": config["asset_lock_sha256"],
        "effective_config_sha256": config["effective_config_sha256"],
        "transcript_postcorrection_applied": False, "target_context_received": False,
        "peer_transcript_received": False, "acoustic_verbatim_completeness": "unknown"}
    for key, expected in fixed.items():
        _same(capture[key], expected, f"native capture.{key}")
    _same(capture["effective_config"], config["effective_config"], "captured effective config")
    if canonical_sha256(capture["effective_config"]) != capture["effective_config_sha256"]:
        raise ValueError("native captured effective config hash mismatch")
    _input(config, capture["input"])
    {QWEN_ADAPTER: _qwen_termination, SENSE_ADAPTER: _sense_termination,
     WHISPER_ADAPTER: _whisper_termination}[config["adapter_id"]](
         capture["termination"], capture["effective_config"])
