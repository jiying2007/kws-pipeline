"""Invented metadata only; no audio, real capture, authentication, or inference."""
import copy

from asr_native_contracts import (ADAPTER_FAMILIES, CAPTURE_CLASS, CONDITIONING_POLICY,
    CONFIG_CLASS, EXECUTION, QWEN_ADAPTER, REQUESTS, SENSE_ADAPTER, WHISPER_ADAPTER,
    canonical_sha256)


def native_fixture(adapter, audio_frames=160):
    """Build structurally valid *invented* metadata through the ordinary schema."""
    effective = {"execution": copy.deepcopy(EXECUTION),
                 "native_request": copy.deepcopy(REQUESTS[adapter])}
    if adapter == QWEN_ADAPTER:
        effective.update(generation_mode="greedy_search", num_beams=1, num_return_sequences=1,
            forced_bos_token_id=None, forced_eos_token_id=None, suppress_tokens=None,
            begin_suppress_tokens=None, bad_words_ids=None, force_words_ids=None,
            sequence_bias=None, stop_strings=None, custom_logits_processors=[],
            custom_stopping_criteria=[], resolved_generation_config_sha256="3" * 64)
        termination = {"kind": "autoregressive_eos", "capture_stage": "returned_generation_sequence",
            "generated_token_ids": [17, 18, 151645], "generated_tokens": 3,
            "stop_reason": "eos_token", "truncated": False}
    elif adapter == SENSE_ADAPTER:
        effective.update(decoding="full_input_greedy_ctc", vad_used=False,
            rich_postprocess_applied=False, vocabulary_size=25055, lexical_unknown_id=0,
            metadata_prefix_tokens=4, tokenizer_sha256="3" * 64,
            parser_contract_sha256="4" * 64, frontend_config_sha256="5" * 64)
        termination = {"kind": "full_input_ctc_return", "inference_returned": True,
            "token_scope": "ctc_collapsed_nonblank_ids_not_framewise_logits",
            "raw_token_ids": [25001, 25002, 25003, 25004, 17, 18],
            "metadata_token_ids": [25001, 25002, 25003, 25004], "lexical_token_ids": [17, 18],
            "metadata_boundary_verified": True, "raw_redecode_matches_capture": True,
            "truncated": False}
    elif adapter == WHISPER_ADAPTER:
        effective.update(decoder="greedy", n_text_ctx=448, effective_sample_len=224,
            tokenizer_sha256="3" * 64, non_speech_token_ids=[200, 201],
            control_token_ids={"transcribe": 101, "translate": 102, "sot": 103,
                "sot_prev": 104, "sot_lm": 105, "no_speech": 106, "language": 107,
                "no_timestamps": 108},
            resolved_suppress_token_ids=[101, 102, 103, 104, 105, 106, 200, 201],
            space_token_ids=[20], initial_blank_suppress_token_ids=[20, 100],
            eot_token_id=100, initial_token_ids=[103, 107, 101, 108])
        termination = {"kind": "whisper_greedy_pre_finalize",
            "capture_stage": "decoder_update_before_finalize", "initial_token_ids": [103, 107, 101, 108],
            "generated_token_ids": [17, 18, 100], "step_count": 3, "pre_finalize_token_count": 7,
            "completed": True, "stop_reason": "eot", "truncated": False}
    else:
        raise ValueError("unknown invented adapter")
    preprocessing = "zero_pad_only_480000" if adapter == WHISPER_ADAPTER else "full_input"
    config = {"schema_version": 2, "evidence_class": CONFIG_CLASS,
        "adapter_id": adapter, "adapter_version": 1, "model_family": ADAPTER_FAMILIES[adapter],
        "source_lock_sha256": "1" * 64, "asset_lock_sha256": "2" * 64,
        "conditioning_policy": CONDITIONING_POLICY, "execution": copy.deepcopy(EXECUTION),
        "input_policy": {"sample_rate": 16000, "channels": 1, "max_audio_frames": 192000,
            "pcm_format": "float32_pcm16_scale_32768", "preprocessing": preprocessing},
        "native_request": copy.deepcopy(REQUESTS[adapter]), "effective_config": effective,
        "effective_config_sha256": canonical_sha256(effective)}
    capture = {"schema_version": 2, "evidence_class": CAPTURE_CLASS,
        "native_config_sha256": canonical_sha256(config), "source_lock_sha256": "1" * 64,
        "asset_lock_sha256": "2" * 64, "effective_config": copy.deepcopy(effective),
        "effective_config_sha256": config["effective_config_sha256"],
        "input": {"sample_rate": 16000, "channels": 1, "audio_frames": audio_frames,
            "processed_start_frame": 0, "processed_end_frame": audio_frames,
            "padding_frames": 480000 - audio_frames if adapter == WHISPER_ADAPTER else 0,
            "trimmed_frames": 0, "pcm_format": "float32_pcm16_scale_32768",
            "preprocessing": preprocessing},
        "termination": termination, "transcript_postcorrection_applied": False,
        "target_context_received": False, "peer_transcript_received": False,
        "acoustic_verbatim_completeness": "unknown"}
    return config, capture


def rebind_native_fixture(config, capture):
    """Rehash invented mutations; this is not an authentication operation."""
    config["effective_config_sha256"] = canonical_sha256(config["effective_config"])
    capture["native_config_sha256"] = canonical_sha256(config)
    capture["effective_config"] = copy.deepcopy(config["effective_config"])
    capture["effective_config_sha256"] = config["effective_config_sha256"]
    return config, capture
