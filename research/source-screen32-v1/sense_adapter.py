"""Source-screen SenseVoice path plumbing; retained scientific semantics unchanged.

Only imported after worker scope/source checks. No launcher, downloader or model
construction. The copied binding differs only by an explicit tokenizer path;
retained qwen6 files and their old default-path behavior are not modified.
"""
from pathlib import Path
import hashlib
import stat

from sense_binding import (_contract, _hash, verify_tokenizer, TOKENIZER_NAME,
    CONTRACT_SHA256, TOKEN_SCOPE, VOCABULARY_SIZE, LEXICAL_UNKNOWN_ID)
from asr_stage.adapters import SENSE, _begin_clip
from asr_stage.decoding import CapturingTokenizer, bounded_text
from asr_stage.runtime_support import describe_cpu_tensor, verify_pcm_array
from vendor.checkpoint_guard.exact_apply import require_verified_receipt

STAGES = frozenset(("runtime_verification", "model_loading", "adapter_dispatch",
    "model_verification", "pcm_binding", "tokenizer_verification", "model_inference",
    "result_contract", "token_capture", "token_binding", "pcm_postcheck", "complete",
    "scope_verification", "input_binding", "adapter_result"))
ERROR_CODES = {
    "Invalid bounded decoder token IDs": "SENSE_TOKEN_IDS_INVALID",
    "Invalid decoder text": "SENSE_DECODER_TEXT_INVALID",
    "Unexpected tokenizer decode call count/options": "SENSE_CAPTURE_CALL_CONTRACT",
    "Tokenizer input was mutated": "SENSE_CAPTURE_INPUT_MUTATED",
    "Raw token capture does not match model output": "SENSE_CAPTURE_TEXT_MISMATCH",
    "Wrong runtime": "SENSE_WRONG_RUNTIME",
    "Relocated SenseVoice model root mismatch": "SENSE_MODEL_ROOT_INVALID",
    "Relocated SenseVoice model root alias": "SENSE_MODEL_ROOT_ALIASED",
    "Relocated SenseVoice tokenizer file missing": "SENSE_TOKENIZER_MISSING",
    "Relocated SenseVoice tokenizer file alias or type": "SENSE_TOKENIZER_ALIASED_OR_TYPE",
    "Pinned SenseVoice tokenizer bytes changed": "SENSE_TOKENIZER_BYTES_MISMATCH",
    "Pinned tokenizer vocabulary/unknown semantics mismatch": "SENSE_TOKENIZER_SEMANTICS_MISMATCH",
    "Frozen corrected SenseVoice contract changed": "SENSE_PARSER_BYTES_MISMATCH",
    "SenseVoice single-clip result contract mismatch": "SENSE_RESULT_SHAPE_MISMATCH",
    "SenseVoice result ID/fields mismatch": "SENSE_RESULT_ASSOCIATION_MISMATCH",
    "SenseVoice captured output association mismatch": "SENSE_CAPTURE_ASSOCIATION_MISMATCH",
    "SenseVoice raw token outside pinned vocabulary": "SENSE_TOKEN_OUT_OF_VOCABULARY",
    "Exact captured raw token re-decode mismatch": "SENSE_TOKEN_REDECODE_MISMATCH",
}


def failure_details(error, state):
    """Fixed checkpoints/codes plus a bounded rendered-message digest; no raw text."""
    stage = state.get("stage")
    if stage not in STAGES:
        stage = "adapter_dispatch"
    name = type(error).__name__
    allowed_types = {"ValueError", "RuntimeError", "TypeError", "KeyError", "OSError",
                     "FileNotFoundError", "TimeoutError", "ClipDeadline", "MemoryError"}
    try:
        message = str(error)
    except Exception:
        message = ""
    prefix = message[:4096].encode("utf-8", "backslashreplace")
    return {"stage": stage, "code": ERROR_CODES.get(message, "UNCLASSIFIED"),
            "exception_type": name if name in allowed_types else "Exception",
            "model_inference_returned": (state["model_inference_returned"]
                if type(state.get("model_inference_returned")) is bool else None),
            "message": {"representation": "utf8_rendered_exception_prefix_not_native_stderr",
                        "characters": len(message), "prefix_limit_characters": 4096,
                        "prefix_bytes": len(prefix), "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                        "prefix_truncated": len(message) > 4096, "raw_text_omitted": True}}


def runtime_tokenizer_path(model_root):
    root = Path(model_root)
    if not root.is_absolute() or root.name != "sensevoice" or root.parent.name != "models":
        raise ValueError("Relocated SenseVoice model root mismatch")
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("Relocated SenseVoice model root alias")
    path = root / TOKENIZER_NAME
    if not path.exists() and not path.is_symlink():
        raise ValueError("Relocated SenseVoice tokenizer file missing")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Relocated SenseVoice tokenizer file alias or type")
    return path


def bound_sense_transcript(decoded, capture, sp, tokenizer_path):
    """Re-decode exact raw IDs and verify metadata token boundary before slicing.

    Malformed rich metadata stays an unknown warning, never silently repaired.
    Metadata emotion EMO_UNKNOWN is not the lexical unk token (ID0).
    """
    from asr_stage.decoding import bounded_text, token_ids
    contract = _contract()
    identity = verify_tokenizer(sp, tokenizer_path)
    if (type(capture) is not dict or set(capture) != {'token_ids', 'decoded', 'token_scope'} or
            capture['decoded'] != decoded or capture['token_scope'] != TOKEN_SCOPE):
        raise ValueError('SenseVoice captured output association mismatch')
    decoded = bounded_text(decoded)
    ids = token_ids(capture['token_ids'], allow_empty=True)
    if any(i >= VOCABULARY_SIZE for i in ids):
        raise ValueError('SenseVoice raw token outside pinned vocabulary')
    passed_ids = list(ids)
    if bounded_text(sp.decode(passed_ids)) != decoded or passed_ids != ids:
        raise ValueError('Exact captured raw token re-decode mismatch')
    pieces = [sp.id_to_piece(i) for i in ids[:4]]
    groups = contract['SENSE_GROUPS']
    boundary = (len(pieces) == 4 and
        all(type(piece) is str and piece.startswith('<|') and piece.endswith('|>') and
            piece[2:-2] in group for piece, group in zip(pieces, groups)) and
        decoded.startswith(''.join(pieces)))
    lexical = ids[4:] if boundary else None
    if boundary:
        extracted = contract['corrected_sense_extract'](decoded, lexical)
        metadata = extracted.pop('metadata')
        row = extracted
    else:
        metadata = []
        row = {'raw_text': decoded, 'completeness': 'unknown', 'quality_flags': ['decoding_warning']}
    binding = {'schema': 'cosyvoice12-sense-token-binding-v1', 'tokenizer': identity,
        'parser_contract_sha256': CONTRACT_SHA256, 'raw_token_ids': ids,
        'raw_decoded_sha256': _hash(decoded.encode('utf-8')),
        'metadata_token_ids': ids[:4], 'metadata_pieces': pieces,
        'metadata_boundary_verified': boundary, 'lexical_token_ids': lexical,
        'lexical_unknown_present': (LEXICAL_UNKNOWN_ID in lexical) if boundary else None,
        'raw_redecode_matches_capture': True, 'token_scope': TOKEN_SCOPE}
    return row, metadata, binding


def infer_sense_once_after_approval(runtime, bound, decoder_manifest_sha256, model_root, state):
    """Exact scientific call, with explicit runtime tokenizer and fixed checkpoints.

    State is diagnostic only. It is never used as evidence of permission, source
    identity or resource enforcement; exceptions keep their original identity.
    """
    state.clear()
    state.update(stage="model_verification", model_inference_returned=False)
    if runtime.model_id != SENSE:
        raise RuntimeError("Wrong runtime")
    wrapper = runtime.wrapper
    require_verified_receipt(wrapper.model, describe_cpu_tensor)
    state["stage"] = "pcm_binding"
    pcm = _begin_clip(runtime, bound, decoder_manifest_sha256)
    state["stage"] = "tokenizer_verification"
    path = runtime_tokenizer_path(model_root)
    verify_tokenizer(wrapper.kwargs["tokenizer"].sp, path)
    capture = CapturingTokenizer(wrapper.kwargs["tokenizer"])
    import torch
    state["stage"] = "model_inference"
    with torch.inference_mode():
        output = wrapper.model.inference(data_in=pcm,key=[bound.opaque_id],tokenizer=capture,
            frontend=wrapper.kwargs["frontend"],device="cpu",language="auto",use_itn=False,
            text_norm="woitn",fs=16000,data_type="sound",output_timestamp=False,ban_emo_unk=False)
    state["model_inference_returned"] = True
    state["stage"] = "result_contract"
    if type(output) is not tuple or len(output)!=2 or type(output[0]) is not list or len(output[0])!=1:
        raise RuntimeError("SenseVoice single-clip result contract mismatch")
    item = output[0][0]
    if type(item) is not dict or item.get("key")!=bound.opaque_id or set(item)!={"key","text"}:
        raise RuntimeError("SenseVoice result ID/fields mismatch")
    decoded = bounded_text(item["text"])
    state["stage"] = "token_capture"
    raw = capture.evidence(decoded)
    state["stage"] = "token_binding"
    row,metadata,token_binding = bound_sense_transcript(decoded,raw,wrapper.kwargs["tokenizer"].sp,path)
    state["stage"] = "pcm_postcheck"
    verify_pcm_array(pcm,bound.descriptor["float32_pcm"])
    state["stage"] = "complete"
    return row,{"raw_result":output,"token_evidence":raw,"metadata":metadata,"token_binding":token_binding,
        "input_binding":bound.descriptor,"use_itn":False,"rich_postprocess_applied":False,
        "completeness_scope":"Returned full-input CTC inference; not acoustic transcript completeness"}
