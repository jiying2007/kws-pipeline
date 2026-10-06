"""New v1 two-phase exact application/readback gate, no PyTorch import.

Failed application is not transactional: discard the model after any load attempt
that fails. Never infer from a boolean flag alone; use require_verified_receipt.
The loader and descriptor are trusted, separately qualified runtime components.
"""
import json
import weakref
from .checkpoint_contract import (CheckpointContractError, canonical_bytes, digest,
    describe_state, need, prepare_contract, schema_of, verify_metadata)

_VERIFIED = {}


def _forget(model):
    """Never call module equality/hash; models can override them or be unhashable."""
    key = id(model)
    entry = _VERIFIED.get(key)
    if entry is not None and entry[0]() is model:
        del _VERIFIED[key]


def _seal(model, receipt_sha256):
    key = id(model)
    def expired(dead_ref):
        entry = _VERIFIED.get(key)
        if entry is not None and entry[0] is dead_ref:
            del _VERIFIED[key]
    _VERIFIED[key] = (weakref.ref(model, expired), receipt_sha256)


def _same_instance_seal(model):
    entry = _VERIFIED.get(id(model))
    return entry[1] if entry is not None and entry[0]() is model else None


def clear_verification(model):
    _forget(model)
    model._kws_weights_verified = False
    model._kws_checkpoint_receipt = None


def _incompatibility_lists(result):
    # PyTorch's actual load_state_dict return is the _IncompatibleKeys namedtuple.
    # A plain dict, None, truthy value or arbitrary object cannot stand in for it.
    need(isinstance(result, tuple) and len(result) == 2 and
         getattr(type(result), "_fields", None) == ("missing_keys", "unexpected_keys"),
         "Unsupported strict-loader return contract")
    need(type(result[0]) is list and type(result[1]) is list,
         "Strict-loader incompatibility fields must be lists")
    need(not result[0] and not result[1], "Strict-loader reported missing or unexpected keys")
    return {"missing_keys": [], "unexpected_keys": []}


def exact_apply(model, checkpoint, describe_tensor, *, expected_schema=None,
                expected_schema_sha256=None, prefix_mappings=(("", ""),),
                metadata_recipe=None):
    """Prevalidate all inputs, load strictly once, then read every tensor back.

    No key renaming, dtype conversion, partial load, allowlist, or mismatch ignore.
    Only independently approved CPU FP32/integer recipes should call this API.
    Runtime callbacks must reject non-CPU/meta/nonfinite/nonstrided tensors and
    fingerprint exact native bytes. This module cannot authenticate a callback.
    """
    mutation = False
    stage = "prevalidation"
    try:
        clear_verification(model)
        # Fail before any weight mutation if the model cannot be identity-tracked.
        # A weakref need not be hashed, so unhashable modules remain supported.
        weakref.ref(model)
        frozen, contract = prepare_contract(checkpoint, model.state_dict(), describe_tensor,
            expected_schema=expected_schema, expected_schema_sha256=expected_schema_sha256,
            prefix_mappings=prefix_mappings, metadata_recipe=metadata_recipe)
        # Recheck checkpoint and architecture immediately before the sole write.
        need(describe_state(frozen, describe_tensor, True) == contract["checkpoint_state"],
             "Checkpoint changed during prevalidation")
        architecture = model.state_dict()
        need(describe_state(architecture, describe_tensor, False) == contract["expected_schema"],
             "Architecture changed during prevalidation")
        verify_metadata(frozen, contract, "checkpoint")
        verify_metadata(architecture, contract, "architecture")
        stage = "strict_application"
        mutation = True
        result = model.load_state_dict(frozen, strict=True)
        loading_info = _incompatibility_lists(result)
        stage = "readback"
        loaded_state = model.state_dict()
        actual = describe_state(loaded_state, describe_tensor, True)
        need(actual == contract["checkpoint_state"], "Loaded state differs from frozen checkpoint contents")
        verify_metadata(loaded_state, contract, "architecture")
        # Do not use a post-load checkpoint hash as the expected value: a broken
        # loader could mutate both the checkpoint and model to the same bad value.
        need(describe_state(frozen, describe_tensor, True) == contract["checkpoint_state"],
             "Strict loader mutated the checkpoint input")
        verify_metadata(frozen, contract, "checkpoint")
        receipt = {
            "schema": "sensevoice-exact-apply-receipt-v1", "contract": contract,
            "loading_info": loading_info, "verified_loaded_state": actual,
            "verified_loaded_state_sha256": digest(actual),
            "complete_checkpoint_verification_passed": True,
            "execution_authorized": False,
            "runtime_tensor_checks_delegated_to_trusted_callback": True,
        }
        sealed = json.loads(canonical_bytes(receipt))
        _seal(model, digest(sealed))
        model._kws_checkpoint_receipt = sealed
        model._kws_weights_verified = True
        return json.loads(canonical_bytes(sealed))
    except Exception as exc:
        # Invalidate even after a previous successful verification. Never roll
        # back tensor state: rollback can itself partially fail or hide failure.
        try:
            clear_verification(model)
        except Exception:
            _forget(model)
        raise CheckpointContractError(
            str(exc) if isinstance(exc, CheckpointContractError) else type(exc).__name__,
            stage=stage, mutation_may_have_occurred=mutation) from exc


def require_verified_receipt(model, describe_tensor=None):
    """Check same-instance receipt AND fresh full readback before inference.

    Exclusive ownership remains necessary between this call and inference.
    This is an accidental-misuse interlock, not tamper resistance or permission.
    """
    try:
        receipt = getattr(model, "_kws_checkpoint_receipt", None)
        need(getattr(model, "_kws_weights_verified", False) is True and
             type(receipt) is dict and _same_instance_seal(model) is not None and
             digest(receipt) == _same_instance_seal(model) and
             receipt.get("complete_checkpoint_verification_passed") is True and
             receipt.get("execution_authorized") is False,
             "A sealed same-instance verification receipt is required")
        current_state = model.state_dict()
        actual = describe_state(current_state, describe_tensor, True)
        need(actual == receipt["verified_loaded_state"], "Model state changed after verified application")
        verify_metadata(current_state, receipt["contract"], "architecture")
        return json.loads(canonical_bytes(receipt))
    except Exception as exc:
        try:
            clear_verification(model)
        except Exception:
            _forget(model)
        raise CheckpointContractError(
            str(exc) if isinstance(exc, CheckpointContractError) else type(exc).__name__,
            stage="pre_inference_readback", mutation_may_have_occurred=False) from exc
