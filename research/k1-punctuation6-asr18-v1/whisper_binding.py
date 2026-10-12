"""One-shot pre-audio lock binding for the unchanged reviewed Whisper adapter.

The original constructor verifies source/assets and loads the original model.
Only its FIRST native-config write is bound to the new resolver-compatible lock
bytes. No overwritten config, historical relabeling, decoder change, or inference.
This function is used in one model's isolated single-threaded constructor only.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def construct_whisper(adapter_module, runtime, output, source_lock_bytes, asset_lock_bytes):
    from asr_native_contracts import validate_native_config, canonical_sha256
    output = Path(output)
    if ((output / "native-config.json").exists()
            or type(source_lock_bytes) is not bytes or type(asset_lock_bytes) is not bytes):
        raise ValueError("fresh native config and exact resolver lock bytes required")
    source, model = json.loads(source_lock_bytes), json.loads(asset_lock_bytes)
    if (source.get("schema") != "retained-asr-source-lock-v1"
            or model.get("schema") != "retained-asr-model-lock-v1"
            or model.get("model_family") != "whisper"):
        raise ValueError("Whisper requires predeclared resolver-compatible locks")
    original_write = adapter_module.write_once
    count = 0
    def first_write(path, value):
        nonlocal count
        if Path(path) == output / "native-config.json":
            if count:
                raise ValueError("Whisper config can only be frozen once")
            count += 1
            # Mutate the new instance's config before its first persistence and
            # before load-receipt.json is constructed. No capture exists yet.
            value["source_lock_sha256"] = hashlib.sha256(source_lock_bytes).hexdigest()
            value["asset_lock_sha256"] = hashlib.sha256(asset_lock_bytes).hexdigest()
            validate_native_config(value)
            result = original_write(path, value)
            # Historical write_once returns None. Read the actually persisted
            # bytes rather than trusting that callback's return convention.
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != canonical_sha256(value):
                raise ValueError("Whisper config was not retained as canonical bytes")
            return result
        return original_write(path, value)
    adapter_module.write_once = first_write
    try:
        recognizer = adapter_module.FixedWhisper(runtime, output)
    finally:
        adapter_module.write_once = original_write
    if count != 1:
        raise ValueError("Whisper native configuration freeze was not observed")
    validate_native_config(recognizer.config)
    if ((output / "native-config.json").read_bytes() != json.dumps(recognizer.config,
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()):
        raise ValueError("Whisper in-memory and frozen configs disagree")
    return recognizer
