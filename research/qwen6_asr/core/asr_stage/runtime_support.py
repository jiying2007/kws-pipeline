"""Deferred runtime functions. Importing this module does not import ML packages."""
import hashlib
import importlib.metadata
from pathlib import Path
import sys
from .architecture import canonical_sha

VERSIONS = {"torch":"2.12.1+cpu","numpy":"1.26.4","transformers":"4.57.6",
            "qwen-asr":"0.0.6","funasr":"1.4.16","kaldi-native-fbank":"1.22.3"}
SOURCE_MEMBERS = {
    "Qwen/Qwen3-ASR-0.6B": frozenset((
        ("qwen-asr","qwen_asr/__init__.py"),
        ("qwen-asr","qwen_asr/inference/qwen3_asr.py"),
        ("qwen-asr","qwen_asr/core/transformers_backend/configuration_qwen3_asr.py"),
        ("qwen-asr","qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"),
        ("qwen-asr","qwen_asr/core/transformers_backend/processing_qwen3_asr.py"))),
    "FunAudioLLM/SenseVoiceSmall": frozenset((
        ("funasr","funasr/auto/auto_model.py"),("funasr","funasr/models/sense_voice/model.py"),
        ("funasr","funasr/models/ctc/ctc.py"),("funasr","funasr/frontends/wav_frontend.py"),
        ("funasr","funasr/utils/load_utils.py"),("funasr","funasr/utils/fbank.py"),
        ("funasr","funasr/tokenizer/sentencepiece_tokenizer.py"))),
}


def require_runtime_versions():
    actual = {name:importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS: raise RuntimeError("Runtime version drift from qualified environment")
    if sys.byteorder != "little": raise RuntimeError("Only reviewed little-endian runtime supported")
    from runtime_version_gate import require_runtime_profile, CPU_WHEEL
    require_runtime_profile(actual, profile="official_cpu_build_native_fbank", torch_wheel=CPU_WHEEL)
    return actual


def verify_source_members(rows, approved_sha256, model_id):
    if type(rows) is not list or not rows or canonical_sha(rows) != approved_sha256:
        raise RuntimeError("Missing or changed predeclared package source lock")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"distribution","relative_path","sha256"}: raise RuntimeError("Malformed package source row")
        key = (row["distribution"],row["relative_path"])
        if key in seen or row["distribution"] not in VERSIONS: raise RuntimeError("Duplicate/unknown package source")
        seen.add(key)
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py": raise RuntimeError("Invalid package source path")
        source = Path(importlib.metadata.distribution(row["distribution"]).locate_file(str(relative)))
        if not source.is_file() or source.stat().st_size > 1024 * 1024: raise RuntimeError("Missing/oversized package source")
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]: raise RuntimeError("Installed package source differs from independently inspected source")
    if seen != SOURCE_MEMBERS[model_id]:
        raise RuntimeError("Incomplete or extra inspected package-source membership")


def describe_cpu_tensor(value, include_content):
    import torch
    if (not isinstance(value,torch.Tensor) or value.layout != torch.strided or
            value.device.type != "cpu" or value.is_meta or not value.is_contiguous()):
        raise RuntimeError("Expected a contiguous strided non-meta CPU tensor")
    allowed = (torch.float32,torch.float64,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool)
    if value.dtype not in allowed: raise RuntimeError("Unreviewed tensor dtype")
    detached = value.detach().reshape(-1)
    # Bound the finite-check scratch allocation instead of allocating a whole-
    # tensor boolean mask for a large embedding matrix.
    for begin in range(0,detached.numel(),262144):
        if not torch.isfinite(detached[begin:begin+262144]).all().item(): raise RuntimeError("Nonfinite tensor")
    out = {"shape":list(value.shape),"dtype":str(value.dtype)}
    if include_content:
        raw = memoryview(detached.view(torch.uint8).numpy()).cast("B")
        h = hashlib.sha256()
        for begin in range(0,len(raw),1024*1024): h.update(raw[begin:begin+1024*1024])
        out["sha256"] = h.hexdigest()
    return out


def describe_storage(value):
    describe_cpu_tensor(value, False)
    storage = value.untyped_storage()
    return {"storage_pointer":storage.data_ptr(),"storage_nbytes":storage.nbytes(),
            "storage_offset":value.storage_offset(),"stride":list(value.stride()),"device":"cpu"}


def make_pcm_array(bound):
    import numpy as np
    row = bound.descriptor["float32_pcm"]
    array = np.frombuffer(bound.pcm_float32_le,dtype="<f4").copy(order="C")
    verify_pcm_array(array,row)
    return array


def verify_pcm_array(array, expected):
    import numpy as np
    if (type(array) is not np.ndarray or array.dtype != np.dtype("<f4") or
        array.shape != tuple(expected["shape"]) or not array.flags.c_contiguous or
        not np.isfinite(array).all()): raise RuntimeError("PCM array shape/dtype/finite/layout mismatch")
    if hashlib.sha256(memoryview(array).cast("B")).hexdigest() != expected["sha256"]:
        raise RuntimeError("PCM array content changed")


def assert_effective_qwen_attention(model):
    observed = []
    for name,module in model.named_modules():
        kind = type(module).__name__
        if kind in ("Qwen3ASRAudioAttention","Qwen3ASRTextAttention"):
            implementation = getattr(module.config,"_attn_implementation",None)
            observed.append({"name":name,"class":kind,"implementation":implementation})
    counts = {kind:sum(r["class"]==kind for r in observed) for kind in ("Qwen3ASRAudioAttention","Qwen3ASRTextAttention")}
    if counts != {"Qwen3ASRAudioAttention":18,"Qwen3ASRTextAttention":28} or any(r["implementation"]!="eager" for r in observed):
        raise RuntimeError("Effective nested Qwen attention is not the reviewed 18/28 eager recipe")
    return observed
