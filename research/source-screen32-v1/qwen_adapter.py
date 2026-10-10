"""Narrow VoiceDesign call adapter; no loader, CLI, downloader or runtime imports.

The future admitted launcher owns exact installed/asset verification, a hard scope
established before runtime imports/loading, persistent attempt claims, deadlines,
EOS capture, native/derived file binding and blind packing. A stdlib checker reads the
actual kernel scope; a missing/unbounded scope fails before runtime imports in the
worker. Mock tests do not prove enforcement. Constructing
a new adapter does not grant a new attempt. This is not a runnable job.
"""
from __future__ import annotations
import hashlib
import math
import random
import struct

from contract import expected_plan, request_preview, require, validate_plan
from runtime_scope import verify_runtime_scope

EXECUTION_READY = False


def validate_loaded_model(wrapper):
    """Check the already loaded objects; construct/import no model dependencies."""
    model = wrapper.model
    require((model.tts_model_type, model.tts_model_size, model.tokenizer_type) ==
            ("voice_design", "1b7", "qwen3_tts_tokenizer_12hz"), "exact VoiceDesign model identity")
    require(model.speaker_encoder is None, "reference-speaker path must be absent")
    require(dict(model.config.talker_config.spk_id) == {} and
            dict(model.config.talker_config.spk_is_dialect) == {}, "no stock/person voice table")
    require(callable(wrapper.generate_voice_design), "locked VoiceDesign API missing")
    for owner in (model, model.speech_tokenizer.model):
        require(owner.training is False, "loaded model must be in evaluation mode")
        parameters = list(owner.parameters())
        require(parameters and all(str(p.device) == "cpu" and str(p.dtype) == "torch.float32"
                                   for p in parameters), "all model parameters must be CPU FP32")
        require(all(str(b.device) == "cpu" for b in owner.buffers()), "non-CPU model buffer")
        attention = [module for _, module in owner.named_modules()
                     if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj"))]
        require(attention and all(getattr(module.config, "_attn_implementation", None) == "eager"
                                  for module in attention), "eager attention required in model and codec")


def validate_output(wavs, rate):
    require(type(rate) is int and rate == 24000, "native output must be 24kHz")
    require(type(wavs) in (list, tuple) and len(wavs) == 1, "exactly one waveform per call")
    waveform = wavs[0]
    require(getattr(waveform, "ndim", None) == 1 and str(getattr(waveform, "dtype", "")) == "float32",
            "native mono float32 waveform required")
    require(0 < len(waveform) <= 12 * rate, "native waveform duration bound")
    values = [float(value) for value in waveform]
    require(all(math.isfinite(value) for value in values), "nonfinite native waveform")
    payload = b"".join(struct.pack("<f", value) for value in values)
    peak = max(abs(value) for value in values)
    flags = (["silent"] if not any(values) else []) + (["source_peak_ge_one"] if peak >= 1.0 else [])
    return waveform, {"sample_rate_hz": rate, "channels": 1, "frames": len(values),
                      "native_float_values_sha256": hashlib.sha256(payload).hexdigest(),
                      "peak": peak, "quality_flags": flags,
                      "termination_verified": False, "acoustic_completeness": "UNKNOWN"}


class FixedQwenAdapter:
    """One in-memory ordered batch; persistent once-only admission remains external.

    torch and numpy are the caller's already verified modules. Only fixed source
    pins and kernel records are read; no shell, network or model load API exists. The hard-scope guard reads the actual kernel scope; no caller boolean or
    callback can stand in for containment or reviewed admission.
    """
    def __init__(self, wrapper, *, torch, numpy):
        self.wrapper, self.torch, self.numpy = wrapper, torch, numpy
        self.plan = expected_plan()
        self.next_index = 0
        self.attempted = []
        self.stopped = False
        verify_runtime_scope()
        validate_loaded_model(wrapper)

    def generate_cell(self, cell_id):
        require(not self.stopped and self.next_index < 16, "source stopped or exhausted")
        try:
            verify_runtime_scope()
            validate_plan(self.plan)
            row = self.plan["cells"][self.next_index]
            require(cell_id == row["cell_id"] and row["lane"] == "qwen-voicedesign", "frozen Qwen cell order")
            validate_loaded_model(self.wrapper)
            preview = request_preview(self.plan, cell_id)
            kwargs = preview["kwargs"]
            require(set(kwargs) == {"text", "instruct", "language", "do_sample", "top_k", "top_p",
                                   "temperature", "repetition_penalty", "subtalker_dosample", "subtalker_top_k",
                                   "subtalker_top_p", "subtalker_temperature", "max_new_tokens", "non_streaming_mode"},
                    "unexpected generation argument")
            # The outer launcher must durably claim this exact cell before entry.
            # Mark it consumed before RNG setup/model call, including exceptions.
            self.attempted.append(cell_id)
            self.next_index += 1
            seed = row["seed"]
            random.seed(seed)
            self.numpy.random.seed(seed)
            self.torch.manual_seed(seed)
            with self.torch.inference_mode():
                wavs, rate = self.wrapper.generate_voice_design(**kwargs)
            waveform, output = validate_output(wavs, rate)
            return waveform, {"cell_id": cell_id, "seed": seed, "attempts": 1,
                              "status": "generated_candidate", "native": output,
                              "role": "source_screen_only", "human_gold": False,
                              "training_admitted": False}
        except BaseException:
            self.stopped = True
            raise
