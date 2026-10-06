"""Independent stdlib state-schema derivation from pinned architecture source.

No weights/runtime imports. These are review candidates until config/source body
locks and a real architecture-state comparison pass. Never infer keys from a
checkpoint and call that independent evidence.
"""
import hashlib
import json
import math

QWEN_SOURCE = "7c6daf77a2421100f5fb066495372c00129d39ff"
SENSE_SOURCE = "904cd18681b8083de5e1039bd0ecebc4f49ede60"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def qwen06_schema():
    """Fixed advertised 0.6B geometry, all retained state in CPU FP32.

    Source: Qwen3ASRForConditionalGeneration -> thinker -> audio_tower/model/lm_head.
    Sinusoidal position and rotary inverse-frequency buffers are nonpersistent.
    No omitted aliases are assumed. Actual checkpoint/runtime mismatch fails.
    """
    state = {}
    def add(key, *shape):
        if key in state: raise ValueError("Duplicate derived key")
        state[key] = {"shape": list(shape), "dtype": "torch.float32"}
    a = "thinker.audio_tower."
    for i in range(18):
        p = a + f"layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            add(p + "self_attn." + proj + ".weight", 896, 896)
            add(p + "self_attn." + proj + ".bias", 896)
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            add(p + norm + ".weight", 896); add(p + norm + ".bias", 896)
        add(p + "fc1.weight", 3584, 896); add(p + "fc1.bias", 3584)
        add(p + "fc2.weight", 896, 3584); add(p + "fc2.bias", 896)
    add(a + "ln_post.weight", 896); add(a + "ln_post.bias", 896)
    for index, inputs in ((1, 1), (2, 480), (3, 480)):
        add(a + f"conv2d{index}.weight", 480, inputs, 3, 3)
        add(a + f"conv2d{index}.bias", 480)
    add(a + "conv_out.weight", 896, 480 * 16)
    add(a + "proj1.weight", 896, 896); add(a + "proj1.bias", 896)
    add(a + "proj2.weight", 1024, 896); add(a + "proj2.bias", 1024)
    t = "thinker.model."
    add(t + "embed_tokens.weight", 151936, 1024)
    for i in range(28):
        p = t + f"layers.{i}."
        for name, output in (("q_proj", 2048), ("k_proj", 1024), ("v_proj", 1024)):
            add(p + "self_attn." + name + ".weight", output, 1024)
        add(p + "self_attn.o_proj.weight", 1024, 2048)
        for norm in ("q_norm", "k_norm"):
            add(p + "self_attn." + norm + ".weight", 128)
        add(p + "mlp.gate_proj.weight", 3072, 1024)
        add(p + "mlp.up_proj.weight", 3072, 1024)
        add(p + "mlp.down_proj.weight", 1024, 3072)
        add(p + "input_layernorm.weight", 1024)
        add(p + "post_attention_layernorm.weight", 1024)
    add(t + "norm.weight", 1024)
    add("thinker.lm_head.weight", 151936, 1024)
    return state


def sensevoice_schema():
    """Fixed 70-layer SANM/CTC state candidate from official Python constructors.

    25055 vocabulary is independently documented by the official SenseVoice
    native-runtime architecture; locked tokenizer must confirm it before use.
    The stateless positional encoder, dropout/specaug/loss add no state tensors.
    """
    state = {}
    def add(key, *shape):
        if key in state: raise ValueError("Duplicate derived key")
        state[key] = {"shape": list(shape), "dtype": "torch.float32"}
    layers = [("encoders0.0", 560)] + [(f"encoders.{i}", 512) for i in range(49)]
    layers += [(f"tp_encoders.{i}", 512) for i in range(20)]
    for name, inputs in layers:
        p = "encoder." + name + "."
        add(p + "self_attn.linear_out.weight", 512, 512)
        add(p + "self_attn.linear_out.bias", 512)
        add(p + "self_attn.linear_q_k_v.weight", 1536, inputs)
        add(p + "self_attn.linear_q_k_v.bias", 1536)
        add(p + "self_attn.fsmn_block.weight", 512, 1, 11)
        add(p + "feed_forward.w_1.weight", 2048, 512)
        add(p + "feed_forward.w_1.bias", 2048)
        add(p + "feed_forward.w_2.weight", 512, 2048)
        add(p + "feed_forward.w_2.bias", 512)
        add(p + "norm1.weight", inputs); add(p + "norm1.bias", inputs)
        add(p + "norm2.weight", 512); add(p + "norm2.bias", 512)
    for name in ("after_norm", "tp_norm"):
        add("encoder." + name + ".weight", 512); add("encoder." + name + ".bias", 512)
    add("ctc.ctc_lo.weight", 25055, 512); add("ctc.ctc_lo.bias", 25055)
    add("embed.weight", 16, 560)
    return state


def schema_summary(state):
    elements = sum(math.prod(row["shape"]) for row in state.values())
    return {"keys": len(state), "elements": elements, "fp32_bytes": elements * 4,
            "canonical_schema_sha256": canonical_sha(state),
            "actual_runtime_or_checkpoint_verified": False}


def validate_qwen_geometry(config):
    if type(config) is not dict or config.get("model_type") != "qwen3_asr":
        raise ValueError("Wrong Qwen architecture")
    thinker = config.get("thinker_config", {})
    for part, expected in (
        ("audio_config", {"d_model":896,"encoder_layers":18,"encoder_ffn_dim":3584,
                          "encoder_attention_heads":14,"downsample_hidden_size":480,
                          "num_mel_bins":128,"output_dim":1024}),
        ("text_config", {"hidden_size":1024,"num_hidden_layers":28,"intermediate_size":3072,
                         "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
                         "vocab_size":151936,"attention_bias":False}),
    ):
        actual = thinker.get(part)
        if type(actual) is not dict:
            raise ValueError("Missing Qwen subconfig")
        for name, value in expected.items():
            if type(actual.get(name)) is not type(value) or actual[name] != value:
                raise ValueError("Qwen architecture geometry differs: " + part + "." + name)


def validate_sense_geometry(config):
    if type(config) is not dict or config.get("model") != "SenseVoiceSmall" or config.get("encoder") != "SenseVoiceEncoderSmall":
        raise ValueError("Wrong SenseVoice architecture")
    for part, expected in (
        ("encoder_conf", {"output_size":512,"num_blocks":50,"tp_blocks":20,
                          "linear_units":2048,"kernel_size":11,"attention_heads":4}),
        ("frontend_conf", {"fs":16000,"n_mels":80,"lfr_m":7,"lfr_n":6}),
    ):
        actual = config.get(part)
        if type(actual) is not dict:
            raise ValueError("Missing SenseVoice subconfig")
        for name, value in expected.items():
            if type(actual.get(name)) is not type(value) or actual[name] != value:
                raise ValueError("SenseVoice geometry differs: " + part + "." + name)
    if config.get("normalize") is not None or config.get("ctc_conf") not in (None, {}):
        raise ValueError("Unreviewed SenseVoice normalization/CTC configuration")
