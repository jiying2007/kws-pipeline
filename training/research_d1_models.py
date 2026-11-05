"""Explicit new D1 candidates, reusing the existing encoder equations.

No Torch import, weights, construction, or inference occurs at module import.
The extended FSMN is a new candidate: retaining old logits does not retain the
old softmax probabilities after two additional outputs are introduced.
"""
from __future__ import annotations

import math

from research.candidate_encoders.fulltext_contract import c1_fulltext_spec, rnn64_fulltext_spec
from research_d1_vocab import BASE_OUTPUTS, OUTPUTS, digest, require

SEED = 20261105
FSMN_WEIGHT = "backbone.out_linear2.linear.weight"
FSMN_BIAS = "backbone.out_linear2.linear.bias"


def build_fresh(kind, vocabulary, reference):
    from research.candidate_encoders.torch_models import build_encoder
    identity = digest(vocabulary)
    if kind == "rnn64-ctc2601":
        spec = rnn64_fulltext_spec(identity)
        return build_encoder(spec, rnn_model=reference(32, 64, OUTPUTS)), spec.to_dict()
    require(kind == "c1-ctc2601", "D1_UNKNOWN_FRESH_CANDIDATE")
    spec = c1_fulltext_spec(identity)
    return build_encoder(spec), spec.to_dict()


def extend_fsmn_state(official_state, *, seed=SEED):
    """Copy every donor tensor and old output row; initialize only two rows.

    The caller authenticates the official checkpoint before reaching this
    function. This numerical boundary also supports small synthetic fixtures.
    Its initializer uses a separate CPU generator and a declared fan-in bound.
    """
    import torch
    from research_weak_train import tensor_sha
    require(type(seed) is int and seed == SEED, "D1_EXTENSION_SEED_CHANGED")
    require(FSMN_WEIGHT in official_state and FSMN_BIAS in official_state, "D1_FSMN_HEAD_MISSING")
    weight, bias = official_state[FSMN_WEIGHT], official_state[FSMN_BIAS]
    require(tuple(weight.shape) == (BASE_OUTPUTS, 140) and tuple(bias.shape) == (BASE_OUTPUTS,) and
            weight.dtype == bias.dtype == torch.float32 and
            weight.device.type == bias.device.type == "cpu" and
            bool(torch.isfinite(weight).all()) and bool(torch.isfinite(bias).all()),
            "D1_OFFICIAL_FSMN_HEAD_GEOMETRY")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bound = 1.0 / math.sqrt(140)
    new_weight = torch.empty(OUTPUTS - BASE_OUTPUTS, 140, dtype=torch.float32).uniform_(
        -bound, bound, generator=generator)
    new_bias = torch.empty(OUTPUTS - BASE_OUTPUTS, dtype=torch.float32).uniform_(
        -bound, bound, generator=generator)
    state = {name: value.detach().clone() for name, value in official_state.items()}
    state[FSMN_WEIGHT] = torch.cat((weight, new_weight))
    state[FSMN_BIAS] = torch.cat((bias, new_bias))
    for name, original in official_state.items():
        restored = state[name][:BASE_OUTPUTS] if name in (FSMN_WEIGHT, FSMN_BIAS) else state[name]
        require(tensor_sha(restored) == tensor_sha(original), "D1_DONOR_TENSOR_CHANGED")
    receipt = {
        "schema": "d1-fsmn-output-extension-v2", "candidate_id": "fsmn-d1-ctc2601-v2",
        "base_outputs": BASE_OUTPUTS, "new_outputs": OUTPUTS,
        "old_weight_rows_sha256": tensor_sha(weight), "old_bias_rows_sha256": tensor_sha(bias),
        "new_weight_rows_sha256": tensor_sha(new_weight), "new_bias_rows_sha256": tensor_sha(new_bias),
        "all_other_donor_tensors_unchanged": True, "old_rows_bitwise_unchanged": True,
        "seed": seed, "initializer": "separate CPU generator; uniform[-1/sqrt(140),1/sqrt(140)]",
        "original_2599_checkpoint_identity": False,
        "old_softmax_probabilities_preserved": False,
    }
    return state, receipt


def build_extended_fsmn(sources, vocabulary):
    import torch
    from research_fsmn_weak import build_model, verify_sources
    from research_weak_train import sha_file
    verify_sources(sources)
    require(sha_file(sources / "tokens.txt") == vocabulary["base_tokens_sha256"],
            "D1_DONOR_VOCABULARY_MISMATCH")
    model, processor = build_model(sources, torch)
    state, receipt = extend_fsmn_state(model.state_dict())
    # Keep the upstream metadata consistent with its replaced affine layer.
    with torch.random.fork_rng(devices=[]):
        replacement = torch.nn.Linear(140, OUTPUTS)
    model.backbone.out_linear2.linear = replacement
    model.backbone.out_linear2.output_dim = OUTPUTS
    model.backbone.output_dim = OUTPUTS
    model.load_state_dict(state, strict=True)
    receipt.update(vocabulary_sha256=digest(vocabulary),
                   donor_checkpoint_sha256=sha_file(sources / "avg_30.pt"))
    return model, processor, receipt


def fulltext_ctc(logits, target, weight, mask, *, normalizer=1.0):
    import torch
    require(logits.ndim == 3 and tuple(logits.shape[1:]) == (1, OUTPUTS),
            "D1_FULLTEXT_LOGIT_GEOMETRY")
    require(type(mask) is bool and type(weight) in (int, float) and math.isfinite(weight) and
            0 <= weight <= 1 and type(normalizer) in (int, float) and
            math.isfinite(normalizer) and normalizer > 0, "D1_LOSS_WEIGHT")
    if not mask:
        require(weight == 0 and target is None, "D1_MASKED_CTC_MUST_RETAIN_NO_TARGET")
        zero = logits.sum() * 0
        return zero, zero.detach()
    require(weight > 0 and type(target) is list and target and
            all(type(value) is int and 0 < value < OUTPUTS for value in target),
            "D1_COMPLETE_NONBLANK_TARGET_REQUIRED")
    minimum = len(target) + sum(a == b for a, b in zip(target, target[1:]))
    require(logits.shape[0] >= minimum, "D1_INFEASIBLE_CTC_GEOMETRY")
    raw = torch.nn.functional.ctc_loss(logits.log_softmax(2), torch.tensor(target, dtype=torch.long),
        torch.tensor([logits.shape[0]], dtype=torch.long), torch.tensor([len(target)], dtype=torch.long),
        blank=0, reduction="none", zero_infinity=False)[0] / len(target)
    require(bool(torch.isfinite(raw)), "D1_NONFINITE_CTC_LOSS")
    return raw * (weight / normalizer), raw.detach()
