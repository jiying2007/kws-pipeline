# B: direct three-class streaming KWS prototype

This is a **research software implementation**, not a trained acoustic result or
an enabled experiment. It supplies the missing non-CTC objective/head for route
B: `background`, `你好小窝`, `小窝小窝`. It does not change the five-token RNN/C1
CTC paths, product trainer/exporter, C runtime, historical plans or shipping gate.
No new data, TTS/ASR run, corpus training or real-world improvement is implied.

## What is different from the existing C1 arm

The existing [C1](../candidate_encoders/README.md) and
[paired runner](../model_comparison/README.md) both emit five pinyin/blank token
logits and train with CTC. Changing the encoder alone did not implement route B.

`build_classifier` explicitly constructs the existing C1 factory and retains
only its input projection and causal depthwise/pointwise residual blocks. The
five-token output layer is discarded. A new three-output `classifier` is the
only prediction head in this model's parameter/state inventory. Neither the CTC
forward interface nor its token decoder is called. Background is an ordinary
cross-entropy class, not a CTC blank or empty sequence target.

This small adapter keeps frozen C1 source and existing checkpoint source hashes
unchanged. It relies on C1's `in_proj`, `blocks` and block `forward_chunk` shape;
the numeric test explicitly checks that reuse. Construction consumes the old
head's initialization RNG before discarding it, so reproduce with the same seed
and both source files. The direct spec has its own architecture, class names and
objective identity; it must not be represented as a CTC spec/checkpoint.

The architecture is a local reduced causal DS-TCN, not a reproduction of Google's
published DS-TC-ResNet, its normalization/pooling recipe, or reported accuracy.
The upstream primary references are:

- [Google Research streaming KWS](https://github.com/google-research/google-research/tree/master/kws_streaming): explicit streaming state and incremental classification.
- [Official ds_tc_resnet.py](https://github.com/google-research/google-research/blob/master/kws_streaming/models/ds_tc_resnet.py): causal temporal depthwise/pointwise residual blocks and a class output.
- [Official train.py](https://github.com/google-research/google-research/blob/master/kws_streaming/train/train.py): sparse categorical cross-entropy for direct class training.

These sources were inspected on 2026-10-16. The implementation uses the existing
local C1 blocks, not a copied TensorFlow implementation or pretrained weights.

## Model and loss API

Torch is a lazy dependency; package import does not initialize or execute a model.
`DirectKWSSpec(hidden_dim, kernels, dilations)` requires explicit dimensions and
reuses the existing 32-feature C1 allocation checks. These limits are software
safety bounds, not device budgets. The spec serializes/deserializes with explicit
three-class and direct-objective identity.

- `init_state` / `reset_state`: fresh caller-owned per-layer history.
- `step`: `[B,32]` plus state to `[B,3]` plus state.
- `forward_chunk`: `[B,T,32]` plus state to `[T,B,3]` plus state.
- `forward`: start a new window/sequence with zero history; `[B,T,32]` to `[T,B,3]`.
- `window_cross_entropy(logits, targets, lengths)`: CE at each window's last valid
  frame. Targets are integer classes `0..2`. Optional lengths exclude padded
  tails; they do not create timing labels or CTC alignments.

The causal history length is `1 + sum((kernel - 1) * dilation)` frames. For an
illustrative width-32, kernel `(5,5,5,5)`, dilation `(1,2,4,8)` topology this is 61
frames (1.2 seconds of past frame offsets). This is not a selected recipe.
The 25 ms frontend window and 20 ms hop are additional timing facts, not zero
end-to-end delay. Empty chunks preserve state, EOF adds no frames, and returned
history has bounded independent storage. Autograd is not implicitly detached;
the training caller owns cross-chunk truncation and total activation memory.

`class_target(expected_keywords, annotation_status="complete")` reuses the
existing clip-presence label convention: `[] -> 0`, `[1] -> 1`, `[2] -> 2`.
Unknown/partial/missing labels and multiple/repeated targets are rejected. It is
only a label mapper, not a source/rights/admission validator.

**A whole-clip presence label is not enough to supervise a causal endpoint.**
The caller must provide labeled windows whose full target is in the model's
receptive context at that endpoint. A long positive clip with trailing silence
cannot silently be labeled positive at EOF. Continuous positives require
reviewed timing/window construction, and negative windows must have complete
annotations. Partial phrases and near-homophones belong in reviewed background
training. This prototype deliberately does not infer boundaries from transcripts
or promote machine labels to human truth.

## Continuous feature and event API

`stream_events(model.eval(), feature_chunks, event_head, recording=...)` evaluates
one complete recording's unpadded `[T,32]` feature chunks. It preserves model
history and the event head across uneven and empty chunks; no per-chunk reset or
EOF flush occurs. Use a new model state/head/clock at each recording boundary or
declared discontinuity. The helper owns the model state for one recording and
requires a fresh event head; it is not a pause/resume serialization interface.

Reuse the existing [frontend](../../training/frontend.py) and the paired runner's
`iter_wav_features` to produce logmel chunks with persistent PCM overlap. A
feature producer must not recompute a stateful PCEN frontend independently on
each chunk. This module does not take over WAV I/O, resampling, AFE or recording
identity checks. Complete-window timing is fixed to the existing mono 16-kHz,
400-sample window, 320-sample hop, starting continuously from recording sample
0. Cropped/skipped frames and other geometries have incompatible timestamps.

`DirectEventHead` requires two explicit thresholds, a consecutive-frame count
and per-word cooldown. A candidate must be the unique three-class argmax and
meet or exceed its word's threshold. A sustained high score emits once until it falls
below threshold or changes class; a rearmed event still obeys cooldown. It keeps
streak, latch, cooldown and frame clock across chunks. Ties emit no event.
Invalid probabilities fail before advancing that state.

Events use the unchanged scorer schema:

```json
{"recording":"example","keyword_id":1,"confidence":0.8,"time_s":0.045}
```

`time_s` is when the complete input frame is available, not a measured acoustic
word endpoint. Feed these events to [eval/score_events.py](../../eval/score_events.py)
with independently labeled references. Its matching windows, denominators,
per-word misses, duplicate events, negative-only FAR and confidence bounds apply
unchanged. Clip-presence labels must stay in the scorer's clip-presence mode;
do not fabricate event windows.

A terminal-window classifier with this event policy is an executable streaming
prototype, **not a complete continuously trained/calibrated KWS system**. The
CTC route and this route have different score semantics and decoder operations.
The existing paired-runner comparator requires equal decoder hashes and cannot
be reused unchanged to pronounce an A/B winner. A future A/B must freeze shared
recording/reset/EOF/matching/refractory rules, document the route-specific token
versus direct-class decoding, and select each threshold using calibration only
at a matched FAR operating point. Equal numeric thresholds are not equal FAR.

## Reproduce the software checks

Use the repository's pinned CPU Torch environment, without dependency changes:

```sh
python tests/test_direct_kws_numeric.py --report /tmp/direct-kws-fixture.json
python tests/test_candidate_encoder_numeric.py --report /tmp/c1-regression.json
python tests/test_candidate_encoder_source.py
python tests/test_candidate_encoder_contract.py
```

The main numeric fixture uses seed 20261016, feature seed 20261017, batch 2,
13 artificial feature frames and a width-8 `(3,1,2)` / dilation `(1,3,2)` toy
topology. One additional handcrafted width-1, four-frame impulse checks that an
incorrect per-chunk model reset would destroy a real nonconstant-output event.
It checks 3-class identity, reuse of the old backbone, finite CE backward,
whole/frame/uneven-chunk output and gradient parity, padded-tail loss exclusion,
causality, state/reset/storage/row isolation, rejected inputs, an in-memory
state-dict round trip, direct events and the real scorer interface. Handwritten
probabilities and reference windows are arithmetic fixtures. There are no audio
files, pretrained weights, optimizer updates, topology sweeps or acoustic runs.
Missing Torch is an error, never a silently skipped pass.

The local 2026-10-16 run passed all 15 direct tests and the existing 12 numeric,
15 source, 11 contract tests (53 total). This is a local focused check, not a
fresh-main CI pass or complete repository suite. Numeric receipts record source
hashes/runtime/fixture scope and intentionally do not contain scientific scores.
The existing product-training-data-contract objective-math job runs these direct
checks on the existing pinned CPU runtime, with a five-minute step limit and a
JSON-only receipt. Both PR/push filters and the existing selector cover direct
source/test changes. Direct-only changes do not enable original training/data
jobs; mixed product, selector or workflow edits preserve their existing gates.
No new workflow or real-data experiment is introduced.

## Next bounded A/B inputs and remaining work

1. Use one rights-cleared, admitted source inventory with disjoint speaker,
   session, source/augmentation and decoded-PCM groups for training, calibration
   and selection. Preserve the existing split/admission tooling. No human
   recording is required merely to continue software implementation; synthetic
   development evidence must retain its limited interpretation.
2. Supply common source windows and labels usable by both the CTC and direct
   objectives. Freeze a bounded sample-exposure/compute budget, topology and
   seeds. Different loss magnitudes are not comparable quality measurements.
3. Implement the bounded direct training/checkpoint runner with exact source,
   data/split/schedule and class identity. The tested state-dict round trip is
   not a provenance-complete retained training checkpoint or product exporter.
4. Freeze continuous event rules; calibrate independent thresholds without
   looking at selection labels, then compare per-word FRR at matched FAR,
   negative exposure/confidence bounds, latency and measured resource cost.
5. Only a supported candidate proceeds to multi-seed confirmation, quantization,
   C/export parity and SSC305/final-AFE qualification. Neither this source nor
   these tests change any release approval.
