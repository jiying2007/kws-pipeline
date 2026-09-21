# Product model training data contract

The shipping-format model trainer must not silently fall back to the legacy tone
generator.

## Current state

The currently pinned model `model-749187ec1d66` was trained at source
`749187ec1d6662658f06aa9c76d47fde835968db`. At that exact source revision:

- `generator.tts.backend` was `tone`;
- no `generator.external_base_dataset` was bound.

That model remains a valid **synthetic-qualified frozen baseline**, but this fact
explains why later speech-like research did not automatically improve the
deployable `.kwm`: the research corpus and the product training lane were
separate execution paths.

## Governed base for future candidates

Future governed `model-training` runs use the immutable release:

`speech-like-base-5204b798033f`

It is the four-split 384-recording production Stage-A corpus whose independent
384x2 generation run `35318933124` produced the same external-base identity:

`5204b798033fe46e5381c743accf638ab7f16d4e4367f42cba38a6bb44c97b35`

and provider identity:

`d12728562db74951191575e9afe4922adc1260934b02d577dda67d91faa4aff9`

The release is immutable and its corpus archive is additionally pinned by SHA256
in `configs/training/product-speech-like-base-v1.json`.

Before training begins, the workflow:

1. requires the exact current `main` source SHA;
2. downloads the immutable release;
3. verifies release ID, target SHA, archive digest and release checksum manifest;
4. regenerates an effective training config with all four external-base splits;
5. re-hashes every index/summary/WAV through `external_base_dataset.py`;
6. refuses protected evidence and tone fallback;
7. uses that exact effective config for base training, refinement, synthetic
   qualification, robustness and diagnostics.

The effective config SHA is bound across the two training jobs by the existing
base-stage receipt.

## Deterministic feature cache

Product RNN training enables the existing deterministic frontend feature cache
with `train.feature_cache_max_items=8192`. Training manifests contain already
rendered PCM WAV files and `Manifest.__getitem__` performs only deterministic
WAV decoding plus frontend feature extraction, so caching does not freeze any
stochastic augmentation or change the training sample stream.

The same cache wrapper is used for base domain rounds and adversarial refinement.
It changes wall-clock work only: epoch count, sample order, optimizer state,
losses, replay composition, thresholds and gates are unchanged. The wrapper is
bound into `training_code_sha256`, and exported model provenance records
`deterministic-feature-cache-v1`, the configured capacity and
`training_math_changed=false`. Promotion rejects future product candidates
that lack this evidence.

## Auditable training invocation

Governed training keeps the existing manual `workflow_dispatch` entry point and
also accepts a versioned request on protected `main`:

`.github/triggers/model-training-request.json`

The expensive training jobs run on a push only when that exact request file
changes. Ordinary main pushes do not start model training. To request another
governed run, change the request ID and reason through a normal protected-main
pull request. After merge, the workflow still verifies that its checkout SHA is
the repository's current `main` before doing any training.

Each run writes `build/model-training/training-invocation.json` into the
retained base-domain state. For a versioned request this binds the exact request
content and SHA256 to the GitHub event SHA/ref; manual dispatches record the same
source identity without claiming a versioned request.

## Promoted model lineage

For new candidates, finalization freezes these next to the selected model:

- `effective-training-config.json`;
- `product-speech-like-base-contract.json`.

`model-promotion` carries them into the immutable model Release. The model
registry mirror then carries the same files into Git with the rest of the model
tuple.

## Product evidence boundary

This speech-like corpus is still offline synthetic speech. Its source model is
8-kHz-bandlimited and normalized to 16 kHz. It is a substantially more realistic
development base than tone synthesis, but it is **not** real-human final-AFE
evidence.

Shipping approval still requires:

1. real-human final-AFE Phase A;
2. physical target-board Phase B.

If Phase A exposes a repeatable model-capacity failure, use a separate
real-human **development** corpus for retraining; never feed the consumed
held-out qualification corpus back into training.

## Replay/resynthesis boundary

The external speech-like base alone is not sufficient: the development loop also
resynthesizes hard negatives and failure cases. Those paths historically read
`generator.tts` directly and could therefore reintroduce tone audio even when
the base corpus was speech-like.

Governed product training now prepares the same hash-bound offline-TTS provider
in provider-only mode and overrides `generator.tts` with its verified command
backend. Replay provider identity must equal the production base-corpus provider
identity.

Replay voice selection is deterministic and restricted to the eight
`train-*` voice slots. Calibration, test and qualification voice identities are
never used for replay.

## Development refinement source

Adversarial refinement is a development-only optimization stage and does not
consume formal qualification evidence. If base iteration already has a strict
calibration/test candidate, refinement keeps using that strict candidate.

If base iteration has no strict candidate, refinement may start from a
development-only fallback selected without qualification feedback. The fallback
is recall-first: it minimizes worst-case FRR across calibration, test, their
3-5 m far-distance slices, and every shipping keyword; then total FRR; then FAR.
This prevents either a near all-reject checkpoint or a checkpoint that collapses
one wake word from winning refinement source selection merely because the frozen
zero-error gate heavily penalizes FAR.

Formal qualification remains unchanged: it still requires a strict
calibration/test development candidate after refinement, and the guarded formal
renderer still rejects any non-strict candidate.

Model-training #339 (run `35501188138`) proved that a single global wake
multiplier is not a sufficient refinement actuator. Its v1 balance was applied
exactly: 432 exact-wake rows had base mass 864, 1,488 tokenized non-wake rows had
mass 2,976, and the derived multiplier 3.4444 raised exact-wake mass to the same
2,976. Even so, calibration/test FRR ended at 0.875/0.9375 while test FAR fell to
493.73 h^-1; keyword 1 (`你好小窝`) remained 32/32 false rejects. The run is
diagnostic-only and is not eligible for promotion.

Refinement therefore uses per-keyword pressure balancing rather than a global
wake scalar. The policy is `per-keyword-exact-wake-pressure-balance-v2`.
Configured exact-wake rows are counted per keyword. Every non-wake target
sequence is assigned to the configured wake sequence(s) with minimum token edit
distance; ties split the row's effective mass equally, and empty-target rows
split equally across all keywords. This conserves the same total non-wake
effective mass used by v1 while exposing asymmetric lexical pressure around
individual shipping wake words.

Each keyword receives its own exact-wake multiplier so its effective wake mass
matches its assigned non-wake pressure, bounded independently to [1, 12]. The
trainer keeps a default wake multiplier of 1.0 and receives the per-keyword map
explicitly. The map and assigned masses are retained in refinement evidence and
model provenance, and model promotion verifies both copies are identical.

The resulting sample weights still apply consistently to all target-sensitive
objectives: per-frame CTC, sequence margin, strict-prefix completion, and the
ordered-token loss added in #158. Ordered-token loss preserves its legacy value
when all participating samples have equal weights; non-uniform per-keyword wake
weights change its gradient in the same way as the other target-sensitive
objectives. Recurrent-release loss remains unweighted because it models
post-utterance blank release rather than wake/non-wake target identity.

Base training, replay counts, loss coefficients, epochs, learning rate,
thresholds, strict gates and formal qualification remain unchanged.

Model-training run `35439346929` was cancelled after this hidden tone replay
path was identified. It is diagnostic-only and is not eligible for promotion.
