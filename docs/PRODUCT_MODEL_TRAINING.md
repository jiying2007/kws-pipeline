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

Future manual `model-training` runs use the immutable release:

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
