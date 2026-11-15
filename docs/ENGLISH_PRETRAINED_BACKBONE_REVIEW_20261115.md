# English pretrained-backbone candidate review — 2026-11-15

Status: source-level review only. No upstream code, model asset, dependency, dataset, audio, or weight is imported into this repository by this review. No training or inference is performed.

## Reviewed upstream identity

- Project: https://github.com/dscripka/openWakeWord
- Reviewed source commit: `368c03716d1e92591906a84949bc477f3a834455` (2025-12-30)
- Upstream LICENSE blob: `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`
- Reviewed source files:
  - `openwakeword/model.py`, blob `1296ac179032fca6d2f65c669d07742c946b3c6e`
  - `openwakeword/train.py`, blob `708cdb23bf0c7fdaafab055f99a981b796ec9ccb`
  - `openwakeword/utils.py`, current source at reviewed commit
  - `pyproject.toml`

The repository source LICENSE is Apache-2.0. The upstream README states that its shared feature extractor is based on Google's Speech Embedding TFHub model and identifies that model as Apache-2.0. The same README explicitly states that the distributed pretrained wake-word models are CC BY-NC-SA 4.0 due to their training data. Those pretrained wake-word models are excluded from any commercial candidate path.

## Architecture observations

The upstream system separates:
1. audio preprocessing / mel spectrogram;
2. a shared general speech embedding extractor;
3. a small keyword classifier.

Its training source supports a small fully-connected head and a two-layer bidirectional LSTM classifier. The latter is not causal and must not be copied as a streaming design. The model runtime supports ONNX or TFLite and exposes frame-wise predictions through an audio-feature pipeline, but this is not a C ABI or evidence of SSC305 suitability.

The feature model files are not tracked in the reviewed source tree; runtime resource resolution downloads or expects separate model files. Therefore the source license alone is insufficient. Before any candidate execution, separately retrieve and pin the exact feature assets, record their license/terms and SHA-256, inspect their tensor shapes/quantization/operator set, and establish a C implementation or supported target runtime. No asset hash or target resource number is asserted here.

## Decision

**KEEP AS A HIGH-PRIORITY REFERENCE; DO NOT ADMIT AS A MODEL OR DEPENDENCY YET.**

Why it is worth prototyping:
- it directly addresses the current weak students' lack of broad pretrained acoustic representation;
- a frozen shared embedding plus a small causal head can reduce trainable parameter count and allow two phrase heads to share computation;
- it offers a concrete alternative to repeatedly changing GRU/DS-TCN clip-level pooling or background loss.

Why it is not yet a deployable choice:
- upstream's common RNN classifier is bidirectional, so the causal head must be newly designed and validated;
- feature extraction, buffering, latency, quantization, C export, and state-reset semantics need a target-specific implementation;
- upstream model assets and negative feature corpora have separate rights and must be audited independently;
- the upstream README reports large negative-data scale; the project cannot assume that synthetic prompts or the already exposed LibriSpeech subset are equivalent;
- no SSC305 benchmark or independent FAR/FRR is available.

## Next bounded engineering steps

1. Retrieve exact Google Speech Embedding asset metadata and license text without running inference; hash the actual asset only after obtaining it through an authorized source.
2. Inspect graph operators, input/output tensor geometry, receptive field, causal/chunk behavior, and required runtime dependencies.
3. Estimate C portability from the exact graph. Prefer a minimal native C implementation of the small head; do not assume ONNX Runtime or TFLite is already available in the SSC305 rootfs.
4. Compare frozen-backbone + causal DNN/GRU head against a causal FSMN-CTC candidate on the same admitted source-disjoint data and fixed decoder.
5. Run actual inference/training only after positive pronunciation review, rights admission, immutable input identities, and a finite predeclared recipe.
6. Keep the original 63 WAVs immutable and exposed. Do not regenerate them or use the 517-source LibriSpeech set as an unseen holdout.

## Non-negotiable completion gates

The same frozen candidate must have reproducible training/source/weights, independent human-confirmed positive coverage for both phrases, continuous natural-negative FAR/hour, per-word FRR, one-sided 95% upper bounds, word-end callback latency, C/ARM parity, and physical SSC305 CPU/RAM/thermal/long-run evidence.

V4 targets remain combined FAR <= 0.1/h, per-word FRR <= 5%, and word-end-to-callback P95 <= 500 ms. This source review does not satisfy any acoustic or board gate.
